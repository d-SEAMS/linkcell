//! CUDA path: consume/produce dlpk tensors, call `lc_gpu_*`.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;

use dlpk::pyo3::PyDLPack;
use dlpk::sys::{
    DLDevice, DLDeviceType, DLManagedTensorVersioned, DLTensor, DLPACK_FLAG_BITMASK_IS_COPIED,
    DLPACK_FLAG_BITMASK_READ_ONLY,
};
use dlpk::{DLPackTensor, GetDLPackDataType};
use linkcell::lc_cell;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[repr(C)]
struct lc_gpu_workspace {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn lc_gpu_available() -> c_int;
    fn lc_gpu_last_error() -> *const c_char;
    fn lc_gpu_workspace_new() -> *mut lc_gpu_workspace;
    fn lc_gpu_workspace_free(ws: *mut lc_gpu_workspace);
    fn lc_gpu_knearest_many(
        ws: *mut lc_gpu_workspace,
        xyz: *const f64,
        n: usize,
        n_frames: usize,
        simbox: *const lc_cell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
        out_d2: *mut f64,
        wait: c_int,
    ) -> c_int;
    fn lc_gpu_knearest_many_dcell(
        ws: *mut lc_gpu_workspace,
        xyz: *const f64,
        n: usize,
        n_frames: usize,
        cell: *const f64,
        cell_n: c_int,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
        out_d2: *mut f64,
        wait: c_int,
    ) -> c_int;
    fn lc_gpu_queue(ws: *mut lc_gpu_workspace) -> *mut c_void;
    fn lc_gpu_alloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn lc_gpu_free(ptr: *mut c_void);
    fn lc_gpu_fill_i32(ptr: *mut c_void, value: c_int, n: usize) -> c_int;
    fn lc_gpu_memcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
}

fn gpu_err() -> PyErr {
    let p = unsafe { lc_gpu_last_error() };
    if p.is_null() {
        PyRuntimeError::new_err("gpu knearest failed")
    } else {
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        PyRuntimeError::new_err(s)
    }
}

pub fn available() -> bool {
    unsafe { lc_gpu_available() != 0 }
}

/// `kind` is `cudaMemcpyKind`: 2 is device to host.
pub fn copy_dtoh(dst: *mut u8, src: *const u8, bytes: usize) -> PyResult<()> {
    if unsafe { lc_gpu_memcpy(dst.cast(), src.cast(), bytes, 2) } != 0 {
        return Err(gpu_err());
    }
    Ok(())
}

fn capsule_on_stream<'py>(
    obj: &Bound<'py, PyAny>,
    stream: *mut c_void,
) -> PyResult<Bound<'py, PyCapsule>> {
    if !stream.is_null() {
        let kwargs = pyo3::types::PyDict::new(obj.py());
        kwargs.set_item("stream", stream as usize)?;
        if let Ok(cap) = obj.call_method("__dlpack__", (), Some(&kwargs)) {
            if let Ok(c) = cap.cast_into::<PyCapsule>() {
                return Ok(c);
            }
        }
    }
    crate::capsule_of(obj)
}

/// Device allocation whose Drop is `lc_gpu_free`. Converted to a
/// `DLPackTensor` with the same manager-context pattern dlpk uses for
/// `Vec` (CPU).
struct DeviceBuf {
    ptr: *mut c_void,
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { lc_gpu_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

struct Mgr {
    buf: DeviceBuf,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

unsafe extern "C" fn deleter(tensor: *mut DLManagedTensorVersioned) {
    unsafe {
        let ctx = (*tensor).manager_ctx.cast::<Mgr>();
        drop(Box::from_raw(ctx));
        drop(Box::from_raw(tensor));
    }
}

fn device_tensor<T: GetDLPackDataType>(
    buf: DeviceBuf,
    shape: Vec<i64>,
    device: DLDevice,
) -> DLPackTensor {
    let ndim = shape.len() as i32;
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut ctx = Box::new(Mgr {
        buf,
        shape: shape.into_boxed_slice(),
        strides: strides.into_boxed_slice(),
    });
    let data = ctx.buf.ptr;
    let dl_tensor = DLTensor {
        data,
        device,
        ndim,
        dtype: T::get_dlpack_data_type(),
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: dlpk::sys::DLPackVersion::current(),
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(deleter),
        flags: DLPACK_FLAG_BITMASK_IS_COPIED,
        dl_tensor,
    });
    unsafe { DLPackTensor::from_ptr(Box::into_raw(managed)) }
}

/// `PyDLPack` rejects `stream=`. After `lc_gpu_wait` the buffer is
/// idle, so torch's stream argument can be ignored.
#[pyclass]
pub struct StreamDlpack {
    inner: PyDLPack,
}

#[pymethods]
impl StreamDlpack {
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
        max_version: Option<Bound<'py, PyAny>>,
        dl_device: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyCapsule>> {
        let _ = stream;
        self.inner
            .__dlpack__(py, None, max_version, dl_device, copy)
    }

    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> PyResult<Py<pyo3::types::PyTuple>> {
        self.inner.__dlpack_device__(py)
    }
}

fn to_stream_dlpack(py: Python<'_>, tensor: DLPackTensor) -> PyResult<Py<PyAny>> {
    let inner =
        PyDLPack::try_from(tensor).map_err(|e| PyRuntimeError::new_err(format!("dlpack: {e}")))?;
    Ok(Py::new(py, StreamDlpack { inner })?.into_any())
}

fn xyz_n(shape: &[i64]) -> PyResult<(usize, usize)> {
    match shape {
        [n, 3] if *n > 0 => Ok((*n as usize, 1)),
        [f, n, 3] if *f > 0 && *n > 0 => Ok((*n as usize, *f as usize)),
        [n] if *n > 0 && *n % 3 == 0 => Ok(((*n as usize) / 3, 1)),
        _ => Err(PyValueError::new_err(
            "xyz must be float64 shape (n, 3), (n_frames, n, 3), or (n*3,)",
        )),
    }
}

unsafe extern "C" fn borrowed_capsule_deleter(tensor: *mut DLManagedTensorVersioned) {
    unsafe {
        let managed = Box::from_raw(tensor);
        drop(Box::from_raw(managed.manager_ctx.cast::<Py<PyCapsule>>()));
    }
}

fn take_cuda(obj: &Bound<'_, PyAny>, stream: *mut c_void) -> PyResult<DLPackTensor> {
    let cap = capsule_on_stream(obj, stream)?;
    if cap.name()?.map(|name| unsafe { name.as_cstr() }) != Some(crate::DLTENSOR) {
        return DLPackTensor::try_from(&cap)
            .map_err(|e| PyValueError::new_err(format!("dlpack: {e}")));
    }
    // The legacy capsule owns its buffer and metadata. Retaining it keeps
    // both alive through the synchronous GPU operation without consuming
    // its deleter or interpreting the legacy layout as a versioned tensor.
    let dl_tensor = crate::with_dltensor(&cap, |tensor| {
        Ok(DLTensor {
            data: tensor.data,
            device: tensor.device,
            ndim: tensor.ndim,
            dtype: tensor.dtype,
            shape: tensor.shape,
            strides: tensor.strides,
            byte_offset: tensor.byte_offset,
        })
    })?;
    let managed = Box::new(DLManagedTensorVersioned {
        version: dlpk::sys::DLPackVersion::current(),
        manager_ctx: Box::into_raw(Box::new(cap.unbind())).cast(),
        deleter: Some(borrowed_capsule_deleter),
        flags: DLPACK_FLAG_BITMASK_READ_ONLY,
        dl_tensor,
    });
    // The manager owns the capsule from which all tensor pointers originate.
    Ok(unsafe { DLPackTensor::from_ptr(Box::into_raw(managed)) })
}

fn cell_n_from_shape(shape: &[i64]) -> PyResult<i32> {
    match shape {
        [3] => Ok(3),
        [3, 3] => Ok(9),
        [4, 3] => Ok(12),
        [n] if *n == 3 || *n == 9 || *n == 12 => Ok(*n as i32),
        _ => Err(PyValueError::new_err(
            "cell must be float64 shape (3,) ortho, (3, 3) lattice rows, or (4, 3) rows plus origin",
        )),
    }
}

pub fn knearest_cuda<'py>(
    py: Python<'py>,
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
    k: usize,
    mask: Option<&Bound<'py, PyAny>>,
    cell_hint: Option<f64>,
    n_frames: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    if !available() {
        return Err(PyRuntimeError::new_err(
            "CUDA xyz but the driver or nvrtc is not loaded",
        ));
    }
    let ws = unsafe { lc_gpu_workspace_new() };
    if ws.is_null() {
        return Err(gpu_err());
    }
    struct Ws(*mut lc_gpu_workspace);
    impl Drop for Ws {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { lc_gpu_workspace_free(self.0) };
            }
        }
    }
    let ws = Ws(ws);
    let stream = unsafe { lc_gpu_queue(ws.0) };
    let xyz_t = take_cuda(xyz, stream)?;
    if xyz_t.device().device_type != DLDeviceType::kDLCUDA {
        return Err(PyValueError::new_err("expected kDLCUDA xyz"));
    }
    let (n, frames_from_xyz) = xyz_n(xyz_t.shape())?;
    let n_frames = if n_frames == 0 {
        frames_from_xyz
    } else {
        n_frames
    };
    if frames_from_xyz != n_frames {
        return Err(PyValueError::new_err("xyz frame count mismatch"));
    }
    let need = n
        .checked_mul(k)
        .and_then(|v| v.checked_mul(n_frames))
        .ok_or_else(|| PyValueError::new_err("n * k * n_frames overflows"))?;
    let mask_t = if let Some(m) = mask {
        Some(take_cuda(m, stream)?)
    } else {
        None
    };
    let mask_ptr = if let Some(ref t) = mask_t {
        if t.device().device_type != DLDeviceType::kDLCUDA {
            return Err(PyValueError::new_err(
                "mask must be on the same CUDA device",
            ));
        }
        t.data_ptr::<c_int>()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?
    } else {
        ptr::null()
    };
    let xyz_ptr = xyz_t
        .data_ptr::<f64>()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let mut out_nn = ptr::null_mut();
    if unsafe { lc_gpu_alloc(&mut out_nn, need * std::mem::size_of::<c_int>()) } != 0 {
        return Err(gpu_err());
    }
    let nn_buf = DeviceBuf { ptr: out_nn };
    if unsafe { lc_gpu_fill_i32(nn_buf.ptr, -1, need) } != 0 {
        return Err(gpu_err());
    }
    let mut out_d2 = ptr::null_mut();
    if unsafe { lc_gpu_alloc(&mut out_d2, need * std::mem::size_of::<f64>()) } != 0 {
        return Err(gpu_err());
    }
    let d2_buf = DeviceBuf { ptr: out_d2 };

    let hint = cell_hint.unwrap_or(0.0);
    let ws_p = ws.0 as usize;
    let xyz_p = xyz_ptr as usize;
    let mask_p = mask_ptr as usize;
    let nn_p = nn_buf.ptr as usize;
    let d2_p = d2_buf.ptr as usize;
    let cell_cuda = crate::peek_cuda(cell)?;
    let st = if cell_cuda {
        let cell_t = take_cuda(cell, stream)?;
        if cell_t.device().device_type != DLDeviceType::kDLCUDA {
            return Err(PyValueError::new_err("expected kDLCUDA cell"));
        }
        if cell_t.device().device_id != xyz_t.device().device_id {
            return Err(PyValueError::new_err(
                "cell and xyz must sit on the same CUDA device",
            ));
        }
        let cell_n = cell_n_from_shape(cell_t.shape())?;
        let cell_ptr = cell_t
            .data_ptr::<f64>()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let cell_p = cell_ptr as usize;
        py.detach(|| unsafe {
            lc_gpu_knearest_many_dcell(
                ws_p as *mut lc_gpu_workspace,
                xyz_p as *const f64,
                n,
                n_frames,
                cell_p as *const f64,
                cell_n,
                k,
                mask_p as *const c_int,
                hint,
                nn_p as *mut c_int,
                d2_p as *mut f64,
                1,
            )
        })
    } else {
        let cell_buf = crate::take_f64(cell)?;
        let (_, raw) = crate::parse_cell(&cell_buf)?;
        py.detach(|| unsafe {
            lc_gpu_knearest_many(
                ws_p as *mut lc_gpu_workspace,
                xyz_p as *const f64,
                n,
                n_frames,
                &raw,
                k,
                mask_p as *const c_int,
                hint,
                nn_p as *mut c_int,
                d2_p as *mut f64,
                1,
            )
        })
    };
    if st != 0 {
        return Err(gpu_err());
    }
    // Kernel done. Move the pointers into dlpk-managed tensors.
    let device = DLDevice::cuda(xyz_t.device().device_id);
    let shape = if n_frames <= 1 {
        vec![n as i64, k as i64]
    } else {
        vec![n_frames as i64, n as i64, k as i64]
    };
    let nn_t = device_tensor::<i32>(nn_buf, shape.clone(), device);
    let d2_t = device_tensor::<f64>(d2_buf, shape, device);
    Ok((to_stream_dlpack(py, nn_t)?, to_stream_dlpack(py, d2_t)?))
}
