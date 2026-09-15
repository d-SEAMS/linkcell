//! Python bindings. Arrays cross the FFI as DLPack (dlpk).
//!
//! `knearest` takes any `__dlpack__()` object. Host tensors are copied
//! out of the capsule while it is still alive, then `knearest_into`
//! runs with the GIL detached. CUDA `xyz` (`kDLCUDA`, including
//! `torch.Tensor` on GPU) is passed by device pointer into `lc_gpu_*`.

use dlpk::pyo3::PyDLPack;
use dlpk::sys::{
    DLDataTypeCode, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned, DLTensor,
};
use dlpk::DLPackTensor;
use linkcell::{knearest_into_d2, knearest_into_many, lc_cell, Cell};
use ndarray::{Array2, Array3};

mod gpu;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyModule};

const DLTENSOR_VERSIONED: &std::ffi::CStr = c_str!("dltensor_versioned");
const DLTENSOR: &std::ffi::CStr = c_str!("dltensor");

pub(crate) fn capsule_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyCapsule>> {
    let cap = obj.call_method0("__dlpack__")?;
    cap.cast_into::<PyCapsule>().map_err(Into::into)
}

pub(crate) fn with_dltensor<R>(
    cap: &Bound<'_, PyCapsule>,
    f: impl FnOnce(&DLTensor) -> PyResult<R>,
) -> PyResult<R> {
    let name = cap.name()?.map(|n| unsafe { n.as_cstr() });
    if name == Some(DLTENSOR_VERSIONED) {
        let ptr = cap
            .pointer_checked(Some(DLTENSOR_VERSIONED))?
            .as_ptr()
            .cast::<DLManagedTensorVersioned>();
        let tensor = unsafe { &(*ptr).dl_tensor };
        return f(tensor);
    }
    if name == Some(DLTENSOR) {
        let ptr = cap
            .pointer_checked(Some(DLTENSOR))?
            .as_ptr()
            .cast::<DLManagedTensor>();
        let tensor = unsafe { &(*ptr).dl_tensor };
        return f(tensor);
    }
    Err(PyValueError::new_err(
        "expected a dltensor or dltensor_versioned capsule",
    ))
}

fn is_f64(t: &DLTensor) -> bool {
    t.dtype.code == DLDataTypeCode::kDLFloat && t.dtype.bits == 64
}

fn is_i32(t: &DLTensor) -> bool {
    matches!(
        t.dtype.code,
        DLDataTypeCode::kDLInt | DLDataTypeCode::kDLUInt
    ) && t.dtype.bits == 32
}

fn is_host(t: &DLTensor) -> bool {
    matches!(
        t.device.device_type,
        DLDeviceType::kDLCPU | DLDeviceType::kDLCUDAHost | DLDeviceType::kDLROCMHost
    )
}

fn is_cuda(t: &DLTensor) -> bool {
    t.device.device_type == DLDeviceType::kDLCUDA
}

pub(crate) fn peek_cuda(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    match obj.call_method0("__dlpack_device__") {
        Ok(dev) => {
            let tup = dev.cast::<pyo3::types::PyTuple>()?;
            let ty: i32 = tup.get_item(0)?.extract()?;
            Ok(ty == 2)
        }
        Err(_) => {
            let peek = capsule_of(obj)?;
            let flag = with_dltensor(&peek, |t| Ok(is_cuda(t)))?;
            Ok(flag)
        }
    }
}

fn shape_of(t: &DLTensor) -> PyResult<&[i64]> {
    if t.ndim < 0 {
        return Err(PyValueError::new_err("negative ndim"));
    }
    let n = t.ndim as usize;
    if n == 0 {
        return Ok(&[]);
    }
    if t.shape.is_null() {
        return Err(PyValueError::new_err("null shape"));
    }
    Ok(unsafe { std::slice::from_raw_parts(t.shape, n) })
}

pub(crate) fn numel(shape: &[i64]) -> PyResult<usize> {
    let mut n: usize = 1;
    for &d in shape {
        if d < 0 {
            return Err(PyValueError::new_err("negative shape"));
        }
        n = n
            .checked_mul(d as usize)
            .ok_or_else(|| PyValueError::new_err("shape overflow"))?;
    }
    Ok(n)
}

fn require_c_contiguous(t: &DLTensor, shape: &[i64]) -> PyResult<()> {
    if t.strides.is_null() {
        return Ok(());
    }
    let strides = unsafe { std::slice::from_raw_parts(t.strides, shape.len()) };
    let mut expect = 1i64;
    for (&dim, &st) in shape.iter().rev().zip(strides.iter().rev()) {
        if dim <= 1 {
            continue;
        }
        if st != expect {
            return Err(PyValueError::new_err(format!(
                "dlpack tensor is not C-contiguous (stride {st}, expected {expect})"
            )));
        }
        expect = expect
            .checked_mul(dim)
            .ok_or_else(|| PyValueError::new_err("stride overflow"))?;
    }
    Ok(())
}

fn data_ptr<T>(t: &DLTensor) -> *const T {
    t.data
        .wrapping_add(t.byte_offset as usize)
        .cast::<T>()
        .cast_const()
}

pub(crate) struct HostF64 {
    data: Vec<f64>,
    shape: Vec<i64>,
}

pub(crate) fn take_f64(obj: &Bound<'_, PyAny>) -> PyResult<HostF64> {
    let cap = capsule_of(obj)?;
    with_dltensor(&cap, |t| {
        if !is_f64(t) {
            return Err(PyValueError::new_err("expected float64 dlpack tensor"));
        }
        let shape = shape_of(t)?.to_vec();
        let n = numel(&shape)?;
        require_c_contiguous(t, &shape)?;
        if is_cuda(t) {
            let ptr = data_ptr::<f64>(t);
            if ptr.is_null() && n != 0 {
                return Err(PyValueError::new_err("null data pointer"));
            }
            let mut data = vec![0.0f64; n];
            gpu::copy_dtoh(
                data.as_mut_ptr().cast(),
                ptr.cast(),
                n * std::mem::size_of::<f64>(),
            )?;
            return Ok(HostF64 { data, shape });
        }
        if !is_host(t) {
            return Err(PyValueError::new_err("unsupported dlpack device"));
        }
        let ptr = data_ptr::<f64>(t);
        if ptr.is_null() && n != 0 {
            return Err(PyValueError::new_err("null data pointer"));
        }
        let data = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
        Ok(HostF64 { data, shape })
    })
}

fn take_mask(obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<bool>> {
    let cap = capsule_of(obj)?;
    with_dltensor(&cap, |t| {
        if !is_host(t) {
            return Err(PyValueError::new_err("mask must be host-accessible"));
        }
        let shape = shape_of(t)?;
        if numel(shape)? != n {
            return Err(PyValueError::new_err("mask length must be n"));
        }
        require_c_contiguous(t, shape)?;
        if is_i32(t) {
            let ptr = data_ptr::<i32>(t);
            let s = unsafe { std::slice::from_raw_parts(ptr, n) };
            return Ok(s.iter().map(|&v| v != 0).collect());
        }
        if is_f64(t) {
            let ptr = data_ptr::<f64>(t);
            let s = unsafe { std::slice::from_raw_parts(ptr, n) };
            return Ok(s.iter().map(|&v| v != 0.0).collect());
        }
        Err(PyValueError::new_err("mask must be int32 or float64"))
    })
}

fn parse_xyz(buf: &HostF64) -> PyResult<(Vec<[f64; 3]>, usize, usize)> {
    let (n, frames) = match buf.shape.as_slice() {
        [n, 3] if *n > 0 => (*n as usize, 1usize),
        [f, n, 3] if *f > 0 && *n > 0 => (*n as usize, *f as usize),
        [n] if *n > 0 && *n % 3 == 0 => ((*n as usize) / 3, 1usize),
        _ => {
            return Err(PyValueError::new_err(
                "xyz must be float64 shape (n, 3), (n_frames, n, 3), or (n*3,)",
            ));
        }
    };
    if buf.data.len() != n * frames * 3 {
        return Err(PyValueError::new_err("xyz size mismatch"));
    }
    let pts = buf
        .data
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    Ok((pts, n, frames))
}

pub(crate) fn parse_cell(buf: &HostF64) -> PyResult<(Cell, lc_cell)> {
    let d = &buf.data;
    let (cell, raw) = match buf.shape.as_slice() {
        [3] => (
            Cell::ortho(d[0], d[1], d[2]).map_err(|e| PyValueError::new_err(e.to_string()))?,
            lc_cell {
                ax: d[0],
                ay: 0.0,
                az: 0.0,
                bx: 0.0,
                by: d[1],
                bz: 0.0,
                cx: 0.0,
                cy: 0.0,
                cz: d[2],
                ox: 0.0,
                oy: 0.0,
                oz: 0.0,
            },
        ),
        [3, 3] => (
            Cell::from_vectors(
                [d[0], d[1], d[2]],
                [d[3], d[4], d[5]],
                [d[6], d[7], d[8]],
                [0.0, 0.0, 0.0],
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
            lc_cell {
                ax: d[0],
                ay: d[1],
                az: d[2],
                bx: d[3],
                by: d[4],
                bz: d[5],
                cx: d[6],
                cy: d[7],
                cz: d[8],
                ox: 0.0,
                oy: 0.0,
                oz: 0.0,
            },
        ),
        [4, 3] => (
            Cell::from_vectors(
                [d[0], d[1], d[2]],
                [d[3], d[4], d[5]],
                [d[6], d[7], d[8]],
                [d[9], d[10], d[11]],
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
            lc_cell {
                ax: d[0],
                ay: d[1],
                az: d[2],
                bx: d[3],
                by: d[4],
                bz: d[5],
                cx: d[6],
                cy: d[7],
                cz: d[8],
                ox: d[9],
                oy: d[10],
                oz: d[11],
            },
        ),
        _ => {
            return Err(PyValueError::new_err(
                "cell must be float64 shape (3,) ortho, (3, 3) lattice rows, or (4, 3) rows plus origin",
            ));
        }
    };
    Ok((cell, raw))
}

fn to_pydlpack<A>(py: Python<'_>, arr: A) -> PyResult<Py<PyAny>>
where
    DLPackTensor: TryFrom<A>,
    <DLPackTensor as TryFrom<A>>::Error: std::fmt::Display,
{
    let tensor = DLPackTensor::try_from(arr)
        .map_err(|e| PyRuntimeError::new_err(format!("ndarray -> dlpack: {e}")))?;
    gpu::to_stream_dlpack(py, tensor)
}

/// Periodic linked-cell k-nearest search.
///
/// `xyz` and `cell` are any `__dlpack__()` objects. `xyz` is float64
/// `(n, 3)` or `(n_frames, n, 3)`. `cell` is float64 `(3,)` (ortho
/// lengths), `(3, 3)` lattice rows, or `(4, 3)` rows plus origin, on
/// any device. `mask`, if given, is length `n`.
///
/// Returns `(indices, dist2)` as two DLPack tensors. Indices are int32
/// (`-1` unused). `dist2` is float64 (`NaN` unused). Shape is `(n, k)`
/// or `(n_frames, n, k)` when `xyz` is `(n_frames, n, 3)`.
///
/// Host tensors run the rayon CPU walk with the GIL detached. CUDA
/// `xyz` is passed by device pointer into `lc_gpu_*`. A CUDA `cell`
/// is inverted on device; the host only reads the four launch ints.
#[pyfunction]
#[pyo3(signature = (xyz, cell, k, mask=None, cell_hint=None))]
fn knearest<'py>(
    py: Python<'py>,
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
    k: usize,
    mask: Option<&Bound<'py, PyAny>>,
    cell_hint: Option<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    if k == 0 {
        return Err(PyValueError::new_err("k must be at least 1"));
    }
    let xyz_cuda = peek_cuda(xyz)?;
    if xyz_cuda {
        return gpu::knearest_cuda(py, xyz, cell, k, mask, cell_hint, 0);
    }
    let cell_buf = take_f64(cell)?;
    let (sim, _) = parse_cell(&cell_buf)?;
    let xyz_buf = take_f64(xyz)?;
    let (pts, n, n_frames) = parse_xyz(&xyz_buf)?;
    let mask_vec = match mask {
        None => None,
        Some(m) => Some(take_mask(m, n)?),
    };
    let need = n
        .checked_mul(k)
        .and_then(|v| v.checked_mul(n_frames))
        .ok_or_else(|| PyValueError::new_err("n * k * n_frames overflows"))?;
    let mut nn = vec![-1i32; need];
    let mut d2 = vec![f64::NAN; need];
    let hint = cell_hint.filter(|h| *h > 0.0);
    py.detach(|| {
        if n_frames == 1 {
            knearest_into_d2(
                &pts,
                &sim,
                k,
                mask_vec.as_deref(),
                hint,
                &mut nn,
                Some(&mut d2),
            )
        } else {
            knearest_into_many(
                &pts,
                n,
                n_frames,
                &sim,
                k,
                mask_vec.as_deref(),
                hint,
                &mut nn,
                Some(&mut d2),
            )
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })?;
    if n_frames == 1 {
        let nn_a = Array2::from_shape_vec((n, k), nn)
            .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
        let d2_a = Array2::from_shape_vec((n, k), d2)
            .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
        Ok((
            to_pydlpack(py, nn_a)?,
            to_pydlpack(py, d2_a)?,
        ))
    } else {
        let nn_a = Array3::from_shape_vec((n_frames, n, k), nn)
            .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
        let d2_a = Array3::from_shape_vec((n_frames, n, k), d2)
            .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
        Ok((
            to_pydlpack(py, nn_a)?,
            to_pydlpack(py, d2_a)?,
        ))
    }
}

#[pyfunction]
fn gpu_available() -> bool {
    gpu::available()
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knearest, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_class::<PyDLPack>()?;
    m.add_class::<gpu::StreamDlpack>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
