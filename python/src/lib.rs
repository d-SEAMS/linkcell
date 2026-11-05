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
use linkcell::{
    cell_graph, cell_grid, halo, halo_width, knearest_into_d2, knearest_into_many, lc_cell,
    pairs_within_columns, partition, Cell, Method, NeighborCalc, PairColumns, PartRequest,
    VesinOptions,
};
use ndarray::{Array1, Array2, Array3};

mod gpu;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyModule};

type PairPack = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

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
        Ok((to_pydlpack(py, nn_a)?, to_pydlpack(py, d2_a)?))
    } else {
        let nn_a = Array3::from_shape_vec((n_frames, n, k), nn)
            .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
        let d2_a = Array3::from_shape_vec((n_frames, n, k), d2)
            .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
        Ok((to_pydlpack(py, nn_a)?, to_pydlpack(py, d2_a)?))
    }
}

/// Cutoff pairs in the vesin / tonari `ijS` layout.
///
/// `xyz` is float64 `(n, 3)`. `cell` is the same shapes as
/// [`knearest`]. Each row is one atom-image whose squared distance is
/// strictly below `cutoff` squared. `S` is the integer shift of the
/// target in the caller's basis. `half` keeps one side of
/// `(i, j, S)` and `(j, i, -S)`.
///
/// Returns `(i, j, S, dist2)`. `i` and `j` are int32 `(n_pairs,)`,
/// `S` is int32 `(n_pairs, 3)`, `dist2` is float64 `(n_pairs,)`.
/// A cutoff neighbour list is this function. `knearest` is the
/// k-nearest list.
#[pyfunction]
#[pyo3(name = "pairs_within", signature = (xyz, cell, cutoff, mask=None, cell_hint=None, half=false))]
fn pairs_within_py<'py>(
    py: Python<'py>,
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
    cutoff: f64,
    mask: Option<&Bound<'py, PyAny>>,
    cell_hint: Option<f64>,
    half: bool,
) -> PyResult<PairPack> {
    if peek_cuda(xyz)? {
        return Err(PyValueError::new_err(
            "pairs_within reads host tensors; move xyz to the CPU",
        ));
    }
    let cell_buf = take_f64(cell)?;
    let (sim, _) = parse_cell(&cell_buf)?;
    let xyz_buf = take_f64(xyz)?;
    let (pts, n, n_frames) = parse_xyz(&xyz_buf)?;
    if n_frames != 1 {
        return Err(PyValueError::new_err(
            "pairs_within takes one frame; xyz shape is (n, 3)",
        ));
    }
    let mask_vec = match mask {
        None => None,
        Some(m) => Some(take_mask(m, n)?),
    };
    let hint = cell_hint.filter(|h| *h > 0.0);
    let mut cols = PairColumns::default();
    py.detach(|| {
        pairs_within_columns(
            &pts,
            &sim,
            cutoff,
            mask_vec.as_deref(),
            hint,
            half,
            &mut cols,
        )
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n_pairs = cols.len();
    // Same allocation read as flat int32: `[i32; 3]` and `i32` share an
    // alignment, and three times the capacity is the same byte count.
    let shift = {
        let mut rows = std::mem::ManuallyDrop::new(cols.shift);
        unsafe {
            Vec::from_raw_parts(
                rows.as_mut_ptr().cast::<i32>(),
                rows.len() * 3,
                rows.capacity() * 3,
            )
        }
    };
    let i_a = Array1::from_vec(cols.i);
    let j_a = Array1::from_vec(cols.j);
    let s_a = Array2::from_shape_vec((n_pairs, 3), shift)
        .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
    let d_a = Array1::from_vec(cols.dist2);
    Ok((
        to_pydlpack(py, i_a)?,
        to_pydlpack(py, j_a)?,
        to_pydlpack(py, s_a)?,
        to_pydlpack(py, d_a)?,
    ))
}

fn parse_method(name: &str) -> PyResult<Method> {
    match name {
        "default" => Ok(linkcell::DEFAULT_METHOD),
        "morton" => Ok(Method::Morton),
        "hilbert" => Ok(Method::Hilbert),
        "rcb" => Ok(Method::Rcb),
        "hilbert-fm" => Ok(Method::HilbertFm),
        "rcb-fm" => Ok(Method::RcbFm),
        "multi" => Ok(Method::Multi),
        "scotch" => Ok(Method::Scotch),
        _ => Err(PyValueError::new_err(
            "method is default, morton, hilbert, rcb, hilbert-fm, rcb-fm, multi, or scotch",
        )),
    }
}

fn frame_points<'py>(
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
) -> PyResult<(Vec<[f64; 3]>, Cell)> {
    if peek_cuda(xyz)? {
        return Err(PyValueError::new_err(
            "partition reads host tensors; move xyz to the CPU",
        ));
    }
    let cell_buf = take_f64(cell)?;
    let (sim, _) = parse_cell(&cell_buf)?;
    let xyz_buf = take_f64(xyz)?;
    let (pts, _n, n_frames) = parse_xyz(&xyz_buf)?;
    if n_frames != 1 {
        return Err(PyValueError::new_err("xyz must be one frame of shape (n, 3)"));
    }
    Ok((pts, sim))
}

/// Point counts on the linked-cell mesh.
///
/// Returns `(nxyz, bin_width, weight)`. `weight[i]` is the point count
/// of cell `i` in the order `((iz * ny) + iy) * nx + ix`.
#[pyfunction]
#[pyo3(name = "cell_grid", signature = (xyz, cell, cutoff, mask=None, cell_hint=None))]
fn cell_grid_py<'py>(
    py: Python<'py>,
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
    cutoff: f64,
    mask: Option<&Bound<'py, PyAny>>,
    cell_hint: Option<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let (pts, sim) = frame_points(xyz, cell)?;
    let n = pts.len();
    let mask_vec = match mask {
        None => None,
        Some(m) => Some(take_mask(m, n)?),
    };
    let grid = py
        .detach(|| cell_grid(&pts, &sim, cutoff, mask_vec.as_deref(), cell_hint))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nxyz = Array1::from_vec(vec![grid.nx, grid.ny, grid.nz]);
    let width = Array1::from_vec(grid.bin_width.to_vec());
    let weight = Array1::from_vec(grid.weight);
    Ok((
        to_pydlpack(py, nxyz)?,
        to_pydlpack(py, width)?,
        to_pydlpack(py, weight)?,
    ))
}

/// Periodic adjacency. Edge weights are pair counts from `pairs_within`.
///
/// Returns `(nxyz, bin_width, weight, offsets, neighbor, edge_weight)`.
#[pyfunction]
#[pyo3(name = "cell_graph", signature = (xyz, cell, cutoff, mask=None, cell_hint=None))]
fn cell_graph_py<'py>(
    py: Python<'py>,
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
    cutoff: f64,
    mask: Option<&Bound<'py, PyAny>>,
    cell_hint: Option<f64>,
) -> PyResult<(
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
)> {
    let (pts, sim) = frame_points(xyz, cell)?;
    let n = pts.len();
    let mask_vec = match mask {
        None => None,
        Some(m) => Some(take_mask(m, n)?),
    };
    let graph = py
        .detach(|| cell_graph(&pts, &sim, cutoff, mask_vec.as_deref(), cell_hint))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nxyz = Array1::from_vec(vec![graph.grid.nx, graph.grid.ny, graph.grid.nz]);
    let width = Array1::from_vec(graph.grid.bin_width.to_vec());
    let weight = Array1::from_vec(graph.grid.weight);
    let offsets = Array1::from_vec(graph.offsets);
    let neighbor = Array1::from_vec(graph.neighbor);
    let edge = Array1::from_vec(graph.edge_weight);
    Ok((
        to_pydlpack(py, nxyz)?,
        to_pydlpack(py, width)?,
        to_pydlpack(py, weight)?,
        to_pydlpack(py, offsets)?,
        to_pydlpack(py, neighbor)?,
        to_pydlpack(py, edge)?,
    ))
}

/// Domain id of each cell.
///
/// `method` is `default`, `morton`, `hilbert`, `rcb`, `hilbert-fm`,
/// `rcb-fm`, `multi`, or `scotch`. `balance` of `None` selects the
/// derived bound `1 + parts * heaviest_cell / total_weight`.
/// `device` runs the curve split on CUDA and raises when that device
/// is absent.
#[pyfunction]
#[pyo3(
    name = "partition",
    signature = (xyz, cell, cutoff, parts, method="default", balance=None, mask=None, cell_hint=None, device=false)
)]
fn partition_py<'py>(
    py: Python<'py>,
    xyz: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyAny>,
    cutoff: f64,
    parts: u32,
    method: &str,
    balance: Option<f64>,
    mask: Option<&Bound<'py, PyAny>>,
    cell_hint: Option<f64>,
    device: bool,
) -> PyResult<Py<PyAny>> {
    let method = parse_method(method)?;
    let (pts, sim) = frame_points(xyz, cell)?;
    let n = pts.len();
    let mask_vec = match mask {
        None => None,
        Some(m) => Some(take_mask(m, n)?),
    };
    let request = PartRequest {
        parts,
        method,
        balance,
        device,
    };
    let assign = py
        .detach(|| partition(&pts, &sim, cutoff, mask_vec.as_deref(), cell_hint, request))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ids: Vec<i32> = assign
        .into_iter()
        .map(|v| i32::try_from(v).map_err(|_| PyRuntimeError::new_err("domain id overflow")))
        .collect::<PyResult<_>>()?;
    to_pydlpack(py, Array1::from_vec(ids))
}

/// Three bin radii for `cutoff * depth`.
#[pyfunction]
#[pyo3(name = "halo_width")]
fn halo_width_py(cutoff: f64, depth: u32, bin_width: Vec<f64>) -> PyResult<(u32, u32, u32)> {
    if bin_width.len() != 3 {
        return Err(PyValueError::new_err("bin_width has length 3"));
    }
    let width = halo_width(cutoff, depth, [bin_width[0], bin_width[1], bin_width[2]])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((width[0], width[1], width[2]))
}

/// Ghost cell ids of `domain`.
#[pyfunction]
#[pyo3(name = "halo")]
fn halo_py<'py>(
    py: Python<'py>,
    nxyz: Vec<i32>,
    assign: Vec<i32>,
    domain: u32,
    width: Vec<u32>,
) -> PyResult<Py<PyAny>> {
    if nxyz.len() != 3 || width.len() != 3 {
        return Err(PyValueError::new_err("nxyz and width have length 3"));
    }
    let ncell = (nxyz[0] as usize)
        .checked_mul(nxyz[1] as usize)
        .and_then(|v| v.checked_mul(nxyz[2] as usize))
        .ok_or_else(|| PyValueError::new_err("nxyz overflow"))?;
    if assign.len() != ncell {
        return Err(PyValueError::new_err("assign length must be nx * ny * nz"));
    }
    let owned: Vec<u32> = assign
        .into_iter()
        .map(|v| u32::try_from(v).map_err(|_| PyValueError::new_err("negative domain")))
        .collect::<PyResult<_>>()?;
    let grid = linkcell::CellGrid {
        nx: nxyz[0],
        ny: nxyz[1],
        nz: nxyz[2],
        bin_width: [0.0; 3],
        weight: vec![0; ncell],
    };
    let ghosts = halo(&grid, &owned, domain, [width[0], width[1], width[2]])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ids: Vec<i32> = ghosts
        .into_iter()
        .map(|v| i32::try_from(v).map_err(|_| PyRuntimeError::new_err("ghost id overflow")))
        .collect::<PyResult<_>>()?;
    to_pydlpack(py, Array1::from_vec(ids))
}

#[pyfunction]
fn gpu_available() -> bool {
    gpu::available()
}

fn periodic_flags(obj: &Bound<'_, PyAny>) -> PyResult<[bool; 3]> {
    if let Ok(flag) = obj.extract::<bool>() {
        return Ok([flag, flag, flag]);
    }
    let flags: Vec<bool> = obj
        .extract()
        .map_err(|_| PyValueError::new_err("periodic must be a bool or three bools"))?;
    if flags.len() != 3 {
        return Err(PyValueError::new_err(
            "periodic must be a bool or three bools",
        ));
    }
    Ok([flags[0], flags[1], flags[2]])
}

/// vesin-compatible calculator. `compute` returns one DLPack tensor per
/// letter of `quantities` (`i`, `j`, `P`, `S`, `d`, `D`).
#[pyclass(name = "NeighborList", module = "linkcell._lib", subclass)]
struct PyNeighborList {
    cutoff: f64,
    full: bool,
    sorted: bool,
    skin: f64,
    threads: i32,
    calc: NeighborCalc,
}

#[pymethods]
impl PyNeighborList {
    #[new]
    #[pyo3(signature = (cutoff, full_list, sorted=false, skin=0.0, n_threads=0))]
    fn new(
        cutoff: f64,
        full_list: bool,
        sorted: bool,
        skin: f64,
        n_threads: i32,
    ) -> PyResult<Self> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(PyValueError::new_err("cutoff must be finite and positive"));
        }
        if !skin.is_finite() {
            return Err(PyValueError::new_err("skin must be finite"));
        }
        if n_threads < 0 {
            return Err(PyValueError::new_err("n_threads must be zero or positive"));
        }
        Ok(Self {
            cutoff,
            full: full_list,
            sorted,
            skin,
            threads: n_threads,
            calc: NeighborCalc::default(),
        })
    }

    /// Rows for `quantities`, in that letter order.
    #[pyo3(signature = (points, cell, periodic=None, quantities="ij"))]
    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
        cell: &Bound<'py, PyAny>,
        periodic: Option<&Bound<'py, PyAny>>,
        quantities: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let periodic = match periodic {
            None => [true, true, true],
            Some(flag) => periodic_flags(flag)?,
        };
        let xyz_buf = take_f64(points)?;
        let (pts, _n, frames) = parse_xyz(&xyz_buf)?;
        if frames != 1 {
            return Err(PyValueError::new_err(
                "points must be one frame of shape (n, 3)",
            ));
        }
        let cell_buf = take_f64(cell)?;
        let (cell, _) = parse_cell(&cell_buf)?;
        let opt = VesinOptions::cpu(
            self.cutoff,
            self.full,
            self.skin,
            self.threads,
            quantities.contains('S'),
            quantities.contains('d'),
            quantities.contains('D'),
        )
        .sorted(self.sorted);
        py.detach(|| self.calc.compute(&pts, &cell, periodic, opt))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut out = Vec::new();
        for letter in quantities.chars() {
            let tensor = match letter {
                'i' => to_pydlpack(
                    py,
                    Array1::from_vec(
                        self.calc
                            .pairs()
                            .iter()
                            .map(|p| p[0] as u64)
                            .collect::<Vec<_>>(),
                    ),
                )?,
                'j' => to_pydlpack(
                    py,
                    Array1::from_vec(
                        self.calc
                            .pairs()
                            .iter()
                            .map(|p| p[1] as u64)
                            .collect::<Vec<_>>(),
                    ),
                )?,
                'P' => {
                    let n = self.calc.pairs().len();
                    let flat: Vec<u64> = self
                        .calc
                        .pairs()
                        .iter()
                        .flat_map(|p| [p[0] as u64, p[1] as u64])
                        .collect();
                    let arr = Array2::from_shape_vec((n, 2), flat)
                        .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
                    to_pydlpack(py, arr)?
                }
                'S' => {
                    let n = self.calc.shifts().len();
                    let flat: Vec<i32> = self
                        .calc
                        .shifts()
                        .iter()
                        .flat_map(|s| [s[0], s[1], s[2]])
                        .collect();
                    let arr = Array2::from_shape_vec((n, 3), flat)
                        .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
                    to_pydlpack(py, arr)?
                }
                'd' => to_pydlpack(py, Array1::from_vec(self.calc.distances().to_vec()))?,
                'D' => {
                    let n = self.calc.vectors().len();
                    let flat: Vec<f64> = self
                        .calc
                        .vectors()
                        .iter()
                        .flat_map(|v| [v[0], v[1], v[2]])
                        .collect();
                    let arr = Array2::from_shape_vec((n, 3), flat)
                        .map_err(|e| PyRuntimeError::new_err(format!("shape: {e}")))?;
                    to_pydlpack(py, arr)?
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown quantity '{letter}', expected i, j, P, S, d, or D"
                    )))
                }
            };
            out.push(tensor);
        }
        Ok(out)
    }
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knearest, m)?)?;
    m.add_function(wrap_pyfunction!(pairs_within_py, m)?)?;
    m.add_function(wrap_pyfunction!(cell_grid_py, m)?)?;
    m.add_function(wrap_pyfunction!(cell_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(partition_py, m)?)?;
    m.add_function(wrap_pyfunction!(halo_width_py, m)?)?;
    m.add_function(wrap_pyfunction!(halo_py, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_class::<PyNeighborList>()?;
    m.add_class::<PyDLPack>()?;
    m.add_class::<gpu::StreamDlpack>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
