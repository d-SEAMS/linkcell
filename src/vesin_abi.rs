//! vesin-compatible CPU entry points.
//!
//! [`vesin_neighbors`] has the same layout and calling convention as
//! vesin's C API: row-major box, `ij` / `S` / distance / vector outputs,
//! half or full lists, and a Verlet skin. Fully periodic cells run here.
//! A non-periodic axis or a CUDA device is forwarded to the platform
//! vesin library (`libvesin.so`, `libvesin.dylib`, or `libvesin.dll`)
//! when the loader can see it.

use std::cell::RefCell;
use std::ffi::CString;
#[cfg(unix)]
use std::os::raw::c_int;
use std::os::raw::{c_char, c_void};
#[cfg(feature = "parallel")]
use std::sync::Mutex;

use crate::cell::Cell;
use crate::pairs::{pairs_within_columns, PairColumns};
use crate::skin::SkinList;
use crate::Error;

const VESIN_CPU: i32 = 1;
const VESIN_CUDA: i32 = 2;

/// Options block, identical in layout to `VesinOptions`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VesinOptions {
    /// Spherical cutoff.
    pub cutoff: f64,
    /// Both `(i, j, S)` and `(j, i, -S)` when true.
    pub full: bool,
    /// Group rows by `i` when true.
    pub sorted: bool,
    _pad0: [u8; 6],
    /// Full-list width. Zero means no padding.
    pub padded_to: usize,
    /// `VesinCellList` (2) or `VesinAutoAlgorithm` (0). Brute force is CPU-unsupported upstream.
    pub algorithm: i32,
    _pad1: u32,
    /// Verlet skin. Negative selects `0.1 * cutoff` on the CPU.
    pub skin: f64,
    /// Thread count. Zero uses the process default.
    pub n_threads: i32,
    /// Store integer shifts.
    pub return_shifts: bool,
    /// Store distances (not squared).
    pub return_distances: bool,
    /// Store shifted vectors `q + H S - p`.
    pub return_vectors: bool,
    _pad2: u8,
}

impl VesinOptions {
    /// CPU options. `skin < 0` is resolved inside [`NeighborCalc::compute`].
    pub fn cpu(
        cutoff: f64,
        full: bool,
        skin: f64,
        n_threads: i32,
        shifts: bool,
        distances: bool,
        vectors: bool,
    ) -> Self {
        Self {
            cutoff,
            full,
            sorted: false,
            _pad0: [0; 6],
            padded_to: 0,
            algorithm: 2,
            _pad1: 0,
            skin,
            n_threads,
            return_shifts: shifts,
            return_distances: distances,
            return_vectors: vectors,
            _pad2: 0,
        }
    }

    /// Ask for rows grouped by the first index.
    pub fn sorted(mut self, sorted: bool) -> Self {
        self.sorted = sorted;
        self
    }

    /// Pad a full list to this many rows per point.
    pub fn padded_to(mut self, n: usize) -> Self {
        self.padded_to = n;
        self
    }
}

/// CPU or CUDA tag, identical in layout to `VesinDevice`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VesinDevice {
    /// 1 is the CPU, 2 is CUDA.
    pub kind: i32,
    /// Device index. Zero on the CPU.
    pub device_id: i32,
}

/// Output list, identical in layout to `VesinNeighborList`.
#[repr(C)]
pub struct VesinNeighborList {
    /// Number of rows.
    pub length: usize,
    /// Device the buffers live on.
    pub device: VesinDevice,
    /// `length` pairs.
    pub pairs: *mut [usize; 2],
    /// `length` shifts, or null.
    pub shifts: *mut [i32; 3],
    /// `length` distances, or null.
    pub distances: *mut f64,
    /// `length` vectors, or null.
    pub vectors: *mut [f64; 3],
    /// Owned [`Opaque`] state.
    pub opaque: *mut c_void,
}

struct Opaque {
    calc: NeighborCalc,
}

/// Reused buffers for repeated [`vesin_neighbors`] calls.
#[derive(Default)]
pub struct NeighborCalc {
    skin_list: Option<SkinList>,
    skin_tag: (u64, u64, bool),
    pairs: Vec<[usize; 2]>,
    shifts: Vec<[i32; 3]>,
    distances: Vec<f64>,
    vectors: Vec<[f64; 3]>,
}

impl NeighborCalc {
    /// Index pairs from the last [`Self::compute`].
    pub fn pairs(&self) -> &[[usize; 2]] {
        &self.pairs
    }

    /// Integer shifts, empty unless they were requested.
    pub fn shifts(&self) -> &[[i32; 3]] {
        &self.shifts
    }

    /// Distances, empty unless they were requested.
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Shifted vectors, empty unless they were requested.
    pub fn vectors(&self) -> &[[f64; 3]] {
        &self.vectors
    }
}

impl NeighborCalc {
    /// One neighbour list.
    ///
    /// `periodic` is three axis flags. A false flag returns
    /// [`Error::Partial`] from this function; the C entry tries vesin.
    pub fn compute(
        &mut self,
        xyz: &[[f64; 3]],
        cell: &Cell,
        periodic: [bool; 3],
        mut opt: VesinOptions,
    ) -> Result<(), Error> {
        if !periodic[0] || !periodic[1] || !periodic[2] {
            return Err(Error::Partial);
        }
        if !opt.cutoff.is_finite() || opt.cutoff <= 0.0 {
            return Err(Error::BadCutoff);
        }
        if !opt.skin.is_finite() {
            return Err(Error::BadCutoff);
        }
        if opt.n_threads < 0 {
            return Err(Error::BadCutoff);
        }
        if opt.padded_to > 0 && !opt.full {
            return Err(Error::Padded);
        }
        for p in xyz {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(Error::NonFinite);
            }
        }
        if xyz.is_empty() {
            return Err(Error::Empty);
        }
        if opt.skin < 0.0 {
            opt.skin = 0.1 * opt.cutoff;
        }
        let threads = if opt.n_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            opt.n_threads as usize
        };
        let half = !opt.full;
        let want_vec = opt.return_vectors;
        with_threads(threads, || {
            let cols = if opt.skin > 0.0 {
                let tag = (opt.cutoff.to_bits(), opt.skin.to_bits(), half);
                if self.skin_tag != tag {
                    self.skin_list = None;
                    self.skin_tag = tag;
                }
                if self.skin_list.is_none() {
                    self.skin_list = Some(SkinList::new(*cell, opt.cutoff, opt.skin, half)?);
                }
                let list = self.skin_list.as_mut().unwrap();
                list.update(xyz, want_vec)?;
                list.columns().clone()
            } else {
                self.skin_list = None;
                let mut cols = PairColumns::default();
                pairs_within_columns(xyz, cell, opt.cutoff, None, None, half, &mut cols)?;
                cols
            };
            self.emit(xyz, cell, &cols, opt)
        })
    }

    fn emit(
        &mut self,
        xyz: &[[f64; 3]],
        cell: &Cell,
        cols: &PairColumns,
        opt: VesinOptions,
    ) -> Result<(), Error> {
        let n_atoms = xyz.len();
        let vectors_ready = opt.return_vectors
            && self
                .skin_list
                .as_ref()
                .is_some_and(|s| s.vectors().len() == cols.len());
        let cached_vec = if vectors_ready {
            self.skin_list.as_ref().unwrap().vectors().to_vec()
        } else {
            Vec::new()
        };
        self.pairs.clear();
        self.shifts.clear();
        self.distances.clear();
        self.vectors.clear();
        if opt.padded_to > 0 {
            return self.emit_padded(xyz, cell, cols, opt, &cached_vec, n_atoms);
        }
        let mut order: Vec<usize> = (0..cols.len()).collect();
        if opt.sorted {
            order.sort_by_key(|&k| cols.i[k]);
        }
        self.pairs.reserve(order.len());
        if opt.return_shifts {
            self.shifts.reserve(order.len());
        }
        if opt.return_distances {
            self.distances.reserve(order.len());
        }
        if opt.return_vectors {
            self.vectors.reserve(order.len());
        }
        for k in order {
            self.push_row(xyz, cell, cols, k, opt, &cached_vec);
        }
        Ok(())
    }

    fn emit_padded(
        &mut self,
        xyz: &[[f64; 3]],
        cell: &Cell,
        cols: &PairColumns,
        opt: VesinOptions,
        cached_vec: &[[f64; 3]],
        n_atoms: usize,
    ) -> Result<(), Error> {
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
        for k in 0..cols.len() {
            let i = cols.i[k] as usize;
            if i >= n_atoms {
                return Err(Error::Overflow);
            }
            buckets[i].push(k);
        }
        for (i, bucket) in buckets.iter().enumerate() {
            if bucket.len() > opt.padded_to {
                let _ = i;
                return Err(Error::Padded);
            }
        }
        let rows = n_atoms * opt.padded_to;
        self.pairs.reserve(rows);
        if opt.return_shifts {
            self.shifts.reserve(rows);
        }
        if opt.return_distances {
            self.distances.reserve(rows);
        }
        if opt.return_vectors {
            self.vectors.reserve(rows);
        }
        for (i, bucket) in buckets.iter().enumerate() {
            for &k in bucket {
                self.push_row(xyz, cell, cols, k, opt, cached_vec);
            }
            for _ in bucket.len()..opt.padded_to {
                self.pairs.push([i, usize::MAX]);
                if opt.return_shifts {
                    self.shifts.push([0, 0, 0]);
                }
                if opt.return_distances {
                    self.distances.push(0.0);
                }
                if opt.return_vectors {
                    self.vectors.push([0.0, 0.0, 0.0]);
                }
            }
        }
        Ok(())
    }

    fn push_row(
        &mut self,
        xyz: &[[f64; 3]],
        cell: &Cell,
        cols: &PairColumns,
        k: usize,
        opt: VesinOptions,
        cached_vec: &[[f64; 3]],
    ) {
        let i = cols.i[k] as usize;
        let j = cols.j[k] as usize;
        self.pairs.push([i, j]);
        if opt.return_shifts {
            self.shifts.push(cols.shift[k]);
        }
        if opt.return_distances {
            self.distances.push(cols.dist2[k].sqrt());
        }
        if opt.return_vectors {
            if !cached_vec.is_empty() {
                self.vectors.push(cached_vec[k]);
            } else {
                let s = cols.shift[k];
                let t = cell.lattice_shift(s[0], s[1], s[2]);
                let p = xyz[i];
                let q = xyz[j];
                self.vectors
                    .push([q[0] + t[0] - p[0], q[1] + t[1] - p[1], q[2] + t[2] - p[2]]);
            }
        }
    }
}

fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    {
        struct Slot {
            n: usize,
            pool: rayon::ThreadPool,
        }
        static SLOTS: Mutex<Vec<Slot>> = Mutex::new(Vec::new());
        let n = n.max(1);
        let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
        if slots.iter().all(|s| s.n != n) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap_or_else(|_| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .expect("rayon")
                });
            slots.push(Slot { n, pool });
        }
        let slot = slots.iter().find(|s| s.n == n).unwrap();
        // The pool outlives the mutex; install while the lock is held so
        // the slot cannot move.
        slot.pool.install(f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        f()
    }
}

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::new("").expect("empty"));
}

fn set_last(msg: &str) -> *const c_char {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("error").expect("plain"));
    LAST.with(|slot| {
        *slot.borrow_mut() = c;
        slot.borrow().as_ptr()
    })
}

fn box_of(rows: [[f64; 3]; 3]) -> Result<Cell, Error> {
    Cell::from_vectors(rows[0], rows[1], rows[2], [0.0, 0.0, 0.0]).map_err(|_| Error::BadBox)
}

type VesinFn = unsafe extern "C" fn(
    *const [f64; 3],
    usize,
    *const [[f64; 3]; 3],
    *const [bool; 3],
    VesinDevice,
    VesinOptions,
    *mut VesinNeighborList,
    *mut *const c_char,
) -> i32;

fn foreign_vesin() -> Option<VesinFn> {
    use std::sync::OnceLock;
    static FOREIGN: OnceLock<Option<VesinFn>> = OnceLock::new();
    *FOREIGN.get_or_init(load_foreign_vesin)
}

#[cfg(all(unix, target_os = "macos"))]
const VESIN_LIB: &[u8] = b"libvesin.dylib\0";
#[cfg(all(unix, not(target_os = "macos")))]
const VESIN_LIB: &[u8] = b"libvesin.so\0";

#[cfg(unix)]
fn load_foreign_vesin() -> Option<VesinFn> {
    unsafe {
        extern "C" {
            fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        }
        let handle = dlopen(VESIN_LIB.as_ptr().cast(), 1);
        if handle.is_null() {
            return None;
        }
        let sym = dlsym(handle, b"vesin_neighbors\0".as_ptr().cast());
        if sym.is_null() || sym == lc_vesin_neighbors as *mut c_void {
            return None;
        }
        Some(std::mem::transmute::<*mut c_void, VesinFn>(sym))
    }
}

#[cfg(windows)]
fn load_foreign_vesin() -> Option<VesinFn> {
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryA(name: *const u8) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        }
        let handle = LoadLibraryA(b"libvesin.dll\0".as_ptr());
        if handle.is_null() {
            return None;
        }
        let sym = GetProcAddress(handle, b"vesin_neighbors\0".as_ptr());
        if sym.is_null() || sym == lc_vesin_neighbors as *mut c_void {
            return None;
        }
        Some(std::mem::transmute::<*mut c_void, VesinFn>(sym))
    }
}

#[cfg(not(any(unix, windows)))]
fn load_foreign_vesin() -> Option<VesinFn> {
    None
}

/// Fully periodic CPU lists are computed here.
///
/// The symbol is `lc_vesin_neighbors`, not `vesin_neighbors`. A program
/// that links both this crate and vesin keeps vesin's symbol. A drop-in
/// object that calls this function is `include/vesin_alias.c`.
///
/// # Safety
///
/// `points` is `n_points` coordinate triples. `box_` is three lattice
/// rows. `periodic` is three booleans. `neighbors` is a zero-initialised
/// or previously returned list. `error_message` is writable.
#[no_mangle]
pub unsafe extern "C" fn lc_vesin_neighbors(
    points: *const [f64; 3],
    n_points: usize,
    box_: *const [[f64; 3]; 3],
    periodic: *const [bool; 3],
    device: VesinDevice,
    options: VesinOptions,
    neighbors: *mut VesinNeighborList,
    error_message: *mut *const c_char,
) -> i32 {
    if error_message.is_null() {
        return 1;
    }
    unsafe {
        *error_message = std::ptr::null();
    }
    if points.is_null() || box_.is_null() || periodic.is_null() || neighbors.is_null() {
        unsafe {
            *error_message = set_last("null pointer");
        }
        return 1;
    }
    let periodic = unsafe { *periodic };
    let forward = device.kind == VESIN_CUDA || !periodic[0] || !periodic[1] || !periodic[2];
    if forward {
        if let Some(f) = foreign_vesin() {
            return unsafe {
                f(
                    points,
                    n_points,
                    box_,
                    &periodic,
                    device,
                    options,
                    neighbors,
                    error_message,
                )
            };
        }
        let msg = if device.kind == VESIN_CUDA {
            Error::Device.to_string()
        } else {
            Error::Partial.to_string()
        };
        unsafe {
            *error_message = set_last(&msg);
        }
        return 1;
    }
    if device.kind != VESIN_CPU && device.kind != 0 {
        unsafe {
            *error_message = set_last(&Error::Device.to_string());
        }
        return 1;
    }
    let rows = unsafe { *box_ };
    let cell = match box_of(rows) {
        Ok(c) => c,
        Err(e) => {
            unsafe {
                *error_message = set_last(&e.to_string());
            }
            return 1;
        }
    };
    let xyz = unsafe { std::slice::from_raw_parts(points, n_points) };
    let opaque = unsafe { &mut *neighbors };
    if opaque.opaque.is_null() {
        let boxed = Box::new(Opaque {
            calc: NeighborCalc::default(),
        });
        opaque.opaque = Box::into_raw(boxed).cast();
    }
    let state = unsafe { &mut *(opaque.opaque as *mut Opaque) };
    if let Err(e) = state.calc.compute(xyz, &cell, periodic, options) {
        unsafe {
            *error_message = set_last(&e.to_string());
        }
        return 1;
    }
    opaque.length = state.calc.pairs.len();
    opaque.device = VesinDevice {
        kind: VESIN_CPU,
        device_id: 0,
    };
    opaque.pairs = state.calc.pairs.as_mut_ptr();
    opaque.shifts = if options.return_shifts {
        state.calc.shifts.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    opaque.distances = if options.return_distances {
        state.calc.distances.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    opaque.vectors = if options.return_vectors {
        state.calc.vectors.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    0
}

/// Release a list produced by [`lc_vesin_neighbors`].
///
/// # Safety
///
/// `neighbors` is null or a list this library filled. Not a list vesin itself allocated.
#[no_mangle]
pub unsafe extern "C" fn lc_vesin_free(neighbors: *mut VesinNeighborList) {
    if neighbors.is_null() {
        return;
    }
    unsafe {
        let nl = &mut *neighbors;
        if !nl.opaque.is_null() {
            drop(Box::from_raw(nl.opaque.cast::<Opaque>()));
        }
        std::ptr::write_bytes(nl, 0, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_match_the_vesin_layout() {
        assert_eq!(std::mem::size_of::<VesinOptions>(), 48);
        assert_eq!(std::mem::size_of::<VesinNeighborList>(), 56);
        assert_eq!(std::mem::align_of::<VesinOptions>(), 8);
    }

    #[test]
    fn cpu_list_matches_columns() {
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let mut calc = NeighborCalc::default();
        let opt = VesinOptions {
            cutoff: 4.0,
            full: true,
            sorted: false,
            _pad0: [0; 6],
            padded_to: 0,
            algorithm: 2,
            _pad1: 0,
            skin: 0.0,
            n_threads: 1,
            return_shifts: true,
            return_distances: true,
            return_vectors: true,
            _pad2: 0,
        };
        calc.compute(&xyz, &cell, [true, true, true], opt).unwrap();
        assert!(!calc.pairs.is_empty());
        assert_eq!(calc.pairs.len(), calc.distances.len());
        assert_eq!(calc.pairs.len(), calc.vectors.len());
        let mut again = NeighborCalc::default();
        let mut opt2 = opt;
        opt2.skin = 0.4;
        again
            .compute(&xyz, &cell, [true, true, true], opt2)
            .unwrap();
        assert_eq!(again.pairs.len(), calc.pairs.len());
    }
}
