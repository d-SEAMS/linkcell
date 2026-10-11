//! Cell-major bins and the certified index-box frontier.
//!
//! Occupants of one bin sit in a contiguous slice (HOOMD / vesin), so the
//! stencil walk is a sequential read. The visited set is a rectangular
//! box in bin index. Every unvisited image lies beyond one of six lattice
//! planes. The perpendicular distance to the nearest of those planes is a
//! lower bound on the unvisited distance, and it is safe to stop once the
//! k-th neighbour is inside it.

use std::cell::RefCell;

use crate::cell::Cell;
use crate::Error;

/// Below this many active points the mesh stays serial. The cubic hot
/// path is 4096 points, and a parallel histogram there costs more than
/// the fold.
#[cfg(feature = "parallel")]
const PARALLEL_MESH: usize = 8_192;

const MAX_CELLS: i64 = 16_777_216;
/// Shrink a cutoff slab so a rounded-up plane cannot drop a pair.
const CERT_REL: f64 = 1.0e-8;
const CERT_ABS: f64 = 1.0e-12;

#[derive(Default)]
struct Recycled {
    frac: Vec<[f64; 3]>,
    folded: Vec<[f64; 3]>,
    slot_frac: Vec<[f64; 3]>,
    slot_folded: Vec<[f64; 3]>,
    key: Vec<u32>,
    offsets: Vec<usize>,
    occupants: Vec<usize>,
    counts: Vec<usize>,
    cursor: Vec<usize>,
    hist: Vec<std::sync::atomic::AtomicUsize>,
}

thread_local! {
    // `const { ... }` is newer than this crate's 1.70 floor.
    #[allow(clippy::missing_const_for_thread_local)]
    static MESH_POOL: RefCell<Option<Recycled>> = RefCell::new(None);
}

fn take_recycled() -> Recycled {
    MESH_POOL.with(|slot| slot.borrow_mut().take().unwrap_or_default())
}

fn recycle(buf: Recycled) {
    const MAX_KEEP: usize = 1 << 20;
    if buf.frac.capacity() > MAX_KEEP || buf.offsets.capacity() > MAX_KEEP {
        return;
    }
    MESH_POOL.with(|slot| {
        let mut guard = slot.borrow_mut();
        if guard.is_none() {
            *guard = Some(buf);
        }
    });
}

/// Fractional bins with cell-major occupants.
pub(crate) struct Mesh {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub widths: [f64; 3],
    pub frac: Vec<[f64; 3]>,
    pub folded: Vec<[f64; 3]>,
    /// `frac` and `folded` of each slot's occupant, in slot order: a walk
    /// in bin order reads them in runs whatever order the atoms came in.
    pub(crate) slot_frac: Vec<[f64; 3]>,
    pub(crate) slot_folded: Vec<[f64; 3]>,
    /// Flat bin of each point, in point order.
    key: Vec<u32>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) occupants: Vec<usize>,
    counts: Vec<usize>,
    cursor: Vec<usize>,
    hist: Vec<std::sync::atomic::AtomicUsize>,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        recycle(Recycled {
            frac: std::mem::take(&mut self.frac),
            folded: std::mem::take(&mut self.folded),
            slot_frac: std::mem::take(&mut self.slot_frac),
            slot_folded: std::mem::take(&mut self.slot_folded),
            key: std::mem::take(&mut self.key),
            offsets: std::mem::take(&mut self.offsets),
            occupants: std::mem::take(&mut self.occupants),
            counts: std::mem::take(&mut self.counts),
            cursor: std::mem::take(&mut self.cursor),
            hist: std::mem::take(&mut self.hist),
        });
    }
}

impl Mesh {
    pub(crate) fn build(
        xyz: &[[f64; 3]],
        simbox: &Cell,
        active: Option<&[usize]>,
        edge: f64,
    ) -> Result<Self, Error> {
        let ids = match active {
            Some(list) => Active::List(list),
            None => Active::All(xyz.len()),
        };
        Self::assemble(xyz, simbox, edge, ids)
    }

    fn assemble(
        xyz: &[[f64; 3]],
        simbox: &Cell,
        edge: f64,
        ids: Active<'_>,
    ) -> Result<Self, Error> {
        let widths = simbox.widths();
        let nx = bins_1d(widths[0], edge)?;
        let ny = bins_1d(widths[1], edge)?;
        let nz = bins_1d(widths[2], edge)?;
        let ncell = (i64::from(nx))
            .checked_mul(i64::from(ny))
            .and_then(|v| v.checked_mul(i64::from(nz)))
            .filter(|&v| v > 0 && v <= MAX_CELLS)
            .ok_or(Error::TooManyCells)? as usize;

        let n = xyz.len();
        let n_active = ids.len();
        let mut buf = take_recycled();
        if buf.frac.len() != n {
            buf.frac.resize(n, [0.0; 3]);
            buf.folded.resize(n, [0.0; 3]);
            buf.key.resize(n, 0);
        }

        #[cfg(feature = "parallel")]
        let parallel = n_active >= PARALLEL_MESH;
        #[cfg(not(feature = "parallel"))]
        let parallel = false;
        if buf.offsets.len() != ncell + 1 {
            buf.offsets.clear();
            buf.offsets.resize(ncell + 1, 0);
        }
        buf.occupants.resize(n_active, 0);

        if parallel {
            #[cfg(feature = "parallel")]
            {
                // Counts, offsets, and cursors in parallel over the bins:
                // there can be more bins than points.
                if buf.hist.len() != ncell {
                    buf.hist.clear();
                    buf.hist
                        .resize_with(ncell, || std::sync::atomic::AtomicUsize::new(0));
                }
                zero_parallel(&buf.hist);
                bin_parallel(
                    xyz,
                    simbox,
                    ids,
                    [nx, ny, nz],
                    MeshScratch {
                        frac: &mut buf.frac,
                        folded: &mut buf.folded,
                        key: &mut buf.key,
                        hist: &buf.hist,
                    },
                );
                prefix_parallel(&buf.hist, &mut buf.offsets);
                scatter_parallel(ids, &buf.key, &buf.offsets, &buf.hist, &mut buf.occupants);
            }
        } else {
            buf.counts.clear();
            buf.counts.resize(ncell, 0);
            let _pop = crate::pop::JobTimer::new();
            let mut start = 0usize;
            if let Active::All(len) = ids {
                let path = crate::simd::select();
                let counts = &mut buf.counts;
                start = fold_mesh_at(
                    path,
                    simbox,
                    xyz,
                    [nx, ny, nz],
                    0,
                    len,
                    (&mut buf.frac, &mut buf.folded, &mut buf.key),
                    |c| counts[c] += 1,
                );
            }
            for slot in start..n_active {
                let i = ids.get(slot);
                let s = simbox.fractional(xyz[i]);
                let b = [
                    bin_coord(s[0], nx),
                    bin_coord(s[1], ny),
                    bin_coord(s[2], nz),
                ];
                let c = flat_cell(b, [nx, ny, nz]);
                buf.frac[i] = s;
                buf.folded[i] = simbox.cartesian(s);
                buf.key[i] = c as u32;
                buf.counts[c] += 1;
            }
            let _pop = crate::pop::JobTimer::new();
            buf.offsets[0] = 0;
            for c in 0..ncell {
                buf.offsets[c + 1] = buf.offsets[c] + buf.counts[c];
            }
            buf.cursor.clear();
            buf.cursor.extend_from_slice(&buf.offsets[..ncell]);
            for slot in 0..n_active {
                let i = ids.get(slot);
                let c = buf.key[i] as usize;
                let dest = buf.cursor[c];
                buf.occupants[dest] = i;
                buf.cursor[c] = dest + 1;
            }
        }

        buf.slot_frac.clear();
        buf.slot_folded.clear();
        {
            let _pop = crate::pop::JobTimer::new();
            let (frac, folded) = (&buf.frac, &buf.folded);
            #[cfg(feature = "parallel")]
            if parallel {
                use rayon::prelude::*;
                buf.slot_frac
                    .par_extend(buf.occupants.par_iter().map(|&i| frac[i]));
                buf.slot_folded
                    .par_extend(buf.occupants.par_iter().map(|&i| folded[i]));
            }
            if buf.slot_frac.len() != n_active {
                buf.slot_frac.clear();
                buf.slot_frac.extend(buf.occupants.iter().map(|&i| frac[i]));
                buf.slot_folded.clear();
                buf.slot_folded
                    .extend(buf.occupants.iter().map(|&i| folded[i]));
            }
        }

        Ok(Self {
            nx,
            ny,
            nz,
            widths,
            frac: buf.frac,
            folded: buf.folded,
            slot_frac: buf.slot_frac,
            slot_folded: buf.slot_folded,
            key: buf.key,
            offsets: buf.offsets,
            occupants: buf.occupants,
            counts: buf.counts,
            cursor: buf.cursor,
            hist: buf.hist,
        })
    }

    /// Bin index plus the integer image `(na, nb, nc)` of an unfolded cell.
    #[inline(always)]
    pub(crate) fn locate(&self, ix: i32, iy: i32, iz: i32) -> (usize, i32, i32, i32) {
        let (cx, na) = split_axis(ix, self.nx);
        let (cy, nb) = split_axis(iy, self.ny);
        let (cz, nc) = split_axis(iz, self.nz);
        let cell = ((cz * self.ny + cy) * self.nx + cx) as usize;
        (cell, na, nb, nc)
    }

    /// Half-box of bins. The search cap is at least this, and at least
    /// the space diagonal in cell heights.
    pub(crate) fn image_reach(&self) -> i32 {
        self.nx.max(self.ny).max(self.nz) / 2 + 2
    }
}

pub(crate) fn target_edge(simbox: &Cell, hint: Option<f64>, fallback: f64) -> f64 {
    let mut edge = hint.unwrap_or(fallback);
    if !edge.is_finite() || edge <= 0.0 {
        edge = fallback;
    }
    let w = simbox.widths();
    edge.min(w[0]).min(w[1]).min(w[2])
}

/// Index box grown from `prev` to `reach`. `prev` of `-1` visits the
/// whole box, including the home cell. Later calls visit only the new
/// faces, so a short axis can grow while a long axis stays put.
pub(crate) fn for_new_layer(prev: [i32; 3], reach: [i32; 3], mut visit: impl FnMut(i32, i32, i32)) {
    let [px, py, pz] = prev;
    let [rx, ry, rz] = reach;
    for dz in -rz..=rz {
        for dy in -ry..=ry {
            for dx in -rx..=rx {
                if dx.abs() <= px && dy.abs() <= py && dz.abs() <= pz {
                    continue;
                }
                visit(dx, dy, dz);
            }
        }
    }
}

/// Perpendicular gap from fractional coordinate `s` to the nearest
/// plane just outside an index interval of radius `reach`.
///
/// Bins are half-open, so an unvisited point lies past this plane. A
/// neighbour sitting on the plane is still inside the visited interval.
pub(crate) fn axis_gap(s: f64, bin: i32, reach: i32, n: i32, w: f64) -> f64 {
    let nf = f64::from(n);
    let plus = (f64::from(bin + reach + 1) / nf - s) * w;
    let minus = (s - f64::from(bin - reach) / nf) * w;
    plus.min(minus)
}

/// Squared lower bound on the distance from `s` to an image bin.
pub(crate) fn slab_dist2(s: [f64; 3], unfolded: [i32; 3], n: [i32; 3], w: [f64; 3]) -> f64 {
    let mut gap = 0.0_f64;
    for a in 0..3 {
        let nf = f64::from(n[a]);
        let lo = f64::from(unfolded[a]) / nf;
        let hi = f64::from(unfolded[a] + 1) / nf;
        let axis = if s[a] < lo {
            (lo - s[a]) * w[a]
        } else if s[a] > hi {
            (s[a] - hi) * w[a]
        } else {
            0.0
        };
        gap = gap.max(axis);
    }
    let d = certify(gap);
    d * d
}

/// Active ids. `All` is `0..n` without a side buffer. `List` is a mask
/// filter. Each index appears once; the parallel fill writes `frac[i]`
/// from one slot only.
#[derive(Clone, Copy)]
enum Active<'a> {
    All(usize),
    List(&'a [usize]),
}

impl Active<'_> {
    fn len(self) -> usize {
        match self {
            Active::All(n) => n,
            Active::List(list) => list.len(),
        }
    }

    fn get(self, slot: usize) -> usize {
        match self {
            Active::All(_) => slot,
            Active::List(list) => list[slot],
        }
    }
}

/// Raw pointer shared across rayon jobs. Each job writes a distinct index.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);

#[cfg(feature = "parallel")]
unsafe impl<T> Send for SyncPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T> Sync for SyncPtr<T> {}

#[cfg(feature = "parallel")]
impl<T> SyncPtr<T> {
    /// # Safety
    /// `index` selects a slot no other job writes for the duration of this call.
    unsafe fn write(self, index: usize, value: T) {
        self.0.add(index).write(value);
    }

    /// The pointer; a closure that calls this captures the whole `SyncPtr`.
    fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(feature = "parallel")]
struct MeshScratch<'a> {
    frac: &'a mut [[f64; 3]],
    folded: &'a mut [[f64; 3]],
    key: &'a mut [u32],
    hist: &'a [std::sync::atomic::AtomicUsize],
}

/// Every counter to zero, in parallel.
#[cfg(feature = "parallel")]
fn zero_parallel(hist: &[std::sync::atomic::AtomicUsize]) {
    use rayon::prelude::*;
    hist.par_chunks(1 << 14).for_each(|c| {
        for h in c {
            h.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// `offsets[c]` is the sum of the counts before bin `c`, in parallel over
/// runs of bins.
#[cfg(feature = "parallel")]
fn prefix_parallel(hist: &[std::sync::atomic::AtomicUsize], offsets: &mut [usize]) {
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;
    let ncell = hist.len();
    let run = (ncell / (4 * rayon::current_num_threads()).max(1)).max(1 << 12);
    let sums: Vec<usize> = hist
        .par_chunks(run)
        .map(|c| c.iter().map(|h| h.load(Ordering::Relaxed)).sum())
        .collect();
    let mut base = Vec::with_capacity(sums.len());
    let mut acc = 0usize;
    for s in &sums {
        base.push(acc);
        acc += s;
    }
    offsets[ncell] = acc;
    offsets[..ncell]
        .par_chunks_mut(run)
        .zip(hist.par_chunks(run))
        .zip(base.par_iter())
        .for_each(|((out, h), &b)| {
            let mut a = b;
            for (o, h) in out.iter_mut().zip(h) {
                *o = a;
                a += h.load(Ordering::Relaxed);
            }
        });
}

#[cfg(feature = "parallel")]
fn bin_parallel(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    ids: Active<'_>,
    nbin: [i32; 3],
    scratch: MeshScratch<'_>,
) {
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;

    let [nx, ny, nz] = nbin;
    let hist = scratch.hist;
    let n_active = ids.len();

    match ids {
        Active::All(_) => {
            const CHUNK: usize = 4096;
            let path = crate::simd::select();
            scratch
                .frac
                .par_chunks_mut(CHUNK)
                .zip(scratch.folded.par_chunks_mut(CHUNK))
                .zip(scratch.key.par_chunks_mut(CHUNK))
                .zip(xyz.par_chunks(CHUNK))
                .for_each_init(crate::pop::JobTimer::new, |_timer, (((f, fold), b), p)| {
                    let len = p.len();
                    let start = fold_mesh_at(
                        path,
                        simbox,
                        p,
                        nbin,
                        0,
                        len,
                        (&mut *f, &mut *fold, &mut *b),
                        |c| {
                            hist[c].fetch_add(1, Ordering::Relaxed);
                        },
                    );
                    for t in start..len {
                        let s = simbox.fractional(p[t]);
                        let c = flat_cell(
                            [
                                bin_coord(s[0], nx),
                                bin_coord(s[1], ny),
                                bin_coord(s[2], nz),
                            ],
                            nbin,
                        );
                        f[t] = s;
                        fold[t] = simbox.cartesian(s);
                        b[t] = c as u32;
                        hist[c].fetch_add(1, Ordering::Relaxed);
                    }
                });
        }
        Active::List(_) => {
            // SAFETY: `ids` does not repeat an index, so each `frac[i]`
            // is written by one iteration. The join of `for_each` happens
            // before the histogram is read.
            let frac_ptr = SyncPtr(scratch.frac.as_mut_ptr());
            let folded_ptr = SyncPtr(scratch.folded.as_mut_ptr());
            let key_ptr = SyncPtr(scratch.key.as_mut_ptr());
            (0..n_active).into_par_iter().for_each_init(
                crate::pop::JobTimer::new,
                |_timer, slot| {
                    let i = ids.get(slot);
                    let s = simbox.fractional(xyz[i]);
                    let c = flat_cell(
                        [
                            bin_coord(s[0], nx),
                            bin_coord(s[1], ny),
                            bin_coord(s[2], nz),
                        ],
                        nbin,
                    );
                    // SAFETY: `ids` lists each point once, so these writes do not alias.
                    unsafe {
                        frac_ptr.write(i, s);
                        folded_ptr.write(i, simbox.cartesian(s));
                        key_ptr.write(i, c as u32);
                    }
                    hist[c].fetch_add(1, Ordering::Relaxed);
                },
            );
        }
    }
}

/// Scatter the points into their bins with `cursor` (the counts) as
/// atomic cursors, then sort each bin of several points, so the order is
/// the serial scatter's, which walks the points in order.
#[cfg(feature = "parallel")]
fn scatter_parallel(
    ids: Active<'_>,
    key: &[u32],
    offsets: &[usize],
    cursor: &[std::sync::atomic::AtomicUsize],
    occupants: &mut [usize],
) {
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;

    let ncell = offsets.len() - 1;
    cursor
        .par_chunks(1 << 14)
        .zip(offsets[..ncell].par_chunks(1 << 14))
        .for_each(|(c, o)| {
            for (c, &o) in c.iter().zip(o) {
                c.store(o, Ordering::Relaxed);
            }
        });
    let occ = SyncPtr(occupants.as_mut_ptr());
    let n_active = ids.len();
    (0..n_active)
        .into_par_iter()
        .for_each_init(crate::pop::JobTimer::new, |_timer, slot| {
            let i = ids.get(slot);
            let dest = cursor[key[i] as usize].fetch_add(1, Ordering::Relaxed);
            // SAFETY: each `fetch_add` returns a distinct slot, and the
            // cell ranges partition `occupants`.
            unsafe {
                occ.write(dest, i);
            }
        });
    // Bins are disjoint runs of slots, so runs of bins sort apart.
    let run = (ncell / (4 * rayon::current_num_threads()).max(1)).max(1 << 12);
    let starts: Vec<usize> = (0..ncell).step_by(run).collect();
    starts.par_iter().for_each(|&c0| {
        for c in c0..(c0 + run).min(ncell) {
            let (lo, hi) = (offsets[c], offsets[c + 1]);
            if hi - lo > 1 {
                // SAFETY: bin `c` is slots `lo..hi`, and no other run of
                // bins holds it.
                let bin = unsafe { std::slice::from_raw_parts_mut(occ.get().add(lo), hi - lo) };
                bin.sort_unstable();
            }
        }
    });
}

fn bins_1d(width: f64, edge: f64) -> Result<i32, Error> {
    let n = (width / edge).floor().max(1.0);
    if !n.is_finite() || n > 1_000_000.0 {
        return Err(Error::TooManyCells);
    }
    Ok(n as i32)
}

/// Box constants of the wide fold. AVX-512 uses eight lanes, AVX2 four,
/// NEON two. The values are the scalar cell's.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) struct Fold8 {
    origin: [f64; 3],
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    hinv: [[f64; 3]; 3],
    ortho: bool,
    n: [i32; 3],
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Fold8 {
    pub(crate) fn new(simbox: &Cell, n: [i32; 3]) -> Self {
        Fold8 {
            origin: simbox.origin(),
            widths: simbox.widths(),
            h: simbox.h(),
            hinv: simbox.hinv(),
            ortho: simbox.is_ortho(),
            n,
        }
    }
}

/// Fractional coordinates, folded positions, and bins of the eight
/// packed `x y z` rows at `base`, with [`Cell::fractional`],
/// [`Cell::cartesian`], and the bin truncation of [`Mesh::build`] done
/// lane by lane in the same order (a division, or the `Hinv` product,
/// then `wrap01`; `H` then the origin), so every value is the same bits.
///
/// # Safety
/// AVX-512F is available and `base` is readable for 24 doubles.
#[allow(clippy::incompatible_msrv, clippy::type_complexity)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn fold8(
    f: &Fold8,
    base: *const f64,
) -> ([[f64; 8]; 3], [[f64; 8]; 3], [[i32; 8]; 3]) {
    use std::arch::x86_64::{_mm256_storeu_si256, _mm512_storeu_pd};
    let (s, p, b) = fold8_regs(f, base);
    let mut so = [[0.0f64; 8]; 3];
    let mut po = [[0.0f64; 8]; 3];
    let mut bo = [[0i32; 8]; 3];
    for ax in 0..3 {
        _mm512_storeu_pd(so[ax].as_mut_ptr(), s[ax]);
        _mm512_storeu_pd(po[ax].as_mut_ptr(), p[ax]);
        _mm256_storeu_si256(bo[ax].as_mut_ptr() as *mut _, b[ax]);
    }
    (so, po, bo)
}

/// [`fold8`] in registers.
///
/// # Safety
/// AVX-512F is available and `base` is readable for 24 doubles.
#[allow(clippy::incompatible_msrv, clippy::type_complexity)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn fold8_regs(
    f: &Fold8,
    base: *const f64,
) -> (
    [std::arch::x86_64::__m512d; 3],
    [std::arch::x86_64::__m512d; 3],
    [std::arch::x86_64::__m256i; 3],
) {
    use std::arch::x86_64::{
        __m512d, _mm256_max_epi32, _mm256_min_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm512_add_pd, _mm512_cmp_pd_mask, _mm512_cvttpd_epi32, _mm512_div_pd, _mm512_loadu_pd,
        _mm512_mask_blend_pd, _mm512_mask_permutex2var_pd, _mm512_mul_pd, _mm512_permutex2var_pd,
        _mm512_roundscale_pd, _mm512_set1_pd, _mm512_setr_epi64, _mm512_setzero_pd, _mm512_sub_pd,
        _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ,
    };
    let bc = |v: f64| _mm512_set1_pd(v);
    let one = bc(1.0);
    let (o, w, h, hinv, n) = (f.origin, f.widths, f.h, f.hinv, f.n);
    // `wrap01`: minimage `floor_fast` (trunc toward zero, `-0` becomes
    // `+0`, then one less when that trunc is above the input), then a
    // value that rounds to one is zero.
    let wrap01 = |s: __m512d| {
        let zero = _mm512_setzero_pd();
        let trunc = _mm512_roundscale_pd::<0x0b>(s);
        let trunc =
            _mm512_mask_blend_pd(_mm512_cmp_pd_mask::<_CMP_EQ_OQ>(trunc, zero), trunc, zero);
        let floor = _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_GT_OQ>(trunc, s),
            trunc,
            _mm512_sub_pd(trunc, one),
        );
        let t = _mm512_sub_pd(s, floor);
        _mm512_mask_blend_pd(_mm512_cmp_pd_mask::<_CMP_GE_OQ>(t, one), t, zero)
    };
    let a = _mm512_loadu_pd(base);
    let b = _mm512_loadu_pd(base.add(8));
    let c = _mm512_loadu_pd(base.add(16));
    let x = _mm512_mask_permutex2var_pd(
        _mm512_permutex2var_pd(a, _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0), b),
        0xc0,
        _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 10, 13),
        c,
    );
    let y = _mm512_mask_permutex2var_pd(
        _mm512_permutex2var_pd(a, _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0), b),
        0xe0,
        _mm512_setr_epi64(0, 0, 0, 0, 0, 8, 11, 14),
        c,
    );
    let z = _mm512_mask_permutex2var_pd(
        _mm512_permutex2var_pd(a, _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0), b),
        0xe0,
        _mm512_setr_epi64(0, 0, 0, 0, 0, 9, 12, 15),
        c,
    );
    let d = [
        _mm512_sub_pd(x, bc(o[0])),
        _mm512_sub_pd(y, bc(o[1])),
        _mm512_sub_pd(z, bc(o[2])),
    ];
    let s = if f.ortho {
        [
            wrap01(_mm512_div_pd(d[0], bc(w[0]))),
            wrap01(_mm512_div_pd(d[1], bc(w[1]))),
            wrap01(_mm512_div_pd(d[2], bc(w[2]))),
        ]
    } else {
        let row = |r: usize| {
            _mm512_add_pd(
                _mm512_add_pd(
                    _mm512_mul_pd(bc(hinv[0][r]), d[0]),
                    _mm512_mul_pd(bc(hinv[1][r]), d[1]),
                ),
                _mm512_mul_pd(bc(hinv[2][r]), d[2]),
            )
        };
        [wrap01(row(0)), wrap01(row(1)), wrap01(row(2))]
    };
    let cart = |r: usize| {
        _mm512_add_pd(
            _mm512_add_pd(
                _mm512_add_pd(
                    _mm512_mul_pd(bc(h[0][r]), s[0]),
                    _mm512_mul_pd(bc(h[1][r]), s[1]),
                ),
                _mm512_mul_pd(bc(h[2][r]), s[2]),
            ),
            bc(o[r]),
        )
    };
    let bin = |ax: usize| {
        let t = _mm512_cvttpd_epi32(_mm512_mul_pd(s[ax], bc(f64::from(n[ax]))));
        _mm256_min_epi32(
            _mm256_max_epi32(t, _mm256_setzero_si256()),
            _mm256_set1_epi32(n[ax] - 1),
        )
    };
    (s, [cart(0), cart(1), cart(2)], [bin(0), bin(1), bin(2)])
}

/// The three per-point arrays of a mesh: fractional, folded, flat bin.
type MeshOut<'a> = (&'a mut [[f64; 3]], &'a mut [[f64; 3]], &'a mut [u32]);

/// Fold `xyz[lo..hi]` with `path`, and add each bin to `count`.
/// Returns the first index left for the scalar fold.
#[allow(clippy::too_many_arguments)]
fn fold_mesh_at(
    path: crate::simd::Path,
    simbox: &Cell,
    xyz: &[[f64; 3]],
    n: [i32; 3],
    lo: usize,
    hi: usize,
    out: MeshOut<'_>,
    count: impl FnMut(usize),
) -> usize {
    match path {
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        crate::simd::Path::Avx512 => fold_mesh_avx512(simbox, xyz, n, lo, hi, out, count),
        #[cfg(target_arch = "x86_64")]
        crate::simd::Path::Avx2 => fold_mesh_avx2(simbox, xyz, n, lo, hi, out, count),
        #[cfg(target_arch = "aarch64")]
        crate::simd::Path::Neon => fold_mesh_neon(simbox, xyz, n, lo, hi, out, count),
        _ => lo,
    }
}

/// Fold `xyz[lo..hi]` into `frac`, `folded`, and `key` at the same
/// indices, eight at a time on AVX-512, and add each to `counts`;
/// returns the first index left for the scalar fold.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
fn fold_mesh_avx512(
    simbox: &Cell,
    xyz: &[[f64; 3]],
    n: [i32; 3],
    lo: usize,
    hi: usize,
    out: MeshOut<'_>,
    mut count: impl FnMut(usize),
) -> usize {
    if !std::is_x86_feature_detected!("avx512f") {
        return lo;
    }
    let (frac, folded, key) = out;
    let f = Fold8::new(simbox, n);
    let flat = xyz.as_ptr() as *const f64;
    let mut k = lo;
    while k + 8 <= hi {
        // Safety: AVX-512F was detected and rows `k..k + 8` are in `xyz`.
        let (s, p, b) = unsafe { fold8(&f, flat.add(3 * k)) };
        for l in 0..8 {
            let i = k + l;
            frac[i] = [s[0][l], s[1][l], s[2][l]];
            folded[i] = [p[0][l], p[1][l], p[2][l]];
            let c = flat_cell([b[0][l], b[1][l], b[2][l]], n);
            key[i] = c as u32;
            count(c);
        }
        k += 8;
    }
    k
}

/// Fractional coordinates, folded positions, and bins of the four
/// packed `x y z` rows at `base`. Lane order matches [`fold8`]: a
/// division, or the `Hinv` product, then `wrap01`; `H` then the origin.
/// Sums are separate multiplies and adds.
///
/// The shuffle control is lane sources, low lane in the low bits:
/// `x` is `(a0, a3, b2, c1)`, `y` is `(a1, b0, b3, c2)`, `z` is
/// `(a2, b1, c0, c3)`, with `a` the first four doubles.
///
/// # Safety
/// AVX2 and FMA are available and `base` is readable for 12 doubles.
#[allow(clippy::incompatible_msrv, clippy::type_complexity)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[inline]
unsafe fn fold4(f: &Fold8, base: *const f64) -> ([[f64; 4]; 3], [[f64; 4]; 3], [[i32; 4]; 3]) {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_blend_pd, _mm256_blendv_pd, _mm256_cmp_pd,
        _mm256_cvttpd_epi32, _mm256_div_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute4x64_pd,
        _mm256_round_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
        _mm_max_epi32, _mm_min_epi32, _mm_set1_epi32, _mm_setzero_si128, _mm_storeu_si128,
        _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ,
    };
    let bc = |v: f64| _mm256_set1_pd(v);
    let one = bc(1.0);
    let (o, w, h, hinv, n) = (f.origin, f.widths, f.h, f.hinv, f.n);
    // `floor_fast`: trunc toward zero, `-0` becomes `+0`, then one less
    // when that trunc is above the input. A result that rounds to one is zero.
    let wrap01 = |s: __m256d| {
        let zero = _mm256_setzero_pd();
        let trunc = _mm256_round_pd::<0x0b>(s);
        let trunc = _mm256_blendv_pd(trunc, zero, _mm256_cmp_pd::<_CMP_EQ_OQ>(trunc, zero));
        let floor = _mm256_blendv_pd(
            trunc,
            _mm256_sub_pd(trunc, one),
            _mm256_cmp_pd::<_CMP_GT_OQ>(trunc, s),
        );
        let t = _mm256_sub_pd(s, floor);
        _mm256_blendv_pd(t, zero, _mm256_cmp_pd::<_CMP_GE_OQ>(t, one))
    };
    let a = _mm256_loadu_pd(base);
    let b = _mm256_loadu_pd(base.add(4));
    let c = _mm256_loadu_pd(base.add(8));
    // a = x0 y0 z0 x1, b = y1 z1 x2 y2, c = z2 x3 y3 z3.
    let x = _mm256_blend_pd::<0b1000>(
        _mm256_blend_pd::<0b0100>(
            _mm256_permute4x64_pd::<0x0c>(a),
            _mm256_permute4x64_pd::<0x20>(b),
        ),
        _mm256_permute4x64_pd::<0x40>(c),
    );
    let y = _mm256_blend_pd::<0b1000>(
        _mm256_blend_pd::<0b0110>(
            _mm256_permute4x64_pd::<0x01>(a),
            _mm256_permute4x64_pd::<0x30>(b),
        ),
        _mm256_permute4x64_pd::<0x80>(c),
    );
    let z = _mm256_blend_pd::<0b1100>(
        _mm256_blend_pd::<0b0010>(
            _mm256_permute4x64_pd::<0x02>(a),
            _mm256_permute4x64_pd::<0x04>(b),
        ),
        _mm256_permute4x64_pd::<0xc5>(c),
    );
    let d = [
        _mm256_sub_pd(x, bc(o[0])),
        _mm256_sub_pd(y, bc(o[1])),
        _mm256_sub_pd(z, bc(o[2])),
    ];
    let s = if f.ortho {
        [
            wrap01(_mm256_div_pd(d[0], bc(w[0]))),
            wrap01(_mm256_div_pd(d[1], bc(w[1]))),
            wrap01(_mm256_div_pd(d[2], bc(w[2]))),
        ]
    } else {
        let row = |r: usize| {
            _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(bc(hinv[0][r]), d[0]),
                    _mm256_mul_pd(bc(hinv[1][r]), d[1]),
                ),
                _mm256_mul_pd(bc(hinv[2][r]), d[2]),
            )
        };
        [wrap01(row(0)), wrap01(row(1)), wrap01(row(2))]
    };
    let cart = |r: usize| {
        _mm256_add_pd(
            _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(bc(h[0][r]), s[0]),
                    _mm256_mul_pd(bc(h[1][r]), s[1]),
                ),
                _mm256_mul_pd(bc(h[2][r]), s[2]),
            ),
            bc(o[r]),
        )
    };
    let bin = |ax: usize| {
        let t = _mm256_cvttpd_epi32(_mm256_mul_pd(s[ax], bc(f64::from(n[ax]))));
        _mm_min_epi32(
            _mm_max_epi32(t, _mm_setzero_si128()),
            _mm_set1_epi32(n[ax] - 1),
        )
    };
    let mut so = [[0.0f64; 4]; 3];
    let mut po = [[0.0f64; 4]; 3];
    let mut bo = [[0i32; 4]; 3];
    let p = [cart(0), cart(1), cart(2)];
    let bins = [bin(0), bin(1), bin(2)];
    for ax in 0..3 {
        _mm256_storeu_pd(so[ax].as_mut_ptr(), s[ax]);
        _mm256_storeu_pd(po[ax].as_mut_ptr(), p[ax]);
        _mm_storeu_si128(bo[ax].as_mut_ptr().cast(), bins[ax]);
    }
    (so, po, bo)
}

/// [`fold_mesh_avx512`] four points at a time.
#[cfg(target_arch = "x86_64")]
fn fold_mesh_avx2(
    simbox: &Cell,
    xyz: &[[f64; 3]],
    n: [i32; 3],
    lo: usize,
    hi: usize,
    out: MeshOut<'_>,
    mut count: impl FnMut(usize),
) -> usize {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return lo;
    }
    let (frac, folded, key) = out;
    let f = Fold8::new(simbox, n);
    let flat = xyz.as_ptr() as *const f64;
    let mut k = lo;
    while k + 4 <= hi {
        // Safety: AVX2 and FMA were detected and rows `k..k + 4` are in `xyz`.
        let (s, p, b) = unsafe { fold4(&f, flat.add(3 * k)) };
        for l in 0..4 {
            let i = k + l;
            frac[i] = [s[0][l], s[1][l], s[2][l]];
            folded[i] = [p[0][l], p[1][l], p[2][l]];
            let c = flat_cell([b[0][l], b[1][l], b[2][l]], n);
            key[i] = c as u32;
            count(c);
        }
        k += 4;
    }
    k
}

/// [`fold4`] for two points. `a` is `x0 y0`, `b` is `z0 x1`, `c` is `y1 z1`.
///
/// # Safety
/// NEON is available and `base` is readable for 6 doubles.
#[allow(clippy::type_complexity)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fold2(f: &Fold8, base: *const f64) -> ([[f64; 2]; 3], [[f64; 2]; 3], [[i32; 2]; 3]) {
    use std::arch::aarch64::{
        float64x2_t, vaddq_f64, vbslq_f64, vceqq_f64, vcgeq_f64, vcgtq_f64, vcombine_f64,
        vcvtq_s64_f64, vdivq_f64, vdupq_n_f64, vget_high_f64, vget_low_f64, vgetq_lane_s64,
        vld1q_f64, vmulq_f64, vrndq_f64, vst1q_f64, vsubq_f64,
    };
    let a = vld1q_f64(base);
    let b = vld1q_f64(base.add(2));
    let c = vld1q_f64(base.add(4));
    let x = vcombine_f64(vget_low_f64(a), vget_high_f64(b));
    let y = vcombine_f64(vget_high_f64(a), vget_low_f64(c));
    let z = vcombine_f64(vget_low_f64(b), vget_high_f64(c));
    let bc = |v: f64| vdupq_n_f64(v);
    let one = bc(1.0);
    let zero = vdupq_n_f64(0.0);
    let (o, w, h, hinv, nbin) = (f.origin, f.widths, f.h, f.hinv, f.n);
    // `floor_fast`: trunc toward zero, `-0` becomes `+0`, then one less
    // when that trunc is above the input. A result that rounds to one is zero.
    let wrap01 = |s: float64x2_t| {
        let trunc = vrndq_f64(s);
        let trunc = vbslq_f64(vceqq_f64(trunc, zero), zero, trunc);
        let floor = vbslq_f64(vcgtq_f64(trunc, s), vsubq_f64(trunc, one), trunc);
        let t = vsubq_f64(s, floor);
        vbslq_f64(vcgeq_f64(t, one), zero, t)
    };
    let d = [
        vsubq_f64(x, bc(o[0])),
        vsubq_f64(y, bc(o[1])),
        vsubq_f64(z, bc(o[2])),
    ];
    let s = if f.ortho {
        [
            wrap01(vdivq_f64(d[0], bc(w[0]))),
            wrap01(vdivq_f64(d[1], bc(w[1]))),
            wrap01(vdivq_f64(d[2], bc(w[2]))),
        ]
    } else {
        let row = |r: usize| {
            vaddq_f64(
                vaddq_f64(
                    vmulq_f64(bc(hinv[0][r]), d[0]),
                    vmulq_f64(bc(hinv[1][r]), d[1]),
                ),
                vmulq_f64(bc(hinv[2][r]), d[2]),
            )
        };
        [wrap01(row(0)), wrap01(row(1)), wrap01(row(2))]
    };
    let cart = |r: usize| {
        vaddq_f64(
            vaddq_f64(
                vaddq_f64(vmulq_f64(bc(h[0][r]), s[0]), vmulq_f64(bc(h[1][r]), s[1])),
                vmulq_f64(bc(h[2][r]), s[2]),
            ),
            bc(o[r]),
        )
    };
    let mut so = [[0.0f64; 2]; 3];
    let mut po = [[0.0f64; 2]; 3];
    let mut bo = [[0i32; 2]; 3];
    let p = [cart(0), cart(1), cart(2)];
    for ax in 0..3 {
        vst1q_f64(so[ax].as_mut_ptr(), s[ax]);
        vst1q_f64(po[ax].as_mut_ptr(), p[ax]);
        let raw = vcvtq_s64_f64(vmulq_f64(s[ax], bc(f64::from(nbin[ax]))));
        bo[ax][0] = bin_i64(vgetq_lane_s64::<0>(raw), nbin[ax]);
        bo[ax][1] = bin_i64(vgetq_lane_s64::<1>(raw), nbin[ax]);
    }
    (so, po, bo)
}

/// `(s * n) as i32` for a truncated lane, then the [`bin_coord`] clamp.
#[cfg(target_arch = "aarch64")]
fn bin_i64(v: i64, n: i32) -> i32 {
    let mut c = if v > i64::from(i32::MAX) {
        i32::MAX
    } else if v < i64::from(i32::MIN) {
        i32::MIN
    } else {
        v as i32
    };
    if c < 0 {
        c = 0;
    }
    if c >= n {
        c = n - 1;
    }
    c
}

/// [`fold_mesh_avx512`] two points at a time.
#[cfg(target_arch = "aarch64")]
fn fold_mesh_neon(
    simbox: &Cell,
    xyz: &[[f64; 3]],
    n: [i32; 3],
    lo: usize,
    hi: usize,
    out: MeshOut<'_>,
    mut count: impl FnMut(usize),
) -> usize {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return lo;
    }
    let (frac, folded, key) = out;
    let f = Fold8::new(simbox, n);
    let flat = xyz.as_ptr() as *const f64;
    let mut k = lo;
    while k + 2 <= hi {
        // Safety: NEON was detected and rows `k..k + 2` are in `xyz`.
        let (s, p, b) = unsafe { fold2(&f, flat.add(3 * k)) };
        for l in 0..2 {
            let i = k + l;
            frac[i] = [s[0][l], s[1][l], s[2][l]];
            folded[i] = [p[0][l], p[1][l], p[2][l]];
            let c = flat_cell([b[0][l], b[1][l], b[2][l]], n);
            key[i] = c as u32;
            count(c);
        }
        k += 2;
    }
    k
}

/// Bins per axis for `edge`, with the same cell cap as [`Mesh::build`].
pub(crate) fn grid_dims(simbox: &Cell, edge: f64) -> Result<[i32; 3], Error> {
    let w = simbox.widths();
    let n = [
        bins_1d(w[0], edge)?,
        bins_1d(w[1], edge)?,
        bins_1d(w[2], edge)?,
    ];
    i64::from(n[0])
        .checked_mul(i64::from(n[1]))
        .and_then(|v| v.checked_mul(i64::from(n[2])))
        .filter(|&v| v > 0 && v <= MAX_CELLS)
        .ok_or(Error::TooManyCells)?;
    Ok(n)
}

/// Folded Cartesian position and bin of `r`, the values [`Mesh::build`]
/// stores, so a cutoff row sees the same coordinates.
#[inline]
pub(crate) fn fold_point(simbox: &Cell, r: [f64; 3], n: [i32; 3]) -> ([f64; 3], [i32; 3]) {
    let s = simbox.fractional(r);
    let b = [
        bin_coord(s[0], n[0]),
        bin_coord(s[1], n[1]),
        bin_coord(s[2], n[2]),
    ];
    (simbox.cartesian(s), b)
}

/// Flat index of an in-range bin.
#[inline]
pub(crate) fn flat_cell(b: [i32; 3], n: [i32; 3]) -> usize {
    ((b[2] * n[1] + b[1]) * n[0] + b[0]) as usize
}

/// Flat index of any bin, folded into the box.
pub(crate) fn wrap_cell(b: [i32; 3], n: [i32; 3]) -> usize {
    cell_index(b[0], b[1], b[2], n[0], n[1], n[2])
}

fn bin_coord(s: f64, n: i32) -> i32 {
    let mut c = (s * f64::from(n)) as i32;
    if c < 0 {
        c = 0;
    }
    if c >= n {
        c = n - 1;
    }
    c
}

#[inline(always)]
fn split_axis(i: i32, n: i32) -> (i32, i32) {
    if i >= 0 && i < n {
        (i, 0)
    } else {
        (i.rem_euclid(n), i.div_euclid(n))
    }
}

fn cell_index(ix: i32, iy: i32, iz: i32, nx: i32, ny: i32, nz: i32) -> usize {
    let (cx, _) = split_axis(ix, nx);
    let (cy, _) = split_axis(iy, ny);
    let (cz, _) = split_axis(iz, nz);
    ((cz * ny + cy) * nx + cx) as usize
}

pub(crate) fn certify(dist: f64) -> f64 {
    if dist <= 0.0 || !dist.is_finite() {
        return 0.0;
    }
    let d = dist - dist * CERT_REL - CERT_ABS;
    if d > 0.0 {
        d
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_fill_keeps_every_point_once() {
        let nside = 22usize;
        let a = 3.125;
        let boxl = nside as f64 * a;
        let cell = crate::Cell::ortho(boxl, boxl, boxl).unwrap();
        let mut xyz = Vec::with_capacity(nside * nside * nside);
        for iz in 0..nside {
            for iy in 0..nside {
                for ix in 0..nside {
                    xyz.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
                }
            }
        }
        let mesh = Mesh::build(&xyz, &cell, None, 3.0).unwrap();
        assert_eq!(mesh.offsets[mesh.offsets.len() - 1], xyz.len());
        let mut occ = mesh.occupants.clone();
        occ.sort_unstable();
        occ.dedup();
        assert_eq!(occ.len(), xyz.len());
        let all: Vec<usize> = (0..xyz.len()).collect();
        let listed = Mesh::build(&xyz, &cell, Some(&all), 3.0).unwrap();
        let mut listed_occ = listed.occupants.clone();
        listed_occ.sort_unstable();
        assert_eq!(listed_occ, all);
    }

    #[test]
    fn new_layer_is_the_cube_then_one_face() {
        let mut first = Vec::new();
        for_new_layer([-1, -1, -1], [1, 1, 1], |dx, dy, dz| {
            first.push((dx, dy, dz));
        });
        assert_eq!(first.len(), 27);
        let mut grown = Vec::new();
        for_new_layer([1, 1, 1], [1, 2, 1], |dx, dy, dz| {
            grown.push((dx, dy, dz));
        });
        assert_eq!(grown.len(), 2 * 3 * 3);
        assert!(grown.iter().all(|(_, y, _)| y.abs() == 2));
    }

    fn bits3(v: [f64; 3]) -> [u64; 3] {
        [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
    }

    fn xorshift(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    fn sample_points(cell: &crate::Cell, n: usize, state: &mut u64) -> Vec<[f64; 3]> {
        let mut xyz = Vec::with_capacity(n);
        for i in 0..n {
            let unit = |state: &mut u64| (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64;
            let s = [unit(state), unit(state), unit(state)];
            let mut p = cell.cartesian(s);
            if i % 11 == 0 {
                p[0] -= cell.widths()[0];
            }
            if i % 13 == 0 {
                p[1] += 2.0 * cell.widths()[1];
            }
            if i % 17 == 0 {
                p[2] = -0.0;
            }
            xyz.push(p);
        }
        if n > 0 {
            xyz[0] = [0.0, 0.0, 0.0];
        }
        xyz
    }

    #[test]
    fn every_simd_path_folds_like_scalar() {
        let cells = [
            crate::Cell::ortho(25.0, 18.0, 30.0).unwrap(),
            crate::Cell::ortho_origin(25.0, 18.0, 30.0, [1.5, -2.0, 0.25]).unwrap(),
            crate::Cell::from_vectors(
                [25.0, 0.0, 0.0],
                [4.0, 24.0, 0.0],
                [-3.0, 2.0, 24.5],
                [0.5, -0.5, 1.0],
            )
            .unwrap(),
            crate::Cell::from_vectors(
                [24.0, 3.0, -2.0],
                [2.5, 23.5, 3.0],
                [-1.5, 2.0, 24.0],
                [0.0; 3],
            )
            .unwrap(),
        ];
        let mut state = 0x853c_49e6_748f_ea9bu64;
        for cell in &cells {
            for &n in &[1usize, 3, 7, 8, 17, 64, 257] {
                let xyz = sample_points(cell, n, &mut state);
                compare_mesh(cell, &xyz);
            }
        }
        let big = sample_points(&cells[0], 9_000, &mut state);
        compare_mesh(&cells[0], &big);
    }

    fn compare_mesh(cell: &crate::Cell, xyz: &[[f64; 3]]) {
        let scalar = {
            let _guard = crate::simd::force(crate::simd::Path::Scalar);
            Mesh::build(xyz, cell, None, 3.0).unwrap()
        };
        for path in crate::simd::paths_to_compare() {
            let got = {
                let _guard = crate::simd::force(path);
                Mesh::build(xyz, cell, None, 3.0).unwrap()
            };
            assert_eq!((got.nx, got.ny, got.nz), (scalar.nx, scalar.ny, scalar.nz));
            for i in 0..xyz.len() {
                assert_eq!(
                    bits3(got.frac[i]),
                    bits3(scalar.frac[i]),
                    "{path:?} frac {i}"
                );
                assert_eq!(
                    bits3(got.folded[i]),
                    bits3(scalar.folded[i]),
                    "{path:?} folded {i}"
                );
                assert_eq!(got.key[i], scalar.key[i], "{path:?} key {i}");
            }
            assert_eq!(got.offsets, scalar.offsets, "{path:?} offsets");
            assert_eq!(got.occupants, scalar.occupants, "{path:?} occupants");
        }
    }

    #[test]
    fn frontier_is_at_least_the_min_cell_bound() {
        // Bin 2 of 8 covers fractional [0.25, 0.375). The source sits inside it.
        let s = [0.30, 0.30, 0.30];
        let bin = [2, 2, 2];
        let n = [8, 8, 8];
        let w = [10.0, 10.0, 10.0];
        let h = w[0] / f64::from(n[0]);
        for reach in 1..=3 {
            let gap = axis_gap(s[0], bin[0], reach, n[0], w[0])
                .min(axis_gap(s[1], bin[1], reach, n[1], w[1]))
                .min(axis_gap(s[2], bin[2], reach, n[2], w[2]));
            let bound = gap * gap;
            let loose = (f64::from(reach) * h) * (f64::from(reach) * h);
            assert!(
                bound + 1e-9 >= loose * (1.0 - 2.0 * CERT_REL),
                "reach {reach}: {bound} vs {loose}"
            );
        }
    }
}
