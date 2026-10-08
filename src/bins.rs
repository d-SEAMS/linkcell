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
    bin: Vec<[i32; 3]>,
    offsets: Vec<usize>,
    occupants: Vec<usize>,
    counts: Vec<usize>,
    cursor: Vec<usize>,
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
    pub bin: Vec<[i32; 3]>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) occupants: Vec<usize>,
    counts: Vec<usize>,
    cursor: Vec<usize>,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        recycle(Recycled {
            frac: std::mem::take(&mut self.frac),
            folded: std::mem::take(&mut self.folded),
            slot_frac: std::mem::take(&mut self.slot_frac),
            slot_folded: std::mem::take(&mut self.slot_folded),
            bin: std::mem::take(&mut self.bin),
            offsets: std::mem::take(&mut self.offsets),
            occupants: std::mem::take(&mut self.occupants),
            counts: std::mem::take(&mut self.counts),
            cursor: std::mem::take(&mut self.cursor),
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
            buf.bin.resize(n, [0; 3]);
        }
        buf.counts.clear();
        buf.counts.resize(ncell, 0);

        #[cfg(feature = "parallel")]
        let parallel = n_active >= PARALLEL_MESH;
        #[cfg(not(feature = "parallel"))]
        let parallel = false;

        if parallel {
            #[cfg(feature = "parallel")]
            bin_parallel(
                xyz,
                simbox,
                ids,
                [nx, ny, nz],
                MeshScratch {
                    frac: &mut buf.frac,
                    folded: &mut buf.folded,
                    bin: &mut buf.bin,
                    counts: &mut buf.counts,
                },
            );
        } else {
            let _pop = crate::pop::JobTimer::new();
            #[allow(unused_mut)]
            let mut start = 0usize;
            #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
            if let Active::All(len) = ids {
                let counts = &mut buf.counts;
                start = fold_mesh_avx512(
                    simbox,
                    xyz,
                    [nx, ny, nz],
                    0,
                    len,
                    (&mut buf.frac, &mut buf.folded, &mut buf.bin),
                    |c| counts[c] += 1,
                );
            }
            for slot in start..n_active {
                let i = ids.get(slot);
                let s = simbox.fractional(xyz[i]);
                let ix = bin_coord(s[0], nx);
                let iy = bin_coord(s[1], ny);
                let iz = bin_coord(s[2], nz);
                buf.frac[i] = s;
                buf.folded[i] = simbox.cartesian(s);
                buf.bin[i] = [ix, iy, iz];
                buf.counts[cell_index(ix, iy, iz, nx, ny, nz)] += 1;
            }
        }

        buf.offsets.clear();
        buf.offsets.resize(ncell + 1, 0);
        {
            let _pop = crate::pop::JobTimer::new();
            for c in 0..ncell {
                buf.offsets[c + 1] = buf.offsets[c] + buf.counts[c];
            }
        }
        buf.cursor.clear();
        buf.cursor.resize(ncell + 1, 0);
        buf.cursor.copy_from_slice(&buf.offsets);
        buf.occupants.resize(n_active, 0);

        if parallel {
            #[cfg(feature = "parallel")]
            scatter_parallel(ids, &buf.bin, &buf.offsets, &mut buf.occupants, nx, ny, nz);
        } else {
            let _pop = crate::pop::JobTimer::new();
            for slot in 0..n_active {
                let i = ids.get(slot);
                let [ix, iy, iz] = buf.bin[i];
                let c = cell_index(ix, iy, iz, nx, ny, nz);
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
            bin: buf.bin,
            offsets: buf.offsets,
            occupants: buf.occupants,
            counts: buf.counts,
            cursor: buf.cursor,
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
}

#[cfg(feature = "parallel")]
struct MeshScratch<'a> {
    frac: &'a mut [[f64; 3]],
    folded: &'a mut [[f64; 3]],
    bin: &'a mut [[i32; 3]],
    counts: &'a mut [usize],
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    let [nx, ny, nz] = nbin;
    let ncell = scratch.counts.len();
    let hist: Vec<AtomicUsize> = (0..ncell).map(|_| AtomicUsize::new(0)).collect();
    let n_active = ids.len();

    match ids {
        Active::All(_) => {
            const CHUNK: usize = 4096;
            scratch
                .frac
                .par_chunks_mut(CHUNK)
                .zip(scratch.folded.par_chunks_mut(CHUNK))
                .zip(scratch.bin.par_chunks_mut(CHUNK))
                .zip(xyz.par_chunks(CHUNK))
                .for_each_init(crate::pop::JobTimer::new, |_timer, (((f, fold), b), p)| {
                    let len = p.len();
                    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
                    let start = fold_mesh_avx512(
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
                    #[cfg(not(all(target_arch = "x86_64", linkcell_avx512)))]
                    let start = 0usize;
                    for t in start..len {
                        let s = simbox.fractional(p[t]);
                        let ix = bin_coord(s[0], nx);
                        let iy = bin_coord(s[1], ny);
                        let iz = bin_coord(s[2], nz);
                        f[t] = s;
                        fold[t] = simbox.cartesian(s);
                        b[t] = [ix, iy, iz];
                        hist[cell_index(ix, iy, iz, nx, ny, nz)].fetch_add(1, Ordering::Relaxed);
                    }
                });
        }
        Active::List(_) => {
            // SAFETY: `ids` does not repeat an index, so each `frac[i]`
            // is written by one iteration. The join of `for_each` happens
            // before the histogram is read.
            let frac_ptr = SyncPtr(scratch.frac.as_mut_ptr());
            let folded_ptr = SyncPtr(scratch.folded.as_mut_ptr());
            let bin_ptr = SyncPtr(scratch.bin.as_mut_ptr());
            (0..n_active).into_par_iter().for_each_init(
                crate::pop::JobTimer::new,
                |_timer, slot| {
                    let i = ids.get(slot);
                    let s = simbox.fractional(xyz[i]);
                    let ix = bin_coord(s[0], nx);
                    let iy = bin_coord(s[1], ny);
                    let iz = bin_coord(s[2], nz);
                    // SAFETY: `ids` lists each point once, so these writes do not alias.
                    unsafe {
                        frac_ptr.write(i, s);
                        folded_ptr.write(i, simbox.cartesian(s));
                        bin_ptr.write(i, [ix, iy, iz]);
                    }
                    hist[cell_index(ix, iy, iz, nx, ny, nz)].fetch_add(1, Ordering::Relaxed);
                },
            );
        }
    }

    let _pop = crate::pop::JobTimer::new();
    for (dst, src) in scratch.counts.iter_mut().zip(&hist) {
        *dst = src.load(Ordering::Relaxed);
    }
}

#[cfg(feature = "parallel")]
fn scatter_parallel(
    ids: Active<'_>,
    bin: &[[i32; 3]],
    offsets: &[usize],
    occupants: &mut [usize],
    nx: i32,
    ny: i32,
    nz: i32,
) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let ncell = offsets.len() - 1;
    let cursor: Vec<AtomicUsize> = offsets[..ncell]
        .iter()
        .copied()
        .map(AtomicUsize::new)
        .collect();
    let occ = SyncPtr(occupants.as_mut_ptr());
    let n_active = ids.len();
    (0..n_active)
        .into_par_iter()
        .for_each_init(crate::pop::JobTimer::new, |_timer, slot| {
            let i = ids.get(slot);
            let [ix, iy, iz] = bin[i];
            let c = cell_index(ix, iy, iz, nx, ny, nz);
            let dest = cursor[c].fetch_add(1, Ordering::Relaxed);
            // SAFETY: each `fetch_add` returns a distinct slot, and the
            // cell ranges partition `occupants`.
            unsafe {
                occ.write(dest, i);
            }
        });
    let _pop = crate::pop::JobTimer::new();
    // Index order matches the serial scatter, which walks ids in order.
    // A full bin of one point is already ordered. Cutoff pairs keep visit
    // order, so a crowded bin is sorted.
    for c in 0..ncell {
        let lo = offsets[c];
        let hi = offsets[c + 1];
        if hi - lo > 1 {
            occupants[lo..hi].sort_unstable();
        }
    }
}

fn bins_1d(width: f64, edge: f64) -> Result<i32, Error> {
    let n = (width / edge).floor().max(1.0);
    if !n.is_finite() || n > 1_000_000.0 {
        return Err(Error::TooManyCells);
    }
    Ok(n as i32)
}

/// Box constants of the eight-wide fold.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
pub(crate) struct Fold8 {
    origin: [f64; 3],
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    hinv: [[f64; 3]; 3],
    ortho: bool,
    n: [i32; 3],
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
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
    use std::arch::x86_64::{
        __m512d, _mm256_max_epi32, _mm256_min_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm512_add_pd, _mm512_cmp_pd_mask, _mm512_cvttpd_epi32, _mm512_div_pd,
        _mm512_loadu_pd, _mm512_mask_blend_pd, _mm512_mask_permutex2var_pd, _mm512_mul_pd,
        _mm512_permutex2var_pd, _mm512_roundscale_pd, _mm512_set1_pd, _mm512_setr_epi64,
        _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd, _CMP_GE_OQ,
    };
    let bc = |v: f64| _mm512_set1_pd(v);
    let one = bc(1.0);
    let (o, w, h, hinv, n) = (f.origin, f.widths, f.h, f.hinv, f.n);
    // `wrap01`: subtract the floor, and a value that rounds to one is zero.
    let wrap01 = |s: __m512d| {
        let t = _mm512_sub_pd(s, _mm512_roundscale_pd::<0x09>(s));
        _mm512_mask_blend_pd(
            _mm512_cmp_pd_mask::<_CMP_GE_OQ>(t, one),
            t,
            _mm512_setzero_pd(),
        )
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
    let mut so = [[0.0f64; 8]; 3];
    let mut po = [[0.0f64; 8]; 3];
    let mut bo = [[0i32; 8]; 3];
    for ax in 0..3 {
        _mm512_storeu_pd(so[ax].as_mut_ptr(), s[ax]);
        _mm512_storeu_pd(po[ax].as_mut_ptr(), cart(ax));
        _mm256_storeu_si256(bo[ax].as_mut_ptr() as *mut _, bin(ax));
    }
    (so, po, bo)
}

/// The three per-point arrays of a mesh: fractional, folded, bin.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
type MeshOut<'a> = (&'a mut [[f64; 3]], &'a mut [[f64; 3]], &'a mut [[i32; 3]]);

/// Fold `xyz[lo..hi]` into `frac`, `folded`, and `bin` at the same
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
    let (frac, folded, bin) = out;
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
            bin[i] = [b[0][l], b[1][l], b[2][l]];
            count(cell_index(b[0][l], b[1][l], b[2][l], n[0], n[1], n[2]));
        }
        k += 8;
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
