//! One OpenMP region for an unmasked k-nearest query.
//!
//! Bin, count, scan, fill, and select are phases of a single `omp parallel`.
//! Each slot keeps a private histogram, a private cursor row, and a private
//! candidate list. Slots meet only at the barriers between phases.

use std::cell::RefCell;
use std::os::raw::c_void;
use std::time::Instant;

use super::{
    box_diameter, select_one_block, walk_cells, BlockScan, Candidates, Geom, KHeap, Mesh, RowsOut,
    TEAM,
};
use crate::bins::{self, Recycled};
use crate::cell::Cell;
use crate::Error;

type Phase = extern "C" fn(*mut c_void, i32, i32, i64, i64);

extern "C" {
    fn lc_omp_max_threads() -> i32;
    fn lc_knn_one_region(
        ctx: *mut c_void,
        pin: i32,
        n_points: i64,
        n_bins: i64,
        n_blocks: i64,
        bin: Phase,
        count: Phase,
        scan_base: Phase,
        scan: Phase,
        fill: Phase,
        select: Phase,
        team: *mut i32,
    );
}

struct ThreadScratch {
    cand: Candidates,
    heap: KHeap,
    k: usize,
}

impl ThreadScratch {
    fn new(k: usize) -> Self {
        Self {
            cand: Candidates::default(),
            heap: KHeap::new(k),
            k,
        }
    }

    fn ensure(&mut self, k: usize) {
        if self.k != k {
            self.heap = KHeap::new(k);
            self.k = k;
        }
    }
}

struct ScratchPool {
    hist: Vec<usize>,
    base: Vec<usize>,
    chunk_base: Vec<usize>,
    scratches: Vec<ThreadScratch>,
}

impl ScratchPool {
    fn prepare(&mut self, threads: usize, ncell: usize, k: usize) {
        let rows = threads.saturating_mul(ncell);
        if self.hist.len() != rows {
            self.hist.resize(rows, 0);
        }
        if self.base.len() != rows {
            self.base.resize(rows, 0);
        }
        if self.chunk_base.len() != threads {
            self.chunk_base.resize(threads, 0);
        }
        if self.scratches.len() < threads {
            self.scratches
                .resize_with(threads, || ThreadScratch::new(k));
        }
        for scratch in &mut self.scratches[..threads] {
            scratch.ensure(k);
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<ScratchPool> = const {
        RefCell::new(ScratchPool {
            hist: Vec::new(),
            base: Vec::new(),
            chunk_base: Vec::new(),
            scratches: Vec::new(),
        })
    };
}

struct OmpCtx {
    xyz: *const [f64; 3],
    n: usize,
    cell: *const Cell,
    ncell: usize,
    frac: *mut [f64; 3],
    folded: *mut [f64; 3],
    key: *mut u32,
    offsets: *mut usize,
    occupants: *mut usize,
    slot_frac: *mut [f64; 3],
    slot_folded: *mut [f64; 3],
    counts: *mut usize,
    hist: *mut usize,
    base: *mut usize,
    chunk_base: *mut usize,
    mesh: *const Mesh,
    geom: Geom,
    scratches: *mut ThreadScratch,
    k: usize,
    max_reach: i32,
    mode: u8,
    /// Negative when the shell walk runs per cell instead of per block.
    side: i32,
    blocks: [i32; 3],
    path: crate::simd::Path,
    time_scan: bool,
    pop: bool,
    #[cfg(feature = "timing")]
    coarse: bool,
    #[cfg(feature = "timing")]
    t0: Instant,
    nn: *mut i32,
    d2: *mut f64,
}

/// OpenMP static share. The same division is in `src/knn_region.c`.
fn share(n: usize, slot: i32, nt: i32) -> (usize, usize) {
    let n = n as i64;
    let begin = (n * i64::from(slot)) / i64::from(nt);
    let end = (n * i64::from(slot + 1)) / i64::from(nt);
    (begin as usize, end as usize)
}

fn timed(ctx: &OmpCtx, slot: i32, body: impl FnOnce()) {
    if !ctx.pop {
        body();
        return;
    }
    let started = Instant::now();
    body();
    crate::pop::add_useful(slot as usize, started.elapsed().as_nanos() as u64);
}

extern "C" fn phase_bin(raw: *mut c_void, slot: i32, nt: i32, begin: i64, end: i64) {
    let ctx = unsafe { &*raw.cast::<OmpCtx>() };
    let begin = begin as usize;
    let end = end as usize;
    debug_assert_eq!((begin, end), share(ctx.n, slot, nt));
    timed(ctx, slot, || {
        let ncell = ctx.ncell;
        // Safety: slot `slot` owns histogram row `slot`, and `begin..end`
        // is that slot's disjoint slice of the point arrays. The OpenMP
        // static schedule gives this iteration to one thread.
        let hist =
            unsafe { std::slice::from_raw_parts_mut(ctx.hist.add(slot as usize * ncell), ncell) };
        hist.fill(0);
        let len = end - begin;
        let xyz = unsafe { std::slice::from_raw_parts(ctx.xyz.add(begin), len) };
        let frac = unsafe { std::slice::from_raw_parts_mut(ctx.frac.add(begin), len) };
        let folded = unsafe { std::slice::from_raw_parts_mut(ctx.folded.add(begin), len) };
        let key = unsafe { std::slice::from_raw_parts_mut(ctx.key.add(begin), len) };
        let cell = unsafe { &*ctx.cell };
        bins::fold_chunk(
            cell,
            xyz,
            ctx.geom.nbin,
            (frac, folded, key),
            hist,
            ctx.path,
        );
    });
}

extern "C" fn phase_count(raw: *mut c_void, slot: i32, nt: i32, begin: i64, end: i64) {
    let ctx = unsafe { &*raw.cast::<OmpCtx>() };
    let begin = begin as usize;
    let end = end as usize;
    debug_assert_eq!((begin, end), share(ctx.ncell, slot, nt));
    timed(ctx, slot, || {
        let ncell = ctx.ncell;
        // Safety: this slot writes `counts[begin..end]` only. Histograms are
        // read-only after the bin barrier.
        let counts = unsafe { std::slice::from_raw_parts_mut(ctx.counts.add(begin), end - begin) };
        for (offset, bin) in (begin..end).enumerate() {
            let mut sum = 0usize;
            for other in 0..nt as usize {
                sum += unsafe { *ctx.hist.add(other * ncell + bin) };
            }
            counts[offset] = sum;
        }
    });
}

extern "C" fn phase_scan_base(raw: *mut c_void, slot: i32, nt: i32, _begin: i64, _end: i64) {
    let ctx = unsafe { &*raw.cast::<OmpCtx>() };
    timed(ctx, slot, || {
        // Safety: `omp single` runs this on one thread. `counts` is complete.
        let counts = unsafe { std::slice::from_raw_parts(ctx.counts, ctx.ncell) };
        let chunk = unsafe { std::slice::from_raw_parts_mut(ctx.chunk_base, nt as usize) };
        let mut acc = 0usize;
        for t in 0..nt {
            let (lo, hi) = share(ctx.ncell, t, nt);
            let mut sum = 0usize;
            for count in &counts[lo..hi] {
                sum += *count;
            }
            chunk[t as usize] = acc;
            acc += sum;
        }
        unsafe { *ctx.offsets.add(ctx.ncell) = acc };
    });
}

extern "C" fn phase_scan(raw: *mut c_void, slot: i32, nt: i32, begin: i64, end: i64) {
    let ctx = unsafe { &*raw.cast::<OmpCtx>() };
    let begin = begin as usize;
    let end = end as usize;
    debug_assert_eq!((begin, end), share(ctx.ncell, slot, nt));
    timed(ctx, slot, || {
        let ncell = ctx.ncell;
        let counts = unsafe { std::slice::from_raw_parts(ctx.counts, ncell) };
        let mut acc = unsafe { *ctx.chunk_base.add(slot as usize) };
        for (bin, &count) in counts.iter().enumerate().skip(begin).take(end - begin) {
            unsafe { *ctx.offsets.add(bin) = acc };
            let mut cursor = acc;
            for other in 0..nt as usize {
                // Safety: this slot owns every thread's cursor at `bin`,
                // and other slots own other bins.
                unsafe { *ctx.base.add(other * ncell + bin) = cursor };
                cursor += unsafe { *ctx.hist.add(other * ncell + bin) };
            }
            acc += count;
        }
    });
}

extern "C" fn phase_fill(raw: *mut c_void, slot: i32, nt: i32, begin: i64, end: i64) {
    let ctx = unsafe { &*raw.cast::<OmpCtx>() };
    let begin = begin as usize;
    let end = end as usize;
    debug_assert_eq!((begin, end), share(ctx.n, slot, nt));
    let _ = nt;
    timed(ctx, slot, || {
        let ncell = ctx.ncell;
        // Safety: key, frac, and folded are read-only after the scan barrier.
        // This slot's cursor row is private. Each cursor value was the
        // exclusive start of this slot's points in that bin, so the
        // destinations do not alias another slot.
        let key = unsafe { std::slice::from_raw_parts(ctx.key, ctx.n) };
        let frac = unsafe { std::slice::from_raw_parts(ctx.frac, ctx.n) };
        let folded = unsafe { std::slice::from_raw_parts(ctx.folded, ctx.n) };
        let cursors =
            unsafe { std::slice::from_raw_parts_mut(ctx.base.add(slot as usize * ncell), ncell) };
        for i in begin..end {
            let bin = key[i] as usize;
            let dest = cursors[bin];
            cursors[bin] = dest + 1;
            unsafe {
                ctx.occupants.add(dest).write(i);
                ctx.slot_frac.add(dest).write(frac[i]);
                ctx.slot_folded.add(dest).write(folded[i]);
            }
        }
    });
}

extern "C" fn phase_select(raw: *mut c_void, slot: i32, nt: i32, begin: i64, end: i64) {
    let ctx = unsafe { &*raw.cast::<OmpCtx>() };
    let begin = begin as usize;
    let end = end as usize;
    let _ = nt;
    #[cfg(feature = "timing")]
    if ctx.coarse && begin == 0 {
        crate::simd::add_mesh_ns(ctx.t0.elapsed().as_nanos());
    }
    timed(ctx, slot, || {
        // Safety: the mesh is complete and read-only. Slot `slot` owns
        // `scratches[slot]`. Each source is in one cell, and each cell is
        // in one block, so output rows do not alias.
        let mesh = unsafe { &*ctx.mesh };
        let nn = RowsOut(ctx.nn);
        let dd = if ctx.d2.is_null() {
            None
        } else {
            Some(RowsOut(ctx.d2))
        };
        if ctx.side < 0 {
            match ctx.mode {
                0 => walk_cells::<0>(mesh, &ctx.geom, ctx.k, ctx.max_reach, (nn, dd), begin, end),
                1 => walk_cells::<1>(mesh, &ctx.geom, ctx.k, ctx.max_reach, (nn, dd), begin, end),
                _ => walk_cells::<2>(mesh, &ctx.geom, ctx.k, ctx.max_reach, (nn, dd), begin, end),
            }
            return;
        }
        let scan = BlockScan {
            mesh,
            geom: &ctx.geom,
            k: ctx.k,
            max_reach: ctx.max_reach,
            side: ctx.side,
            blocks: ctx.blocks,
            path: ctx.path,
            time_scan: ctx.time_scan,
            nn,
            dd,
        };
        let scratch = unsafe { &mut *ctx.scratches.add(slot as usize) };
        for blk in begin..end {
            match ctx.mode {
                0 => select_one_block::<0>(&scan, &mut scratch.cand, &mut scratch.heap, blk),
                1 => select_one_block::<1>(&scan, &mut scratch.cand, &mut scratch.heap, blk),
                _ => select_one_block::<2>(&scan, &mut scratch.cand, &mut scratch.heap, blk),
            }
        }
    });
}

fn assemble(parts: Recycled, nbin: [i32; 3], widths: [f64; 3]) -> Mesh {
    Mesh {
        nx: nbin[0],
        ny: nbin[1],
        nz: nbin[2],
        widths,
        frac: parts.frac,
        folded: parts.folded,
        slot_frac: parts.slot_frac,
        slot_folded: parts.slot_folded,
        key: parts.key,
        offsets: parts.offsets,
        occupants: parts.occupants,
        counts: parts.counts,
        cursor: parts.cursor,
        hist: parts.hist,
    }
}

fn debug_mesh(mesh: &Mesh, n: usize) {
    if !cfg!(debug_assertions) {
        return;
    }
    let ncell = mesh.offsets.len() - 1;
    assert_eq!(mesh.offsets[ncell], n, "bin fill dropped a point");
    let mut seen = vec![0u8; n];
    for bin in 0..ncell {
        let (lo, hi) = (mesh.offsets[bin], mesh.offsets[bin + 1]);
        assert!(lo <= hi && hi <= n);
        for pair in mesh.occupants[lo..hi].windows(2) {
            assert!(pair[0] < pair[1], "bin {bin} is not in index order");
        }
        for &i in &mesh.occupants[lo..hi] {
            assert!(i < n && seen[i] == 0, "occupant {i}");
            seen[i] = 1;
        }
    }
    assert!(seen.iter().all(|&flag| flag == 1));
}

pub(super) fn run(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    edge: f64,
    out_nn: &mut [i32],
    mut out_d2: Option<&mut [f64]>,
) -> Result<(), Error> {
    let (nbin, widths, ncell) = bins::grid(simbox, edge)?;
    let n = xyz.len();
    let n_points = i64::try_from(n).map_err(|_| Error::Overflow)?;
    let n_bins = i64::try_from(ncell).map_err(|_| Error::Overflow)?;
    let threads = unsafe { lc_omp_max_threads() }.max(1) as usize;
    let mut parts = bins::take_recycled();
    if parts.frac.len() != n {
        parts.frac.resize(n, [0.0; 3]);
        parts.folded.resize(n, [0.0; 3]);
        parts.key.resize(n, 0);
    }
    if parts.offsets.len() != ncell + 1 {
        parts.offsets.resize(ncell + 1, 0);
    }
    parts.occupants.resize(n, 0);
    parts.slot_frac.resize(n, [0.0; 3]);
    parts.slot_folded.resize(n, [0.0; 3]);
    if parts.counts.len() != ncell {
        parts.counts.resize(ncell, 0);
    }
    let mut mesh = assemble(parts, nbin, widths);
    let side = super::block_side(&mesh, nbin);
    let (n_select, blocks, side_i) = if let Some(side) = side {
        let blocks = [
            (nbin[0] + side - 1) / side,
            (nbin[1] + side - 1) / side,
            (nbin[2] + side - 1) / side,
        ];
        let nblock = (blocks[0] as usize)
            .saturating_mul(blocks[1] as usize)
            .saturating_mul(blocks[2] as usize);
        (nblock, blocks, side)
    } else {
        (ncell, [0; 3], -1)
    };
    let n_blocks = i64::try_from(n_select).map_err(|_| Error::Overflow)?;
    let cell_min = (widths[0] / f64::from(nbin[0]))
        .min(widths[1] / f64::from(nbin[1]))
        .min(widths[2] / f64::from(nbin[2]));
    let need = (box_diameter(simbox) / cell_min).ceil();
    let image = nbin[0].max(nbin[1]).max(nbin[2]) / 2 + 2;
    let max_reach = if need.is_finite() && need < 1.0e7 {
        (need as i32).saturating_add(3).max(image)
    } else {
        image
    };
    let geom = Geom {
        cols: [simbox.a(), simbox.b(), simbox.c()],
        lengths: simbox.widths(),
        nbin,
        widths,
    };
    let mode = if simbox.is_ortho() {
        0
    } else if simbox.is_restricted() {
        1
    } else {
        2
    };
    let path = crate::simd::select();
    #[cfg(feature = "timing")]
    let time_scan = crate::simd::scan_timing_enabled();
    #[cfg(not(feature = "timing"))]
    let time_scan = false;
    #[cfg(feature = "timing")]
    let coarse = crate::simd::coarse_timing_enabled();
    let pop = crate::pop::enabled();
    let pin = i32::from(super::PIN_LEAST_LOADED.load(std::sync::atomic::Ordering::Relaxed));
    let d2 = out_d2
        .as_mut()
        .map(|row| row.as_mut_ptr())
        .unwrap_or(std::ptr::null_mut());

    SCRATCH.with(|slot| {
        let pool = &mut *slot.borrow_mut();
        pool.prepare(threads, ncell, k);
        // Field pointers come from one raw mesh pointer. Phases before
        // select write disjoint ranges through them. Select then reads
        // the mesh, after those borrows have ended at the barriers.
        let mesh_ptr = &mut mesh as *mut Mesh;
        let frac = unsafe { (*mesh_ptr).frac.as_mut_ptr() };
        let folded = unsafe { (*mesh_ptr).folded.as_mut_ptr() };
        let key = unsafe { (*mesh_ptr).key.as_mut_ptr() };
        let offsets = unsafe { (*mesh_ptr).offsets.as_mut_ptr() };
        let occupants = unsafe { (*mesh_ptr).occupants.as_mut_ptr() };
        let slot_frac = unsafe { (*mesh_ptr).slot_frac.as_mut_ptr() };
        let slot_folded = unsafe { (*mesh_ptr).slot_folded.as_mut_ptr() };
        let counts = unsafe { (*mesh_ptr).counts.as_mut_ptr() };
        let mut ctx = OmpCtx {
            xyz: xyz.as_ptr(),
            n,
            cell: simbox,
            ncell,
            frac,
            folded,
            key,
            offsets,
            occupants,
            slot_frac,
            slot_folded,
            counts,
            hist: pool.hist.as_mut_ptr(),
            base: pool.base.as_mut_ptr(),
            chunk_base: pool.chunk_base.as_mut_ptr(),
            mesh: mesh_ptr,
            geom,
            scratches: pool.scratches.as_mut_ptr(),
            k,
            max_reach,
            mode,
            side: side_i,
            blocks,
            path,
            time_scan,
            pop,
            #[cfg(feature = "timing")]
            coarse,
            #[cfg(feature = "timing")]
            t0: Instant::now(),
            nn: out_nn.as_mut_ptr(),
            d2,
        };
        let mut team = 0i32;
        unsafe {
            lc_knn_one_region(
                (&mut ctx as *mut OmpCtx).cast(),
                pin,
                n_points,
                n_bins,
                n_blocks,
                phase_bin,
                phase_count,
                phase_scan_base,
                phase_scan,
                phase_fill,
                phase_select,
                &mut team,
            );
        }
        TEAM.store(team.max(1) as usize, std::sync::atomic::Ordering::Relaxed);
        if pop {
            crate::pop::set_threads(team.max(1) as usize);
        }
    });
    debug_mesh(&mesh, n);
    Ok(())
}
