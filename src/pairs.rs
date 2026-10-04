//! Cutoff pair list with integer cell shifts.
//!
//! [`knearest`](crate::knearest) unique-indexes the neighbour and
//! drops the image. This walk keeps every atom-image pair whose
//! squared distance is strictly below `cutoff²`, including periodic
//! self-images. Each unordered pair is tested once. Hits from a bin
//! are buffered, then the rows are written. A full list writes both
//! `(i, j, S)` and `(j, i, -S)`. Displacement is `q - p + lattice_shift(S)`,
//! which is [`Cell::dist2_shifted`](crate::Cell::dist2_shifted). This is
//! not [`dist2_ortho_diffs`](crate::dist2_ortho_diffs): that Highway
//! kernel wraps a raw difference into the central cell, and a second
//! wrap on an already shifted delta changes which rows survive.

use crate::bins::{self, axis_gap, Mesh};
use crate::cell::Cell;
use crate::Error;

const MAX_IMAGES: i64 = 1 << 24;

/// One atom-image pair inside the cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    /// Source index.
    pub i: usize,
    /// Target index.
    pub j: usize,
    /// Integer cell shift applied to the target (`S` in vesin/tonari).
    pub shift: [i32; 3],
    /// Squared distance after the shift.
    pub dist2: f64,
}

fn keep_half(i: usize, j: usize, shift: [i32; 3]) -> bool {
    if i != j {
        return i < j;
    }
    for s in shift {
        if s != 0 {
            return s < 0;
        }
    }
    true
}

/// Pairs with `dist2 < cutoff²`, one row per atom-image.
///
/// `half` keeps the canonical side of `(i, j, S)` vs `(j, i, -S)`.
/// Zero-shift self pairs are omitted. Periodic self-images stay.
/// Image count above 2^24 is [`Error::TooManyImages`].
pub fn pairs_within(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    half: bool,
) -> Result<Vec<Pair>, Error> {
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(Error::BadCutoff);
    }
    if xyz.is_empty() {
        return Err(Error::Empty);
    }
    let n = xyz.len();
    if n > u32::MAX as usize {
        return Err(Error::Overflow);
    }
    if let Some(m) = mask {
        if m.len() != n {
            return Err(Error::MaskLen);
        }
    }
    let active: Vec<usize> = (0..n)
        .filter(|&i| mask.map(|m| m[i]).unwrap_or(true))
        .collect();
    if active.is_empty() {
        return Ok(Vec::new());
    }
    retain_pair_pages();

    let w = simbox.widths();
    // Reserve the pair buffer before the mesh. A repeated call can then
    // reuse that chunk instead of letting smaller allocs split it.
    let rows = pair_capacity(active.len(), w, cutoff, half);
    // Reserve the pair buffer before the mesh so a repeated call reuses
    // that chunk. Several threads later write disjoint ranges of it.
    let found = Vec::with_capacity(rows);
    let mut scratch = Scratch::new();
    let parallel_walk = {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads() > 1 && active.len() >= PARALLEL_PAIRS
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    };
    if !parallel_walk {
        let hits = if half { rows } else { (rows + 1) / 2 };
        scratch.reserve_more(hits);
    }
    let mut image_count: i64 = 1;
    let mut repeats = [0i32; 3];
    for a in 0..3 {
        let r = (cutoff / w[a]).ceil();
        if !r.is_finite() || r > i32::MAX as f64 {
            return Err(Error::TooManyImages);
        }
        repeats[a] = r as i32;
        let factor = 2 * i64::from(repeats[a]) + 1;
        if image_count > MAX_IMAGES / factor {
            return Err(Error::TooManyImages);
        }
        image_count *= factor;
    }

    let edge = bins::target_edge(simbox, cell_hint, cutoff);
    let mesh = Mesh::build(xyz, simbox, Some(&active), edge)?;
    let cell_min = (mesh.widths[0] / f64::from(mesh.nx))
        .min(mesh.widths[1] / f64::from(mesh.ny))
        .min(mesh.widths[2] / f64::from(mesh.nz));
    let reach_cut = (cutoff / cell_min).ceil();
    if !reach_cut.is_finite() || reach_cut > i32::MAX as f64 {
        return Err(Error::TooManyImages);
    }
    let max_reach = (reach_cut as i32)
        .max(repeats[0] * mesh.nx)
        .max(repeats[1] * mesh.ny)
        .max(repeats[2] * mesh.nz)
        .max(1);
    let cut2 = cutoff * cutoff;
    // One reach for every atom: the shorter gap at either edge of a bin.
    // The box is symmetric, so each unordered pair is visited from one
    // side and written out in both shift directions when `half` is off.
    let reach = uniform_reach([mesh.nx, mesh.ny, mesh.nz], mesh.widths, cut2, max_reach);
    let nslot = mesh.occupants.len();
    let mut coords = Coords {
        x: vec![0.0; nslot],
        y: vec![0.0; nslot],
        z: vec![0.0; nslot],
    };
    for (slot, &i) in mesh.occupants.iter().enumerate() {
        let p = mesh.folded[i];
        coords.x[slot] = p[0];
        coords.y[slot] = p[1];
        coords.z[slot] = p[2];
    }
    let partners = build_partners(&mesh, simbox, reach, cut2);
    let walk = Walk {
        mesh: &mesh,
        coords: &coords,
        cut2,
        half,
        partners: &partners,
        simd: simd_mode(),
    };
    Ok(walk.collect(found, scratch))
}

fn pair_capacity(n_src: usize, widths: [f64; 3], cutoff: f64, half: bool) -> usize {
    let volume = (widths[0] * widths[1] * widths[2]).max(1.0e-30);
    let shell = 4.1887902047863905 * cutoff * cutoff * cutoff;
    let neighbors = ((n_src.max(1) as f64) * shell / volume).ceil().max(1.0);
    let rows = neighbors * if half { 0.5 } else { 1.0 };
    ((n_src as f64) * rows * 1.25) as usize + 16
}

/// glibc drops a large free buffer (`MADV_DONTNEED`) and the next cutoff
/// list faults every page. Keep those pages on the heap. The knobs are
/// `M_TRIM_THRESHOLD` and `M_MMAP_THRESHOLD`.
fn retain_pair_pages() {
    #[cfg(target_os = "linux")]
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            // Safety: mallopt is process-wide and these two knobs are integers.
            unsafe {
                extern "C" {
                    fn mallopt(param: i32, value: i32) -> i32;
                }
                let _ = mallopt(-1, -1);
                let _ = mallopt(-3, 256 << 20);
            }
        });
    }
}

struct Partner {
    lo: usize,
    hi: usize,
    shift_s: [i32; 3],
    shift: [f64; 3],
}

struct PartnerList {
    off: Vec<usize>,
    items: Vec<Partner>,
}

fn bin_gap(src: i32, dst: i32, width: f64) -> f64 {
    let lo_s = f64::from(src) * width;
    let hi_s = f64::from(src + 1) * width;
    let lo_d = f64::from(dst) * width;
    let hi_d = f64::from(dst + 1) * width;
    if hi_s <= lo_d {
        lo_d - hi_s
    } else if hi_d <= lo_s {
        lo_s - hi_d
    } else {
        0.0
    }
}

fn bins_too_far(ortho: bool, src: [i32; 3], dst: [i32; 3], width: [f64; 3], cut2: f64) -> bool {
    let gap = [
        bin_gap(src[0], dst[0], width[0]),
        bin_gap(src[1], dst[1], width[1]),
        bin_gap(src[2], dst[2], width[2]),
    ];
    let bound = if ortho {
        let d = (gap[0] * gap[0] + gap[1] * gap[1] + gap[2] * gap[2]).sqrt();
        let d = bins::certify(d);
        d * d
    } else {
        let d = bins::certify(gap[0].max(gap[1]).max(gap[2]));
        d * d
    };
    bound >= cut2
}

fn build_partners(mesh: &Mesh, simbox: &Cell, reach: [i32; 3], cut2: f64) -> PartnerList {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    let ncell = (nx as usize) * (ny as usize) * (nz as usize);
    let width = [
        mesh.widths[0] / f64::from(nx),
        mesh.widths[1] / f64::from(ny),
        mesh.widths[2] / f64::from(nz),
    ];
    let ortho = simbox.is_ortho();
    let [rx, ry, rz] = reach;
    let mut off = vec![0usize; ncell + 1];
    let mut items = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let cell = ((iz * ny + iy) * nx + ix) as usize;
                let src = [ix, iy, iz];
                for dz in -rz..=rz {
                    for dy in -ry..=ry {
                        for dx in -rx..=rx {
                            if !keep_dir(dx, dy, dz) {
                                continue;
                            }
                            let dst = [ix + dx, iy + dy, iz + dz];
                            if bins_too_far(ortho, src, dst, width, cut2) {
                                continue;
                            }
                            let jc = mesh.cell_of(dst[0], dst[1], dst[2]);
                            let lo = mesh.offsets[jc];
                            let hi = mesh.offsets[jc + 1];
                            if lo == hi {
                                continue;
                            }
                            let shift_s = [
                                dst[0].div_euclid(nx),
                                dst[1].div_euclid(ny),
                                dst[2].div_euclid(nz),
                            ];
                            items.push(Partner {
                                lo,
                                hi,
                                shift_s,
                                shift: simbox.lattice_shift(shift_s[0], shift_s[1], shift_s[2]),
                            });
                        }
                    }
                }
                off[cell + 1] = items.len();
            }
        }
    }
    PartnerList { off, items }
}

struct Coords {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

fn simd_mode() -> u8 {
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return 2;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            return 1;
        }
    }
    0
}

/// `true` for exactly one of `(d)` and `(-d)`. The zero offset is the home cell.
fn keep_dir(dx: i32, dy: i32, dz: i32) -> bool {
    if dx != 0 {
        return dx > 0;
    }
    if dy != 0 {
        return dy > 0;
    }
    if dz != 0 {
        return dz > 0;
    }
    false
}

fn uniform_reach(nbin: [i32; 3], widths: [f64; 3], cut2: f64, max_reach: i32) -> [i32; 3] {
    let mut reach = [1i32; 3];
    loop {
        let mut grew = false;
        for a in 0..3 {
            let n = nbin[a];
            let w = widths[a];
            let nf = f64::from(n);
            let s_hi = (1.0 - 1.0e-12) / nf;
            let gap = bins::certify(
                axis_gap(0.0, 0, reach[a], n, w).min(axis_gap(s_hi, 0, reach[a], n, w)),
            );
            let bound = if gap > 0.0 && gap.is_finite() {
                gap * gap
            } else {
                0.0
            };
            if bound < cut2 && reach[a] < max_reach {
                reach[a] += 1;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    reach
}

struct Walk<'a> {
    mesh: &'a Mesh,
    coords: &'a Coords,
    cut2: f64,
    half: bool,
    partners: &'a PartnerList,
    /// 2 = AVX-512, 1 = AVX, 0 = scalar.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    simd: u8,
}

#[derive(Clone, Copy)]
struct Block {
    i_lo: usize,
    i_hi: usize,
    j_lo: usize,
    j_hi: usize,
    shift_s: [i32; 3],
    shift: [f64; 3],
    /// Home cell: source `s` only sees occupants `s + 1 ..`.
    tri: bool,
}

/// Atom count where row ranges split across threads. Below this the
/// pair buffer is smaller than the spawn, so the walk stays on one thread.
#[cfg(feature = "parallel")]
const PARALLEL_PAIRS: usize = 512;

/// Hits for the whole chunk. The distance loop appends here, then one
/// pass writes the rows. Runs share a shift so the inner loop does not.
struct Scratch {
    atom: Vec<u32>,
    js: Vec<u64>,
    d2: Vec<f64>,
    run_shift: Vec<[i32; 3]>,
    run_end: Vec<usize>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            atom: Vec::new(),
            js: Vec::new(),
            d2: Vec::new(),
            run_shift: Vec::new(),
            run_end: Vec::new(),
        }
    }

    fn reserve_more(&mut self, extra: usize) {
        let need = self.js.len().saturating_add(extra);
        if self.js.capacity() < need {
            self.atom.reserve(extra);
            self.js.reserve(extra);
            self.d2.reserve(extra);
        }
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    fn finish(&mut self, n: usize) {
        debug_assert!(n <= self.js.capacity());
        unsafe {
            self.atom.set_len(n);
            self.js.set_len(n);
            self.d2.set_len(n);
        }
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    fn note(&mut self, before: usize, shift: [i32; 3]) {
        let after = self.js.len();
        if after > before {
            self.run_shift.push(shift);
            self.run_end.push(after);
        }
    }
}

/// Worst-case hits kept in the scratch buffer. Larger blocks use the scalar walk.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const MAX_SCRATCH: usize = 1 << 20;

#[inline(never)]
fn write_full(
    dst: &mut [std::mem::MaybeUninit<Pair>],
    atom: &[u32],
    js: &[u64],
    d2: &[f64],
    shifts: &[[i32; 3]],
    ends: &[usize],
) {
    let dp = dst.as_mut_ptr();
    let ap = atom.as_ptr();
    let jp = js.as_ptr();
    let yp = d2.as_ptr();
    let mut out = 0usize;
    let mut lo = 0usize;
    for (r, &hi) in ends.iter().enumerate() {
        let shift = shifts[r];
        let neg = [-shift[0], -shift[1], -shift[2]];
        for k in lo..hi {
            unsafe {
                let i = (*ap.add(k)) as usize;
                let j = (*jp.add(k)) as usize;
                let dist2 = *yp.add(k);
                (*dp.add(out)).write(Pair { i, j, shift, dist2 });
                (*dp.add(out + 1)).write(Pair {
                    i: j,
                    j: i,
                    shift: neg,
                    dist2,
                });
            }
            out += 2;
        }
        lo = hi;
    }
}

#[inline(never)]
fn write_half(
    dst: &mut [std::mem::MaybeUninit<Pair>],
    atom: &[u32],
    js: &[u64],
    d2: &[f64],
    shifts: &[[i32; 3]],
    ends: &[usize],
) {
    let dp = dst.as_mut_ptr();
    let ap = atom.as_ptr();
    let jp = js.as_ptr();
    let yp = d2.as_ptr();
    let mut out = 0usize;
    let mut lo = 0usize;
    for (r, &hi) in ends.iter().enumerate() {
        let shift = shifts[r];
        let neg = [-shift[0], -shift[1], -shift[2]];
        for k in lo..hi {
            unsafe {
                let i = (*ap.add(k)) as usize;
                let j = (*jp.add(k)) as usize;
                let dist2 = *yp.add(k);
                let pair = if keep_half(i, j, shift) {
                    Pair { i, j, shift, dist2 }
                } else {
                    Pair {
                        i: j,
                        j: i,
                        shift: neg,
                        dist2,
                    }
                };
                (*dp.add(out)).write(pair);
            }
            out += 1;
        }
        lo = hi;
    }
}

impl Walk<'_> {
    fn record(&self, found: &mut Vec<Pair>, i: usize, j: usize, shift: [i32; 3], dist2: f64) {
        let neg = [-shift[0], -shift[1], -shift[2]];
        if !self.half {
            found.push(Pair { i, j, shift, dist2 });
            found.push(Pair {
                i: j,
                j: i,
                shift: neg,
                dist2,
            });
            return;
        }
        if keep_half(i, j, shift) {
            found.push(Pair { i, j, shift, dist2 });
        } else {
            found.push(Pair {
                i: j,
                j: i,
                shift: neg,
                dist2,
            });
        }
    }

    fn commit(&self, found: &mut Vec<Pair>, scratch: &Scratch) {
        let n = scratch.js.len();
        if n == 0 {
            return;
        }
        debug_assert_eq!(scratch.atom.len(), n);
        debug_assert_eq!(scratch.d2.len(), n);
        let rows = if self.half { n } else { n * 2 };
        found.reserve(rows);
        let len0 = found.len();
        {
            let spare = &mut found.spare_capacity_mut()[..rows];
            if self.half {
                write_half(
                    spare,
                    &scratch.atom,
                    &scratch.js,
                    &scratch.d2,
                    &scratch.run_shift,
                    &scratch.run_end,
                );
            } else {
                write_full(
                    spare,
                    &scratch.atom,
                    &scratch.js,
                    &scratch.d2,
                    &scratch.run_shift,
                    &scratch.run_end,
                );
            }
        }
        unsafe {
            found.set_len(len0 + rows);
        }
    }

    fn scan_block(&self, found: &mut Vec<Pair>, scratch: &mut Scratch, block: &Block) {
        if block.i_lo >= block.i_hi || block.j_lo >= block.j_hi {
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = scratch;
            self.scan_scalar(found, block);
        }
        #[cfg(target_arch = "x86_64")]
        {
            let ns = block.i_hi - block.i_lo;
            let span = block.j_hi - block.j_lo;
            if span == 0 || ns > MAX_SCRATCH / span {
                self.scan_scalar(found, block);
                return;
            }
            let before = scratch.js.len();
            scratch.reserve_more(ns * span);
            #[cfg(linkcell_avx512)]
            if self.simd == 2 {
                unsafe {
                    avx512_scan(
                        &self.coords.x,
                        &self.coords.y,
                        &self.coords.z,
                        &self.mesh.occupants,
                        self.cut2,
                        block,
                        scratch,
                    );
                }
                scratch.note(before, block.shift_s);
                return;
            }
            if self.simd >= 1 {
                unsafe {
                    avx_scan(
                        &self.coords.x,
                        &self.coords.y,
                        &self.coords.z,
                        &self.mesh.occupants,
                        self.cut2,
                        block,
                        scratch,
                    );
                }
                scratch.note(before, block.shift_s);
                return;
            }
            self.scan_scalar(found, block);
        }
    }

    fn scan_scalar(&self, found: &mut Vec<Pair>, block: &Block) {
        let sx = block.shift[0];
        let sy = block.shift[1];
        let sz = block.shift[2];
        for s in block.i_lo..block.i_hi {
            let j_lo = if block.tri { s + 1 } else { block.j_lo };
            if j_lo >= block.j_hi {
                continue;
            }
            let i = self.mesh.occupants[s];
            let px = self.coords.x[s] - sx;
            let py = self.coords.y[s] - sy;
            let pz = self.coords.z[s] - sz;
            for slot in j_lo..block.j_hi {
                let dx = self.coords.x[slot] - px;
                let dy = self.coords.y[slot] - py;
                let dz = self.coords.z[slot] - pz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < self.cut2 {
                    let j = self.mesh.occupants[slot];
                    if j == i && block.shift_s == [0, 0, 0] {
                        continue;
                    }
                    self.record(found, i, j, block.shift_s, d2);
                }
            }
        }
    }

    fn append_cell(&self, found: &mut Vec<Pair>, scratch: &mut Scratch, cell: usize) {
        let lo = self.mesh.offsets[cell];
        let hi = self.mesh.offsets[cell + 1];
        if lo == hi {
            return;
        }
        if hi - lo >= 2 {
            self.scan_block(
                found,
                scratch,
                &Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo: lo,
                    j_hi: hi,
                    shift_s: [0, 0, 0],
                    shift: [0.0; 3],
                    tri: true,
                },
            );
        }
        let p0 = self.partners.off[cell];
        let p1 = self.partners.off[cell + 1];
        for partner in &self.partners.items[p0..p1] {
            self.scan_block(
                found,
                scratch,
                &Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo: partner.lo,
                    j_hi: partner.hi,
                    shift_s: partner.shift_s,
                    shift: partner.shift,
                    tri: false,
                },
            );
        }
    }

    fn guess_rows(&self, n_src: usize) -> usize {
        let w = self.mesh.widths;
        let volume = (w[0] * w[1] * w[2]).max(1.0e-30);
        let radius = self.cut2.sqrt();
        let shell = 4.1887902047863905 * radius * radius * radius;
        let n = self.mesh.folded.len().max(1) as f64;
        let neighbors = (n * shell / volume).ceil().max(1.0);
        let rows = neighbors * if self.half { 0.5 } else { 1.0 };
        ((n_src as f64) * rows * 1.25) as usize
    }

    #[cfg(feature = "parallel")]
    fn row_count(&self, scratch: &Scratch) -> usize {
        let n = scratch.js.len();
        if self.half {
            n
        } else {
            n * 2
        }
    }

    fn fill_only(&self, found: &mut Vec<Pair>, scratch: &mut Scratch, start: usize, end: usize) {
        let n_src = self.mesh.offsets[end] - self.mesh.offsets[start];
        let rows = self.guess_rows(n_src);
        let hits = if self.half { rows } else { (rows + 1) / 2 };
        scratch.reserve_more(hits);
        for cell in start..end {
            self.append_cell(found, scratch, cell);
        }
    }

    fn gather_into(&self, found: &mut Vec<Pair>, scratch: &mut Scratch, start: usize, end: usize) {
        let n_src = self.mesh.offsets[end] - self.mesh.offsets[start];
        let rows = self.guess_rows(n_src);
        // reserve(n) makes capacity >= len + n.
        if found.capacity() < rows {
            found.reserve(rows - found.len());
        }
        self.fill_only(found, scratch, start, end);
        self.commit(found, scratch);
    }

    fn collect(&self, found: Vec<Pair>, mut scratch: Scratch) -> Vec<Pair> {
        let ncell = self.mesh.offsets.len() - 1;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let threads = rayon::current_num_threads().max(1);
            if threads == 1 || ncell <= 1 || self.mesh.occupants.len() < PARALLEL_PAIRS {
                let _timer = crate::pop::JobTimer::new();
                let mut found = found;
                self.gather_into(&mut found, &mut scratch, 0, ncell);
                return found;
            }
            let ranges = cell_ranges(&self.mesh.offsets, threads);
            if ranges.len() <= 1 {
                let _timer = crate::pop::JobTimer::new();
                let mut found = found;
                self.gather_into(&mut found, &mut scratch, 0, ncell);
                return found;
            }
            let chunks: Vec<(Scratch, Vec<Pair>)> = ranges
                .into_par_iter()
                .map(|(start, end)| {
                    let _timer = crate::pop::JobTimer::new();
                    let mut scratch = Scratch::new();
                    let mut extra = Vec::new();
                    self.fill_only(&mut extra, &mut scratch, start, end);
                    (scratch, extra)
                })
                .collect();
            self.place_chunks(found, &chunks)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _timer = crate::pop::JobTimer::new();
            let mut found = found;
            self.gather_into(&mut found, &mut scratch, 0, ncell);
            found
        }
    }

    #[cfg(feature = "parallel")]
    fn place_chunks(&self, mut found: Vec<Pair>, chunks: &[(Scratch, Vec<Pair>)]) -> Vec<Pair> {
        use rayon::prelude::*;
        let mut rows = Vec::with_capacity(chunks.len());
        let mut total = 0usize;
        for (scratch, extra) in chunks {
            let n = self.row_count(scratch) + extra.len();
            rows.push(n);
            total += n;
        }
        // reserve(n) makes capacity >= len + n. The ideal-gas estimate
        // may already hold a shorter buffer.
        if found.capacity() < total {
            found.reserve(total - found.len());
        }
        let mut off = Vec::with_capacity(chunks.len() + 1);
        off.push(0usize);
        for n in &rows {
            off.push(off.last().copied().unwrap() + n);
        }
        let base = SharePtr(found.as_mut_ptr() as *mut std::mem::MaybeUninit<Pair>);
        let half = self.half;
        // Safety: each job writes `rows[t]` slots starting at `off[t]`.
        // Those ranges partition `0..total` and nothing reads them until
        // every job has joined. `found.len()` stays 0 until then.
        chunks
            .par_iter()
            .enumerate()
            .for_each(|(t, (scratch, extra))| {
                let _timer = crate::pop::JobTimer::new();
                let mut at = off[t];
                let hit_rows = if half {
                    scratch.js.len()
                } else {
                    scratch.js.len() * 2
                };
                if hit_rows > 0 {
                    let dst = unsafe { std::slice::from_raw_parts_mut(base.slot(at), hit_rows) };
                    if half {
                        write_half(
                            dst,
                            &scratch.atom,
                            &scratch.js,
                            &scratch.d2,
                            &scratch.run_shift,
                            &scratch.run_end,
                        );
                    } else {
                        write_full(
                            dst,
                            &scratch.atom,
                            &scratch.js,
                            &scratch.d2,
                            &scratch.run_shift,
                            &scratch.run_end,
                        );
                    }
                    at += hit_rows;
                }
                if !extra.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            extra.as_ptr(),
                            base.slot(at) as *mut Pair,
                            extra.len(),
                        );
                    }
                }
            });
        unsafe {
            found.set_len(total);
        }
        found
    }
}

/// Shared destination for disjoint pair-row ranges.
///
/// Safety: jobs write distinct slots and do not read a slot another job writes.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SharePtr(*mut std::mem::MaybeUninit<Pair>);
#[cfg(feature = "parallel")]
unsafe impl Send for SharePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SharePtr {}
#[cfg(feature = "parallel")]
impl SharePtr {
    unsafe fn slot(self, index: usize) -> *mut std::mem::MaybeUninit<Pair> {
        self.0.add(index)
    }
}

#[cfg(feature = "parallel")]
fn cell_ranges(offsets: &[usize], threads: usize) -> Vec<(usize, usize)> {
    let ncell = offsets.len() - 1;
    if ncell == 0 {
        return Vec::new();
    }
    let threads = threads.max(1).min(ncell);
    if threads == 1 {
        return vec![(0, ncell)];
    }
    let total = offsets[ncell];
    if total == 0 {
        return vec![(0, ncell)];
    }
    let mut ranges = Vec::with_capacity(threads);
    let mut start = 0usize;
    for t in 1..threads {
        let target = total * t / threads;
        let mut end = start;
        while end < ncell && offsets[end] < target {
            end += 1;
        }
        if end == start {
            end = (start + 1).min(ncell);
        }
        if end <= start {
            break;
        }
        ranges.push((start, end));
        start = end;
    }
    if start < ncell {
        ranges.push((start, ncell));
    }
    ranges
}

/// # Safety
/// `block` indexes `xs`, `ys`, `zs`, and `ids`. `scratch` holds every pair in the block.
#[allow(clippy::incompatible_msrv)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn avx512_scan(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ids: &[usize],
    cut2: f64,
    block: &Block,
    scratch: &mut Scratch,
) {
    use std::arch::x86_64::_mm512_mask_compressstoreu_epi64;
    use std::arch::x86_64::_mm512_mask_compressstoreu_pd;
    use std::arch::x86_64::{
        __m512i, _mm512_add_pd, _mm512_cmp_pd_mask, _mm512_loadu_pd, _mm512_loadu_si512,
        _mm512_mul_pd, _mm512_set1_pd, _mm512_sub_pd, _CMP_LT_OQ,
    };
    let xp = xs.as_ptr();
    let yp = ys.as_ptr();
    let zp = zs.as_ptr();
    let idp = ids.as_ptr();
    let atom_p = scratch.atom.as_mut_ptr();
    let js_p = scratch.js.as_mut_ptr();
    let d2_p = scratch.d2.as_mut_ptr();
    let mut n = scratch.js.len();
    let sx = block.shift[0];
    let sy = block.shift[1];
    let sz = block.shift[2];
    let skip_self = block.shift_s == [0, 0, 0];
    let cutv = _mm512_set1_pd(cut2);

    macro_rules! take8 {
        ($mask:expr, $d2v:expr, $jids:expr, $iu:expr) => {{
            let mask = $mask;
            if mask != 0 {
                let c = mask.count_ones() as usize;
                _mm512_mask_compressstoreu_pd(d2_p.add(n), mask, $d2v);
                _mm512_mask_compressstoreu_epi64(js_p.add(n) as *mut i64, mask, $jids);
                let iu = $iu;
                let end = n + c;
                let mut k = n;
                while k < end {
                    atom_p.add(k).write(iu);
                    k += 1;
                }
                n = end;
            }
        }};
    }
    macro_rules! dist8 {
        ($jx:expr, $jy:expr, $jz:expr, $bx:expr, $by:expr, $bz:expr) => {{
            let dx = _mm512_sub_pd($jx, $bx);
            let dy = _mm512_sub_pd($jy, $by);
            let dz = _mm512_sub_pd($jz, $bz);
            _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
                _mm512_mul_pd(dz, dz),
            )
        }};
    }
    macro_rules! keep_one {
        ($i:expr, $iu:expr, $px:expr, $py:expr, $pz:expr, $j:expr, $jx:expr, $jy:expr, $jz:expr) => {{
            let i = $i;
            let j = $j;
            if !(skip_self && j == i) {
                let dx = $jx - $px;
                let dy = $jy - $py;
                let dz = $jz - $pz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < cut2 {
                    atom_p.add(n).write($iu);
                    js_p.add(n).write(j as u64);
                    d2_p.add(n).write(d2);
                    n += 1;
                }
            }
        }};
    }
    macro_rules! one_src {
        ($s:expr, $j_lo:expr) => {{
            let s = $s;
            let j_lo = $j_lo;
            if j_lo < block.j_hi {
                let i = *idp.add(s);
                let iu = i as u32;
                let px = *xp.add(s) - sx;
                let py = *yp.add(s) - sy;
                let pz = *zp.add(s) - sz;
                let bx = _mm512_set1_pd(px);
                let by = _mm512_set1_pd(py);
                let bz = _mm512_set1_pd(pz);
                let span = block.j_hi - j_lo;
                let end = j_lo + (span & !7);
                let mut slot = j_lo;
                while slot < end {
                    let jx = _mm512_loadu_pd(xp.add(slot));
                    let jy = _mm512_loadu_pd(yp.add(slot));
                    let jz = _mm512_loadu_pd(zp.add(slot));
                    let jids = _mm512_loadu_si512(idp.add(slot) as *const __m512i);
                    let d2v = dist8!(jx, jy, jz, bx, by, bz);
                    let mask = _mm512_cmp_pd_mask(d2v, cutv, _CMP_LT_OQ);
                    take8!(mask, d2v, jids, iu);
                    slot += 8;
                }
                for slot in end..block.j_hi {
                    let j = *idp.add(slot);
                    let jx = *xp.add(slot);
                    let jy = *yp.add(slot);
                    let jz = *zp.add(slot);
                    keep_one!(i, iu, px, py, pz, j, jx, jy, jz);
                }
            }
        }};
    }

    if !block.tri {
        let j_lo = block.j_lo;
        let j_hi = block.j_hi;
        let end = j_lo + ((j_hi - j_lo) & !7);
        let mut s = block.i_lo;
        while s + 4 <= block.i_hi {
            let i0 = *idp.add(s);
            let i1 = *idp.add(s + 1);
            let i2 = *idp.add(s + 2);
            let i3 = *idp.add(s + 3);
            let iu0 = i0 as u32;
            let iu1 = i1 as u32;
            let iu2 = i2 as u32;
            let iu3 = i3 as u32;
            let p0x = *xp.add(s) - sx;
            let p0y = *yp.add(s) - sy;
            let p0z = *zp.add(s) - sz;
            let p1x = *xp.add(s + 1) - sx;
            let p1y = *yp.add(s + 1) - sy;
            let p1z = *zp.add(s + 1) - sz;
            let p2x = *xp.add(s + 2) - sx;
            let p2y = *yp.add(s + 2) - sy;
            let p2z = *zp.add(s + 2) - sz;
            let p3x = *xp.add(s + 3) - sx;
            let p3y = *yp.add(s + 3) - sy;
            let p3z = *zp.add(s + 3) - sz;
            let b0x = _mm512_set1_pd(p0x);
            let b0y = _mm512_set1_pd(p0y);
            let b0z = _mm512_set1_pd(p0z);
            let b1x = _mm512_set1_pd(p1x);
            let b1y = _mm512_set1_pd(p1y);
            let b1z = _mm512_set1_pd(p1z);
            let b2x = _mm512_set1_pd(p2x);
            let b2y = _mm512_set1_pd(p2y);
            let b2z = _mm512_set1_pd(p2z);
            let b3x = _mm512_set1_pd(p3x);
            let b3y = _mm512_set1_pd(p3y);
            let b3z = _mm512_set1_pd(p3z);
            let mut slot = j_lo;
            while slot < end {
                let jx = _mm512_loadu_pd(xp.add(slot));
                let jy = _mm512_loadu_pd(yp.add(slot));
                let jz = _mm512_loadu_pd(zp.add(slot));
                let jids = _mm512_loadu_si512(idp.add(slot) as *const __m512i);
                let d0 = dist8!(jx, jy, jz, b0x, b0y, b0z);
                let d1 = dist8!(jx, jy, jz, b1x, b1y, b1z);
                let d2v = dist8!(jx, jy, jz, b2x, b2y, b2z);
                let d3 = dist8!(jx, jy, jz, b3x, b3y, b3z);
                take8!(_mm512_cmp_pd_mask(d0, cutv, _CMP_LT_OQ), d0, jids, iu0);
                take8!(_mm512_cmp_pd_mask(d1, cutv, _CMP_LT_OQ), d1, jids, iu1);
                take8!(_mm512_cmp_pd_mask(d2v, cutv, _CMP_LT_OQ), d2v, jids, iu2);
                take8!(_mm512_cmp_pd_mask(d3, cutv, _CMP_LT_OQ), d3, jids, iu3);
                slot += 8;
            }
            for slot in end..j_hi {
                let j = *idp.add(slot);
                let jx = *xp.add(slot);
                let jy = *yp.add(slot);
                let jz = *zp.add(slot);
                keep_one!(i0, iu0, p0x, p0y, p0z, j, jx, jy, jz);
                keep_one!(i1, iu1, p1x, p1y, p1z, j, jx, jy, jz);
                keep_one!(i2, iu2, p2x, p2y, p2z, j, jx, jy, jz);
                keep_one!(i3, iu3, p3x, p3y, p3z, j, jx, jy, jz);
            }
            s += 4;
        }
        while s < block.i_hi {
            one_src!(s, j_lo);
            s += 1;
        }
    } else {
        for s in block.i_lo..block.i_hi {
            one_src!(s, s + 1);
        }
    }
    scratch.finish(n);
}

/// # Safety
/// `block` indexes `xs`, `ys`, `zs`, and `ids`. `scratch` holds every pair in the block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline(never)]
unsafe fn avx_scan(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ids: &[usize],
    cut2: f64,
    block: &Block,
    scratch: &mut Scratch,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_cmp_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_LT_OQ,
    };
    let xp = xs.as_ptr();
    let yp = ys.as_ptr();
    let zp = zs.as_ptr();
    let idp = ids.as_ptr();
    let atom_p = scratch.atom.as_mut_ptr();
    let js_p = scratch.js.as_mut_ptr();
    let d2_p = scratch.d2.as_mut_ptr();
    let mut n = scratch.js.len();
    let sx = block.shift[0];
    let sy = block.shift[1];
    let sz = block.shift[2];
    let skip_self = block.shift_s == [0, 0, 0];
    let cutv = _mm256_set1_pd(cut2);

    macro_rules! take4 {
        ($bits:expr, $d2v:expr, $slot:expr, $iu:expr, $iatom:expr) => {{
            if $bits != 0 {
                let mut lane = [0.0f64; 4];
                _mm256_storeu_pd(lane.as_mut_ptr(), $d2v);
                let iatom = $iatom;
                if $bits & 1 != 0 {
                    let j = *idp.add($slot);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(lane[0]);
                        n += 1;
                    }
                }
                if $bits & 2 != 0 {
                    let j = *idp.add($slot + 1);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(lane[1]);
                        n += 1;
                    }
                }
                if $bits & 4 != 0 {
                    let j = *idp.add($slot + 2);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(lane[2]);
                        n += 1;
                    }
                }
                if $bits & 8 != 0 {
                    let j = *idp.add($slot + 3);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(lane[3]);
                        n += 1;
                    }
                }
            }
        }};
    }
    macro_rules! dist4 {
        ($jx:expr, $jy:expr, $jz:expr, $bx:expr, $by:expr, $bz:expr) => {{
            let dx = _mm256_sub_pd($jx, $bx);
            let dy = _mm256_sub_pd($jy, $by);
            let dz = _mm256_sub_pd($jz, $bz);
            _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                _mm256_mul_pd(dz, dz),
            )
        }};
    }
    macro_rules! one_src {
        ($s:expr, $j_lo:expr) => {{
            let s = $s;
            let j_lo = $j_lo;
            if j_lo < block.j_hi {
                let i = *idp.add(s);
                let iu = i as u32;
                let px = *xp.add(s) - sx;
                let py = *yp.add(s) - sy;
                let pz = *zp.add(s) - sz;
                let bx = _mm256_set1_pd(px);
                let by = _mm256_set1_pd(py);
                let bz = _mm256_set1_pd(pz);
                let span = block.j_hi - j_lo;
                let end = j_lo + (span & !3);
                let mut slot = j_lo;
                while slot < end {
                    let jx = _mm256_loadu_pd(xp.add(slot));
                    let jy = _mm256_loadu_pd(yp.add(slot));
                    let jz = _mm256_loadu_pd(zp.add(slot));
                    let d2v = dist4!(jx, jy, jz, bx, by, bz);
                    let bits = _mm256_movemask_pd(_mm256_cmp_pd(d2v, cutv, _CMP_LT_OQ));
                    take4!(bits, d2v, slot, iu, i);
                    slot += 4;
                }
                for slot in end..block.j_hi {
                    let j = *idp.add(slot);
                    if skip_self && j == i {
                        continue;
                    }
                    let dx = *xp.add(slot) - px;
                    let dy = *yp.add(slot) - py;
                    let dz = *zp.add(slot) - pz;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < cut2 {
                        atom_p.add(n).write(iu);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(d2);
                        n += 1;
                    }
                }
            }
        }};
    }

    if !block.tri {
        let j_lo = block.j_lo;
        let j_hi = block.j_hi;
        let end = j_lo + ((j_hi - j_lo) & !3);
        let mut s = block.i_lo;
        while s + 2 <= block.i_hi {
            let i0 = *idp.add(s);
            let i1 = *idp.add(s + 1);
            let iu0 = i0 as u32;
            let iu1 = i1 as u32;
            let p0x = *xp.add(s) - sx;
            let p0y = *yp.add(s) - sy;
            let p0z = *zp.add(s) - sz;
            let p1x = *xp.add(s + 1) - sx;
            let p1y = *yp.add(s + 1) - sy;
            let p1z = *zp.add(s + 1) - sz;
            let b0x = _mm256_set1_pd(p0x);
            let b0y = _mm256_set1_pd(p0y);
            let b0z = _mm256_set1_pd(p0z);
            let b1x = _mm256_set1_pd(p1x);
            let b1y = _mm256_set1_pd(p1y);
            let b1z = _mm256_set1_pd(p1z);
            let mut slot = j_lo;
            while slot < end {
                let jx = _mm256_loadu_pd(xp.add(slot));
                let jy = _mm256_loadu_pd(yp.add(slot));
                let jz = _mm256_loadu_pd(zp.add(slot));
                let d0 = dist4!(jx, jy, jz, b0x, b0y, b0z);
                let d1 = dist4!(jx, jy, jz, b1x, b1y, b1z);
                take4!(
                    _mm256_movemask_pd(_mm256_cmp_pd(d0, cutv, _CMP_LT_OQ)),
                    d0,
                    slot,
                    iu0,
                    i0
                );
                take4!(
                    _mm256_movemask_pd(_mm256_cmp_pd(d1, cutv, _CMP_LT_OQ)),
                    d1,
                    slot,
                    iu1,
                    i1
                );
                slot += 4;
            }
            for slot in end..j_hi {
                let j = *idp.add(slot);
                let jx = *xp.add(slot);
                let jy = *yp.add(slot);
                let jz = *zp.add(slot);
                if !(skip_self && j == i0) {
                    let dx = jx - p0x;
                    let dy = jy - p0y;
                    let dz = jz - p0z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < cut2 {
                        atom_p.add(n).write(iu0);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(d2);
                        n += 1;
                    }
                }
                if !(skip_self && j == i1) {
                    let dx = jx - p1x;
                    let dy = jy - p1y;
                    let dz = jz - p1z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < cut2 {
                        atom_p.add(n).write(iu1);
                        js_p.add(n).write(j as u64);
                        d2_p.add(n).write(d2);
                        n += 1;
                    }
                }
            }
            s += 2;
        }
        if s < block.i_hi {
            one_src!(s, j_lo);
        }
    } else {
        for s in block.i_lo..block.i_hi {
            one_src!(s, s + 1);
        }
    }
    scratch.finish(n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cell;

    #[test]
    fn tight_cluster_matches_brute_force() {
        // Every pair sits inside the cutoff, and the box is large, so the
        // ideal-gas row estimate is a handful. The walk still has to
        // return the full list. 520 is above the parallel split.
        let n = 520usize;
        let mut xyz = vec![[0.0f64; 3]; n];
        let g = 9i32;
        let mut k = 0usize;
        'fill: for z in 0..g {
            for y in 0..g {
                for x in 0..g {
                    if k >= n {
                        break 'fill;
                    }
                    xyz[k] = [x as f64 * 0.05, y as f64 * 0.05, z as f64 * 0.05];
                    k += 1;
                }
            }
        }
        let sim = Cell::ortho(80.0, 80.0, 80.0).unwrap();
        let cutoff = 2.0;
        let cut2 = cutoff * cutoff;
        let mut brute: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = xyz[j][0] - xyz[i][0];
                let dy = xyz[j][1] - xyz[i][1];
                let dz = xyz[j][2] - xyz[i][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < cut2 {
                    brute.push((i, j, d2));
                    brute.push((j, i, d2));
                }
            }
        }
        brute.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let run = || pairs_within(&xyz, &sim, cutoff, None, None, false).unwrap();
        #[cfg(feature = "parallel")]
        let got = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(run)
        };
        #[cfg(not(feature = "parallel"))]
        let got = run();
        let mut rows: Vec<(usize, usize, f64)> = got.iter().map(|p| (p.i, p.j, p.dist2)).collect();
        rows.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        assert_eq!(rows.len(), n * (n - 1));
        assert_eq!(rows.len(), brute.len());
        for (got_row, brute_row) in rows.iter().zip(brute.iter()) {
            assert_eq!(got_row.0, brute_row.0);
            assert_eq!(got_row.1, brute_row.1);
            assert!((got_row.2 - brute_row.2).abs() < 1e-9);
        }
    }

    #[test]
    fn far_image_is_not_rewrapped() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 3.0, None, Some(10.0), false).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|p| p.shift == [0, 0, 0]));
        assert!(pairs.iter().all(|p| (p.dist2 - 0.01).abs() < 1e-12));
    }

    #[test]
    fn highway_mic_matches_wrap_pair_distance() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 1.0, None, None, true).unwrap();
        let mut mic = [0.0];
        crate::dist2_ortho_diffs(
            &[xyz[1][0] - xyz[0][0]],
            &[xyz[1][1] - xyz[0][1]],
            &[xyz[1][2] - xyz[0][2]],
            10.0,
            10.0,
            10.0,
            &mut mic,
        )
        .unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].shift, [-1, 0, 0]);
        assert!((pairs[0].dist2 - mic[0]).abs() < 1e-12);
        assert!((mic[0] - 0.64).abs() < 1e-12);
    }

    #[test]
    fn wrap_pair_keeps_shift() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 1.0, None, None, false).unwrap();
        assert_eq!(pairs.len(), 2);
        let a = pairs.iter().find(|p| p.i == 0 && p.j == 1).unwrap();
        assert_eq!(a.shift, [-1, 0, 0]);
        assert!((a.dist2 - 0.64).abs() < 1e-12);
    }

    #[test]
    fn half_list_is_canonical() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 1.0, None, None, true).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].i, 0);
        assert_eq!(pairs[0].j, 1);
    }

    #[test]
    fn sheared_cutoff_matches_shift_scan() {
        let sim = Cell::from_vectors(
            [4.0, 0.0, 0.0],
            [1.5, 3.0, 0.0],
            [0.2, -0.4, 3.5],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        let xyz = [
            sim.cartesian([0.05, 0.10, 0.20]),
            sim.cartesian([0.90, 0.15, 0.80]),
            sim.cartesian([0.40, 0.70, 0.30]),
            sim.cartesian([0.12, 0.85, 0.05]),
        ];
        let cutoff = 2.5;
        let got = pairs_within(&xyz, &sim, cutoff, None, Some(1.0), false).unwrap();
        let mut want = Vec::new();
        let cut2 = cutoff * cutoff;
        for i in 0..xyz.len() {
            for j in 0..xyz.len() {
                for na in -2..=2 {
                    for nb in -2..=2 {
                        for nc in -2..=2 {
                            if i == j && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let shift = sim.lattice_shift(na, nb, nc);
                            let pi = sim.cartesian(sim.fractional(xyz[i]));
                            let pj = sim.cartesian(sim.fractional(xyz[j]));
                            let d2 = sim.dist2_shifted(pi, pj, shift);
                            if d2 < cut2 {
                                want.push((i, j, [na, nb, nc]));
                            }
                        }
                    }
                }
            }
        }
        let mut got_key: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
        got_key.sort();
        want.sort();
        assert_eq!(got_key, want);
    }

    #[test]
    fn wide_cube_matches_shift_scan() {
        let sim = Cell::ortho(18.0, 18.0, 18.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..8 {
            for iy in 0..8 {
                for ix in 0..8 {
                    xyz.push([
                        (ix as f64 + 0.5) * 18.0 / 8.0,
                        (iy as f64 + 0.5) * 18.0 / 8.0,
                        (iz as f64 + 0.5) * 18.0 / 8.0,
                    ]);
                }
            }
        }
        let cutoff = 4.0;
        let cut2 = cutoff * cutoff;
        for half in [false, true] {
            let got = pairs_within(&xyz, &sim, cutoff, None, None, half).unwrap();
            let mut want = Vec::new();
            for i in 0..xyz.len() {
                for j in 0..xyz.len() {
                    for na in -1..=1 {
                        for nb in -1..=1 {
                            for nc in -1..=1 {
                                if i == j && na == 0 && nb == 0 && nc == 0 {
                                    continue;
                                }
                                let shift = [na, nb, nc];
                                let d2 = sim.dist2_shifted(
                                    xyz[i],
                                    xyz[j],
                                    sim.lattice_shift(na, nb, nc),
                                );
                                if d2 < cut2 && (!half || keep_half(i, j, shift)) {
                                    want.push((i, j, shift));
                                }
                            }
                        }
                    }
                }
            }
            let mut got_key: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
            got_key.sort();
            want.sort();
            assert_eq!(got_key, want);
        }
    }

    #[test]
    fn dense_cube_matches_shift_scan() {
        let sim = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    xyz.push([
                        (ix as f64 + 0.5) * 1.2,
                        (iy as f64 + 0.5) * 1.2,
                        (iz as f64 + 0.5) * 1.2,
                    ]);
                }
            }
        }
        let cutoff = 2.0;
        let got = pairs_within(&xyz, &sim, cutoff, None, None, false).unwrap();
        let mut keys: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
        keys.sort();
        let cut2 = cutoff * cutoff;
        let mut want = Vec::new();
        for i in 0..xyz.len() {
            for j in 0..xyz.len() {
                for na in -1..=1 {
                    for nb in -1..=1 {
                        for nc in -1..=1 {
                            if i == j && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let shift = [na, nb, nc];
                            let pi = sim.cartesian(sim.fractional(xyz[i]));
                            let pj = sim.cartesian(sim.fractional(xyz[j]));
                            let d2 = sim.dist2_shifted(pi, pj, sim.lattice_shift(na, nb, nc));
                            if d2 < cut2 {
                                want.push((i, j, shift));
                            }
                        }
                    }
                }
            }
        }
        want.sort();
        assert_eq!(keys, want);
    }

    #[test]
    fn bad_cutoff() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.0, 0.0, 0.0]];
        assert_eq!(
            pairs_within(&xyz, &sim, 0.0, None, None, false).unwrap_err(),
            Error::BadCutoff
        );
    }
}
