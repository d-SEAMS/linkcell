//! SoA candidate filter for a Verlet skin.
//!
//! Candidates are grouped by source atom, the same blocking vesin uses
//! for a Verlet list. Squared distances are written once, in candidate
//! order. AVX-512 gathers the eight target coordinates of a chunk and
//! adds `H S` in registers. Threads then compact disjoint atom ranges
//! into the output columns.

#![allow(
    clippy::incompatible_msrv,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use crate::cell::Cell;
use crate::pairs::PairColumns;
use crate::simd::{self, Path};
use crate::Error;

/// Candidates of one frame, grouped by source atom.
pub(crate) struct Soa {
    /// `offsets[i]..offsets[i + 1]` are the candidates whose source is `i`.
    offsets: Vec<u32>,
    j: Vec<u32>,
    sx: Vec<i32>,
    sy: Vec<i32>,
    sz: Vec<i32>,
    /// Packed shift in `[-3, 3]^3` when every candidate fits.
    /// Empty when [`Self::coded`] is false and the full shift columns are used.
    code: Vec<u8>,
    coded: bool,
}

/// Integer code of shift `(0, 0, 0)` in [`ISHIFT`].
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
const ZERO_CODE: u8 = (3 * 49 + 3 * 7 + 3) as u8;

/// Eight zero-shift codes, for the aligned fast path.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
const ZERO8: u64 = {
    let b = ZERO_CODE as u64;
    b | (b << 8) | (b << 16) | (b << 24) | (b << 32) | (b << 40) | (b << 48) | (b << 56)
};

const fn ishift_table() -> [[i32; 3]; 343] {
    let mut t = [[0; 3]; 343];
    let mut s = 0;
    while s < 343 {
        let sx = (s / 49) as i32 - 3;
        let sy = ((s % 49) / 7) as i32 - 3;
        let sz = (s % 7) as i32 - 3;
        t[s] = [sx, sy, sz];
        s += 1;
    }
    t
}

/// `code -> (sx, sy, sz)` for shifts inside `[-3, 3]`.
const ISHIFT: [[i32; 3]; 343] = ishift_table();

fn encode_shift(sx: i32, sy: i32, sz: i32) -> Option<u8> {
    if !(-3..=3).contains(&sx) || !(-3..=3).contains(&sy) || !(-3..=3).contains(&sz) {
        return None;
    }
    Some(((sx + 3) as u8) * 49 + ((sy + 3) as u8) * 7 + (sz + 3) as u8)
}

impl Soa {
    pub(crate) fn empty() -> Self {
        Self {
            offsets: vec![0],
            j: Vec::new(),
            sx: Vec::new(),
            sy: Vec::new(),
            sz: Vec::new(),
            code: Vec::new(),
            coded: true,
        }
    }

    /// Group `cols` by source atom. Order inside an atom is the order
    /// the columns arrived in.
    pub(crate) fn from_columns(cols: &PairColumns, n_atoms: usize) -> Result<Self, Error> {
        let n = cols.len();
        if cols.j.len() != n || cols.shift.len() != n {
            return Err(Error::BufferSize);
        }
        let mut counts = vec![0u32; n_atoms];
        for &i in &cols.i {
            let i = i as usize;
            if i >= n_atoms {
                return Err(Error::Overflow);
            }
            counts[i] = counts[i].checked_add(1).ok_or(Error::Overflow)?;
        }
        let mut offsets = vec![0u32; n_atoms + 1];
        for i in 0..n_atoms {
            offsets[i + 1] = offsets[i].checked_add(counts[i]).ok_or(Error::Overflow)?;
        }
        let mut cursor = offsets.clone();
        let mut j = vec![0u32; n];
        let mut sx = vec![0i32; n];
        let mut sy = vec![0i32; n];
        let mut sz = vec![0i32; n];
        let mut code = vec![0u8; n];
        let mut coded = true;
        for k in 0..n {
            let i = cols.i[k] as usize;
            let at = cursor[i] as usize;
            cursor[i] += 1;
            j[at] = cols.j[k] as u32;
            let s = cols.shift[k];
            sx[at] = s[0];
            sy[at] = s[1];
            sz[at] = s[2];
            if coded {
                if let Some(c) = encode_shift(s[0], s[1], s[2]) {
                    code[at] = c;
                } else {
                    coded = false;
                }
            }
        }
        if !coded {
            code.clear();
        }
        Ok(Self {
            offsets,
            j,
            sx,
            sy,
            sz,
            code,
            coded,
        })
    }

    /// `(source, target)` of every candidate, source-major.
    pub(crate) fn each_end(&self, mut f: impl FnMut(u32, u32)) {
        let atoms = self.offsets.len().saturating_sub(1);
        for i in 0..atoms {
            let a = self.offsets[i] as usize;
            let b = self.offsets[i + 1] as usize;
            for k in a..b {
                f(i as u32, self.j[k]);
            }
        }
    }

    /// Keep rows with squared distance strictly below `cut2`.
    pub(crate) fn filter(
        &self,
        xyz: &[[f64; 3]],
        cell: &Cell,
        cut2: f64,
        want_vector: bool,
        d2: &mut Vec<f64>,
        out: &mut PairColumns,
        vectors: &mut Vec<[f64; 3]>,
    ) -> Result<(), Error> {
        let atoms = self.offsets.len().saturating_sub(1);
        if xyz.len() != atoms {
            return Err(Error::BufferSize);
        }
        let n = self.j.len();
        let threads = filter_threads(n);
        let ranges = atom_ranges(&self.offsets, threads);
        let use_avx = simd::select() == Path::Avx512;
        let ortho = cell.is_ortho();
        let h = cell.h();
        let widths = cell.widths();
        if self.coded && !want_vector {
            vectors.clear();
            return filter_coded(
                xyz,
                &self.offsets,
                &self.j,
                &self.code,
                &ranges,
                cut2,
                ortho,
                widths,
                h,
                use_avx,
                d2,
                out,
            );
        }
        if d2.len() < n {
            d2.resize(n, 0.0);
        }
        let counts = count_ranges(
            xyz,
            &self.offsets,
            &self.j,
            &self.sx,
            &self.sy,
            &self.sz,
            &ranges,
            cut2,
            ortho,
            widths,
            h,
            use_avx,
            d2,
        );
        let mut off = Vec::with_capacity(ranges.len() + 1);
        off.push(0usize);
        for c in &counts {
            off.push(off[off.len() - 1] + c);
        }
        let total = *off.last().unwrap_or(&0);
        prepare(out, vectors, total, want_vector);
        write_ranges(
            xyz,
            self,
            &ranges,
            &off,
            d2,
            cut2,
            want_vector,
            ortho,
            widths,
            h,
            use_avx,
            out,
            vectors,
        );
        Ok(())
    }

    pub(crate) fn is_coded(&self) -> bool {
        self.coded
    }

    /// Filter every bead from this candidate list in one pass.
    ///
    /// `xyz` is bead-major, `beads * atoms` points. A chunk of candidates
    /// is loaded once; each bead then gathers its own coordinates into the
    /// same lanes and keeps the rows under `cut2`.
    pub(crate) fn filter_beads(
        &self,
        xyz: &[[f64; 3]],
        beads: usize,
        cell: &Cell,
        cut2: f64,
        outs: &mut [PairColumns],
    ) -> Result<(), Error> {
        let atoms = self.offsets.len().saturating_sub(1);
        if beads == 0 || outs.len() != beads || xyz.len() != beads * atoms || !self.coded {
            return Err(Error::BufferSize);
        }
        let threads = filter_threads(self.j.len().saturating_mul(beads.max(1)));
        let ranges = atom_ranges(&self.offsets, threads);
        let use_avx = simd::select() == Path::Avx512;
        let (tx, ty, tz) = cartesian_tables(cell.is_ortho(), cell.widths(), cell.h());
        let counts = bead_pass::<false>(
            xyz,
            atoms,
            beads,
            &self.offsets,
            &self.j,
            &self.code,
            &ranges,
            &[],
            cut2,
            &tx,
            &ty,
            &tz,
            use_avx,
            &[],
        );
        let mut cursor = vec![0usize; beads];
        let mut off = vec![vec![0usize; ranges.len()]; beads];
        for (t, row) in counts.iter().enumerate() {
            for b in 0..beads {
                off[b][t] = cursor[b];
                cursor[b] += row[b];
            }
        }
        let mut vectors = Vec::new();
        let mut ptrs = Vec::with_capacity(beads);
        for b in 0..beads {
            prepare(&mut outs[b], &mut vectors, cursor[b], false);
            ptrs.push((
                outs[b].i.as_mut_ptr() as usize,
                outs[b].j.as_mut_ptr() as usize,
                outs[b].shift.as_mut_ptr() as usize,
                outs[b].dist2.as_mut_ptr() as usize,
            ));
        }
        let written = bead_pass::<true>(
            xyz,
            atoms,
            beads,
            &self.offsets,
            &self.j,
            &self.code,
            &ranges,
            &off,
            cut2,
            &tx,
            &ty,
            &tz,
            use_avx,
            &ptrs,
        );
        debug_assert_eq!(written, counts);
        Ok(())
    }
}

fn cartesian_tables(
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
) -> ([f64; 343], [f64; 343], [f64; 343]) {
    let mut tx = [0.0; 343];
    let mut ty = [0.0; 343];
    let mut tz = [0.0; 343];
    for (s, shift) in ISHIFT.iter().enumerate() {
        let (x, y, z) = shift_xyz(shift[0], shift[1], shift[2], ortho, widths, h);
        tx[s] = x;
        ty[s] = y;
        tz[s] = z;
    }
    (tx, ty, tz)
}

fn filter_coded(
    xyz: &[[f64; 3]],
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    ranges: &[(usize, usize)],
    cut2: f64,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    use_avx: bool,
    _d2: &mut Vec<f64>,
    out: &mut PairColumns,
) -> Result<(), Error> {
    let (tx, ty, tz) = cartesian_tables(ortho, widths, h);
    let counts = coded_pass::<false>(
        xyz,
        offsets,
        j,
        code,
        ranges,
        &[],
        cut2,
        &tx,
        &ty,
        &tz,
        use_avx,
        0,
        0,
        0,
        0,
        0,
    );
    let mut off = Vec::with_capacity(ranges.len() + 1);
    off.push(0usize);
    for c in &counts {
        off.push(off[off.len() - 1] + c);
    }
    let total = *off.last().unwrap_or(&0);
    let mut vectors = Vec::new();
    prepare(out, &mut vectors, total, false);
    let written = coded_pass::<true>(
        xyz,
        offsets,
        j,
        code,
        ranges,
        &off,
        cut2,
        &tx,
        &ty,
        &tz,
        use_avx,
        out.i.as_mut_ptr() as usize,
        out.j.as_mut_ptr() as usize,
        out.shift.as_mut_ptr() as usize,
        out.dist2.as_mut_ptr() as usize,
        0,
    );
    debug_assert_eq!(written, counts);
    Ok(())
}

fn coded_pass<const WRITE: bool>(
    xyz: &[[f64; 3]],
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    ranges: &[(usize, usize)],
    off: &[usize],
    cut2: f64,
    tx: &[f64; 343],
    ty: &[f64; 343],
    tz: &[f64; 343],
    use_avx: bool,
    ip: usize,
    jp: usize,
    sp: usize,
    dp: usize,
    cand: usize,
) -> Vec<usize> {
    let run = |t: usize, lo: usize, hi: usize| -> usize {
        let at = if WRITE { off[t] } else { 0 };
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        if use_avx {
            return unsafe {
                coded_avx::<WRITE>(
                    xyz, offsets, j, code, lo, hi, cut2, tx, ty, tz, at, ip, jp, sp, dp, cand,
                )
            };
        }
        let _ = use_avx;
        coded_scalar::<WRITE>(
            xyz, offsets, j, code, lo, hi, cut2, tx, ty, tz, at, ip, jp, sp, dp, cand,
        )
    };
    #[cfg(feature = "parallel")]
    if ranges.len() > 1 {
        use rayon::prelude::*;
        return ranges
            .par_iter()
            .enumerate()
            .map(|(t, &(lo, hi))| run(t, lo, hi))
            .collect();
    }
    ranges
        .iter()
        .enumerate()
        .map(|(t, &(lo, hi))| run(t, lo, hi))
        .collect()
}

fn coded_scalar<const WRITE: bool>(
    xyz: &[[f64; 3]],
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    lo: usize,
    hi: usize,
    cut2: f64,
    tx: &[f64; 343],
    ty: &[f64; 343],
    tz: &[f64; 343],
    mut at: usize,
    ip: usize,
    jp: usize,
    sp: usize,
    dp: usize,
    cand: usize,
) -> usize {
    let mut hits = 0usize;
    for atom in lo..hi {
        let p = xyz[atom];
        let a = offsets[atom] as usize;
        let b = offsets[atom + 1] as usize;
        for k in a..b {
            let q = xyz[j[k] as usize];
            let c = code[k] as usize;
            let dx = q[0] + tx[c] - p[0];
            let dy = q[1] + ty[c] - p[1];
            let dz = q[2] + tz[c] - p[2];
            let dist = (dx * dx + dy * dy) + dz * dz;
            let _ = cand;
            if dist < cut2 {
                if WRITE {
                    unsafe {
                        *(ip as *mut i32).add(at) = atom as i32;
                        *(jp as *mut i32).add(at) = j[k] as i32;
                        *(sp as *mut [i32; 3]).add(at) = ISHIFT[c];
                        *(dp as *mut f64).add(at) = dist;
                    }
                }
                at += 1;
                hits += 1;
            }
        }
    }
    hits
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f,avx512vl,avx2")]
unsafe fn coded_avx<const WRITE: bool>(
    xyz: &[[f64; 3]],
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    lo: usize,
    hi: usize,
    cut2: f64,
    tx: &[f64; 343],
    ty: &[f64; 343],
    tz: &[f64; 343],
    mut at: usize,
    ip: usize,
    jp: usize,
    sp: usize,
    dp: usize,
    cand: usize,
) -> usize {
    use std::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm256_add_epi32, _mm256_cvtepu8_epi32, _mm256_loadu_si256,
        _mm256_mask_compressstoreu_epi32, _mm256_maskz_compress_epi32, _mm256_mullo_epi32,
        _mm256_set1_epi32, _mm256_setzero_si256, _mm256_storeu_si256, _mm512_add_pd,
        _mm512_cmp_pd_mask, _mm512_i32gather_pd, _mm512_mask_compressstoreu_pd, _mm512_mul_pd,
        _mm512_set1_pd, _mm512_setzero_si512, _mm512_storeu_pd, _mm512_storeu_si512, _mm512_sub_pd,
        _mm_loadl_epi64, _mm_prefetch, _CMP_LT_OQ, _MM_HINT_T0,
    };
    let mut hits = 0usize;
    let cut = _mm512_set1_pd(cut2);
    let bytes = xyz.as_ptr() as *const f64;
    let txp = tx.as_ptr();
    let typ = ty.as_ptr();
    let tzp = tz.as_ptr();
    let three = _mm256_set1_epi32(3);
    let one = _mm256_set1_epi32(1);
    let two = _mm256_set1_epi32(2);
    for atom in lo..hi {
        let p = xyz[atom];
        let px = _mm512_set1_pd(p[0]);
        let py = _mm512_set1_pd(p[1]);
        let pz = _mm512_set1_pd(p[2]);
        let a = offsets[atom] as usize;
        let b = offsets[atom + 1] as usize;
        let mut t = a;
        while t + 8 <= b {
            if t + 16 <= b {
                for s in 0..8 {
                    let jj = unsafe { *j.as_ptr().add(t + 8 + s) } as usize;
                    unsafe {
                        _mm_prefetch(bytes.add(jj * 3) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(bytes.add(jj * 3 + 2) as *const i8, _MM_HINT_T0);
                    }
                }
            }
            let raw = unsafe { (code.as_ptr().add(t) as *const u64).read_unaligned() };
            let zero = raw == ZERO8;
            let jv = unsafe { _mm256_loadu_si256(j.as_ptr().add(t) as *const __m256i) };
            let ix = _mm256_mullo_epi32(jv, three);
            let qx = unsafe { _mm512_i32gather_pd(ix, bytes, 8) };
            let qy = unsafe { _mm512_i32gather_pd(_mm256_add_epi32(ix, one), bytes, 8) };
            let qz = unsafe { _mm512_i32gather_pd(_mm256_add_epi32(ix, two), bytes, 8) };
            let (dx, dy, dz, c32) = if zero {
                (
                    _mm512_sub_pd(qx, px),
                    _mm512_sub_pd(qy, py),
                    _mm512_sub_pd(qz, pz),
                    _mm256_set1_epi32(i32::from(ZERO_CODE)),
                )
            } else {
                let c32 = _mm256_cvtepu8_epi32(unsafe {
                    _mm_loadl_epi64(code.as_ptr().add(t) as *const __m128i)
                });
                let sx = unsafe { _mm512_i32gather_pd(c32, txp, 8) };
                let sy = unsafe { _mm512_i32gather_pd(c32, typ, 8) };
                let sz = unsafe { _mm512_i32gather_pd(c32, tzp, 8) };
                (
                    _mm512_sub_pd(_mm512_add_pd(qx, sx), px),
                    _mm512_sub_pd(_mm512_add_pd(qy, sy), py),
                    _mm512_sub_pd(_mm512_add_pd(qz, sz), pz),
                    c32,
                )
            };
            let dist = _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
                _mm512_mul_pd(dz, dz),
            );
            let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(dist, cut);
            let n = mask.count_ones() as usize;
            hits += n;
            let _ = cand;
            if WRITE && n > 0 {
                unsafe {
                    let iptr = ip as *mut i32;
                    let jptr = jp as *mut i32;
                    let sptr = sp as *mut [i32; 3];
                    let dptr = dp as *mut f64;
                    let iv = _mm256_set1_epi32(atom as i32);
                    if n == 8 && zero {
                        _mm256_storeu_si256(iptr.add(at) as *mut __m256i, iv);
                        _mm256_storeu_si256(jptr.add(at) as *mut __m256i, jv);
                        _mm512_storeu_pd(dptr.add(at), dist);
                        let sbase = (sptr as *mut i32).add(at * 3);
                        _mm512_storeu_si512(sbase as *mut __m512i, _mm512_setzero_si512());
                        _mm256_storeu_si256(sbase.add(16) as *mut __m256i, _mm256_setzero_si256());
                    } else {
                        _mm256_mask_compressstoreu_epi32(iptr.add(at), mask, iv);
                        _mm256_mask_compressstoreu_epi32(jptr.add(at), mask, jv);
                        _mm512_mask_compressstoreu_pd(dptr.add(at), mask, dist);
                        if zero {
                            for hhit in 0..n {
                                *sptr.add(at + hhit) = [0, 0, 0];
                            }
                        } else {
                            let packed = _mm256_maskz_compress_epi32(mask, c32);
                            let mut cs = [0i32; 8];
                            _mm256_storeu_si256(cs.as_mut_ptr() as *mut __m256i, packed);
                            for hhit in 0..n {
                                *sptr.add(at + hhit) = ISHIFT[cs[hhit] as usize];
                            }
                        }
                    }
                }
                at += n;
            }
            t += 8;
        }
        let p = xyz[atom];
        for k in t..b {
            let q = xyz[j[k] as usize];
            let c = code[k] as usize;
            let dx = q[0] + tx[c] - p[0];
            let dy = q[1] + ty[c] - p[1];
            let dz = q[2] + tz[c] - p[2];
            let dist = (dx * dx + dy * dy) + dz * dz;
            let _ = cand;
            if dist < cut2 {
                if WRITE {
                    unsafe {
                        *(ip as *mut i32).add(at) = atom as i32;
                        *(jp as *mut i32).add(at) = j[k] as i32;
                        *(sp as *mut [i32; 3]).add(at) = ISHIFT[c];
                        *(dp as *mut f64).add(at) = dist;
                    }
                }
                at += 1;
                hits += 1;
            }
        }
    }
    hits
}

fn bead_pass<const WRITE: bool>(
    xyz: &[[f64; 3]],
    atoms: usize,
    beads: usize,
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    ranges: &[(usize, usize)],
    off: &[Vec<usize>],
    cut2: f64,
    tx: &[f64; 343],
    ty: &[f64; 343],
    tz: &[f64; 343],
    use_avx: bool,
    ptrs: &[(usize, usize, usize, usize)],
) -> Vec<Vec<usize>> {
    let run = |t: usize, lo: usize, hi: usize| -> Vec<usize> {
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        if use_avx {
            return unsafe {
                bead_avx::<WRITE>(
                    xyz, atoms, beads, offsets, j, code, lo, hi, t, off, cut2, tx, ty, tz, ptrs,
                )
            };
        }
        let _ = use_avx;
        bead_scalar::<WRITE>(
            xyz, atoms, beads, offsets, j, code, lo, hi, t, off, cut2, tx, ty, tz, ptrs,
        )
    };
    #[cfg(feature = "parallel")]
    if ranges.len() > 1 {
        use rayon::prelude::*;
        return ranges
            .par_iter()
            .enumerate()
            .map(|(t, &(lo, hi))| run(t, lo, hi))
            .collect();
    }
    ranges
        .iter()
        .enumerate()
        .map(|(t, &(lo, hi))| run(t, lo, hi))
        .collect()
}

fn bead_scalar<const WRITE: bool>(
    xyz: &[[f64; 3]],
    atoms: usize,
    beads: usize,
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    lo: usize,
    hi: usize,
    t: usize,
    off: &[Vec<usize>],
    cut2: f64,
    tx: &[f64; 343],
    ty: &[f64; 343],
    tz: &[f64; 343],
    ptrs: &[(usize, usize, usize, usize)],
) -> Vec<usize> {
    let mut hits = vec![0usize; beads];
    let mut at: Vec<usize> = (0..beads)
        .map(|b| if WRITE { off[b][t] } else { 0 })
        .collect();
    for atom in lo..hi {
        let a = offsets[atom] as usize;
        let b = offsets[atom + 1] as usize;
        for k in a..b {
            let c = code[k] as usize;
            let target = j[k] as usize;
            let (sx, sy, sz) = (tx[c], ty[c], tz[c]);
            let shift = ISHIFT[c];
            for bead in 0..beads {
                let p = xyz[bead * atoms + atom];
                let q = xyz[bead * atoms + target];
                let dx = q[0] + sx - p[0];
                let dy = q[1] + sy - p[1];
                let dz = q[2] + sz - p[2];
                let dist = (dx * dx + dy * dy) + dz * dz;
                if dist < cut2 {
                    if WRITE {
                        let (ip, jp, sp, dp) = ptrs[bead];
                        let atb = at[bead];
                        unsafe {
                            *(ip as *mut i32).add(atb) = atom as i32;
                            *(jp as *mut i32).add(atb) = target as i32;
                            *(sp as *mut [i32; 3]).add(atb) = shift;
                            *(dp as *mut f64).add(atb) = dist;
                        }
                    }
                    at[bead] += 1;
                    hits[bead] += 1;
                }
            }
        }
    }
    hits
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f,avx512vl,avx2")]
unsafe fn bead_avx<const WRITE: bool>(
    xyz: &[[f64; 3]],
    atoms: usize,
    beads: usize,
    offsets: &[u32],
    j: &[u32],
    code: &[u8],
    lo: usize,
    hi: usize,
    t: usize,
    off: &[Vec<usize>],
    cut2: f64,
    tx: &[f64; 343],
    ty: &[f64; 343],
    tz: &[f64; 343],
    ptrs: &[(usize, usize, usize, usize)],
) -> Vec<usize> {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_cvtepu8_epi32, _mm256_loadu_si256,
        _mm256_mask_compressstoreu_epi32, _mm256_maskz_compress_epi32, _mm256_mullo_epi32,
        _mm256_set1_epi32, _mm256_storeu_si256, _mm512_add_pd, _mm512_cmp_pd_mask,
        _mm512_i32gather_pd, _mm512_mask_compressstoreu_pd, _mm512_mul_pd, _mm512_set1_pd,
        _mm512_sub_pd, _mm_loadl_epi64, _CMP_LT_OQ,
    };
    let mut hits = vec![0usize; beads];
    let mut at: Vec<usize> = (0..beads)
        .map(|b| if WRITE { off[b][t] } else { 0 })
        .collect();
    let cut = _mm512_set1_pd(cut2);
    let txp = tx.as_ptr();
    let typ = ty.as_ptr();
    let tzp = tz.as_ptr();
    let three = _mm256_set1_epi32(3);
    let one = _mm256_set1_epi32(1);
    let two = _mm256_set1_epi32(2);
    for atom in lo..hi {
        let a = offsets[atom] as usize;
        let b = offsets[atom + 1] as usize;
        let mut k = a;
        while k + 8 <= b {
            let raw = unsafe { (code.as_ptr().add(k) as *const u64).read_unaligned() };
            let zero = raw == ZERO8;
            let jv = unsafe { _mm256_loadu_si256(j.as_ptr().add(k) as *const __m256i) };
            let ix = _mm256_mullo_epi32(jv, three);
            let (sxv, syv, szv, c32) = if zero {
                (
                    _mm512_set1_pd(0.0),
                    _mm512_set1_pd(0.0),
                    _mm512_set1_pd(0.0),
                    _mm256_set1_epi32(i32::from(ZERO_CODE)),
                )
            } else {
                let c32 = _mm256_cvtepu8_epi32(unsafe {
                    _mm_loadl_epi64(code.as_ptr().add(k) as *const __m128i)
                });
                (
                    unsafe { _mm512_i32gather_pd(c32, txp, 8) },
                    unsafe { _mm512_i32gather_pd(c32, typ, 8) },
                    unsafe { _mm512_i32gather_pd(c32, tzp, 8) },
                    c32,
                )
            };
            for bead in 0..beads {
                let base = unsafe { xyz.as_ptr().add(bead * atoms) } as *const f64;
                let p = xyz[bead * atoms + atom];
                let px = _mm512_set1_pd(p[0]);
                let py = _mm512_set1_pd(p[1]);
                let pz = _mm512_set1_pd(p[2]);
                let qx = unsafe { _mm512_i32gather_pd(ix, base, 8) };
                let qy = unsafe { _mm512_i32gather_pd(_mm256_add_epi32(ix, one), base, 8) };
                let qz = unsafe { _mm512_i32gather_pd(_mm256_add_epi32(ix, two), base, 8) };
                let dx = _mm512_sub_pd(_mm512_add_pd(qx, sxv), px);
                let dy = _mm512_sub_pd(_mm512_add_pd(qy, syv), py);
                let dz = _mm512_sub_pd(_mm512_add_pd(qz, szv), pz);
                let dist = _mm512_add_pd(
                    _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
                    _mm512_mul_pd(dz, dz),
                );
                let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(dist, cut);
                let n = mask.count_ones() as usize;
                hits[bead] += n;
                if WRITE && n > 0 {
                    let (ip, jp, sp, dp) = ptrs[bead];
                    let atb = at[bead];
                    unsafe {
                        let iv = _mm256_set1_epi32(atom as i32);
                        _mm256_mask_compressstoreu_epi32((ip as *mut i32).add(atb), mask, iv);
                        _mm256_mask_compressstoreu_epi32((jp as *mut i32).add(atb), mask, jv);
                        _mm512_mask_compressstoreu_pd((dp as *mut f64).add(atb), mask, dist);
                        if zero {
                            let sptr = (sp as *mut [i32; 3]).add(atb);
                            for hhit in 0..n {
                                *sptr.add(hhit) = [0, 0, 0];
                            }
                        } else {
                            let packed = _mm256_maskz_compress_epi32(mask, c32);
                            let mut cs = [0i32; 8];
                            _mm256_storeu_si256(cs.as_mut_ptr() as *mut __m256i, packed);
                            let sptr = (sp as *mut [i32; 3]).add(atb);
                            for hhit in 0..n {
                                *sptr.add(hhit) = ISHIFT[cs[hhit] as usize];
                            }
                        }
                    }
                    at[bead] += n;
                }
            }
            k += 8;
        }
        for idx in k..b {
            let c = code[idx] as usize;
            let target = j[idx] as usize;
            let (sx, sy, sz) = (tx[c], ty[c], tz[c]);
            let shift = ISHIFT[c];
            for bead in 0..beads {
                let p = xyz[bead * atoms + atom];
                let q = xyz[bead * atoms + target];
                let dx = q[0] + sx - p[0];
                let dy = q[1] + sy - p[1];
                let dz = q[2] + sz - p[2];
                let dist = (dx * dx + dy * dy) + dz * dz;
                if dist < cut2 {
                    if WRITE {
                        let (ip, jp, sp, dp) = ptrs[bead];
                        let atb = at[bead];
                        unsafe {
                            *(ip as *mut i32).add(atb) = atom as i32;
                            *(jp as *mut i32).add(atb) = target as i32;
                            *(sp as *mut [i32; 3]).add(atb) = shift;
                            *(dp as *mut f64).add(atb) = dist;
                        }
                    }
                    at[bead] += 1;
                    hits[bead] += 1;
                }
            }
        }
    }
    hits
}

fn write_ranges(
    xyz: &[[f64; 3]],
    soa: &Soa,
    ranges: &[(usize, usize)],
    off: &[usize],
    d2: &[f64],
    cut2: f64,
    want_vector: bool,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    use_avx: bool,
    out: &mut PairColumns,
    vectors: &mut Vec<[f64; 3]>,
) {
    let ip = out.i.as_mut_ptr() as usize;
    let jp = out.j.as_mut_ptr() as usize;
    let sp = out.shift.as_mut_ptr() as usize;
    let dp = out.dist2.as_mut_ptr() as usize;
    let vp = vectors.as_mut_ptr() as usize;
    let job = move |t: usize, lo: usize, hi: usize| {
        let mut at = off[t];
        for atom in lo..hi {
            let a = soa.offsets[atom] as usize;
            let b = soa.offsets[atom + 1] as usize;
            let p = xyz[atom];
            if use_avx && !want_vector && b - a >= 8 {
                at += compact_atom(soa, d2, atom, a, b, cut2, at, ip, jp, sp, dp);
            } else {
                for k in a..b {
                    let dist = d2[k];
                    if dist < cut2 {
                        let s = [soa.sx[k], soa.sy[k], soa.sz[k]];
                        unsafe {
                            *(ip as *mut i32).add(at) = atom as i32;
                            *(jp as *mut i32).add(at) = soa.j[k] as i32;
                            *(sp as *mut [i32; 3]).add(at) = s;
                            *(dp as *mut f64).add(at) = dist;
                            if want_vector {
                                let q = *xyz.get_unchecked(soa.j[k] as usize);
                                let (tx, ty, tz) = shift_xyz(s[0], s[1], s[2], ortho, widths, h);
                                *(vp as *mut [f64; 3]).add(at) =
                                    [q[0] + tx - p[0], q[1] + ty - p[1], q[2] + tz - p[2]];
                            }
                        }
                        at += 1;
                    }
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    if ranges.len() > 1 {
        use rayon::prelude::*;
        ranges
            .par_iter()
            .enumerate()
            .for_each(|(t, &(lo, hi))| job(t, lo, hi));
        return;
    }
    if let Some(&(lo, hi)) = ranges.first() {
        job(0, lo, hi);
    }
}

fn compact_atom(
    soa: &Soa,
    d2: &[f64],
    atom: usize,
    a: usize,
    b: usize,
    cut2: f64,
    at: usize,
    ip: usize,
    jp: usize,
    sp: usize,
    dp: usize,
) -> usize {
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    unsafe {
        compact_avx512(soa, d2, atom, a, b, cut2, at, ip, jp, sp, dp)
    }
    #[cfg(not(all(target_arch = "x86_64", linkcell_avx512)))]
    {
        let _ = (soa, d2, atom, a, b, cut2, at, ip, jp, sp, dp);
        0
    }
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f,avx512vl,avx2")]
unsafe fn compact_avx512(
    soa: &Soa,
    d2: &[f64],
    atom: usize,
    a: usize,
    b: usize,
    cut2: f64,
    mut at: usize,
    ip: usize,
    jp: usize,
    sp: usize,
    dp: usize,
) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_maskz_compress_epi32, _mm256_storeu_si256,
        _mm512_cmp_pd_mask, _mm512_loadu_pd, _mm512_maskz_compress_pd, _mm512_set1_pd,
        _mm512_storeu_pd, _CMP_LT_OQ,
    };
    let start = at;
    let cut = _mm512_set1_pd(cut2);
    let mut t = a;
    let mut js = [0i32; 8];
    let mut xs = [0i32; 8];
    let mut ys = [0i32; 8];
    let mut zs = [0i32; 8];
    let mut ds = [0f64; 8];
    while t + 8 <= b {
        let dist = unsafe { _mm512_loadu_pd(d2.as_ptr().add(t)) };
        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(dist, cut);
        let n = mask.count_ones() as usize;
        if n > 0 {
            unsafe {
                _mm256_storeu_si256(
                    js.as_mut_ptr() as *mut __m256i,
                    _mm256_maskz_compress_epi32(
                        mask,
                        _mm256_loadu_si256(soa.j.as_ptr().add(t) as *const __m256i),
                    ),
                );
                _mm256_storeu_si256(
                    xs.as_mut_ptr() as *mut __m256i,
                    _mm256_maskz_compress_epi32(
                        mask,
                        _mm256_loadu_si256(soa.sx.as_ptr().add(t) as *const __m256i),
                    ),
                );
                _mm256_storeu_si256(
                    ys.as_mut_ptr() as *mut __m256i,
                    _mm256_maskz_compress_epi32(
                        mask,
                        _mm256_loadu_si256(soa.sy.as_ptr().add(t) as *const __m256i),
                    ),
                );
                _mm256_storeu_si256(
                    zs.as_mut_ptr() as *mut __m256i,
                    _mm256_maskz_compress_epi32(
                        mask,
                        _mm256_loadu_si256(soa.sz.as_ptr().add(t) as *const __m256i),
                    ),
                );
                _mm512_storeu_pd(ds.as_mut_ptr(), _mm512_maskz_compress_pd(mask, dist));
            }
            let ii = atom as i32;
            for h in 0..n {
                unsafe {
                    *(ip as *mut i32).add(at) = ii;
                    *(jp as *mut i32).add(at) = js[h];
                    *(sp as *mut [i32; 3]).add(at) = [xs[h], ys[h], zs[h]];
                    *(dp as *mut f64).add(at) = ds[h];
                }
                at += 1;
            }
        }
        t += 8;
    }
    for k in t..b {
        let dist = d2[k];
        if dist < cut2 {
            unsafe {
                *(ip as *mut i32).add(at) = atom as i32;
                *(jp as *mut i32).add(at) = soa.j[k] as i32;
                *(sp as *mut [i32; 3]).add(at) = [soa.sx[k], soa.sy[k], soa.sz[k]];
                *(dp as *mut f64).add(at) = dist;
            }
            at += 1;
        }
    }
    at - start
}

#[allow(clippy::uninit_vec)]
fn prepare(out: &mut PairColumns, vectors: &mut Vec<[f64; 3]>, total: usize, want_vector: bool) {
    out.i.clear();
    out.j.clear();
    out.shift.clear();
    out.dist2.clear();
    vectors.clear();
    out.i.reserve(total);
    out.j.reserve(total);
    out.shift.reserve(total);
    out.dist2.reserve(total);
    unsafe {
        out.i.set_len(total);
        out.j.set_len(total);
        out.shift.set_len(total);
        out.dist2.set_len(total);
    }
    if want_vector {
        vectors.reserve(total);
        unsafe { vectors.set_len(total) };
    }
}

fn filter_threads(n: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::current_num_threads().max(1);
        if pool > 1 && n >= 32_768 {
            pool
        } else {
            1
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        1
    }
}

fn atom_ranges(offsets: &[u32], threads: usize) -> Vec<(usize, usize)> {
    let atoms = offsets.len().saturating_sub(1);
    if atoms == 0 {
        return Vec::new();
    }
    let threads = threads.max(1).min(atoms);
    let total = offsets[atoms] as usize;
    if threads == 1 || total == 0 {
        return vec![(0, atoms)];
    }
    let mut ranges = Vec::with_capacity(threads);
    let mut start = 0usize;
    for t in 1..threads {
        let target = total * t / threads;
        let mut end = start;
        while end < atoms && (offsets[end] as usize) < target {
            end += 1;
        }
        if end == start {
            end = (start + 1).min(atoms);
        }
        ranges.push((start, end));
        start = end;
    }
    if start < atoms {
        ranges.push((start, atoms));
    }
    ranges
}

fn count_one(
    xyz: &[[f64; 3]],
    offsets: &[u32],
    j: &[u32],
    sx: &[i32],
    sy: &[i32],
    sz: &[i32],
    lo: usize,
    hi: usize,
    cut2: f64,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    use_avx: bool,
    d2: &mut [f64],
) -> usize {
    let c0 = offsets[lo] as usize;
    let mut hits = 0usize;
    let mut local = 0usize;
    for atom in lo..hi {
        let a = offsets[atom] as usize;
        let b = offsets[atom + 1] as usize;
        let count = b - a;
        if count == 0 {
            continue;
        }
        hits += distance_block(
            xyz,
            j,
            sx,
            sy,
            sz,
            a,
            count,
            atom,
            cut2,
            ortho,
            widths,
            h,
            use_avx,
            &mut d2[local..local + count],
        );
        local += count;
    }
    let _ = c0;
    hits
}

fn count_ranges(
    xyz: &[[f64; 3]],
    offsets: &[u32],
    j: &[u32],
    sx: &[i32],
    sy: &[i32],
    sz: &[i32],
    ranges: &[(usize, usize)],
    cut2: f64,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    use_avx: bool,
    d2: &mut [f64],
) -> Vec<usize> {
    #[cfg(feature = "parallel")]
    if ranges.len() > 1 {
        use rayon::prelude::*;
        let ptr = d2.as_mut_ptr() as usize;
        return ranges
            .par_iter()
            .map(|&(lo, hi)| {
                let c0 = offsets[lo] as usize;
                let c1 = offsets[hi] as usize;
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (ptr + c0 * std::mem::size_of::<f64>()) as *mut f64,
                        c1 - c0,
                    )
                };
                count_one(
                    xyz, offsets, j, sx, sy, sz, lo, hi, cut2, ortho, widths, h, use_avx, slice,
                )
            })
            .collect();
    }
    ranges
        .iter()
        .map(|&(lo, hi)| {
            let c0 = offsets[lo] as usize;
            let c1 = offsets[hi] as usize;
            count_one(
                xyz,
                offsets,
                j,
                sx,
                sy,
                sz,
                lo,
                hi,
                cut2,
                ortho,
                widths,
                h,
                use_avx,
                &mut d2[c0..c1],
            )
        })
        .collect()
}

fn distance_block(
    xyz: &[[f64; 3]],
    j: &[u32],
    sx: &[i32],
    sy: &[i32],
    sz: &[i32],
    base: usize,
    count: usize,
    atom: usize,
    cut2: f64,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    use_avx: bool,
    d2: &mut [f64],
) -> usize {
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    if use_avx {
        return unsafe {
            distance_avx512(
                xyz, j, sx, sy, sz, base, count, atom, cut2, ortho, widths, h, d2,
            )
        };
    }
    let _ = use_avx;
    distance_scalar(
        xyz, j, sx, sy, sz, base, count, atom, cut2, ortho, widths, h, d2,
    )
}

fn distance_scalar(
    xyz: &[[f64; 3]],
    j: &[u32],
    sx: &[i32],
    sy: &[i32],
    sz: &[i32],
    base: usize,
    count: usize,
    atom: usize,
    cut2: f64,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    d2: &mut [f64],
) -> usize {
    let p = xyz[atom];
    let mut hits = 0usize;
    for t in 0..count {
        let q = xyz[j[base + t] as usize];
        let (tx, ty, tz) = shift_xyz(sx[base + t], sy[base + t], sz[base + t], ortho, widths, h);
        let dx = q[0] + tx - p[0];
        let dy = q[1] + ty - p[1];
        let dz = q[2] + tz - p[2];
        let dist = (dx * dx + dy * dy) + dz * dz;
        d2[t] = dist;
        if dist < cut2 {
            hits += 1;
        }
    }
    hits
}

pub(crate) fn shift_xyz(
    sx: i32,
    sy: i32,
    sz: i32,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
) -> (f64, f64, f64) {
    if ortho {
        (
            f64::from(sx) * widths[0],
            f64::from(sy) * widths[1],
            f64::from(sz) * widths[2],
        )
    } else {
        let (a, b, c) = (f64::from(sx), f64::from(sy), f64::from(sz));
        (
            a * h[0][0] + b * h[1][0] + c * h[2][0],
            a * h[0][1] + b * h[1][1] + c * h[2][1],
            a * h[0][2] + b * h[1][2] + c * h[2][2],
        )
    }
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f,avx512vl,avx2")]
unsafe fn distance_avx512(
    xyz: &[[f64; 3]],
    j: &[u32],
    sx: &[i32],
    sy: &[i32],
    sz: &[i32],
    base: usize,
    count: usize,
    atom: usize,
    cut2: f64,
    ortho: bool,
    widths: [f64; 3],
    h: [[f64; 3]; 3],
    d2: &mut [f64],
) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_set1_epi32,
        _mm512_add_pd, _mm512_cmp_pd_mask, _mm512_cvtepi32_pd, _mm512_i32gather_pd, _mm512_mul_pd,
        _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd, _CMP_LT_OQ,
    };
    let p = xyz[atom];
    let bytes = xyz.as_ptr() as *const f64;
    let jptr = j.as_ptr().add(base);
    let sxp = sx.as_ptr().add(base);
    let syp = sy.as_ptr().add(base);
    let szp = sz.as_ptr().add(base);
    let outp = d2.as_mut_ptr();
    let cut = _mm512_set1_pd(cut2);
    let px = _mm512_set1_pd(p[0]);
    let py = _mm512_set1_pd(p[1]);
    let pz = _mm512_set1_pd(p[2]);
    let three = _mm256_set1_epi32(3);
    let one = _mm256_set1_epi32(1);
    let two = _mm256_set1_epi32(2);
    let mut hits = 0usize;
    let mut t = 0usize;
    while t + 8 <= count {
        let jv = unsafe { _mm256_loadu_si256(jptr.add(t) as *const __m256i) };
        let ix = _mm256_mullo_epi32(jv, three);
        let qx = unsafe { _mm512_i32gather_pd(ix, bytes, 8) };
        let qy = unsafe { _mm512_i32gather_pd(_mm256_add_epi32(ix, one), bytes, 8) };
        let qz = unsafe { _mm512_i32gather_pd(_mm256_add_epi32(ix, two), bytes, 8) };
        let sxf = _mm512_cvtepi32_pd(unsafe { _mm256_loadu_si256(sxp.add(t) as *const __m256i) });
        let syf = _mm512_cvtepi32_pd(unsafe { _mm256_loadu_si256(syp.add(t) as *const __m256i) });
        let szf = _mm512_cvtepi32_pd(unsafe { _mm256_loadu_si256(szp.add(t) as *const __m256i) });
        let (tx, ty, tz) = if ortho {
            (
                _mm512_mul_pd(sxf, _mm512_set1_pd(widths[0])),
                _mm512_mul_pd(syf, _mm512_set1_pd(widths[1])),
                _mm512_mul_pd(szf, _mm512_set1_pd(widths[2])),
            )
        } else {
            (
                _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_mul_pd(sxf, _mm512_set1_pd(h[0][0])),
                        _mm512_mul_pd(syf, _mm512_set1_pd(h[1][0])),
                    ),
                    _mm512_mul_pd(szf, _mm512_set1_pd(h[2][0])),
                ),
                _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_mul_pd(sxf, _mm512_set1_pd(h[0][1])),
                        _mm512_mul_pd(syf, _mm512_set1_pd(h[1][1])),
                    ),
                    _mm512_mul_pd(szf, _mm512_set1_pd(h[2][1])),
                ),
                _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_mul_pd(sxf, _mm512_set1_pd(h[0][2])),
                        _mm512_mul_pd(syf, _mm512_set1_pd(h[1][2])),
                    ),
                    _mm512_mul_pd(szf, _mm512_set1_pd(h[2][2])),
                ),
            )
        };
        let dx = _mm512_sub_pd(_mm512_add_pd(qx, tx), px);
        let dy = _mm512_sub_pd(_mm512_add_pd(qy, ty), py);
        let dz = _mm512_sub_pd(_mm512_add_pd(qz, tz), pz);
        let dist = _mm512_add_pd(
            _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
            _mm512_mul_pd(dz, dz),
        );
        unsafe { _mm512_storeu_pd(outp.add(t), dist) };
        hits += _mm512_cmp_pd_mask::<_CMP_LT_OQ>(dist, cut).count_ones() as usize;
        t += 8;
    }
    if t < count {
        hits += distance_scalar(
            xyz,
            j,
            sx,
            sy,
            sz,
            base + t,
            count - t,
            atom,
            cut2,
            ortho,
            widths,
            h,
            &mut d2[t..],
        );
    }
    hits
}
