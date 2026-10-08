//! Hooks for an external autotuner such as Kernel Tuner: the cutoff
//! walk's thresholds and paths as runtime values, and a timed call that
//! reports a checksum of its rows, so every configuration can be checked
//! against the default one. Built only with the `tune` feature; the
//! shipped header does not declare these symbols, and
//! `scripts/tune-pairs.py` drives them.

use std::os::raw::c_int;
use std::time::Instant;

use crate::capi::lc_cell;
use crate::pairs::knobs;
use crate::Cell;

/// Set knob `key` of the cutoff walk to `value` for this process:
/// 0 = expected pairs where the walk splits across threads,
/// 1 = active atoms from which a split walk builds its bins on several
/// threads, 2 = one thread writes a full list from the tile (nonzero) or
/// buffers its hits first (zero), 3 = bin ranges per thread in a split
/// search. Returns 0, or 1 for an unknown key or a negative value.
#[no_mangle]
pub extern "C" fn lc_tune_set(key: c_int, value: f64) -> c_int {
    let slot = match key {
        0 => knobs::SPLIT_PAIRS,
        1 => knobs::GRID_ATOMS,
        2 => knobs::FUSED,
        3 => knobs::CHUNKS,
        _ => return 1,
    };
    if value.is_nan() || value < 0.0 {
        return 1;
    }
    let v = if value >= usize::MAX as f64 {
        usize::MAX
    } else {
        value as usize
    };
    knobs::VALUES[slot].store(v, std::sync::atomic::Ordering::Relaxed);
    0
}

/// Three warm calls of [`crate::pairs_within`] on a full list, then
/// `reps` timed ones. Writes `out[0]` = milliseconds per timed call,
/// `out[1]` = rows, `out[2]` and `out[3]` = the low and high 32 bits of a
/// checksum of the `(i, j, S)` rows that does not depend on their order,
/// and `out[4]` = the sum of `dist2`, which moves in its last bits with
/// the bin edge. Returns 0, or 1 on a null pointer, a bad box, or a
/// failed search.
///
/// # Safety
/// `xyz` holds `3 * n` doubles, `simbox` one cell, and `out` five doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_tune_pairs(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    cell_hint: f64,
    reps: usize,
    out: *mut f64,
) -> c_int {
    if xyz.is_null() || simbox.is_null() || out.is_null() || n == 0 {
        return 1;
    }
    let b = *simbox;
    let sim = match Cell::from_vectors(
        [b.ax, b.ay, b.az],
        [b.bx, b.by, b.bz],
        [b.cx, b.cy, b.cz],
        [b.ox, b.oy, b.oz],
    ) {
        Ok(sim) => sim,
        Err(_) => return 1,
    };
    let pts = std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n);
    let hint = if cell_hint > 0.0 {
        Some(cell_hint)
    } else {
        None
    };
    let run = || crate::pairs_within(pts, &sim, cutoff, None, hint, false);
    let mut rows = match run() {
        Ok(rows) => rows,
        Err(_) => return 1,
    };
    for _ in 0..2 {
        rows = match run() {
            Ok(rows) => rows,
            Err(_) => return 1,
        };
    }
    let reps = reps.max(1);
    let t0 = Instant::now();
    for _ in 0..reps {
        rows = match run() {
            Ok(rows) => rows,
            Err(_) => return 1,
        };
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    // A sum of a mix of each row: the same rows in any order give the
    // same value.
    let (mut sum, mut d2) = (0u64, 0.0f64);
    for p in &rows {
        let mut h = (p.i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= (p.j as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        for (k, s) in p.shift.iter().enumerate() {
            h ^= (*s as i64 as u64).wrapping_mul(0x1656_67b1_9e37_79f9 << k);
        }
        h ^= h >> 31;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 29;
        sum = sum.wrapping_add(h);
        d2 += p.dist2;
    }
    *out = ms;
    *out.add(1) = rows.len() as f64;
    *out.add(2) = (sum & 0xffff_ffff) as f64;
    *out.add(3) = (sum >> 32) as f64;
    *out.add(4) = d2;
    0
}
