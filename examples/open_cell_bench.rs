//! Open-cell list against the padded periodic box.
//!
//! A 27-atom cluster, one corner atom moved 400 times. `OPEN_REPS`
//! (default 8) is the number of timed repetitions. The report is the
//! median and the interquartile range of those repetitions, in
//! milliseconds, and the core count. No host name.

use std::time::Instant;

use linkcell::{pairs_open, pairs_within, stencil_builds, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= state.wrapping_shl(13);
    *state ^= state.wrapping_shr(7);
    *state ^= state.wrapping_shl(17);
    *state
}

fn cluster() -> Vec<[f64; 3]> {
    let mut xyz = Vec::with_capacity(27);
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                xyz.push([(ix + 2) as f64, (iy + 2) as f64, (iz + 2) as f64]);
            }
        }
    }
    xyz
}

fn nudge(xyz: &mut [[f64; 3]], state: &mut u64) {
    let step = 0.25;
    for coord in &mut xyz[26] {
        let draw = xorshift(state);
        let unit = (draw % 10001) as f64 / 10000.0;
        *coord += unit * 2.0 * step - step;
    }
}

fn padded_box(xyz: &[[f64; 3]], cutoff: f64) -> Cell {
    let mut length = [0.0; 3];
    for axis in 0..3 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for point in xyz {
            lo = lo.min(point[axis]);
            hi = hi.max(point[axis]);
        }
        let span = (hi - lo).max(0.0);
        length[axis] = 2.0 * span + 4.0 * cutoff + 10.0;
    }
    Cell::ortho(length[0], length[1], length[2]).expect("box")
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() - 1;
    let pos = q * last as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = pos - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

fn summary(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let median = percentile(samples, 0.5);
    let iqr = percentile(samples, 0.75) - percentile(samples, 0.25);
    (median, iqr)
}

fn time_open(moves: usize) -> (f64, usize) {
    let cutoff = 2.5;
    let hint = Some(8.0);
    let mut xyz = cluster();
    let mut state = 42u64;
    let mut rows = 0usize;
    let start = Instant::now();
    for _ in 0..moves {
        rows += pairs_open(&xyz, cutoff, hint).expect("open").len();
        nudge(&mut xyz, &mut state);
    }
    let ms = start.elapsed().as_secs_f64() * 1.0e3;
    (ms, rows)
}

fn time_padded(moves: usize) -> (f64, usize) {
    let cutoff = 2.5;
    let hint = Some(8.0);
    let mut xyz = cluster();
    let mut state = 42u64;
    let mut rows = 0usize;
    let start = Instant::now();
    for _ in 0..moves {
        let cell = padded_box(&xyz, cutoff);
        rows += pairs_within(&xyz, &cell, cutoff, None, hint, true)
            .expect("padded")
            .len();
        nudge(&mut xyz, &mut state);
    }
    let ms = start.elapsed().as_secs_f64() * 1.0e3;
    (ms, rows)
}

fn main() {
    let reps = env_usize("OPEN_REPS", 8).max(8);
    let moves = env_usize("OPEN_MOVES", 400);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let _ = time_open(moves);
    let _ = time_padded(moves);
    let before = stencil_builds();
    let mut open_ms = Vec::with_capacity(reps);
    let mut padded_ms = Vec::with_capacity(reps);
    let mut open_rows = 0usize;
    let mut padded_rows = 0usize;
    for _ in 0..reps {
        let (ms, rows) = time_open(moves);
        open_ms.push(ms);
        open_rows += rows;
        let (ms, rows) = time_padded(moves);
        padded_ms.push(ms);
        padded_rows += rows;
    }
    let (open_median, open_iqr) = summary(&mut open_ms);
    let (padded_median, padded_iqr) = summary(&mut padded_ms);
    println!("cores={cores} reps={reps} moves={moves} atoms=27");
    println!("open_ms median={open_median:.4} iqr={open_iqr:.4} rows={open_rows}");
    println!("padded_ms median={padded_median:.4} iqr={padded_iqr:.4} rows={padded_rows}");
    println!("stencil_builds_during_bench={}", stencil_builds() - before);
}
