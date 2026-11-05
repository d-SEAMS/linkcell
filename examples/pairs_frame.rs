//! Larger cutoff-list probe. The 18 Å cube hides a method gap inside a
//! two-millisecond call. This frame holds about a million points in a
//! wider orthorhombic cell, so the bin columns miss the 2 MB cache and
//! the timed call is long enough for the ratio to settle.
//!
//! Env: `PAIRS_N` (default 1_000_000), `PAIRS_BOX` (default 207.5 Å, about
//! thirty ideal-gas neighbours at a 4 Å cutoff), `PAIRS_REPS` (default 1).
//! `RAYON_NUM_THREADS` is read at startup. Three warmups, then the timed
//! repetitions.

use std::time::Instant;

use linkcell::{pairs_within, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn fill(n: usize, boxl: f64) -> Vec<[f64; 3]> {
    let m = (n as f64).cbrt().ceil() as usize;
    let mut xyz = Vec::with_capacity(n);
    let mut count = 0usize;
    'outer: for iz in 0..m {
        for iy in 0..m {
            for ix in 0..m {
                if count >= n {
                    break 'outer;
                }
                xyz.push([
                    (ix as f64 + 0.5) * boxl / m as f64,
                    (iy as f64 + 0.5) * boxl / m as f64,
                    (iz as f64 + 0.5) * boxl / m as f64,
                ]);
                count += 1;
            }
        }
    }
    xyz
}

fn main() {
    let n = env_usize("PAIRS_N", 1_000_000);
    let reps = env_usize("PAIRS_REPS", 1).max(1);
    let boxl = env_f64("PAIRS_BOX", 207.5);
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let xyz = fill(n, boxl);
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    linkcell::pop_engage();
    linkcell::pop_prepare();
    let mut pairs = Vec::new();
    for _ in 0..3 {
        pairs = pairs_within(&xyz, &cell, 4.0, None, None, false).expect("pairs");
    }
    linkcell::pop_reset();
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..reps {
        let got = pairs_within(&xyz, &cell, 4.0, None, None, false).expect("pairs");
        acc += got.len();
    }
    let wall_ns = t0.elapsed().as_nanos() as u64;
    let ms = wall_ns as f64 / 1e6 / reps as f64;
    println!(
        "n={} box={boxl} pairs={} reps={reps} ms_per={ms:.4} acc={acc}",
        xyz.len(),
        pairs.len()
    );
    println!(
        "pop shape=frame-{n} threads={threads} n={n} reps={reps} wall_ns={wall_ns}"
    );
}
