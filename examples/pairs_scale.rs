//! Cutoff-list probe for the periodic 18 Å cube at a 4 Å cutoff.
//!
//! Env: `PAIRS_N` (default 4096), `PAIRS_REPS` (default 5),
//! `PAIRS_HALF` (default 0). `RAYON_NUM_THREADS` is read at startup.

use std::time::Instant;

use linkcell::{pairs_within, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn fill(n: usize) -> Vec<[f64; 3]> {
    let boxl = 18.0;
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
    let n = env_usize("PAIRS_N", 4096);
    let reps = env_usize("PAIRS_REPS", 5).max(1);
    let half = std::env::var("PAIRS_HALF").ok().as_deref() == Some("1");
    let hint = std::env::var("PAIRS_HINT")
        .ok()
        .and_then(|s| s.parse().ok());
    let xyz = fill(n);
    let cell = Cell::ortho(18.0, 18.0, 18.0).expect("box");
    // The first calls fault the pair buffer. Time the calls after that.
    let mut warmup = Vec::new();
    for _ in 0..3 {
        warmup = pairs_within(&xyz, &cell, 4.0, None, hint, half).expect("pairs");
    }
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..reps {
        let pairs = pairs_within(&xyz, &cell, 4.0, None, hint, half).expect("pairs");
        acc += pairs.len();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "n={} half={} hint={:?} pairs={} reps={} ms_per={:.4} acc={}",
        xyz.len(),
        half as u8,
        hint,
        warmup.len(),
        reps,
        ms / reps as f64,
        acc
    );
}
