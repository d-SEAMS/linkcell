//! One-call k-nearest timings.
//!
//! The lattice matches `knn_scale`: spacing 3.125, orthorhombic box,
//! `k` from `KNN_BENCH_K` (default 4), bin edge 3.0. `KNN_BENCH_N` is
//! rounded to a cube. `KNN_BENCH_REPS` (default 11) is the number of
//! timed calls after one warmup. `KNN_BENCH_PIN=1` opts into
//! least-loaded CPU placement through the library API.

use std::time::Instant;

use linkcell::{knearest_into, knearest_team_threads, pin_knearest_to_least_loaded, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn cubic(nside: usize) -> (Vec<[f64; 3]>, Cell, f64) {
    let spacing = 3.125;
    let boxl = nside as f64 * spacing;
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let mut xyz = Vec::with_capacity(nside * nside * nside);
    for iz in 0..nside {
        for iy in 0..nside {
            for ix in 0..nside {
                xyz.push([
                    ix as f64 * spacing,
                    iy as f64 * spacing,
                    iz as f64 * spacing,
                ]);
            }
        }
    }
    (xyz, cell, spacing)
}

fn main() {
    let nside = ((env_usize("KNN_BENCH_N", 110_592) as f64).cbrt().round() as usize).max(1);
    let k = env_usize("KNN_BENCH_K", 4);
    let reps = env_usize("KNN_BENCH_REPS", 11).max(1);
    if std::env::var("KNN_BENCH_PIN").ok().as_deref() == Some("1") {
        pin_knearest_to_least_loaded(true);
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let (xyz, cell, spacing) = cubic(nside);
    let mut out = vec![-1i32; xyz.len() * k];
    knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("warmup");
    let mut samples = Vec::with_capacity(reps);
    let mut acc = 0i32;
    for _ in 0..reps {
        let started = Instant::now();
        knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("knearest");
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        acc = acc.wrapping_add(out[0]);
    }
    std::hint::black_box(acc);
    let threads = knearest_team_threads();
    let last = xyz.last().copied().unwrap_or([0.0; 3]);
    println!(
        "n={} k={k} threads={threads} cores={cores} reps={reps} spacing={spacing} box={:.6} first={:.6},{:.6},{:.6} last={:.6},{:.6},{:.6}",
        xyz.len(),
        cell.a()[0],
        xyz[0][0],
        xyz[0][1],
        xyz[0][2],
        last[0],
        last[1],
        last[2],
    );
    for ms in samples {
        println!("sample_ms={ms:.6}");
    }
}
