//! Time k-nearest search under the selected SIMD path.
//!
//! `SIMD_BENCH_N` (default 10000), `SIMD_BENCH_K` (default 4),
//! `SIMD_BENCH_REPS` (default 5). `LINKCELL_SIMD` forces the path
//! (`scalar`, `avx2`, `avx512`, `neon`). `RAYON_NUM_THREADS` is the
//! caller's.

use std::time::Instant;

use linkcell::{knearest_into, simd_path, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn cpu_model() -> String {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return std::env::consts::ARCH.to_string();
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            return rest.trim_start_matches([' ', '\t', ':']).to_string();
        }
    }
    std::env::consts::ARCH.to_string()
}

fn main() {
    let n = env_usize("SIMD_BENCH_N", 10_000);
    let k = env_usize("SIMD_BENCH_K", 4).max(1);
    let reps = env_usize("SIMD_BENCH_REPS", 5).max(1);
    let nside = (n as f64).cbrt().ceil() as usize;
    let spacing = 3.125;
    let boxl = nside as f64 * spacing;
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let mut xyz = Vec::with_capacity(n);
    'fill: for iz in 0..nside {
        for iy in 0..nside {
            for ix in 0..nside {
                xyz.push([
                    ix as f64 * spacing,
                    iy as f64 * spacing,
                    iz as f64 * spacing,
                ]);
                if xyz.len() == n {
                    break 'fill;
                }
            }
        }
    }
    let mut out = vec![-1i32; n * k];
    knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("warmup");
    let started = Instant::now();
    let mut acc = 0i32;
    for _ in 0..reps {
        knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("knearest");
        acc = acc.wrapping_add(out[0]);
    }
    std::hint::black_box(acc);
    let ms = started.elapsed().as_secs_f64() * 1e3;
    println!(
        "cpu={} path={} n={n} k={k} reps={reps} ms={ms:.3} ms_per={:.3}",
        cpu_model(),
        simd_path(),
        ms / reps as f64
    );
}
