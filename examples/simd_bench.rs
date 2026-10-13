//! Time k-nearest search under the selected SIMD path.
//!
//! `SIMD_BENCH_N` (default 10000), `SIMD_BENCH_K` (default 4),
//! `SIMD_BENCH_REPS` (default 5), `SIMD_BENCH_SHAPE` (`lattice` or
//! `jitter`). `LINKCELL_SIMD` forces the path (`scalar`, `avx2`,
//! `avx512`, `neon`). `RAYON_NUM_THREADS` is the caller's.
//!
//! Build with `--features timing`. `LINKCELL_SCAN_TIME` fills
//! [`linkcell::SimdTiming::scan_ns`]. `LINKCELL_COARSE_TIME` fills
//! `mesh_ns` and `knn_ns`.

use std::time::Instant;

use linkcell::{knearest_into, simd_path, simd_timing, Cell};

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

fn xorshift(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

fn lattice(n: usize) -> (Vec<[f64; 3]>, Cell) {
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
    (xyz, cell)
}

fn jitter(xyz: &mut [[f64; 3]]) {
    // A few ulps of the lattice spacing. Bitwise ulps of a zero coordinate
    // wrap through the sign bit and become NaN.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let step = f64::from_bits(3.125f64.to_bits() + 1) - 3.125;
    for p in xyz {
        for c in p.iter_mut() {
            let ulps = (xorshift(&mut state) % 7) as i64 - 3;
            *c += ulps as f64 * step;
        }
    }
}

fn main() {
    let n = env_usize("SIMD_BENCH_N", 10_000);
    let k = env_usize("SIMD_BENCH_K", 4).max(1);
    let reps = env_usize("SIMD_BENCH_REPS", 5).max(1);
    let shape = std::env::var("SIMD_BENCH_SHAPE").unwrap_or_else(|_| "lattice".to_string());
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let (mut xyz, cell) = lattice(n);
    if shape == "jitter" {
        jitter(&mut xyz);
    }
    let mut out = vec![-1i32; n * k];
    knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("warmup");
    let before = simd_timing();
    let started = Instant::now();
    let mut acc = 0i32;
    for _ in 0..reps {
        knearest_into(&xyz, &cell, k, None, Some(3.0), &mut out).expect("knearest");
        acc = acc.wrapping_add(out[0]);
    }
    std::hint::black_box(acc);
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let after = simd_timing();
    println!(
        "cpu={} path={} shape={shape} n={n} k={k} threads={threads} reps={reps} ms={ms:.3} ms_per={:.3} mesh_ns={} scan_ns={} knn_ns={}",
        cpu_model(),
        simd_path(),
        ms / reps as f64,
        after.mesh_ns.saturating_sub(before.mesh_ns),
        after.scan_ns.saturating_sub(before.scan_ns),
        after.knn_ns.saturating_sub(before.knn_ns),
    );
}
