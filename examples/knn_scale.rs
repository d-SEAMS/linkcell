//! Strong-scaling probe for one fixed problem.
//!
//! The box grows with `n` so the spacing stays 3.125, the cubic hot-path
//! density. Set `RAYON_NUM_THREADS` before the process starts. Env:
//! `KNN_SCALE_N` (default 262144), `KNN_SCALE_K` (default 4),
//! `KNN_SCALE_REPS` (default 6), `KNN_SCALE_SHAPE` (`cubic` or `tilt`),
//! `KNN_SCALE_HINT` (bin edge, default 3.0).
//! `KNN_SCALE_SHUFFLE` (a seed) permutes the atoms, so their order no longer follows
//! their position, as in a snapshot after a run.

use std::time::Instant;

use linkcell::{knearest_into, Cell};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn nside_of(n: usize) -> usize {
    ((n as f64).cbrt().round() as usize).max(1)
}

fn cubic(nside: usize) -> (Vec<[f64; 3]>, Cell) {
    let boxl = nside as f64 * 3.125;
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let mut xyz = Vec::with_capacity(nside * nside * nside);
    let a = 3.125;
    for iz in 0..nside {
        for iy in 0..nside {
            for ix in 0..nside {
                xyz.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
            }
        }
    }
    (xyz, cell)
}

/// Restricted triclinic box, tilt-reduced, scaled so the a edge matches
/// the cubic spacing times `nside`.
fn tilt(nside: usize) -> (Vec<[f64; 3]>, Cell) {
    let lx = nside as f64 * 3.125;
    let cell = Cell::from_vectors(
        [lx, 0.0, 0.0],
        [lx * 0.2, lx * 0.9, 0.0],
        [lx * 0.05, lx * -0.08, lx * 0.95],
        [0.0, 0.0, 0.0],
    )
    .expect("tilt");
    let mut xyz = Vec::with_capacity(nside * nside * nside);
    let nf = nside as f64;
    for iz in 0..nside {
        for iy in 0..nside {
            for ix in 0..nside {
                let s = [
                    (ix as f64 + 0.5) / nf,
                    (iy as f64 + 0.5) / nf,
                    (iz as f64 + 0.5) / nf,
                ];
                xyz.push(cell.cartesian(s));
            }
        }
    }
    (xyz, cell)
}

/// Fisher-Yates with an xorshift stream from `seed`.
fn shuffle(xyz: &mut [[f64; 3]], seed: u64) {
    let mut s = seed.max(1);
    for i in (1..xyz.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        xyz.swap(i, (s % (i as u64 + 1)) as usize);
    }
}

fn main() {
    let nside = nside_of(env_usize("KNN_SCALE_N", 262_144));
    let k = env_usize("KNN_SCALE_K", 4);
    let reps = env_usize("KNN_SCALE_REPS", 6);
    let shape = std::env::var("KNN_SCALE_SHAPE").unwrap_or_else(|_| "cubic".to_string());
    let hint: f64 = std::env::var("KNN_SCALE_HINT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let (mut xyz, cell) = match shape.as_str() {
        "cubic" => cubic(nside),
        "tilt" => tilt(nside),
        other => panic!("unknown KNN_SCALE_SHAPE {other}"),
    };
    if let Some(seed) = std::env::var("KNN_SCALE_SHUFFLE")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        shuffle(&mut xyz, seed);
    }
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let mut out = vec![-1i32; xyz.len() * k];
    linkcell::pop_engage();
    linkcell::pop_prepare();
    knearest_into(&xyz, &cell, k, None, Some(hint), &mut out).expect("warmup");
    linkcell::pop_reset();
    let mut acc = 0i32;
    let started = Instant::now();
    for _ in 0..reps {
        knearest_into(&xyz, &cell, k, None, Some(hint), &mut out).expect("knearest");
        acc = acc.wrapping_add(out[0]);
    }
    let wall_ns = started.elapsed().as_nanos() as u64;
    std::hint::black_box(acc);
    let useful = linkcell::pop_snapshot();
    let ms = wall_ns as f64 / 1e6;
    println!(
        "shape={shape} threads={threads} n={} k={k} reps={reps} ms={ms:.1} ms_per={:.2}",
        xyz.len(),
        ms / reps as f64
    );
    let (lb, ce, pe) = linkcell::pop_efficiencies(&useful, wall_ns);
    let list = useful
        .iter()
        .map(|ns| ns.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "pop shape={shape} threads={threads} n={} reps={reps} wall_ns={wall_ns} useful_ns={list} lb={lb:.4} ce={ce:.4} pe={pe:.4}",
        xyz.len()
    );
}
