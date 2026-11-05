//! Time `pairs_within` on one mask written by `pairs_ranks`.
//!
//! Env: `PAIRS_N`, `PAIRS_BOX`, `PAIRS_MASK` (path of a `u8` mask).

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
    let boxl = env_f64("PAIRS_BOX", 207.5);
    let path = std::env::var("PAIRS_MASK").expect("PAIRS_MASK");
    let bytes = std::fs::read(&path).expect("read mask");
    assert_eq!(bytes.len(), n);
    let mask: Vec<bool> = bytes.iter().map(|b| *b != 0).collect();
    let xyz = fill(n, boxl);
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let active = mask.iter().filter(|m| **m).count();
    let _ = pairs_within(&xyz, &cell, 4.0, Some(&mask), None, false).expect("warm");
    let t0 = Instant::now();
    let got = pairs_within(&xyz, &cell, 4.0, Some(&mask), None, false).expect("pairs");
    let pairs_ns = t0.elapsed().as_nanos() as u64;
    println!("mask={path} active={active} pairs={} pairs_ns={pairs_ns}", got.len());
}
