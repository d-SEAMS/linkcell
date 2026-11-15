//! Time `pairs_within` on one mask written by `pairs_ranks`.
//!
//! Env: `PAIRS_N`, `PAIRS_BOX`, `PAIRS_MASK` (path of a `u8` mask).

use std::time::Instant;

use linkcell::{pairs_within_into, Cell, Pair};

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
    let mut rows = Vec::<Pair>::new();
    let _ = pairs_within_into(&xyz, &cell, 4.0, Some(&mask), None, false, &mut rows);
    let (f0, j0, rss0) = usage();
    let t0 = Instant::now();
    pairs_within_into(&xyz, &cell, 4.0, Some(&mask), None, false, &mut rows).expect("pairs");
    let pairs_ns = t0.elapsed().as_nanos() as u64;
    let (f1, j1, rss1) = usage();
    println!(
        "active={active} pairs={} pairs_ns={pairs_ns} minflt={} majflt={} rss_kb={rss0}->{rss1}",
        rows.len(),
        f1.saturating_sub(f0),
        j1.saturating_sub(j0)
    );
}

fn usage() -> (u64, u64, u64) {
    let raw = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = raw.split_once(')').map(|(_, r)| r).unwrap_or("");
    let bits: Vec<&str> = rest.split_whitespace().collect();
    let num = |i: usize| bits.get(i).and_then(|s| s.parse().ok()).unwrap_or(0u64);
    let rss = std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (num(7), num(9), rss)
}
