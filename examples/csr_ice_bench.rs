//! Cutoff CSR rows on an ice-like frame against the plain pair search.
//!
//! Cubic ice (diamond lattice, O-O 2.76 Å, 0.0311 per cubic angstrom)
//! with a 0.15 Å jitter, in lattice order, cutoff 3.5 Å, orthorhombic.
//! `ICE_CELLS` unit cells a side give `8 * cells^3` atoms: 8 is 4096,
//! 24 is 110592. Each case is timed call by call after `ICE_WARM`
//! (default 3) calls, `ICE_REPS` (default 15) times, at each width in
//! `ICE_THREADS` (default `1,8`, set by `lc_set_threads`):
//!
//! - `csr`: `lc_pairs_within_csr`, full list, no distances.
//! - `csr_d2`: the same with distances.
//! - `rows`: `lc_pairs_within_rows`, full list, into a large enough
//!   buffer: the plain search with its rows written.
//! - `count`: `pairs_within_count`, full list: the search alone.
//!
//! `ICE_CASES` (comma separated) picks a subset. `ICE_SHUFFLE=1` numbers
//! the atoms in a random order instead. Each case prints one `median`
//! line with the min and max.

use std::os::raw::c_int;
use std::ptr::{null, null_mut};
use std::time::Instant;

use linkcell::{lc_cell, lc_pair, lc_pairs_within_csr, lc_pairs_within_rows, lc_set_threads};

const CUTOFF: f64 = 3.5;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn ice(cells: usize) -> (Vec<[f64; 3]>, f64) {
    let a = 4.0 * 2.76 / 3f64.sqrt();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let edge = a * cells as f64;
    let mut xyz = Vec::with_capacity(8 * cells * cells * cells);
    for iz in 0..cells {
        for iy in 0..cells {
            for ix in 0..cells {
                for b in basis {
                    let p = [
                        a * (ix as f64 + b[0]) + 0.3 * next(),
                        a * (iy as f64 + b[1]) + 0.3 * next(),
                        a * (iz as f64 + b[2]) + 0.3 * next(),
                    ];
                    xyz.push(p.map(|v| v.rem_euclid(edge)));
                }
            }
        }
    }
    (xyz, edge)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

fn main() {
    let cells = env_usize("ICE_CELLS", 24);
    let reps = env_usize("ICE_REPS", 15).max(1);
    let warm = env_usize("ICE_WARM", 3);
    let widths: Vec<i32> = std::env::var("ICE_THREADS")
        .unwrap_or_else(|_| "1,8".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (mut xyz, edge) = ice(cells);
    let n = xyz.len();
    if env_usize("ICE_SHUFFLE", 0) != 0 {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for k in (1..n).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            xyz.swap(k, (state % (k as u64 + 1)) as usize);
        }
    }
    let cell = lc_cell {
        ax: edge,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: edge,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: edge,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let rcell = linkcell::Cell::ortho(edge, edge, edge).unwrap();
    println!(
        "# n={n} edge={edge:.3} reps={reps} warm={warm} shuffle={} simd={:?} version={}",
        env_usize("ICE_SHUFFLE", 0),
        linkcell::simd_path(),
        env!("CARGO_PKG_VERSION")
    );
    let cap = 64 * n;
    let mut offsets: Vec<c_int> = vec![0; n + 1];
    let mut cols: Vec<c_int> = vec![0; cap];
    let mut d2: Vec<f64> = vec![0.0; cap];
    let mut rows: Vec<lc_pair> = vec![lc_pair::default(); cap];
    for &t in &widths {
        lc_set_threads(t);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t as usize)
            .build()
            .unwrap();
        let cases = std::env::var("ICE_CASES").unwrap_or_else(|_| "csr,csr_d2,rows,count".into());
        for case in cases.split(',') {
            let mut run = || -> usize {
                let (mut no, mut nc) = (0usize, 0usize);
                match case {
                    "csr" | "csr_d2" => {
                        let out_d2 = if case == "csr_d2" {
                            d2.as_mut_ptr()
                        } else {
                            null_mut()
                        };
                        let rc = unsafe {
                            lc_pairs_within_csr(
                                xyz.as_ptr().cast(),
                                n,
                                &cell,
                                null(),
                                CUTOFF,
                                null(),
                                0.0,
                                0,
                                offsets.as_mut_ptr(),
                                n + 1,
                                cols.as_mut_ptr(),
                                out_d2,
                                cap,
                                &mut no,
                                &mut nc,
                            )
                        };
                        assert_eq!(rc, 0);
                        nc
                    }
                    "rows" => {
                        let rc = unsafe {
                            lc_pairs_within_rows(
                                xyz.as_ptr().cast(),
                                n,
                                &cell,
                                CUTOFF,
                                null(),
                                0.0,
                                0,
                                rows.as_mut_ptr(),
                                cap,
                                &mut nc,
                            )
                        };
                        assert_eq!(rc, 0);
                        nc
                    }
                    _ => pool.install(|| {
                        linkcell::pairs_within_count(&xyz, &rcell, CUTOFF, None, None, false)
                            .unwrap()
                    }),
                }
            };
            for _ in 0..warm {
                run();
            }
            let mut ms = Vec::with_capacity(reps);
            let mut size = 0;
            for _ in 0..reps {
                let start = Instant::now();
                size = run();
                ms.push(start.elapsed().as_secs_f64() * 1e3);
            }
            let lo = ms.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = ms.iter().copied().fold(0.0, f64::max);
            println!(
                "median n={n} threads={t} case={case} ms={:.3} min={lo:.3} max={hi:.3} size={size}",
                median(&mut ms)
            );
        }
    }
    lc_set_threads(0);
}
