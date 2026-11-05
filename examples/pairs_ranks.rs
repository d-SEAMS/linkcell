//! RCB domains of the million-point frame, then `pairs_within` on each
//! rank's owned cells plus one cutoff of halo. Masks are written so the
//! public main binary can time the same subsets.
//!
//! Env: `PAIRS_N`, `PAIRS_BOX`, `PAIRS_MASK_DIR`, `PAIRS_PARTS` (default
//! `1,2,4,8`).

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use linkcell::{
    cell_grid, halo, halo_width, pairs_within, partition, Cell, Method, PartRequest,
};

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

fn bin_coord(s: f64, n: i32) -> i32 {
    let mut c = (s * f64::from(n)) as i32;
    if c < 0 {
        c = 0;
    }
    if c >= n {
        c = n - 1;
    }
    c
}

fn main() {
    let n = env_usize("PAIRS_N", 1_000_000);
    let boxl = env_f64("PAIRS_BOX", 207.5);
    let dir = PathBuf::from(std::env::var("PAIRS_MASK_DIR").expect("PAIRS_MASK_DIR"));
    fs::create_dir_all(&dir).expect("mask dir");
    let parts_list: Vec<u32> = std::env::var("PAIRS_PARTS")
        .unwrap_or_else(|_| "1,2,4,8".to_string())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let xyz = fill(n, boxl);
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let grid = cell_grid(&xyz, &cell, 4.0, None, None).expect("grid");
    let width = halo_width(4.0, 1, grid.bin_width).expect("halo width");
    println!(
        "frame n={} box={boxl} cells={} width={}x{}x{}",
        xyz.len(),
        grid.ncell(),
        width[0],
        width[1],
        width[2]
    );
    for parts in parts_list {
        let t0 = Instant::now();
        let assign = partition(
            &xyz,
            &cell,
            4.0,
            None,
            None,
            PartRequest {
                parts,
                method: Method::Rcb,
                balance: None,
                device: false,
            },
        )
        .expect("partition");
        let part_ns = t0.elapsed().as_nanos() as u64;
        println!("partition parts={parts} method=rcb part_ns={part_ns}");
        for rank in 0..parts {
            let ghosts = halo(&grid, &assign, rank, width).expect("halo");
            let mut live = vec![false; grid.ncell()];
            for (c, &domain) in assign.iter().enumerate() {
                if domain == rank {
                    live[c] = true;
                }
            }
            for g in &ghosts {
                live[*g as usize] = true;
            }
            let mut mask = vec![false; xyz.len()];
            let mut owned = 0usize;
            for (i, p) in xyz.iter().enumerate() {
                let s = cell.fractional(*p);
                let ix = bin_coord(s[0], grid.nx) as usize;
                let iy = bin_coord(s[1], grid.ny) as usize;
                let iz = bin_coord(s[2], grid.nz) as usize;
                let c = (iz * grid.ny as usize + iy) * grid.nx as usize + ix;
                if assign[c] == rank {
                    owned += 1;
                }
                mask[i] = live[c];
            }
            let active = mask.iter().filter(|m| **m).count();
            let bytes: Vec<u8> = mask.iter().map(|m| u8::from(*m)).collect();
            let path = dir.join(format!("mask-{parts}-{rank}.bin"));
            fs::write(&path, &bytes).expect("write mask");
            // One warmup, then the timed call. The warmup pays the first
            // fault of this subset.
            let _ = pairs_within(&xyz, &cell, 4.0, Some(&mask), None, false).expect("warm");
            let t1 = Instant::now();
            let got = pairs_within(&xyz, &cell, 4.0, Some(&mask), None, false).expect("pairs");
            let pairs_ns = t1.elapsed().as_nanos() as u64;
            println!(
                "rank parts={parts} rank={rank} owned={owned} active={active} halo_cells={} pairs={} pairs_ns={pairs_ns}",
                ghosts.len(),
                got.len()
            );
        }
    }
}
