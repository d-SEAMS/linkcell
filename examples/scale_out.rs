//! One RCB of the million-point frame, then `pairs_within` on each rank.
//!
//! The clock splits into the cell partition, the halo masks, and each
//! rank's pair list. Masks are rewritten so the public main binary times
//! the same subsets.
//!
//! Env: `PAIRS_N`, `PAIRS_BOX`, `PAIRS_PARTS` (one part count), `PAIRS_MASK_DIR`.

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
    let parts = env_usize("PAIRS_PARTS", 8) as u32;
    let dir = PathBuf::from(std::env::var("PAIRS_MASK_DIR").expect("PAIRS_MASK_DIR"));
    fs::create_dir_all(&dir).expect("mask dir");
    let xyz = fill(n, boxl);
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let t_part = Instant::now();
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
    let part_ns = t_part.elapsed().as_nanos() as u64;
    let grid = cell_grid(&xyz, &cell, 4.0, None, None).expect("grid");
    let width = halo_width(4.0, 1, grid.bin_width).expect("width");
    // One bin index per point. Each rank then paints its halo and scans
    // those indices, so the halo does not fold the coordinates again.
    let mut point_cell = vec![0usize; xyz.len()];
    for (i, p) in xyz.iter().enumerate() {
        let s = cell.fractional(*p);
        let ix = bin_coord(s[0], grid.nx) as usize;
        let iy = bin_coord(s[1], grid.ny) as usize;
        let iz = bin_coord(s[2], grid.nz) as usize;
        point_cell[i] = (iz * grid.ny as usize + iy) * grid.nx as usize + ix;
    }
    let t_halo = Instant::now();
    let mut masks = Vec::with_capacity(parts as usize);
    let mut owned_n = Vec::with_capacity(parts as usize);
    let mut halo_n = Vec::with_capacity(parts as usize);
    for rank in 0..parts {
        let ghosts = halo(&grid, &assign, rank, width).expect("halo");
        let mut live = vec![false; grid.ncell()];
        let mut owned = 0usize;
        for (c, &domain) in assign.iter().enumerate() {
            if domain == rank {
                live[c] = true;
            }
        }
        for g in &ghosts {
            live[*g as usize] = true;
        }
        let mut mask = vec![false; xyz.len()];
        for (i, &c) in point_cell.iter().enumerate() {
            if assign[c] == rank {
                owned += 1;
            }
            mask[i] = live[c];
        }
        let bytes: Vec<u8> = mask.iter().map(|m| u8::from(*m)).collect();
        fs::write(dir.join(format!("mask-{parts}-{rank}.bin")), &bytes).expect("mask");
        masks.push(mask);
        owned_n.push(owned);
        halo_n.push(ghosts.len());
    }
    let halo_ns = t_halo.elapsed().as_nanos() as u64;
    println!("parts={parts} part_ns={part_ns} halo_ns={halo_ns} cells={}", grid.ncell());
    for rank in 0..parts {
        let mask = &masks[rank as usize];
        let active = mask.iter().filter(|m| **m).count();
        let _ = pairs_within(&xyz, &cell, 4.0, Some(mask), None, false).expect("warm");
        let t_pairs = Instant::now();
        let got = pairs_within(&xyz, &cell, 4.0, Some(mask), None, false).expect("pairs");
        let pairs_ns = t_pairs.elapsed().as_nanos() as u64;
        println!(
            "rank={rank} owned={} active={active} halo_cells={} pairs={} pairs_ns={pairs_ns}",
            owned_n[rank as usize],
            halo_n[rank as usize],
            got.len()
        );
    }
}
