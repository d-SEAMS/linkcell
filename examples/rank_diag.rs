//! Why a 2-rank slab is slower than the full list: counts, faults, core.

use std::fs;
use std::time::Instant;

use linkcell::{
    cell_grid, halo, halo_width, pairs_within, pairs_within_count, partition, Cell, Method,
    PartRequest,
};

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

fn stat_fields() -> (u64, u64, u64, i32) {
    let raw = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = raw.split_once(')').map(|(_, r)| r).unwrap_or("");
    let bits: Vec<&str> = rest.split_whitespace().collect();
    // after the comm, field 10 minflt is index 8 (fields 3.. start at 0)
    // Fields after `comm`: 0 is state (man field 3). minflt is field 10,
    // majflt field 12, rss field 24, processor field 39.
    let num = |i: usize| bits.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
    (num(7), num(9), num(21), num(36) as i32)
}

fn rss_kb() -> u64 {
    fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn main() {
    let n = 1_000_000usize;
    let boxl = 207.5f64;
    let xyz = fill(n, boxl);
    let cell = Cell::ortho(boxl, boxl, boxl).unwrap();
    let assign = partition(
        &xyz,
        &cell,
        4.0,
        None,
        None,
        PartRequest {
            parts: 2,
            method: Method::Rcb,
            balance: None,
            device: false,
        },
    )
    .unwrap();
    let grid = cell_grid(&xyz, &cell, 4.0, None, None).unwrap();
    let width = halo_width(4.0, 1, grid.bin_width).unwrap();
    let mut point_cell = vec![0usize; n];
    for (i, p) in xyz.iter().enumerate() {
        let s = cell.fractional(*p);
        let ix = bin_coord(s[0], grid.nx) as usize;
        let iy = bin_coord(s[1], grid.ny) as usize;
        let iz = bin_coord(s[2], grid.nz) as usize;
        point_cell[i] = (iz * grid.ny as usize + iy) * grid.nx as usize + ix;
    }
    println!(
        "grid {}x{}x{} width {}x{}x{}",
        grid.nx, grid.ny, grid.nz, width[0], width[1], width[2]
    );

    let time_mask = |label: &str, mask: Option<&[bool]>, pts: &[[f64; 3]], sim: &Cell| {
        let (m0, j0, _, cpu0) = stat_fields();
        let r0 = rss_kb();
        let _ = pairs_within(pts, sim, 4.0, mask, None, false).unwrap();
        let t_c = Instant::now();
        let counted = pairs_within_count(pts, sim, 4.0, mask, None, false).unwrap();
        let count_ns = t_c.elapsed().as_nanos() as u64;
        let (m1, j1, _, _) = stat_fields();
        let t_w = Instant::now();
        let got = pairs_within(pts, sim, 4.0, mask, None, false).unwrap();
        let write_ns = t_w.elapsed().as_nanos() as u64;
        let (m2, j2, _, cpu) = stat_fields();
        let r2 = rss_kb();
        println!(
            "{label} n={} pairs={} count={counted} count_ns={count_ns} write_ns={write_ns} minflt_count={} majflt_count={} minflt_write={} majflt_write={} rss_kb={r0}->{r2} cpu={cpu} cpu0={cpu0} dmin_warm={} dmaj_warm={}",
            pts.len(),
            got.len(),
            m1 - m0,
            j1 - j0,
            m2 - m1,
            j2 - j1,
            m0,
            j0
        );
    };

    time_mask("full", None, &xyz, &cell);

    for rank in 0..2u32 {
        let ghosts = halo(&grid, &assign, rank, width).unwrap();
        let mut live = vec![false; grid.ncell()];
        let mut owned = 0usize;
        for (c, &d) in assign.iter().enumerate() {
            if d == rank {
                live[c] = true;
            }
        }
        for g in &ghosts {
            live[*g as usize] = true;
        }
        let mut mask = vec![false; n];
        let mut compact = Vec::new();
        for (i, &c) in point_cell.iter().enumerate() {
            if assign[c] == rank {
                owned += 1;
            }
            if live[c] {
                mask[i] = true;
                compact.push(xyz[i]);
            }
        }
        let active = mask.iter().filter(|m| **m).count();
        let frac = (active - owned) as f64 / owned.max(1) as f64;
        println!(
            "rank {rank} owned={owned} active={active} halo_frac={frac:.4} halo_cells={} compact={}",
            ghosts.len(),
            compact.len()
        );
        time_mask(&format!("mask{rank}"), Some(&mask), &xyz, &cell);
        time_mask(&format!("compact{rank}"), None, &compact, &cell);
    }
}
