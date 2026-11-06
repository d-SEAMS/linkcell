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
    cell_grid, halo, halo_width, pairs_within_into, partition, Cell, Method, PartRequest, Pair,
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
    println!(
        "parts={parts} part_ns={part_ns} halo_ns={halo_ns} cells={} cores={}",
        grid.ncell(),
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    );
    // One thread per rank, pinned for both the warmup and the timed
    // call. A fresh thread would allocate again and fault the row buffer.
    let ncores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (mf0, mj0) = faults();
    let t_wall = Instant::now();
    let rows: Vec<_> = std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for rank in 0..parts {
            let mask = &masks[rank as usize];
            let xyz = &xyz;
            let cell = &cell;
            joins.push(scope.spawn(move || {
                pin((rank as usize) % ncores);
                let mut rows = Vec::<Pair>::new();
                let _ = pairs_within_into(xyz, cell, 4.0, Some(mask), None, false, &mut rows);
                let cpu = current_cpu();
                let (f0, j0) = faults();
                let t_pairs = Instant::now();
                pairs_within_into(xyz, cell, 4.0, Some(mask), None, false, &mut rows)
                    .expect("pairs");
                let pairs_ns = t_pairs.elapsed().as_nanos() as u64;
                let (f1, j1) = faults();
                (
                    rank,
                    cpu,
                    rows.len(),
                    pairs_ns,
                    f1.saturating_sub(f0),
                    j1.saturating_sub(j0),
                )
            }));
        }
        joins
            .into_iter()
            .map(|join| join.join().expect("rank"))
            .collect()
    });
    let wall_ns = t_wall.elapsed().as_nanos() as u64;
    let (mf1, mj1) = faults();
    println!(
        "parallel wall_ns={wall_ns} minflt={} majflt={}",
        mf1.saturating_sub(mf0),
        mj1.saturating_sub(mj0)
    );
    for (rank, cpu, pairs, pairs_ns, minflt, majflt) in rows {
        let active = masks[rank as usize].iter().filter(|m| **m).count();
        println!(
            "rank={rank} cpu={cpu} owned={} active={active} halo_cells={} pairs={pairs} pairs_ns={pairs_ns} minflt={minflt} majflt={majflt}",
            owned_n[rank as usize],
            halo_n[rank as usize]
        );
    }
}

fn pin(cpu: usize) {
    let mut set = [0u8; 128];
    if cpu < 1024 {
        set[cpu / 8] |= 1u8 << (cpu % 8);
    }
    unsafe {
        sched_setaffinity(0, set.len(), set.as_ptr());
    }
}

fn current_cpu() -> i32 {
    unsafe { sched_getcpu() }
}

fn faults() -> (u64, u64) {
    let raw = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = raw.split_once(')').map(|(_, r)| r).unwrap_or("");
    let bits: Vec<&str> = rest.split_whitespace().collect();
    let num = |i: usize| bits.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
    (num(7), num(9))
}

extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
    fn sched_getcpu() -> i32;
}
