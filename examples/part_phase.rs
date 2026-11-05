//! Phase clocks for one RCB of the million-point frame.
//!
//! The cell-count pass is the link-cell bin. The split is
//! [`partition_graph`] on those counts with no pair arcs. The full call
//! is [`partition`], which builds pair arcs only when the method reads
//! them.

use std::time::Instant;

use linkcell::{
    cell_grid, partition, partition_graph, profile_ns, profile_reset, Cell, CellGraph, Method,
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

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env_usize("PAIRS_N", 1_000_000);
    let boxl = std::env::var("PAIRS_BOX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(207.5);
    let parts: u32 = env_usize("PAIRS_PARTS", 8) as u32;
    let xyz = fill(n, boxl);
    let cell = Cell::ortho(boxl, boxl, boxl).expect("box");
    let request = PartRequest {
        parts,
        method: Method::Rcb,
        balance: None,
        device: false,
    };
    let t_count = Instant::now();
    let grid = cell_grid(&xyz, &cell, 4.0, None, None).expect("grid");
    let count_ns = t_count.elapsed().as_nanos() as u64;
    let ncell = grid.ncell();
    let bare = CellGraph {
        grid,
        offsets: vec![0; ncell + 1],
        neighbor: Vec::new(),
        edge_weight: Vec::new(),
    };
    profile_reset();
    let t_rcb = Instant::now();
    let from_counts = partition_graph(&bare, request).expect("rcb");
    let rcb_ns = t_rcb.elapsed().as_nanos() as u64;
    let (select, fm, coarse, pack, repair, check, schedule) = profile_ns();
    let t_full = Instant::now();
    let from_call = partition(&xyz, &cell, 4.0, None, None, request).expect("partition");
    let full_ns = t_full.elapsed().as_nanos() as u64;
    let same = from_counts == from_call;
    println!(
        "n={n} box={boxl} parts={parts} cells={ncell} count_ns={count_ns} rcb_ns={rcb_ns} full_ns={full_ns} same={same}"
    );
    println!(
        "profile select_ns={select} fm_ns={fm} coarse_ns={coarse} pack_ns={pack} repair_ns={repair} check_ns={check} schedule_ns={schedule}"
    );
}
