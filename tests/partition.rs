//! Cell partition: one domain per cell, stable under a lattice move
//! and a relabel, and at least the curve cut on a small grid.

use linkcell::{
    cell_graph, cell_grid, curve_split_device, derived_balance, edge_cut, fused_masks, halo,
    halo_width, mesh_cut, partition, score, Cell, CellGraph, CellGrid, Method, PartRequest,
};

fn cloud(n: usize, box_l: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };
    (0..n)
        .map(|_| [next() * box_l, next() * box_l, next() * box_l])
        .collect()
}

fn assert_partition(assign: &[u32], ncell: usize, parts: u32) {
    assert_eq!(assign.len(), ncell);
    let mut seen = vec![0u32; ncell];
    for (cell, &domain) in assign.iter().enumerate() {
        assert!(domain < parts);
        seen[cell] += 1;
    }
    assert!(seen.iter().all(|&c| c == 1));
}

fn methods() -> [Method; 5] {
    [
        Method::Morton,
        Method::Hilbert,
        Method::Rcb,
        Method::HilbertFm,
        Method::RcbFm,
    ]
}

#[test]
fn each_cell_once_and_the_calls_repeat() {
    let sim = Cell::ortho(12.0, 12.0, 12.0).unwrap();
    let xyz = cloud(80, 12.0, 7);
    let graph = cell_graph(&xyz, &sim, 3.0, None, None).unwrap();
    for method in methods() {
        let request = PartRequest {
            parts: 4,
            method,
            balance: None,
            device: false,
        };
        let a = partition(&xyz, &sim, 3.0, None, None, request).unwrap();
        let b = partition(&xyz, &sim, 3.0, None, None, request).unwrap();
        assert_eq!(a, b);
        assert_partition(&a, graph.grid.ncell(), 4);
        let width = halo_width(3.0, 1, graph.grid.bin_width).unwrap();
        let got = score(&graph, &a, 4, width).unwrap();
        assert_eq!(got.cut, edge_cut(&graph, &a));
        assert!(got.imbalance.is_finite());
        assert!(got.halo_fraction.is_finite());
    }
}

#[test]
fn a_lattice_shift_and_a_relabel_keep_the_cells() {
    let sim = Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [2.0, 8.0, 0.0],
        [0.5, -0.4, 9.0],
        [0.2, 0.3, 0.4],
    )
    .unwrap();
    let xyz = cloud(60, 8.0, 11);
    let request = PartRequest::new(3);
    let base = partition(&xyz, &sim, 2.5, None, None, request).unwrap();
    let shift = [10.0 + 2.0 + 0.5, 8.0 - 0.4, 9.0];
    let moved: Vec<[f64; 3]> = xyz
        .iter()
        .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
        .collect();
    let after = partition(&moved, &sim, 2.5, None, None, request).unwrap();
    assert_eq!(base, after);
    let mut order: Vec<usize> = (0..xyz.len()).collect();
    let mut state = 99u64;
    for i in (1..order.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        order.swap(i, j);
    }
    let shuffled: Vec<[f64; 3]> = order.iter().map(|&i| xyz[i]).collect();
    let relabel = partition(&shuffled, &sim, 2.5, None, None, request).unwrap();
    assert_eq!(base, relabel);
}

#[test]
fn halo_radius_is_the_covering_integer() {
    assert_eq!(halo_width(4.0, 2, [2.0, 2.0, 2.0]).unwrap(), [4, 4, 4]);
    assert_eq!(halo_width(3.0, 1, [2.0, 2.0, 2.0]).unwrap(), [2, 2, 2]);
    assert_eq!(halo_width(0.0, 4, [1.0, 1.0, 1.0]).unwrap(), [0, 0, 0]);
    let grid = CellGrid {
        nx: 4,
        ny: 4,
        nz: 1,
        bin_width: [1.0, 1.0, 1.0],
        weight: vec![1; 16],
    };
    let mut assign = vec![0u32; 16];
    for c in 0..16 {
        assign[c] = if c < 8 { 0 } else { 1 };
    }
    let ghosts = halo(&grid, &assign, 0, [0, 1, 0]).unwrap();
    assert!(ghosts.iter().all(|&c| assign[c as usize] == 1));
    assert!(!ghosts.is_empty());
}

#[test]
fn pair_counts_are_the_edge_weights() {
    let sim = Cell::ortho(8.0, 8.0, 8.0).unwrap();
    let xyz = cloud(30, 8.0, 3);
    let graph = cell_graph(&xyz, &sim, 2.5, None, None).unwrap();
    let grid = cell_grid(&xyz, &sim, 2.5, None, None).unwrap();
    assert_eq!(graph.grid.weight, grid.weight);
    assert_eq!(graph.grid.weight.iter().sum::<u64>(), xyz.len() as u64);
    let mut directed = 0u64;
    for w in &graph.edge_weight {
        directed += *w;
    }
    assert_eq!(directed % 2, 0);
}

#[test]
fn the_device_curve_reports_no_device_here() {
    let sim = Cell::ortho(6.0, 6.0, 6.0).unwrap();
    let xyz = cloud(16, 6.0, 1);
    let err = curve_split_device(&xyz, &sim, 2.0, None, None, 2, Method::Morton).unwrap_err();
    assert_eq!(err, linkcell::Error::NoDevice);
}

fn brute_cut(graph: &CellGraph, parts: u32, balance_num: u64, balance_den: u64) -> u64 {
    let n = graph.grid.ncell();
    let total: u64 = graph.grid.weight.iter().sum();
    let mut best = u64::MAX;
    let space = (parts as u64).pow(n as u32);
    for mask in 0..space {
        let mut assign = vec![0u32; n];
        let mut cursor = mask;
        let mut part_w = vec![0u64; parts as usize];
        for cell in 0..n {
            let domain = (cursor % parts as u64) as u32;
            cursor /= parts as u64;
            assign[cell] = domain;
            part_w[domain as usize] += graph.grid.weight[cell];
        }
        let legal = part_w.iter().all(|&w| {
            total == 0
                || (w as u128) * (parts as u128) * (balance_den as u128)
                    <= (total as u128) * (balance_num as u128)
        });
        if !legal {
            continue;
        }
        let cut = edge_cut(graph, &assign);
        if cut < best {
            best = cut;
        }
    }
    best
}

#[test]
fn multilevel_keeps_a_lattice_balanced() {
    let sim = Cell::ortho(12.0, 12.0, 12.0).unwrap();
    let mut xyz = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                xyz.push([x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5]);
            }
        }
    }
    let graph = cell_graph(&xyz, &sim, 1.2, None, None).unwrap();
    let assign = linkcell::partition_graph(
        &graph,
        PartRequest {
            parts: 4,
            method: Method::Multi,
            balance: None,
            device: false,
        },
    )
    .unwrap();
    assert_partition(&assign, graph.grid.ncell(), 4);
    let mut part = vec![0u64; 4];
    for (c, &d) in assign.iter().enumerate() {
        part[d as usize] += graph.grid.weight[c];
    }
    let total: u64 = part.iter().sum();
    let mean = total as f64 / 4.0;
    let imb = part.iter().copied().max().unwrap() as f64 / mean;
    assert!(
        imb < 2.0,
        "imbalance {imb} weights {part:?} ncell {} assign {assign:?}",
        graph.grid.ncell()
    );
}

fn square_graph() -> CellGraph {
    // 2 by 2 by 1. Heavy edges run along x. Light edges run along y.
    let mut offsets = vec![0u32; 5];
    let mut neighbor = Vec::new();
    let mut edge_weight = Vec::new();
    let undirected = [(0u32, 1u32, 5u64), (2, 3, 5), (0, 2, 1), (1, 3, 1)];
    let mut arcs = Vec::new();
    for (a, b, w) in undirected {
        arcs.push((a, b, w));
        arcs.push((b, a, w));
    }
    arcs.sort_unstable();
    for (src, dst, w) in arcs {
        offsets[src as usize + 1] += 1;
        neighbor.push(dst);
        edge_weight.push(w);
    }
    for c in 0..4 {
        offsets[c + 1] += offsets[c];
    }
    CellGraph {
        grid: CellGrid {
            nx: 2,
            ny: 2,
            nz: 1,
            bin_width: [1.0, 1.0, 1.0],
            weight: vec![1, 1, 1, 1],
        },
        offsets,
        neighbor,
        edge_weight,
    }
}

#[test]
fn small_grid_cuts_meet_the_balanced_optimum() {
    let graph = square_graph();
    let (num, den) = derived_balance(&graph.grid.weight, 2);
    let optimal = brute_cut(&graph, 2, num, den);
    assert_eq!(optimal, 2);
    for method in [
        Method::Morton,
        Method::Hilbert,
        Method::HilbertFm,
        Method::Rcb,
        Method::RcbFm,
    ] {
        let assign = linkcell::partition_graph(
            &graph,
            PartRequest {
                parts: 2,
                method,
                balance: None,
                device: false,
            },
        )
        .unwrap();
        assert_partition(&assign, 4, 2);
        assert!(edge_cut(&graph, &assign) >= optimal);
    }
    let morton = linkcell::partition_graph(
        &graph,
        PartRequest {
            parts: 2,
            method: Method::Morton,
            balance: None,
            device: false,
        },
    )
    .unwrap();
    assert_eq!(edge_cut(&graph, &morton), optimal);
    let loose = brute_cut(&graph, 2, 3, 2);
    assert_eq!(loose, 2);
    let refined = linkcell::partition_graph(
        &graph,
        PartRequest {
            parts: 2,
            method: Method::RcbFm,
            balance: Some(1.5),
            device: false,
        },
    )
    .unwrap();
    assert_eq!(edge_cut(&graph, &refined), loose);
    let plain = linkcell::partition_graph(
        &graph,
        PartRequest {
            parts: 2,
            method: Method::Rcb,
            balance: Some(1.5),
            device: false,
        },
    )
    .unwrap();
    assert!(edge_cut(&graph, &refined) <= edge_cut(&graph, &plain));
}

#[test]
fn rcb_on_cell_counts_matches_the_pair_graph() {
    let sim = Cell::ortho(18.0, 18.0, 12.0).unwrap();
    let xyz = cloud(3000, 18.0, 11);
    let graph = cell_graph(&xyz, &sim, 4.0, None, None).unwrap();
    let request = PartRequest {
        parts: 8,
        method: Method::Rcb,
        balance: None,
        device: false,
    };
    let from_graph = linkcell::partition_graph(&graph, request).unwrap();
    let from_counts = partition(&xyz, &sim, 4.0, None, None, request).unwrap();
    assert_eq!(from_counts, from_graph);
    let (num, den) = derived_balance(&graph.grid.weight, 8);
    let total: u64 = graph.grid.weight.iter().sum();
    let cap = (u128::from(total) * u128::from(num)) / (8 * u128::from(den));
    let mut part = vec![0u64; 8];
    for (c, &d) in from_counts.iter().enumerate() {
        part[d as usize] += graph.grid.weight[c];
    }
    assert!(part.iter().all(|&w| u128::from(w) <= cap));
}

#[test]
fn fused_halo_matches_the_cell_halo() {
    let sim = Cell::ortho(18.0, 18.0, 12.0).unwrap();
    let xyz = cloud(3000, 18.0, 11);
    let request = PartRequest {
        parts: 4,
        method: Method::Rcb,
        balance: None,
        device: false,
    };
    let cut = mesh_cut(&xyz, &sim, 4.0, None, None, request).unwrap();
    let plain = partition(&xyz, &sim, 4.0, None, None, request).unwrap();
    assert_eq!(cut.assign, plain);
    let width = halo_width(4.0, 1, cut.grid.bin_width).unwrap();
    let ranks = fused_masks(&cut.grid, &cut.assign, &cut.point_cell, 4, width).unwrap();
    for (rank, view) in ranks.iter().enumerate() {
        let ghosts = halo(&cut.grid, &cut.assign, rank as u32, width).unwrap();
        assert_eq!(view.ghosts, ghosts.len());
        let mut live = vec![false; cut.grid.ncell()];
        for (cell, &domain) in cut.assign.iter().enumerate() {
            if domain == rank as u32 {
                live[cell] = true;
            }
        }
        for ghost in ghosts {
            live[ghost as usize] = true;
        }
        let mut owned = 0usize;
        for (i, &cell) in cut.point_cell.iter().enumerate() {
            assert_eq!(view.mask[i], live[cell as usize]);
            if cut.assign[cell as usize] == rank as u32 {
                owned += 1;
            }
        }
        assert_eq!(view.owned, owned);
    }
}

#[test]
fn caller_weights_move_the_bisection() {
    let nx = 8i32;
    let ny = 4i32;
    let nz = 4i32;
    let ncell = (nx * ny * nz) as usize;
    let mut weight = vec![1u64; ncell];
    // The low-x slab is the expensive one.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..2 {
                let cell = ((iz * ny + iy) * nx + ix) as usize;
                weight[cell] = 40;
            }
        }
    }
    let graph = CellGraph {
        grid: CellGrid {
            nx,
            ny,
            nz,
            bin_width: [1.0; 3],
            weight: weight.clone(),
        },
        offsets: vec![0; ncell + 1],
        neighbor: Vec::new(),
        edge_weight: Vec::new(),
    };
    let assign = linkcell::partition_graph(
        &graph,
        PartRequest {
            parts: 2,
            method: Method::Rcb,
            balance: None,
            device: false,
        },
    )
    .unwrap();
    assert_partition(&assign, ncell, 2);
    let mut load = [0u64; 2];
    for (cell, &domain) in assign.iter().enumerate() {
        load[domain as usize] += weight[cell];
    }
    let mean = (load[0] + load[1]) as f64 / 2.0;
    let imbalance = (load[0].max(load[1]) as f64) / mean;
    let (bound_num, bound_den) = derived_balance(&weight, 2);
    let bound = bound_num as f64 / bound_den as f64;
    assert!(imbalance <= bound + 1e-9, "{imbalance} > {bound}");
    let heavy = assign[0];
    let mut heavy_cells = 0usize;
    let mut light_cells = 0usize;
    for &domain in &assign {
        if domain == heavy {
            heavy_cells += 1;
        } else {
            light_cells += 1;
        }
    }
    assert!(heavy_cells < light_cells);
}

#[test]
fn reach_keeps_the_neighbour_and_drops_the_next_cell() {
    let xyz = vec![[0.2, 0.2, 0.2], [1.2, 0.2, 0.2], [3.2, 0.2, 0.2]];
    let assign = vec![0u32, 1, 1];
    let point_cell = vec![0u32, 1, 2];
    let hit = linkcell::mark_reached(
        &xyz,
        [6.0, 4.0, 4.0],
        [3, 1, 1],
        &assign,
        &point_cell,
        [1, 0, 0],
        0,
        1.5,
    )
    .unwrap();
    assert_eq!(hit[0], 2);
    assert_eq!(hit[1], 1);
    assert_eq!(hit[2], 0);
}
