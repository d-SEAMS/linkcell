//! Domain partition of the periodic cell grid.
//!
//! The mesh is the one [`pairs_within`](crate::pairs_within) builds for
//! the same cutoff. A cell weight is the number of points in that bin.
//! An edge weight is the number of cutoff pairs with one end in each
//! bin. [`partition_graph`] cuts that graph. [`halo`] lists the ghost
//! bins of one part.
//!
//! With `balance: None`, a part may exceed the ideal share by at most
//! the heaviest cell. That is `1 + parts * w_max / total`. The same
//! formula is used for every system. Serial Scotch is the `scotch`
//! feature. Distributed Scotch is the `ptscotch` feature. Both are off
//! unless compiled in.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::OnceLock;
use std::time::Instant;

use crate::bins::{self, flat_cell};
use crate::cell::Cell;
use crate::pairs::{self, pairs_within};
use crate::Error;

/// Smallest advance `Instant` reported on this process.
///
/// The join classifier uses one tick of this clock. The value is the
/// median of 32 observed steps, so it is the host timer, not a constant
/// chosen per system.
pub fn timer_resolution_ns() -> u64 {
    static TICK: OnceLock<u64> = OnceLock::new();
    *TICK.get_or_init(|| {
        let mut samples = [1u64; 32];
        for slot in &mut samples {
            let start = Instant::now();
            let mut next = Instant::now();
            let mut spins = 0u32;
            while next <= start && spins < 1_000_000 {
                next = Instant::now();
                spins += 1;
            }
            *slot = next.saturating_duration_since(start).as_nanos() as u64;
            if *slot == 0 {
                *slot = 1;
            }
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    })
}

/// Balance allowed by one indivisible cell.
///
/// A weighted median gives the straddling cell to one part. That part
/// may therefore exceed the ideal share `total / parts` by at most the
/// heaviest cell `w_max`. The bound is `(total + parts * w_max) / total`,
/// which is `1 + parts * w_max / total`. No cell-aligned cut can promise
/// a tighter bound. `lc_partition` selects it when `balance <= 0`, which
/// is the same per-frame rule the d-SEAMS domain driver derives.
/// The same formula is used for every system. `balance: None` selects it.
pub fn derived_balance(weight: &[u64], parts: u32) -> (u64, u64) {
    let total: u64 = weight.iter().copied().sum();
    if total == 0 || parts == 0 {
        return (1, 1);
    }
    let w_max = weight.iter().copied().max().unwrap_or(0);
    let extra = u64::from(parts).saturating_mul(w_max);
    (total.saturating_add(extra), total)
}

/// The derived bound as a real ratio, for drivers that take a float.
pub fn derived_balance_f64(weight: &[u64], parts: u32) -> f64 {
    let (num, den) = derived_balance(weight, parts);
    if den == 0 {
        1.0
    } else {
        num as f64 / den as f64
    }
}

/// How [`partition_graph`] cuts the cell graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Method {
    /// Morton order, then a weighted prefix split.
    Morton = 0,
    /// Hilbert order, then a weighted prefix split.
    Hilbert = 1,
    /// Weighted recursive coordinate bisection.
    Rcb = 2,
    /// Hilbert prefix split, then Fiduccia-Mattheyses.
    HilbertFm = 3,
    /// Recursive coordinate bisection, then Fiduccia-Mattheyses.
    RcbFm = 4,
    /// Serial Scotch `SCOTCH_graphPart`, feature `scotch` only.
    Scotch = 5,
    /// Heavy-edge coarsening, coarse bisection, refinement on the way back.
    Multi = 6,
    /// PT-Scotch `SCOTCH_dgraphPart`, feature `ptscotch` only.
    PtScotch = 7,
}

impl Method {
    /// True when the initial cut is a curve prefix.
    pub fn uses_curve(self) -> bool {
        matches!(self, Method::Morton | Method::Hilbert | Method::HilbertFm)
    }

    /// True when a Fiduccia-Mattheyses pass follows the initial cut.
    pub fn refines(self) -> bool {
        matches!(self, Method::HilbertFm | Method::RcbFm)
    }

    /// True when the cut reads pair-count arcs.
    ///
    /// A plain recursive coordinate bisection and a curve prefix use the
    /// link-cell point counts only. Refinement, coarsening, and Scotch
    /// walk the arcs, so those methods still build them.
    fn needs_edges(self) -> bool {
        matches!(
            self,
            Method::HilbertFm | Method::RcbFm | Method::Scotch | Method::PtScotch | Method::Multi
        )
    }

    fn from_code(code: u8) -> Result<Self, Error> {
        match code {
            0 => Ok(Method::Morton),
            1 => Ok(Method::Hilbert),
            2 => Ok(Method::Rcb),
            3 => Ok(Method::HilbertFm),
            4 => Ok(Method::RcbFm),
            5 => Ok(Method::Scotch),
            6 => Ok(Method::Multi),
            7 => Ok(Method::PtScotch),
            _ => Err(Error::BadPartition),
        }
    }
}

/// Default method.
///
/// The panel sums edge cuts over the five systems. The default is the
/// native method with the smallest sum among those that stay within
/// the balance bound on every system. A tie breaks toward the smaller
/// sum of median wall times.
pub const DEFAULT_METHOD: Method = Method::RcbFm;

/// One call to [`partition`] or [`partition_graph`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartRequest {
    /// Number of domains. Must be at least 1.
    pub parts: u32,
    /// Cut to run.
    pub method: Method,
    /// `None` selects [`derived_balance`] for these cell weights.
    pub balance: Option<f64>,
    /// Run weights, curve keys, and the prefix split on the device.
    pub device: bool,
}

impl PartRequest {
    /// `parts` domains, [`DEFAULT_METHOD`], the default balance, on the host.
    pub fn new(parts: u32) -> Self {
        Self {
            parts,
            method: DEFAULT_METHOD,
            balance: None,
            device: false,
        }
    }
}

/// Fractional bins and the point count in each bin.
#[derive(Clone, Debug, PartialEq)]
pub struct CellGrid {
    /// Bins along the first lattice axis.
    pub nx: i32,
    /// Bins along the second lattice axis.
    pub ny: i32,
    /// Bins along the third lattice axis.
    pub nz: i32,
    /// Perpendicular width of one bin on each axis.
    pub bin_width: [f64; 3],
    /// Point count of cell `((iz * ny) + iy) * nx + ix`.
    pub weight: Vec<u64>,
}

impl CellGrid {
    /// Number of bins.
    pub fn ncell(&self) -> usize {
        self.weight.len()
    }
}

/// Place `copies` disjoint copies of `graph` side by side along x.
///
/// Cell count and arc count both equal the base counts times `copies`.
/// Each copy keeps the base neighbourhood, so the work per copy stays
/// the base cell count and the base arc count.
pub fn replicate_graph(graph: &CellGraph, copies: usize) -> Result<CellGraph, Error> {
    if copies == 0 {
        return Err(Error::BadGraph);
    }
    if copies == 1 {
        return Ok(graph.clone());
    }
    let nx = graph.grid.nx;
    let ny = graph.grid.ny;
    let nz = graph.grid.nz;
    if nx <= 0 || ny <= 0 || nz <= 0 {
        return Err(Error::BadGraph);
    }
    let n = graph.grid.ncell();
    let expect = (nx as usize)
        .checked_mul(ny as usize)
        .and_then(|v| v.checked_mul(nz as usize))
        .ok_or(Error::BadGraph)?;
    if expect != n || graph.offsets.len() != n + 1 {
        return Err(Error::BadGraph);
    }
    let copies_i = i32::try_from(copies).map_err(|_| Error::BadGraph)?;
    let big_nx = nx.checked_mul(copies_i).ok_or(Error::BadGraph)?;
    let big_n = n.checked_mul(copies).ok_or(Error::BadGraph)?;
    let nx_u = nx as usize;
    let ny_u = ny as usize;
    let big_nx_u = big_nx as usize;
    let mut weight = vec![0u64; big_n];
    for copy in 0..copies {
        for iz in 0..nz as usize {
            for iy in 0..ny_u {
                for ix in 0..nx_u {
                    let src = ((iz * ny_u) + iy) * nx_u + ix;
                    let dst = ((iz * ny_u) + iy) * big_nx_u + copy * nx_u + ix;
                    weight[dst] = graph.grid.weight[src];
                }
            }
        }
    }
    let mut neighbor = Vec::with_capacity(graph.neighbor.len().saturating_mul(copies));
    let mut edge_weight = Vec::with_capacity(graph.edge_weight.len().saturating_mul(copies));
    let mut offsets = vec![0u32; big_n + 1];
    for dst in 0..big_n {
        let plane = big_nx_u * ny_u;
        let iz = dst / plane;
        let rem = dst % plane;
        let iy = rem / big_nx_u;
        let ix_big = rem % big_nx_u;
        let copy = ix_big / nx_u;
        let ix = ix_big % nx_u;
        let src = ((iz * ny_u) + iy) * nx_u + ix;
        let lo = graph.offsets[src] as usize;
        let hi = graph.offsets[src + 1] as usize;
        for edge in lo..hi {
            let nbr = graph.neighbor[edge] as usize;
            if nbr >= n {
                return Err(Error::BadGraph);
            }
            let nplane = nx_u * ny_u;
            let niz = nbr / nplane;
            let nrem = nbr % nplane;
            let niy = nrem / nx_u;
            let nix = nrem % nx_u;
            let mapped = ((niz * ny_u) + niy) * big_nx_u + copy * nx_u + nix;
            neighbor.push(mapped as u32);
            edge_weight.push(graph.edge_weight[edge]);
        }
        offsets[dst + 1] = u32::try_from(neighbor.len()).map_err(|_| Error::BadGraph)?;
    }
    Ok(CellGraph {
        grid: CellGrid {
            nx: big_nx,
            ny,
            nz,
            bin_width: graph.grid.bin_width,
            weight,
        },
        offsets,
        neighbor,
        edge_weight,
    })
}

/// Periodic cell adjacency with pair-count edge weights.
///
/// Each undirected pair is stored as two arcs of the same weight.
/// A cut counts each undirected pair once.
#[derive(Clone, Debug, PartialEq)]
pub struct CellGraph {
    /// Bin counts and point weights.
    pub grid: CellGrid,
    /// CSR offsets, length `ncell + 1`.
    pub offsets: Vec<u32>,
    /// Neighbour cell of each arc.
    pub neighbor: Vec<u32>,
    /// Pair count of each arc.
    pub edge_weight: Vec<u64>,
}

/// Cut, imbalance, and halo fraction of one assignment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartScore {
    /// Sum of undirected edge weights whose ends lie in different parts.
    pub cut: u64,
    /// Heaviest part weight divided by the mean part weight.
    /// Equal empty parts score `1`.
    pub imbalance: f64,
    /// Largest `ghost / owned` ratio over parts that own a cell.
    /// A part that owns nothing is skipped. No owned cell scores `0`.
    pub halo_fraction: f64,
}

/// Bin radii that cover a length of `cutoff * depth`.
///
/// Axis `i` gets the smallest integer `r` with `r * bin_width[i]` at
/// least that length. `depth` is the number of cutoff shells the
/// caller must cover. Zero cutoff or zero depth is a zero radius.
pub fn halo_width(cutoff: f64, depth: u32, bin_width: [f64; 3]) -> Result<[u32; 3], Error> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(Error::BadCutoff);
    }
    let length = cutoff * f64::from(depth);
    if !length.is_finite() {
        return Err(Error::BadCutoff);
    }
    let mut out = [0u32; 3];
    for i in 0..3 {
        out[i] = bins_needed(length, bin_width[i])?;
    }
    Ok(out)
}

/// Ghost cells of `domain`.
///
/// `width` is a bin radius on each axis, from [`halo_width`] or from
/// the caller. The result is sorted and does not include cells whose
/// assignment equals `domain`.
pub fn halo(
    grid: &CellGrid,
    assign: &[u32],
    domain: u32,
    width: [u32; 3],
) -> Result<Vec<u32>, Error> {
    let n = check_assign(grid, assign)?;
    let mut mark = vec![0u8; n];
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    for c in 0..n {
        if assign[c] != domain {
            continue;
        }
        let (ix, iy, iz) = unravel(c, nx, ny);
        paint(&mut mark, nx, ny, nz, ix, iy, iz, width);
    }
    let mut ghosts = Vec::new();
    for c in 0..n {
        if mark[c] != 0 && assign[c] != domain {
            ghosts.push(c as u32);
        }
    }
    Ok(ghosts)
}

/// Point counts on the mesh `pairs_within` would use.
pub fn cell_grid(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
) -> Result<CellGrid, Error> {
    prepared(xyz, sim, cutoff, mask, cell_hint, false).map(|p| p.graph.grid)
}

/// Cell grid plus pair-count arcs from [`pairs_within`].
pub fn cell_graph(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
) -> Result<CellGraph, Error> {
    prepared(xyz, sim, cutoff, mask, cell_hint, true).map(|p| p.graph)
}

/// Domain id of each cell. See [`PartRequest`].
pub fn partition(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    request: PartRequest,
) -> Result<Vec<u32>, Error> {
    // The pair list is a full sort of every cutoff arc. A method that
    // only reads cell counts must not build it: on the million-point
    // frame that list is the whole call, and the bisection is the
    // median of the link-cell bins.
    let prep = prepared(
        xyz,
        sim,
        cutoff,
        mask,
        cell_hint,
        request.method.needs_edges(),
    )?;
    partition_prepared(&prep, request)
}

/// Partition an existing graph. `device` needs the point bins, so it
/// returns [`Error::NoDevice`] here. Build the assignment from points
/// with [`partition`] when the device path is required.
pub fn partition_graph(graph: &CellGraph, request: PartRequest) -> Result<Vec<u32>, Error> {
    if request.device {
        return Err(Error::NoDevice);
    }
    let t_check = Instant::now();
    check_graph(graph)?;
    CHECK_NS.fetch_add(t_check.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    let t_ratio = Instant::now();
    let ratio = balance_ratio(request.balance, &graph.grid.weight, request.parts)?;
    PACK_NS.fetch_add(t_ratio.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    if request.parts == 0 {
        return Err(Error::BadPartition);
    }
    dispatch(graph, request.method, request.parts, ratio, None)
}

/// Curve split on the device, including the copies both ways.
///
/// The result is the prefix assignment, with no refinement pass.
/// [`Error::NoDevice`] means the driver is absent. A mismatch with the
/// host histogram is [`Error::DeviceMismatch`].
pub fn curve_split_device(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    parts: u32,
    method: Method,
) -> Result<Vec<u32>, Error> {
    if !method.uses_curve() {
        return Err(Error::NoDevice);
    }
    if parts == 0 {
        return Err(Error::BadPartition);
    }
    let prep = prepared(xyz, sim, cutoff, mask, cell_hint, false)?;
    device_curve(&prep, parts, method)
}

/// Cut, imbalance, and halo fraction.
pub fn score(
    graph: &CellGraph,
    assign: &[u32],
    parts: u32,
    width: [u32; 3],
) -> Result<PartScore, Error> {
    check_graph(graph)?;
    check_assign(&graph.grid, assign)?;
    if parts == 0 {
        return Err(Error::BadPartition);
    }
    for &d in assign {
        if d >= parts {
            return Err(Error::BadPartition);
        }
    }
    Ok(PartScore {
        cut: edge_cut(graph, assign),
        imbalance: max_imbalance(&graph.grid.weight, assign, parts),
        halo_fraction: halo_fraction(&graph.grid, assign, parts, width)?,
    })
}

/// Undirected edge cut.
pub fn edge_cut(graph: &CellGraph, assign: &[u32]) -> u64 {
    let mut cut = 0u64;
    let n = graph.grid.ncell();
    for c in 0..n {
        let a = assign[c];
        let lo = graph.offsets[c] as usize;
        let hi = graph.offsets[c + 1] as usize;
        for e in lo..hi {
            let nbr = graph.neighbor[e] as usize;
            if nbr > c && assign[nbr] != a {
                cut += graph.edge_weight[e];
            }
        }
    }
    cut
}

struct Prepared {
    graph: CellGraph,
    /// Flat bin of each active point, in point order with masked points omitted.
    active_bins: Vec<u32>,
    /// Bin of every point. A masked point keeps bin 0.
    point_cell: Vec<u32>,
}

/// Point counts on the link-cell grid. Threads own disjoint point ranges
/// and private histograms, then the histograms add. The active-bin list
/// stays in point order.
fn bin_points(
    xyz: &[[f64; 3]],
    sim: &Cell,
    mask: Option<&[bool]>,
    dims: [i32; 3],
    ncell: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u32>) {
    let n = xyz.len();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads = if n >= 8192 { threads.max(1) } else { 1 };
    let chunk = (n + threads - 1) / threads;
    let mut bin_of = vec![0u32; n];
    let mut weight = vec![0u64; ncell];
    let mut hits: Vec<(usize, Vec<(usize, u32)>, Vec<u64>)> = Vec::new();
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for t in 0..threads {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            if lo >= hi {
                continue;
            }
            joins.push(scope.spawn(move || {
                let mut local = vec![0u64; ncell];
                let mut active = Vec::new();
                for i in lo..hi {
                    if mask.is_some_and(|m| !m[i]) {
                        continue;
                    }
                    let (_, b) = bins::fold_point(sim, xyz[i], dims);
                    let c = flat_cell(b, dims) as u32;
                    local[c as usize] += 1;
                    active.push((i, c));
                }
                (t, active, local)
            }));
        }
        for join in joins {
            hits.push(join.join().expect("bin"));
        }
    });
    hits.sort_by_key(|(t, _, _)| *t);
    let mut active_bins = Vec::new();
    for (_, active, local) in hits {
        for (c, w) in local.into_iter().enumerate() {
            weight[c] += w;
        }
        for (i, c) in active {
            bin_of[i] = c;
            active_bins.push(c);
        }
    }
    (weight, active_bins, bin_of)
}

fn prepared(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    with_edges: bool,
) -> Result<Prepared, Error> {
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(Error::BadCutoff);
    }
    if xyz.is_empty() {
        return Err(Error::Empty);
    }
    if let Some(m) = mask {
        if m.len() != xyz.len() {
            return Err(Error::MaskLen);
        }
    }
    let mut n_act = 0usize;
    for (i, p) in xyz.iter().enumerate() {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return Err(Error::NonFinite);
        }
        if mask.is_some_and(|m| !m[i]) {
            continue;
        }
        n_act += 1;
    }
    let edge = bins::target_edge(sim, cell_hint, pairs::default_edge(sim, n_act, cutoff));
    let dims = bins::grid_dims(sim, edge)?;
    let ncell = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    let box_w = sim.widths();
    let bin_width = [
        box_w[0] / f64::from(dims[0]),
        box_w[1] / f64::from(dims[1]),
        box_w[2] / f64::from(dims[2]),
    ];
    let (weight, active_bins, bin_of) = bin_points(xyz, sim, mask, dims, ncell);
    let (offsets, neighbor, edge_weight) = if with_edges {
        arcs_from_pairs(xyz, sim, cutoff, mask, cell_hint, &bin_of, ncell)?
    } else {
        (vec![0u32; ncell + 1], Vec::new(), Vec::new())
    };
    Ok(Prepared {
        graph: CellGraph {
            grid: CellGrid {
                nx: dims[0],
                ny: dims[1],
                nz: dims[2],
                bin_width,
                weight,
            },
            offsets,
            neighbor,
            edge_weight,
        },
        active_bins,
        point_cell: bin_of,
    })
}

/// Assignment, the grid it was cut on, and the bin of every point.
///
/// One binning feeds the cut. A later frame that keeps this cut does not
/// bin again.
pub struct MeshCut {
    /// Domain of each cell.
    pub assign: Vec<u32>,
    /// Bins and point weights.
    pub grid: CellGrid,
    /// Bin of each point, in point order.
    pub point_cell: Vec<u32>,
}

/// Cut the mesh once and keep the point bins.
pub fn mesh_cut(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    request: PartRequest,
) -> Result<MeshCut, Error> {
    let prep = prepared(
        xyz,
        sim,
        cutoff,
        mask,
        cell_hint,
        request.method.needs_edges(),
    )?;
    let assign = partition_prepared(&prep, request)?;
    Ok(MeshCut {
        assign,
        grid: prep.graph.grid,
        point_cell: prep.point_cell,
    })
}

/// Point mask of one rank after its halo is painted.
pub struct RankMask {
    /// Points whose bin is owned or lies in the halo.
    pub mask: Vec<bool>,
    /// Points in cells this rank owns.
    pub owned: usize,
    /// Ghost cells, the painted bins this rank does not own.
    pub ghosts: usize,
}

/// Paint each rank's halo while walking the cells that rank owns.
///
/// The owned lists come from one shared pass over the assignment. Each
/// rank then paints its stencil from those cells and reads the point
/// bins already stored on the cut. The ghost count matches [`halo`].
pub fn fused_masks(
    grid: &CellGrid,
    assign: &[u32],
    point_cell: &[u32],
    parts: u32,
    width: [u32; 3],
) -> Result<Vec<RankMask>, Error> {
    let n = check_assign(grid, assign)?;
    if parts == 0 {
        return Err(Error::BadPartition);
    }
    for &domain in assign {
        if domain >= parts {
            return Err(Error::BadPartition);
        }
    }
    for &cell in point_cell {
        if cell as usize >= n {
            return Err(Error::BadPartition);
        }
    }
    let owned = owned_cells(assign, parts);
    let built = std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for rank in 0..parts {
            let cells = &owned[rank as usize];
            joins
                .push(scope.spawn(move || rank_mask(grid, assign, point_cell, cells, rank, width)));
        }
        joins
            .into_iter()
            .map(|join| join.join().expect("rank mask"))
            .collect()
    });
    Ok(built)
}

/// Cells of each domain, in cell order.
pub fn owned_cells(assign: &[u32], parts: u32) -> Vec<Vec<u32>> {
    let n = assign.len();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads = if n >= 8192 { threads.max(1) } else { 1 };
    let chunk = (n + threads - 1) / threads;
    let mut hits: Vec<(usize, Vec<Vec<u32>>)> = Vec::new();
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for t in 0..threads {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            if lo >= hi {
                continue;
            }
            joins.push(scope.spawn(move || {
                let mut rows = vec![Vec::new(); parts as usize];
                for cell in lo..hi {
                    rows[assign[cell] as usize].push(cell as u32);
                }
                (t, rows)
            }));
        }
        for join in joins {
            hits.push(join.join().expect("owned"));
        }
    });
    hits.sort_by_key(|(t, _)| *t);
    let mut owned = vec![Vec::new(); parts as usize];
    for (_, rows) in hits {
        for (rank, cells) in rows.into_iter().enumerate() {
            owned[rank].extend(cells);
        }
    }
    owned
}

/// Halo and point mask for one rank, from that rank's own cells.
///
/// The stencil is painted in the same walk that visits those cells.
pub fn rank_mask(
    grid: &CellGrid,
    assign: &[u32],
    point_cell: &[u32],
    cells: &[u32],
    rank: u32,
    width: [u32; 3],
) -> RankMask {
    let n = grid.ncell();
    let mut mark = vec![0u8; n];
    for &cell in cells {
        let (ix, iy, iz) = unravel(cell as usize, grid.nx, grid.ny);
        paint(&mut mark, grid.nx, grid.ny, grid.nz, ix, iy, iz, width);
    }
    let mut ghosts = 0usize;
    for (cell, &flag) in mark.iter().enumerate() {
        if flag != 0 && assign[cell] != rank {
            ghosts += 1;
        }
    }
    let mut mask = vec![false; point_cell.len()];
    let mut owned_pts = 0usize;
    for (i, &cell) in point_cell.iter().enumerate() {
        let cell = cell as usize;
        if assign[cell] == rank {
            owned_pts += 1;
        }
        mask[i] = mark[cell] != 0;
    }
    RankMask {
        mask,
        owned: owned_pts,
        ghosts,
    }
}

fn arcs_from_pairs(
    xyz: &[[f64; 3]],
    sim: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    bin_of: &[u32],
    ncell: usize,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u64>), Error> {
    let pairs = pairs_within(xyz, sim, cutoff, mask, cell_hint, true)?;
    let mut ends: Vec<(u32, u32)> = Vec::new();
    for pair in pairs {
        let a = bin_of[pair.i];
        let b = bin_of[pair.j];
        if a == b {
            continue;
        }
        if a < b {
            ends.push((a, b));
        } else {
            ends.push((b, a));
        }
    }
    ends.sort_unstable();
    let mut undirected: Vec<(u32, u32, u64)> = Vec::new();
    let mut cursor = 0;
    while cursor < ends.len() {
        let (a, b) = ends[cursor];
        let mut count = 1u64;
        cursor += 1;
        while cursor < ends.len() && ends[cursor] == (a, b) {
            count += 1;
            cursor += 1;
        }
        undirected.push((a, b, count));
    }
    let mut arcs: Vec<(u32, u32, u64)> = Vec::with_capacity(undirected.len() * 2);
    for (a, b, w) in undirected {
        arcs.push((a, b, w));
        arcs.push((b, a, w));
    }
    arcs.sort_unstable();
    if arcs.len() > u32::MAX as usize {
        return Err(Error::Overflow);
    }
    let mut offsets = vec![0u32; ncell + 1];
    let mut neighbor = Vec::with_capacity(arcs.len());
    let mut edge_weight = Vec::with_capacity(arcs.len());
    for (src, dst, w) in arcs {
        offsets[src as usize + 1] += 1;
        neighbor.push(dst);
        edge_weight.push(w);
    }
    for c in 0..ncell {
        offsets[c + 1] += offsets[c];
    }
    Ok((offsets, neighbor, edge_weight))
}

fn partition_prepared(prep: &Prepared, request: PartRequest) -> Result<Vec<u32>, Error> {
    if request.parts == 0 {
        return Err(Error::BadPartition);
    }
    let ratio = balance_ratio(request.balance, &prep.graph.grid.weight, request.parts)?;
    let device_assign = if request.device {
        if !request.method.uses_curve() {
            return Err(Error::NoDevice);
        }
        Some(device_curve(prep, request.parts, request.method)?)
    } else {
        None
    };
    dispatch(
        &prep.graph,
        request.method,
        request.parts,
        ratio,
        device_assign,
    )
}

/// Largest weight one contracted vertex may have.
///
/// [`part_cap`] is the most weight `under_balance` allows one part to
/// hold. With the derived ratio that cap is the ideal share plus the
/// heaviest fine cell, so the stop is tied to the part count and
/// `w_max`. A coarse vertex above the cap cannot sit in any part, and
/// matching refuses a pair that would cross it.
fn cluster_cap(total: u64, parts: u32, w_max: u64, ratio: (u64, u64)) -> u64 {
    part_cap(total, parts, ratio).max(w_max).max(1)
}

fn multi_assign(graph: &CellGraph, parts: u32, ratio: (u64, u64)) -> Vec<u32> {
    let n = graph.grid.ncell();
    let t_pack = Instant::now();
    let mut coords = Vec::with_capacity(n);
    for c in 0..n {
        let (ix, iy, iz) = unravel(c, graph.grid.nx, graph.grid.ny);
        coords.push((f64::from(ix), f64::from(iy), f64::from(iz)));
    }
    PACK_NS.fetch_add(t_pack.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    // Geometric bisection is the incumbent. A later cycle replaces it
    // only when the integer cut falls and every part stays inside the
    // balance bound. The cut cannot fall forever, so the loop ends when
    // a cycle brings no gain.
    let mut best = rcb_assign(graph, parts, true, ratio);
    let t_cut = Instant::now();
    let mut best_cut = edge_cut(graph, &best);
    COARSE_NS.fetch_add(t_cut.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    loop {
        let next = v_cycle(graph, &coords, &best, parts, ratio);
        let t_cut = Instant::now();
        let next_cut = edge_cut(graph, &next);
        let keep = next_cut < best_cut && holds_balance(&graph.grid.weight, &next, parts, ratio);
        COARSE_NS.fetch_add(t_cut.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        if keep {
            best = next;
            best_cut = next_cut;
        } else {
            break;
        }
    }
    best
}

fn holds_balance(weight: &[u64], assign: &[u32], parts: u32, ratio: (u64, u64)) -> bool {
    if parts == 0 {
        return false;
    }
    let mut part_w = vec![0u64; parts as usize];
    let mut total = 0u64;
    for (c, &d) in assign.iter().enumerate() {
        if d >= parts {
            return false;
        }
        part_w[d as usize] += weight[c];
        total += weight[c];
    }
    part_w
        .iter()
        .all(|&w| under_balance(w, total, parts, ratio))
}

/// One uncoarsening pass. Matching stays inside the current parts, so the
/// projected partition is the incumbent until refinement moves a cluster.
fn v_cycle(
    graph: &CellGraph,
    coords: &[(f64, f64, f64)],
    assign: &[u32],
    parts: u32,
    ratio: (u64, u64),
) -> Vec<u32> {
    let n = graph.grid.ncell();
    let t_prep = Instant::now();
    let total: u64 = graph.grid.weight.iter().copied().sum();
    let w_max = graph.grid.weight.iter().copied().max().unwrap_or(0);
    let cap = cluster_cap(total, parts, w_max, ratio);
    COARSE_NS.fetch_add(t_prep.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    // One vertex per part is already a legal assignment. A vertex above
    // the cap cannot be paired without producing an unsplittable part.
    if n <= parts as usize || graph.grid.weight.iter().any(|&w| w > cap) {
        let mut out = assign.to_vec();
        refine(&mut out, graph, parts, ratio);
        return out;
    }
    let t0 = Instant::now();
    let mate = heavy_match(graph, cap, Some(assign));
    let shrunk = mate.iter().enumerate().any(|(v, &m)| m != v as u32);
    if !shrunk {
        COARSE_NS.fetch_add(t0.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        let mut out = assign.to_vec();
        refine(&mut out, graph, parts, ratio);
        return out;
    }
    let (coarse, coarse_coords, owner) = contract(graph, coords, &mate);
    COARSE_NS.fetch_add(t0.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    let mut up = vec![0u32; coarse.grid.ncell()];
    for (fine, &cell) in owner.iter().enumerate() {
        up[cell as usize] = assign[fine];
    }
    let up = v_cycle(&coarse, &coarse_coords, &up, parts, ratio);
    let t_down = Instant::now();
    let mut out = vec![0u32; n];
    for (fine, &cell) in owner.iter().enumerate() {
        out[fine] = up[cell as usize];
    }
    COARSE_NS.fetch_add(t_down.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    refine(&mut out, graph, parts, ratio);
    out
}

fn heavy_match(graph: &CellGraph, cap: u64, same: Option<&[u32]>) -> Vec<u32> {
    let n = graph.grid.ncell();
    let mut mate = vec![u32::MAX; n];
    for v in 0..n {
        if mate[v] != u32::MAX {
            continue;
        }
        let wv = graph.grid.weight[v];
        // Prefer a heavy edge between light vertices: compare edge/combined.
        let mut best: Option<(u64, u64, u32)> = None;
        let lo = graph.offsets[v] as usize;
        let hi = graph.offsets[v + 1] as usize;
        for e in lo..hi {
            let u = graph.neighbor[e];
            if u as usize == v || mate[u as usize] != u32::MAX {
                continue;
            }
            let wu = graph.grid.weight[u as usize];
            if wv.saturating_add(wu) > cap {
                continue;
            }
            if let Some(parts_of) = same {
                if parts_of[u as usize] != parts_of[v] {
                    continue;
                }
            }
            let w = graph.edge_weight[e];
            let denom = wv.max(1).saturating_add(wu.max(1));
            let take = match best {
                None => true,
                Some((bw, bden, bu)) => {
                    let left = u128::from(w) * u128::from(bden);
                    let right = u128::from(bw) * u128::from(denom);
                    left > right || (left == right && u < bu)
                }
            };
            if take {
                best = Some((w, denom, u));
            }
        }
        if let Some((_, _, u)) = best {
            mate[v] = u;
            mate[u as usize] = v as u32;
        } else {
            mate[v] = v as u32;
        }
    }
    mate
}

fn contract(
    graph: &CellGraph,
    coords: &[(f64, f64, f64)],
    mate: &[u32],
) -> (CellGraph, Vec<(f64, f64, f64)>, Vec<u32>) {
    let n = graph.grid.ncell();
    let mut coarse_of = vec![u32::MAX; n];
    let mut members: Vec<Vec<u32>> = Vec::new();
    for v in 0..n {
        let lead = (v as u32).min(mate[v]) as usize;
        if coarse_of[lead] == u32::MAX {
            coarse_of[lead] = members.len() as u32;
            members.push(Vec::new());
        }
        let id = coarse_of[lead];
        coarse_of[v] = id;
        members[id as usize].push(v as u32);
    }
    let nc = members.len();
    let mut weight = vec![0u64; nc];
    let mut ccoords = vec![(0.0, 0.0, 0.0); nc];
    for (c, group) in members.iter().enumerate() {
        let mut mass = 0u64;
        let mut acc = (0.0, 0.0, 0.0);
        for &v in group {
            let w = graph.grid.weight[v as usize];
            mass += w;
            let p = coords[v as usize];
            acc.0 += p.0 * w as f64;
            acc.1 += p.1 * w as f64;
            acc.2 += p.2 * w as f64;
            weight[c] += w;
        }
        if mass == 0 {
            let p = coords[group[0] as usize];
            ccoords[c] = p;
        } else {
            let mass = mass as f64;
            ccoords[c] = (acc.0 / mass, acc.1 / mass, acc.2 / mass);
        }
    }
    let mut ends: Vec<(u32, u32, u64)> = Vec::new();
    for v in 0..n {
        let cv = coarse_of[v];
        let lo = graph.offsets[v] as usize;
        let hi = graph.offsets[v + 1] as usize;
        for e in lo..hi {
            let u = graph.neighbor[e] as usize;
            if u <= v {
                continue;
            }
            let cu = coarse_of[u];
            if cu == cv {
                continue;
            }
            let (a, b) = if cv < cu { (cv, cu) } else { (cu, cv) };
            ends.push((a, b, graph.edge_weight[e]));
        }
    }
    ends.sort_unstable_by(|p, q| p.0.cmp(&q.0).then(p.1.cmp(&q.1)));
    let mut arcs: Vec<(u32, u32, u64)> = Vec::new();
    let mut i = 0;
    while i < ends.len() {
        let (a, b, mut w) = ends[i];
        i += 1;
        while i < ends.len() && ends[i].0 == a && ends[i].1 == b {
            w += ends[i].2;
            i += 1;
        }
        // Each fine arc was stored once per direction, and the pair was
        // ordered, so the merged weight is the undirected total. Emit both.
        arcs.push((a, b, w));
        arcs.push((b, a, w));
    }
    arcs.sort_unstable_by(|p, q| p.0.cmp(&q.0).then(p.1.cmp(&q.1)));
    let mut offsets = vec![0u32; nc + 1];
    let mut neighbor = Vec::with_capacity(arcs.len());
    let mut edge_weight = Vec::with_capacity(arcs.len());
    for (src, dst, w) in arcs {
        offsets[src as usize + 1] += 1;
        neighbor.push(dst);
        edge_weight.push(w);
    }
    for c in 0..nc {
        offsets[c + 1] += offsets[c];
    }
    let coarse = CellGraph {
        grid: CellGrid {
            nx: nc as i32,
            ny: 1,
            nz: 1,
            bin_width: [1.0, 1.0, 1.0],
            weight,
        },
        offsets,
        neighbor,
        edge_weight,
    };
    (coarse, ccoords, coarse_of)
}

fn rcb_on(
    graph: &CellGraph,
    coords: &[(f64, f64, f64)],
    parts: u32,
    refine_cuts: bool,
    ratio: (u64, u64),
) -> Vec<u32> {
    let n = graph.grid.ncell();
    let t_pack = Instant::now();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for (i, p) in coords.iter().enumerate() {
        x[i] = p.0;
        y[i] = p.1;
        z[i] = p.2;
    }
    let geom = RcbGeom {
        x,
        y,
        z,
        w: graph.grid.weight.clone(),
    };
    let mut ids: Vec<u32> = (0..n as u32).collect();
    let mut out = vec![0u32; n];
    PACK_NS.fetch_add(t_pack.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    let phase = rcb_slice(
        &mut ids,
        parts,
        0,
        &geom,
        SendPtr(out.as_mut_ptr()),
        graph,
        refine_cuts,
        ratio,
    );
    record_phase(phase);
    out
}

fn ptscotch_assign(graph: &CellGraph, parts: u32, ratio: (u64, u64)) -> Result<Vec<u32>, Error> {
    #[cfg(not(feature = "ptscotch"))]
    {
        let _ = (graph, parts, ratio);
        Err(Error::ScotchDisabled)
    }
    #[cfg(feature = "ptscotch")]
    {
        ptscotch_linked(graph, parts, ratio)
    }
}

#[cfg(feature = "ptscotch")]
fn ptscotch_linked(graph: &CellGraph, parts: u32, ratio: (u64, u64)) -> Result<Vec<u32>, Error> {
    let n = graph.grid.ncell();
    if n > i32::MAX as usize || parts > i32::MAX as u32 {
        return Err(Error::Overflow);
    }
    let mut offsets = Vec::with_capacity(n + 1);
    let mut edges = Vec::with_capacity(graph.neighbor.len());
    let mut vwgt = Vec::with_capacity(n);
    let mut ewgt = Vec::with_capacity(graph.edge_weight.len());
    for &v in &graph.offsets {
        if v > i32::MAX as u32 {
            return Err(Error::Overflow);
        }
        offsets.push(v as i32);
    }
    for &v in &graph.neighbor {
        if v > i32::MAX as u32 {
            return Err(Error::Overflow);
        }
        edges.push(v as i32);
    }
    for &v in &graph.grid.weight {
        if v > i32::MAX as u64 {
            return Err(Error::Overflow);
        }
        vwgt.push(v as i32);
    }
    for &v in &graph.edge_weight {
        if v > i32::MAX as u64 {
            return Err(Error::Overflow);
        }
        ewgt.push(v as i32);
    }
    let mut part = vec![0i32; n];
    let (num, den) = ratio;
    let imbalance = if den == 0 {
        0.0
    } else {
        (num as f64 / den as f64) - 1.0
    };
    let mut seconds = 0.0f64;
    let rc = unsafe {
        lc_ptscotch_part(
            n as i32,
            offsets.as_ptr(),
            edges.as_ptr(),
            vwgt.as_ptr(),
            ewgt.as_ptr(),
            parts as i32,
            imbalance,
            part.as_mut_ptr(),
            &mut seconds,
        )
    };
    if rc != 0 {
        return Err(Error::ScotchFailed);
    }
    let _ = seconds;
    let mut out = Vec::with_capacity(n);
    for v in part {
        if v < 0 || v as u32 >= parts {
            return Err(Error::ScotchFailed);
        }
        out.push(v as u32);
    }
    Ok(out)
}

#[cfg(feature = "ptscotch")]
unsafe extern "C" {
    fn lc_ptscotch_part(
        n: i32,
        offsets: *const i32,
        edges: *const i32,
        vwgt: *const i32,
        ewgt: *const i32,
        parts: i32,
        imbalance: f64,
        part: *mut i32,
        seconds_max: *mut f64,
    ) -> i32;
}

fn dispatch(
    graph: &CellGraph,
    method: Method,
    parts: u32,
    ratio: (u64, u64),
    device_assign: Option<Vec<u32>>,
) -> Result<Vec<u32>, Error> {
    let mut assign = match method {
        Method::Scotch => scotch_assign(graph, parts, ratio)?,
        Method::PtScotch => ptscotch_assign(graph, parts, ratio)?,
        Method::Multi => multi_assign(graph, parts, ratio),
        Method::Rcb | Method::RcbFm => rcb_assign(graph, parts, method == Method::RcbFm, ratio),
        Method::Morton | Method::Hilbert | Method::HilbertFm => match device_assign {
            Some(done) => done,
            None => curve_assign(graph, parts, method)?,
        },
    };
    if method.refines() {
        refine(&mut assign, graph, parts, ratio);
    }
    if matches!(method, Method::RcbFm | Method::HilbertFm | Method::Multi) {
        let t_repair = Instant::now();
        repair_balance(&mut assign, graph, parts, ratio);
        if matches!(method, Method::RcbFm | Method::Multi)
            && !holds_balance(&graph.grid.weight, &assign, parts, ratio)
        {
            REPAIR_NS.fetch_add(
                t_repair.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            assign = rcb_assign(graph, parts, false, ratio);
        } else {
            REPAIR_NS.fetch_add(
                t_repair.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
        }
    }
    Ok(assign)
}

fn part_cap(total: u64, parts: u32, ratio: (u64, u64)) -> u64 {
    let (num, den) = ratio;
    if parts == 0 || den == 0 {
        return total;
    }
    ((u128::from(total) * u128::from(num)) / (u128::from(parts) * u128::from(den))) as u64
}

/// Move boundary weight off any part that finished above the cap.
fn repair_balance(assign: &mut [u32], graph: &CellGraph, parts: u32, ratio: (u64, u64)) {
    if parts <= 1 {
        return;
    }
    let n = assign.len();
    let weight = &graph.grid.weight;
    let total: u64 = weight.iter().copied().sum();
    let cap = part_cap(total, parts, ratio);
    let mut part_w = vec![0u64; parts as usize];
    for (c, &d) in assign.iter().enumerate() {
        if (d as usize) < part_w.len() {
            part_w[d as usize] += weight[c];
        }
    }
    for _ in 0..n {
        let Some(src) = part_w.iter().position(|&w| w > cap) else {
            return;
        };
        let mut best: Option<(i64, usize, u32)> = None;
        for cell in 0..n {
            if assign[cell] as usize != src {
                continue;
            }
            let w = weight[cell];
            if w == 0 || part_w[src] < w || !part_boundary(cell, assign, graph) {
                continue;
            }
            let lo = graph.offsets[cell] as usize;
            let hi = graph.offsets[cell + 1] as usize;
            let mut dests: Vec<u32> = Vec::new();
            let mut to_src = 0i64;
            let mut to_dest_w = vec![0i64; parts as usize];
            for e in lo..hi {
                let nbr = graph.neighbor[e] as usize;
                let dest = assign[nbr];
                let ww = graph.edge_weight[e] as i64;
                if dest as usize == src {
                    to_src += ww;
                } else if (dest as usize) < to_dest_w.len() {
                    to_dest_w[dest as usize] += ww;
                    if !dests.contains(&dest) {
                        dests.push(dest);
                    }
                }
            }
            for dest in dests {
                if part_w[dest as usize].saturating_add(w) > cap {
                    continue;
                }
                let gain = to_dest_w[dest as usize] - to_src;
                let take = match best {
                    None => true,
                    Some((bg, bcell, bdest)) => {
                        gain > bg
                            || (gain == bg && (cell < bcell || (cell == bcell && dest < bdest)))
                    }
                };
                if take {
                    best = Some((gain, cell, dest));
                }
            }
        }
        let Some((_, cell, dest)) = best else {
            return;
        };
        let w = weight[cell];
        part_w[src] -= w;
        part_w[dest as usize] += w;
        assign[cell] = dest;
    }
}

fn curve_assign(graph: &CellGraph, parts: u32, method: Method) -> Result<Vec<u32>, Error> {
    let n = graph.grid.ncell();
    let mut order: Vec<u32> = (0..n as u32).collect();
    match method {
        Method::Morton => {
            order.sort_by(|&a, &b| morton_ord(a, b, &graph.grid).then(a.cmp(&b)));
        }
        Method::Hilbert | Method::HilbertFm => {
            let bits = hilbert_bits(graph.grid.nx, graph.grid.ny, graph.grid.nz);
            let mut keys = vec![0u64; n];
            for c in 0..n {
                let (x, y, z) = unravel(c, graph.grid.nx, graph.grid.ny);
                keys[c] = hilbert_key(x as u32, y as u32, z as u32, bits);
            }
            order.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]).then(a.cmp(&b)));
        }
        _ => return Err(Error::BadPartition),
    }
    Ok(prefix_assign(&order, &graph.grid.weight, parts))
}

fn device_curve(prep: &Prepared, parts: u32, method: Method) -> Result<Vec<u32>, Error> {
    let n = prep.graph.grid.ncell();
    let coords = cell_coords(&prep.graph.grid);
    let mut assign = vec![0u32; n];
    let mut got_w = vec![0u64; n];
    let rc = unsafe {
        call_device_curve(
            prep.active_bins.as_ptr(),
            prep.active_bins.len(),
            coords.as_ptr(),
            n,
            parts,
            assign.as_mut_ptr(),
            got_w.as_mut_ptr(),
            method as u8,
        )
    };
    match rc {
        0 => {}
        2 => return Err(Error::NoDevice),
        _ => return Err(Error::NoDevice),
    }
    if got_w != prep.graph.grid.weight {
        return Err(Error::DeviceMismatch);
    }
    for &d in &assign {
        if d >= parts {
            return Err(Error::DeviceMismatch);
        }
    }
    Ok(assign)
}

fn cell_coords(grid: &CellGrid) -> Vec<u32> {
    let mut coords = Vec::with_capacity(grid.ncell() * 3);
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                coords.push(ix as u32);
                coords.push(iy as u32);
                coords.push(iz as u32);
            }
        }
    }
    coords
}

/// Weighted prefix of `order`. A zero total weight treats every cell as weight 1.
pub fn prefix_assign(order: &[u32], weight: &[u64], parts: u32) -> Vec<u32> {
    let n = weight.len();
    let mut assign = vec![0u32; n];
    if parts == 0 || n == 0 {
        return assign;
    }
    let total: u64 = weight.iter().copied().sum();
    let mut before = 0u64;
    for &cell in order {
        let w = if total == 0 { 1 } else { weight[cell as usize] };
        let tot = if total == 0 { n as u64 } else { total };
        assign[cell as usize] = owner_of(before, w, tot, parts);
        before += w;
    }
    assign
}

fn owner_of(before: u64, weight: u64, total: u64, parts: u32) -> u32 {
    if total == 0 || parts == 0 {
        return 0;
    }
    let num = (2u128 * u128::from(before) + u128::from(weight)) * u128::from(parts);
    let den = 2u128 * u128::from(total);
    let mut owner = (num / den) as u32;
    if owner >= parts {
        owner = parts - 1;
    }
    owner
}

static SELECT_NS: AtomicU64 = AtomicU64::new(0);
static FM_NS: AtomicU64 = AtomicU64::new(0);
static COARSE_NS: AtomicU64 = AtomicU64::new(0);
static PACK_NS: AtomicU64 = AtomicU64::new(0);
static REPAIR_NS: AtomicU64 = AtomicU64::new(0);
static CHECK_NS: AtomicU64 = AtomicU64::new(0);
static SCHEDULE_NS: AtomicU64 = AtomicU64::new(0);

/// Zero the phase counters used by the next partition.
pub fn profile_reset() {
    SELECT_NS.store(0, AtomicOrdering::Relaxed);
    FM_NS.store(0, AtomicOrdering::Relaxed);
    COARSE_NS.store(0, AtomicOrdering::Relaxed);
    PACK_NS.store(0, AtomicOrdering::Relaxed);
    REPAIR_NS.store(0, AtomicOrdering::Relaxed);
    CHECK_NS.store(0, AtomicOrdering::Relaxed);
    SCHEDULE_NS.store(0, AtomicOrdering::Relaxed);
}

/// Critical-path wall time, in nanoseconds.
///
/// The fields are the weighted median, boundary refinement, coarsening,
/// packing, the balance repair, the graph check, and join scheduling.
/// Packing fills coordinates, measures the span, and splits cell ids.
/// Scheduling is the join wall beyond the child that ran. Each value is
/// the sum over the calls since [`profile_reset`]. One thread adds the
/// children, because the join runs them in order.
pub fn profile_ns() -> (u64, u64, u64, u64, u64, u64, u64) {
    (
        SELECT_NS.load(AtomicOrdering::Relaxed),
        FM_NS.load(AtomicOrdering::Relaxed),
        COARSE_NS.load(AtomicOrdering::Relaxed),
        PACK_NS.load(AtomicOrdering::Relaxed),
        REPAIR_NS.load(AtomicOrdering::Relaxed),
        CHECK_NS.load(AtomicOrdering::Relaxed),
        SCHEDULE_NS.load(AtomicOrdering::Relaxed),
    )
}

fn axis_coord(geom: &RcbGeom, axis: usize, id: u32) -> f64 {
    match axis {
        0 => geom.x[id as usize],
        1 => geom.y[id as usize],
        _ => geom.z[id as usize],
    }
}

fn coord_before(geom: &RcbGeom, axis: usize, a: u32, b: u32) -> bool {
    let ca = axis_coord(geom, axis, a);
    let cb = axis_coord(geom, axis, b);
    ca < cb || (ca == cb && a < b)
}

fn partition_at(ids: &mut [u32], geom: &RcbGeom, axis: usize, lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    let a = ids[lo];
    let b = ids[mid];
    let c = ids[hi - 1];
    let pivot = if coord_before(geom, axis, a, b) {
        if coord_before(geom, axis, b, c) {
            b
        } else if coord_before(geom, axis, a, c) {
            c
        } else {
            a
        }
    } else if coord_before(geom, axis, a, c) {
        a
    } else if coord_before(geom, axis, b, c) {
        c
    } else {
        b
    };
    let mut pivot_at = lo;
    if ids[mid] == pivot {
        pivot_at = mid;
    } else if ids[hi - 1] == pivot {
        pivot_at = hi - 1;
    }
    ids.swap(pivot_at, hi - 1);
    let pivot = ids[hi - 1];
    let mut store = lo;
    for i in lo..hi - 1 {
        if coord_before(geom, axis, ids[i], pivot) {
            ids.swap(store, i);
            store += 1;
        }
    }
    ids.swap(store, hi - 1);
    store
}

fn clamp_cut(cut: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    cut.clamp(1, n - 1)
}

fn subset_cut(ids: &[u32], cut: usize, graph: &CellGraph, side: &mut [u8]) -> u64 {
    side.fill(0);
    for &id in &ids[..cut] {
        side[id as usize] = 1;
    }
    for &id in &ids[cut..] {
        side[id as usize] = 2;
    }
    let side_ref: &[u8] = side;
    let left = &ids[..cut];
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() > 1 && left.len() > 1 {
            use rayon::prelude::*;
            let chunk = work_chunk(left.len());
            return left
                .par_chunks(chunk)
                .map(|group| crossing_weight(group, graph, side_ref))
                .sum();
        }
    }
    crossing_weight(left, graph, side_ref)
}

fn crossing_weight(group: &[u32], graph: &CellGraph, side: &[u8]) -> u64 {
    let mut crossed = 0u64;
    for &id in group {
        let lo = graph.offsets[id as usize] as usize;
        let hi = graph.offsets[id as usize + 1] as usize;
        for e in lo..hi {
            let nbr = graph.neighbor[e] as usize;
            if side[nbr] == 2 {
                crossed += graph.edge_weight[e];
            }
        }
    }
    crossed
}

fn work_chunk(n: usize) -> usize {
    let threads = {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads().max(1)
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    };
    (n / threads).max(1)
}

/// Weighted median on each axis. Keeps the split with the smaller edge cut.
/// Ties keep `prefer`, which is the longest geometric axis.
fn lightest_axis(
    ids: &mut [u32],
    geom: &RcbGeom,
    graph: &CellGraph,
    total: u64,
    parts: u32,
    parts_l: u32,
    prefer: usize,
) -> usize {
    let saved = ids.to_vec();
    let mut side = vec![0u8; graph.grid.ncell()];
    let mut best_score = u64::MAX;
    let mut best_order = saved.clone();
    let mut best_at = 0usize;
    for axis_try in 0..3 {
        ids.copy_from_slice(&saved);
        let at = clamp_cut(
            if total == 0 {
                ids.len() * parts_l as usize / parts.max(1) as usize
            } else {
                weighted_cut(ids, geom, axis_try, total, parts, parts_l)
            },
            ids.len(),
        );
        let score = subset_cut(ids, at, graph, &mut side);
        let better = score < best_score || (score == best_score && axis_try == prefer);
        if better {
            best_score = score;
            best_order.copy_from_slice(ids);
            best_at = at;
        }
    }
    ids.copy_from_slice(&best_order);
    best_at
}

/// Light side of a weighted median. Average linear in the slice length.
///
/// A slice already ordered by cell id uses one histogram per axis. The
/// bin count is the integer span of that coordinate, which is fixed by
/// the cells in the slice. Ties stay on cell id, so the light set matches
/// the serial quickselect.
fn weighted_cut(
    ids: &mut [u32],
    geom: &RcbGeom,
    axis: usize,
    total: u64,
    parts: u32,
    parts_l: u32,
) -> usize {
    if total > 0 && ids_increase(ids) {
        if let Some(cut) = weighted_cut_bins(ids, geom, axis, total, parts, parts_l) {
            return cut;
        }
    }
    weighted_cut_quick(ids, geom, axis, total, parts, parts_l)
}

fn ids_increase(ids: &[u32]) -> bool {
    ids.windows(2).all(|pair| pair[0] < pair[1])
}

fn coord_i64(geom: &RcbGeom, axis: usize, id: u32) -> Option<i64> {
    let value = axis_coord(geom, axis, id);
    if !value.is_finite() {
        return None;
    }
    let whole = value as i64;
    if whole as f64 != value {
        return None;
    }
    Some(whole)
}

fn weighted_cut_bins(
    ids: &mut [u32],
    geom: &RcbGeom,
    axis: usize,
    total: u64,
    parts: u32,
    parts_l: u32,
) -> Option<usize> {
    let n = ids.len();
    if n < 2 {
        return Some(clamp_cut(0, n));
    }
    let mut min_c = i64::MAX;
    let mut max_c = i64::MIN;
    for &id in ids.iter() {
        let c = coord_i64(geom, axis, id)?;
        min_c = min_c.min(c);
        max_c = max_c.max(c);
    }
    let bins = (max_c - min_c + 1) as usize;
    if bins == 0 || bins > n {
        return None;
    }
    let (hist_w, hist_n) = bin_hist(ids, geom, axis, min_c, bins)?;
    let mut before_w = 0u64;
    let mut before_n = 0usize;
    let mut cut_bin = None;
    for b in 0..bins {
        let w = hist_w[b];
        let count = hist_n[b];
        if count == 0 {
            continue;
        }
        if owner_of(before_w.saturating_add(w), 0, total, parts) < parts_l {
            before_w = before_w.saturating_add(w);
            before_n += count;
            continue;
        }
        cut_bin = Some((b, before_w, before_n));
        break;
    }
    let (cut_b, before_w, before_n) = cut_bin.unwrap_or((bins, before_w, before_n));
    let cut_c = min_c + cut_b as i64;
    let take = if cut_b >= bins {
        0
    } else if owner_of(before_w, 0, total, parts) >= parts_l {
        0
    } else {
        bin_take(ids, geom, axis, cut_c, before_w, total, parts, parts_l)
    };
    let light_n = before_n + take;
    let cut = clamp_cut(light_n, n);
    arrange_cut(ids, geom, axis, cut_c, take, cut);
    Some(cut)
}

fn bin_hist(
    ids: &[u32],
    geom: &RcbGeom,
    axis: usize,
    min_c: i64,
    bins: usize,
) -> Option<(Vec<u64>, Vec<usize>)> {
    let one = |group: &[u32]| -> Option<(Vec<u64>, Vec<usize>)> {
        let mut weight = vec![0u64; bins];
        let mut count = vec![0usize; bins];
        for &id in group {
            let c = coord_i64(geom, axis, id)?;
            let b = (c - min_c) as usize;
            if b >= bins {
                return None;
            }
            weight[b] = weight[b].saturating_add(geom.w[id as usize]);
            count[b] += 1;
        }
        Some((weight, count))
    };
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() > 1 && ids.len() > 1 {
            use rayon::prelude::*;
            let chunk = work_chunk(ids.len());
            let parts: Vec<Option<(Vec<u64>, Vec<usize>)>> =
                ids.par_chunks(chunk).map(one).collect();
            let mut weight = vec![0u64; bins];
            let mut count = vec![0usize; bins];
            for part in parts {
                let (w, c) = part?;
                for b in 0..bins {
                    weight[b] = weight[b].saturating_add(w[b]);
                    count[b] += c[b];
                }
            }
            return Some((weight, count));
        }
    }
    one(ids)
}

fn bin_take(
    ids: &[u32],
    geom: &RcbGeom,
    axis: usize,
    cut_c: i64,
    mut before_w: u64,
    total: u64,
    parts: u32,
    parts_l: u32,
) -> usize {
    let mut take = 0usize;
    for &id in ids {
        let Some(c) = coord_i64(geom, axis, id) else {
            continue;
        };
        if c != cut_c {
            continue;
        }
        let w = geom.w[id as usize];
        if owner_of(before_w, w, total, parts) >= parts_l {
            break;
        }
        before_w = before_w.saturating_add(w);
        take += 1;
    }
    take
}

fn arrange_cut(ids: &mut [u32], geom: &RcbGeom, axis: usize, cut_c: i64, take: usize, cut: usize) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && ids.len() >= 4096 {
        arrange_cut_parallel(ids, geom, axis, cut_c, take);
        return;
    }
    let _ = cut;
    arrange_cut_serial(ids, geom, axis, cut_c, take);
}

/// Split the slice in encounter order. Cells below the cut bin, then the
/// first `take` cells of that bin, stay on the light side.
fn arrange_cut_serial(ids: &mut [u32], geom: &RcbGeom, axis: usize, cut_c: i64, take: usize) {
    let mut seen = 0usize;
    let mut light = Vec::with_capacity(ids.len());
    let mut heavy = Vec::with_capacity(ids.len());
    for &id in ids.iter() {
        let Some(c) = coord_i64(geom, axis, id) else {
            heavy.push(id);
            continue;
        };
        if c < cut_c || (c == cut_c && seen < take) {
            if c == cut_c {
                seen += 1;
            }
            light.push(id);
        } else {
            heavy.push(id);
        }
    }
    let n = light.len();
    ids[..n].copy_from_slice(&light);
    ids[n..].copy_from_slice(&heavy);
}

/// Same light set and same order as [`arrange_cut_serial`], with one
/// chunk of the slice per thread.
#[cfg(feature = "parallel")]
fn arrange_cut_parallel(ids: &mut [u32], geom: &RcbGeom, axis: usize, cut_c: i64, take: usize) {
    use rayon::prelude::*;
    let chunk = work_chunk(ids.len());
    let groups: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = ids
        .par_chunks(chunk)
        .map(|group| {
            let mut below = Vec::new();
            let mut equal = Vec::new();
            let mut above = Vec::new();
            for &id in group {
                match coord_i64(geom, axis, id) {
                    Some(c) if c < cut_c => below.push(id),
                    Some(c) if c == cut_c => equal.push(id),
                    _ => above.push(id),
                }
            }
            (below, equal, above)
        })
        .collect();
    let mut below = Vec::new();
    let mut equal = Vec::new();
    let mut above = Vec::new();
    for (lo, mid, hi) in groups {
        below.extend(lo);
        equal.extend(mid);
        above.extend(hi);
    }
    let take = take.min(equal.len());
    let light_n = below.len() + take;
    ids[..below.len()].copy_from_slice(&below);
    ids[below.len()..light_n].copy_from_slice(&equal[..take]);
    let rest = equal.len() - take;
    ids[light_n..light_n + rest].copy_from_slice(&equal[take..]);
    ids[light_n + rest..].copy_from_slice(&above);
}

/// Span and weight of a slice. Threads own disjoint id ranges.
fn slice_span(ids: &[u32], geom: &RcbGeom) -> ([f64; 3], [f64; 3], u64) {
    let one = |group: &[u32]| {
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        let mut total = 0u64;
        for &id in group {
            let c = [
                geom.x[id as usize],
                geom.y[id as usize],
                geom.z[id as usize],
            ];
            for axis in 0..3 {
                min[axis] = min[axis].min(c[axis]);
                max[axis] = max[axis].max(c[axis]);
            }
            total = total.saturating_add(geom.w[id as usize]);
        }
        (min, max, total)
    };
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && ids.len() >= 4096 {
        use rayon::prelude::*;
        let chunk = work_chunk(ids.len());
        let parts: Vec<_> = ids.par_chunks(chunk).map(one).collect();
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        let mut total = 0u64;
        for (lo, hi, weight) in parts {
            for axis in 0..3 {
                min[axis] = min[axis].min(lo[axis]);
                max[axis] = max[axis].max(hi[axis]);
            }
            total = total.saturating_add(weight);
        }
        return (min, max, total);
    }
    one(ids)
}

fn weighted_cut_quick(
    ids: &mut [u32],
    geom: &RcbGeom,
    axis: usize,
    total: u64,
    parts: u32,
    parts_l: u32,
) -> usize {
    fn walk(
        ids: &mut [u32],
        geom: &RcbGeom,
        axis: usize,
        lo: usize,
        hi: usize,
        before: u64,
        total: u64,
        parts: u32,
        parts_l: u32,
    ) -> usize {
        if hi <= lo {
            return lo;
        }
        if hi - lo == 1 {
            let w = geom.w[ids[lo] as usize];
            return if owner_of(before, w, total, parts) >= parts_l {
                lo
            } else {
                lo + 1
            };
        }
        let pivot = partition_at(ids, geom, axis, lo, hi);
        let mut left_w = before;
        for id in &ids[lo..pivot] {
            left_w += geom.w[*id as usize];
        }
        let pivot_w = geom.w[ids[pivot] as usize];
        if owner_of(left_w, pivot_w, total, parts) >= parts_l {
            walk(ids, geom, axis, lo, pivot, before, total, parts, parts_l)
        } else {
            walk(
                ids,
                geom,
                axis,
                pivot + 1,
                hi,
                left_w + pivot_w,
                total,
                parts,
                parts_l,
            )
        }
    }
    let cut = walk(ids, geom, axis, 0, ids.len(), 0, total, parts, parts_l);
    cut.clamp(1, ids.len().saturating_sub(1))
}

fn rcb_assign(graph: &CellGraph, parts: u32, refine_cuts: bool, ratio: (u64, u64)) -> Vec<u32> {
    let n = graph.grid.ncell();
    let t_pack = Instant::now();
    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];
    for c in 0..n {
        let (ix, iy, iz) = unravel(c, graph.grid.nx, graph.grid.ny);
        x[c] = f64::from(ix);
        y[c] = f64::from(iy);
        z[c] = f64::from(iz);
    }
    let geom = RcbGeom {
        x,
        y,
        z,
        w: graph.grid.weight.clone(),
    };
    let mut ids: Vec<u32> = (0..n as u32).collect();
    let mut out = vec![0u32; n];
    PACK_NS.fetch_add(t_pack.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    let phase = rcb_slice(
        &mut ids,
        parts,
        0,
        &geom,
        SendPtr(out.as_mut_ptr()),
        graph,
        refine_cuts,
        ratio,
    );
    record_phase(phase);
    out
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Wall time of one bisection subtree.
///
/// `select`, `fm`, and `pack` are sequential on a single branch. A join
/// keeps the slower child when more than one thread is in the pool, and
/// adds the children when the join is serial.
#[derive(Clone, Copy, Default)]
struct Phase {
    select: u64,
    fm: u64,
    pack: u64,
    schedule: u64,
}

fn phase_wall(phase: Phase) -> u64 {
    phase
        .select
        .saturating_add(phase.fm)
        .saturating_add(phase.pack)
        .saturating_add(phase.schedule)
}

fn record_phase(phase: Phase) {
    SELECT_NS.fetch_add(phase.select, AtomicOrdering::Relaxed);
    FM_NS.fetch_add(phase.fm, AtomicOrdering::Relaxed);
    PACK_NS.fetch_add(phase.pack, AtomicOrdering::Relaxed);
    SCHEDULE_NS.fetch_add(phase.schedule, AtomicOrdering::Relaxed);
}

fn join_measured(a: Phase, b: Phase, wall: u64) -> Phase {
    let sum_w = phase_wall(a).saturating_add(phase_wall(b));
    let max_w = phase_wall(a).max(phase_wall(b));
    // A sequential join lasts about as long as the two children.
    // An overlapped join finishes before that sum.
    // The slack is one tick of Instant, measured on this host.
    let overlapped = wall.saturating_add(timer_resolution_ns()) < sum_w;
    let mut child = if overlapped {
        if phase_wall(a) >= phase_wall(b) {
            a
        } else {
            b
        }
    } else {
        Phase {
            select: a.select.saturating_add(b.select),
            fm: a.fm.saturating_add(b.fm),
            pack: a.pack.saturating_add(b.pack),
            schedule: a.schedule.saturating_add(b.schedule),
        }
    };
    let base = if overlapped { max_w } else { sum_w };
    if wall > base {
        child.schedule = child.schedule.saturating_add(wall - base);
    }
    child
}

#[cfg_attr(feature = "parallel", allow(dead_code))]
fn phase_join(a: Phase, b: Phase) -> Phase {
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() > 1 {
            return if phase_wall(a) >= phase_wall(b) { a } else { b };
        }
    }
    Phase {
        select: a.select.saturating_add(b.select),
        fm: a.fm.saturating_add(b.fm),
        pack: a.pack.saturating_add(b.pack),
        schedule: a.schedule.saturating_add(b.schedule),
    }
}

fn phase_after(select_here: u64, fm_here: u64, pack_here: u64, child: Phase) -> Phase {
    Phase {
        select: select_here.saturating_add(child.select),
        fm: fm_here.saturating_add(child.fm),
        pack: pack_here.saturating_add(child.pack),
        schedule: child.schedule,
    }
}

struct RcbGeom {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<u64>,
}

fn rcb_slice(
    ids: &mut [u32],
    parts: u32,
    part0: u32,
    geom: &RcbGeom,
    out: SendPtr,
    graph: &CellGraph,
    refine_cuts: bool,
    ratio: (u64, u64),
) -> Phase {
    if ids.is_empty() {
        return Phase::default();
    }
    if parts <= 1 || ids.len() == 1 {
        let t_pack = Instant::now();
        for &id in ids.iter() {
            // Each cell id is written by exactly one bisection branch.
            unsafe { out.0.add(id as usize).write(part0) };
        }
        return Phase {
            select: 0,
            fm: 0,
            pack: t_pack.elapsed().as_nanos() as u64,
            schedule: 0,
        };
    }
    let parts_l = parts / 2;
    let parts_r = parts - parts_l;
    let t_pack = Instant::now();
    let (min, max, total) = slice_span(ids, geom);
    let span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let axis = if span[0] >= span[1] && span[0] >= span[2] {
        0
    } else if span[1] >= span[2] {
        1
    } else {
        2
    };
    let pack_span = t_pack.elapsed().as_nanos() as u64;
    let t_sel = Instant::now();
    let mut cut = if refine_cuts {
        // The longest axis is not the lightest cut when pair counts differ.
        lightest_axis(ids, geom, graph, total, parts, parts_l, axis)
    } else if total == 0 {
        ids.len() * parts_l as usize / parts as usize
    } else {
        weighted_cut(ids, geom, axis, total, parts, parts_l)
    };
    let select_here = t_sel.elapsed().as_nanos() as u64;
    if cut == 0 {
        cut = 1;
    }
    if cut >= ids.len() {
        cut = ids.len() - 1;
    }
    // The weighted cut already lays the light cells in the prefix. The two
    // sides descend from that prefix, with no second copy of the ids.
    if !refine_cuts {
        let (left_ids, right_ids) = ids.split_at_mut(cut);
        let child = descend(
            left_ids,
            parts_l,
            part0,
            right_ids,
            parts_r,
            part0 + parts_l,
            geom,
            out,
            graph,
            refine_cuts,
            ratio,
        );
        return phase_after(select_here, 0, pack_span, child);
    }
    let t_side = Instant::now();
    let mut side = vec![1u8; ids.len()];
    for s in side.iter_mut().take(cut) {
        *s = 0;
    }
    let pack_side = t_side.elapsed().as_nanos() as u64;
    let t_fm = Instant::now();
    bipartition_fm(ids, &mut side, geom, graph, parts_l, parts_r, ratio);
    let fm_here = t_fm.elapsed().as_nanos() as u64;
    let t_split = Instant::now();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if side[i] == 0 {
            left.push(id);
        } else {
            right.push(id);
        }
    }
    let pack_here = pack_span
        .saturating_add(pack_side)
        .saturating_add(t_split.elapsed().as_nanos() as u64);
    let child = if left.is_empty() || right.is_empty() {
        let (left_ids, right_ids) = ids.split_at_mut(cut);
        descend(
            left_ids,
            parts_l,
            part0,
            right_ids,
            parts_r,
            part0 + parts_l,
            geom,
            out,
            graph,
            refine_cuts,
            ratio,
        )
    } else {
        descend(
            &mut left,
            parts_l,
            part0,
            &mut right,
            parts_r,
            part0 + parts_l,
            geom,
            out,
            graph,
            refine_cuts,
            ratio,
        )
    };
    let t_drop = Instant::now();
    drop(left);
    drop(right);
    drop(side);
    let pack_drop = t_drop.elapsed().as_nanos() as u64;
    phase_after(
        select_here,
        fm_here,
        pack_here.saturating_add(pack_drop),
        child,
    )
}

fn descend(
    left: &mut [u32],
    parts_l: u32,
    part_l: u32,
    right: &mut [u32],
    parts_r: u32,
    part_r: u32,
    geom: &RcbGeom,
    out: SendPtr,
    graph: &CellGraph,
    refine_cuts: bool,
    ratio: (u64, u64),
) -> Phase {
    #[cfg(feature = "parallel")]
    {
        let left_out = out;
        let right_out = out;
        let t_join = Instant::now();
        let (a, b) = rayon::join(
            move || {
                rcb_slice(
                    left,
                    parts_l,
                    part_l,
                    geom,
                    left_out,
                    graph,
                    refine_cuts,
                    ratio,
                )
            },
            move || {
                rcb_slice(
                    right,
                    parts_r,
                    part_r,
                    geom,
                    right_out,
                    graph,
                    refine_cuts,
                    ratio,
                )
            },
        );
        let wall = t_join.elapsed().as_nanos() as u64;
        return join_measured(a, b, wall);
    }
    #[cfg(not(feature = "parallel"))]
    {
        let a = rcb_slice(left, parts_l, part_l, geom, out, graph, refine_cuts, ratio);
        let b = rcb_slice(right, parts_r, part_r, geom, out, graph, refine_cuts, ratio);
        phase_join(a, b)
    }
}

/// Fiduccia-Mattheyses across one bisection. `side` is 0 or 1.
fn bipartition_fm(
    ids: &[u32],
    side: &mut [u8],
    geom: &RcbGeom,
    graph: &CellGraph,
    parts_l: u32,
    parts_r: u32,
    ratio: (u64, u64),
) {
    let parts = parts_l + parts_r;
    if parts_l == 0 || parts_r == 0 {
        return;
    }
    let mut index = vec![usize::MAX; graph.grid.ncell()];
    let mut weight_l = 0u64;
    let mut weight_r = 0u64;
    for (i, &id) in ids.iter().enumerate() {
        index[id as usize] = i;
        if side[i] == 0 {
            weight_l += geom.w[id as usize];
        } else {
            weight_r += geom.w[id as usize];
        }
    }
    let total = weight_l + weight_r;
    loop {
        if !boundary_pass(
            ids,
            side,
            &index,
            geom,
            graph,
            parts_l,
            parts_r,
            parts,
            ratio,
            total,
            &mut weight_l,
            &mut weight_r,
        ) {
            break;
        }
    }
}

fn side_limit(total: u64, side_parts: u32, parts: u32, ratio: (u64, u64)) -> u64 {
    if total == 0 || parts == 0 {
        return 0;
    }
    let (num, den) = ratio;
    let num = u128::from(total) * u128::from(side_parts) * u128::from(num);
    let den = u128::from(parts) * u128::from(den);
    (num / den) as u64
}

fn cross_gain(slot: usize, ids: &[u32], side: &[u8], index: &[usize], graph: &CellGraph) -> i64 {
    let id = ids[slot] as usize;
    let src = side[slot];
    let mut to_src = 0i64;
    let mut to_dest = 0i64;
    let lo = graph.offsets[id] as usize;
    let hi = graph.offsets[id + 1] as usize;
    for e in lo..hi {
        let nbr = graph.neighbor[e] as usize;
        let other = index[nbr];
        if other == usize::MAX {
            continue;
        }
        let ww = graph.edge_weight[e] as i64;
        if side[other] == src {
            to_src += ww;
        } else {
            to_dest += ww;
        }
    }
    to_dest - to_src
}

fn side_fits(
    slot: usize,
    ids: &[u32],
    side: &[u8],
    geom: &RcbGeom,
    weight_l: u64,
    weight_r: u64,
    limit_l: u64,
    limit_r: u64,
) -> bool {
    let w = geom.w[ids[slot] as usize];
    let (new_l, new_r) = if side[slot] == 0 {
        (weight_l - w, weight_r + w)
    } else {
        (weight_l + w, weight_r - w)
    };
    new_l <= limit_l && new_r <= limit_r
}

#[allow(dead_code)]
fn bucket_drop(slot: usize, bins: &mut BTreeMap<i64, Vec<u32>>, loc: &mut [Option<(i64, usize)>]) {
    let Some((gain, at)) = loc[slot] else { return };
    let bin = bins.get_mut(&gain).unwrap();
    let last = bin.len() - 1;
    bin.swap(at, last);
    let moved = bin[at] as usize;
    if moved != slot {
        if let Some(entry) = loc[moved].as_mut() {
            entry.1 = at;
        }
    }
    bin.pop();
    if bin.is_empty() {
        bins.remove(&gain);
    }
    loc[slot] = None;
}

#[allow(dead_code)]
fn bucket_add(
    slot: usize,
    bins: &mut BTreeMap<i64, Vec<u32>>,
    loc: &mut [Option<(i64, usize)>],
    ids: &[u32],
    side: &[u8],
    index: &[usize],
    graph: &CellGraph,
    geom: &RcbGeom,
    locked: &[bool],
    weight_l: u64,
    weight_r: u64,
    limit_l: u64,
    limit_r: u64,
) {
    if locked[slot] {
        return;
    }
    bucket_drop(slot, bins, loc);
    if !is_boundary(slot, ids, side, index, graph) {
        return;
    }
    if !side_fits(slot, ids, side, geom, weight_l, weight_r, limit_l, limit_r) {
        return;
    }
    let gain = cross_gain(slot, ids, side, index, graph);
    let bin = bins.entry(gain).or_default();
    loc[slot] = Some((gain, bin.len()));
    bin.push(slot as u32);
}

#[allow(dead_code)]
fn boundary_pass(
    ids: &[u32],
    side: &mut [u8],
    index: &[usize],
    geom: &RcbGeom,
    graph: &CellGraph,
    parts_l: u32,
    parts_r: u32,
    parts: u32,
    ratio: (u64, u64),
    total: u64,
    weight_l: &mut u64,
    weight_r: &mut u64,
) -> bool {
    let t0 = Instant::now();
    let n = ids.len();
    let limit_l = side_limit(total, parts_l, parts, ratio);
    let limit_r = side_limit(total, parts_r, parts, ratio);
    let mut locked = vec![false; n];
    let mut loc: Vec<Option<(i64, usize)>> = vec![None; n];
    let mut bins: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
    // Gains are a pure function of the current sides. The parallel scan
    // matches the serial scan, and slots enter each bucket in index order.
    let gains = bisect_gains(
        ids, side, index, graph, geom, *weight_l, *weight_r, limit_l, limit_r,
    );
    for slot in 0..n {
        let Some(gain) = gains[slot] else {
            continue;
        };
        let bin = bins.entry(gain).or_default();
        loc[slot] = Some((gain, bin.len()));
        bin.push(slot as u32);
    }
    let mut moves: Vec<(usize, u8)> = Vec::new();
    let mut cum: i128 = 0;
    let mut best: i128 = 0;
    let mut best_len = 0usize;
    // The balance bound lets a part absorb one extra heaviest cell.
    // The pass may try that many non-positive moves, then it stops.
    let heaviest = ids.iter().map(|&id| geom.w[id as usize]).max().unwrap_or(1);
    let mut neg_left = heaviest.max(1) as usize;
    while let Some((&gain, _)) = bins.iter().next_back() {
        if gain <= 0 {
            neg_left = neg_left.saturating_sub(1);
            if neg_left == 0 {
                break;
            }
        }
        let bin = bins.get_mut(&gain).unwrap();
        let slot = bin.pop().unwrap() as usize;
        if bin.is_empty() {
            bins.remove(&gain);
        }
        loc[slot] = None;
        if locked[slot] {
            continue;
        }
        if !side_fits(
            slot, ids, side, geom, *weight_l, *weight_r, limit_l, limit_r,
        ) {
            continue;
        }
        let fresh = cross_gain(slot, ids, side, index, graph);
        if fresh != gain {
            bucket_add(
                slot, &mut bins, &mut loc, ids, side, index, graph, geom, &locked, *weight_l,
                *weight_r, limit_l, limit_r,
            );
            continue;
        }
        let id = ids[slot] as usize;
        let w = geom.w[id];
        let src = side[slot];
        if src == 0 {
            *weight_l -= w;
            *weight_r += w;
            side[slot] = 1;
        } else {
            *weight_r -= w;
            *weight_l += w;
            side[slot] = 0;
        }
        locked[slot] = true;
        moves.push((slot, src));
        cum += i128::from(fresh);
        if cum > best {
            best = cum;
            best_len = moves.len();
        }
        let lo = graph.offsets[id] as usize;
        let hi = graph.offsets[id + 1] as usize;
        for e in lo..hi {
            let other = index[graph.neighbor[e] as usize];
            if other != usize::MAX && !locked[other] {
                bucket_add(
                    other, &mut bins, &mut loc, ids, side, index, graph, geom, &locked, *weight_l,
                    *weight_r, limit_l, limit_r,
                );
            }
        }
    }
    if best <= 0 {
        best_len = 0;
    }
    for &(slot, src) in moves
        .iter()
        .rev()
        .take(moves.len().saturating_sub(best_len))
    {
        let id = ids[slot] as usize;
        let w = geom.w[id];
        if side[slot] == 0 {
            *weight_l -= w;
            *weight_r += w;
        } else {
            *weight_r -= w;
            *weight_l += w;
        }
        side[slot] = src;
    }
    let _ = t0;
    best > 0
}

fn is_boundary(slot: usize, ids: &[u32], side: &[u8], index: &[usize], graph: &CellGraph) -> bool {
    let id = ids[slot] as usize;
    let src = side[slot];
    let lo = graph.offsets[id] as usize;
    let hi = graph.offsets[id + 1] as usize;
    for e in lo..hi {
        let other = index[graph.neighbor[e] as usize];
        if other != usize::MAX && side[other] != src {
            return true;
        }
    }
    false
}

fn bisect_gains(
    ids: &[u32],
    side: &[u8],
    index: &[usize],
    graph: &CellGraph,
    geom: &RcbGeom,
    weight_l: u64,
    weight_r: u64,
    limit_l: u64,
    limit_r: u64,
) -> Vec<Option<i64>> {
    let n = ids.len();
    let one = |slot: usize| -> Option<i64> {
        if !is_boundary(slot, ids, side, index, graph) {
            return None;
        }
        if !side_fits(slot, ids, side, geom, weight_l, weight_r, limit_l, limit_r) {
            return None;
        }
        Some(cross_gain(slot, ids, side, index, graph))
    };
    #[cfg(feature = "parallel")]
    {
        // The parallel map writes one slot per index, so it matches the
        // serial map at every length. One thread stays on the serial map.
        if rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            return (0..n).into_par_iter().map(one).collect();
        }
    }
    (0..n).map(one).collect()
}

fn refine(assign: &mut [u32], graph: &CellGraph, parts: u32, ratio: (u64, u64)) {
    if parts <= 1 {
        return;
    }
    let t0 = Instant::now();
    let weight = &graph.grid.weight;
    let mut part_w = vec![0u64; parts as usize];
    for (c, &d) in assign.iter().enumerate() {
        part_w[d as usize] += weight[c];
    }
    let total: u64 = part_w.iter().copied().sum();
    loop {
        if !fm_pass(assign, graph, parts, ratio, &mut part_w, total, weight) {
            break;
        }
    }
    FM_NS.fetch_add(t0.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
}

fn kway_proposal(
    assign: &[u32],
    graph: &CellGraph,
    parts: u32,
    ratio: (u64, u64),
    part_w: &[u64],
    total: u64,
    weight: &[u64],
) -> Vec<Option<(i64, u32)>> {
    let n = assign.len();
    let one = |cell: usize| -> Option<(i64, u32)> {
        if !part_boundary(cell, assign, graph) {
            return None;
        }
        let mut scratch = vec![0u64; parts as usize];
        best_move(
            cell as u32,
            assign,
            graph,
            parts,
            ratio,
            part_w,
            total,
            weight,
            &mut scratch,
        )
    };
    #[cfg(feature = "parallel")]
    {
        // Same rule as the bisection gain scan: every length, and only
        // when the pool has more than one thread.
        if rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            return (0..n).into_par_iter().map(one).collect();
        }
    }
    (0..n).map(one).collect()
}

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq)]
struct Cand {
    gain: i64,
    cell: u32,
    dest: u32,
    stamp: u32,
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .cmp(&other.gain)
            .then(other.cell.cmp(&self.cell))
            .then(other.dest.cmp(&self.dest))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(dead_code)]
fn fm_pass(
    assign: &mut [u32],
    graph: &CellGraph,
    parts: u32,
    ratio: (u64, u64),
    part_w: &mut [u64],
    total: u64,
    weight: &[u64],
) -> bool {
    let t0 = Instant::now();
    let n = assign.len();
    let mut locked = vec![false; n];
    let mut stamp = vec![0u32; n];
    let mut heap: BinaryHeap<Cand> = BinaryHeap::new();
    let mut scratch = vec![0u64; parts as usize];
    // The proposal is a pure function of the current assignment. Cells
    // are pushed in index order, so the heap matches a serial scan.
    let proposal = kway_proposal(assign, graph, parts, ratio, part_w, total, weight);
    for cell in 0..n {
        let Some((gain, dest)) = proposal[cell] else {
            continue;
        };
        stamp[cell] = stamp[cell].wrapping_add(1);
        heap.push(Cand {
            gain,
            cell: cell as u32,
            dest,
            stamp: stamp[cell],
        });
    }
    let mut moves: Vec<(u32, u32, u32)> = Vec::new();
    let mut cum: i128 = 0;
    let mut best: i128 = 0;
    let mut best_len = 0usize;
    let heaviest = weight.iter().copied().max().unwrap_or(1);
    let mut neg_left = heaviest.max(1) as usize;
    while let Some(cand) = heap.pop() {
        if cand.gain <= 0 {
            neg_left = neg_left.saturating_sub(1);
            if neg_left == 0 {
                break;
            }
        }
        let cell = cand.cell as usize;
        if locked[cell] || stamp[cell] != cand.stamp {
            continue;
        }
        let fresh = best_move(
            cand.cell,
            assign,
            graph,
            parts,
            ratio,
            part_w,
            total,
            weight,
            &mut scratch,
        );
        let Some((gain, dest)) = fresh else {
            continue;
        };
        if gain != cand.gain || dest != cand.dest {
            stamp[cell] = stamp[cell].wrapping_add(1);
            heap.push(Cand {
                gain,
                cell: cand.cell,
                dest,
                stamp: stamp[cell],
            });
            continue;
        }
        let src = assign[cell];
        let w = weight[cell];
        part_w[src as usize] -= w;
        part_w[dest as usize] += w;
        assign[cell] = dest;
        locked[cell] = true;
        moves.push((cand.cell, src, dest));
        cum += i128::from(gain);
        if cum > best {
            best = cum;
            best_len = moves.len();
        }
        let lo = graph.offsets[cell] as usize;
        let hi = graph.offsets[cell + 1] as usize;
        for e in lo..hi {
            let nbr = graph.neighbor[e] as usize;
            if !locked[nbr] {
                push_best(
                    nbr as u32,
                    assign,
                    graph,
                    parts,
                    ratio,
                    part_w,
                    total,
                    weight,
                    &locked,
                    &mut stamp,
                    &mut scratch,
                    &mut heap,
                );
            }
        }
    }
    if best <= 0 {
        best_len = 0;
    }
    for (cell, src, dest) in moves.iter().rev().take(moves.len() - best_len) {
        let w = weight[*cell as usize];
        part_w[*dest as usize] -= w;
        part_w[*src as usize] += w;
        assign[*cell as usize] = *src;
    }
    let _ = t0;
    best > 0
}

fn part_boundary(cell: usize, assign: &[u32], graph: &CellGraph) -> bool {
    let src = assign[cell];
    let lo = graph.offsets[cell] as usize;
    let hi = graph.offsets[cell + 1] as usize;
    for e in lo..hi {
        if assign[graph.neighbor[e] as usize] != src {
            return true;
        }
    }
    false
}

#[allow(dead_code)]
fn push_best(
    cell: u32,
    assign: &[u32],
    graph: &CellGraph,
    parts: u32,
    ratio: (u64, u64),
    part_w: &[u64],
    total: u64,
    weight: &[u64],
    locked: &[bool],
    stamp: &mut [u32],
    scratch: &mut [u64],
    heap: &mut BinaryHeap<Cand>,
) {
    if locked[cell as usize] || !part_boundary(cell as usize, assign, graph) {
        return;
    }
    let Some((gain, dest)) = best_move(
        cell, assign, graph, parts, ratio, part_w, total, weight, scratch,
    ) else {
        return;
    };
    stamp[cell as usize] = stamp[cell as usize].wrapping_add(1);
    heap.push(Cand {
        gain,
        cell,
        dest,
        stamp: stamp[cell as usize],
    });
}

fn best_move(
    cell: u32,
    assign: &[u32],
    graph: &CellGraph,
    parts: u32,
    ratio: (u64, u64),
    part_w: &[u64],
    total: u64,
    weight: &[u64],
    scratch: &mut [u64],
) -> Option<(i64, u32)> {
    scratch.fill(0);
    let src = assign[cell as usize];
    let lo = graph.offsets[cell as usize] as usize;
    let hi = graph.offsets[cell as usize + 1] as usize;
    for e in lo..hi {
        let nbr = graph.neighbor[e] as usize;
        let part = assign[nbr] as usize;
        scratch[part] = scratch[part].saturating_add(graph.edge_weight[e]);
    }
    let w = weight[cell as usize];
    let mut best: Option<(i64, u32)> = None;
    for dest in 0..parts {
        if dest == src {
            continue;
        }
        let new_src = part_w[src as usize] - w;
        let new_dest = part_w[dest as usize] + w;
        if !under_balance(new_src, total, parts, ratio)
            || !under_balance(new_dest, total, parts, ratio)
        {
            continue;
        }
        let gain = scratch[dest as usize] as i64 - scratch[src as usize] as i64;
        let take = match best {
            None => true,
            Some((g, d)) => gain > g || (gain == g && dest < d),
        };
        if take {
            best = Some((gain, dest));
        }
    }
    best
}

fn scotch_assign(graph: &CellGraph, parts: u32, ratio: (u64, u64)) -> Result<Vec<u32>, Error> {
    #[cfg(not(feature = "scotch"))]
    {
        let _ = (graph, parts, ratio);
        Err(Error::ScotchDisabled)
    }
    #[cfg(feature = "scotch")]
    {
        scotch_assign_linked(graph, parts, ratio)
    }
}

#[cfg(feature = "scotch")]
fn scotch_assign_linked(
    graph: &CellGraph,
    parts: u32,
    ratio: (u64, u64),
) -> Result<Vec<u32>, Error> {
    let n = graph.grid.ncell();
    if n > i32::MAX as usize || parts > i32::MAX as u32 {
        return Err(Error::Overflow);
    }
    let mut offsets = Vec::with_capacity(n + 1);
    let mut edges = Vec::with_capacity(graph.neighbor.len());
    let mut vwgt = Vec::with_capacity(n);
    let mut ewgt = Vec::with_capacity(graph.edge_weight.len());
    for &v in &graph.offsets {
        if v > i32::MAX as u32 {
            return Err(Error::Overflow);
        }
        offsets.push(v as i32);
    }
    for &v in &graph.neighbor {
        if v > i32::MAX as u32 {
            return Err(Error::Overflow);
        }
        edges.push(v as i32);
    }
    for &v in &graph.grid.weight {
        if v > i32::MAX as u64 {
            return Err(Error::Overflow);
        }
        vwgt.push(v as i32);
    }
    for &v in &graph.edge_weight {
        if v > i32::MAX as u64 {
            return Err(Error::Overflow);
        }
        ewgt.push(v as i32);
    }
    let mut part = vec![0i32; n];
    let (num, den) = ratio;
    let imbalance = (num - den) as f64 / den as f64;
    let rc = unsafe {
        lc_scotch_part(
            n as i32,
            offsets.as_ptr(),
            edges.as_ptr(),
            vwgt.as_ptr(),
            ewgt.as_ptr(),
            parts as i32,
            imbalance,
            part.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(Error::ScotchFailed);
    }
    let mut out = Vec::with_capacity(n);
    for v in part {
        if v < 0 || v as u32 >= parts {
            return Err(Error::ScotchFailed);
        }
        out.push(v as u32);
    }
    Ok(out)
}

#[cfg(feature = "scotch")]
unsafe extern "C" {
    fn lc_scotch_part(
        n: i32,
        offsets: *const i32,
        edges: *const i32,
        vwgt: *const i32,
        ewgt: *const i32,
        parts: i32,
        imbalance: f64,
        part: *mut i32,
    ) -> i32;
}

fn max_imbalance(weight: &[u64], assign: &[u32], parts: u32) -> f64 {
    let mut part_w = vec![0u64; parts as usize];
    let mut total = 0u64;
    for (c, &d) in assign.iter().enumerate() {
        part_w[d as usize] += weight[c];
        total += weight[c];
    }
    if total == 0 {
        return 1.0;
    }
    let mean = total as f64 / parts as f64;
    let heavy = part_w.iter().copied().max().unwrap_or(0) as f64;
    heavy / mean
}

fn halo_fraction(
    grid: &CellGrid,
    assign: &[u32],
    parts: u32,
    width: [u32; 3],
) -> Result<f64, Error> {
    let mut best = 0.0f64;
    for domain in 0..parts {
        let owned = assign.iter().filter(|&&d| d == domain).count();
        if owned == 0 {
            continue;
        }
        let ghosts = halo(grid, assign, domain, width)?.len();
        let ratio = ghosts as f64 / owned as f64;
        if ratio > best {
            best = ratio;
        }
    }
    Ok(best)
}

fn balance_ratio(balance: Option<f64>, weight: &[u64], parts: u32) -> Result<(u64, u64), Error> {
    match balance {
        None => Ok(derived_balance(weight, parts)),
        Some(value) => {
            if !value.is_finite() || value < 1.0 {
                return Err(Error::BadPartition);
            }
            // Decimal resolution of a caller-supplied bound.
            let den = 1_000_000u64;
            let scaled = value * den as f64;
            if !scaled.is_finite() || scaled > u64::MAX as f64 {
                return Err(Error::BadPartition);
            }
            let num = scaled.round() as u64;
            if num < den {
                return Err(Error::BadPartition);
            }
            Ok((num, den))
        }
    }
}

fn under_balance(weight: u64, total: u64, parts: u32, ratio: (u64, u64)) -> bool {
    if total == 0 {
        return true;
    }
    let (num, den) = ratio;
    u128::from(weight) * u128::from(parts) * u128::from(den) <= u128::from(total) * u128::from(num)
}

fn bins_needed(length: f64, width: f64) -> Result<u32, Error> {
    if !(length > 0.0) || !(width > 0.0) {
        return Ok(0);
    }
    let q = length / width;
    if !q.is_finite() {
        return Err(Error::Overflow);
    }
    let ceil = q.ceil();
    if !ceil.is_finite() || ceil > u32::MAX as f64 {
        return Err(Error::Overflow);
    }
    let mut n = ceil as u32;
    while (n as f64) * width < length {
        n = n.saturating_add(1);
        if n == u32::MAX {
            break;
        }
    }
    Ok(n)
}

fn check_graph(graph: &CellGraph) -> Result<usize, Error> {
    let n = graph.grid.ncell();
    if graph.grid.nx <= 0 || graph.grid.ny <= 0 || graph.grid.nz <= 0 {
        return Err(Error::BadGraph);
    }
    let expect = (graph.grid.nx as usize)
        .checked_mul(graph.grid.ny as usize)
        .and_then(|v| v.checked_mul(graph.grid.nz as usize))
        .ok_or(Error::BadGraph)?;
    if expect != n || graph.offsets.len() != n + 1 || graph.offsets[0] != 0 {
        return Err(Error::BadGraph);
    }
    if !offsets_increase(&graph.offsets) {
        return Err(Error::BadGraph);
    }
    let narc = graph.offsets[n] as usize;
    if graph.neighbor.len() != narc || graph.edge_weight.len() != narc {
        return Err(Error::BadGraph);
    }
    if !neighbors_in_range(&graph.neighbor, n) {
        return Err(Error::BadGraph);
    }
    Ok(n)
}

fn offsets_increase(offsets: &[u32]) -> bool {
    let check = |start: usize, part: &[u32]| -> bool {
        let mut prev = if start == 0 { 0 } else { offsets[start - 1] };
        for &off in part {
            if off < prev {
                return false;
            }
            prev = off;
        }
        true
    };
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() > 1 && offsets.len() > 1 {
            use rayon::prelude::*;
            let chunk = work_chunk(offsets.len());
            return offsets
                .par_chunks(chunk)
                .enumerate()
                .all(|(index, part)| check(index * chunk, part));
        }
    }
    check(0, offsets)
}

fn neighbors_in_range(neighbor: &[u32], n: usize) -> bool {
    let inside = |part: &[u32]| part.iter().all(|&nbr| (nbr as usize) < n);
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() > 1 && neighbor.len() > 1 {
            use rayon::prelude::*;
            let chunk = work_chunk(neighbor.len());
            return neighbor.par_chunks(chunk).all(inside);
        }
    }
    inside(neighbor)
}

fn check_assign(grid: &CellGrid, assign: &[u32]) -> Result<usize, Error> {
    let n = grid.ncell();
    if assign.len() != n {
        return Err(Error::BadPartition);
    }
    let expect = (grid.nx as usize)
        .saturating_mul(grid.ny as usize)
        .saturating_mul(grid.nz as usize);
    if expect != n {
        return Err(Error::BadGraph);
    }
    Ok(n)
}

fn unravel(c: usize, nx: i32, ny: i32) -> (i32, i32, i32) {
    let nx = nx as usize;
    let ny = ny as usize;
    let ix = (c % nx) as i32;
    let iy = ((c / nx) % ny) as i32;
    let iz = (c / (nx * ny)) as i32;
    (ix, iy, iz)
}

fn paint(mark: &mut [u8], nx: i32, ny: i32, nz: i32, ix: i32, iy: i32, iz: i32, width: [u32; 3]) {
    let rx = width[0] as i32;
    let ry = width[1] as i32;
    let rz = width[2] as i32;
    for dz in -rz..=rz {
        for dy in -ry..=ry {
            for dx in -rx..=rx {
                let x = rem_euclid(ix + dx, nx);
                let y = rem_euclid(iy + dy, ny);
                let z = rem_euclid(iz + dz, nz);
                let c = ((z as usize * ny as usize) + y as usize) * nx as usize + x as usize;
                mark[c] = 1;
            }
        }
    }
}

fn rem_euclid(v: i32, n: i32) -> i32 {
    let mut r = v % n;
    if r < 0 {
        r += n;
    }
    r
}

fn morton_ord(a: u32, b: u32, grid: &CellGrid) -> Ordering {
    let (ax, ay, az) = unravel(a as usize, grid.nx, grid.ny);
    let (bx, by, bz) = unravel(b as usize, grid.nx, grid.ny);
    morton_cmp(
        [ax as u32, ay as u32, az as u32],
        [bx as u32, by as u32, bz as u32],
    )
}

/// Morton order matching [`morton_key`]: at each bit, `z`, then `y`, then `x`.
///
/// `x` occupies the least significant bit of each triplet, so it varies
/// first along the curve.
pub fn morton_cmp(a: [u32; 3], b: [u32; 3]) -> Ordering {
    for bit in (0..32).rev() {
        let mask = 1u32 << bit;
        let az = a[2] & mask;
        let bz = b[2] & mask;
        if az != bz {
            return az.cmp(&bz);
        }
        let ay = a[1] & mask;
        let by = b[1] & mask;
        if ay != by {
            return ay.cmp(&by);
        }
        let ax = a[0] & mask;
        let bx = b[0] & mask;
        if ax != bx {
            return ax.cmp(&bx);
        }
    }
    Ordering::Equal
}

/// Packed Morton key. Coordinates at or above `2^21` do not fit.
pub fn morton_key(x: u32, y: u32, z: u32) -> Option<u64> {
    if x >= (1 << 21) || y >= (1 << 21) || z >= (1 << 21) {
        return None;
    }
    Some(spread21(x) | (spread21(y) << 1) | (spread21(z) << 2))
}

fn spread21(v: u32) -> u64 {
    let mut x = u64::from(v & 0x1f_ffff);
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

fn hilbert_bits(nx: i32, ny: i32, nz: i32) -> u32 {
    let n = (nx as u32).max(ny as u32).max(nz as u32);
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Hilbert index on a cube of `2^bits` cells per axis (Skilling, 2004).
pub fn hilbert_key(x: u32, y: u32, z: u32, bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    let mut p = [x, y, z];
    let mut q = 1u32 << (bits - 1);
    while q > 1 {
        let mask = q - 1;
        for i in 0..3 {
            if p[i] & q != 0 {
                p[0] ^= mask;
            } else {
                let t = (p[0] ^ p[i]) & mask;
                p[0] ^= t;
                p[i] ^= t;
            }
        }
        q >>= 1;
    }
    p[1] ^= p[0];
    p[2] ^= p[1];
    let mut t = 0u32;
    q = 1u32 << (bits - 1);
    while q > 1 {
        if p[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for i in 0..3 {
        p[i] ^= t;
    }
    let mut key = 0u64;
    for b in (0..bits).rev() {
        for i in 0..3 {
            key = (key << 1) | u64::from((p[i] >> b) & 1);
        }
    }
    key
}

/// Histogram of flat bin ids.
pub fn histogram(bins_of: &[u32], ncell: usize) -> Vec<u64> {
    let mut weight = vec![0u64; ncell];
    for &b in bins_of {
        weight[b as usize] += 1;
    }
    weight
}

/// Exclusive prefix. `out[i + 1] - out[i] = values[i]`, `out[0] = 0`.
pub fn exclusive_prefix(values: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; values.len() + 1];
    for (i, &v) in values.iter().enumerate() {
        out[i + 1] = out[i] + v;
    }
    out
}

unsafe extern "C" {
    fn lc_gpu_curve_split(
        atom_bin: *const u32,
        natom: usize,
        coords: *const u32,
        ncell: usize,
        parts: u32,
        assign: *mut u32,
        out_weight: *mut u64,
        method: u8,
    ) -> i32;
}

unsafe fn call_device_curve(
    atom_bin: *const u32,
    natom: usize,
    coords: *const u32,
    ncell: usize,
    parts: u32,
    assign: *mut u32,
    out_weight: *mut u64,
    method: u8,
) -> i32 {
    unsafe {
        lc_gpu_curve_split(
            atom_bin, natom, coords, ncell, parts, assign, out_weight, method,
        )
    }
}

/// C ABI method code. Unknown codes are [`Error::BadPartition`].
pub fn method_from_c(code: u8) -> Result<Method, Error> {
    Method::from_code(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_matches_the_bit_order() {
        assert_eq!(morton_key(0, 0, 0), Some(0));
        assert_eq!(morton_key(1, 0, 0), Some(1));
        assert_eq!(morton_key(0, 1, 0), Some(2));
        assert_eq!(morton_key(0, 0, 1), Some(4));
        let mut coords = Vec::new();
        for z in 0..8u32 {
            for y in 0..8 {
                for x in 0..8 {
                    coords.push([x, y, z]);
                }
            }
        }
        let mut by_key = coords.clone();
        by_key.sort_by(|a, b| {
            morton_key(a[0], a[1], a[2])
                .unwrap()
                .cmp(&morton_key(b[0], b[1], b[2]).unwrap())
        });
        let mut by_cmp = coords.clone();
        by_cmp.sort_by(|a, b| morton_cmp(*a, *b));
        assert_eq!(by_key, by_cmp);
    }

    #[test]
    fn histogram_median_matches_quickselect() {
        let mut weight = vec![1u64; 60];
        weight[3] = 5;
        weight[11] = 8;
        weight[40] = 2;
        let geom = RcbGeom {
            x: (0..60).map(|i| (i % 9) as f64).collect(),
            y: (0..60).map(|i| (i % 4) as f64).collect(),
            z: (0..60).map(|i| (i % 3) as f64).collect(),
            w: weight,
        };
        let total = geom.w.iter().sum();
        for axis in 0..3 {
            let mut hist = (0..60u32).collect::<Vec<_>>();
            let mut quick = hist.clone();
            let got = weighted_cut_bins(&mut hist, &geom, axis, total, 8, 4).unwrap();
            let expect = weighted_cut_quick(&mut quick, &geom, axis, total, 8, 4);
            assert_eq!(got, expect, "axis {axis}");
            let mut a = hist[..got].to_vec();
            let mut b = quick[..expect].to_vec();
            a.sort();
            b.sort();
            assert_eq!(a, b, "set axis {axis}");
        }
    }

    #[test]
    fn histogram_and_prefix_are_exact() {
        let bins_of = [0u32, 1, 1, 2, 0];
        let weight = histogram(&bins_of, 4);
        assert_eq!(weight, vec![2, 2, 1, 0]);
        let prefix = exclusive_prefix(&weight);
        assert_eq!(prefix, vec![0, 2, 4, 5, 5]);
        assert_eq!(prefix.windows(2).map(|w| w[1] - w[0]).sum::<u64>(), 5);
    }

    #[test]
    fn hilbert_keys_permute_a_cube() {
        for bits in 1u32..=4 {
            let side = 1u32 << bits;
            let mut seen = vec![false; (side * side * side) as usize];
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        let key = hilbert_key(x, y, z, bits) as usize;
                        assert!(key < seen.len());
                        assert!(!seen[key]);
                        seen[key] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&v| v));
        }
        assert_eq!(hilbert_key(0, 0, 0, 1), 0);
        assert_eq!(hilbert_key(1, 0, 0, 1), 7);
        assert_eq!(hilbert_key(0, 1, 0, 2), 1);
        assert_eq!(hilbert_key(1, 1, 1, 2), 5);
    }

    #[test]
    fn replicated_copies_keep_cells_and_arcs() {
        let cell = Cell::ortho(8.0, 8.0, 8.0).unwrap();
        let xyz: Vec<[f64; 3]> = (0..4).map(|i| [1.0 + i as f64, 1.0, 1.0]).collect();
        let graph = cell_graph(&xyz, &cell, 2.0, None, None).unwrap();
        let copies = 4;
        let tiled = replicate_graph(&graph, copies).unwrap();
        assert_eq!(tiled.grid.ncell(), graph.grid.ncell() * copies);
        assert_eq!(tiled.neighbor.len(), graph.neighbor.len() * copies);
        assert_eq!(tiled.grid.ncell() / copies, graph.grid.ncell());
        assert_eq!(tiled.neighbor.len() / copies, graph.neighbor.len());
    }

    #[test]
    fn prefix_puts_each_cell_in_one_part() {
        let weight = [1u64, 1, 1, 1];
        let order = [0u32, 1, 2, 3];
        let assign = prefix_assign(&order, &weight, 2);
        assert_eq!(assign.iter().filter(|&&d| d == 0).count(), 2);
        assert_eq!(assign.iter().filter(|&&d| d == 1).count(), 2);
        assert!(assign.iter().all(|&d| d < 2));
    }
}
