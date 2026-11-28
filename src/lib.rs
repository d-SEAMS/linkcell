//! Periodic linked-cell k-nearest neighbour search.
//!
//! vesin builds cutoff pair lists. nanoflann builds Euclidean KD-trees
//! without a minimum-image convention. This crate is the missing piece:
//! Allen and Tildesley's linked cells, a k-heap per source, and an
//! index box that grows one axis at a time until the k-th neighbour
//! cannot lie past an unvisited face. The pair itself is Rapaport's
//! cell shift, not a per-pair nearest integer. [`pairs_within`] is the
//! cutoff pair list with integer cell shifts (vesin/tonari `ijS`);
//! [`knearest`] unique-indexes.
//!
//! # Algorithm
//!
//! 1. **Fold** every position into the primary cell
//!    ([`Cell::fractional`], then [`Cell::cartesian`]).
//! 2. **Bin** the folded points on a fractional mesh. The bin count
//!    along each axis uses the perpendicular face width, so a sheared
//!    dump is not treated as orthogonal.
//! 3. **Grow an index box** of linked cells. The first visit is the
//!    3×3×3 around the home bin. After that, only an axis whose
//!    unvisited plane is still inside the k-th neighbour grows.
//!    Occupants are stored in cell order. A neighbour cell whose slab
//!    cannot beat the worst heap entry is skipped. The cap is the
//!    space diagonal over the minimum cell height, so a shift outside
//!    `{-1,0,1}` is still visited when it is the Euclidean image.
//!
//! The walk basis depends on the cell. An orthorhombic box, including
//! a rectangular film, is used as stored. A restricted triclinic box
//! (LAMMPS dump bounds, a CON angle, a hexagonal prism) is
//! tilt-reduced in place and the shift is triangular. A general
//! orientation is Minkowski-reduced. If either reduction fails, the
//! walk keeps the caller's cell. [`pairs_within`] keeps the caller's
//! basis on every call, because the reported shift is an integer
//! combination of that H.
//!
//! Pair distances in the walk are not a per-pair minimum-image wrap.
//! After the fold, a neighbour stencil `(jx, jy, jz)` contributes
//! [`Cell::lattice_shift`] and [`Cell::dist2_shifted`]: a Cartesian
//! subtract plus one lattice vector (the vesin / LAMMPS ghost trick).
//! Orthorhombic boxes set the [`Cell::is_ortho`] flag and skip the two
//! 3x3 matvecs: three independent wraps, and the shift is a scaled
//! diagonal. The Highway batch ([`dist2_ortho_diffs`]) is that wrap
//! for a raw difference. [`knearest_brute`] uses it on an orthorhombic
//! box. [`pairs_within`] keeps the stencil image and does not wrap
//! the shifted difference again.
//!
//! # Why a unique-cell stamp is wrong
//!
//! A stamp that visits each linked cell once, with a **single** shift
//! for every occupant, is not a minimum-image convention. Occupants of
//! one cell can need **different** images of the same source (a wide
//! bin, or a source near a face). The walk keys on the integer stencil
//! `(jx, jy, jz)`, including wraps: the same `rem_euclid` bin may
//! appear with more than one [`Cell::lattice_shift`].
//!
//! # Output
//!
//! [`knearest`] returns one [`Neighbors`] row per input point, nearest
//! first. [`knearest_into`] writes a packed `n * k` index buffer
//! (`-1` in unused slots), the same layout as `lc_knearest`.
//!
//! # Cost
//!
//! Binning is linear in the active points plus the bin count. More
//! than 1,000,000 bins on one axis, or more than 16,777,216 bins in
//! the mesh, is [`Error::TooManyCells`].
//!
//! [`pairs_within`] keeps the caller's basis. The image count is the
//! product of `2 * ceil(cutoff / width) + 1` over the three axes, and
//! a count above 2^24 is [`Error::TooManyImages`]. Those caps, and a
//! checked reservation of the estimated rows, run before any buffer is
//! reserved. A face width that is zero or non-finite is
//! [`Error::Degenerate`]. A positive finite width is not rejected by
//! its ratio to the longest edge. The image count depends on the
//! cutoff. The mesh edge is clamped to the shortest width, so a face
//! much shorter than the other edges can exceed the bin cap at every
//! cutoff. When both caps hold, the search returns the pairs. The
//! partner stencil is capped at 2^31 slots, enough for a reach of two
//! bins on every axis of a mesh at the bin cap; a larger reach box is
//! [`Error::TooManyImages`] and is not stored. With the default
//! bin edge the reach is at most a small constant (the occupancy
//! integer is clamped to `1..=6`) whenever that edge is shorter than
//! every box width. One bin that holds every point makes the candidate
//! tests quadratic.
//!
//! [`knearest`] takes no cutoff. Each source keeps a heap of `k`
//! entries (`k <= 16` stays on the stack) and grows an index box until
//! the worst entry is at most the squared gap to every unvisited plane,
//! or until the radius reaches the space diagonal of the walk cell in
//! units of the shortest bin edge. A filled lattice stops in the first
//! shells. `k = n - 1`, or every point in a few bins, reads every other
//! point.
//!
//! # Features
//!
//! `parallel` is on by default. An unmasked k-nearest query of at least
//! 8192 points runs on one linkcell thread team when this build has it: bin,
//! count, scan, fill, and select, with a private histogram and scratch
//! on each thread. Masked and smaller queries use rayon.
//! `--no-default-features` drops rayon and serializes the walk (re-enable
//! `capi` if the C ABI is still required). The per-source k-heap
//! (`KHeap`) stays on the stack for `k <= 16`; larger `k` spills to the
//! heap.
//!
//! The mesh fold, the k-nearest SoA scan, and cutoff pairs use AVX-512
//! (eight lanes), AVX2 (four lanes), or NEON (two lanes) when the CPU
//! has that path. AVX-512 also requires AVX-512VL, POPCNT, and FMA.
//! AVX2 is selected only when FMA is present too. Stored sums are
//! separate multiplies and adds, in the scalar order, so every stored
//! value matches the scalar path bit for bit. Pair screening may use
//! FMA inside its margin test. The value it stores is the direct sum
//! of squares. The fold floor is minimage `floor_fast`, truncation on
//! every architecture. `LINKCELL_SIMD` forces `scalar`, `avx2`, `avx512`, or
//! `neon`. [`simd_path`] reports the path in use. Neighbour rows, pair
//! rows, and the fold are bitwise identical across thread counts. The
//! `timing` feature exports `SimdTiming` for benchmark counters.
//!
//! # Examples
//!
//! Periodic image, not the raw Cartesian vector:
//!
//! ```
//! use linkcell::{knearest, Cell};
//!
//! # fn main() -> Result<(), linkcell::Error> {
//! let sim = Cell::ortho(10.0, 10.0, 10.0)?;
//! let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
//! let rows = knearest(&xyz, &sim, 1, None, None)?;
//! assert_eq!(rows[0].indices, vec![1]);
//! assert!((rows[0].dist2[0] - 0.64).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```
//!
//! Two occupants of one linked cell (the whole box is a single bin)
//! need different images of the source. A unique-cell stamp with one
//! shift cannot serve both.
//!
//! ```
//! use linkcell::{knearest, Cell};
//!
//! # fn main() -> Result<(), linkcell::Error> {
//! let sim = Cell::ortho(10.0, 10.0, 10.0)?;
//! let xyz = [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [9.8, 0.0, 0.0]];
//! let rows = knearest(&xyz, &sim, 2, None, Some(10.0))?;
//! assert_eq!(rows[0].indices, vec![2, 1]);
//! assert!((rows[0].dist2[0] - 0.09).abs() < 1e-12);
//! assert!((rows[0].dist2[1] - 0.81).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```
//!
//! Packed `n * k` indices, unused slots `-1`:
//!
//! ```
//! use linkcell::{knearest_into, Cell};
//!
//! # fn main() -> Result<(), linkcell::Error> {
//! let sim = Cell::ortho(10.0, 10.0, 10.0)?;
//! let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
//! let mut out = [0; 4];
//! knearest_into(&xyz, &sim, 2, None, None, &mut out)?;
//! assert_eq!(out, [1, -1, 0, -1]);
//! # Ok(())
//! # }
//! ```
//!
//! The C ABI (`lc_*`) is the hourglass waist: packed `n * k` indices,
//! matching squared distances, and `lc_pairs_within` for a cutoff
//! list with the caller's shift. C++ lives in `include/linkcell.hpp`
//! as a RAII header over that ABI.

#![deny(missing_docs)]

mod bins;
mod cell;
mod error;
mod knearest;
mod nd;
mod open_cell;
mod pairs;
mod partition;
#[cfg(any(feature = "capi", test))]
mod place;
mod pop;
mod replica;
#[cfg(feature = "request")]
#[allow(missing_docs, dead_code, clippy::all)]
#[rustfmt::skip]
mod neighbor_capnp;
#[cfg(feature = "request")]
mod request;
mod simd;
mod skin;
mod skin_simd;
mod team;
#[cfg(feature = "capi")]
mod vesin_abi;
#[cfg(feature = "capi")]
pub use vesin_abi::{NeighborCalc, VesinOptions};

pub use cell::{
    dist2_many, dist2_ortho_diffs, dist2_pairs, reduce_pairs, reduce_pairs_packed, wrap_many, Cell,
};
pub use error::Error;
pub use knearest::{
    knearest, knearest_brute, knearest_csr, knearest_into, knearest_into_d2, knearest_into_many,
    knearest_team_threads, pin_knearest_to_least_loaded, shutdown_knearest_team, Neighbors,
};
pub use nd::{
    ContiguousSpace, DbscanResult, Euclidean, ExactDist2, Metric, NeighborHit, NeighbourIndex,
    OwnedContiguousSpace, PairHit, PeriodicEuclidean, PeriodicSpace, Space, VpTreeIndex,
};
pub use open_cell::{cell_for_pbc, pairs_open, pairs_within_pbc, stencil_builds, PbcQuery};
pub use pairs::{
    pairs_within, pairs_within_columns, pairs_within_count, pairs_within_csr, pairs_within_into,
    Pair, PairColumns, PairCsr, Row, RowCursor, RowGrid,
};
pub use partition::{
    cell_graph, cell_grid, curve_split_device, derived_balance, derived_balance_f64, edge_cut,
    exclusive_prefix, fused_masks, halo, halo_width, hilbert_key, histogram, mark_reached,
    mesh_cut, morton_cmp, morton_key, owned_cells, partition, partition_graph, prefix_assign,
    profile_ns, profile_reset, rank_mask, replicate_graph, score, timer_resolution_ns, CellGraph,
    CellGrid, MeshCut, Method, PartRequest, PartScore, RankMask, DEFAULT_METHOD,
};
pub use pop::{
    efficiencies as pop_efficiencies, engage as pop_engage, prepare as pop_prepare,
    reset as pop_reset, snapshot as pop_snapshot,
};
pub use replica::ReplicaList;
pub use simd::{avx512_ready, simd_path};
#[cfg(feature = "timing")]
pub use simd::{simd_timing, SimdTiming};
pub use skin::SkinList;

#[cfg(feature = "parallel")]
thread_local! {
    static ON_CALLER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True while a search asked to run on one thread is on the calling thread.
#[cfg(feature = "parallel")]
pub(crate) fn on_caller() -> bool {
    rayon::current_thread_index().is_none() && ON_CALLER.with(std::cell::Cell::get)
}

/// Threads the walk may split over. A one-thread search on the caller is
/// one: asked from outside a pool, rayon builds its global pool at
/// `RAYON_NUM_THREADS` or the cpu count, and the walk then splits over it.
#[cfg(feature = "parallel")]
pub(crate) fn pool_width() -> usize {
    if on_caller() {
        1
    } else {
        rayon::current_num_threads()
    }
}

/// Runs `work` on this thread as a one-thread search.
#[cfg(all(feature = "parallel", feature = "capi"))]
pub(crate) fn run_on_caller<R>(work: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ON_CALLER.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(ON_CALLER.with(|flag| flag.replace(true)));
    work()
}

#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "tune")]
mod tune;
#[cfg(feature = "capi")]
pub use capi::*;
