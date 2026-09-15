//! Periodic linked-cell k-nearest neighbour search.
//!
//! vesin builds cutoff pair lists. nanoflann builds Euclidean KD-trees
//! without a minimum-image convention. This crate is the missing piece:
//! Allen and Tildesley's linked cells, a k-heap per source, expanding
//! Chebyshev shells until the k-th neighbour cannot lie outside the
//! visited cube. [`pairs_within`] is the cutoff pair list with integer
//! cell shifts (vesin/tonari `ijS`); [`knearest`] unique-indexes.
//!
//! # Algorithm
//!
//! 1. **Fold** every position into the primary cell
//!    ([`Cell::fractional`], then [`Cell::cartesian`]).
//! 2. **Bin** the folded points on a fractional mesh. The bin count
//!    along each axis uses the perpendicular face width, so a sheared
//!    dump is not treated as orthogonal.
//! 3. **Expand Chebyshev shells** of linked cells (`reach = 1, 2, ...`).
//!    The walk stops when the k-heap is full and the k-th squared
//!    distance cannot hide beyond `(reach * min_subcell_height)^2`.
//!
//! Pair distances in the walk are not a per-pair minimum-image wrap.
//! After the fold, a neighbour stencil `(jx, jy, jz)` contributes
//! [`Cell::lattice_shift`] and [`Cell::dist2_shifted`]: a Cartesian
//! subtract plus one lattice vector (the vesin / LAMMPS ghost trick).
//! Orthorhombic boxes set the [`Cell::is_ortho`] flag and skip the two
//! 3x3 matvecs: three independent wraps, and the shift is a scaled
//! diagonal.
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
//! # Features
//!
//! `parallel` (rayon over sources) is on by default for wall-clock.
//! `--no-default-features` drops rayon and serializes the walk (re-enable
//! `capi` if the C ABI is still required). The per-source k-heap
//! (`KHeap`) stays on the stack for `k <= 16`; larger `k` spills to the
//! heap.
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
//! The C ABI (`lc_*`) is the hourglass waist: packed `n * k` indices
//! and matching squared distances. C++ lives in `include/linkcell.hpp`
//! as a RAII header over that ABI.

#![deny(missing_docs)]

mod cell;
mod error;
mod knearest;
mod pairs;

pub use cell::Cell;
pub use error::Error;
pub use knearest::{
    knearest, knearest_brute, knearest_into, knearest_into_d2, knearest_into_many, Neighbors,
};
pub use pairs::{pairs_within, Pair};

#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "capi")]
pub use capi::*;
