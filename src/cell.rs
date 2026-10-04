//! Periodic cell: re-export of [`minimage::Cell`].
//!
//! Lattice vectors are the columns of H. Constructors, the fractional
//! wrap, [`Cell::dist2`], [`Cell::lattice_shift`], and
//! [`Cell::dist2_shifted`] live in minimage. This module keeps the
//! linkcell names so the walk does not own a second H.
//!
//! [`dist2_ortho_diffs`] is the orthorhombic Highway kernel: packed
//! `dx, dy, dz`, one reciprocal per axis, then
//! `dr -= box * round(dr / box)` after `abs`. That is a minimum-image
//! wrap of a raw difference. [`crate::pairs_within`] does not call it.
//! The stencil has already chosen the image `S`, and the row distance
//! is [`Cell::dist2_shifted`]. A second wrap pulls a far image back
//! inside the cutoff and labels it with the wrong `S`.
//! [`crate::knearest_brute`] does call it on an orthorhombic box,
//! where that wrap is the whole distance.
//!
//! ```
//! use linkcell::dist2_ortho_diffs;
//!
//! # fn main() -> Result<(), linkcell::Error> {
//! let dx = [9.2];
//! let dy = [0.0];
//! let dz = [0.0];
//! let mut out = [0.0];
//! dist2_ortho_diffs(&dx, &dy, &dz, 10.0, 10.0, 10.0, &mut out)?;
//! assert!((out[0] - 0.64).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

pub use minimage::Cell;
pub use minimage::{
    dist2_many, dist2_ortho_diffs, dist2_pairs, reduce_pairs, reduce_pairs_packed, wrap_many,
};
