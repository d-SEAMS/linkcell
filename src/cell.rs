//! Periodic cell: re-export of [`minimage::Cell`].
//!
//! Lattice vectors are the columns of H. Constructors, the fractional
//! wrap, [`Cell::dist2`], [`Cell::lattice_shift`], and
//! [`Cell::dist2_shifted`] live in minimage. This module keeps the
//! linkcell names so the walk does not own a second H.

pub use minimage::Cell;
