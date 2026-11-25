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

/// Shortest periodic face altitude, as a fraction of the longest periodic
/// edge, at or below which the lattice is collapsed.
///
/// A linked-cell image count of `ceil(cutoff / altitude)` is not a
/// neighbour list on that face. The fraction is far below a real tilt
/// and above the roundoff that leaves a near-flat triple product just
/// large enough for [`Cell::from_vectors`].
pub(crate) const COLLAPSED_FACE: f64 = 1.0e-6;

/// `true` when a selected periodic face has collapsed.
///
/// `periodic[k]` selects axis `k`. The altitude is [`Cell::widths`] on
/// that axis and the edge is the length of the corresponding column of
/// [`Cell::h`]. No selected axis is not collapsed.
pub(crate) fn periodic_face_collapsed(cell: &Cell, periodic: [bool; 3]) -> bool {
    let widths = cell.widths();
    let columns = cell.h();
    let mut longest = 0.0;
    let mut narrowest = f64::INFINITY;
    let mut any = false;
    for axis in 0..3 {
        if !periodic[axis] {
            continue;
        }
        any = true;
        let row = columns[axis];
        let edge = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        if edge > longest {
            longest = edge;
        }
        if widths[axis] < narrowest {
            narrowest = widths[axis];
        }
    }
    if !any {
        return false;
    }
    if !longest.is_finite() || longest <= 0.0 {
        return true;
    }
    !narrowest.is_finite() || narrowest <= COLLAPSED_FACE * longest
}
