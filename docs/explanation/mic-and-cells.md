# Minimum image and linked cells

The production walk does not compute a fractional minimum image on
every pair. The Hinv formula `s = H^{-1} (r_j - r_i)`, wrap each
component to the central cell, then `r = H s`, is the brute-force
`Cell::dist2` path on a general box. The linked-cell search folds
once, then adds a lattice shift per neighbour cell.

## Fold once, then shift

Every active point is wrapped into the primary cell (`fractional`
then `cartesian`) and binned there. After that, a neighbour in a
periodic image is the same point plus an integer combination of the
lattice vectors. vesin and LAMMPS call those copies ghosts.

`Cell::lattice_shift(na, nb, nc)` is that translation. The pair
loop is `dist2_shifted`: `|q + shift - p|^2`. No wrap, no `Hinv`,
one subtract. The device walk uses the same shift: `na a + nb b +
nc c`. Orthorhombic boxes keep the cheap path (`H` diagonal).

Orthorhombic boxes skip the two 3x3 matvecs everywhere they can:
`fractional` divides by the three widths, `lattice_shift` multiplies
those widths, and brute `dist2` is three independent wraps.

## One shift per cell is wrong

The bins live in the primary cell. Cell `(ix + nx, iy, iz)` is the
same linked list as `(ix, iy, iz)`, but it is a different wrap:
`lattice_shift(1, 0, 0)`, not the identity.

If the walk records each unique bin once and applies a single
shift (the minimum image of the cell centre, or the first visit),
it drops the other images of those points. A pair close across a
face, edge, or corner can sit in a wrap that visit never reaches.

The walk in `knearest` therefore iterates integer cell offsets
`(dx, dy, dz)` and takes `div_euclid` for the shift. The same
primary bin is visited once per wrap the Chebyshev shell reaches.
That is the vesin-style construction: every wrap is a visit. One
shift per cell is correct only when every wrap is visited.

The offsets are taken in a Minkowski-reduced basis of the same
lattice, so a short vector such as `a - b` is a cell edge. Reduction
does not change which Cartesian images exist. It does not by itself
make the nearest image one of the 27 shifts: an unreduced dump can
need an original shift such as `(-7, 7, 0)`. The walk keeps expanding
shells until the k-th neighbour is inside the perpendicular distance
to the unvisited frontier, or until the shell covers a space
diagonal. `pairs_within` does not reduce: its shift is the caller's
`S`.

## Cutoff pair lists

vesin answers "who is inside radius r". This crate answers "who are
the k nearest, with the periodic image". The search takes no
cutoff. `cell_hint` only sizes the bins. Shells grow until the
k-heap is exact against the `reach * cell_min` bound.

nanoflann answers the same k question in Euclidean space without a
minimum-image convention. A periodic dump still needs the fold and
the wraps above.
