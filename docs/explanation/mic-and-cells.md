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
shells until the k-th neighbour is no farther than the unvisited
frontier plane, or until the shell covers a space diagonal.
`pairs_within` does not reduce: its shift is the caller's `S`.

## Cutoff pair lists

vesin answers "who is inside radius r". This crate answers "who are
the k nearest, with the periodic image". The search takes no
cutoff. `cell_hint` only sizes the bins. Shells grow until the
k-th neighbour is certified against the frontier plane. The device
walk still stops on the looser `reach * cell_min` bound, which is
that plane distance when the source sits on the outer face of its
cell.

nanoflann answers the same k question in Euclidean space without a
minimum-image convention. A periodic dump still needs the fold and
the wraps above.

## What the literature actually specifies

The linked-cell search is Quentrec and Brot, *J. Comput. Phys.* **13**,
430 (1973). Allen and Tildesley, *Computer Simulation of Liquids*, is
the textbook form: a linked list, a half stencil of 13 neighbour
cells, and a per-pair nearest integer. Rapaport, *The Art of
Molecular Dynamics Simulation* (Cambridge, 2nd ed., 2004), stores the
same cells and forms the pair as the Cartesian difference plus the
lattice shift of that neighbour cell. `dist2_shifted` is Rapaport's
pair, not Allen and Tildesley's `ANINT`.

Exact k-nearest neighbours, with no cutoff, are not in those books.
Bentley, Weide, and Yao, *ACM Trans. Math. Softw.* **6**, 563 (1980),
state the cell technique for that problem: bin a bounded distribution,
then expand square layers, and stop when every unsearched bin misses
the ball of the current neighbour. The Chebyshev shells and the
frontier plane are that test on a periodic lattice. A precomputed
stencil (LAMMPS `NStencil`, Allen and Tildesley's `MAP`) is the same
set when the radius is an input. A k-nearest radius is not known
until the heap fills, so the shell grows.

Nguyen and Stehlé, *ACM Trans. Algorithms* **5**, 46 (2009), produce
the Minkowski-reduced basis this walk bins in. Conway and Sloane,
*Proc. R. Soc. Lond. A* **436**, 55 (1992), prove a different fact:
every three-dimensional lattice has an obtuse superbase (Selling,
1874), and the Voronoi vectors are then `{-1,0,1}` combinations of
that superbase. Nearest-integer rounding in a Minkowski basis is not
that certificate. The walk does not stop at 27 images.

Welling and Germano, *Comput. Phys. Commun.* **182**, 611 (2011),
compare the later cell-list variants on the same footing. Gonnet's
projection sort beats a plain cell list only above roughly 19
particles per cell. The cubic hot path has one particle per cell, so
the walk computes the distance instead of projecting.

## Codes that answer a different question

LAMMPS, GROMACS, HOOMD-blue, OpenMM, and NAMD build a cutoff list,
usually with a skin and a rebuild. A half stencil (Newton's third
law, GROMACS's 14 images, Allen and Tildesley's 13) stores each pair
once. That is wrong for k-nearest neighbours: `j` can be among the
`k` nearest of `i` while `i` is not among the `k` nearest of `j`.
GROMACS and OpenMM also require a tilt-reduced triclinic box and a
cutoff short enough that one image suffices. Copying that stop onto
an unreduced cell drops a closer shift.

vesin and ASE `neighbor_list` are cutoff cell lists. The stencil is
sized from the face distances, so a long cutoff walks past 27 images,
and the reported vector is `r_j - r_i + S H`. Sorting that list is a
k-nearest answer only when the cutoff already contains the k-th
neighbour. freud (`AABBQuery` / `LinkCell`, Ramasubramani et al.,
*Comput. Phys. Commun.* **254**, 107275, 2020) does take a neighbour
count on a triclinic box. The query grows a ball. It is the closest
peer, and it is still a ball of limited images, not this frontier
certificate. `scipy.spatial.cKDTree` with `boxsize` is exact on an
orthorhombic torus and has no triclinic metric.

A bounding-volume hierarchy (Howard, Anderson, Nikoubashman, Glotzer,
and Panagiotopoulos, *Comput. Phys. Commun.* **203**, 45, 2016) wins
for a cutoff when particle sizes differ. HOOMD's own comparison is
that the cell list wins when every particle has essentially the same
range. Verlet skins reuse a cutoff list across timesteps. This
search is one configuration and has no radius it may legally freeze.

## Geometry, algebra, robotics, tensors, graphs

These are the places a molecular-dynamics reading does not look.
None of them replaces the cell walk on a filled cubic lattice. Two
of them are the exact certificates for the skewed cell, and they
answer a different query than "the k nearest sites."

**Closest lattice vector.** Agrell, Eriksson, Vardy, and Zeger,
*IEEE Trans. Inf. Theory* **48**, 2201 (2002), survey closest-point
search. Schnorr–Euchner enumeration walks lattice layers in order of
orthogonal distance and stops when the next layer is farther than the
best point already found. The layer index is not confined to
`{-1,0,1}`. Babai's nearest plane (*Combinatorica* **6**, 1, 1986)
keeps only one layer per dimension and can miss the same image a
27-image loop misses. McKilliam, Grant, and Clarkson, *SIAM J.
Discrete Math.* **28**, 1405 (2014), start from the component-wise
floor in an obtuse superbasis and add a series of relevant vectors.
Every lattice of dimension less than 4 has such a superbasis, and
the series reaches a closest lattice point in at most as many steps
as the dimension. The vector it returns is the closest lattice point
to one displacement. It does not say which other site is the k-th
neighbour.
On the cubic hot path the closest coefficient is already inside the
first shell, so the certificate does not change the neighbour list
and does not remove any bin visits.

**Periodic Delaunay.** Caroli and Teillaud, ESA 2009, compute the
Delaunay triangulation of the cubic flat torus, keeping one copy of
each point once the complex is simplicial, and a finite-sheeted cover
otherwise. CGAL's `Periodic_3_Delaunay_triangulation_3` implements
that cubic torus and can return the nearest vertex. Osang,
Rouxel-Labbé, and Teillaud, ESA 2020, extend it to a general lattice
by reducing to an obtuse superbase and folding each point with a
closest-vector query. The dual is an exact 1-nearest structure. The
link of a vertex is not the 4 nearest neighbours: a site's fourth
neighbour need not be a Delaunay edge. Dickerson and Eppstein,
*Comput. Geom.* **5**, 277 (1996), recover the k nearest from a
Delaunay search in `O(k n log n)` in Euclidean space, with no
periodic quotient. Lee, *IEEE Trans. Comput.* **C-31**, 478 (1982),
builds the order-k Voronoi diagram in the plane. The proved cost of
the 3D periodic triangulation is the same `O(n^2)` worst case as a
Delaunay triangulation in `R^3`. A cell list on this sample is linear.
Callahan and Kosaraju, *J. ACM* **42**, 67 (1995), get all k-nearest
neighbours from a well-separated pair decomposition in Euclidean
space. A decomposition of one fundamental domain misses a pair that
is close only after a lattice translation. No tensor train, Gröbner
basis, or cylindrical algebraic decomposition turned up as the
decision procedure for this query. The identity that decides the
closest lattice vector in three dimensions is the obtuse superbase
already cited above.

**Robotics.** Yershova and LaValle, *IEEE Trans. Robot.* **23**, 151
(2007), extend Arya and Mount's kd-tree to a product of lines,
circles, and `RP^3`. On `(S^1)^3` the distance is the Euclidean
minimum image of an orthorhombic box, and a node is discarded only
when the circular distance to its rectangle is already worse than
the best point seen. The proof is for one neighbour. The same prune
is the wrong Euclidean distance on a skewed cell, because the metric
no longer splits across axes. Ichnowski and Alterovitz, WAFR 2014,
search `SO(3)` and `SE(3)` with four kd-trees on the quaternion
3-sphere. That distance is a great-arc plus a translation, not
`R^3` modulo a lattice. OMPL's torus state space is the embedded
doughnut surface, and GNAT answers queries in the metric that
surface supplies. Cover trees and GNAT likewise answer queries in
the metric they are given. Neither constructs the lattice images of
a periodic box. Pan,
Lauterbach, and Manocha, IROS 2010, use locality-sensitive hashing,
which they state is approximate.

**Tensors.** Jégou, Douze, and Schmid, *IEEE TPAMI* **33**, 117
(2011), product-quantize a Cartesian product of subspaces. The
codes are approximate. Pham and Pagh, KDD 2013, sketch a polynomial
kernel. Novikov, Gneushev, Kadeishvili, and Oseledets, arXiv:2410.04462
(2024), use a tensor train as an approximate point-cloud index, with
no periodic cell. The orthorhombic minimum image is a sum of three
circular distances, which is why the ortho path never multiplies by
`H`. That is the product metric above, not a tensor decomposition.

**Graph construction.** Crystal networks take a cutoff ball, or k
neighbours from inside a ball. Xie and Grossman, *Phys. Rev. Lett.*
**120**, 145301 (2018), keep 12 neighbours inside a fixed radius, so
the 12th neighbour outside that ball is absent. ALIGNN grows the
radius until 12 neighbours exist and then keeps the whole shell.
fairchem's periodic radius graph repeats the lattice out to
`ceil(radius / face height)` and then caps the list. Park and
Wolverton, *Phys. Rev. Materials* **4**, 063801 (2020), connect
Voronoi neighbours instead. A Voronoi face can be longer than a
non-neighbour, so that graph is not the k nearest sites. Ruff,
Reiser, Stühmer, and Friederich, *Digital Discovery* **3**, 594
(2024), compare periodic k-nearest, radius, and Voronoi edges and
keep k = 24. NequIP, Allegro, and MACE consume a cutoff list. None
of these is an uncapped Euclidean k-nearest walk, and none of them
adds a bin test this search does not already apply.
