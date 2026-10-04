#!/usr/bin/env python3
"""Distance-test and flop counts for the cutoff list and vesin's cell list.

vesin 0.6.2 (`CellList::foreach_pair`) walks the full (2 r + 1)^3 stencil and
calls the distance callback on every ordered pair. `pairs_within` keeps one
direction of that stencil and the upper triangle of the home cell, then writes
both `(i, j, S)` and `(j, i, -S)` from that one test.

When the bin edge is the cutoff, the search radius is 1 and the gap test
rejects nothing inside that stencil. The two loops then differ by exactly two.
"""

import sympy as sp


def main() -> None:
    rho = sp.symbols("rho", integer=True, positive=True)
    # Full 3x3x3 stencil, including the home cell. Self pairs at shift 0 drop.
    t_vesin = 27 * rho - 1
    # Home triangle plus the 13 cells with a positive half-direction.
    t_ours = (rho - 1) / 2 + 13 * rho
    ratio = sp.simplify(t_vesin / t_ours)
    difference = sp.factor(t_vesin - 2 * t_ours)

    # vesin CellShift::cartesian is an integer vector times the 3x3 box, on
    # every candidate. The distance is then |r_j - r_i + S H|^2.
    mul, add = sp.symbols("mul add", integer=True, positive=True)
    vesin_shift = 9 * mul + 6 * add
    vesin_delta = 3 * add + 3 * add  # r_j - r_i, then add the shift
    vesin_dot = 3 * mul + 2 * add
    flops_vesin = sp.simplify(vesin_shift + vesin_delta + vesin_dot)
    # The Cartesian shift is one value per cell pair, outside the atom loop.
    flops_ours = 3 * add + 3 * mul + 2 * add
    flop_ratio = sp.simplify(
        sp.simplify(flops_vesin.subs({mul: 1, add: 1}))
        * t_vesin
        / (sp.simplify(flops_ours.subs({mul: 1, add: 1})) * t_ours)
    )

    print("per atom, rho atoms in the home cell")
    print(f"  vesin tests  = {t_vesin}")
    print(f"  ours tests   = {t_ours}")
    print(f"  vesin / ours = {ratio}")
    print(f"  vesin - 2*ours = {difference}")
    print(f"  vesin flops / candidate = {flops_vesin}")
    print(f"  ours flops / candidate  = {flops_ours}")
    print(f"  flop ratio (vesin / ours) = {flop_ratio}")

    # Consumer cube: 18 Å box, 4 Å cutoff, 16^3 lattice.
    n_bin = sp.Integer(4)
    atoms = n_bin**3 * 64
    rho_c = sp.Integer(64)
    hits = 729088  # measured full list; both codes return this set
    tests_v = int(t_vesin.subs(rho, rho_c) * atoms)
    tests_u = int(t_ours.subs(rho, rho_c) * atoms)
    fv = int(flops_vesin.subs({mul: 1, add: 1}))
    fo = int(flops_ours.subs({mul: 1, add: 1}))
    print()
    print(f"18 Å cube, cutoff 4 Å, {atoms} atoms, {rho_c} per cell")
    print(f"  vesin distance tests = {tests_v}")
    print(f"  ours distance tests  = {tests_u}")
    print(f"  vesin flops          = {tests_v * fv}")
    print(f"  ours flops           = {tests_u * fo}")
    # Separate arrays: size_t[2] + int32[3]. Pair is 40 bytes with the pad.
    bytes_vesin = hits * (8 * 2 + 4 * 3)
    bytes_pair = hits * 40
    bytes_scratch = (hits // 2) * (4 + 8 + 8)
    print(f"  full-list rows       = {hits}")
    print(f"  vesin shift bytes    = {bytes_vesin}")
    print(f"  pair bytes           = {bytes_pair}")
    print(f"  scratch bytes        = {bytes_scratch}")
    print(
        "  extra traffic is the scratch and the 40-byte row; "
        "the distance loop is the smaller one"
    )


if __name__ == "__main__":
    main()
