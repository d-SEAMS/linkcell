#!/usr/bin/env python3
"""Boundary checks for the floating geometry used by the cell grid.

Cell weights, edge cuts, and the owner index are exact integers. Ozaki
schemes, error-free transforms, and symbolic regression do not apply to
those operations: there is no rounded dot product and no expression to
fit. The floating work is the cutoff, the bin width, and the halo radius.
This script checks those with rational arithmetic and a decimal interval.
"""

import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import sympy as sp


def balance_identity():
    parts, w_max, total = sp.symbols("parts w_max total", integer=True, positive=True)
    ratio = 1 + parts * w_max / total
    num = total + parts * w_max
    difference = sp.simplify(ratio - num / total)
    if difference != 0:
        raise SystemExit(f"balance identity failed: {difference}")
    print("sympy balance 1 + parts * w_max / total = (total + parts * w_max) / total")


def bin_radius_covers():
    """ceil(w / h) * h covers w for positive h."""
    samples = [
        (Fraction(75, 10), Fraction(4, 1)),
        (Fraction(43065, 10000), Fraction(3, 1)),
        (Fraction(1, 3), Fraction(1, 7)),
    ]
    for width, edge in samples:
        bins = -(-width // edge)  # positive ceil via floor division
        if bins * edge < width:
            raise SystemExit(f"bin radius missed {width} {edge}")
    print("rational bin_radius covers the checked widths")


def cutoff_interval():
    sigma = Decimal("2.3925")
    factor = Decimal("1.8")
    with localcontext() as ctx:
        ctx.prec = 50
        product = factor * sigma
    # The panel prints 4.306500. The exact product is 4.3065.
    if product != Decimal("4.3065"):
        raise SystemExit(f"mW cutoff product {product}")
    lo = product - Decimal("1e-12")
    hi = product + Decimal("1e-12")
    if not (lo < Decimal("4.3065") < hi):
        raise SystemExit("cutoff interval missed 4.3065")
    print(f"decimal mW cutoff {product} inside [{lo}, {hi}]")
    print("ozaki not_applicable exact integer weights and cuts")
    print("error_free_transform not_applicable no rounded dot product in the cut")
    print("pysr not_applicable the balance formula is derived, not fitted")


def main():
    balance_identity()
    bin_radius_covers()
    cutoff_interval()
    return 0


if __name__ == "__main__":
    sys.exit(main())
