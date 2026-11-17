"""README quickstart plus an independent periodic k-nearest reference."""

import sys

import numpy as np

import linkcell


ORTHO_XYZ = np.array(
    [
        [0.20, 0.00, 0.00],
        [9.40, 0.00, 0.00],
        [0.20, 2.50, 0.10],
        [5.00, 5.00, 5.00],
        [0.00, 0.10, 9.70],
        [3.30, 1.20, 4.80],
        [8.90, 8.10, 0.40],
        [1.40, 7.20, 2.60],
        [6.60, 0.30, 8.80],
        [4.20, 9.50, 3.30],
        [2.20, 3.80, 6.10],
        [7.70, 4.40, 1.90],
    ],
    dtype=np.float64,
)

SHEAR_H = np.array(
    [[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 10.0]],
    dtype=np.float64,
)
SHEAR_FRAC = np.array(
    [
        [0.02, 0.10, 0.20],
        [0.90, 0.05, 0.15],
        [0.40, 0.70, 0.30],
        [0.15, 0.40, 0.80],
        [0.60, 0.20, 0.55],
        [0.33, 0.88, 0.10],
    ],
    dtype=np.float64,
)

TRI_H = np.array(
    [[10.0, 0.2, 0.1], [1.2, 8.5, -0.3], [0.4, 0.5, 9.2]],
    dtype=np.float64,
)
TRI_FRAC = np.array(
    [
        [0.02, 0.00, 0.00],
        [0.94, 0.00, 0.01],
        [0.10, 0.20, 0.30],
        [0.50, 0.50, 0.50],
        [0.80, 0.15, 0.90],
        [0.33, 0.71, 0.22],
        [0.61, 0.08, 0.44],
        [0.12, 0.88, 0.67],
    ],
    dtype=np.float64,
)


def image_dist2(pi, pj, lattice):
    best = 1.0e300
    for ia in range(-2, 3):
        for ib in range(-2, 3):
            for ic in range(-2, 3):
                shift = ia * lattice[0] + ib * lattice[1] + ic * lattice[2]
                delta = pj + shift - pi
                best = min(best, float(delta @ delta))
    return best


def library_knn(xyz, cell, k):
    indices, dist2 = linkcell.knearest(np.ascontiguousarray(xyz), np.ascontiguousarray(cell), k)
    return np.from_dlpack(indices), np.from_dlpack(dist2)


def check(label, xyz, cell, lattice, k):
    indices, dist2 = library_knn(xyz, cell, k)
    max_abs = 0.0
    n = xyz.shape[0]
    for i in range(n):
        ranked = sorted(
            ((image_dist2(xyz[i], xyz[j], lattice), j) for j in range(n) if j != i)
        )
        gap = ranked[k][0] - ranked[k - 1][0]
        if gap <= 1.0e-6:
            raise SystemExit(f"{label}: ambiguous neighbour gap {gap} at {i}")
        for rank in range(k):
            got_j = int(indices[i, rank])
            got_d = float(dist2[i, rank])
            exp_d, exp_j = ranked[rank]
            err = abs(got_d - exp_d)
            if got_j != exp_j or err > 1.0e-8:
                raise SystemExit(
                    f"{label}: i={i} rank={rank} got ({got_j}, {got_d:.16g}) "
                    f"expected ({exp_j}, {exp_d:.16g})"
                )
            max_abs = max(max_abs, err)
    print(f"{label} ok n={n} k={k} max_abs_d2={max_abs:.3e}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: periodic.py <version>")
    expected = sys.argv[1]
    print("linkcell", linkcell.__version__, linkcell.__file__)
    if linkcell.__version__ != expected:
        raise SystemExit(f"version {linkcell.__version__} != {expected}")

    xyz = np.array([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    nn, d2 = linkcell.knearest(xyz, cell, 1)
    nn = np.from_dlpack(nn)
    d2 = np.from_dlpack(d2)
    print("quickstart", nn.tolist(), d2.tolist())
    if int(nn[0, 0]) != 1 or int(nn[1, 0]) != 0:
        raise SystemExit(f"quickstart indices {nn.tolist()}")
    if abs(float(d2[0, 0]) - 0.64) > 1.0e-12 or abs(float(d2[1, 0]) - 0.64) > 1.0e-12:
        raise SystemExit(f"quickstart dist2 {d2.tolist()}")

    ortho = np.diag([10.0, 10.0, 10.0])
    check("ortho-lengths", ORTHO_XYZ, np.array([10.0, 10.0, 10.0]), ortho, 4)
    check("ortho-matrix", ORTHO_XYZ, ortho, ortho, 4)
    check("sheared", SHEAR_FRAC @ SHEAR_H, SHEAR_H, SHEAR_H, 3)
    check("triclinic", TRI_FRAC @ TRI_H, TRI_H, TRI_H, 3)


if __name__ == "__main__":
    main()
