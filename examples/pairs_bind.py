"""Python cutoff-list probe: the call eOn makes, numpy arrays out.

Env: PAIRS_N (default 4096), PAIRS_REPS (default 8), PAIRS_HALF (0).
RAYON_NUM_THREADS is read when the extension starts. The line reports
the first call, then the mean of the timed calls after three warmups.

numpy's OpenBLAS threads spin on the same cores for a while after
import and after each BLAS call, and they slow any wide thread pool in
the process. The probe keeps that pool at one thread unless
OPENBLAS_NUM_THREADS is already set.
"""

import math
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import linkcell  # noqa: E402


def fill(n, boxl=18.0):
    m = math.ceil(n ** (1.0 / 3.0) - 1e-9)
    while m**3 < n:
        m += 1
    g = (np.arange(m) + 0.5) * boxl / m
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)[:n]
    return np.ascontiguousarray(xyz, dtype=np.float64)


def call(xyz, cell, half):
    i, j, s, d2 = linkcell.pairs_within(xyz, cell, 4.0, half=half)
    return np.from_dlpack(i), np.from_dlpack(j), np.from_dlpack(s), np.from_dlpack(d2)


def main():
    n = int(os.environ.get("PAIRS_N", "4096"))
    reps = max(1, int(os.environ.get("PAIRS_REPS", "8")))
    half = os.environ.get("PAIRS_HALF", "0") == "1"
    threads = os.environ.get("RAYON_NUM_THREADS", "default")
    xyz = fill(n)
    cell = np.ascontiguousarray([18.0, 18.0, 18.0], dtype=np.float64)
    early = []
    for _ in range(3):
        t = time.perf_counter()
        out = call(xyz, cell, half)
        early.append((time.perf_counter() - t) * 1e3)
    t0 = time.perf_counter()
    acc = 0
    for _ in range(reps):
        acc += call(xyz, cell, half)[0].shape[0]
    ms = (time.perf_counter() - t0) * 1e3 / reps
    print(
        f"py n={n} half={int(half)} threads={threads} pairs={out[0].shape[0]} "
        f"cold={early[0]:.3f} call2={early[1]:.3f} call3={early[2]:.3f} "
        f"reps={reps} ms_per={ms:.4f} acc={acc}"
    )


if __name__ == "__main__":
    main()
