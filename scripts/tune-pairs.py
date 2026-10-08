"""Autotune the cutoff walk's thresholds and paths with Kernel Tuner.

    cargo build --release --features tune
    RAYON_NUM_THREADS=8 python3 scripts/tune-pairs.py [--n 4096] [--jitter 0.3]

Kernel Tuner (https://github.com/KernelTuner/kernel_tuner) compiles one
small C function per configuration with its tunable parameters as
`#define`s. That function sets the knobs through `lc_tune_set` and times
`pairs_within` through `lc_tune_pairs`, which returns the milliseconds
per call, the row count, and a checksum of the rows. Every configuration
is checked against the default one: the same `(i, j, S)` rows, and the
same `dist2` sum to within 1e-9. The thread count is the pool's, set by
`RAYON_NUM_THREADS` before the library starts, so each count is its own
run. The problem is the probe's: a lattice of `n` points in an 18 Å cube
at a 4 Å cutoff, moved by `--jitter` times the lattice step at random.

Knobs: `cell_hint` (bin edge, 0 for the cutoff), `split_pairs` (expected
pairs where the walk splits across threads), `grid_atoms` (atoms from
which a split walk builds its bins on several threads), `fused` (one
thread writes a full list from the tile), and `chunks` (bin ranges per
thread in a split search). Only the knobs that can act at the pool's
thread count are searched.
"""

import argparse
import ctypes
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBDIR = os.path.join(ROOT, "target", "release")
LIB = os.path.join(LIBDIR, "liblinkcell.so")

KERNEL = r"""
#include <stddef.h>
extern "C" int lc_tune_set(int, double);
extern "C" int lc_tune_pairs(const double *, size_t, const void *, double,
                             double, size_t, double *);

extern "C" float tune_pairs(const double *xyz, const double *simbox,
                            double *out, int n) {
    lc_tune_set(0, split_pairs);
    lc_tune_set(1, grid_atoms);
    lc_tune_set(2, fused);
    lc_tune_set(3, chunks);
    if (lc_tune_pairs(xyz, (size_t)n, simbox, @CUTOFF@, cell_hint, @REPS@, out) != 0)
        return 1.0e30f;
    return (float)out[0];
}
"""

DEFAULTS = {"split_pairs": 10_000, "grid_atoms": 1_024, "fused": 1, "chunks": 1, "cell_hint": 0.0}
KEYS = {"split_pairs": 0, "grid_atoms": 1, "fused": 2, "chunks": 3}


def lattice(n, boxl, jitter, seed):
    m = int(np.ceil(np.cbrt(n)))
    while m**3 < n:
        m += 1
    idx = np.arange(m**3)[:n]
    pts = np.stack([idx % m, (idx // m) % m, idx // (m * m)], axis=1).astype(np.float64)
    rng = np.random.default_rng(seed)
    pts += 0.5 + jitter * (rng.random(pts.shape) - 0.5)
    return np.ascontiguousarray(pts * boxl / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--box", type=float, default=18.0)
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=50, help="timed calls per Kernel Tuner run")
    ap.add_argument("--iterations", type=int, default=5, help="Kernel Tuner runs per configuration")
    ap.add_argument("--strategy", default="brute_force")
    ap.add_argument("--cache", default=None, help="Kernel Tuner cache file (JSON)")
    args = ap.parse_args()
    # Kernel Tuner writes its sources and objects in the working directory.
    os.chdir(os.environ.get("TMPDIR", "/tmp"))

    threads = int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count()))
    # The library, and the worker pool inside it, stay loaded while Kernel
    # Tuner loads and unloads one wrapper per configuration.
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    lib.lc_tune_pairs.restype = ctypes.c_int
    lib.lc_tune_set.argtypes = [ctypes.c_int, ctypes.c_double]

    from kernel_tuner import tune_kernel

    n = args.n
    xyz = lattice(n, args.box, args.jitter, args.seed)
    simbox = np.array([args.box, 0, 0, 0, args.box, 0, 0, 0, args.box, 0, 0, 0], dtype=np.float64)
    out = np.zeros(5, dtype=np.float64)

    def call(knobs, hint, reps):
        for key, slot in KEYS.items():
            lib.lc_tune_set(slot, float(knobs[key]))
        res = np.zeros(5, dtype=np.float64)
        ok = lib.lc_tune_pairs(
            xyz.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(n),
            simbox.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(args.cutoff),
            ctypes.c_double(hint),
            ctypes.c_size_t(reps),
            res.ctypes.data_as(ctypes.c_void_p),
        )
        if ok != 0:
            sys.exit("lc_tune_pairs failed")
        return res

    ref = call(DEFAULTS, 0.0, args.reps)
    print(f"default: {ref[0]:.4f} ms per call, {int(ref[1])} rows, {threads} threads", flush=True)

    edges = [0.0] + [round(f * args.cutoff, 4) for f in (0.5, 0.5625, 0.6, 0.75, 0.9, 1.125, 1.25, 1.5)]
    tune_params = {"cell_hint": edges}
    if threads > 1:
        tune_params["split_pairs"] = [2_500, 5_000, 10_000, 20_000, 40_000]
        tune_params["grid_atoms"] = [512, 1_024, 2_048, 4_096, 8_192, 1 << 30]
        tune_params["chunks"] = [1, 2, 3, 4]
        tune_params["fused"] = [DEFAULTS["fused"]]
    else:
        tune_params["fused"] = [1, 0]
        tune_params["split_pairs"] = [DEFAULTS["split_pairs"]]
        tune_params["grid_atoms"] = [DEFAULTS["grid_atoms"]]
        tune_params["chunks"] = [DEFAULTS["chunks"]]

    def verify(answer, result, atol=None):
        want, got = answer[2], result[2]
        same_rows = np.array_equal(want[1:4], got[1:4])
        same_d2 = abs(want[4] - got[4]) <= 1e-9 * max(1.0, abs(want[4]))
        return bool(same_rows and same_d2)

    source = KERNEL.replace("@CUTOFF@", repr(args.cutoff)).replace("@REPS@", str(args.reps))
    results, _ = tune_kernel(
        "tune_pairs",
        source,
        n,
        [xyz, simbox, out, np.int32(n)],
        tune_params,
        lang="C",
        compiler_options=["-O2", f"-L{LIBDIR}", "-llinkcell", f"-Wl,-rpath,{LIBDIR}"],
        answer=[None, None, ref.copy(), None],
        verify=verify,
        iterations=args.iterations,
        strategy=args.strategy,
        cache=args.cache,
        quiet=True,
    )
    ok = [r for r in results if isinstance(r.get("time"), float)]
    ok.sort(key=lambda r: r["time"])
    print("best configurations (ms per call):")
    for r in ok[:8]:
        print("  " + json.dumps({k: r[k] for k in ["time"] + list(tune_params)}))
    best = ok[0]
    knobs = {k: best[k] for k in KEYS}
    check = call(knobs, best["cell_hint"], args.reps * 4)
    again = call(DEFAULTS, 0.0, args.reps * 4)
    print(
        f"recheck: best {check[0]:.4f} ms, default {again[0]:.4f} ms "
        f"({again[0] / check[0]:.2f}x), {len(results)} configurations"
    )


if __name__ == "__main__":
    main()
