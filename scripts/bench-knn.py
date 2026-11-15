"""k-nearest timings against OVITO on the same cubic oxygen lattice.

The lattice is the one `examples/knn_bench.rs` builds: nside^3 sites at
spacing 3.125 in an orthorhombic box, k=4, bin edge 3.0. One process
runs every row. Contended rows share the machine with one tight loop
per core. Bound rows set OMP_PROC_BIND=close; unbound rows unset it.

    python3 scripts/bench-knn.py [--reps 11] [--pin]

Prints median and IQR in milliseconds. No host name.
"""

import argparse
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACING = 3.125
K = 4


def cubic(nside):
    xyz = []
    for iz in range(nside):
        for iy in range(nside):
            for ix in range(nside):
                xyz.append((ix * SPACING, iy * SPACING, iz * SPACING))
    return xyz


def nside_of(n):
    return max(1, round(n ** (1.0 / 3.0)))


def quartiles(samples):
    ordered = sorted(samples)
    # Inclusive method matches statistics.quantiles(..., method="inclusive").
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return statistics.median(ordered), q3 - q1, ordered[0], ordered[-1]


def bimodal(samples):
    ordered = sorted(samples)
    if len(ordered) < 4:
        return False
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    biggest = max(gaps)
    med = statistics.median(ordered)
    # A gap larger than a quarter of the median, and not at the extreme
    # sample, splits the run into two clusters.
    index = gaps.index(biggest)
    return biggest > 0.25 * med and 0 < index < len(gaps) - 1


def start_burn(cores):
    code = (
        "import itertools\n"
        "x = 0\n"
        "for _ in itertools.count():\n"
        "    x = (x * 1664525 + 1013904223) & 0xFFFFFFFF\n"
    )
    procs = []
    for _ in range(cores):
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    time.sleep(0.3)
    return procs


def stop_burn(procs):
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def linkcell_samples(binary, n, reps, bind, pin):
    env = os.environ.copy()
    env.pop("OMP_PROC_BIND", None)
    env.pop("GOMP_CPU_AFFINITY", None)
    env.pop("KNN_BENCH_PIN", None)
    if bind:
        env["OMP_PROC_BIND"] = "close"
    if pin:
        env["KNN_BENCH_PIN"] = "1"
    env["KNN_BENCH_N"] = str(n)
    env["KNN_BENCH_K"] = str(K)
    env["KNN_BENCH_REPS"] = str(reps)
    out = subprocess.check_output([binary], env=env, text=True)
    header = ""
    samples = []
    for line in out.splitlines():
        if line.startswith("n="):
            header = line
        elif line.startswith("sample_ms="):
            samples.append(float(line.split("=", 1)[1]))
    return header, samples


def ovito_samples(n, reps):
    import numpy as np
    from ovito.data import DataCollection, NearestNeighborFinder, Particles, SimulationCell

    side = nside_of(n)
    xyz = np.asarray(cubic(side), dtype=np.float64)
    box = side * SPACING
    data = DataCollection()
    cell = SimulationCell(pbc=(True, True, True))
    cell[:, :] = [
        [box, 0.0, 0.0, 0.0],
        [0.0, box, 0.0, 0.0],
        [0.0, 0.0, box, 0.0],
    ]
    particles = Particles()
    particles.create_property("Position", data=xyz)
    data.objects.append(particles)
    data.objects.append(cell)

    def once():
        finder = NearestNeighborFinder(K, data)
        idx, vec = finder.find_all()
        return idx, vec

    once()
    samples = []
    idx = vec = None
    for _ in range(reps):
        started = time.perf_counter()
        idx, vec = once()
        samples.append((time.perf_counter() - started) * 1_000.0)
    if idx.shape != (len(xyz), K) or not np.isfinite(vec).all():
        raise RuntimeError("OVITO neighbours are not a finite k-list")
    return samples, len(xyz)


def row(name, samples):
    med, iqr, lo, hi = quartiles(samples)
    flag = "bimodal" if bimodal(samples) else "unimodal"
    listed = " ".join(f"{v:.3f}" for v in samples)
    print(
        f"{name} median_ms={med:.3f} iqr_ms={iqr:.3f} min_ms={lo:.3f} max_ms={hi:.3f} {flag} n={len(samples)}"
    )
    print(f"  samples_ms {listed}")
    return med, flag


def run_rows(args):
    cores = os.cpu_count() or 1
    sizes = [110592, 13824]
    policy = os.environ.get("OMP_WAIT_POLICY", "unset")
    print(f"cores={cores} k={K} reps={args.reps} spacing={SPACING} policy={policy}", flush=True)
    burn = []
    try:
        for load, contended in (("idle", False), ("contended", True)):
            if contended:
                burn = start_burn(cores)
            else:
                stop_burn(burn)
                burn = []
            for n in sizes:
                for bind_name, bind in (("unbound", False), ("bound", True)):
                    header, samples = linkcell_samples(args.binary, n, args.reps, bind, False)
                    print(f"linkcell header {header}", flush=True)
                    row(f"linkcell n={nside_of(n) ** 3} {load} {bind_name} policy={policy}", samples)
                    if args.pin and n == 110592 and not bind:
                        header, samples = linkcell_samples(args.binary, n, args.reps, False, True)
                        print(f"linkcell header {header}", flush=True)
                        row(
                            f"linkcell n={nside_of(n) ** 3} {load} unbound pin policy={policy}",
                            samples,
                        )
                o_samples, o_n = ovito_samples(n, args.reps)
                row(f"ovito n={o_n} {load} policy={policy}", o_samples)
    finally:
        stop_burn(burn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--binary", default=os.path.join(ROOT, "target", "release", "examples", "knn_bench"))
    ap.add_argument("--pin", action="store_true", help="also time the opt-in least-loaded pin")
    args = ap.parse_args()
    if args.reps < 8:
        raise SystemExit("reps must be at least 8")
    # Each policy is a fresh process so both linkcell and OVITO initialise
    # OpenMP under that same environment. The library does not override it.
    if os.environ.get("LC_BENCH_CHILD") != "1":
        script = os.path.abspath(__file__)
        for policy in ("unset", "PASSIVE"):
            env = os.environ.copy()
            env["LC_BENCH_CHILD"] = "1"
            env.pop("OMP_PROC_BIND", None)
            env.pop("GOMP_CPU_AFFINITY", None)
            if policy == "PASSIVE":
                env["OMP_WAIT_POLICY"] = "PASSIVE"
            else:
                env.pop("OMP_WAIT_POLICY", None)
            cmd = [sys.executable, script, "--reps", str(args.reps), "--binary", args.binary]
            if args.pin:
                cmd.append("--pin")
            print(f"policy_block={policy}", flush=True)
            subprocess.check_call(cmd, env=env)
        return
    run_rows(args)


if __name__ == "__main__":
    main()
