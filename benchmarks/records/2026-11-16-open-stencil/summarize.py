"""Median, IQR and before/after ratio per bench point from run-ab.sh logs.

    python3 summarize.py RAW_DIR BEFORE AFTER [AFTER ...]

Samples of one point are pooled over rounds. open_band and knn
contribute every timed sample, open_cell the median of each process,
and pairs the per-call mean of each process. The ratio is the after
median over the before median, so below 1 is faster. The 95% interval
of that ratio is a percentile bootstrap (seed 1, 4000 resamples).
A point regresses when the whole interval sits above 1 and the median
moved by more than the larger of the two IQRs.
"""

import glob
import os
import random
import re
import statistics
import sys


def quartiles(samples):
    q1, _, q3 = statistics.quantiles(sorted(samples), n=4, method="inclusive")
    return statistics.median(samples), q3 - q1


def collect(raw, variant):
    points = {}
    builds = {}

    def add(key, values):
        points.setdefault(key, []).extend(values)

    for path in sorted(glob.glob(os.path.join(raw, f"open_band.{variant}.r*.log"))):
        for line in open(path):
            if not line.startswith("point="):
                continue
            fields = dict(item.split("=", 1) for item in line.split())
            key = f"open_band {fields['point']}"
            add(key, [float(v) for v in fields["samples_us"].split(",")])
            builds.setdefault(key, set()).add(fields["builds_per_sample"])
    for path in sorted(glob.glob(os.path.join(raw, f"open_cell.{variant}.r*.log"))):
        text = open(path).read()
        for arm in ("open_ms", "padded_ms"):
            value = re.search(rf"^{arm} median=([0-9.]+)", text, re.M)
            add(f"open_cell {arm} (27 atoms, 400 moves)", [float(value.group(1))])
    for path in sorted(glob.glob(os.path.join(raw, f"knn.n*.{variant}.r*.log"))):
        n = re.search(r"knn\.n(\d+)\.", path).group(1)
        values = [float(v) for v in re.findall(r"^sample_ms=([0-9.]+)", open(path).read(), re.M)]
        add(f"knn n={n} k=4 1 thread", values)
    for path in sorted(glob.glob(os.path.join(raw, f"pairs.n*.{variant}.r*.log"))):
        n = re.search(r"pairs\.n(\d+)\.", path).group(1)
        value = re.search(r"ms_per=([0-9.]+)", open(path).read()).group(1)
        add(f"pairs n={n} 1 thread", [float(value)])
    return points, builds


def unit(key):
    return "us/call" if key.startswith("open_band") else "ms"


def bootstrap(before, after, rng, draws=4000):
    ratios = []
    for _ in range(draws):
        b = statistics.median(rng.choices(before, k=len(before)))
        a = statistics.median(rng.choices(after, k=len(after)))
        ratios.append(a / b)
    ratios.sort()
    return ratios[int(0.025 * draws)], ratios[int(0.975 * draws) - 1]


def main():
    raw, before, *afters = sys.argv[1:]
    base, base_builds = collect(raw, before)
    for after in afters:
        rng = random.Random(1)
        other, other_builds = collect(raw, after)
        print(f"## {before} -> {after}\n")
        print(
            "| point | unit | n | before median | before IQR | after median | after IQR "
            "| after/before | 95% CI | builds before -> after | verdict |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for key in base:
            b, a = base[key], other[key]
            bm, bi = quartiles(b)
            am, ai = quartiles(a)
            lo, hi = bootstrap(b, a, rng)
            moved = abs(am - bm) > max(bi, ai)
            if lo > 1.0 and moved:
                verdict = "slower"
            elif hi < 1.0 and moved:
                verdict = "faster"
            else:
                verdict = "within noise"
            built = "-"
            if key in base_builds:
                built = f"{'/'.join(sorted(base_builds[key]))} -> {'/'.join(sorted(other_builds[key]))}"
            print(
                f"| {key} | {unit(key)} | {len(b)}/{len(a)} | {bm:.4f} | {bi:.4f} | {am:.4f} | {ai:.4f} "
                f"| {am / bm:.3f} | {lo:.3f}-{hi:.3f} | {built} | {verdict} |"
            )
        print()


if __name__ == "__main__":
    main()
