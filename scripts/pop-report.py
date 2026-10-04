#!/usr/bin/env python3
"""Turn `pop ...` lines into the POP strong-scaling hierarchy.

Reference is the 1-thread row of the same shape. Computation scaling is
sum(useful at 1 thread) / sum(useful here), per repetition. Global
efficiency is parallel efficiency times computation scaling.
"""

import sys


def parse(line):
    item = {}
    for tok in line.split()[1:]:
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        item[key] = val
    return item


def useful_sum(item):
    parts = [int(x) for x in item["useful_ns"].split(",") if x]
    return sum(parts)


rows = []
for line in sys.stdin:
    line = line.strip()
    if line.startswith("pop "):
        rows.append(parse(line))

ref = {}
for row in rows:
    shape = row.get("shape", row.get("backend", ""))
    threads = int(row["threads"])
    if threads == 1:
        ref[shape] = row

print(
    "shape threads wall_ms lb ce pe comps ge"
)
for row in rows:
    shape = row.get("shape", row.get("backend", ""))
    threads = int(row["threads"])
    reps = int(row["reps"])
    wall_ms = int(row["wall_ns"]) / 1e6 / reps
    lb = float(row["lb"])
    ce = float(row["ce"])
    pe = float(row["pe"])
    base = ref.get(shape)
    if base is None:
        comps = 1.0
    else:
        here = useful_sum(row) / reps
        origin = useful_sum(base) / int(base["reps"])
        comps = origin / here if here else 0.0
    ge = pe * comps
    print(
        f"{shape} {threads} {wall_ms:.2f} {lb:.3f} {ce:.3f} {pe:.3f} {comps:.3f} {ge:.3f}"
    )
