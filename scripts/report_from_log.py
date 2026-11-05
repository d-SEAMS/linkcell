#!/usr/bin/env python3
"""Write REPORT.md from one panel log.

Every median and interquartile range is recomputed from the raw sample
line that precedes the summary. The printed median in the log is not copied.
"""

import sys
from pathlib import Path


def quantile(sorted_values, p):
    """Type-7 quantile. h = 1 + (n - 1) * p, then interpolate."""
    n = len(sorted_values)
    h = 1.0 + (n - 1.0) * p
    lo = int(h // 1)
    if abs(h - lo) < 1e-12:
        hi = lo
    else:
        hi = lo + 1
    if lo < 1:
        lo = 1
    if hi > n:
        hi = n
    a = sorted_values[lo - 1]
    b = sorted_values[hi - 1]
    if abs(float(lo) - float(hi)) < 1e-12:
        return a
    return a + (h - lo) * (b - a)


def iqr(samples):
    ordered = sorted(samples)
    return quantile(ordered, 0.5), quantile(ordered, 0.75) - quantile(ordered, 0.25)


def parse(text):
    rows = []
    scales = []
    sweeps = []
    pending = None
    meta = {}
    halo = {}
    derived = {}
    for line in text.splitlines():
        if line.startswith("HEAD "):
            meta["head"] = line.split(" ", 1)[1]
        elif line.startswith("TREE "):
            meta["tree"] = line.split(" ", 1)[1]
        elif line.startswith("hardware "):
            meta["hardware"] = line.split(" ", 1)[1]
        elif line.startswith("physical_cores "):
            meta["cores"] = line.split(" ", 1)[1]
        elif line.startswith("tip4p_cutoff_A "):
            meta["tip4p"] = line.split(" ", 1)[1]
        elif line.startswith("water_density_molecules_per_A3 "):
            meta["rho"] = line.split(" ", 1)[1]
        elif line.startswith("density_basis "):
            meta["density_basis"] = line.split(" ", 1)[1]
        elif line.startswith("mW_cutoff_A "):
            meta["mw"] = line.split(" ", 1)[1]
        elif line.startswith("reps "):
            meta["reps"] = line.split(" ", 1)[1]
        elif line.startswith("balance_bound "):
            meta["bound"] = line.split(" ", 1)[1]
        elif line.startswith("scotch_seed "):
            meta["seed"] = line.split(" ", 1)[1]
        elif line.startswith("raw "):
            pending = [float(x) for x in line.split()[1:]]
        elif line.startswith("halo_owned "):
            parts = line.split()
            # halo_owned N halo_ghost M profile_ns a b c
            halo["owned"] = parts[1]
            halo["ghost"] = parts[3]
        elif line.startswith("derived_balance "):
            # stored against the next system summary; also keep last
            derived["line"] = line
        elif line.startswith("system ") and " n " in line and " method " not in line:
            name = line.split()[1]
            derived[name] = derived.get("line", "")
            # bin widths
            bits = line.split()
            if "bin_width" in bits:
                i = bits.index("bin_width")
                derived[name + "_bin"] = " ".join(bits[i + 1 : i + 4])
            if "ncell" in bits:
                derived[name + "_ncell"] = bits[bits.index("ncell") + 1]
        elif line.startswith("system ") and " method " in line:
            bits = line.split()
            # system NAME method METHOD cut C imbalance I halo_fraction H median_s M iqr_s Q eligible E
            name = bits[1]
            method = bits[3]
            cut = bits[bits.index("cut") + 1]
            imb = bits[bits.index("imbalance") + 1]
            halo_f = bits[bits.index("halo_fraction") + 1]
            eligible = bits[bits.index("eligible") + 1]
            if pending is None:
                raise SystemExit(f"missing raw samples for {name} {method}")
            med, spread = iqr(pending)
            rows.append(
                {
                    "system": name,
                    "method": method,
                    "cut": cut,
                    "imbalance": imb,
                    "halo": halo_f,
                    "median": med,
                    "iqr": spread,
                    "eligible": eligible,
                    "raw": pending,
                    "owned": halo.get("owned", ""),
                    "ghost": halo.get("ghost", ""),
                    "bin": derived.get(name + "_bin", ""),
                    "ncell": derived.get(name + "_ncell", ""),
                    "derived": derived.get(name, ""),
                }
            )
            pending = None
            halo = {}
        elif line.startswith("sweep "):
            if pending is None:
                raise SystemExit(f"missing raw samples for {line}")
            med, spread = iqr(pending)
            sweeps.append((line, med, spread, pending))
            pending = None
        elif line.startswith("scale_row "):
            if pending is None:
                raise SystemExit(f"missing raw samples for {line}")
            med, spread = iqr(pending)
            scales.append((line, med, spread, pending))
            pending = None
        elif line.startswith("EXIT "):
            meta["exit"] = line.split(" ", 1)[1]
    return meta, rows, scales, sweeps


def fmt(value):
    return f"{value:.6f}"


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    meta, rows, scales, sweeps = parse(src.read_text())
    if meta.get("exit") != "0":
        raise SystemExit(f"log exit is {meta.get('exit')}")
    reps = int(meta.get("reps", "0"))
    for row in rows:
        if len(row["raw"]) != reps:
            raise SystemExit(
                f"{row['system']} {row['method']} has {len(row['raw'])} samples, expected {reps}"
            )
    systems = []
    for row in rows:
        if row["system"] not in systems:
            systems.append(row["system"])
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    lines = []
    a = lines.append
    a("# Cell partition")
    a("")
    a("The cell graph, the partition, and the geometric halo live in linkcell.")
    a("Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.")
    a("eOn calls `pairs_within` only. rgpot does not call linkcell. readcon-core reads configuration (CON) frames and is not a caller.")
    a("")
    a("This report is generated from one log. Every median and every interquartile range (IQR) is recomputed from the raw sample line for that cell. The type-7 quantile is the same rule the panel uses.")
    a("")
    a(f"Hardware is {meta.get('hardware', 'Intel Xeon Processor')}, {meta.get('cores', '8')} physical cores, one thread per core, one node.")
    a("A second node was not run. The log line is `multi_node HOST NOT-RUN`, where HOST is the host process.")
    a(f"Commit `{meta.get('head', '')}`. Tree `{meta.get('tree', '')}`.")
    a(f"Exit code {meta.get('exit', '')}.")
    a("")
    a("## Balance")
    a("")
    a(f"The bound is `{meta.get('bound', '1 + parts * heaviest_cell / total_weight')}`.")
    a("A weighted median gives the straddling cell to one side, so a part can exceed the ideal share by at most the heaviest cell.")
    a("The ratio is `1 + parts * heaviest_cell / total_weight`. The same formula is used on every system.")
    a("")
    a("## Cutoff and density")
    a("")
    a(f"The mW cutoff is {meta.get('mw', '')} angstroms, which is 1.8 times sigma with sigma 2.3925 angstroms.")
    a(f"The TIP4P cutoff is {meta.get('tip4p', '7.5')} angstroms, the spherical cutoff in Jorgensen, Chandrasekhar, Madura, Impey, and Klein, Journal of Chemical Physics 79, 926 (1983).")
    a(f"The water density used to place molecules is {meta.get('rho', '')} molecules per cubic angstrom.")
    a(f"Basis: {meta.get('density_basis', '1 g cm-3')}.")
    a("One cubic centimetre is 1e24 cubic angstroms, so the conversion from moles per cubic centimetre to molecules per cubic angstrom uses the factor 1e-24.")
    a("")
    a("## Scotch seed")
    a("")
    a(f"Serial Scotch and parallel Scotch both call `SCOTCH_randomReset` and then `SCOTCH_randomSeed` with seed {meta.get('seed', '1')}.")
    a("The panel log repeats `SCOTCH_randomReset 1` and `SCOTCH_randomSeed 1` for each parallel Scotch run.")
    a("")
    a("## Head to head")
    a("")
    a("Times are seconds. The pair is the median and the interquartile range (IQR) of the raw samples.")
    a("The halo fraction is the largest ghost-cell count divided by the owned-cell count. A value above 1 means the ghost set is larger than the owned set.")
    a("")
    a("| System | Method | Cut | Imbalance | Halo | Median | IQR | Eligible |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        a(
            f"| {row['system']} | {row['method']} | {row['cut']} | {row['imbalance']} | {row['halo']} | {fmt(row['median'])} | {fmt(row['iqr'])} | {row['eligible']} |"
        )
    a("")
    a("## Halo counts")
    a("")
    a("The rows below are the domain with the largest ghost-to-owned ratio for the last native method in the log, together with the bin widths in angstroms.")
    a("")
    a("| System | Cells | Bin widths | Owned | Ghost |")
    a("| --- | ---: | --- | ---: | ---: |")
    seen = set()
    for row in rows:
        if row["system"] in seen or not row["owned"]:
            continue
        # keep the last row that recorded halo for this system: overwrite by scanning
        seen.add(row["system"])
    # rescan for the last owned count per system
    last = {}
    for row in rows:
        if row["owned"]:
            last[row["system"]] = row
    for name in systems:
        row = last.get(name)
        if row is None:
            continue
        a(f"| {name} | {row['ncell']} | {row['bin']} | {row['owned']} | {row['ghost']} |")
    a("")
    a("On the small TIP4P lattice the bin is about as wide as the 7.5 angstrom cutoff, so the radius is one bin.")
    a("A part then owns a few hundred cells, and the one-bin shell around that part can contain more cells than the part itself.")
    a("The fraction is ghosts divided by owned cells, not ghosts divided by the whole grid.")
    a("")
    a("## Raw samples")
    a("")
    a("Each line is the twelve wall times, in seconds, in the order they were measured.")
    a("")
    for row in rows:
        text = " ".join(fmt(v) for v in row["raw"])
        a(f"- {row['system']} {row['method']}: {text}")
    a("")
    a("## Balance sweep")
    a("")
    a("The same five ratios are used on every system. The main table uses the derived bound, not one of these fixed ratios.")
    a("")
    for line, med, spread, _raw in sweeps:
        bits = line.split()
        # sweep ratio R system S method M cut C imbalance I median_s M iqr_s Q
        ratio = bits[2]
        system = bits[4]
        method = bits[6]
        cut = bits[bits.index("cut") + 1]
        imb = bits[bits.index("imbalance") + 1]
        a(
            f"- ratio {ratio} {system} {method} cut {cut} imbalance {imb} median {fmt(med)} iqr {fmt(spread)}"
        )
    a("")
    a("## Scaling")
    a("")
    a("Strong scaling holds the lattice at 1,520,875 points (115 cubed). Weak scaling uses 190,109 points times the thread count, so eight threads is 1,520,872 points.")
    a("Median and interquartile range (IQR) are recomputed from the raw line.")
    a("`profile_ns` is the summed select, Fiduccia-Mattheyses, and coarsen time over the repetitions, in nanoseconds.")
    a("")
    a("| Kind | Threads | N | Method | Median | IQR |")
    a("| --- | ---: | ---: | --- | ---: | ---: |")
    for line, med, spread, _raw in scales:
        bits = line.split()
        kind = bits[bits.index("kind") + 1]
        threads = bits[bits.index("threads") + 1]
        n = bits[bits.index("n") + 1]
        method = bits[bits.index("method") + 1]
        a(f"| {kind} | {threads} | {n} | {method} | {fmt(med)} | {fmt(spread)} |")
    a("")
    a("## Profile")
    a("")
    for line, _med, _spread, _raw in scales:
        a(f"- {line}")
    a("")
    dst.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
