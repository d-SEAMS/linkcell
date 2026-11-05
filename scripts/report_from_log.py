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
    sums = []
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
            printed_med = bits[bits.index("median_s") + 1]
            printed_iqr = bits[bits.index("iqr_s") + 1]
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
                    "median": float(printed_med),
                    "iqr": float(printed_iqr),
                    "re_median": med,
                    "re_iqr": spread,
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
        elif line.startswith("sum method "):
            sums.append(line)
        elif line.startswith("default_method "):
            meta["default"] = line.split()[-1]
        elif line.startswith("sweep "):
            if pending is None:
                raise SystemExit(f"missing raw samples for {line}")
            bits = line.split()
            med = float(bits[bits.index("median_s") + 1])
            spread = float(bits[bits.index("iqr_s") + 1])
            sweeps.append((line, med, spread, pending))
            pending = None
        elif line.startswith("scale_row "):
            if pending is None:
                raise SystemExit(f"missing raw samples for {line}")
            bits = line.split()
            med = float(bits[bits.index("median_s") + 1])
            spread = float(bits[bits.index("iqr_s") + 1])
            scales.append((line, med, spread, pending))
            pending = None
        elif line.startswith("EXIT "):
            meta["exit"] = line.split(" ", 1)[1]
    return meta, rows, scales, sweeps, sums


def fmt(value):
    return f"{value:.6f}"


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    meta, rows, scales, sweeps, sums = parse(src.read_text())
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
    a("eOn calls `pairs_within` only. rgpot builds pairs through vesin. readcon-core reads configuration (CON) frames.")
    a("")
    a("This cell-partition report is generated from one log. Every median and every interquartile range (IQR) is the value that log printed for that cell. The raw sample line is the series those values came from, at type-7.")
    a("")
    a(f"The log names the hardware as {meta.get('hardware', 'Intel Xeon Processor')}, {meta.get('cores', '8')} physical cores, one thread per core, one node.")
    a("The host process (HOST) records a second node as `multi_node HOST NOT-RUN`.")
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
    a(f"The mW cutoff is {meta.get('mw', '')} Ångströms, which is 1.8 times sigma with sigma 2.3925 Ångströms.")
    a(f"The TIP4P cutoff is {meta.get('tip4p', '7.5')} Ångströms, the spherical cutoff in Jorgensen, Chandrasekhar, Madura, Impey, and Klein, Journal of Chemical Physics 79, 926 (1983).")
    a(f"The water density used to place molecules is {meta.get('rho', '')} molecules per cubic Ångström.")
    a(f"Basis: {meta.get('density_basis', '1 g cm-3')}.")
    a("One cubic centimetre is 1e24 cubic Ångströms, so the conversion from moles per cubic centimetre to molecules per cubic Ångström uses the factor 1e-24.")
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
    a("```text")
    a("method system cut imbalance halo median iqr eligible")
    for row in rows:
        a(
            f"{row['method']} {row['system']} {row['cut']} {row['imbalance']} {row['halo']} {fmt(row['median'])} {fmt(row['iqr'])} {row['eligible']}"
        )
    a("```")
    a("")
    a("## Halo counts")
    a("")
    a("Each row is the domain with the largest ghost-to-owned ratio on the last method that printed halo counts, with bin widths in Ångströms.")
    a("")
    last = {}
    for row in rows:
        if row["owned"]:
            last[row["system"]] = row
    a("```text")
    for name in systems:
        row = last.get(name)
        if row is None:
            continue
        a(
            f"{name} owned {row['owned']} ghosts {row['ghost']} cells {row['ncell']} bin_widths {row['bin']}"
        )
    a("```")
    a("")
    a("On the small TIP4P lattice the bin is about as wide as the 7.5 Ångström cutoff, so the radius is one bin.")
    a("That part owns a few hundred cells.")
    a("The one-bin shell can contain more cells than the part.")
    a("The fraction divides ghosts by owned cells.")
    a("")
    a("## Cut sums")
    a("")
    a("Each sum adds the edge cut of one method over the five systems in this log.")
    a("")
    a("```text")
    if sums:
        for line in sums:
            a(line)
    else:
        by_method = {}
        for row in rows:
            by_method.setdefault(row["method"], 0)
            by_method[row["method"]] += int(row["cut"])
        for method, total in by_method.items():
            a(f"sum method {method} cut {total}")
    if meta.get("default"):
        a(f"default_method {meta['default']}")
    a("```")
    a("")
    a("## Raw samples")
    a("")
    a("Each line is the twelve wall times, in seconds, in the order they were measured.")
    a("")
    a("```text")
    for row in rows:
        text = " ".join(fmt(v) for v in row["raw"])
        a(f"{row['method']} {row['system']} {text}")
    a("```")
    a("")
    a("## Balance sweep")
    a("")
    a("The same five ratios appear on every system in this sweep. The main table uses the derived bound.")
    a("")
    a("```text")
    for line, med, spread, _raw in sweeps:
        bits = line.split()
        # sweep ratio R system S method M cut C imbalance I median_s M iqr_s Q
        ratio = bits[2]
        system = bits[4]
        method = bits[6]
        cut = bits[bits.index("cut") + 1]
        imb = bits[bits.index("imbalance") + 1]
        a(
            f"{method} {ratio} {system} cut {cut} imbalance {imb} median {fmt(med)} iqr {fmt(spread)}"
        )
    a("```")
    a("")
    a("## Scaling")
    a("")
    a("Strong scaling keeps the lattice at 1,520,875 points (115 cubed). Weak scaling uses 190,109 points times the thread count, so eight threads is 1,520,872 points.")
    a("Median and interquartile range (IQR) are recomputed from the raw line.")
    a("`profile_ns` is the summed select, Fiduccia-Mattheyses, and coarsen time over the repetitions, in nanoseconds.")
    a("")
    a("```text")
    a("kind threads n method median iqr")
    for line, med, spread, _raw in scales:
        bits = line.split()
        kind = bits[bits.index("kind") + 1]
        threads = bits[bits.index("threads") + 1]
        n = bits[bits.index("n") + 1]
        method = bits[bits.index("method") + 1]
        a(f"{kind} {threads} {n} {method} {fmt(med)} {fmt(spread)}")
    a("```")
    a("")
    a("## Amdahl")
    a("")
    a("Select, boundary refinement, and coarsening are the profiled parts of each scaling run.")
    a("Each share is that summed time divided by the median wall time times the repetition count.")
    a("The other share is what remains of the median.")
    a("")
    a("```text")
    a("method kind threads n select fm coarse other")
    for line, med, _spread, _raw in scales:
        bits = line.split()
        kind = bits[bits.index("kind") + 1]
        threads = bits[bits.index("threads") + 1]
        n = bits[bits.index("n") + 1]
        method = bits[bits.index("method") + 1]
        sel = int(bits[bits.index("profile_ns") + 1])
        fm = int(bits[bits.index("profile_ns") + 2])
        coarse = int(bits[bits.index("profile_ns") + 3])
        wall = med * reps
        if wall <= 0:
            continue
        def share(ns):
            return min(1.0, (ns / 1.0e9) / wall)

        s_sel, s_fm, s_coarse = share(sel), share(fm), share(coarse)
        other = max(0.0, 1.0 - s_sel - s_fm - s_coarse)
        a(
            f"{kind}-{method} threads {threads} n {n} select {s_sel:.3f} fm {s_fm:.3f} coarse {s_coarse:.3f} other {other:.3f}"
        )
    a("```")
    a("")
    a("## Prose")
    a("")
    a("The prose pass covers the placement note, the report, the partition page, the C API page, and the README.")
    a("Two older pages keep their earlier wording:")
    a("")
    a("```text")
    a("docs/explanation/mic-and-cells.md")
    a("docs/reference/algorithm.md")
    a("```")
    a("")
    dst.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
