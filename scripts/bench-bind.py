"""Cutoff list through every entry point, two builds side by side.

Each round runs the Rust probe (`pairs_scale`), the C and C++ probe
(`pairs_bind_cpp`), and the Python probe (`examples/pairs_bind.py`)
for one build, then the other, so drift on the host hits both. The
table is the median warm milliseconds per call over the rounds.

    python3 scripts/bench-bind.py --base DIR [--new DIR] [--py-base PATH]

DIR is a checkout with `target/release/examples/pairs_scale` and
`target/release/pairs_bind_cpp` built:

    cargo build --release --lib --examples
    g++ -O2 -std=c++17 -Iinclude examples/pairs_bind.cpp \\
        -o target/release/pairs_bind_cpp target/release/liblinkcell.a \\
        -lpthread -ldl -lm

`--py-base` is a directory holding the base `linkcell` package (for
example `pip install --target DIR` of its wheel); the new one is
whatever imports. Env: NS (atom counts), TS (threads), REPS (calls per
run), ROUNDS.
"""

import argparse
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(out):
    rows = {}
    for line in out.splitlines():
        head = line.split()[0] if line.strip() else ""
        if line.startswith("n="):
            kind = "rust"
        elif head in ("cpp", "c", "py"):
            kind = head
        else:
            continue
        fields = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        rows[kind] = (float(fields["ms_per"]), int(fields["pairs"]))
    return rows


def run_build(tree, py_path, n, t, reps):
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    env.update(RAYON_NUM_THREADS=str(t), PAIRS_N=str(n), PAIRS_REPS=str(reps))
    out = subprocess.check_output([f"{tree}/target/release/examples/pairs_scale"], env=env, text=True)
    out += subprocess.check_output([f"{tree}/target/release/pairs_bind_cpp"], env=env, text=True)
    py_env = dict(env)
    if py_path:
        py_env["PYTHONPATH"] = py_path
    out += subprocess.check_output(
        [sys.executable, f"{ROOT}/examples/pairs_bind.py"], env=py_env, text=True
    )
    return parse(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", default=ROOT)
    ap.add_argument("--py-base", default="")
    args = ap.parse_args()
    ns = [int(x) for x in os.environ.get("NS", "256 1024 4096").split()]
    ts = [int(x) for x in os.environ.get("TS", "1 8").split()]
    reps = int(os.environ.get("REPS", "8"))
    rounds = int(os.environ.get("ROUNDS", "5"))
    print("entry atoms threads base_ms new_ms base/new pairs")
    for n in ns:
        for t in ts:
            got = {"base": {}, "new": {}}
            pairs = {}
            for _ in range(rounds):
                for name, tree, py in (("base", args.base, args.py_base), ("new", args.new, "")):
                    for kind, (ms, count) in run_build(tree, py, n, t, reps).items():
                        got[name].setdefault(kind, []).append(ms)
                        pairs.setdefault(kind, set()).add(count)
            for kind in ("rust", "c", "cpp", "py"):
                if kind not in got["base"] or kind not in got["new"]:
                    continue
                b = statistics.median(got["base"][kind])
                w = statistics.median(got["new"][kind])
                same = "/".join(str(p) for p in sorted(pairs[kind]))
                print(f"{kind} {n} {t} {b:.4f} {w:.4f} {b / w:.2f} {same}")


if __name__ == "__main__":
    main()
