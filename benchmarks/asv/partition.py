"""asv timings of every in-process partition arm, plus the external tools.

The Snakemake panel is the five-system whole-operation table. Each
`time_*` method here calls the same entry, including graph build, on
the 4096-atom cell. PT-Scotch and Zoltan shell out to the panel binaries.
"""

import os
import subprocess

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")


class Partition:
    repeat = 12
    number = 1
    processes = 1
    rounds = 1

    def setup(self):
        import linkcell

        self.linkcell = linkcell
        n = 16
        spacing = 3.1
        xyz = np.array(
            [
                [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing]
                for iz in range(n)
                for iy in range(n)
                for ix in range(n)
            ],
            dtype=np.float64,
        )
        self.xyz = np.ascontiguousarray(xyz)
        self.cell = np.array([n * spacing, n * spacing, n * spacing], dtype=np.float64)
        self.cutoff = 4.3065
        self.tol = "0.03"
        work = "benchmarks/work" if os.path.isdir("benchmarks/work") else "work"
        os.makedirs(work, exist_ok=True)
        self.graph_path = os.path.join(work, "asv-ptscotch.bin")
        self.assign_path = os.path.join(work, "asv-assign.txt")
        self.grid_path = os.path.join(work, "asv-grid.txt")
        _write_graphs(self)

    def time_morton(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="morton")

    def time_hilbert(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="hilbert")

    def time_hilbert_fm(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="hilbert-fm")

    def time_rcb(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="rcb")

    def time_rcb_fm(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="rcb-fm")

    def time_multi(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="multi")

    def time_scotch(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="scotch")

    def time_zoltan(self):
        self._external("zoltan_rcb", ranks=None)

    def time_ptscotch(self):
        self._external("ptscotch_part", ranks=1)

    def _external(self, name, ranks):
        binary = _tool(name)
        graph = self.graph_path if name == "ptscotch_part" else self.grid_path
        assign = self.assign_path
        if name == "ptscotch_part":
            cmd = ["mpirun", "-np", str(ranks), binary, graph, assign, self.tol]
        else:
            cmd = [binary, graph, assign, self.tol]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


def _write_graphs(bench):
    nxyz, _width, weight, offsets, neighbor, edge = bench.linkcell.cell_graph(
        bench.xyz, bench.cell, bench.cutoff
    )
    nxyz = np.from_dlpack(nxyz)
    weight = np.from_dlpack(weight).astype(np.int64)
    offsets = np.from_dlpack(offsets).astype(np.int32)
    neighbor = np.from_dlpack(neighbor).astype(np.int32)
    edge = np.from_dlpack(edge).astype(np.int32)
    n = int(offsets.shape[0] - 1)
    parts = 8
    with open(bench.graph_path, "wb") as handle:
        handle.write(np.int32(n).tobytes())
        handle.write(np.int32(parts).tobytes())
        handle.write(offsets.astype("<i4").tobytes())
        handle.write(neighbor.astype("<i4").tobytes())
        handle.write(weight.astype("<i4").tobytes())
        handle.write(edge.astype("<i4").tobytes())
    nx, ny, nz = (int(v) for v in nxyz)
    with open(bench.grid_path, "w", encoding="utf-8") as handle:
        handle.write(f"{n} {parts}\n")
        for cell in range(n):
            ix = cell % nx
            iy = (cell // nx) % ny
            iz = cell // (nx * ny)
            handle.write(f"{ix + 0.5} {iy + 0.5} {iz + 0.5} {int(weight[cell])}\n")


def _tool(name):
    for path in (os.path.join("benchmarks", "work", name), os.path.join("work", name)):
        if os.path.isfile(path):
            return path
    raise RuntimeError(f"missing {name}")
