"""asv timings of the cell partition on one water lattice.

The Snakemake panel is the five-system table. This benchmark times the
same `partition` entry on the 4096-atom cell, serially, twelve repeats.
"""

import os

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

    def time_hilbert_fm(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="hilbert-fm")

    def time_rcb(self):
        self.linkcell.partition(self.xyz, self.cell, self.cutoff, 8, method="rcb")
