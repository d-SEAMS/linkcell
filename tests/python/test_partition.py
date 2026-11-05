"""DLPack partition: one domain per cell, stable under a lattice shift."""

import numpy as np

import linkcell


def test_partition_covers_each_cell_once():
    xyz = np.array(
        [[0.2, 0.2, 0.2], [3.2, 0.2, 0.2], [0.2, 3.2, 0.2], [6.2, 6.2, 6.2]],
        dtype=np.float64,
    )
    cell = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    domain = np.from_dlpack(linkcell.partition(xyz, cell, 3.0, 2))
    assert domain.ndim == 1
    assert set(domain.tolist()) <= {0, 1}
    nxyz = np.from_dlpack(linkcell.cell_grid(xyz, cell, 3.0)[0])
    assert domain.shape[0] == int(nxyz[0] * nxyz[1] * nxyz[2])
    shifted = xyz + np.array([10.0, 0.0, 0.0])
    again = np.from_dlpack(linkcell.partition(shifted, cell, 3.0, 2))
    assert np.array_equal(domain, again)
    width = linkcell.halo_width(3.0, 1, [2.0, 2.0, 2.0])
    assert width == (2, 2, 2)
