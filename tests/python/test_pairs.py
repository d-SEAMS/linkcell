"""Cutoff pairs are the vesin ijS list, not k-nearest with k = n - 1."""

from __future__ import annotations

import numpy as np

import linkcell


def _pairs(xyz, cell, cutoff, **kwargs):
    i, j, shift, d2 = linkcell.pairs_within(xyz, cell, cutoff, **kwargs)
    return (
        np.from_dlpack(i),
        np.from_dlpack(j),
        np.from_dlpack(shift),
        np.from_dlpack(d2),
    )


def test_wrap_pair_keeps_the_caller_shift():
    xyz = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    i, j, shift, d2 = _pairs(xyz, cell, 1.0)
    assert i.shape == (2,)
    assert shift.shape == (2, 3)
    row = int(np.where(i == 0)[0][0])
    assert int(j[row]) == 1
    assert shift[row].tolist() == [-1, 0, 0]
    assert abs(float(d2[row]) - 0.64) < 1e-12


def test_half_list_is_one_side():
    xyz = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    i, j, shift, _d2 = _pairs(xyz, cell, 1.0, half=True)
    assert i.tolist() == [0]
    assert j.tolist() == [1]
    assert shift[0].tolist() == [-1, 0, 0]


def test_sheared_rows_match_a_shift_scan():
    rows = np.ascontiguousarray(
        [[8.0, 0.0, 0.0], [2.0, 7.0, 0.0], [0.4, -0.2, 9.0]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(4)
    frac = rng.random((6, 3))
    xyz = np.ascontiguousarray(frac @ rows, dtype=np.float64)
    cutoff = 3.5
    i, j, shift, d2 = _pairs(xyz, rows, cutoff, cell_hint=2.0)
    got = {
        (int(i[t]), int(j[t]), tuple(int(s) for s in shift[t])): float(d2[t])
        for t in range(i.shape[0])
    }
    want = {}
    cut2 = cutoff * cutoff
    for a in range(6):
        for b in range(6):
            for na in (-1, 0, 1):
                for nb in (-1, 0, 1):
                    for nc in (-1, 0, 1):
                        if a == b and na == nb == nc == 0:
                            continue
                        delta = xyz[b] - xyz[a] + np.array([na, nb, nc]) @ rows
                        dist2 = float(delta @ delta)
                        if dist2 < cut2:
                            want[(a, b, (na, nb, nc))] = dist2
    assert set(got) == set(want)
    for key, dist2 in want.items():
        assert abs(got[key] - dist2) < 1e-8
