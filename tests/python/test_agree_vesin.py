"""linkcell k-NN vs vesin pairs sorted by minimage.dist2."""

from __future__ import annotations

import numpy as np
import pytest

import linkcell

vesin = pytest.importorskip("vesin")
minimage = pytest.importorskip("minimage")


def test_knearest_returns_indices_and_dist2():
    xyz = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    raw = linkcell.knearest(xyz, cell, 1)
    assert isinstance(raw, tuple) and len(raw) == 2
    nn = np.from_dlpack(raw[0])
    d2 = np.from_dlpack(raw[1])
    assert int(nn[0, 0]) == 1
    assert abs(float(d2[0, 0]) - 0.64) < 1e-12


def test_knn_matches_brute_minimage():
    rng = np.random.default_rng(1)
    n, k = 20, 3
    box_len = 12.0
    xyz = np.ascontiguousarray(rng.random((n, 3)) * box_len, dtype=np.float64)
    cell = np.ascontiguousarray([box_len, box_len, box_len], dtype=np.float64)
    mi = minimage.Cell.ortho(box_len, box_len, box_len)
    nn, d2 = linkcell.knearest(xyz, cell, k)
    nn = np.from_dlpack(nn)
    d2 = np.from_dlpack(d2)
    for a in range(n):
        ranked = sorted(
            (float(mi.dist2(xyz[a].tolist(), xyz[b].tolist())), b)
            for b in range(n)
            if b != a
        )
        expect = [b for _d, b in ranked[:k]]
        got = [int(nn[a, t]) for t in range(k)]
        assert set(expect) == set(got)
        for t in range(k):
            b = int(nn[a, t])
            assert abs(float(d2[a, t]) - float(mi.dist2(xyz[a].tolist(), xyz[b].tolist()))) < 1e-12


def test_vesin_pairs_are_a_cutoff_subset():
    rng = np.random.default_rng(1)
    n = 20
    box_len = 12.0
    cutoff = 4.0
    xyz = np.ascontiguousarray(rng.random((n, 3)) * box_len, dtype=np.float64)
    rows = np.eye(3) * box_len
    nl = vesin.NeighborList(cutoff=cutoff, full_list=True)
    i, j = nl.compute(xyz, rows, periodic=True, quantities="ij")
    mi = minimage.Cell.ortho(box_len, box_len, box_len)
    for a, b in zip(i.tolist(), j.tolist()):
        if a == b:
            continue
        d = float(mi.dist2(xyz[int(a)].tolist(), xyz[int(b)].tolist())) ** 0.5
        assert d <= cutoff + 1e-9


def test_knn_sheared_matches_minimage_euclidean():
    rows = np.ascontiguousarray(
        [[10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    xyz = np.ascontiguousarray(
        [[0.2, 0.1, 1.0], [9.7, 0.1, 1.0], [5.0, 4.0, 5.0]], dtype=np.float64
    )
    raw = linkcell.knearest(xyz, rows, 1)
    nn = np.from_dlpack(raw[0])
    d2 = np.from_dlpack(raw[1])
    mi = minimage.Cell.from_vesin(rows)
    assert int(nn[0, 0]) == 1
    got = float(d2[0, 0])
    assert abs(got - float(mi.dist2(xyz[0].tolist(), xyz[1].tolist()))) < 1e-12
    assert abs(got - float(mi.dist2_euclidean(xyz[0].tolist(), xyz[1].tolist()))) < 1e-12


def test_knn_hex_body_diagonal_matches_minimage_euclidean():
    rows = np.ascontiguousarray(
        [[10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    mi = minimage.Cell.from_vesin(rows)
    origin = [0.0, 0.0, 0.0]
    body = (
        0.49 * np.array([10.0, 0.0, 0.0])
        + 0.49 * np.array([5.0, 8.660254037844386, 0.0])
        + 0.49 * np.array([0.0, 0.0, 10.0])
    ).tolist()
    xyz = np.ascontiguousarray([origin, body], dtype=np.float64)
    raw = linkcell.knearest(xyz, rows, 1)
    nn = np.from_dlpack(raw[0])
    d2 = np.from_dlpack(raw[1])
    euc = float(mi.dist2_euclidean(origin, body))
    frac = float(mi.dist2(origin, body))
    assert int(nn[0, 0]) == 1
    assert abs(float(d2[0, 0]) - euc) < 1e-12
    assert euc + 1e-8 < frac


def test_knn_unreduced_skew_beats_unreduced_27_image():
    rows = np.ascontiguousarray(
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mi = minimage.Cell.from_vesin(rows)
    origin = [0.0, 0.0, 0.0]
    lattice = [0.02, -0.02, 0.0]
    xyz = np.ascontiguousarray([origin, lattice], dtype=np.float64)
    raw = linkcell.knearest(xyz, rows, 1)
    nn = np.from_dlpack(raw[0])
    d2 = np.from_dlpack(raw[1])
    euc = float(mi.dist2_euclidean(origin, lattice))
    a = np.array(rows[0])
    b = np.array(rows[1])
    c = np.array(rows[2])
    q = np.array(lattice)
    best27 = min(
        float((q + na * a + nb * b + nc * c) @ (q + na * a + nb * b + nc * c))
        for na in (-1, 0, 1)
        for nb in (-1, 0, 1)
        for nc in (-1, 0, 1)
    )
    assert int(nn[0, 0]) == 1
    assert abs(float(d2[0, 0]) - euc) < 1e-12
    assert euc < 1e-24
    assert euc + 1e-8 < best27
