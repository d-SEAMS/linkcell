"""Cutoff pairs are the vesin ijS list, not k-nearest with k = n - 1."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_open_cluster_is_the_cartesian_half_list():
    # This frame is not used by any other test, so a stencil left in the
    # process cache cannot already match it.
    xyz = np.ascontiguousarray(
        [[0.05, 0.17, 0.03], [1.11, 0.08, 0.04], [1.19, 0.46, 0.02], [7.4, 0.2, 0.15]],
        dtype=np.float64,
    )
    cutoff = 1.61
    before = linkcell.stencil_builds()
    i, j, shift, d2 = _open(xyz, cutoff)
    built = linkcell.stencil_builds() - before
    assert built <= 1
    i2, j2, shift2, d2_2 = _open(xyz, cutoff)
    assert linkcell.stencil_builds() - before == built
    assert i.tolist() == i2.tolist()
    assert j.tolist() == j2.tolist()
    assert shift.tolist() == shift2.tolist() == [[0, 0, 0]] * len(i)
    assert d2.tolist() == d2_2.tolist()
    cut2 = cutoff * cutoff
    want = set()
    for a in range(len(xyz)):
        for b in range(a + 1, len(xyz)):
            delta = xyz[b] - xyz[a]
            if float(delta @ delta) < cut2:
                want.add((a, b))
    got = {(int(i[t]), int(j[t])) for t in range(len(i))}
    assert got == want
    assert all(float(d2[t]) < cut2 for t in range(len(d2)))


def _open(xyz, cutoff):
    i, j, shift, d2 = linkcell.pairs_open(xyz, cutoff)
    return (
        np.from_dlpack(i),
        np.from_dlpack(j),
        np.from_dlpack(shift),
        np.from_dlpack(d2),
    )


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


def _near_flat_rows():
    lx = ly = 13.0
    lz = 23.3
    alpha = np.deg2rad(86.892)
    beta = np.deg2rad(16.892)
    gamma = np.deg2rad(70.0)
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)
    factor = (cos_a - cos_b * cos_g) / sin_g
    inside = 1.0 - cos_b * cos_b - factor * factor
    cz = lz * np.sqrt(inside)
    return np.ascontiguousarray(
        [
            [lx, 0.0, 0.0],
            [ly * cos_g, ly * sin_g, 0.0],
            [lz * cos_b, lz * factor, cz],
        ],
        dtype=np.float64,
    )


_NEAR_FLAT_CU = np.ascontiguousarray(
    [
        [2.7431309316168475, 6.108002035108404, 0.0],
        [13.232056168881522, 6.352322116512741, 0.2],
        [4.03338785589701, 0.3664801221065042, 0.0],
        [7.720397346483967, 10.13928337827995, -0.2],
        [8.19626621664133, 0.45462981824422477, 0.0412786853694912],
        [13.762071462898417, 10.626955433611426, -0.03622779513074964],
        [11.205363013255461, 7.1146751150038385, 0.027634492938765798],
        [4.492545207753816, 4.035049921555236, -0.19606254358135186],
        [16.50539159880175, 12.057883571293173, 0.15819319967883666],
        [4.199134240190456, 10.099861087740562, 0.2068132352186905],
        [6.126746340992574, 8.464007919000748, -0.11518997428508809],
        [9.814929593400677, 8.903620103137243, -0.1682723449043315],
        [2.2857588312509485, 2.9564446590077442, 0.12103716729274078],
        [5.9510448718450855, 0.20132540631560192, -0.08736864655602095],
        [11.368678075155005, 10.008767396697653, -0.05551644303227138],
        [15.870150122394206, 8.195507208935341, 0.1519522101214148],
        [10.42169020181862, 1.7934744461922287, 0.10864204430766089],
        [8.351747146366385, 3.7093332382537727, 0.02911048113793241],
        [4.732924885010005, 6.6345398008188905, 0.1202865161255377],
        [11.924591995381464, 4.883493475804255, -0.06332713818033944],
        [12.308118627317418, 2.485340130933891, -0.24174491728692993],
        [0.9990718536434593, 0.9081763430908019, 0.09809706259195394],
        [8.401410062080734, 6.873783879816667, 0.223667971540852],
        [5.947793909899321, 2.4918353256604986, -0.27449681317456937],
    ],
    dtype=np.float64,
)


def _pbc_pairs(xyz, cell, periodic, cutoff):
    i, j, shift, d2 = linkcell.pairs_within_pbc(
        xyz,
        cell,
        periodic,
        cutoff,
        half=True,
        exclude_self_images=True,
    )
    return (
        np.from_dlpack(i),
        np.from_dlpack(j),
        np.from_dlpack(shift),
        np.from_dlpack(d2),
    )


def test_near_flat_slab_matches_minimum_image():
    rows = _near_flat_rows()
    assert 1.0e-7 < float(rows[2, 2]) < 1.5e-7
    cutoff = 2.6
    cut2 = cutoff * cutoff
    xyz = _NEAR_FLAT_CU
    for periodic in (
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, False, False),
    ):
        i, j, shift, d2 = _pbc_pairs(xyz, rows, list(periodic), cutoff)
        got = {
            (int(i[t]), int(j[t]), tuple(int(s) for s in shift[t])): float(d2[t])
            for t in range(i.shape[0])
        }
        want = {}
        span = range(-6, 7)
        for a in range(len(xyz)):
            for b in range(a + 1, len(xyz)):
                for na in span:
                    for nb in span:
                        for nc in span:
                            image = (na, nb, nc)
                            if not periodic[0] and na:
                                continue
                            if not periodic[1] and nb:
                                continue
                            if not periodic[2] and nc:
                                continue
                            delta = xyz[b] - xyz[a] + np.array(image, dtype=np.float64) @ rows
                            dist2 = float(delta @ delta)
                            if dist2 < cut2:
                                want[(a, b, image)] = dist2
        assert set(got) == set(want)
        for key, dist2 in want.items():
            assert abs(got[key] - dist2) < 1.0e-8


def test_near_flat_slab_fully_periodic_is_degenerate():
    with pytest.raises(RuntimeError, match="face width has collapsed"):
        linkcell.pairs_within_pbc(_NEAR_FLAT_CU, _near_flat_rows(), [True, True, True], 2.6)
