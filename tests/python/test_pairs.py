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


def test_near_flat_slab_fully_periodic_is_too_many_images():
    with pytest.raises(RuntimeError, match="periodic image count exceeds 2\\^24"):
        linkcell.pairs_within_pbc(_NEAR_FLAT_CU, _near_flat_rows(), [True, True, True], 2.6)


def _thin_brute(xyz, edges, cutoff):
    """Minimum image on an orthorhombic box, half list, strict cutoff."""
    cut2 = cutoff * cutoff
    reps = [int(np.ceil(cutoff / float(edges[a]))) for a in range(3)]
    spans = [np.arange(-rep, rep + 1, dtype=np.int32) for rep in reps]
    sa, sb, sc = np.meshgrid(*spans, indexing="ij")
    shift = np.stack((sa.ravel(), sb.ravel(), sc.ravel()), axis=1).astype(np.float64)
    delta = shift * edges
    want = {}
    for a in range(len(xyz)):
        for b in range(a, len(xyz)):
            diff = xyz[b] - xyz[a] + delta
            dist2 = np.einsum("ij,ij->i", diff, diff)
            keep = dist2 < cut2
            if a == b:
                zero = np.all(shift == 0.0, axis=1)
                keep &= ~zero
            images = shift[keep]
            values = dist2[keep]
            for row in range(images.shape[0]):
                image = images[row]
                if a == b:
                    sign = 0
                    for component in image:
                        if component != 0.0:
                            sign = component
                            break
                    if sign >= 0.0:
                        continue
                key = (a, b, (int(image[0]), int(image[1]), int(image[2])))
                want[key] = float(values[row])
    return want


def _assert_pairs(xyz, edges, cutoff):
    i, j, shift, d2 = _pairs(xyz, edges, cutoff, half=True)
    pbc = linkcell.pairs_within_pbc(xyz, edges, [True, True, True], cutoff, half=True)
    pi, pj, pshift, pd2 = (np.from_dlpack(part) for part in pbc)
    assert pi.shape == i.shape
    got = {}
    for row in range(i.shape[0]):
        key = (
            int(i[row]),
            int(j[row]),
            (int(shift[row, 0]), int(shift[row, 1]), int(shift[row, 2])),
        )
        got[key] = float(d2[row])
        pkey = (
            int(pi[row]),
            int(pj[row]),
            (int(pshift[row, 0]), int(pshift[row, 1]), int(pshift[row, 2])),
        )
        assert pkey == key
        assert abs(float(pd2[row]) - float(d2[row])) <= 1e-9 * max(1.0, abs(float(d2[row])))
    want = _thin_brute(xyz, edges, cutoff)
    assert set(got) == set(want)
    for key, dist2 in want.items():
        assert abs(got[key] - dist2) <= 1e-9 * max(1.0, abs(dist2))


def test_thin_periodic_ratios_match_or_error():
    ratios = (5e-7, 9.9e-7, 1e-6, 1.1e-6, 2e-6, 1e-5, 1e-4)
    cutoffs = (0.1, 2.6, 10.0)
    long = 10.0
    for ratio in ratios:
        thin = long * ratio
        edges = np.array([long, long, thin], dtype=np.float64)
        xyz = np.array(
            [[1.0, 1.0, thin * 0.30], [1.4, 1.2, thin * 0.70]],
            dtype=np.float64,
        )
        for cutoff in cutoffs:
            with pytest.raises(RuntimeError) as err:
                linkcell.pairs_within(xyz, edges, cutoff, half=True)
            text = str(err.value)
            assert "face width is zero or non-finite" not in text
            assert (
                "periodic image count exceeds 2^24" in text
                or "linked-cell mesh is too fine" in text
            )
            with pytest.raises(RuntimeError) as err:
                linkcell.pairs_within_pbc(xyz, edges, [True, True, True], cutoff, half=True)
            assert "face width is zero or non-finite" not in str(err.value)
    # Face ratio 1e-4 with the short face equal to the cutoff. The mesh
    # then has 10_000 bins on the long axis and a reach of 1, which the
    # bin cap accepts, and the list matches the minimum image.
    for cutoff in cutoffs:
        thin = cutoff
        long = cutoff / 1.0e-4
        edges = np.array([long, thin, thin], dtype=np.float64)
        xyz = np.array(
            [
                [long * 0.20, thin * 0.25, thin * 0.30],
                [long * 0.20 + 0.40 * cutoff, thin * 0.55, thin * 0.40],
            ],
            dtype=np.float64,
        )
        _assert_pairs(xyz, edges, cutoff)
    long = 100.0
    thin = long * 1.1e-6
    edges = np.array([long, thin, thin], dtype=np.float64)
    xyz = np.tile(np.array([[long * 0.2, thin * 0.5, thin * 0.5]], dtype=np.float64), (512, 1))
    with pytest.raises(RuntimeError, match="periodic image count exceeds 2\\^24"):
        linkcell.pairs_within(xyz, edges, 2.6, half=True)
    with pytest.raises(RuntimeError, match="periodic image count exceeds 2\\^24"):
        linkcell.pairs_within_pbc(xyz, edges, [True, True, True], 2.6)


def test_flat_and_needle_are_typed_errors():
    # Widths 10, 10, 1.1e-5, cutoff 10: under the image cap, over the bin cap.
    # Eight points make the ideal-gas full list about 12 GB. The mesh error
    # comes back before that buffer is reserved. half defaults to false.
    thin = 1.1e-5
    edges = np.array([10.0, 10.0, thin], dtype=np.float64)
    xyz = np.tile(np.array([[1.0, 2.0, thin * 0.3]], dtype=np.float64), (8, 1))
    for half in (False, True):
        with pytest.raises(RuntimeError, match="linked-cell mesh is too fine"):
            linkcell.pairs_within(xyz, edges, 10.0, half=half)
        with pytest.raises(RuntimeError, match="linked-cell mesh is too fine"):
            linkcell.pairs_within_pbc(xyz, edges, [True, True, True], 10.0, half=half)
    # Ratio 1e-6.
    edges = np.array([10.0, 10.0, 1.0e-5], dtype=np.float64)
    xyz = np.tile(np.array([[1.0, 2.0, 1.0e-5 * 0.3]], dtype=np.float64), (4, 1))
    for cutoff in (0.1, 2.6, 10.0):
        with pytest.raises(RuntimeError) as err:
            linkcell.pairs_within(xyz, edges, cutoff, half=False)
        text = str(err.value)
        assert "face width is zero or non-finite" not in text
        assert (
            "periodic image count exceeds 2^24" in text
            or "linked-cell mesh is too fine" in text
        )
    # Needle: image cap and bin cap both hold; the partner stencil does not.
    edges = np.array([0.001, 0.001, 10.0], dtype=np.float64)
    xyz = np.array([[0.0002, 0.0003, 1.0], [0.0005, 0.0007, 1.04]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="periodic image count exceeds 2\\^24"):
        linkcell.pairs_within(xyz, edges, 0.1, half=False)
    with pytest.raises(RuntimeError, match="periodic image count exceeds 2\\^24"):
        linkcell.pairs_within_pbc(xyz, edges, [True, True, True], 0.1, half=True)


def test_wide_partner_stencils_match_brute_force():
    # A cube of side 80 at cutoff 1 builds a reach-two stencil of about
    # 6e7 slots, past 2^24 and inside the 2^31 partner cap.
    xyz = np.array([[0.2, 0.2, 0.2], [0.7, 0.2, 0.2]], dtype=np.float64)
    _assert_pairs(xyz, np.array([80.0, 80.0, 80.0]), 1.0)
    # Widths 0.01, 0.01, 10 at cutoff 0.2: the image cap and the bin cap
    # both hold, and the reach box is about 7e7 slots.
    xyz = np.array([[0.002, 0.003, 1.0], [0.005, 0.007, 1.05]], dtype=np.float64)
    _assert_pairs(xyz, np.array([0.01, 0.01, 10.0]), 0.2)
