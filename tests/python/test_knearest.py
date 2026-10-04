import numpy as np
import pytest

import linkcell


def _pair(xyz, cell, k, **kwargs):
    nn, d2 = linkcell.knearest(xyz, cell, k, **kwargs)
    return np.from_dlpack(nn), np.from_dlpack(d2)


def test_version():
    assert linkcell.__version__ == "0.3.6"


def test_periodic_image():
    xyz = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    nn, d2 = _pair(xyz, cell, 1)
    assert nn.shape == (2, 1)
    assert d2.shape == (2, 1)
    assert int(nn[0, 0]) == 1
    assert int(nn[1, 0]) == 0
    assert abs(float(d2[0, 0]) - 0.64) < 1e-12
    assert abs(float(d2[1, 0]) - 0.64) < 1e-12


def test_lattice_rows():
    xyz = np.ascontiguousarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    cell = np.ascontiguousarray(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float64
    )
    nn, d2 = _pair(xyz, cell, 1)
    assert int(nn[0, 0]) == 1
    assert abs(float(d2[0, 0]) - 1.0) < 1e-12


def test_mask_drops_source():
    xyz = np.ascontiguousarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    mask = np.ascontiguousarray([1, 0, 1], dtype=np.int32)
    nn, _d2 = _pair(xyz, cell, 1, mask=mask)
    assert int(nn[1, 0]) == -1
    assert int(nn[0, 0]) == 2
    assert int(nn[2, 0]) == 0


def test_batched_frames():
    frame = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    xyz = np.stack([frame, frame], axis=0)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    nn, d2 = _pair(xyz, cell, 1)
    assert nn.shape == (2, 2, 1)
    assert d2.shape == (2, 2, 1)
    assert int(nn[0, 0, 0]) == 1
    assert int(nn[1, 1, 0]) == 0
    assert abs(float(d2[0, 0, 0]) - 0.64) < 1e-12


def test_k_zero():
    xyz = np.zeros((2, 3), dtype=np.float64)
    cell = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    with pytest.raises(ValueError, match="k must"):
        linkcell.knearest(xyz, cell, 0)


def test_gpu_available_is_bool():
    assert isinstance(linkcell.gpu_available(), bool)
