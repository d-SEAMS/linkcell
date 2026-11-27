import numpy as np
import pytest

import linkcell

torch = pytest.importorskip("torch")


def _host_pair(xyz, cell, k):
    nn, d2 = linkcell.knearest(xyz, cell, k)
    return np.from_dlpack(nn).copy(), np.from_dlpack(d2).copy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_cuda_matches_host():
    xyz_h = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    host_nn, host_d2 = _host_pair(xyz_h, cell, 1)
    xyz_d = torch.from_numpy(xyz_h).cuda()
    nn, d2 = linkcell.knearest(xyz_d, cell, 1)
    out = torch.from_dlpack(nn)
    out_d2 = torch.from_dlpack(d2)
    assert out.device.type == "cuda"
    assert out_d2.device.type == "cuda"
    assert out.shape == (2, 1)
    assert int(out[0, 0].item()) == int(host_nn[0, 0])
    assert int(out[1, 0].item()) == int(host_nn[1, 0])
    assert abs(float(out_d2[0, 0].item()) - float(host_d2[0, 0])) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_cuda_cell_stays_on_device():
    xyz_h = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell_h = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    host_nn, host_d2 = _host_pair(xyz_h, cell_h, 1)
    xyz_d = torch.from_numpy(xyz_h).cuda()
    cell_d = torch.from_numpy(cell_h).cuda()
    nn, d2 = linkcell.knearest(xyz_d, cell_d, 1)
    out = torch.from_dlpack(nn)
    out_d2 = torch.from_dlpack(d2)
    assert out.device.type == "cuda"
    assert int(out[0, 0].item()) == int(host_nn[0, 0])
    assert abs(float(out_d2[0, 0].item()) - 0.64) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_host_xyz_cuda_cell():
    xyz_h = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell_h = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    cell_d = torch.from_numpy(cell_h).cuda()
    nn, d2 = linkcell.knearest(xyz_h, cell_d, 1)
    out = np.from_dlpack(nn)
    out_d2 = np.from_dlpack(d2)
    assert int(out[0, 0]) == 1
    assert abs(float(out_d2[0, 0]) - 0.64) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_cuda_batched():
    frame = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    xyz_h = np.stack([frame, frame], axis=0)
    cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
    xyz_d = torch.from_numpy(xyz_h).cuda()
    nn, d2 = linkcell.knearest(xyz_d, cell, 1)
    out = torch.from_dlpack(nn)
    out_d2 = torch.from_dlpack(d2)
    assert tuple(out.shape) == (2, 2, 1)
    assert int(out[0, 0, 0].item()) == 1
    assert abs(float(out_d2[1, 1, 0].item()) - 0.64) < 1e-12


def test_gpu_available_is_bool():
    assert isinstance(linkcell.gpu_available(), bool)


def test_cpu_export_to_torch_retains_buffer():
    import gc

    xyz = np.array([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
    cell = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    nn, d2 = linkcell.knearest(xyz, cell, 1)
    indices = torch.from_dlpack(nn)
    distances = torch.from_dlpack(d2)
    del nn, d2
    gc.collect()
    assert indices.tolist() == [[1], [0]]
    torch.testing.assert_close(distances, torch.full((2, 1), 0.64, dtype=torch.float64))
