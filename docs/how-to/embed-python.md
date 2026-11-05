# Embed linkcell from Python

The Python module wraps the C waist with DLPack (dlpk). It does not
take `numpy.ndarray` as the ABI. Any object with `__dlpack__()` is
valid input. The return value implements `__dlpack__()`; consume it
with `numpy.from_dlpack` or `torch.from_dlpack`.

```python
import numpy as np
import linkcell

xyz = np.ascontiguousarray([[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]], dtype=np.float64)
cell = np.ascontiguousarray([10.0, 10.0, 10.0], dtype=np.float64)
nn, d2 = linkcell.knearest(xyz, cell, k=1)
nn = np.from_dlpack(nn)
d2 = np.from_dlpack(d2)
assert int(nn[0, 0]) == 1
assert abs(float(d2[0, 0]) - 0.64) < 1e-12
```

`xyz` is float64 `(n, 3)` or `(n_frames, n, 3)`. `cell` is float64
`(3,)` (ortho lengths), `(3, 3)` lattice rows (vesin / `lc_cell`
order), or `(4, 3)` rows plus origin, on any DLPack device.
Optional `mask` is length `n`. The GIL is detached for the rayon
CPU walk.

The return is `(indices, dist2)`. Indices are int32 (`-1` unused).
`dist2` is float64 (`NaN` unused). A batched `xyz` yields shape
`(n_frames, n, k)`.

CUDA `xyz` (`kDLCUDA`, including a `torch.Tensor` on GPU) is passed
by device pointer into `lc_gpu_*`. A CUDA `cell` is inverted on
device; the host only reads the four launch ints (`nx`, `ny`,
`nz`, `nC`). The result capsules are on the same CUDA device;
`torch.from_dlpack` takes them without a host bounce.
`linkcell.gpu_available()` is true when the driver and nvrtc load
(gpulite, no CUDA SDK at build time).

Wheels:

- `abi3-py312`: one wheel per platform, GIL CPython 3.12+
- `abi3t-py315`: one wheel per platform, CPython 3.15+ GIL and
  free-threaded (PEP 803). Build with maturin 1.14+ and
  `--features abi3t`.
