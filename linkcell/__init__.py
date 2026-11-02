"""Periodic linked-cell k-nearest search.

Arrays are DLPack. Pass any ``__dlpack__()`` object (numpy, torch,
jax, cupy, metatomic) for ``xyz`` and ``cell``. ``knearest`` returns
``(indices, dist2)``. ``pairs_within`` returns ``(i, j, S, dist2)``,
the cutoff list. Use ``numpy.from_dlpack`` or ``torch.from_dlpack``
on each tensor.
"""

from linkcell._lib import NeighborList as _NeighborList
from linkcell._lib import __version__, gpu_available, knearest, pairs_within

__all__ = [
    "NeighborList",
    "__version__",
    "gpu_available",
    "knearest",
    "pairs_within",
]


class NeighborList(_NeighborList):
    """Cutoff neighbour list with the vesin ``compute`` arguments.

    ``quantities`` is a string of ``i``, ``j``, ``P``, ``S``, ``d`` and
    ``D``. The return value is one array per letter. When numpy is
    importable each array is a numpy ndarray; otherwise it is a DLPack
    capsule.
    """

    def compute(self, points, box, periodic=True, quantities="ij"):
        raw = super().compute(points, box, periodic, quantities)
        try:
            import numpy as np
        except ImportError:
            return tuple(raw)
        return tuple(np.from_dlpack(array) for array in raw)
