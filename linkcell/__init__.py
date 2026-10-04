"""Periodic linked-cell k-nearest search.

Arrays are DLPack. Pass any ``__dlpack__()`` object (numpy, torch,
jax, cupy, metatomic) for ``xyz`` and ``cell``. ``knearest`` returns
``(indices, dist2)``. ``pairs_within`` returns ``(i, j, S, dist2)``,
the cutoff list. Use ``numpy.from_dlpack`` or ``torch.from_dlpack``
on each tensor.
"""

from linkcell._lib import __version__, gpu_available, knearest, pairs_within

__all__ = ["__version__", "gpu_available", "knearest", "pairs_within"]
