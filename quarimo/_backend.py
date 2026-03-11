"""
_backend.py
===========
Single source of truth for quarimo backend availability.

All backend detection logic lives here.  Every other module — kernel files,
``_forest.py``, ``conftest.py``, test files — imports from this module rather
than performing its own detection.

Design
------
``_BackendCapabilities`` is a lazy, cached capability object.  Each backend is
probed at most once per interpreter session via ``@cached_property``; the probe
runs only when first accessed, not at import time.

The module-level ``backends`` singleton is the intended entry point::

    from quarimo._backend import backends

    if backends.cuda:
        # GPU path
    elif backends.numba:
        # CPU-parallel path
    else:
        # Python fallback

Backend priority (ascending — ``backends.best()`` returns the last available):
    ``python`` < ``cpu-parallel`` < ``mlx`` < ``cuda``

This ordering reflects real hardware: CUDA on a dedicated NVIDIA GPU is the
fastest option; MLX on Apple Silicon (integrated GPU, Unified Memory) is the
Apple equivalent; cpu-parallel (Numba JIT + OpenMP) is next; pure Python is
always the baseline fallback.  On any given machine, ``cuda`` and ``mlx`` are
mutually exclusive in practice (Apple Silicon cannot run CUDA).

Backward-compatible public API
-------------------------------
The module also exports thin-wrapper functions (``check_numba_available``,
``check_cuda_available``, ``get_available_backends``, etc.) so that existing
call sites keep working without changes.
"""

from __future__ import annotations

from functools import cached_property
from typing import List, Optional, Tuple


# ============================================================================ #
# Capability registry                                                           #
# ============================================================================ #


class _BackendCapabilities:
    """
    Lazy, cached detection of all available quarimo backends.

    Use the module-level ``backends`` singleton rather than instantiating this
    class directly.  Properties are evaluated at most once per session.

    Properties
    ----------
    numba : bool
        True if the ``numba`` package is importable.  Required for both the
        ``cpu-parallel`` and ``cuda`` backends.
    cuda : bool
        True if Numba is available **and** ``numba.cuda.is_available()`` is
        True (i.e. a CUDA-capable GPU and driver are present).
    mlx : bool
        True if ``mlx`` is importable **and** a Metal device initialises
        successfully.  Apple Silicon only; mutually exclusive with ``cuda``
        in practice.
    """

    @cached_property
    def numba(self) -> bool:
        """True if Numba is installed (enables ``cpu-parallel`` backend)."""
        try:
            import numba  # noqa: F401
            return True
        except ImportError:
            return False

    @cached_property
    def cuda(self) -> bool:
        """True if Numba + a CUDA-capable GPU are present (enables ``cuda`` backend)."""
        if not self.numba:
            return False
        try:
            from numba import cuda
            return bool(cuda.is_available())
        except Exception:
            return False

    @cached_property
    def mlx(self) -> bool:
        """True if MLX + Metal GPU are available (enables ``mlx`` backend, Apple Silicon)."""
        try:
            import mlx.core as mx
            mx.eval(mx.array([0], dtype=mx.int32))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def available(self) -> List[str]:
        """
        Return available backend names in ascending preference order.

        The last element is always the best (highest-performance) backend.

        Examples
        --------
        >>> backends.available()
        ['python', 'cpu-parallel', 'cuda']   # NVIDIA Linux
        >>> backends.available()
        ['python', 'cpu-parallel', 'mlx']    # Apple Silicon
        >>> backends.available()
        ['python']                           # minimal install
        """
        result = ["python"]
        if self.numba:
            result.append("cpu-parallel")
        if self.mlx:
            result.append("mlx")
        if self.cuda:
            result.append("cuda")
        return result

    def best(self) -> str:
        """Return the highest-priority available backend name."""
        return self.available()[-1]

    def resolve(self, backend: str) -> str:
        """
        Validate and return a concrete backend name.

        Parameters
        ----------
        backend : str
            ``'best'`` or an explicit backend name (``'python'``,
            ``'cpu-parallel'``, ``'cuda'``, ``'mlx'``).

        Returns
        -------
        str
            The resolved backend name.

        Raises
        ------
        ValueError
            If *backend* is not in ``available()``.
        """
        if backend == "best":
            return self.best()
        available = self.available()
        if backend not in available:
            raise ValueError(
                f"Backend '{backend}' not available. "
                f"Available backends: {', '.join(available)}"
            )
        return backend


# Module-level singleton — the single source of truth for backend availability
backends = _BackendCapabilities()


# ============================================================================ #
# Backward-compatible public API                                                #
# (thin wrappers; prefer importing `backends` directly in new code)            #
# ============================================================================ #


def check_numba_available() -> bool:
    """Return True if Numba is installed."""
    return backends.numba


def check_cuda_available() -> Tuple[bool, bool]:
    """Return ``(numba_available, cuda_available)`` for backward compatibility."""
    return (backends.numba, backends.cuda)


def check_mlx_available() -> bool:
    """Return True if MLX with Metal GPU is available (Apple Silicon)."""
    return backends.mlx


def get_available_backends() -> List[str]:
    """Return available backend names in ascending preference order."""
    return backends.available()


def get_best_backend() -> str:
    """Return the highest-priority available backend name."""
    return backends.best()


def resolve_backend(backend: str) -> str:
    """Validate and resolve a backend name; raises ValueError if unavailable."""
    return backends.resolve(backend)


def get_backend_info() -> dict:
    """Return a comprehensive dict of backend availability status."""
    return {
        "numba_available": backends.numba,
        "cuda_available": backends.cuda,
        "mlx_available": backends.mlx,
        "backends": backends.available(),
        "best_backend": backends.best(),
    }
