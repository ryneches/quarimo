"""
conftest.py
===========
Session-level pytest configuration for the quarimo test and benchmark suites.

Custom markers
--------------
Hardware-conditional markers are registered here.  The
``pytest_collection_modifyitems`` hook auto-skips tests whose marker's
required backend is absent, so test files can use simple declarative marks
instead of embedding ``pytest.mark.skipif`` logic.

``requires_cuda``
    Skip unless Numba + a CUDA-capable GPU are present (``backends.cuda``).

``requires_mlx``
    Skip unless MLX + Metal GPU are present on Apple Silicon (``backends.mlx``).

``requires_cpu_parallel``
    Skip unless Numba is installed (``backends.numba``).

``large_scale``
    Slow benchmarks — excluded from CI with ``-m 'not large_scale'``.

``polytomy``
    Polytomy-specific topology tests.

Extending
---------
To add a new backend:
1. Register its marker in ``pytest_configure``.
2. Add a guard clause in ``pytest_collection_modifyitems``.
3. Decorate the relevant tests.

Warning filters
---------------
NumbaPerformanceWarning is suppressed globally.  These warnings about GPU
under-utilisation are expected with small test datasets and are not
informative for correctness testing.
"""

import warnings

import pytest

from quarimo._backend import backends


def pytest_configure(config):
    """Register custom markers and suppress known-noisy warnings."""
    config.addinivalue_line(
        "markers",
        "requires_cuda: skip if CUDA (Numba + NVIDIA GPU) is not available",
    )
    config.addinivalue_line(
        "markers",
        "requires_mlx: skip if MLX with Metal GPU is not available (Apple Silicon only)",
    )
    config.addinivalue_line(
        "markers",
        "requires_cpu_parallel: skip if Numba is not installed",
    )
    config.addinivalue_line(
        "markers",
        "large_scale: slow benchmark tests — opt in with -m large_scale",
    )
    config.addinivalue_line(
        "markers",
        "polytomy: topology tests for polytomous/tied quartet-tree pairs",
    )

    try:
        from numba.core.errors import NumbaPerformanceWarning
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Auto-skip hardware-conditional tests when the backend is unavailable."""
    skip_cuda = pytest.mark.skip(reason="CUDA (Numba + NVIDIA GPU) not available")
    skip_mlx = pytest.mark.skip(reason="MLX with Metal GPU not available (Apple Silicon)")
    skip_parallel = pytest.mark.skip(reason="Numba not installed")

    for item in items:
        if "requires_cuda" in item.keywords and not backends.cuda:
            item.add_marker(skip_cuda)
        if "requires_mlx" in item.keywords and not backends.mlx:
            item.add_marker(skip_mlx)
        if "requires_cpu_parallel" in item.keywords and not backends.numba:
            item.add_marker(skip_parallel)


def pytest_unconfigure(config):
    warnings.resetwarnings()
