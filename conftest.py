"""
tests/conftest.py
=================
Session-level pytest configuration for the test and benchmark suites.

Custom marks
------------
large_scale
    Applied to benchmarks that exercise tree or quartet counts large enough
    to take tens of seconds on a single CPU core.  Excluded from the default
    benchmark run; opt in with ``-m large_scale``.

    Registration here suppresses PytestUnknownMarkWarning and makes the mark
    visible in ``pytest --markers``.

Warning filters
---------------
NumbaPerformanceWarning messages are filtered out during tests. These warnings
about GPU under-utilization are expected with small test data and are not
informative for correctness testing.
"""

import pytest
import warnings


def pytest_configure(config):
    """
    Configure pytest before test collection begins.
    
    This runs very early in the pytest lifecycle, before any test modules
    are imported, which is important for catching warnings from numba
    kernel compilation.
    """
    config.addinivalue_line(
        "markers",
        "large_scale: benchmark at large n_trees/n_quartets scales "
        "(slow — opt in with -m large_scale)",
    )
    
    # Suppress NumbaPerformanceWarning during tests
    # This must happen early, before any kernels are compiled
    try:
        from numba.core.errors import NumbaPerformanceWarning
        warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
    except ImportError:
        # Numba not available, no warnings to suppress
        pass


def pytest_unconfigure(config):
    """
    Clean up after all tests complete.
    
    Restore default warning behavior.
    """
    warnings.resetwarnings()
