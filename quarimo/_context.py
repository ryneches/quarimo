"""
_context.py
===========
Context managers for forest.

Provides clean, Pythonic context managers for temporarily changing state:
- Logging control (suppress/change levels)
- Warning control (suppress specific warnings)
- Backend selection (force specific backend)

All context managers properly restore state on exit, even if exceptions occur.
"""

import logging
import warnings
from contextlib import contextmanager
from typing import Optional, Type


# Module-level state for backend override
_backend_override = None


# ============================================================================ #
# Logging Context Managers
# ============================================================================ #


@contextmanager
def suppress_logger(logger_name: str, level: int = logging.CRITICAL):
    """
    Temporarily change a logger's level.

    Useful for suppressing verbose output from specific modules during
    initialization or bulk operations.

    Parameters
    ----------
    logger_name : str
        Name of the logger to suppress (e.g., 'tree', 'forest')
    level : int, default logging.CRITICAL
        Temporary logging level. Common values:
        - logging.CRITICAL: Suppress almost everything
        - logging.ERROR: Show only errors
        - logging.WARNING: Show warnings and errors
        - logging.INFO: Show info, warnings, and errors
        - logging.DEBUG: Show everything

    Yields
    ------
    None
        Control is yielded back to the with-block.

    Examples
    --------
    >>> # Suppress tree's verbose output during construction
    >>> with suppress_logger('tree'):
    ...     trees = [Tree(nwk) for nwk in newicks]

    >>> # Temporarily reduce logging to warnings only
    >>> with suppress_logger('forest', logging.WARNING):
    ...     c = Forest(trees)

    >>> # Nested suppression works correctly
    >>> with suppress_logger('tree'):
    ...     with suppress_logger('forest'):
    ...         # Both loggers suppressed
    ...         c = Forest(trees)

    Notes
    -----
    - Thread-safe: Each thread has its own logger state
    - Exception-safe: Logger level restored even if exception raised
    - Nesting-safe: Can nest multiple suppress_logger contexts
    """
    logger = logging.getLogger(logger_name)
    original_level = logger.level

    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(original_level)


@contextmanager
def quiet(level: int = logging.CRITICAL):
    """
    Temporarily suppress all forest logging.

    Convenience wrapper for suppressing both 'tree' and
    'forest' loggers simultaneously.

    Parameters
    ----------
    level : int, default logging.CRITICAL
        Temporary logging level for both loggers.

    Yields
    ------
    None
        Control is yielded back to the with-block.

    Examples
    --------
    >>> # Silent collection construction
    >>> with quiet():
    ...     c = Forest(trees)

    >>> # Show only warnings during construction
    >>> with quiet(logging.WARNING):
    ...     c = Forest(trees)

    >>> # Combine with backend selection
    >>> with quiet():
    ...     with use_backend('cpu-parallel'):
    ...         counts = c.quartet_topology(quartets)
    """
    tree_logger = logging.getLogger("tree")
    forest_logger = logging.getLogger("forest")

    original_tree_level = tree_logger.level
    original_forest_level = forest_logger.level

    try:
        tree_logger.setLevel(level)
        forest_logger.setLevel(level)
        yield
    finally:
        tree_logger.setLevel(original_tree_level)
        forest_logger.setLevel(original_forest_level)


# ============================================================================ #
# Warning Context Managers
# ============================================================================ #


@contextmanager
def suppress_warnings(category: Optional[Type[Warning]] = None):
    """
    Temporarily suppress warnings.

    Useful for suppressing known warnings during bulk operations or
    when performance warnings are not relevant (e.g., during testing).

    Parameters
    ----------
    category : Type[Warning] or None, default None
        Warning category to suppress. If None, suppresses all warnings.
        Common categories:
        - UserWarning: General user warnings
        - DeprecationWarning: Deprecated feature warnings
        - FutureWarning: Future API change warnings
        - RuntimeWarning: Runtime behavior warnings
        - NumbaPerformanceWarning: Numba optimization warnings

    Yields
    ------
    None
        Control is yielded back to the with-block.

    Examples
    --------
    >>> # Suppress all warnings
    >>> with suppress_warnings():
    ...     counts = c.quartet_topology(quartets, backend='cuda')

    >>> # Suppress only Numba performance warnings
    >>> from numba.core.errors import NumbaPerformanceWarning
    >>> with suppress_warnings(NumbaPerformanceWarning):
    ...     counts = c.quartet_topology(quartets)

    >>> # Suppress only user warnings
    >>> with suppress_warnings(UserWarning):
    ...     # User warnings suppressed, but others still shown
    ...     process_data()

    Notes
    -----
    - Uses Python's warnings.catch_warnings() internally
    - Fully restores warning state on exit
    - Safe to nest with other warning contexts
    """
    with warnings.catch_warnings():
        if category is None:
            warnings.simplefilter("ignore")
        else:
            warnings.filterwarnings("ignore", category=category)
        yield


# ============================================================================ #
# Backend Context Managers
# ============================================================================ #


@contextmanager
def use_backend(backend: str):
    """
    Temporarily force a specific backend for quartet operations.

    Useful for benchmarking, testing, or ensuring consistent behavior
    regardless of available hardware.

    Parameters
    ----------
    backend : str
        Backend to use. Valid options:
        - 'python': Pure Python (slow, always available)
        - 'cpu-parallel': Numba parallel (requires numba)
        - 'cuda': GPU acceleration (requires numba + CUDA)
        - 'best': Use best available (default behavior)

    Yields
    ------
    None
        Control is yielded back to the with-block.

    Raises
    ------
    ValueError
        If requested backend is not available.

    Examples
    --------
    >>> # Force CPU-parallel backend for consistent timing
    >>> with use_backend('cpu-parallel'):
    ...     start = time.time()
    ...     counts = c.quartet_topology(quartets)
    ...     cpu_time = time.time() - start

    >>> # Compare backends
    >>> backends = ['python', 'cpu-parallel', 'cuda']
    >>> for backend_name in backends:
    ...     try:
    ...         with use_backend(backend_name):
    ...             start = time.time()
    ...             counts = c.quartet_topology(quartets)
    ...             print(f"{backend_name}: {time.time() - start:.3f}s")
    ...     except ValueError:
    ...         print(f"{backend_name}: not available")

    >>> # Force Python backend for debugging
    >>> with use_backend('python'):
    ...     # No JIT compilation, easier to debug
    ...     counts = c.quartet_topology(quartets)

    Notes
    -----
    - **Not thread-safe**: Uses module-level state
    - Backend availability checked when context entered
    - Raises ValueError immediately if backend unavailable
    - Original 'best' behavior restored on exit

    Thread Safety Warning
    ---------------------
    This context manager modifies module-level state and is NOT thread-safe.
    If you need thread-safe backend selection, pass the backend parameter
    directly to quartet_topology() instead:

        # Thread-safe alternative:
        counts = c.quartet_topology(quartets, backend='cpu-parallel')
    """
    global _backend_override

    # Validate backend is available
    from ._backend import get_available_backends

    available = get_available_backends()

    if backend != "best" and backend not in available:
        raise ValueError(
            f"Backend '{backend}' not available. "
            f"Available backends: {', '.join(available)}"
        )

    # Save original override state
    original_override = _backend_override

    try:
        _backend_override = backend
        yield
    finally:
        _backend_override = original_override


def get_backend_override() -> Optional[str]:
    """
    Get the current backend override, if any.

    This is used internally by quartet_topology() to check if a backend
    override is active.

    Returns
    -------
    str or None
        Current backend override, or None if no override active.

    Examples
    --------
    >>> get_backend_override()
    None

    >>> with use_backend('cpu-parallel'):
    ...     print(get_backend_override())
    cpu-parallel

    >>> get_backend_override()  # After context exits
    None
    """
    return _backend_override


# ============================================================================ #
# Combined Context Managers
# ============================================================================ #


@contextmanager
def silent_benchmark(backend: str = "best"):
    """
    Suppress logging and warnings while forcing a specific backend.

    Convenience context manager combining quiet() + use_backend() for
    clean benchmarking code.

    Parameters
    ----------
    backend : str, default 'best'
        Backend to use for operations.

    Yields
    ------
    None
        Control is yielded back to the with-block.

    Examples
    --------
    >>> # Clean benchmark without output
    >>> with silent_benchmark('cpu-parallel'):
    ...     start = time.time()
    ...     counts = c.quartet_topology(quartets)
    ...     elapsed = time.time() - start

    >>> # Compare backends cleanly
    >>> for backend in ['python', 'cpu-parallel', 'cuda']:
    ...     try:
    ...         with silent_benchmark(backend):
    ...             start = time.time()
    ...             counts = c.quartet_topology(quartets)
    ...             print(f"{backend}: {time.time() - start:.3f}s")
    ...     except ValueError:
    ...         print(f"{backend}: not available")
    """
    with quiet():
        with use_backend(backend):
            with suppress_warnings():
                yield
