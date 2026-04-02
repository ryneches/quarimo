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

_logger = logging.getLogger('quarimo')

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
        Name of the logger to suppress (e.g., ``'quarimo'``, ``'quarimo.forest'``)
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
    >>> # Suppress all quarimo logging during construction
    >>> with suppress_logger('quarimo'):
    ...     forest = Forest(newicks)

    >>> # Temporarily reduce forest logging to warnings only
    >>> with suppress_logger('quarimo.forest', logging.WARNING):
    ...     forest = Forest(trees)

    >>> # Nested suppression works correctly
    >>> with suppress_logger('quarimo.forest'):
    ...     with suppress_logger('quarimo.backend'):
    ...         counts = forest.quartet_topology(quartets)

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
    Temporarily suppress all quarimo logging.

    Suppresses the ``quarimo`` parent logger, which covers all child loggers
    (``quarimo.forest``, ``quarimo.backend``, etc.).

    Parameters
    ----------
    level : int, default logging.CRITICAL
        Temporary logging level.

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
    quarimo_logger = logging.getLogger("quarimo")
    original_level = quarimo_logger.level

    try:
        quarimo_logger.setLevel(level)
        yield
    finally:
        quarimo_logger.setLevel(original_level)


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
    Temporarily force a specific backend for all quartet operations.

    Validates that the requested backend is available on entry and raises
    ``ValueError`` immediately if it is not — so failures are explicit and
    located at the context boundary rather than buried in kernel dispatch.

    Parameters
    ----------
    backend : str
        Backend to use.  Valid values:

        ``'python'``
            Pure-Python fallback.  Always available.
        ``'cpu-parallel'``
            Numba-JIT parallel kernel.  Requires ``numba``.
        ``'cuda'``
            NVIDIA GPU kernel.  Requires ``numba`` and a CUDA-capable device.
        ``'mlx'``
            Apple Metal GPU kernel.  Requires ``mlx`` and an M-series chip.
        ``'best'``
            Resolve to the highest-priority available backend (default
            behavior when no context is active).

    Yields
    ------
    str
        The resolved concrete backend name (never ``'best'``).

    Raises
    ------
    ValueError
        Raised immediately on context entry if *backend* is unavailable.
        The message names both the requested backend and the full list of
        available ones, e.g.::

            ValueError: Backend 'cuda' not available.
            Available backends: python, cpu-parallel

    Examples
    --------
    Force a specific backend for a single call:

    >>> with use_backend('cpu-parallel'):
    ...     counts = forest.quartet_topology(quartets)

    Compare backends (graceful skip when unavailable):

    >>> for name in ['python', 'cpu-parallel', 'cuda']:
    ...     try:
    ...         with use_backend(name) as b:
    ...             counts = forest.quartet_topology(quartets)
    ...             print(f"{b}: ok")
    ...     except ValueError as e:
    ...         print(f"skipped — {e}")

    Notes
    -----
    - **Not thread-safe**: Uses module-level state.  For thread-safe backend
      selection pass ``backend=`` directly to :meth:`Forest.quartet_topology`.
    - The context manager stores the *resolved* concrete name, so
      ``use_backend('best')`` inside the block reports the actual backend
      that was selected.
    - The original override (or absence of one) is always restored on exit,
      even when an exception propagates out of the ``with`` block.
    """
    global _backend_override

    from ._backend import backends as _backends

    resolved = _backends.resolve(backend)  # raises ValueError if unavailable

    original_override = _backend_override
    try:
        _backend_override = resolved
        yield resolved
    finally:
        _backend_override = original_override


@contextmanager
def use_kernel(kernel: str, backend: str = "best"):
    """
    Resolve a backend for a specific kernel, with graceful fallback.

    Unlike :func:`use_backend`, this context manager consults the kernel
    implementation registry (``_backend.KERNEL_IMPLEMENTATIONS``) and
    automatically falls back to the next-best available backend when the
    requested one has no implementation for *kernel*.  A ``WARNING`` is
    emitted when a fallback occurs.

    Parameters
    ----------
    kernel : str
        Kernel name (key in ``quarimo._backend.KERNEL_IMPLEMENTATIONS``).
        Example: ``'quartet_topology'``.
    backend : str, default ``'best'``
        Requested backend, or ``'best'`` to let the registry pick the
        highest-priority implemented backend.

    Yields
    ------
    str
        The concrete backend that will be used — either the one requested or
        the fallback.

    Examples
    --------
    >>> with use_kernel("quartet_topology", effective_backend) as b:
    ...     if b == "cuda":
    ...         # GPU path
    ...     elif b == "cpu-parallel":
    ...         # Numba path
    """
    from ._backend import backends as _backends

    effective, fell_back_from = _backends.resolve_for_kernel(kernel, backend)

    if fell_back_from is not None:
        _logger.warning(
            "Backend '%s' has no '%s' kernel; falling back to '%s'",
            fell_back_from,
            kernel,
            effective,
        )

    yield effective


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
            yield
