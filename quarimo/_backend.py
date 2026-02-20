"""
_backend.py
===========
Backend detection and selection for forest.

This module detects available execution backends (Python, CPU-parallel via numba,
CUDA GPU) and provides functions to query and select the best backend.

Functions in this module have NO side effects - they only query system state.
Logging is done by the calling code, not here.
"""

from typing import List, Tuple, Optional


# ============================================================================ #
# Backend Detection (No Side Effects)
# ============================================================================ #


def check_numba_available() -> bool:
    """
    Check if numba is available for CPU parallelization.

    Returns
    -------
    bool
        True if numba can be imported, False otherwise.
    """
    try:
        import numba

        return True
    except ImportError:
        return False


def check_cuda_available() -> Tuple[bool, bool]:
    """
    Check if CUDA GPU acceleration is available.

    Returns
    -------
    tuple[bool, bool]
        (numba_available, cuda_available)
        - numba_available: Whether numba is installed
        - cuda_available: Whether CUDA GPU is available
    """
    try:
        import numba
        from numba import cuda

        return (True, cuda.is_available())
    except ImportError:
        return (False, False)
    except Exception:
        # numba available but CUDA check failed
        return (True, False)


def get_available_backends() -> List[str]:
    """
    Get list of available execution backends.

    Returns
    -------
    list[str]
        List of available backends in preference order.
        Always includes 'python'.
        May include 'cpu-parallel' if numba is available.
        May include 'cuda' if CUDA GPU is available.

    Examples
    --------
    >>> get_available_backends()
    ['python']  # No numba installed

    >>> get_available_backends()
    ['python', 'cpu-parallel']  # Numba installed, no GPU

    >>> get_available_backends()
    ['python', 'cpu-parallel', 'cuda']  # Full stack
    """
    backends = ["python"]  # Always available

    numba_available = check_numba_available()
    if numba_available:
        backends.append("cpu-parallel")

        # Check CUDA (requires numba)
        _, cuda_available = check_cuda_available()
        if cuda_available:
            backends.append("cuda")

    return backends


def get_best_backend() -> str:
    """
    Get the most optimized available backend.

    Returns
    -------
    str
        Best available backend in preference order:
        'cuda' > 'cpu-parallel' > 'python'

    Examples
    --------
    >>> get_best_backend()
    'cuda'  # If GPU available

    >>> get_best_backend()
    'cpu-parallel'  # If numba but no GPU

    >>> get_best_backend()
    'python'  # Fallback
    """
    backends = get_available_backends()
    # List is in preference order, last is best
    return backends[-1]


def resolve_backend(backend: str) -> str:
    """
    Resolve a backend specification to an actual backend.

    Parameters
    ----------
    backend : str
        Backend specification:
        - 'best': Use the best available backend
        - 'python', 'cpu-parallel', 'cuda': Use specific backend

    Returns
    -------
    str
        Resolved backend name.

    Raises
    ------
    ValueError
        If requested backend is not available.

    Examples
    --------
    >>> resolve_backend('best')
    'cuda'  # Returns best available

    >>> resolve_backend('cpu-parallel')
    'cpu-parallel'  # If available

    >>> resolve_backend('cuda')
    ValueError  # If CUDA not available
    """
    if backend == "best":
        return get_best_backend()

    # Validate requested backend is available
    available = get_available_backends()
    if backend not in available:
        raise ValueError(
            f"Backend '{backend}' not available. "
            f"Available backends: {', '.join(available)}"
        )

    return backend


# ============================================================================ #
# Kernel Import Helpers
# ============================================================================ #


def import_cpu_kernels() -> Tuple[
    bool, Optional[object], Optional[object], Optional[object]
]:
    """
    Try to import CPU kernels from _cpu_kernels module.

    Returns
    -------
    tuple
        (success, rmq_kernel, counts_kernel, steiner_kernel)
        - success: Whether import succeeded
        - rmq_kernel: _rmq_csr_nb function or None
        - counts_kernel: _quartet_counts_njit function or None
        - steiner_kernel: _quartet_steiner_njit function or None
    """
    try:
        from quarimo._cpu_kernels import (
            _rmq_csr_nb,
            _quartet_counts_njit,
            _quartet_steiner_njit,
        )

        return (True, _rmq_csr_nb, _quartet_counts_njit, _quartet_steiner_njit)
    except ImportError:
        return (False, None, None, None)


def import_cuda_kernels() -> Tuple[
    bool, Optional[object], Optional[object], Optional[object]
]:
    """
    Try to import CUDA kernels from _cuda_kernels module.

    Returns
    -------
    tuple
        (success, counts_kernel, steiner_kernel, compute_grid)
        - success: Whether import succeeded
        - counts_kernel: _quartet_counts_cuda function or None
        - steiner_kernel: _quartet_steiner_cuda function or None
        - compute_grid: _compute_cuda_grid function or None
    """
    try:
        from quarimo._cuda_kernels import (
            _quartet_counts_cuda,
            _quartet_steiner_cuda,
            _compute_cuda_grid,
        )

        return (True, _quartet_counts_cuda, _quartet_steiner_cuda, _compute_cuda_grid)
    except ImportError:
        return (False, None, None, None)


# ============================================================================ #
# Module-Level State Query (Read-Only)
# ============================================================================ #


def get_backend_info() -> dict:
    """
    Get comprehensive backend information.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'numba_available': bool
        - 'cuda_available': bool
        - 'backends': list[str]
        - 'best_backend': str
        - 'cpu_kernels_available': bool
        - 'cuda_kernels_available': bool

    Examples
    --------
    >>> info = get_backend_info()
    >>> info['best_backend']
    'cuda'
    >>> info['backends']
    ['python', 'cpu-parallel', 'cuda']
    """
    numba_available = check_numba_available()
    _, cuda_available = check_cuda_available()
    backends = get_available_backends()
    best = get_best_backend()

    cpu_kernels_ok, _, _, _ = import_cpu_kernels()
    cuda_kernels_ok, _, _, _ = import_cuda_kernels()

    return {
        "numba_available": numba_available,
        "cuda_available": cuda_available,
        "backends": backends,
        "best_backend": best,
        "cpu_kernels_available": cpu_kernels_ok,
        "cuda_kernels_available": cuda_kernels_ok,
    }
