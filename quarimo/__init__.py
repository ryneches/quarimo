"""
Public API for quarimo.

Core classes  (_forest, _tree, _quartets)
-----------------------------------------
Forest    : Collection of trees; dispatches quartet topology queries to backends
Tree      : Single phylogenetic tree; NEWICK parsing, LCA via RMQ, Steiner lengths
Quartets  : Deterministic quartet sequence; explicit lists, random sampling, on-GPU generation

Context managers  (_context)
-----------------------------
use_backend       : Force a specific computational backend for a block of code
use_kernel        : Force a backend for a named kernel; falls back with a warning
quiet             : Temporarily suppress quarimo logging
suppress_logger   : Suppress a single named logger
suppress_warnings : Suppress specific warning categories
silent_benchmark  : Combines quiet + use_backend + suppress_warnings

Backend inspection  (_backend)
-------------------------------
get_available_backends : List backends available on this machine, in priority order
get_backend_info       : Human-readable backend status summary
check_numba_available  : True if numba is importable
check_cuda_available   : True if a CUDA-capable GPU is present
check_mlx_available    : True if mlx (Apple Silicon Metal) is available

Utilities  (_utils)
--------------------
jaccard_similarity : Jaccard similarity between two sets
validate_quartet   : Validate a four-taxon tuple
format_newick      : Normalise a NEWICK string
"""

from quarimo._version import __version__

__author__ = "Russell Neches"
__license__ = "BSD-3-Clause"

# Main classes
from ._forest import Forest
from ._tree import Tree
from ._quartets import Quartets

# Result dataclasses
from ._results import QuartetTopologyResult, QEDResult

# Context managers (user-facing utilities)
from ._context import (
    suppress_logger,
    quiet,
    suppress_warnings,
    use_backend,
    silent_benchmark,
)

# Utilities (generally useful functions)
from ._utils import (
    jaccard_similarity,
    validate_quartet,
    format_newick,
)

# Backend information (useful for checking capabilities)
from ._backend import (
    get_available_backends,
    get_backend_info,
    check_numba_available,
    check_cuda_available,
    check_mlx_available,
)

# Public API
__all__ = [
    # Main classes
    "Forest",
    "Tree",
    "Quartets",
    # Result dataclasses
    "QuartetTopologyResult",
    "QEDResult",
    # Context managers
    "suppress_logger",
    "quiet",
    "suppress_warnings",
    "use_backend",
    "silent_benchmark",
    # Utilities
    "jaccard_similarity",
    "validate_quartet",
    "format_newick",
    # Backend information
    "get_available_backends",
    "get_backend_info",
    "check_numba_available",
    "check_cuda_available",
    "check_mlx_available",
    # Version info
    "__version__",
]
