"""
quarimo (クアリモ)
==================

Quartet-based entropy analysis for detecting parallel evolution across
phylogenetic tree ensembles.

*Quarimo* (from "quartet" + "mori" 森 = forest) uses cross-entropy products
of quartet frequencies to estimate the likelihood of parallel evolution.

Main Classes
------------
Forest : Collection of phylogenetic trees with efficient quartet queries
Tree : Single phylogenetic tree with NEWICK parsing and algorithms

Context Managers
----------------
quiet : Suppress logging during operations
suppress_logger : Suppress specific logger
suppress_warnings : Suppress specific warnings
use_backend : Force specific computational backend
silent_benchmark : Combine quiet + backend selection + warning suppression

Utilities
---------
jaccard_similarity : Compute Jaccard similarity between sets
validate_quartet : Validate quartet specification
format_newick : Format NEWICK strings consistently

Backend Information
-------------------
get_available_backends : Query available computational backends
get_backend_info : Get comprehensive backend status
check_numba_available : Check if numba is available
check_cuda_available : Check if CUDA GPU is available

Examples
--------
Basic usage:

>>> from quarimo import Forest
>>> trees = ['((A:1,B:1):1,(C:1,D:1):1);', '((A:1,C:1):1,(B:1,D:1):1);']
>>> forest = Forest(trees)
>>> counts = forest.quartet_topology([('A', 'B', 'C', 'D')])
>>> print(counts)
[[1 1 0]]

With context managers:

>>> from quarimo import Forest, quiet, use_backend
>>> with quiet():
...     forest = Forest(large_tree_list)
>>> with use_backend('cpu-parallel'):
...     counts = forest.quartet_topology(quartets)

Benchmarking:

>>> from quarimo import silent_benchmark
>>> with silent_benchmark('cuda'):
...     counts = forest.quartet_topology(quartets)
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Main classes
from ._forest import Forest
from ._tree import Tree
from ._quartets import Quartets

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
)

# Backward compatibility aliases (deprecated)
import warnings

# Public API
__all__ = [
    # Main classes
    "Forest",
    "Tree",
    "Quartets",
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
    # Version info
    "__version__",
]
