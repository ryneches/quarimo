"""
_logging.py
===========
Logging functions for phylo_tree_collection.

All functions in this module have NO side effects except logging. They take
computed data as parameters and format/emit log messages.

This separation ensures:
- Logging can be easily disabled/mocked in tests
- Computation is separate from presentation
- Clear boundaries between analysis and reporting
"""

import logging
import numpy as np
from itertools import combinations
from typing import List, Dict, Set, Any

# Import utility functions
from quarimo._utils import jaccard_similarity


logger = logging.getLogger(__name__)


# ============================================================================ #
# System and Backend Logging (called at module import time)
# ============================================================================ #


def log_optimization_status(numba_available: bool) -> None:
    """
    Log system capabilities and optimization library availability at INFO level.

    Called once at module import time. Reports CPU count, memory, numba version
    (if available), LLVM/CUDA info, and threading configuration.

    Parameters
    ----------
    numba_available : bool
        Whether numba was successfully imported.
    """
    import os
    import platform

    # Basic system info
    cpu_count = os.cpu_count() or 1
    logger.info(
        f"System: {platform.machine()} ({platform.system()}), "
        f"{cpu_count} CPU cores, Python {platform.python_version()}"
    )

    # Memory info (optional psutil)
    try:
        import psutil

        mem = psutil.virtual_memory()
        logger.info(
            f"Memory: {mem.total / (1024**3):.1f} GB total, "
            f"{mem.available / (1024**3):.1f} GB available"
        )
    except ImportError:
        pass  # psutil not required

    # Numba availability and version info
    if numba_available:
        import numba

        logger.info(f"Numba {numba.__version__} loaded successfully")

        # LLVM version from llvmlite (numba's backend)
        try:
            import llvmlite

            logger.info(f"LLVM backend: llvmlite {llvmlite.__version__}")
        except (ImportError, AttributeError):
            pass  # LLVM version unavailable

        # Threading configuration
        try:
            num_threads = numba.get_num_threads()
            threading_layer = numba.threading_layer()
            logger.info(
                f"Numba threading: {threading_layer} layer, "
                f"{num_threads} threads active"
            )
        except Exception:
            pass  # Threading info unavailable in some configs

        # CUDA availability
        try:
            from numba import cuda

            if cuda.is_available():
                gpus = cuda.gpus
                logger.info(f"CUDA available: {len(gpus)} GPU(s) detected")
                for i, gpu in enumerate(gpus):
                    logger.info(f"  GPU {i}: {gpu.name.decode()}")
            else:
                logger.info("CUDA not available (no compatible GPU)")
        except Exception:
            logger.info("CUDA backend unavailable")

    else:
        logger.info("Numba not installed — quartet kernels will run as pure Python")
        logger.info("Install numba for ~10-100x speedup: pip install numba")


def install_numba_warning_filter(numba_available: bool) -> None:
    """
    Capture NumbaPerformanceWarning and route it through our logger.

    numba issues performance warnings (e.g., "parallel=True but no prange found")
    via Python's warnings module. This filter intercepts them and logs them at
    WARNING level via our logger so they appear in the same stream as other
    phylo_tree_collection diagnostics.

    Parameters
    ----------
    numba_available : bool
        Whether numba was successfully imported.
    """
    import warnings

    if not numba_available:
        return  # No numba, no warnings to capture

    try:
        from numba.core.errors import NumbaPerformanceWarning

        def numba_warning_handler(
            message, category, filename, lineno, file=None, line=None
        ):
            """Custom showwarning that routes NumbaPerformanceWarning to logger."""
            if issubclass(category, NumbaPerformanceWarning):
                # Extract just the warning message text
                msg = str(message)
                logger.warning(f"Numba performance issue: {msg}")
                logger.warning(f"  at {filename}:{lineno}")
                # Return True to suppress the default warning display
                return True
            # Return False to use default handling for other warnings
            return False

        # Install a warnings filter that calls our handler
        original_showwarning = warnings.showwarning

        def custom_showwarning(
            message, category, filename, lineno, file=None, line=None
        ):
            if numba_warning_handler(message, category, filename, lineno, file, line):
                return  # Handled by our custom handler
            # Fall back to original for non-numba warnings
            original_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = custom_showwarning

    except ImportError:
        pass  # NumbaPerformanceWarning not available in this numba version


def log_backend_availability(
    backends_available: List[str], numba_available: bool
) -> None:
    """
    Log which execution backends are available for quartet kernels.

    Parameters
    ----------
    backends_available : List[str]
        List of available backends (e.g., ['python', 'cpu-parallel', 'cuda'])
    numba_available : bool
        Whether numba was successfully imported.
    """
    logger.info(f"Available backends: {', '.join(backends_available)}")

    if "cpu-parallel" in backends_available:
        logger.info("  cpu-parallel: LLVM-compiled parallel code (numba.njit + prange)")

    if "cuda" in backends_available:
        logger.info("  cuda: GPU acceleration via CUDA")
    else:
        if numba_available:
            logger.info("  cuda: unavailable (no compatible GPU detected)")

    if "python" in backends_available:
        logger.info("  python: unoptimized reference implementation")

    # Log default behavior
    best = backends_available[-1]  # Last in list is most optimized
    logger.info(f"Default backend='best' will use: {best}")


# ============================================================================ #
# Tree Collection Logging (called during initialization)
# ============================================================================ #


def log_multifurcation_warning(
    n_multifurcating: int, multifurcating_indices: List[int], n_trees: int
) -> None:
    """
    Emit consolidated multifurcation warning.

    Parameters
    ----------
    n_multifurcating : int
        Number of trees that required multifurcation resolution.
    multifurcating_indices : List[int]
        Indices of trees that were multifurcating.
    n_trees : int
        Total number of trees in collection.
    """
    if n_multifurcating > 0:
        if n_multifurcating == 1:
            logger.warning(
                "1 tree required multifurcation resolution (tree index %d). "
                "Zero-length branches were added to enforce bifurcation.",
                multifurcating_indices[0],
            )
        elif n_multifurcating <= 5:
            logger.warning(
                "%d trees required multifurcation resolution (tree indices: %s). "
                "Zero-length branches were added to enforce bifurcation.",
                n_multifurcating,
                ", ".join(map(str, multifurcating_indices)),
            )
        else:
            logger.warning(
                "%d trees required multifurcation resolution (%.1f%% of total). "
                "Zero-length branches were added to enforce bifurcation.",
                n_multifurcating,
                100.0 * n_multifurcating / n_trees,
            )


def log_group_statistics(
    n_groups: int, unique_groups: List[str], group_to_tree_indices: Dict[str, List[int]]
) -> None:
    """
    Log group membership statistics.

    Parameters
    ----------
    n_groups : int
        Number of distinct groups.
    unique_groups : List[str]
        Ordered list of unique group labels.
    group_to_tree_indices : Dict[str, List[int]]
        Mapping from group label to list of tree indices.
    """
    if n_groups == 1:
        # Single group - already logged in main flow
        return

    logger.info("Tree groups: %d distinct labels", n_groups)
    for group_name in unique_groups:
        n_trees_in_group = len(group_to_tree_indices[group_name])
        logger.info("  Group '%s': %d trees", group_name, n_trees_in_group)


def log_jaccard_similarities(
    unique_groups: List[str],
    group_to_tree_indices: Dict[str, List[int]],
    tree_taxa_sets: List[Set[str]],
    group_taxa_sets: Dict[str, Set[str]],
) -> None:
    """
    Log Jaccard similarities within and between groups.

    This function only logs - the caller must compute tree_taxa_sets and
    group_taxa_sets beforehand.

    Parameters
    ----------
    unique_groups : List[str]
        Ordered list of unique group labels.
    group_to_tree_indices : Dict[str, List[int]]
        Mapping from group label to list of tree indices.
    tree_taxa_sets : List[Set[str]]
        Taxa sets for each tree (list of sets).
    group_taxa_sets : Dict[str, Set[str]]
        Taxa sets for each group (union of tree taxa).
    """
    # Log per-group taxa counts
    for group_name in unique_groups:
        n_taxa = len(group_taxa_sets[group_name])
        logger.info("  Group '%s': %d distinct taxa", group_name, n_taxa)

    # Within-group similarities (mean pairwise Jaccard)
    logger.info("Within-group taxa Jaccard similarities (avg pairwise):")
    for group_name in unique_groups:
        tree_indices = group_to_tree_indices[group_name]

        if len(tree_indices) < 2:
            logger.info("  %s: N/A (only 1 tree)", group_name)
        else:
            similarities = [
                jaccard_similarity(tree_taxa_sets[i], tree_taxa_sets[j])
                for i, j in combinations(tree_indices, 2)
            ]
            avg = np.mean(similarities)
            logger.info("  %s: %.3f", group_name, avg)

    # Between-group similarities
    between_pairs = list(combinations(unique_groups, 2))
    if between_pairs:
        logger.info("Between-group taxa Jaccard similarities:")
        for group_a, group_b in between_pairs:
            jac = jaccard_similarity(group_taxa_sets[group_a], group_taxa_sets[group_b])
            logger.info("  %s ↔ %s: %.3f", group_a, group_b, jac)


def log_collection_statistics(
    n_trees: int,
    n_global_taxa: int,
    total_leaves: int,
    taxa_per_tree_mean: float,
    trees_per_taxon_mean: float,
    total_nodes: int,
    total_tour: int,
    total_sp: int,
    total_lg: int,
    memory_bytes: int,
) -> None:
    """
    Log collection statistics: taxon counts, memory usage, namespace overlap.

    Parameters
    ----------
    n_trees : int
        Number of trees in collection.
    n_global_taxa : int
        Number of distinct taxa across all trees.
    total_leaves : int
        Total number of leaves across all trees.
    taxa_per_tree_mean : float
        Average number of taxa per tree.
    trees_per_taxon_mean : float
        Average number of trees per taxon.
    total_nodes : int
        Total number of nodes across all trees.
    total_tour : int
        Total Euler tour length.
    total_sp : int
        Total sparse table entries.
    total_lg : int
        Total log2 table entries.
    memory_bytes : int
        Total memory footprint in bytes.
    """
    # Basic counts
    logger.info(
        "Collection built: %d trees, %d global taxa, %d total leaves",
        n_trees,
        n_global_taxa,
        total_leaves,
    )

    # Namespace overlap statistics
    logger.info(
        "Namespace overlap: %.1f taxa/tree (avg), %.1f trees/taxon (avg)",
        taxa_per_tree_mean,
        trees_per_taxon_mean,
    )

    # Warn if namespace is highly disjoint (low overlap)
    if trees_per_taxon_mean < 2.0 and n_trees > 2:
        logger.warning(
            "Low namespace overlap detected: average taxon appears in %.1f trees. "
            "This may indicate highly disjoint taxon sets across trees, which "
            "reduces the number of viable quartet queries.",
            trees_per_taxon_mean,
        )

    # Array dimension summary
    logger.info(
        "Array dimensions: nodes=%d, tour=%d, sparse_table=%d, log2_table=%d",
        total_nodes,
        total_tour,
        total_sp,
        total_lg,
    )

    # Memory footprint
    mem_mb = memory_bytes / (1024**2)
    mem_gb = memory_bytes / (1024**3)

    if mem_gb >= 1.0:
        logger.info("Total memory footprint: %.2f GB", mem_gb)
    else:
        logger.info("Total memory footprint: %.1f MB", mem_mb)

    # Memory warnings (default thresholds: 80% of 16 GB system, 80% of 8 GB GPU)
    system_threshold_gb = 16 * 0.8  # 12.8 GB
    gpu_threshold_gb = 8 * 0.8  # 6.4 GB

    if mem_gb > system_threshold_gb:
        logger.warning(
            "Memory footprint (%.2f GB) exceeds typical system memory threshold "
            "(%.1f GB = 80%% of 16 GB). Consider filtering taxa or using fewer trees.",
            mem_gb,
            system_threshold_gb,
        )

    if mem_gb > gpu_threshold_gb:
        logger.warning(
            "Memory footprint (%.2f GB) exceeds typical GPU memory threshold "
            "(%.1f GB = 80%% of 8 GB). GPU operations may require batching or "
            "a higher-memory device.",
            mem_gb,
            gpu_threshold_gb,
        )


# ============================================================================ #
# Helper Functions for Computing Data (not logging)
# ============================================================================ #


def compute_tree_taxa_sets(trees: List[Any]) -> List[Set[str]]:
    """
    Build taxa sets per tree (filter out None/empty names).

    Parameters
    ----------
    trees : List[Any]
        List of PhyloTree objects.

    Returns
    -------
    List[Set[str]]
        Taxa sets for each tree.
    """
    return [{name for name in tree.names if name} for tree in trees]


def compute_group_taxa_sets(
    group_to_tree_indices: Dict[str, List[int]], tree_taxa_sets: List[Set[str]]
) -> Dict[str, Set[str]]:
    """
    Build taxa sets per group (union of tree taxa sets).

    Parameters
    ----------
    group_to_tree_indices : Dict[str, List[int]]
        Mapping from group label to list of tree indices.
    tree_taxa_sets : List[Set[str]]
        Taxa sets for each tree.

    Returns
    -------
    Dict[str, Set[str]]
        Taxa sets for each group.
    """
    return {
        group_name: set().union(*(tree_taxa_sets[i] for i in indices))
        for group_name, indices in group_to_tree_indices.items()
    }


def compute_memory_footprint(collection: Any) -> int:
    """
    Compute total memory footprint of collection arrays.

    Parameters
    ----------
    collection : Any
        The PhyloTreeCollection object.

    Returns
    -------
    int
        Total memory in bytes.
    """
    mem_bytes = 0

    # Integer arrays (int32 = 4 bytes)
    int32_arrays = [
        collection.all_parent,
        collection.all_left_child,
        collection.all_right_child,
        collection.all_depth,
        collection.all_first_occ,
        collection.all_euler_tour,
        collection.all_euler_depth,
        collection.all_sparse_table,
        collection.all_log2_table,
        collection.per_tree_n_nodes,
        collection.per_tree_n_leaves,
        collection.per_tree_roots,
        collection.per_tree_max_depth,
        collection.sp_log_widths,
        collection.sp_tour_widths,
        collection.global_to_local.ravel(),
        collection.local_to_global,
    ]
    for arr in int32_arrays:
        mem_bytes += arr.nbytes

    # Float arrays (float64 = 8 bytes)
    float64_arrays = [
        collection.all_distance,
        collection.all_support,
        collection.all_root_distance,
    ]
    for arr in float64_arrays:
        mem_bytes += arr.nbytes

    # Offset arrays (int64 = 8 bytes)
    int64_arrays = [
        collection.node_offsets,
        collection.tour_offsets,
        collection.sp_offsets,
        collection.lg_offsets,
        collection.leaf_offsets,
    ]
    for arr in int64_arrays:
        mem_bytes += arr.nbytes

    # Bool arrays (bool = 1 byte)
    mem_bytes += collection.taxa_present.nbytes

    return mem_bytes
