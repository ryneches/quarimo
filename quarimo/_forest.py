"""
phylo_tree_collection.py
========================
A memory-efficient collection of phylogenetic trees in a CSR-like flat-packed
layout, designed for bulk quartet topology queries and eventual GPU transfer.

Public API
----------
  Forest(newick_strings)
      Constructor.  Accepts a list of NEWICK strings.  Parses each into a
      Tree, builds a global taxon namespace, then packs all arrays into
      contiguous numpy buffers.

  .branch_distance(taxon_a, taxon_b) -> np.ndarray[float64, n_trees]
      Patristic (branch-length) distance between two taxa across every tree.
      NaN where either taxon is absent from a tree.

Logging
-------
The module uses Python's standard logging framework with one logger:

  logging.getLogger('phylo_tree_collection')
      INFO level:    System capabilities (CPU, memory, numba version),
                     optimization status (LLVM, CUDA, threading),
                     construction stages, taxa/tree counts, memory footprint,
                     namespace overlap statistics, array dimensions.
      WARNING level: Multifurcation corrections, high memory usage alerts,
                     numba performance warnings (e.g., parallel=True but no
                     prange loops detected).

Module-level optimization logging
----------------------------------
On first import, the module logs system and optimization library status at
INFO level.  This happens once per Python session and reports:

  - CPU count, memory (if psutil available), Python version
  - Numba version, llvmlite version, threading configuration (if numba available)
  - CUDA availability and GPU details (if CUDA backend available)

NumbaPerformanceWarning messages are intercepted and routed through the
logger at WARNING level, ensuring all diagnostics appear in one stream.

Users can control logging in the standard way:

    import logging
    # Silence INFO messages, keep warnings:
    logging.getLogger('phylo_tree_collection').setLevel(logging.WARNING)

    # Silence all messages from this module:
    logging.getLogger('phylo_tree_collection').setLevel(logging.CRITICAL)

    # Or configure via root logger before importing:
    logging.basicConfig(level=logging.WARNING)

Memory layout
-------------
All per-tree arrays are concatenated into flat 1-D numpy buffers.  A set of
CSR-style offset vectors (one entry per tree plus a sentinel) allows O(1)
slicing into any tree's data:

  Per-node data  (indexed by local node ID 0..n_nodes-1):
    all_parent, all_distance, all_support
    all_left_child, all_right_child
    all_depth, all_root_distance, all_first_occ

  Per-tour data  (indexed by local tour position 0..2n-2):
    all_euler_tour, all_euler_depth

  Sparse table   (LOG × tour_len per tree, row-major, concatenated):
    all_sparse_table
    sp_log_widths   — LOG_i for tree i  (rows of that tree's table)
    sp_tour_widths  — tour_len_i         (columns / stride)

  Log2 table     (length tour_len+1 per tree):
    all_log2_table

  Offset arrays  (length n_trees+1, int64):
    node_offsets   — into all_parent / all_root_distance / all_first_occ …
    tour_offsets   — into all_euler_tour / all_euler_depth
    sp_offsets     — into all_sparse_table
    lg_offsets     — into all_log2_table
    leaf_offsets   — into local_to_global (leaves only, 0..n_leaves-1)

  Per-tree scalar arrays  (length n_trees, int32):
    per_tree_n_nodes, per_tree_n_leaves, per_tree_roots, per_tree_max_depth

Global taxon namespace
----------------------
Taxon names are collected across all trees, sorted deterministically (ASCII
order), and assigned a contiguous integer *global ID* (gid) 0..G-1 where
G = n_global_taxa.

  global_names : list[str]   — global_names[gid] = taxon name
  n_global_taxa : int        — G

  global_to_local : int32 ndarray (n_trees, G)
      global_to_local[tree_idx, gid] = local leaf ID in that tree, or -1
      if the taxon is absent.  Dense 2-D layout enables O(1) presence checks
      and is GPU-coalesced when querying one taxon across all trees (column
      slice) or all taxa in one tree (row slice).

  local_to_global : int32 ndarray (total_leaves,)
      Flat array indexed by leaf_offsets.
      local_to_global[leaf_offsets[ti] + local_leaf_id] = gid.

  taxa_present : bool ndarray (n_trees, G)
      taxa_present[ti, gid] = (global_to_local[ti, gid] >= 0).
      Convenience mask for filtering quartets before RMQ calls.

Tiling notes
------------
The current design assumes all tree data fits in a single device allocation.
If tiling becomes necessary (Case 2/3 from the design discussion), the offset
arrays define the natural tile boundaries:

  * Tree-axis tiling: slice [node_offsets[a]:node_offsets[b]] etc.
  * Quartet-axis tiling: the global_to_local and taxa_present arrays remain
    resident; only the per-tree data needs to be streamed.

GPU / numba notes
-----------------
Every flat array is a contiguous numpy array with an explicit dtype.  The
branch-length kernel uses only integer offsets and array reads — no Python
objects.  The _rmq_csr() static method is the primary candidate for
@numba.njit decoration; its signature contains only plain scalars and 1-D
arrays.
"""

import hashlib
import logging
from itertools import combinations
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np

from quarimo._tree import Tree

# Import logging functions from separate module
from quarimo._logging import (
    log_optimization_status,
    install_numba_warning_filter,
    log_backend_availability,
    log_multifurcation_warning,
    log_group_statistics,
    log_jaccard_similarities,
    log_collection_statistics,
    compute_tree_taxa_sets,
    compute_group_taxa_sets,
    compute_memory_footprint,
)

# Import backend detection from separate module
from quarimo._backend import (
    check_numba_available,
    check_cuda_available,
    get_available_backends,
    get_best_backend,
    resolve_backend,
    import_cpu_kernels,
    import_cuda_kernels,
)

# Import utilities from separate module
from quarimo._utils import jaccard_similarity, validate_quartet, format_newick

# Import context managers from separate module
from quarimo._context import suppress_logger, get_backend_override

# Backward compatibility alias for tests
_jaccard = jaccard_similarity

# ── Optional numba acceleration ──────────────────────────────────────────────
# Import CPU kernels using backend module
_NUMBA_AVAILABLE = check_numba_available()
_cpu_import_ok, _rmq_csr_nb, _quartet_counts_njit, _quartet_steiner_njit = (
    import_cpu_kernels()
)

# Fallback for prange if numba not available
if _NUMBA_AVAILABLE:
    from numba import prange
else:
    prange = range  # serial fallback

logger = logging.getLogger(__name__)


# ── System and optimization info logging ─────────────────────────────────────
# These functions are now in _logging.py and imported above


# ── Backend availability detection ──────────────────────────────────────────
# Use backend module for detection
_, _CUDA_AVAILABLE = check_cuda_available()
_BACKENDS_AVAILABLE = get_available_backends()


# Track first calls to kernels for compilation logging
_kernel_first_call = {
    "cpu-parallel-counts": True,
    "cpu-parallel-steiner": True,
    "cuda-counts": True,
    "cuda-steiner": True,
}


# Log system info and backend availability on module import
log_optimization_status(_NUMBA_AVAILABLE)
log_backend_availability(_BACKENDS_AVAILABLE, _NUMBA_AVAILABLE)
install_numba_warning_filter(_NUMBA_AVAILABLE)


# ======================================================================== #
# CPU Kernels imported from _kernels_cpu.py                                #
# ======================================================================== #
# The following functions are imported from _kernels_cpu:                  #
#   - _rmq_csr_nb: O(1) RMQ helper for CSR-packed arrays                  #
#   - _quartet_counts_njit: Parallel counts-only kernel                   #
#   - _quartet_steiner_njit: Parallel kernel with Steiner distances       #
#                                                                           #
# These are module-level numba-JIT functions, separated into their own     #
# file to isolate numba dependencies and reduce import-time complexity.    #
# ======================================================================== #


# ======================================================================== #
# CUDA kernels imported from _cuda_kernels.py                              #
# ======================================================================== #
# The following functions are imported from _cuda_kernels:                  #
#   - _rmq_csr_cuda: Device function for GPU RMQ                           #
#   - _quartet_counts_cuda: GPU counts-only kernel                         #
#   - _quartet_steiner_cuda: GPU kernel with Steiner distances             #
#   - _compute_cuda_grid: Helper for CUDA grid dimensions                  #
#                                                                           #
# These are cuda.jit functions, separated into their own file to isolate   #
# CUDA dependencies and reduce import-time complexity.                      #
# ======================================================================== #

if _CUDA_AVAILABLE:
    _cuda_import_ok, _quartet_counts_cuda, _quartet_steiner_cuda, _compute_cuda_grid = (
        import_cuda_kernels()
    )
    # Also need the device function for RMQ
    if _cuda_import_ok:
        from quarimo._cuda_kernels import _rmq_csr_cuda

        # Import cuda module for device operations (cuda.to_device, cuda.device_array, etc.)
        from numba import cuda
    else:
        _CUDA_AVAILABLE = False


# ======================================================================== #
# Helper functions                                                          #
# ======================================================================== #


class Forest:
    """
    An immutable collection of phylogenetic trees in CSR flat-packed layout.

    Parameters
    ----------
    newick_input : list[str] or dict[str, list[str]]
        Either:
        - **list of NEWICK strings** → auto-labeled as single group with
          deterministic 10-character hash label generated via BLAKE2b.
        - **dict mapping group labels to lists of NEWICK strings** → explicit
          group labels for comparing quartet topology frequencies and distance
          distributions between labeled groups.

        Multifurcating trees are automatically resolved to bifurcating form
        with a warning (delegated to Tree).

    Attributes (read-only after construction)
    -----------------------------------------
    **Core attributes:**

    n_trees          : int
    n_global_taxa    : int
    global_names     : list[str]          sorted taxon names
    global_to_local  : int32 (n_trees, G) local leaf ID or -1
    local_to_global  : int32 (total_leaves,)
    taxa_present     : bool  (n_trees, G)
    leaf_offsets     : int64 (n_trees+1,)

    **Group label attributes:**

    group_labels     : list[str]          per-tree group label (n_trees,)
    unique_groups    : list[str]          sorted unique group names
    n_groups         : int                number of distinct groups
    group_to_tree_indices : dict[str, np.ndarray]
                                          maps group name to tree indices
    tree_to_group_idx : int32 (n_trees,) tree index → group index
    group_offsets    : int64 (n_groups+1,)
                                          CSR offset array for groups

    **CSR layout arrays:**

    node_offsets, tour_offsets, sp_offsets, lg_offsets : int64 (n_trees+1,)
    sp_log_widths, sp_tour_widths                      : int32 (n_trees,)
    per_tree_n_nodes, per_tree_n_leaves                : int32 (n_trees,)
    per_tree_roots, per_tree_max_depth                 : int32 (n_trees,)

    all_parent, all_left_child, all_right_child        : int32
    all_depth, all_first_occ                           : int32
    all_euler_tour, all_euler_depth                    : int32
    all_sparse_table                                   : int32
    all_log2_table                                     : int32
    all_distance, all_support                          : float64
    all_root_distance                                  : float64

    Examples
    --------
    **Explicit groups (dict input):**

    >>> groups = {
    ...     'species_A': ['((a1:1,a2:1):1,(a3:1,a4:1):1);', ...],
    ...     'species_B': ['((b1:1,b2:1):1,(b3:1,b4:1):1);', ...],
    ... }
    >>> c = Forest(groups)
    >>> c.n_groups
    2
    >>> c.unique_groups
    ['species_A', 'species_B']

    **Auto-labeled single group (list input):**

    >>> trees = ['((A:1,B:1):1,(C:1,D:1):1);', ...]
    >>> c = Forest(trees)
    >>> c.n_groups
    1
    >>> c.unique_groups[0]  # 10-character hash
    'a1b2c3d4e5'

    **Group-aware quartet analysis:**

    >>> counts, dists = c.quartet_topology(quartets, steiner=True)
    >>> by_group = c.split_quartet_results_by_group(counts, dists)
    >>> for group_name, (group_counts, group_dists) in by_group.items():
    ...     print(f"{group_name}: {group_counts[0]}")
    """

    # ================================================================== #
    # Construction                                                         #
    # ================================================================== #

    def __init__(self, newick_input) -> None:
        """Initialize collection from NEWICK strings with optional group labels."""

        # Convert list input to dict with auto-generated hash key
        if isinstance(newick_input, (list, tuple)):
            # Generate deterministic label from input NEWICK strings
            sorted_newicks = sorted(newick_input)
            combined = "".join(sorted_newicks)
            h = hashlib.blake2b(combined.encode("utf-8"), digest_size=5)
            auto_label = h.hexdigest()

            logger.info("Auto-generated group label: %s", auto_label)

            # Convert to dict format
            newick_input = {auto_label: list(newick_input)}
        elif not isinstance(newick_input, dict):
            raise TypeError(
                f"newick_input must be list or dict, got {type(newick_input).__name__}"
            )

        # Parse groups
        self.unique_groups = sorted(newick_input.keys())
        self.n_groups = len(self.unique_groups)

        # Track multifurcation corrections globally
        n_multifurcating = 0
        multifurcating_indices = []

        # Flatten into ordered list with group tracking
        self._trees = []
        self.group_labels = []
        tree_idx = 0

        # Suppress phylo_tree logger during tree construction
        with suppress_logger("quarimo._tree"):
            for group_name in self.unique_groups:
                newicks = newick_input[group_name]
                if not newicks:
                    raise ValueError(f"Group '{group_name}' is empty")

                for newick in newicks:
                    # Check for multifurcation
                    s = newick.strip().rstrip(";")
                    if s.count("(") < s.count(","):
                        n_multifurcating += 1
                        multifurcating_indices.append(tree_idx)

                    self._trees.append(Tree(newick))
                    self.group_labels.append(group_name)
                    tree_idx += 1

        # Set n_trees BEFORE logging multifurcation warning
        self.n_trees = len(self._trees)

        # Emit consolidated multifurcation warning
        self._log_multifurcation_warning_method(
            n_multifurcating, multifurcating_indices
        )

        # Build group mappings
        self._build_group_mappings()

        # Log tree count and group organization
        logger.info("Loading %d tree(s) from NEWICK strings...", self.n_trees)
        if self.n_groups > 1:
            logger.info("  Organized into %d labeled groups", self.n_groups)
        else:
            logger.info("  Single group: %s", self.unique_groups[0])

        # Build namespace and CSR layout
        logger.info("Building global taxon namespace...")
        self._build_global_namespace()

        logger.info("Packing arrays into CSR flat layout...")
        self._pack_csr()

        # Log group statistics and Jaccard similarities
        self._log_group_statistics_method()
        self._log_jaccard_similarities_method()

        # Log overall statistics
        self._log_statistics_method()

    def _build_group_mappings(self) -> None:
        """Build group → tree indices mappings."""
        # Forward mapping: group_name -> array of tree indices
        self.group_to_tree_indices = {
            group_name: np.array(
                [i for i, g in enumerate(self.group_labels) if g == group_name],
                dtype=np.int64,
            )
            for group_name in self.unique_groups
        }

        # Reverse mapping: tree_idx -> group index in unique_groups
        group_to_idx = {g: i for i, g in enumerate(self.unique_groups)}
        self.tree_to_group_idx = np.array(
            [group_to_idx[g] for g in self.group_labels], dtype=np.int32
        )

    def _log_multifurcation_warning_method(
        self, n_multifurcating: int, multifurcating_indices: List[int]
    ) -> None:
        """Emit consolidated multifurcation warning."""
        log_multifurcation_warning(
            n_multifurcating, multifurcating_indices, self.n_trees
        )

    def _log_group_statistics_method(self) -> None:
        """Log group membership statistics."""
        log_group_statistics(
            self.n_groups, self.unique_groups, self.group_to_tree_indices
        )

    def _log_jaccard_similarities_method(self) -> None:
        """Compute and log Jaccard similarities within and between groups."""
        # Compute taxa sets (separate computation from logging)
        tree_taxa_sets = compute_tree_taxa_sets(self._trees)
        group_taxa_sets = compute_group_taxa_sets(
            self.group_to_tree_indices, tree_taxa_sets
        )

        # Log the results
        log_jaccard_similarities(
            self.unique_groups,
            self.group_to_tree_indices,
            tree_taxa_sets,
            group_taxa_sets,
        )

    # ================================================================== #
    # Public methods                                                       #
    # ================================================================== #

    def branch_distance(self, taxon_a, taxon_b) -> np.ndarray:
        """
        Compute the patristic (sum-of-branch-lengths) distance between
        *taxon_a* and *taxon_b* through every tree in the collection.

        Parameters
        ----------
        taxon_a, taxon_b : str | int
            Taxon names (looked up in the global namespace) or global IDs.

        Returns
        -------
        np.ndarray[float64, shape=(n_trees,)]
            Distance for each tree.  NaN where either taxon is absent.

        Raises
        ------
        KeyError   if a name is not found in the global namespace.

        Complexity
        ----------
        O(n_trees) with one O(1) RMQ per tree that contains both taxa.
        """
        ga = self._resolve_global(taxon_a)
        gb = self._resolve_global(taxon_b)

        result = np.full(self.n_trees, np.nan, dtype=np.float64)

        g2l = self.global_to_local
        fo = self.all_first_occ
        rd = self.all_root_distance
        et = self.all_euler_tour
        ed = self.all_euler_depth
        sp = self.all_sparse_table
        lg = self.all_log2_table
        no = self.node_offsets
        to_ = self.tour_offsets
        so = self.sp_offsets
        lo_ = self.lg_offsets
        stw = self.sp_tour_widths

        for ti in range(self.n_trees):
            la = int(g2l[ti, ga])
            lb = int(g2l[ti, gb])
            if la < 0 or lb < 0:
                continue
            if la == lb:
                result[ti] = 0.0
                continue

            node_base = int(no[ti])
            tour_base = int(to_[ti])
            sp_base = int(so[ti])
            lg_base = int(lo_[ti])
            tw = int(stw[ti])

            l = int(fo[node_base + la])
            r = int(fo[node_base + lb])
            if l > r:
                l, r = r, l

            lca_local = Forest._rmq_csr(
                l, r, sp_base, tw, sp, ed, lg, lg_base, tour_base, et
            )

            result[ti] = (
                float(rd[node_base + la])
                + float(rd[node_base + lb])
                - 2.0 * float(rd[node_base + lca_local])
            )

        return result

    def quartet_topology(
        self, quartets: Any, steiner: bool = False, backend: str = "best"
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Count the three possible unrooted quartet topologies for one or more
        quartets across all trees in the collection.

        This is the single public entry-point for all quartet topology queries.
        Both single-quartet and bulk queries use the same calling convention:
        pass an iterable of 4-tuples.  For a single quartet, wrap it in a list:
        ``[[a, b, c, d]]``.  The return shapes are consistent regardless of
        how many quartets are queried, so results are directly usable in either
        case without special-casing by the caller.

        Topology encoding
        -----------------
        For each quartet the four taxa are sorted by global ID to canonical
        order n0 < n1 < n2 < n3.  The three unrooted topologies are:

          Index 0:  (n0, n1) | (n2, n3)
          Index 1:  (n0, n2) | (n1, n3)
          Index 2:  (n0, n3) | (n1, n2)

        Parameters
        ----------
        quartets : iterable of (a, b, c, d)
            Each element is a 4-tuple of taxon names (str) or global IDs (int),
            in any order.  The iterable is fully consumed once; generators are
            accepted.
        steiner : bool, default False
            If True, also return per-tree Steiner distances for the winning
            topology in each quartet.

        Returns
        -------
        counts : np.ndarray[int32, shape=(n_quartets, 3)]
            counts[qi, k] = number of trees where quartet qi has topology k.
            Trees where any of the four taxa are absent do not contribute.

        steiner_distances : np.ndarray[float64, shape=(n_quartets, n_trees, 3)]
            Only returned when steiner=True.

            steiner_distances[qi, ti, k] =
              • The true Steiner spanning length for quartet qi in tree ti
                — if topology k won and all four taxa are present.
              • 0.0 otherwise (taxon absent, or topology k ≠ winning topology).

            Exactly one entry per (qi, ti) row is non-zero for trees where the
            quartet is fully present; the non-zero column identifies the winning
            topology and its value is the true Steiner spanning length.

            Useful aggregations (all vectorised with numpy, no Python loops):
              counts[qi, k] == (dists[qi, :, k] > 0).sum()
              mean_steiner[qi, k] = dists[qi, :, k].sum() / counts[qi, k]
              total_steiner[qi]   = dists[qi].sum()

        Steiner formula
        ---------------
        Let rdXY = root_distance[LCA(nX, nY)], the six pairwise LCA root
        distances already computed for topology detection.  Define:

          r0 = rd01 + rd23,  r1 = rd02 + rd13,  r2 = rd03 + rd12
          r_winner = max(r0, r1, r2)
          leaf_rd_sum = root_distance[n0] + … + root_distance[n3]

        The Steiner length of the winning topology is:

          S = leaf_rd_sum − (r_winner + r0 + r1 + r2) / 2

        No additional RMQ calls are needed beyond those used for topology
        detection; Steiner mode costs 4 float reads and 9 float ops per tree.

        Pre-processing
        --------------
        The ``quartets`` iterable is materialised into a contiguous
        ``int32 (n_quartets, 4)`` array of sorted global IDs before the kernel
        is called.  This array is the sole quartet-varying input to the kernel;
        it can be cached and reused across multiple calls if the same set of
        quartets is queried for different collections sharing a namespace.

        GPU parallelisation
        -------------------
        The kernel exposes a natural 2D thread space:

          axis 0 (qi): quartet index  — n_quartets threads
          axis 1 (ti): tree index     — n_trees threads

        For @numba.cuda.jit conversion, replace the two Python for-loops with:

          qi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
          ti = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        steiner_out[qi, ti, topo] = S is conflict-free (each thread owns a
        unique (qi, ti) row); only counts_out[qi, topo] += 1 requires
        cuda.atomic.add.

        Complexity
        ----------
        O(n_quartets × n_trees × 6) RMQ calls, each O(1).
        For GPU: O(6) per thread with n_threads = n_quartets × n_trees.

        Examples
        --------
        Single quartet:

        >>> counts = c.quartet_topology([['A', 'B', 'C', 'D']])
        >>> print(counts[0])          # topology distribution: (3,) view of row 0
        [10, 3, 7]

        >>> counts, dists = c.quartet_topology([['A', 'B', 'C', 'D']], steiner=True)
        >>> print(dists[0, :, :].sum(axis=0))   # total Steiner per topology
        [34.1   0.   0. ]

        Multiple quartets:

        >>> quartets = [('A','B','C','D'), ('A','B','C','E'), ('B','C','D','E')]
        >>> counts = c.quartet_topology(quartets)
        >>> print(counts.shape)
        (3, 3)

        >>> counts, dists = c.quartet_topology(quartets, steiner=True)
        >>> print(dists.shape)
        (3, 20, 3)          # n_quartets × n_trees × 3 topologies

        Backend selection:

        >>> counts = c.quartet_topology(quartets)                        # 'best' (default)
        >>> counts = c.quartet_topology(quartets, backend='python')      # unoptimized
        >>> counts = c.quartet_topology(quartets, backend='cpu-parallel') # numba CPU
        >>> counts = c.quartet_topology(quartets, backend='cuda')        # GPU (if available)

        Backend
        -------
        backend : str, default 'best'
            Selects the execution mode for the quartet kernel:

            'best'
                Automatically selects the most optimized backend available.
                Preference order: cuda > cpu-parallel > python.
                This is the recommended default for production use.

            'cpu-parallel'
                LLVM-compiled parallel code via numba.njit(parallel=True).
                The outer loop over quartets runs in parallel using prange
                (OpenMP threads).  On first call the kernel is JIT-compiled;
                subsequent calls load from cache.  Falls back to 'python' if
                numba is not available.

                Thread count: numba.set_num_threads(n) before calling.

            'cuda'
                GPU acceleration via numba.cuda.jit.  Transfers data to GPU,
                executes the kernel on device, and copies results back.
                Falls back to 'cpu-parallel' if CUDA unavailable, or 'python'
                if numba unavailable.

            'python'
                Pure-Python reference implementation.  Always available,
                portable, debuggable.  Serves as the correctness baseline.

        Execution mode logging
        ----------------------
        Each call to quartet_topology() logs the selected backend at INFO
        level.  On first invocation of a JIT-compiled backend, an additional
        message indicates whether the kernel was compiled or loaded from cache.
        """
        # ── 1. Materialise and validate the input ────────────────────────
        quartet_list = list(quartets)
        n_quartets = len(quartet_list)
        if n_quartets == 0:
            counts_out = np.zeros((0, 3), dtype=np.int32)
            if steiner:
                return counts_out, np.zeros((0, self.n_trees, 3), dtype=np.float64)
            return counts_out

        # ── 2. Resolve names → global IDs, sort each quartet ────────────
        # Validation (KeyError on unknown names) happens here, before the
        # kernel, so callers get an immediate, informative error rather than
        # a silent wrong result or an index fault inside the kernel.
        sorted_ids = np.empty((n_quartets, 4), dtype=np.int32)
        for qi, quad in enumerate(quartet_list):
            ids = sorted([self._resolve_global(x) for x in quad])
            sorted_ids[qi, 0] = ids[0]
            sorted_ids[qi, 1] = ids[1]
            sorted_ids[qi, 2] = ids[2]
            sorted_ids[qi, 3] = ids[3]

        # ── 3. Pre-allocate outputs and call kernel ──────────────────────
        common_args = (
            sorted_ids,
            self.global_to_local,
            self.all_first_occ,
            self.all_root_distance,
            self.all_euler_tour,
            self.all_euler_depth,
            self.all_sparse_table,
            self.all_log2_table,
            self.node_offsets,
            self.tour_offsets,
            self.sp_offsets,
            self.lg_offsets,
            self.sp_tour_widths,
            n_quartets,
            self.n_trees,
        )

        # ── 3. Resolve backend and log execution mode ────────────────────
        # Check for backend override from context manager
        backend_override = get_backend_override()
        if backend_override is not None:
            backend = backend_override

        try:
            resolved_backend = resolve_backend(backend)
        except ValueError as e:
            # Backend not available, fall back to best available
            logger.warning(str(e))
            resolved_backend = get_best_backend()

        # Log execution mode
        mode_str = "Steiner" if steiner else "counts-only"
        logger.info(f"quartet_topology({mode_str}, backend={resolved_backend!r})")

        # ── 4. Dispatch to the selected backend ──────────────────────────
        counts_out = np.zeros((n_quartets, 3), dtype=np.int32)

        if resolved_backend == "cuda":
            # Track compilation status
            kernel_key = f"cuda-{'steiner' if steiner else 'counts'}"
            if _kernel_first_call.get(kernel_key, False):
                logger.info(
                    f"  Compiling {kernel_key} kernel (cached for future calls)"
                )
                _kernel_first_call[kernel_key] = False

            # ── Log data transfer: Host → Device ─────────────────────────
            # Calculate transfer sizes
            tree_data_arrays = [
                ("global_to_local", self.global_to_local),
                ("all_first_occ", self.all_first_occ),
                ("all_root_distance", self.all_root_distance),
                ("all_euler_tour", self.all_euler_tour),
                ("all_euler_depth", self.all_euler_depth),
                ("all_sparse_table", self.all_sparse_table),
                ("all_log2_table", self.all_log2_table),
                ("node_offsets", self.node_offsets),
                ("tour_offsets", self.tour_offsets),
                ("sp_offsets", self.sp_offsets),
                ("lg_offsets", self.lg_offsets),
                ("sp_tour_widths", self.sp_tour_widths),
            ]

            query_data_arrays = [
                ("sorted_quartet_ids", sorted_ids),
                ("counts_out (zeros)", counts_out),
            ]

            tree_bytes = sum(arr.nbytes for _, arr in tree_data_arrays)
            query_bytes = sum(arr.nbytes for _, arr in query_data_arrays)

            logger.info(f"  Transferring data to GPU device:")
            logger.info(
                f"    Tree data: {len(tree_data_arrays)} arrays, "
                f"{tree_bytes / (1024**2):.2f} MB"
            )
            logger.info(
                f"      - global_to_local: {self.global_to_local.shape} {self.global_to_local.dtype}"
            )
            logger.info(
                f"      - CSR packed arrays: {self.n_trees} trees, "
                f"{self.all_sparse_table.size} sparse table entries"
            )
            logger.info(
                f"    Query data: {len(query_data_arrays)} arrays, "
                f"{query_bytes / (1024**2):.2f} MB"
            )
            logger.info(
                f"      - sorted_quartet_ids: {sorted_ids.shape} {sorted_ids.dtype}"
            )

            if steiner:
                steiner_bytes = n_quartets * self.n_trees * 3 * 8  # float64
                logger.info(
                    f"    Output arrays: {steiner_bytes / (1024**2):.2f} MB "
                    f"(steiner_out {(n_quartets, self.n_trees, 3)} float64)"
                )
                total_bytes = tree_bytes + query_bytes + steiner_bytes
            else:
                total_bytes = tree_bytes + query_bytes

            logger.info(f"    Total H→D transfer: {total_bytes / (1024**2):.2f} MB")

            # Transfer data to GPU
            d_sorted_ids = cuda.to_device(sorted_ids)
            d_global_to_local = cuda.to_device(self.global_to_local)
            d_all_first_occ = cuda.to_device(self.all_first_occ)
            d_all_root_distance = cuda.to_device(self.all_root_distance)
            d_all_euler_tour = cuda.to_device(self.all_euler_tour)
            d_all_euler_depth = cuda.to_device(self.all_euler_depth)
            d_all_sparse_table = cuda.to_device(self.all_sparse_table)
            d_all_log2_table = cuda.to_device(self.all_log2_table)
            d_node_offsets = cuda.to_device(self.node_offsets)
            d_tour_offsets = cuda.to_device(self.tour_offsets)
            d_sp_offsets = cuda.to_device(self.sp_offsets)
            d_lg_offsets = cuda.to_device(self.lg_offsets)
            d_sp_tour_widths = cuda.to_device(self.sp_tour_widths)
            d_counts_out = cuda.to_device(counts_out)

            # Compute grid dimensions
            blocks_per_grid, threads_per_block = _compute_cuda_grid(
                n_quartets, self.n_trees
            )
            total_threads = (
                blocks_per_grid[0]
                * threads_per_block[0]
                * blocks_per_grid[1]
                * threads_per_block[1]
            )
            active_threads = n_quartets * self.n_trees

            logger.info(f"  Launching CUDA kernel:")
            logger.info(
                f"    Grid: {blocks_per_grid[0]}×{blocks_per_grid[1]} blocks, "
                f"{threads_per_block[0]}×{threads_per_block[1]} threads/block"
            )
            logger.info(
                f"    Total threads: {total_threads:,} "
                f"(active: {active_threads:,}, idle: {total_threads - active_threads:,})"
            )

            if steiner:
                steiner_out = np.zeros((n_quartets, self.n_trees, 3), dtype=np.float64)
                d_steiner_out = cuda.to_device(steiner_out)

                # Launch kernel
                _quartet_steiner_cuda[blocks_per_grid, threads_per_block](
                    d_sorted_ids,
                    d_global_to_local,
                    d_all_first_occ,
                    d_all_root_distance,
                    d_all_euler_tour,
                    d_all_euler_depth,
                    d_all_sparse_table,
                    d_all_log2_table,
                    d_node_offsets,
                    d_tour_offsets,
                    d_sp_offsets,
                    d_lg_offsets,
                    d_sp_tour_widths,
                    n_quartets,
                    self.n_trees,
                    d_counts_out,
                    d_steiner_out,
                )

                # Copy results back
                result_bytes = counts_out.nbytes + steiner_out.nbytes
                logger.info(f"  Transferring results from GPU device:")
                logger.info(
                    f"    counts_out: {counts_out.shape} {counts_out.dtype}, "
                    f"{counts_out.nbytes / 1024:.1f} KB"
                )
                logger.info(
                    f"    steiner_out: {steiner_out.shape} {steiner_out.dtype}, "
                    f"{steiner_out.nbytes / (1024**2):.2f} MB"
                )
                logger.info(
                    f"    Total D→H transfer: {result_bytes / (1024**2):.2f} MB"
                )

                d_counts_out.copy_to_host(counts_out)
                d_steiner_out.copy_to_host(steiner_out)
                return counts_out, steiner_out
            else:
                # Launch kernel
                _quartet_counts_cuda[blocks_per_grid, threads_per_block](
                    d_sorted_ids,
                    d_global_to_local,
                    d_all_first_occ,
                    d_all_root_distance,
                    d_all_euler_tour,
                    d_all_euler_depth,
                    d_all_sparse_table,
                    d_all_log2_table,
                    d_node_offsets,
                    d_tour_offsets,
                    d_sp_offsets,
                    d_lg_offsets,
                    d_sp_tour_widths,
                    n_quartets,
                    self.n_trees,
                    d_counts_out,
                )

                # Copy results back
                logger.info(f"  Transferring results from GPU device:")
                logger.info(
                    f"    counts_out: {counts_out.shape} {counts_out.dtype}, "
                    f"{counts_out.nbytes / 1024:.1f} KB"
                )
                logger.info(
                    f"    Total D→H transfer: {counts_out.nbytes / 1024:.1f} KB"
                )

                d_counts_out.copy_to_host(counts_out)
                return counts_out

        elif resolved_backend == "cpu-parallel":
            # Track compilation status
            kernel_key = f"cpu-parallel-{'steiner' if steiner else 'counts'}"
            if _kernel_first_call.get(kernel_key, False):
                logger.info(
                    f"  Compiling {kernel_key} kernel (cached for future calls)"
                )
                _kernel_first_call[kernel_key] = False

            if steiner:
                steiner_out = np.zeros((n_quartets, self.n_trees, 3), dtype=np.float64)
                _quartet_steiner_njit(*common_args, counts_out, steiner_out)
                return counts_out, steiner_out
            else:
                _quartet_counts_njit(*common_args, counts_out)
                return counts_out

        elif resolved_backend == "python":
            steiner_out = (
                np.zeros((n_quartets, self.n_trees, 3), dtype=np.float64)
                if steiner
                else np.empty(0, dtype=np.float64)
            )
            Forest._quartet_kernel(*common_args, counts_out, steiner_out)
            return (counts_out, steiner_out) if steiner else counts_out

        else:
            # This should never be reached due to validation above
            raise RuntimeError(
                f"Internal error: unhandled backend {resolved_backend!r}"
            )

    def split_quartet_results_by_group(
        self, counts: np.ndarray, steiner_distances: Optional[np.ndarray]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split quartet topology results by tree group.

        This method requires per-tree data to determine which group contributed
        which topology decisions. Since counts-only mode aggregates across all
        trees, you must call quartet_topology(..., steiner=True) to get the
        necessary per-tree information.

        Parameters
        ----------
        counts : np.ndarray[int32, shape=(n_quartets, 3)]
            Total topology counts from quartet_topology().
        steiner_distances : np.ndarray[float64, shape=(n_quartets, n_trees, 3)]
            Per-tree Steiner distances from quartet_topology(..., steiner=True).
            Cannot be None.

        Returns
        -------
        dict[str, tuple[np.ndarray, np.ndarray]]
            Maps each group label to (group_counts, group_steiner_distances).

            group_counts : np.ndarray[int32, shape=(n_quartets, 3)]
                Topology counts recomputed for this group only.

            group_steiner_distances : np.ndarray[float64, shape=(n_quartets, n_trees_in_group, 3)]
                Steiner distances for trees in this group only.

        Raises
        ------
        ValueError
            If steiner_distances is None (counts-only mode).

        Examples
        --------
        >>> groups = {
        ...     'species_A': [...],
        ...     'species_B': [...],
        ... }
        >>> c = Forest(groups)
        >>>
        >>> # Must use steiner=True to get per-tree data
        >>> counts, dists = c.quartet_topology(quartets, steiner=True)
        >>>
        >>> # Split by group
        >>> by_group = c.split_quartet_results_by_group(counts, dists)
        >>>
        >>> # Analyze each group
        >>> for group_name, (group_counts, group_dists) in by_group.items():
        ...     print(f"{group_name}: {group_counts[0]}")
        """
        if steiner_distances is None:
            raise ValueError(
                "Cannot split counts without per-tree data. "
                "Re-run: counts, dists = collection.quartet_topology(..., steiner=True)"
            )

        results = {}

        for group_name in self.unique_groups:
            # Get tree indices for this group
            tree_indices = self.group_to_tree_indices[group_name]

            # Extract steiner distances for this group's trees
            group_dists = steiner_distances[:, tree_indices, :]

            # Recompute counts from distances
            # A tree contributes to topology k if dists[qi, ti, k] > 0
            group_counts = (group_dists > 0).sum(axis=1).astype(np.int32)

            results[group_name] = (group_counts, group_dists)

        return results

    # ================================================================== #
    # Private instance methods                                             #
    # ================================================================== #

    def _build_global_namespace(self) -> None:
        """
        **Private.**  Collect all unique taxon names across all trees, assign
        sorted global IDs, and build the global↔local mapping arrays.
        """
        name_set: set = set()
        for t in self._trees:
            for i in range(t.n_leaves):
                if t.names[i]:
                    name_set.add(t.names[i])

        self.global_names = sorted(name_set)
        self.n_global_taxa = len(self.global_names)
        self._name_to_global: dict = {n: i for i, n in enumerate(self.global_names)}

        G = self.n_global_taxa
        NT = self.n_trees

        # Leaf offset array
        leaf_sizes = np.array([t.n_leaves for t in self._trees], dtype=np.int64)
        self.leaf_offsets = np.zeros(NT + 1, dtype=np.int64)
        self.leaf_offsets[1:] = np.cumsum(leaf_sizes)
        total_leaves = int(self.leaf_offsets[-1])

        # Dense global→local map and flat local→global map
        self.global_to_local = np.full((NT, G), -1, dtype=np.int32)
        self.local_to_global = np.full(total_leaves, -1, dtype=np.int32)

        for ti, t in enumerate(self._trees):
            lo = int(self.leaf_offsets[ti])
            # Ensure name index is built
            if t._name_index is None:
                t._build_name_index()
            for name, local_id in t._name_index.items():
                if name in self._name_to_global:
                    gid = self._name_to_global[name]
                    self.global_to_local[ti, gid] = local_id
                    self.local_to_global[lo + local_id] = gid

        # Convenience presence mask
        self.taxa_present = self.global_to_local >= 0

    def _pack_csr(self) -> None:
        """
        **Private.**  Concatenate all per-tree numpy arrays into flat buffers
        and build the CSR offset vectors.

        Sparse-table layout
        -------------------
        Tree i's sparse table has shape (LOG_i, tour_len_i) and is stored
        row-major (level is the outer axis) in all_sparse_table.  The stride
        within tree i's slice is sp_tour_widths[i].  To access element
        [k, pos] for tree i:

            idx = sp_offsets[i] + k * sp_tour_widths[i] + pos
            value = all_sparse_table[idx]

        The value is a *local* tour position (0-based within tree i's tour);
        add tour_offsets[i] to obtain the global position in all_euler_tour.
        """
        trees = self._trees
        NT = self.n_trees

        # ---- Per-tree dimension vectors -------------------------------- #
        node_sizes = np.array([t.n_nodes for t in trees], dtype=np.int64)
        tour_sizes = np.array([len(t.euler_tour) for t in trees], dtype=np.int64)
        lg_sizes = np.array([len(t.log2_table) for t in trees], dtype=np.int64)

        sp_log_widths = np.array(
            [t.sparse_table.shape[0] for t in trees], dtype=np.int32
        )
        sp_tour_widths = np.array(
            [t.sparse_table.shape[1] for t in trees], dtype=np.int32
        )
        sp_sizes = sp_log_widths.astype(np.int64) * sp_tour_widths.astype(np.int64)

        self.sp_log_widths = sp_log_widths
        self.sp_tour_widths = sp_tour_widths

        # ---- CSR offset arrays ---------------------------------------- #
        self.node_offsets = np.zeros(NT + 1, dtype=np.int64)
        self.tour_offsets = np.zeros(NT + 1, dtype=np.int64)
        self.sp_offsets = np.zeros(NT + 1, dtype=np.int64)
        self.lg_offsets = np.zeros(NT + 1, dtype=np.int64)

        self.node_offsets[1:] = np.cumsum(node_sizes)
        self.tour_offsets[1:] = np.cumsum(tour_sizes)
        self.sp_offsets[1:] = np.cumsum(sp_sizes)
        self.lg_offsets[1:] = np.cumsum(lg_sizes)

        # ---- Per-tree scalar arrays ------------------------------------ #
        self.per_tree_n_nodes = np.array([t.n_nodes for t in trees], dtype=np.int32)
        self.per_tree_n_leaves = np.array([t.n_leaves for t in trees], dtype=np.int32)
        self.per_tree_roots = np.array([t.root for t in trees], dtype=np.int32)
        self.per_tree_max_depth = np.array([t.max_depth for t in trees], dtype=np.int32)

        # ---- Flat per-node arrays (int32) ------------------------------ #
        self.all_parent = np.concatenate([t.parent for t in trees])
        self.all_left_child = np.concatenate([t.left_child for t in trees])
        self.all_right_child = np.concatenate([t.right_child for t in trees])
        self.all_depth = np.concatenate([t.depth for t in trees])
        self.all_first_occ = np.concatenate([t.first_occurrence for t in trees])

        # ---- Flat per-node arrays (float64) ---------------------------- #
        self.all_distance = np.concatenate([t.distance for t in trees])
        self.all_support = np.concatenate([t.support for t in trees])
        self.all_root_distance = np.concatenate([t.root_distance for t in trees])

        # ---- Flat per-tour arrays (int32) ------------------------------ #
        self.all_euler_tour = np.concatenate([t.euler_tour for t in trees])
        self.all_euler_depth = np.concatenate([t.euler_depth for t in trees])

        # ---- Flat sparse tables (int32, row-major per tree) ------------ #
        # Each tree's sparse_table is (LOG_i, tour_len_i); .ravel() gives
        # the row-major flattening with stride sp_tour_widths[i].
        self.all_sparse_table = np.concatenate(
            [t.sparse_table.ravel() for t in trees]
        ).astype(np.int32)

        # ---- Flat log2 tables (int32) ---------------------------------- #
        self.all_log2_table = np.concatenate([t.log2_table for t in trees])

        # ---- Group offset array (CSR-style for groups) ----------------- #
        self.group_offsets = np.zeros(self.n_groups + 1, dtype=np.int64)
        for i, group_name in enumerate(self.unique_groups):
            n_trees_in_group = len(self.group_to_tree_indices[group_name])
            self.group_offsets[i + 1] = self.group_offsets[i] + n_trees_in_group

        # Sanity check
        assert self.group_offsets[-1] == self.n_trees, (
            "Group offsets don't sum to total trees"
        )

    def _log_statistics_method(self) -> None:
        """
        **Private.**  Log collection statistics: taxon counts, memory usage,
        namespace overlap, and potential memory warnings.

        Called automatically at the end of __init__.
        """
        # Compute namespace overlap statistics
        taxa_per_tree = self.taxa_present.sum(axis=1)
        trees_per_taxon = self.taxa_present.sum(axis=0)
        mean_taxa = float(taxa_per_tree.mean())
        mean_trees = float(trees_per_taxon.mean())

        # Compute array dimensions
        total_nodes = int(self.node_offsets[-1])
        total_tour = int(self.tour_offsets[-1])
        total_sp = int(self.sp_offsets[-1])
        total_lg = int(self.lg_offsets[-1])

        # Compute memory footprint
        mem_bytes = compute_memory_footprint(self)

        # Log everything
        log_collection_statistics(
            self.n_trees,
            self.n_global_taxa,
            int(self.leaf_offsets[-1]),
            mean_taxa,
            mean_trees,
            total_nodes,
            total_tour,
            total_sp,
            total_lg,
            mem_bytes,
        )

    # ================================================================== #
    # Private helper methods                                               #
    # ================================================================== #

    def _resolve_global(self, taxon: Union[str, int]) -> int:
        """
        **Private.**  Return the global taxon ID for *taxon*.

        Accepts a taxon name (str) or an already-resolved global ID (int).
        """
        if isinstance(taxon, str):
            if taxon not in self._name_to_global:
                raise KeyError(
                    f"Taxon '{taxon}' not found in global namespace. "
                    f"Known taxa: {self.global_names}"
                )
            return self._name_to_global[taxon]
        return int(taxon)

    # ================================================================== #
    # Private static methods (kernel candidates for numba.njit)           #
    # ================================================================== #

    @staticmethod
    def _quartet_kernel(
        sorted_quartet_ids: np.ndarray,
        global_to_local: np.ndarray,
        all_first_occ: np.ndarray,
        all_root_distance: np.ndarray,
        all_euler_tour: np.ndarray,
        all_euler_depth: np.ndarray,
        all_sparse_table: np.ndarray,
        all_log2_table: np.ndarray,
        node_offsets: np.ndarray,
        tour_offsets: np.ndarray,
        sp_offsets: np.ndarray,
        lg_offsets: np.ndarray,
        sp_tour_widths: np.ndarray,
        n_quartets: int,
        n_trees: int,
        counts_out: np.ndarray,
        steiner_out: np.ndarray,
    ) -> None:
        """
        **Private static.**  Unified bulk quartet kernel.

        Computes topology counts for every (quartet, tree) pair, and
        optionally the Steiner spanning length of the winning topology.

        Parameters
        ----------
        sorted_quartet_ids : int32 (n_quartets, 4)
            Each row holds four global taxon IDs sorted in ascending order.
            Produced by the public method; not validated here.
        global_to_local : int32 (n_trees, G)
        all_first_occ … sp_tour_widths : CSR-packed arrays (see class docs)
        n_quartets, n_trees : int
        counts_out : int32 (n_quartets, 3), pre-filled with zeros
            Filled in-place.  counts_out[qi, k] = number of trees where
            quartet qi has topology k.
        steiner_out : float64 array, pre-filled with zeros
            If steiner_out.size == 0 (sentinel: pass np.empty(0)):
                Steiner calculation is skipped entirely.
            Otherwise, must have shape (n_quartets, n_trees, 3).
                steiner_out[qi, ti, k] = Steiner length if topology k won for
                quartet qi in tree ti; 0.0 otherwise.

        Steiner bypass
        --------------
        The sentinel value ``steiner_out = np.empty(0, dtype=np.float64)``
        signals counts-only mode.  Inside the per-(qi,ti) body, the Steiner
        block (4 float reads + 9 float ops + 1 array write) is skipped with a
        single ``if steiner_out.size > 0`` guard.  No branching overhead is
        incurred in the 6-RMQ topology core.

        On GPU (numba.cuda.jit), ``steiner_out.size`` resolves at kernel-launch
        time to a compile-time constant when the array is device-resident, so
        the branch is eliminated entirely by the JIT compiler.

        GPU conversion
        --------------
        Replace the two Python for-loops with:

          qi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
          ti = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
          if qi >= n_quartets or ti >= n_trees: return

        steiner_out[qi, ti, topo] = S  is a plain global-memory store (no
        atomic needed — each thread owns a unique (qi, ti) row).
        counts_out[qi, topo] += 1  requires cuda.atomic.add.

        Eligible for @numba.njit (CPU) or @numba.cuda.jit (GPU) decoration
        without modification.

        Statistical moment notes
        ------------------------
        MOMENT B — replace steiner_out[qi, ti, topo] = S with Welford
          accumulation into mean_k[qi, topo] and M2_k[qi, topo], reducing
          output from (n_quartets, n_trees, 3) to (n_quartets, 3, 3).

        MOMENT C — replace with: weighted_counts[qi, topo] += 1.0 / S
          Output shrinks to (n_quartets, 3) float64.

        MOMENT D — replace with: hist[qi, topo, int(S / bin_width)] += 1
          Output is (n_quartets, 3, n_bins).
        """
        compute_steiner = steiner_out.size > 0

        for qi in range(n_quartets):
            n0 = int(sorted_quartet_ids[qi, 0])
            n1 = int(sorted_quartet_ids[qi, 1])
            n2 = int(sorted_quartet_ids[qi, 2])
            n3 = int(sorted_quartet_ids[qi, 3])

            for ti in range(n_trees):
                ln0 = int(global_to_local[ti, n0])
                ln1 = int(global_to_local[ti, n1])
                ln2 = int(global_to_local[ti, n2])
                ln3 = int(global_to_local[ti, n3])

                if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                    continue

                nb = int(node_offsets[ti])
                tb = int(tour_offsets[ti])
                sb = int(sp_offsets[ti])
                lb = int(lg_offsets[ti])
                tw = int(sp_tour_widths[ti])

                l = int(all_first_occ[nb + ln0])
                r = int(all_first_occ[nb + ln1])
                if l > r:
                    l, r = r, l
                lca = Forest._rmq_csr(
                    l,
                    r,
                    sb,
                    tw,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    lb,
                    tb,
                    all_euler_tour,
                )
                rd01 = float(all_root_distance[nb + lca])

                l = int(all_first_occ[nb + ln0])
                r = int(all_first_occ[nb + ln2])
                if l > r:
                    l, r = r, l
                lca = Forest._rmq_csr(
                    l,
                    r,
                    sb,
                    tw,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    lb,
                    tb,
                    all_euler_tour,
                )
                rd02 = float(all_root_distance[nb + lca])

                l = int(all_first_occ[nb + ln0])
                r = int(all_first_occ[nb + ln3])
                if l > r:
                    l, r = r, l
                lca = Forest._rmq_csr(
                    l,
                    r,
                    sb,
                    tw,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    lb,
                    tb,
                    all_euler_tour,
                )
                rd03 = float(all_root_distance[nb + lca])

                l = int(all_first_occ[nb + ln1])
                r = int(all_first_occ[nb + ln2])
                if l > r:
                    l, r = r, l
                lca = Forest._rmq_csr(
                    l,
                    r,
                    sb,
                    tw,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    lb,
                    tb,
                    all_euler_tour,
                )
                rd12 = float(all_root_distance[nb + lca])

                l = int(all_first_occ[nb + ln1])
                r = int(all_first_occ[nb + ln3])
                if l > r:
                    l, r = r, l
                lca = Forest._rmq_csr(
                    l,
                    r,
                    sb,
                    tw,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    lb,
                    tb,
                    all_euler_tour,
                )
                rd13 = float(all_root_distance[nb + lca])

                l = int(all_first_occ[nb + ln2])
                r = int(all_first_occ[nb + ln3])
                if l > r:
                    l, r = r, l
                lca = Forest._rmq_csr(
                    l,
                    r,
                    sb,
                    tw,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    lb,
                    tb,
                    all_euler_tour,
                )
                rd23 = float(all_root_distance[nb + lca])

                r0 = rd01 + rd23  # (n0,n1)|(n2,n3)
                r1 = rd02 + rd13  # (n0,n2)|(n1,n3)
                r2 = rd03 + rd12  # (n0,n3)|(n1,n2)

                if r0 > r1:
                    if r0 > r2:
                        topo = 0
                        r_winner = r0
                    else:
                        topo = 2
                        r_winner = r2
                else:
                    if r1 > r2:
                        topo = 1
                        r_winner = r1
                    else:
                        topo = 2
                        r_winner = r2

                counts_out[qi, topo] += 1

                if compute_steiner:
                    leaf_rd_sum = (
                        float(all_root_distance[nb + ln0])
                        + float(all_root_distance[nb + ln1])
                        + float(all_root_distance[nb + ln2])
                        + float(all_root_distance[nb + ln3])
                    )

                    S = leaf_rd_sum - (r_winner + r0 + r1 + r2) * 0.5

                    # Plain store — no atomic needed: (qi, ti) is unique per thread.
                    steiner_out[qi, ti, topo] = S

                    # MOMENT B: replace the store above with Welford accumulation
                    # into mean_k[qi, topo] and M2_k[qi, topo], reducing the
                    # output from (n_quartets, n_trees, 3) to (n_quartets, 3, 3).

                    # MOMENT C: replace the store above with:
                    #   weighted_counts[qi, topo] += 1.0 / S
                    # Output shrinks to (n_quartets, 3) float64.

                    # MOMENT D: replace the store above with:
                    #   hist[qi, topo, int(S / bin_width)] += 1
                    # Output is (n_quartets, 3, n_bins).

    @staticmethod
    def _rmq_csr(
        l: int,
        r: int,
        sp_base: int,
        sp_stride: int,
        sparse_table,
        euler_depth,
        log2_table,
        lg_base: int,
        tour_base: int,
        euler_tour,
    ) -> int:
        """
        **Private static.**  O(1) RMQ using CSR-packed arrays for a single tree.

        All index arguments are *absolute* positions in the flat buffers.
        Sparse-table values are *local* tour indices (0-based within the tree);
        the caller adds *tour_base* when accessing euler_tour/euler_depth.

        Parameters
        ----------
        l, r        : int  Inclusive local tour range (l ≤ r).
        sp_base     : int  Offset of this tree's sparse table in all_sparse_table.
        sp_stride   : int  sp_tour_widths[ti] — column stride in the sparse table.
        sparse_table: int32 array  all_sparse_table.
        euler_depth : int32 array  all_euler_depth.
        log2_table  : int32 array  all_log2_table.
        lg_base     : int  Offset of this tree's log2_table in all_log2_table.
        tour_base   : int  Offset of this tree's tour in all_euler_tour/depth.
        euler_tour  : int32 array  all_euler_tour.

        Returns
        -------
        int   Local node ID of the minimum-depth node in tour[l..r].

        Notes
        -----
        Eligible for @numba.njit decoration without modification.
        In C:  int32_t rmq_csr(int l, int r, ...)
        """
        length = r - l + 1
        k = int(log2_table[lg_base + length])
        half = 1 << k

        li = int(sparse_table[sp_base + k * sp_stride + l])
        ri = int(sparse_table[sp_base + k * sp_stride + (r - half + 1)])

        if int(euler_depth[tour_base + ri]) < int(euler_depth[tour_base + li]):
            lca_tour_local = ri
        else:
            lca_tour_local = li

        return int(euler_tour[tour_base + lca_tour_local])
