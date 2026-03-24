"""
_forest.py
==========
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
      WARNING level: High memory usage alerts, low namespace overlap,
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

import functools
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from time import time, perf_counter
from itertools import combinations
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np

from quarimo._tree import Tree
from quarimo._quartets import Quartets

# Import logging functions from separate module
from quarimo._logging import (
    log_optimization_status,
    install_numba_warning_filter,
    log_backend_availability,
    log_polytomy_statistics,
    log_zero_length_branch_warning,
    log_multifurcation_resolved_warning,
    log_group_statistics,
    log_namespace_coverage,
    log_collection_statistics,
    compute_memory_footprint,
)

# Import backend detection from separate module
from quarimo._backend import backends

# Import utilities from separate module
from quarimo._utils import jaccard_similarity, validate_quartet, format_newick, normalize_input

# Import context managers from separate module
from quarimo._context import suppress_logger, get_backend_override, use_kernel

# Import result dataclasses
from quarimo._results import BranchDistanceResult, QEDResult, QuartetTopologyResult

# Import kernel data packaging
from quarimo._kernel_data import ForestKernelData, QuartetKernelArgs

# Backward compatibility alias for tests
_jaccard = jaccard_similarity

if backends.numba:
    from quarimo._cpu_kernels import (
        _rmq_csr_nb,
        _quartet_counts_njit,
        _quartet_steiner_njit,
        _quartet_counts_delta_nb,
    )

logger = logging.getLogger(__name__)

# ── Global CPU thread cap ─────────────────────────────────────────────────────
# None means "no cap — use all available CPUs".  A future public setter will
# expose this for HPC deployments that must cap worker threads across the
# whole process (e.g. SLURM slots, OMP_NUM_THREADS parity).
_GLOBAL_THREAD_CAP: Optional[int] = None

# ── Multiprocessing configuration ────────────────────────────────────────────
# 'fork' on Unix: workers inherit the parent's already-imported modules at
# near-zero startup cost.  Safe here because numba's parallel thread pool has
# not yet been initialised when Forest.__init__ runs.  If you call Forest()
# after quartet_topology() in the same process, pass n_threads=1 to be safe.
# 'spawn' on Windows (fork is unavailable).
_MP_CONTEXT: str = "fork" if sys.platform != "win32" else "spawn"

# Total NEWICK character count below which sequential parsing is used.
# Subprocess overhead (~tens of ms per worker for fork) outweighs the
# parallelism gain for small or tiny-tree datasets.
# ~200 KB ≈ four 1 000-taxon trees or forty 100-taxon trees.
_MP_CHAR_THRESHOLD: int = 200_000


def _resolve_n_threads(n_threads: Optional[int]) -> int:
    """Return the effective worker-thread count for parallel tree loading.

    Priority:  explicit *n_threads* argument  >  _GLOBAL_THREAD_CAP  >  os.cpu_count().
    The result is always ≥ 1.
    """
    effective = n_threads if n_threads is not None else (os.cpu_count() or 1)
    if _GLOBAL_THREAD_CAP is not None:
        effective = min(effective, _GLOBAL_THREAD_CAP)
    return max(1, effective)


# ── System and optimization info logging ─────────────────────────────────────
# These functions are now in _logging.py and imported above


# Track first calls to kernels for compilation logging
_kernel_first_call = {
    "cpu-parallel-counts": True,
    "cpu-parallel-steiner": True,
    "cuda-counts": True,
    "cuda-steiner": True,
}

# Log system info and backend availability on module import
log_optimization_status(backends.numba)
log_backend_availability(backends)
install_numba_warning_filter(backends.numba)

if backends.cuda:
    from quarimo._cuda_kernels import (
        _rmq_csr_cuda,
        _quartet_counts_cuda,
        _quartet_steiner_cuda,
        _compute_cuda_grid,
        _quartet_counts_delta_cuda,
    )
    from numba import cuda  # noqa: F401 — needed for device ops


# ======================================================================== #
# Helper functions                                                          #
# ======================================================================== #


def _parse_newick_worker(newick: str, polytomy_strategy: str = "multifurcation") -> "Tree":
    """Subprocess worker: parse one NEWICK string into a Tree.

    Kept at module scope so it is picklable by :mod:`multiprocessing`.
    """
    return Tree(newick, polytomy_strategy=polytomy_strategy)


def _load_tree_data(
    newick_input: Dict[str, List[str]],
    ordered_groups: List[str],
    n_threads: Optional[int] = None,
    polytomy_strategy: str = "multifurcation",
) -> Tuple[List["Tree"], List[str]]:
    """Parse NEWICK strings into :class:`Tree` instances, optionally in parallel.

    Trees are returned in *ordered_groups* order so that the caller can build
    deterministic group → tree mappings.

    Parameters
    ----------
    newick_input : dict[str, list[str]]
        Mapping from group label to list of NEWICK strings.  Every group
        named in *ordered_groups* must be present.
    ordered_groups : list[str]
        Group labels in the desired construction order.
    n_threads : int or None
        Maximum number of worker *processes* to use.  ``None`` (default) →
        all available CPUs, subject to :data:`_GLOBAL_THREAD_CAP`.  Pass
        ``1`` to force sequential parsing.

        Parallel parsing uses :class:`~concurrent.futures.ProcessPoolExecutor`
        (real OS processes, bypassing the GIL) but only when total NEWICK
        input exceeds :data:`_MP_CHAR_THRESHOLD`; smaller datasets are always
        parsed sequentially to avoid subprocess overhead.

    Returns
    -------
    trees : list[Tree]
        Parsed :class:`Tree` objects, in the same order as the flattened
        (group, newick) sequence.
    group_labels : list[str]
        Per-tree group label, parallel to *trees*.

    Raises
    ------
    ValueError
        If any group in *ordered_groups* is present in *newick_input* but
        has an empty list of NEWICK strings.
    """
    # ── 1. Flatten ─────────────────────────────────────────────────────────
    flat_groups: List[str] = []
    flat_newicks: List[str] = []

    for group_name in ordered_groups:
        newicks = newick_input[group_name]
        if not newicks:
            raise ValueError(f"Group '{group_name}' is empty")
        for newick in newicks:
            flat_groups.append(group_name)
            flat_newicks.append(newick)

    n_total = len(flat_newicks)

    # ── 2. Decide whether to use subprocesses ─────────────────────────────
    effective_workers = min(_resolve_n_threads(n_threads), n_total)
    total_chars = sum(len(nwk) for nwk in flat_newicks)
    use_mp = effective_workers > 1 and total_chars >= _MP_CHAR_THRESHOLD

    # ── 3. Parse sequentially or in parallel ─────────────────────────────
    if use_mp:
        logger.debug(
            "Parsing %d NEWICK strings with %d worker process(es) "
            "(%d chars total, threshold=%d)",
            n_total,
            effective_workers,
            total_chars,
            _MP_CHAR_THRESHOLD,
        )
        ctx = multiprocessing.get_context(_MP_CONTEXT)
        worker = functools.partial(_parse_newick_worker, polytomy_strategy=polytomy_strategy)
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            trees: List["Tree"] = list(pool.map(worker, flat_newicks))
    else:
        logger.debug(
            "Parsing %d NEWICK strings sequentially (%d chars total, threshold=%d)",
            n_total,
            total_chars,
            _MP_CHAR_THRESHOLD,
        )
        trees = [Tree(newick, polytomy_strategy=polytomy_strategy) for newick in flat_newicks]

    return trees, flat_groups


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

    >>> result = c.quartet_topology(quartets, steiner=True)
    >>> result.counts.shape   # (n_quartets, n_groups, 3)
    >>> result.to_frame()     # long-form Polars DataFrame
    """

    # ================================================================== #
    # Construction                                                         #
    # ================================================================== #

    def __init__(
        self,
        newick_input,
        taxon_map: Optional[Dict[str, str]] = None,
        polytomy_strategy: str = "multifurcation",
    ) -> None:
        """Initialize collection from NEWICK strings with optional group labels.

        Parameters
        ----------
        newick_input
            NEWICK strings in any supported form: a list of strings, a dict
            mapping group labels to lists of strings, a single multi-line
            string, or a file path.
        taxon_map : dict[str, str] | None
            Optional mapping from leaf name (as it appears in the NEWICK) to
            genome name.  Leaves that map to the same genome name form a
            paralog group (MUL-tree support).  When *None* (default),
            duplicate leaf names across all trees raise ``ValueError`` as
            before.
        polytomy_strategy : str
            How to treat zero-length branches and multifurcations.  Passed
            through to each :class:`Tree` constructor.

            ``'multifurcation'`` (default)
                Only true NEWICK multifurcations are treated as polytomies.
                User-provided zero-length branches are treated as real branches
                and trigger a warning.

            ``'zero-length-branch'``
                Zero-length internal branches in the NEWICK are treated as
                polytomies.  True multifurcations are resolved silently with a
                warning.

            ``'both'``
                Both forms are accepted as polytomies.  No warnings emitted.
        """
        _VALID_STRATEGIES = {"multifurcation", "zero-length-branch", "both"}
        if polytomy_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"polytomy_strategy must be one of {sorted(_VALID_STRATEGIES)!r}, "
                f"got {polytomy_strategy!r}"
            )
        self._polytomy_strategy = polytomy_strategy

        # Normalize all supported input forms to dict[str, list[str]].
        # Accepts: dict, list/tuple of NEWICK strings or paths, a single
        # multiline string, or a path (str or Path) to a NEWICK file.
        newick_input = normalize_input(newick_input)

        # Parse groups
        self.unique_groups = sorted(newick_input.keys())
        self.n_groups = len(self.unique_groups)

        # Log before parsing so the message appears while work is in progress
        _n_total = sum(len(v) for v in newick_input.values())
        logger.info("🌲 Loading %d tree(s) from NEWICK strings...", _n_total)
        if self.n_groups > 1:
            logger.info("  Organized into %d labeled groups", self.n_groups)
        else:
            logger.info("  Single group: %s", self.unique_groups[0])

        # Parse NEWICK strings into Tree objects (parallel when dataset is large)
        self._trees, self.group_labels = _load_tree_data(
            newick_input, self.unique_groups, polytomy_strategy=polytomy_strategy
        )
        self.n_trees = len(self._trees)

        # Apply taxon_map (paralog renaming) after parsing so the subprocess
        # worker does not need to serialise the map.
        self._taxon_map: Optional[Dict[str, str]] = taxon_map
        if taxon_map is not None:
            for t in self._trees:
                t._apply_taxon_map(taxon_map)

        # Deduplicate completely identical trees (same topology + branch lengths)
        # within each group.  Populates private _stored_trees, _n_stored_trees,
        # _stored_group_labels, _tree_multiplicities.  Public n_trees unchanged.
        self._deduplicate_tree_data()

        # Build group mappings
        self._build_group_mappings()

        # Build namespace and CSR layout
        logger.info("🔖 Building global taxon namespace...")
        self._build_global_namespace()
        self._build_paralog_csr()

        logger.info("📦 Packing arrays into CSR flat layout...")
        self._pack_csr()
        log_polytomy_statistics(self.polytomy_offsets, self._n_stored_trees)

        n_zero_trees = sum(1 for t in self._trees if t.n_zero_length_branches > 0)
        if n_zero_trees > 0:
            total_zero = sum(t.n_zero_length_branches for t in self._trees)
            log_zero_length_branch_warning(n_zero_trees, self.n_trees, total_zero)

        if polytomy_strategy == "zero-length-branch":
            n_multifurc_trees = sum(1 for t in self._trees if t.had_multifurcations)
            if n_multifurc_trees > 0:
                log_multifurcation_resolved_warning(n_multifurc_trees, self.n_trees)

        # Package all forest arrays into a single kernel-dispatch object
        self._build_kernel_data()

        # Upload tree structure to GPU once (if available)
        self._cuda_kernel_data: Optional[ForestKernelData] = None
        if backends.cuda:
            self._upload_to_gpu()

        # Log group statistics and namespace coverage
        self._log_group_statistics_method()
        self._log_namespace_coverage_method()

        # Log overall statistics
        self._log_statistics_method()

    def _build_group_mappings(self) -> None:
        """Build group → tree indices mappings (input-semantics + stored-semantics)."""
        group_to_idx = {g: i for i, g in enumerate(self.unique_groups)}

        # Public: input-tree semantics
        self.group_to_tree_indices = {
            group_name: np.array(
                [i for i, g in enumerate(self.group_labels) if g == group_name],
                dtype=np.int64,
            )
            for group_name in self.unique_groups
        }
        self.tree_to_group_idx = np.array(
            [group_to_idx[g] for g in self.group_labels], dtype=np.int32
        )

        # Private: stored-tree semantics (used by kernels and logging)
        self._stored_tree_to_group_idx = np.array(
            [group_to_idx[g] for g in self._stored_group_labels], dtype=np.int32
        )
        self._stored_group_to_tree_indices = {
            group_name: np.array(
                [i for i, g in enumerate(self._stored_group_labels) if g == group_name],
                dtype=np.int64,
            )
            for group_name in self.unique_groups
        }

    def _deduplicate_tree_data(self) -> None:
        """
        Identify duplicate trees (same topology and branch lengths within the
        same group) and build the stored-tree representation.

        Creates private attributes without modifying any public-facing attrs:
            ``self._stored_trees``        — unique Tree objects (first occurrences).
            ``self._stored_group_labels`` — group label per stored tree.
            ``self._n_stored_trees``      — number of unique stored trees.
            ``self._tree_multiplicities`` — int32[_n_stored_trees] weight per tree.

        Cross-group trees are never merged (they accumulate into different
        output slots).  The order of first occurrences is preserved.
        """
        seen: dict = {}          # (hash_bytes, group_label) -> sequential stored index
        keep_indices: list = []  # indices into self._trees / self.group_labels

        for i, (tree, grp) in enumerate(zip(self._trees, self.group_labels)):
            key = (tree.tree_hash(), grp)
            if key not in seen:
                seen[key] = len(keep_indices)
                keep_indices.append(i)

        n_dupes = self.n_trees - len(keep_indices)
        if n_dupes > 0:
            logger.info(
                "🔁 Deduplication: %d/%d trees are duplicates; storing %d unique trees.",
                n_dupes, self.n_trees, len(keep_indices),
            )

        # Compute multiplicities: count how many input trees map to each stored tree.
        mult: dict = {}  # stored sequential index -> count
        for tree, grp in zip(self._trees, self.group_labels):
            key = (tree.tree_hash(), grp)
            new_idx = seen[key]
            mult[new_idx] = mult.get(new_idx, 0) + 1

        self._stored_trees = [self._trees[i] for i in keep_indices]
        self._stored_group_labels = [self.group_labels[i] for i in keep_indices]
        self._n_stored_trees = len(self._stored_trees)
        self._tree_multiplicities = np.array(
            [mult[new_i] for new_i in range(len(keep_indices))], dtype=np.int32
        )

    def _build_kernel_data(self) -> None:
        """Package all forest arrays into a ``ForestKernelData`` for kernel dispatch."""
        self._kernel_data = ForestKernelData(
            global_to_local=self.global_to_local,
            all_first_occ=self.all_first_occ,
            all_root_distance=self.all_root_distance,
            all_euler_tour=self.all_euler_tour,
            all_euler_depth=self.all_euler_depth,
            all_sparse_table=self.all_sparse_table,
            all_log2_table=self.all_log2_table,
            node_offsets=self.node_offsets,
            tour_offsets=self.tour_offsets,
            sp_offsets=self.sp_offsets,
            lg_offsets=self.lg_offsets,
            sp_tour_widths=self.sp_tour_widths,
            tree_to_group_idx=self._stored_tree_to_group_idx,
            polytomy_offsets=self.polytomy_offsets,
            polytomy_nodes=self.polytomy_nodes,
            tree_multiplicities=self._tree_multiplicities,
            n_trees=self._n_stored_trees,
            n_global_taxa=self.n_global_taxa,
            n_groups=self.n_groups,
        )

    def _upload_to_gpu(self) -> None:
        """Upload all CSR tree-structure arrays to the GPU once at construction time."""
        logger.info(
            "📤 Uploading %d tree arrays (%.2f MB) to GPU...",
            len(self._kernel_data.device_arrays()),
            self._kernel_data.upload_bytes / (1024**2),
        )
        self._cuda_kernel_data = self._kernel_data.to_device()

    def __del__(self) -> None:
        """Free GPU device arrays when the Forest is garbage-collected."""
        if getattr(self, "_cuda_kernel_data", None) is None:
            return
        try:
            # Only attempt cleanup when a CUDA context is still live.
            # Importing cuda here avoids errors if numba was never initialised.
            from numba import cuda as _cuda

            if _cuda.is_available():
                for arr in self._cuda_kernel_data.device_arrays():
                    del arr
        except Exception:
            pass
        self._cuda_kernel_data = None

    def _log_group_statistics_method(self) -> None:
        """Log group membership statistics."""
        log_group_statistics(
            self.n_groups, self.unique_groups, self.group_to_tree_indices
        )

    def _log_namespace_coverage_method(self) -> None:
        """Log taxon namespace coverage within and between groups."""
        log_namespace_coverage(
            self.unique_groups,
            self._stored_group_to_tree_indices,
            self.taxa_present,
        )

    # ================================================================== #
    # Public methods                                                       #
    # ================================================================== #

    def branch_distance(self, taxon_a, taxon_b) -> BranchDistanceResult:
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

        distances_out = np.full(self._n_stored_trees, np.nan, dtype=np.float64)

        kd = self._kernel_data
        g2l = kd.global_to_local
        fo = kd.all_first_occ
        rd = kd.all_root_distance
        et = kd.all_euler_tour
        ed = kd.all_euler_depth
        sp = kd.all_sparse_table
        lg = kd.all_log2_table
        no = kd.node_offsets
        to_ = kd.tour_offsets
        so = kd.sp_offsets
        lo_ = kd.lg_offsets
        stw = kd.sp_tour_widths

        for ti in range(self._n_stored_trees):
            la = int(g2l[ti, ga])
            lb = int(g2l[ti, gb])
            if la < 0 or lb < 0:
                continue
            if la == lb:
                distances_out[ti] = 0.0
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

            distances_out[ti] = (
                float(rd[node_base + la])
                + float(rd[node_base + lb])
                - 2.0 * float(rd[node_base + lca_local])
            )

        # ── Suture: assemble result ────────────────────────────────────────────
        return BranchDistanceResult(distances=distances_out)

    def quartet_topology(
        self, quartets: Quartets, steiner: bool = False, backend: str = "best",
    ) -> QuartetTopologyResult:
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
        quartets : Quartets
            A Quartets object specifying which quartets to analyze.

            Use Quartets.from_list() for explicit quartet lists:
                >>> from quarimo._quartets import Quartets
                >>> q = Quartets.from_list(forest, [('A','B','C','D')])
                >>> counts = forest.quartet_topology(q)

            Use Quartets.random() for random sampling with GPU generation:
                >>> q = Quartets.random(forest, count=1_000_000, seed=42)
                >>> counts = forest.quartet_topology(q)  # 8-24× faster!

            See quarimo._quartets.Quartets for complete documentation.
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
        >>> from quarimo._quartets import Quartets

        Single quartet:

        >>> q = Quartets.from_list(forest, [('A', 'B', 'C', 'D')])
        >>> counts = forest.quartet_topology(q)
        >>> print(counts[0])          # topology distribution: (3,) view of row 0
        [10, 3, 7]

        >>> counts, dists = forest.quartet_topology(q, steiner=True)
        >>> print(dists[0, :, :].sum(axis=0))   # total Steiner per topology
        [34.1   0.   0. ]

        Multiple quartets:

        >>> quartet_list = [('A','B','C','D'), ('A','B','C','E'), ('B','C','D','E')]
        >>> q = Quartets.from_list(forest, quartet_list)
        >>> counts = forest.quartet_topology(q)
        >>> print(counts.shape)
        (3, 3)

        >>> counts, dists = forest.quartet_topology(q, steiner=True)
        >>> print(dists.shape)
        (3, 20, 3)          # n_quartets × n_trees × 3 topologies

        Random sampling (8-24× faster on GPU):

        >>> q = Quartets.random(forest, count=1_000_000, seed=42)
        >>> counts = forest.quartet_topology(q)
        >>> print(counts.shape)
        (1000000, 3)

        Backend selection:

        >>> q = Quartets.from_list(forest, quartet_list)
        >>> counts = forest.quartet_topology(q)                        # 'best' (default)
        >>> counts = forest.quartet_topology(q, backend='python')      # unoptimized
        >>> counts = forest.quartet_topology(q, backend='cpu-parallel') # numba CPU
        >>> counts = forest.quartet_topology(q, backend='cuda')        # GPU (if available)

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
        # ── 1. Validate input type ───────────────────────────────────────
        if not isinstance(quartets, Quartets):
            raise TypeError(
                f"quartets must be a Quartets object, got {type(quartets).__name__}. "
                f"Use Quartets.from_list(forest, your_quartets) or "
                f"Quartets.random(forest, count=N)."
            )

        n_quartets = len(quartets)
        if n_quartets == 0:
            empty = (0, self.n_groups, 4)
            return QuartetTopologyResult(
                counts=np.zeros(empty, dtype=np.int32),
                steiner=np.zeros(empty, dtype=np.float64) if steiner else None,
                steiner_min=np.full(empty, np.inf, dtype=np.float64) if steiner else None,
                steiner_max=np.full(empty, -np.inf, dtype=np.float64) if steiner else None,
                steiner_var=np.zeros(empty, dtype=np.float64) if steiner else None,
                groups=self.unique_groups,
                quartets=quartets,
                global_names=self.global_names,
            )

        # ── 2. Resolve backend and dispatch ──────────────────────────────
        # Context manager override takes precedence over caller's argument.
        effective = get_backend_override() or backend

        with use_kernel("quartet_topology", effective) as resolved_backend:
            mode_str = "Steiner" if steiner else "counts-only"
            logger.info(f"🧬 quartet_topology({mode_str}, backend={resolved_backend!r})")

            # ── 3. Dispatch ───────────────────────────────────────────────
            steiner_out = None
            steiner_min_out = None
            steiner_max_out = None
            steiner_sum_sq_out = None

            if resolved_backend == "cuda":
                counts_out, steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out = (
                    self._quartet_topology_cuda_unified(quartets, steiner)
                )

            elif resolved_backend == "mlx":
                from quarimo._mlx_kernels import quartet_counts_mlx, quartet_steiner_mlx

                _t_ql0 = perf_counter()
                sorted_ids = np.array(list(quartets), dtype=np.int32)
                _t_ql = perf_counter() - _t_ql0

                _t_c0 = perf_counter()
                if steiner:
                    counts_out, steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out = (
                        quartet_steiner_mlx(
                            self._kernel_data, sorted_ids, n_quartets, self.n_groups
                        )
                    )
                else:
                    counts_out = quartet_counts_mlx(
                        self._kernel_data, sorted_ids, n_quartets, self.n_groups
                    )
                _t_c = perf_counter() - _t_c0

                # On Apple Silicon UMA there is no host/device boundary; retrieve is ~0.
                logger.info(
                    "⏱ t_query_load=%.6f t_calc=%.6f t_retrieve=%.6f", _t_ql, _t_c, 0.0
                )

            else:
                # CPU/Python backends: materialise quartets to array
                _t_ql0 = perf_counter()
                sorted_ids = np.array(list(quartets), dtype=np.int32)
                common_args = self._kernel_data.cpu_common_args(sorted_ids, n_quartets)

                counts_out = np.zeros((n_quartets, self.n_groups, 4), dtype=np.int32)

                if steiner:
                    shape = (n_quartets, self.n_groups, 4)
                    steiner_out = np.zeros(shape, dtype=np.float64)
                    steiner_min_out = np.full(shape, np.inf, dtype=np.float64)
                    steiner_max_out = np.full(shape, -np.inf, dtype=np.float64)
                    steiner_sum_sq_out = np.zeros(shape, dtype=np.float64)
                _t_ql = perf_counter() - _t_ql0

                _t_c0 = perf_counter()
                if resolved_backend == "cpu-parallel":
                    kernel_key = f"cpu-parallel-{'steiner' if steiner else 'counts'}"
                    if _kernel_first_call.get(kernel_key, False):
                        logger.info(
                            f"  Compiling {kernel_key} kernel (cached for future calls)"
                        )
                        _kernel_first_call[kernel_key] = False

                    if steiner:
                        _quartet_steiner_njit(
                            *common_args, counts_out,
                            steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
                        )
                    else:
                        _quartet_counts_njit(*common_args, counts_out)

                else:  # python
                    if steiner:
                        steiner_arg = steiner_out
                        smin_arg = steiner_min_out
                        smax_arg = steiner_max_out
                        ssq_arg = steiner_sum_sq_out
                    else:
                        steiner_arg = np.empty(0, dtype=np.float64)  # counts-only sentinel
                        smin_arg = np.empty(0, dtype=np.float64)
                        smax_arg = np.empty(0, dtype=np.float64)
                        ssq_arg = np.empty(0, dtype=np.float64)
                    Forest._quartet_kernel(
                        *common_args, counts_out, steiner_arg, smin_arg, smax_arg, ssq_arg,
                    )
                _t_c = perf_counter() - _t_c0

                # CPU: results are already in host numpy arrays; no retrieve step.
                logger.info(
                    "⏱ t_query_load=%.6f t_calc=%.6f t_retrieve=%.6f", _t_ql, _t_c, 0.0
                )

        # ── Suture: assemble result ────────────────────────────────────────────
        # Compute variance from accumulated sum-of-squares.
        # var = sum_sq/n - (sum/n)^2; NaN for cells where count == 0.
        steiner_var_out = None
        if steiner_sum_sq_out is not None:
            mask = counts_out > 0
            denom = np.where(mask, counts_out, 1).astype(np.float64)
            steiner_var_out = np.where(
                mask,
                steiner_sum_sq_out / denom - (steiner_out / denom) ** 2,
                np.nan,
            )

        return QuartetTopologyResult(
            counts=counts_out,
            steiner=steiner_out,
            steiner_min=steiner_min_out,
            steiner_max=steiner_max_out,
            steiner_var=steiner_var_out,
            groups=self.unique_groups,
            quartets=quartets,
            global_names=self.global_names,
        )

    def resolve_paralogs(
        self,
        quartets,
        max_iter: int = 100,
        rng_seed=None,
        backend: str = "best",
    ):
        """
        Run the full paralog copy-slot optimisation pipeline.

        Executes the full quartet topology pass to get initial counts, then
        runs coordinate-descent optimisation via :class:`ParalogOptimizer`
        to find the copy-slot assignment that maximises mean QED.

        After this call ``self.global_to_local`` reflects the optimal
        assignment.  Run ``quartet_topology(quartets, steiner=True)`` again
        with the updated mapping if you need accurate Steiner statistics for
        the optimal assignment.

        Parameters
        ----------
        quartets : Quartets
            Quartet sequence for the analysis.
        max_iter : int
            Maximum number of coordinate-descent sweeps (default 100).
        rng_seed : optional
            Reserved for future stochastic tie-breaking; ignored for now.
        backend : str
            Computational backend for both the full pass and delta kernel.

        Returns
        -------
        (QuartetTopologyResult, OptimizationResult)
            The topology counts after optimisation (steiner=None) and the
            full optimisation record including assignment history and QED
            trajectory.

        Raises
        ------
        ValueError
            If the forest has no paralog genomes.
        """
        from quarimo._paralog import build_paralog_data as _bpd, ParalogOptimizer
        from quarimo._results import QuartetTopologyResult

        if not self.paralog_genome_names:
            raise ValueError(
                "Forest has no paralog genomes. "
                "Provide taxon_map with at least one genome appearing > 1 time."
            )

        # Full pass with the current (default) assignment
        initial = self.quartet_topology(quartets, backend=backend)

        # Build the inverted quartet index and paralog data
        pd = _bpd(self, quartets)

        # Run the optimiser
        optimizer = ParalogOptimizer(self, quartets, initial.counts, pd, backend=backend)
        opt_result = optimizer.optimize(max_iter=max_iter, rng_seed=rng_seed)

        # Wrap the final counts in a QuartetTopologyResult (no steiner)
        final_result = QuartetTopologyResult(
            counts=optimizer.counts,
            steiner=None,
            steiner_min=None,
            steiner_max=None,
            steiner_var=None,
            groups=initial.groups,
            quartets=quartets,
            global_names=self.global_names,
        )
        return final_result, opt_result

    def build_paralog_data(self, quartets):
        """
        Build a :class:`ParalogData` for this forest and *quartets*.

        Packages the forest's paralog arrays together with an inverted
        taxon → quartet index built by iterating over all quartets once.

        Parameters
        ----------
        quartets : Quartets
            Quartet sequence for which the inverted index is built.

        Returns
        -------
        ParalogData

        Raises
        ------
        ValueError
            If the forest has no paralog genomes (``taxon_map`` was not
            provided or produced no groups with > 1 copy).
        """
        from quarimo._paralog import build_paralog_data as _build

        if not self.paralog_genome_names:
            raise ValueError(
                "Forest has no paralog genomes. "
                "Provide taxon_map with at least one genome appearing > 1 time."
            )
        return _build(self, quartets)

    def qed(
        self,
        counts: Union[np.ndarray, QuartetTopologyResult],
        group_pairs: Optional[np.ndarray] = None,
    ) -> QEDResult:
        """
        Compute the Quartet Ensemble Discordance (QED) for all quartet × group-pair combinations.

        QED is an entropy-like similarity score in [-1, +1] computed from the
        per-group topology counts returned by :meth:`quartet_topology`.

        Score interpretation
        --------------------
        +1.0  The dominant topology is the same in both ensembles and receives
              all the signal — perfect concordance.
        0.0   No topology dominates, or one ensemble has no trees — no
              information.
        -1.0  Conflicting topologies dominate the two ensembles — perfect
              discordance.

        The score captures more than just whether the dominant topologies
        agree.  Because each quartet has exactly three possible topologies,
        QED is the lowest-order subtree measure that can distinguish signal
        from noise in the minority topologies:

        * If the same *two* topologies are supported in both ensembles (one
          dominant, one minority), QED tends toward a positive value in (0, 1),
          reflecting partial but concordant signal.
        * If *different* two topologies are each dominant in one ensemble, QED
          tends toward a negative value in (-1, 0), reflecting discordant
          signal.
        * If both non-dominant topologies are strongly represented in one or
          both ensembles, QED tends toward 0, reflecting noise rather than
          structured signal.

        Topology encoding
        -----------------
        Uses the same k=0,1,2 indices as :meth:`quartet_topology`.

        Parameters
        ----------
        counts : np.ndarray, int32, shape (n_quartets, n_groups, 3)
            Per-group topology count array.  Typically the return value of
            ``forest.quartet_topology(quartets)``.  Must have exactly 3
            topology slots (last axis size 3) and exactly ``self.n_groups``
            groups (middle axis).
        group_pairs : np.ndarray or None, optional
            int32 array of shape (n_pairs, 2).  Each row ``[g1, g2]`` is an
            ordered pair of group indices to compare.  Indices must satisfy
            ``0 <= g1, g2 < self.n_groups``.  Defaults to all unordered pairs
            ``{g1, g2}`` with ``g1 < g2``, enumerated in lexicographic order.
            For a forest with a single group the default is an empty array and
            the return value has shape ``(n_quartets, 0)``.

        Returns
        -------
        qed : np.ndarray, float64, shape (n_quartets, n_pairs)
            ``qed[qi, pi]`` is the QED score comparing groups
            ``group_pairs[pi, 0]`` and ``group_pairs[pi, 1]`` for quartet
            ``qi``.

        Raises
        ------
        ValueError
            If ``counts`` has the wrong shape or ``group_pairs`` contains
            out-of-range indices.

        Examples
        --------
        >>> q = Quartets.from_list(forest, [('A', 'B', 'C', 'D')])
        >>> counts = forest.quartet_topology(q)   # (1, n_groups, 3)
        >>> scores = forest.qed(counts)           # QEDResult, shape (1, n_pairs)
        >>> df = scores.to_frame('wide')          # join on ['a','b','c','d']
        """
        # Extract metadata before unwrapping
        _quartets = None
        _global_names = None
        if isinstance(counts, QuartetTopologyResult):
            _quartets = counts.quartets
            _global_names = counts.global_names
            counts = counts.counts
        counts = np.asarray(counts, dtype=np.int32)
        if counts.ndim != 3 or counts.shape[1] != self.n_groups or counts.shape[2] < 3:
            raise ValueError(
                f"counts must have shape (n_quartets, {self.n_groups}, 3+), "
                f"got {counts.shape}"
            )
        n_quartets = counts.shape[0]

        # Build default group pairs: all (g1, g2) with g1 < g2
        if group_pairs is None:
            pairs = [
                (g1, g2)
                for g1 in range(self.n_groups)
                for g2 in range(g1 + 1, self.n_groups)
            ]
            group_pairs = np.array(pairs, dtype=np.int32).reshape(-1, 2)
        else:
            group_pairs = np.asarray(group_pairs, dtype=np.int32)
            if group_pairs.ndim != 2 or group_pairs.shape[1] != 2:
                raise ValueError(
                    f"group_pairs must have shape (n_pairs, 2), got {group_pairs.shape}"
                )
            if group_pairs.size > 0:
                if group_pairs.min() < 0 or group_pairs.max() >= self.n_groups:
                    raise ValueError(
                        f"group_pairs indices must be in [0, {self.n_groups}), "
                        f"got range [{group_pairs.min()}, {group_pairs.max()}]"
                    )

        n_pairs = len(group_pairs)
        out = np.zeros((n_quartets, n_pairs), dtype=np.float64)

        if n_quartets == 0 or n_pairs == 0:
            return QEDResult(
                scores=out,
                groups=self.unique_groups,
                group_pairs=group_pairs,
                quartets=_quartets,
                global_names=_global_names,
            )

        if backends.numba:
            from quarimo._cpu_kernels import _qed_njit

            _qed_njit(counts, group_pairs, n_quartets, n_pairs, out)
        else:
            Forest._qed_kernel(counts, group_pairs, n_quartets, n_pairs, out)

        return QEDResult(
            scores=out,
            groups=self.unique_groups,
            group_pairs=group_pairs,
            quartets=_quartets,
            global_names=_global_names,
        )

    def _cuda_output_bytes_per_quartet(self, steiner: bool) -> int:
        """Bytes of GPU output array space required per quartet."""
        # counts: n_groups × 4 × int32 per quartet
        bpq = self.n_groups * 4 * 4
        if steiner:
            # steiner_out + steiner_min_out + steiner_max_out + steiner_sum_sq_out
            bpq += 4 * self.n_groups * 4 * 8
        return bpq

    def _cuda_batch_size(self, steiner: bool) -> int:
        """
        Maximum number of quartets whose output arrays fit in free GPU memory.

        Uses 80 % of currently-free device memory so the kernel, stack frames,
        and other runtime overhead have headroom.  Falls back to a conservative
        512 MB budget if the query fails.
        """
        try:
            free_bytes, _ = cuda.current_context().get_memory_info()
            available = int(free_bytes * 0.80)
        except Exception:
            available = 512 * 1024 * 1024  # 512 MB conservative fallback

        bpq = self._cuda_output_bytes_per_quartet(steiner)
        return max(1, available // bpq)

    def _quartet_topology_cuda_unified(
        self, quartets: Quartets, steiner: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        GPU implementation using unified kernel with on-GPU quartet generation.

        Processes quartets in batches whose output arrays fit in free GPU
        memory, so arbitrarily large random-sampling runs never OOM the device.
        The tree-structure arrays were uploaded once in the constructor and
        remain resident; only the tiny seed-quartet array is transferred per
        call.

        Parameters
        ----------
        quartets : Quartets
            Quartet specification with seed and generation parameters.
        steiner : bool
            Whether to compute Steiner distances.

        Returns
        -------
        counts : ndarray, int32, shape (n_quartets, n_groups, 4)
        steiner : ndarray or None, float64, shape (n_quartets, n_groups, 4)
        steiner_min : ndarray or None, float64, shape (n_quartets, n_groups, 4)
        steiner_max : ndarray or None, float64, shape (n_quartets, n_groups, 4)
        """
        from quarimo._cuda_kernels import (
            _compute_cuda_grid,
            generate_quartets_cuda,
            quartet_counts_cuda_unified,
            quartet_steiner_cuda_unified,
        )

        n_quartets = len(quartets)

        # Track first-call compilation
        kernel_key = f"cuda-unified-{'steiner' if steiner else 'counts'}"
        if _kernel_first_call.get(kernel_key, False):
            logger.info("  🔨 Compiling %s kernel (cached for future calls)", kernel_key)
            _kernel_first_call[kernel_key] = False

        # ── Seed array (tiny per-call upload) ─────────────────────────────
        _t_ql0 = perf_counter()
        seed_array = np.array(quartets.seed, dtype=np.int32)
        d_seed_quartets = cuda.to_device(seed_array)
        logger.info(
            "  🌱 Quartet seed: %s %s, %d bytes → generates %d quartets on GPU",
            seed_array.shape,
            seed_array.dtype,
            seed_array.nbytes,
            n_quartets,
        )

        # ── Pre-uploaded tree arrays ───────────────────────────────────────
        d_fkd = self._cuda_kernel_data
        d_forest_args = d_fkd.cuda_forest_args()

        # ── Batch size: how many quartets' output fits in free VRAM ───────
        batch_size = self._cuda_batch_size(steiner)
        n_batches = (n_quartets + batch_size - 1) // batch_size

        bpq = self._cuda_output_bytes_per_quartet(steiner)
        logger.info(
            "  📊 Output: %.2f MB total, batch_size=%d (%d batch%s)",
            n_quartets * bpq / (1024**2),
            batch_size,
            n_batches,
            "es" if n_batches != 1 else "",
        )

        # ── Host accumulation arrays ───────────────────────────────────────
        shape = (n_quartets, self.n_groups, 4)
        counts_out = np.zeros(shape, dtype=np.int32)
        steiner_out: Optional[np.ndarray] = None
        steiner_min_out: Optional[np.ndarray] = None
        steiner_max_out: Optional[np.ndarray] = None
        steiner_sum_sq_out: Optional[np.ndarray] = None
        if steiner:
            steiner_out = np.zeros(shape, dtype=np.float64)
            steiner_min_out = np.full(shape, np.inf, dtype=np.float64)
            steiner_max_out = np.full(shape, -np.inf, dtype=np.float64)
            steiner_sum_sq_out = np.zeros(shape, dtype=np.float64)

        # ── Batched kernel dispatch ────────────────────────────────────────
        _t_ql = perf_counter() - _t_ql0
        _t_c_total = 0.0
        _t_r_total = 0.0
        q_args = QuartetKernelArgs.from_quartets(quartets)

        for batch_idx in range(n_batches):
            bs = batch_idx * batch_size
            bc = min(batch_size, n_quartets - bs)
            batch_offset = quartets.offset + bs

            blocks, tpb = _compute_cuda_grid(bc, self.n_trees)
            logger.info(
                "  Batch %d/%d: %d quartets, %s blocks × %s threads/block",
                batch_idx + 1,
                n_batches,
                bc,
                blocks,
                tpb,
            )

            # ── Quartet source for the 2D processing kernel ────────────────
            # If the batch contains any randomly-generated quartets, run a
            # cheap 1D kernel first to materialise all bc quartets into a
            # scratch array.  The 2D kernel then reads from that scratch array
            # (n_seed=bc, offset=0) and takes the fast global-read path for
            # every thread — identical to the deterministic case.
            # Without this step, every (qi, ti) thread would independently run
            # XorShift128 for the same qi, causing n_trees-fold RNG duplication.
            batch_needs_rng = batch_offset + bc > q_args.n_seed
            if batch_needs_rng:
                d_quartet_batch = cuda.device_array((bc, 4), dtype=np.int32)
                gen_blocks = (bc + 255) // 256
                generate_quartets_cuda[gen_blocks, 256](
                    d_seed_quartets,
                    q_args.n_seed,
                    batch_offset,
                    bc,
                    q_args.rng_seed,
                    d_fkd.n_global_taxa,
                    d_quartet_batch,
                )
                proc_seed = d_quartet_batch
                proc_n_seed = bc
                proc_offset = 0
            else:
                # Explicit seed quartets — served directly from the seed array.
                proc_seed = d_seed_quartets
                proc_n_seed = q_args.n_seed
                proc_offset = batch_offset

            batch_args = q_args.cuda_batch_args(proc_seed, proc_offset, bc,
                                                  n_seed=proc_n_seed)

            # Zero-initialised counts and steiner (atomic.add requires it)
            d_counts_b = cuda.to_device(
                np.zeros((bc, self.n_groups, 4), dtype=np.int32)
            )

            if steiner:
                # Pre-initialise steiner_min to +inf, steiner_max to -inf so that
                # cuda.atomic.min / cuda.atomic.max work correctly.
                d_steiner_b = cuda.to_device(
                    np.zeros((bc, self.n_groups, 4), dtype=np.float64)
                )
                d_steiner_min_b = cuda.to_device(
                    np.full((bc, self.n_groups, 4), np.inf, dtype=np.float64)
                )
                d_steiner_max_b = cuda.to_device(
                    np.full((bc, self.n_groups, 4), -np.inf, dtype=np.float64)
                )
                d_steiner_sum_sq_b = cuda.to_device(
                    np.zeros((bc, self.n_groups, 4), dtype=np.float64)
                )
                logger.info("  🧮 launching Steiner kernel...")
                _t_c0b = perf_counter()
                quartet_steiner_cuda_unified[blocks, tpb](
                    *batch_args, d_fkd.n_global_taxa, *d_forest_args,
                    d_counts_b, d_steiner_b, d_steiner_min_b, d_steiner_max_b,
                    d_steiner_sum_sq_b,
                )
                cuda.synchronize()
                _t_cb = perf_counter() - _t_c0b
                _t_c_total += _t_cb
                logger.info(
                    "  💾 computation completed in %.3f seconds, moving data...",
                    _t_cb,
                )
                # copy_to_host(ary=...) writes directly into the pre-allocated output
                # slice — one GPU→CPU transfer, no temporary array, no extra CPU memcpy.
                _t_r0b = perf_counter()
                d_counts_b.copy_to_host(ary=counts_out[bs : bs + bc])
                d_steiner_b.copy_to_host(ary=steiner_out[bs : bs + bc])
                d_steiner_min_b.copy_to_host(ary=steiner_min_out[bs : bs + bc])
                d_steiner_max_b.copy_to_host(ary=steiner_max_out[bs : bs + bc])
                d_steiner_sum_sq_b.copy_to_host(ary=steiner_sum_sq_out[bs : bs + bc])
                _t_r_total += perf_counter() - _t_r0b
                del d_steiner_b, d_steiner_min_b, d_steiner_max_b, d_steiner_sum_sq_b
            else:
                logger.info("  🧮 launching counts kernel...")
                _t_c0b = perf_counter()
                quartet_counts_cuda_unified[blocks, tpb](
                    *batch_args, d_fkd.n_global_taxa, *d_forest_args,
                    d_counts_b,
                )
                cuda.synchronize()
                _t_cb = perf_counter() - _t_c0b
                _t_c_total += _t_cb
                logger.info(
                    "  💾 computation completed in %.3f seconds, moving data...",
                    _t_cb,
                )
                _t_r0b = perf_counter()
                d_counts_b.copy_to_host(ary=counts_out[bs : bs + bc])
                _t_r_total += perf_counter() - _t_r0b

            if batch_needs_rng:
                del d_quartet_batch
            del d_counts_b

        total_bytes = counts_out.nbytes
        if steiner_out is not None:
            total_bytes += (
                steiner_out.nbytes + steiner_min_out.nbytes
                + steiner_max_out.nbytes + steiner_sum_sq_out.nbytes
            )
        logger.info("  ✅ D→H total: %.2f MB", total_bytes / (1024**2))
        logger.info(
            "⏱ t_query_load=%.6f t_calc=%.6f t_retrieve=%.6f",
            _t_ql, _t_c_total, _t_r_total,
        )

        return counts_out, steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out

    # ================================================================== #
    # Private instance methods                                             #
    # ================================================================== #

    def _build_global_namespace(self) -> None:
        """
        **Private.**  Collect all unique taxon names across all trees, assign
        sorted global IDs, and build the global↔local mapping arrays.

        When a ``taxon_map`` was provided at construction time, leaves with the
        same genome name form paralog groups.  Each genome that appears with
        k > 1 copies in any tree receives k global IDs named
        ``"{genome}_copy0"`` … ``"{genome}_copy{k-1}"``.  Genomes that appear
        at most once across all trees receive a single global ID (plain name).

        Without ``taxon_map``, duplicate leaf names across any tree raise
        ``ValueError`` (existing behaviour).
        """
        NT = self._n_stored_trees

        # ---- Count max copies per genome name across all trees ----------- #
        # For non-paralog forests this is trivially 1 everywhere.
        name_max_copies: Dict[str, int] = {}
        for t in self._stored_trees:
            # Count occurrences of each name within this tree
            tree_counts: Dict[str, int] = {}
            for i in range(t.n_leaves):
                name = t.names[i]
                if name:
                    tree_counts[name] = tree_counts.get(name, 0) + 1
            for name, cnt in tree_counts.items():
                if name_max_copies.get(name, 0) < cnt:
                    name_max_copies[name] = cnt

        # ---- Guard: duplicates without taxon_map are an error ------------ #
        if self._taxon_map is None:
            for name, copies in name_max_copies.items():
                if copies > 1:
                    raise ValueError(
                        f"Duplicate taxon name '{name}' found in a tree. "
                        "Pass taxon_map to enable MUL-tree (paralog) support."
                    )

        # ---- Separate singleton vs paralog genomes ----------------------- #
        # paralog = any genome that appears >1 time in at least one tree.
        paralog_genome_names = sorted(n for n, k in name_max_copies.items() if k > 1)
        singleton_names = sorted(n for n, k in name_max_copies.items() if k == 1)
        self.paralog_genome_names: List[str] = paralog_genome_names
        n_paralog_genomes = len(paralog_genome_names)

        # ---- Build global_names ------------------------------------------ #
        # Layout: all singleton names (sorted) first, then copy slots for
        # each paralog genome in sorted order.
        global_names: List[str] = list(singleton_names)
        paralog_genome_to_first_gid: Dict[str, int] = {}
        for genome in paralog_genome_names:
            k = name_max_copies[genome]
            paralog_genome_to_first_gid[genome] = len(global_names)
            for ci in range(k):
                global_names.append(f"{genome}_copy{ci}")

        self.global_names = global_names
        self.n_global_taxa = len(global_names)
        self._name_to_global: Dict[str, int] = {n: i for i, n in enumerate(global_names)}

        G = self.n_global_taxa

        # ---- Leaf offset array ------------------------------------------- #
        leaf_sizes = np.array([t.n_leaves for t in self._stored_trees], dtype=np.int64)
        self.leaf_offsets = np.zeros(NT + 1, dtype=np.int64)
        self.leaf_offsets[1:] = np.cumsum(leaf_sizes)
        total_leaves = int(self.leaf_offsets[-1])

        # ---- Dense global→local map and flat local→global map ------------ #
        self.global_to_local = np.full((NT, G), -1, dtype=np.int32)
        self.local_to_global = np.full(total_leaves, -1, dtype=np.int32)

        for ti, t in enumerate(self._stored_trees):
            lo = int(self.leaf_offsets[ti])

            if not paralog_genome_names:
                # Fast path: no paralogs — use name index directly.
                if t._name_index is None:
                    t._build_name_index()
                for name, local_id in t._name_index.items():
                    if name in self._name_to_global:
                        gid = self._name_to_global[name]
                        self.global_to_local[ti, gid] = local_id
                        self.local_to_global[lo + local_id] = gid
            else:
                # Paralog path: iterate leaves directly.
                # For singleton names, assign normally.
                # For paralog genomes, group leaves by genome name then
                # assign copy slots in ascending leaf-ID order.
                genome_leaf_lists: Dict[str, List[int]] = {}
                for leaf_id in range(t.n_leaves):
                    name = t.names[leaf_id]
                    if not name:
                        continue
                    if name in paralog_genome_to_first_gid:
                        genome_leaf_lists.setdefault(name, []).append(leaf_id)
                    elif name in self._name_to_global:
                        gid = self._name_to_global[name]
                        self.global_to_local[ti, gid] = leaf_id
                        self.local_to_global[lo + leaf_id] = gid

                for genome, leaf_ids in genome_leaf_lists.items():
                    first_gid = paralog_genome_to_first_gid[genome]
                    for ci, leaf_id in enumerate(sorted(leaf_ids)):
                        gid = first_gid + ci
                        self.global_to_local[ti, gid] = leaf_id
                        self.local_to_global[lo + leaf_id] = gid

        # ---- Paralog CSR arrays ------------------------------------------ #
        # paralog_copy_offsets[li] .. [li+1] gives the global IDs for genome li.
        if n_paralog_genomes > 0:
            copy_counts = np.array(
                [name_max_copies[g] for g in paralog_genome_names], dtype=np.int32
            )
            self.paralog_copy_offsets = np.zeros(n_paralog_genomes + 1, dtype=np.int32)
            self.paralog_copy_offsets[1:] = np.cumsum(copy_counts)
            total_copies = int(self.paralog_copy_offsets[-1])
            self.paralog_copy_global_ids = np.empty(total_copies, dtype=np.int32)
            for li, genome in enumerate(paralog_genome_names):
                first_gid = paralog_genome_to_first_gid[genome]
                k = name_max_copies[genome]
                start = int(self.paralog_copy_offsets[li])
                self.paralog_copy_global_ids[start : start + k] = np.arange(
                    first_gid, first_gid + k, dtype=np.int32
                )
        else:
            self.paralog_copy_offsets = np.zeros(1, dtype=np.int32)
            self.paralog_copy_global_ids = np.empty(0, dtype=np.int32)

        # Convenience presence mask
        self.taxa_present = self.global_to_local >= 0

    def _build_paralog_csr(self) -> None:
        """
        **Private.**  Build per-genome per-tree leaf CSR tables and the
        copy-slot assignment table.  Must be called after
        ``_build_global_namespace`` (requires ``global_to_local``,
        ``paralog_genome_names``, ``paralog_copy_offsets``,
        ``paralog_copy_global_ids``).

        Builds
        ------
        paralog_leaf_offsets : int32 [n_paralog_genomes * (n_trees + 1)]
            Packed CSR: for genome ``li``, the sub-array
            ``[li*(n_trees+1) : (li+1)*(n_trees+1)]`` is a standard CSR
            offset vector (length n_trees+1) mapping each tree index to a
            range in ``paralog_leaf_nodes``.

        paralog_leaf_nodes : int32 [total_paralog_leaves]
            Local leaf node IDs for all (genome, tree) combinations,
            in the order encoded by ``paralog_leaf_offsets``.

        paralog_assignments : int32 [n_paralog_genomes, max_copies, n_trees]
            ``paralog_assignments[li, ci, ti]`` = local leaf node ID assigned
            to copy slot ``ci`` of genome ``li`` in tree ``ti``, or ``-1`` if
            that genome has fewer than ``max_copies`` copies in tree ``ti``.
        """
        n_paralog = len(self.paralog_genome_names)
        NT = self._n_stored_trees

        if n_paralog == 0:
            # No paralogs — build empty sentinel arrays so downstream code
            # can always reference these attributes without guarding.
            self.paralog_leaf_offsets = np.zeros(0, dtype=np.int32)
            self.paralog_leaf_nodes = np.empty(0, dtype=np.int32)
            self.paralog_assignments = np.empty((0, 0, NT), dtype=np.int32)
            return

        # ---- Determine max copies across all paralog genomes ------------ #
        copy_counts = np.diff(self.paralog_copy_offsets)  # length n_paralog
        max_copies = int(copy_counts.max())

        # ---- Build leaf CSR + assignment table in one pass -------------- #
        # We accumulate (genome, tree) → list[leaf_id] before flattening.

        # paralog_leaf_offsets: shape (n_paralog * (NT + 1),)
        # We fill it incrementally as we count leaves per (genome, tree).
        leaf_counts = np.zeros((n_paralog, NT), dtype=np.int32)

        # assignment table — initialise to -1 (absent)
        assignments = np.full((n_paralog, max_copies, NT), -1, dtype=np.int32)

        for li in range(n_paralog):
            first_gid = int(self.paralog_copy_offsets[li])
            k = int(copy_counts[li])  # actual copy count for this genome
            for ti in range(NT):
                cnt = 0
                for ci in range(k):
                    gid = int(self.paralog_copy_global_ids[first_gid + ci])
                    local_id = int(self.global_to_local[ti, gid])
                    if local_id >= 0:
                        assignments[li, ci, ti] = local_id
                        cnt += 1
                leaf_counts[li, ti] = cnt

        # Build paralog_leaf_offsets CSR.
        # Offsets are *global* into paralog_leaf_nodes — each genome's sub-array
        # continues from where the previous genome ended.
        offsets = np.zeros(n_paralog * (NT + 1), dtype=np.int32)
        running = 0
        for li in range(n_paralog):
            base = li * (NT + 1)
            offsets[base] = running
            for ti in range(NT):
                offsets[base + ti + 1] = offsets[base + ti] + leaf_counts[li, ti]
            running = int(offsets[base + NT])
        total_leaves = running

        # Build paralog_leaf_nodes (flat leaf IDs in CSR order)
        leaf_nodes = np.full(total_leaves, -1, dtype=np.int32)
        for li in range(n_paralog):
            base = li * (NT + 1)
            first_gid = int(self.paralog_copy_offsets[li])
            k = int(copy_counts[li])
            for ti in range(NT):
                pos = int(offsets[base + ti])
                for ci in range(k):
                    gid = int(self.paralog_copy_global_ids[first_gid + ci])
                    local_id = int(self.global_to_local[ti, gid])
                    if local_id >= 0:
                        leaf_nodes[pos] = local_id
                        pos += 1

        self.paralog_leaf_offsets = offsets
        self.paralog_leaf_nodes = leaf_nodes
        self.paralog_assignments = assignments

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
        trees = self._stored_trees
        NT = self._n_stored_trees

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

        # ---- Polytomy CSR arrays ------------------------------------- #
        # polytomy_offsets[ti+1] - polytomy_offsets[ti] = number of polytomy-
        # inserted internal nodes in tree ti.  polytomy_nodes holds their
        # local node IDs.  Both arrays are pre-uploaded to GPU in _upload_to_gpu.
        poly_sizes = np.array([len(t.polytomy_node_ids) for t in trees], dtype=np.int64)
        self.polytomy_offsets = np.zeros(NT + 1, dtype=np.int32)
        self.polytomy_offsets[1:] = np.cumsum(poly_sizes).astype(np.int32)
        total_poly = int(self.polytomy_offsets[-1])
        if total_poly > 0:
            self.polytomy_nodes = np.concatenate(
                [np.array(t.polytomy_node_ids, dtype=np.int32) for t in trees]
            )
        else:
            self.polytomy_nodes = np.empty(0, dtype=np.int32)

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
        tree_to_group_idx: np.ndarray,
        polytomy_offsets: np.ndarray,
        polytomy_nodes: np.ndarray,
        tree_multiplicities: np.ndarray,
        counts_out: np.ndarray,
        steiner_out: np.ndarray,
        steiner_min_out: np.ndarray,
        steiner_max_out: np.ndarray,
        steiner_sum_sq_out: np.ndarray,
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
        tree_to_group_idx : int32 (n_trees,)
            Maps each tree index to its group index.
        counts_out : int32 (n_quartets, n_groups, 4), pre-filled with zeros
            counts_out[qi, gi, k] = number of trees in group gi where quartet
            qi has topology k.  k=3 accumulates unresolved (polytomy) counts.
        steiner_out : float64 array, pre-filled with zeros
            If steiner_out.size == 0 (sentinel: pass np.empty(0)):
                Steiner calculation is skipped entirely.
            Otherwise, shape (n_quartets, n_groups, 4).
                steiner_out[qi, gi, k] = summed Steiner lengths.
        steiner_min_out : float64 (n_quartets, n_groups, 4), pre-filled +inf
            Per-cell minimum Steiner length.  Meaningful only when
            steiner_out.size > 0.
        steiner_max_out : float64 (n_quartets, n_groups, 4), pre-filled -inf
            Per-cell maximum Steiner length.  Meaningful only when
            steiner_out.size > 0.
        steiner_sum_sq_out : float64 (n_quartets, n_groups, 4), pre-filled 0
            Per-cell sum of squared Steiner lengths.  Used post-kernel to
            compute variance = sum_sq/n - (sum/n)^2.  Meaningful only when
            steiner_out.size > 0.

        Steiner bypass
        --------------
        The sentinel ``steiner_out = np.empty(0, dtype=np.float64)`` signals
        counts-only mode.  All Steiner updates are skipped with a single
        ``if compute_steiner`` guard; no branching overhead in the topology core.
        """
        compute_steiner = steiner_out.size > 0

        for qi in range(n_quartets):
            t0 = int(sorted_quartet_ids[qi, 0])
            t1 = int(sorted_quartet_ids[qi, 1])
            t2 = int(sorted_quartet_ids[qi, 2])
            t3 = int(sorted_quartet_ids[qi, 3])

            for ti in range(n_trees):
                ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
                    Forest._resolve_quartet(
                        t0,
                        t1,
                        t2,
                        t3,
                        ti,
                        global_to_local,
                        node_offsets,
                        tour_offsets,
                        sp_offsets,
                        lg_offsets,
                        sp_tour_widths,
                    )
                if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                    continue

                fo0 = int(all_first_occ[node_base + ln0])
                fo1 = int(all_first_occ[node_base + ln1])
                fo2 = int(all_first_occ[node_base + ln2])
                fo3 = int(all_first_occ[node_base + ln3])
                poly_start = int(polytomy_offsets[ti])
                poly_end = int(polytomy_offsets[ti + 1])
                topo, r0, r1, r2, r_winner = Forest._quartet_topology_and_rd(
                    fo0,
                    fo1,
                    fo2,
                    fo3,
                    node_base,
                    tour_base,
                    sp_base,
                    lg_base,
                    sp_stride,
                    all_root_distance,
                    all_sparse_table,
                    all_euler_depth,
                    all_log2_table,
                    all_euler_tour,
                    poly_start,
                    poly_end,
                    polytomy_nodes,
                )

                gi = int(tree_to_group_idx[ti])
                mult = int(tree_multiplicities[ti])
                counts_out[qi, gi, topo] += mult

                if compute_steiner:
                    sl = Forest._steiner_length(
                        ln0, ln1, ln2, ln3, node_base, r0, r1, r2, r_winner, all_root_distance,
                    )
                    Forest._accumulate_steiner(
                        qi, gi, topo, sl, mult,
                        steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
                    )

    @staticmethod
    def _qed_kernel(
        counts: np.ndarray,
        pair_indices: np.ndarray,
        n_quartets: int,
        n_pairs: int,
        out: np.ndarray,
    ) -> None:
        """
        **Private static.**  Pure-Python fallback for the QED kernel.

        Mirrors ``_qed_njit`` exactly; used when Numba is unavailable.
        """
        import math as _math

        LOG3_INV = 0.9102392266268374
        for qi in range(n_quartets):
            for pi in range(n_pairs):
                g1 = int(pair_indices[pi, 0])
                g2 = int(pair_indices[pi, 1])

                ca0 = int(counts[qi, g1, 0])
                ca1 = int(counts[qi, g1, 1])
                ca2 = int(counts[qi, g1, 2])
                cb0 = int(counts[qi, g2, 0])
                cb1 = int(counts[qi, g2, 1])
                cb2 = int(counts[qi, g2, 2])

                Na = ca0 + ca1 + ca2
                Nb = cb0 + cb1 + cb2

                if Na == 0 or Nb == 0:
                    out[qi, pi] = 0.0
                    continue

                pa0, pa1, pa2 = ca0 / Na, ca1 / Na, ca2 / Na
                pb0, pb1, pb2 = cb0 / Nb, cb1 / Nb, cb2 / Nb

                if pa0 >= pa1 and pa0 >= pa2:
                    argmax_a = 0
                elif pa1 >= pa2:
                    argmax_a = 1
                else:
                    argmax_a = 2

                if pb0 >= pb1 and pb0 >= pb2:
                    argmax_b = 0
                elif pb1 >= pb2:
                    argmax_b = 1
                else:
                    argmax_b = 2

                J = 1.0 if argmax_a == argmax_b else -1.0

                q0, q1, q2 = pa0 * pb0, pa1 * pb1, pa2 * pb2
                s = 0.0
                if q0 > 0.0:
                    s += q0 * _math.log(q0) * LOG3_INV
                if q1 > 0.0:
                    s += q1 * _math.log(q1) * LOG3_INV
                if q2 > 0.0:
                    s += q2 * _math.log(q2) * LOG3_INV

                out[qi, pi] = J * (1.0 + s)

    @staticmethod
    def _resolve_quartet(
        t0,
        t1,
        t2,
        t3,
        ti,
        global_to_local,
        node_offsets,
        tour_offsets,
        sp_offsets,
        lg_offsets,
        sp_tour_widths,
    ):
        """
        **Private static.**  Map four global taxon IDs to tree-local positions
        for tree *ti*.

        Returns a 9-tuple
        ``(ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride)``
        where:

        ln0..ln3 : int
            Local leaf IDs in tree *ti* (-1 if the taxon is absent).
        node_base, tour_base, sp_base, lg_base : int
            Node / tour / sparse-table / log2-table CSR offsets for tree *ti*.
        sp_stride : int
            Sparse-table column stride for tree *ti*.

        The caller is responsible for checking::

            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue  # skip this (qi, ti) pair

        before accessing ``all_first_occ[node_base + ln0]`` etc.

        Pure-Python counterpart of ``_resolve_quartet_nb`` and
        ``_resolve_quartet_cuda``.
        """
        ln0       = int(global_to_local[ti, t0])
        ln1       = int(global_to_local[ti, t1])
        ln2       = int(global_to_local[ti, t2])
        ln3       = int(global_to_local[ti, t3])
        node_base = int(node_offsets[ti])
        tour_base = int(tour_offsets[ti])
        sp_base   = int(sp_offsets[ti])
        lg_base   = int(lg_offsets[ti])
        sp_stride = int(sp_tour_widths[ti])
        return ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride

    @staticmethod
    def _quartet_topology_and_rd(
        fo0,
        fo1,
        fo2,
        fo3,
        node_base,
        tour_base,
        sp_base,
        lg_base,
        sp_stride,
        all_root_distance,
        all_sparse_table,
        all_euler_depth,
        all_log2_table,
        all_euler_tour,
        poly_start,
        poly_end,
        polytomy_nodes,
    ):
        """
        **Private static.**  Six RMQ calls + four-point condition → topology
        and pair-sums.

        Computes all six pairwise LCA root-distances for four taxa (whose first
        Euler-tour occurrences are ``fo0..fo3``), applies the four-point
        condition to determine the winning unrooted split, and returns the
        topology index alongside all three pair-sums and the winning pair-sum.

        Parameters
        ----------
        fo0..fo3 : int
            First Euler-tour occurrences for the four taxa in tree *ti*.
            All must be valid — the caller has already checked ``ln0..ln3 >= 0``
            before computing these.
        node_base, tour_base, sp_base, lg_base, sp_stride : int
            CSR offsets and sparse-table stride for tree *ti*
            (from ``Forest._resolve_quartet``).
        all_root_distance, all_sparse_table, all_euler_depth,
        all_log2_table, all_euler_tour : np.ndarray
            CSR-packed tree arrays.

        Returns
        -------
        topo : int
            Winning topology index: 0 = (t0,t1)|(t2,t3),
                                    1 = (t0,t2)|(t1,t3),
                                    2 = (t0,t3)|(t1,t2).
        r0, r1, r2 : float
            Pair-sums for each of the three topologies.
        r_winner : float
            Score of the winning topology (max of r0, r1, r2).

        Pure-Python counterpart of ``_quartet_topology_and_rd_nb`` and
        ``_quartet_topology_and_rd_cuda``.
        """

        def lca_id(oa, ob):
            l, r = (oa, ob) if oa <= ob else (ob, oa)
            return Forest._rmq_csr(
                l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_base, tour_base, all_euler_tour,
            )

        lid01 = lca_id(fo0, fo1)
        lid02 = lca_id(fo0, fo2)
        lid03 = lca_id(fo0, fo3)
        lid12 = lca_id(fo1, fo2)
        lid13 = lca_id(fo1, fo3)
        lid23 = lca_id(fo2, fo3)

        rd01 = float(all_root_distance[node_base + lid01])
        rd02 = float(all_root_distance[node_base + lid02])
        rd03 = float(all_root_distance[node_base + lid03])
        rd12 = float(all_root_distance[node_base + lid12])
        rd13 = float(all_root_distance[node_base + lid13])
        rd23 = float(all_root_distance[node_base + lid23])

        r0 = rd01 + rd23  # topology 0: (t0,t1)|(t2,t3)
        r1 = rd02 + rd13  # topology 1: (t0,t2)|(t1,t3)
        r2 = rd03 + rd12  # topology 2: (t0,t3)|(t1,t2)

        # CSR-based polytomy detection (zero overhead for trees without polytomies).
        # CSR is the pre-filter; numerical tie r0==r1==r2 confirms an unresolvable
        # quartet.  A polytomy-inserted node can be an LCA for many resolved
        # quartets — only the all-equal tie means the quartet truly spans the
        # polytomy and is unresolvable (k=3).
        if poly_end > poly_start:
            for j in range(poly_start, poly_end):
                pn = int(polytomy_nodes[j])
                if pn in (lid01, lid02, lid03, lid12, lid13, lid23):
                    if r0 == r1 and r1 == r2:
                        return 3, r0, r1, r2, r0
                    break  # polytomy node is LCA but quartet is resolved; fall through

        if r0 >= r1 and r0 >= r2:
            topo = 0
            r_winner = r0
        elif r1 >= r0 and r1 >= r2:
            topo = 1
            r_winner = r1
        else:
            topo = 2
            r_winner = r2

        return topo, r0, r1, r2, r_winner

    @staticmethod
    def _steiner_length(
        ln0, ln1, ln2, ln3, node_base, r0, r1, r2, r_winner, all_root_distance
    ):
        """
        **Private static.**  Steiner spanning length of the winning quartet
        topology.

        Given the four local leaf IDs and the three pair-sums already computed
        by ``Forest._quartet_topology_and_rd``, returns the Steiner spanning
        length of the minimal subtree connecting the four taxa.

        Parameters
        ----------
        ln0..ln3 : int
            Local leaf IDs in tree *ti* (all must be >= 0).
        node_base : int
            Node-array CSR offset for tree *ti*.
        r0, r1, r2 : float
            Pair-sums for the three topologies.
        r_winner : float
            Score of the winning topology (max of r0, r1, r2).
        all_root_distance : np.ndarray
            CSR-packed root distances.

        Returns
        -------
        float
            Steiner spanning length S >= 0.

        Notes
        -----
        Formula: S = Σ rd(leaf_i) − 0.5 * (r_winner + r0 + r1 + r2)

        Pure-Python counterpart of ``_steiner_length_nb`` and
        ``_steiner_length_cuda``.
        """
        leaf_sum = (
            float(all_root_distance[node_base + ln0])
            + float(all_root_distance[node_base + ln1])
            + float(all_root_distance[node_base + ln2])
            + float(all_root_distance[node_base + ln3])
        )
        return leaf_sum - (r_winner + r0 + r1 + r2) * 0.5

    @staticmethod
    def _accumulate_steiner(
        qi, gi, topo, sl, mult,
        steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
    ):
        """
        Accumulate one Steiner observation into the four per-cell stat arrays.

        Pure-Python counterpart of ``_accumulate_steiner_nb`` and
        ``_accumulate_steiner_cuda``.  No atomics — called from the
        single-threaded Python fallback kernel.

        ``mult`` is the tree multiplicity weight.  ``steiner_out`` and
        ``steiner_sum_sq_out`` are weighted; min/max are not (identical
        duplicate trees have the same Steiner length).
        """
        steiner_out[qi, gi, topo] += sl * mult
        if sl < steiner_min_out[qi, gi, topo]:
            steiner_min_out[qi, gi, topo] = sl
        if sl > steiner_max_out[qi, gi, topo]:
            steiner_max_out[qi, gi, topo] = sl
        steiner_sum_sq_out[qi, gi, topo] += sl * sl * mult

    @staticmethod
    def _python_quartet_counts_delta(
        delta_quartet_ids,
        delta_quartet_global_idx,
        old_global_to_local,
        new_global_to_local,
        delta_tree_ids,
        all_first_occ,
        all_root_distance,
        all_euler_tour,
        all_euler_depth,
        all_sparse_table,
        all_log2_table,
        node_offsets,
        tour_offsets,
        sp_offsets,
        lg_offsets,
        sp_tour_widths,
        tree_to_group_idx,
        polytomy_offsets,
        polytomy_nodes,
        tree_multiplicities,
        counts_out,
    ):
        """
        Pure-Python fallback for the accelerated delta kernels.

        Applies signed ±mult updates to ``counts_out`` in-place for each
        (affected_quartet, affected_tree) pair where the topology changes
        between the old and new copy-slot assignments.  ``mult`` is the
        tree multiplicity weight from ``tree_multiplicities[ti]``.
        """
        n_affected_quartets = delta_quartet_ids.shape[0]
        for qi_local in range(n_affected_quartets):
            t0 = int(delta_quartet_ids[qi_local, 0])
            t1 = int(delta_quartet_ids[qi_local, 1])
            t2 = int(delta_quartet_ids[qi_local, 2])
            t3 = int(delta_quartet_ids[qi_local, 3])
            qi = int(delta_quartet_global_idx[qi_local])

            for ti in delta_tree_ids:
                ti = int(ti)
                gi = int(tree_to_group_idx[ti])

                # ---- Old assignment --------------------------------------- #
                ln0_old, ln1_old, ln2_old, ln3_old, \
                    node_base, tour_base, sp_base, lg_base, sp_stride = \
                    Forest._resolve_quartet(
                        t0, t1, t2, t3, ti,
                        old_global_to_local, node_offsets, tour_offsets,
                        sp_offsets, lg_offsets, sp_tour_widths,
                    )
                if ln0_old < 0 or ln1_old < 0 or ln2_old < 0 or ln3_old < 0:
                    continue

                fo0 = int(all_first_occ[node_base + ln0_old])
                fo1 = int(all_first_occ[node_base + ln1_old])
                fo2 = int(all_first_occ[node_base + ln2_old])
                fo3 = int(all_first_occ[node_base + ln3_old])
                poly_start = int(polytomy_offsets[ti])
                poly_end   = int(polytomy_offsets[ti + 1])
                old_topo, _, _, _, _ = Forest._quartet_topology_and_rd(
                    fo0, fo1, fo2, fo3,
                    node_base, tour_base, sp_base, lg_base, sp_stride,
                    all_root_distance, all_sparse_table, all_euler_depth,
                    all_log2_table, all_euler_tour,
                    poly_start, poly_end, polytomy_nodes,
                )

                # ---- New assignment --------------------------------------- #
                ln0_new = int(new_global_to_local[ti, t0])
                ln1_new = int(new_global_to_local[ti, t1])
                ln2_new = int(new_global_to_local[ti, t2])
                ln3_new = int(new_global_to_local[ti, t3])
                if ln0_new < 0 or ln1_new < 0 or ln2_new < 0 or ln3_new < 0:
                    continue

                fo0 = int(all_first_occ[node_base + ln0_new])
                fo1 = int(all_first_occ[node_base + ln1_new])
                fo2 = int(all_first_occ[node_base + ln2_new])
                fo3 = int(all_first_occ[node_base + ln3_new])
                new_topo, _, _, _, _ = Forest._quartet_topology_and_rd(
                    fo0, fo1, fo2, fo3,
                    node_base, tour_base, sp_base, lg_base, sp_stride,
                    all_root_distance, all_sparse_table, all_euler_depth,
                    all_log2_table, all_euler_tour,
                    poly_start, poly_end, polytomy_nodes,
                )

                # ---- Apply signed delta ----------------------------------- #
                if old_topo != new_topo:
                    mult = int(tree_multiplicities[ti])
                    counts_out[qi, gi, old_topo] -= mult
                    counts_out[qi, gi, new_topo] += mult

    def apply_quartet_counts_delta(
        self,
        paralog_data,
        li: int,
        perm: np.ndarray,
        counts_out: np.ndarray,
        backend: str = "best",
    ) -> np.ndarray:
        """
        Apply a paralog copy-slot permutation to ``counts_out`` in-place.

        Computes the topology change for every (affected_quartet, affected_tree)
        pair and applies ±1 signed updates to ``counts_out``.

        Parameters
        ----------
        paralog_data : ParalogData
            Built from this forest via ``build_paralog_data(forest, quartets)``.
        li : int
            Genome index within ``paralog_data.genome_names``.
        perm : int array [k]
            Copy-slot permutation.  ``perm[ci]`` = old slot whose leaf is
            placed at copy position *ci*.  Identity = ``[0, 1, …, k-1]``.
        counts_out : int32 ndarray [n_quartets, n_groups, 4]
            Modified in-place.
        backend : str
            Computational backend.  ``'best'`` selects the fastest available.
            ``'cuda'`` runs a 2D CUDA kernel (atomics; device-resident counts
            bypass this call when ``ParalogOptimizer`` is used directly).
            ``'mlx'`` runs a 1D Metal kernel (no atomics; UMA means wrapping
            host arrays as ``mx.array`` is essentially free).
            ``'cpu-parallel'`` uses Numba prange.
            ``'python'`` is the pure-Python fallback.
            Default ``'best'``.

        Returns
        -------
        trial_global_to_local : int32 ndarray [n_trees, n_global_taxa]
            The modified global→local mapping corresponding to the applied
            permutation (the caller may use this to update the current state).
        """
        trial_g2l, affected_tree_ids, affected_taxa, affected_qi = \
            paralog_data.apply_permutation(li, perm, self.global_to_local)

        if len(affected_qi) == 0 or len(affected_tree_ids) == 0:
            return trial_g2l

        resolved = get_backend_override() or backends.resolve(backend)
        kd = self._kernel_data

        if resolved == "cuda" and backends.cuda:
            # Upload small per-call arrays; structural arrays come from the
            # pre-uploaded _cuda_kernel_data (never re-uploaded per delta call).
            ckd = self._cuda_kernel_data
            d_delta_taxa  = cuda.to_device(affected_taxa)
            d_delta_qi    = cuda.to_device(affected_qi)
            d_old_g2l     = cuda.to_device(self.global_to_local)
            d_new_g2l     = cuda.to_device(trial_g2l)
            d_tree_ids    = cuda.to_device(affected_tree_ids)
            d_counts      = cuda.to_device(counts_out)

            n_aq = len(affected_qi)
            n_at = len(affected_tree_ids)
            tpb = (16, 16)
            bpg = (
                (n_aq + tpb[0] - 1) // tpb[0],
                (n_at + tpb[1] - 1) // tpb[1],
            )
            _quartet_counts_delta_cuda[bpg, tpb](
                d_delta_taxa,
                d_delta_qi,
                np.int32(n_aq),
                d_old_g2l,
                d_new_g2l,
                d_tree_ids,
                np.int32(n_at),
                ckd.all_first_occ,
                ckd.all_root_distance,
                ckd.all_euler_tour,
                ckd.all_euler_depth,
                ckd.all_sparse_table,
                ckd.all_log2_table,
                ckd.node_offsets,
                ckd.tour_offsets,
                ckd.sp_offsets,
                ckd.lg_offsets,
                ckd.sp_tour_widths,
                ckd.tree_to_group_idx,
                ckd.polytomy_offsets,
                ckd.polytomy_nodes,
                ckd.tree_multiplicities,
                d_counts,
            )
            d_counts.copy_to_host(counts_out)
        elif resolved == "mlx" and backends.mlx:
            from quarimo._mlx_kernels import quartet_counts_delta_mlx
            quartet_counts_delta_mlx(
                kd,
                affected_taxa,
                affected_qi,
                self.global_to_local,
                trial_g2l,
                affected_tree_ids,
                counts_out,
                self.n_groups,
            )
        elif resolved == "cpu-parallel" and backends.numba:
            _quartet_counts_delta_nb(
                affected_taxa,
                affected_qi,
                np.int32(len(affected_qi)),
                self.global_to_local,
                trial_g2l,
                affected_tree_ids,
                np.int32(len(affected_tree_ids)),
                kd.all_first_occ,
                kd.all_root_distance,
                kd.all_euler_tour,
                kd.all_euler_depth,
                kd.all_sparse_table,
                kd.all_log2_table,
                kd.node_offsets,
                kd.tour_offsets,
                kd.sp_offsets,
                kd.lg_offsets,
                kd.sp_tour_widths,
                kd.tree_to_group_idx,
                kd.polytomy_offsets,
                kd.polytomy_nodes,
                kd.tree_multiplicities,
                counts_out,
            )
        else:
            Forest._python_quartet_counts_delta(
                affected_taxa,
                affected_qi,
                self.global_to_local,
                trial_g2l,
                affected_tree_ids,
                kd.all_first_occ,
                kd.all_root_distance,
                kd.all_euler_tour,
                kd.all_euler_depth,
                kd.all_sparse_table,
                kd.all_log2_table,
                kd.node_offsets,
                kd.tour_offsets,
                kd.sp_offsets,
                kd.lg_offsets,
                kd.sp_tour_widths,
                kd.tree_to_group_idx,
                kd.polytomy_offsets,
                kd.polytomy_nodes,
                kd.tree_multiplicities,
                counts_out,
            )

        return trial_g2l

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
