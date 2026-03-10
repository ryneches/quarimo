"""
_kernel_data.py
===============
Dataclasses that package forest arrays and per-call quartet parameters
for dispatch to computation kernels across all backends (Python, CPU-parallel,
CUDA).

These dataclasses serve four purposes:

  1. Reduce argument-list length at kernel call sites.
  2. Centralise array layout documentation in one place.
  3. Make adding a new forest-wide array a single-point change.
  4. Provide a consistent interface across all backends and kernels.

The kernel signatures themselves stay flat — required for Numba ``@cuda.jit``
and beneficial for readability of PTX-compiled code.  Packaging and unpacking
happen exclusively at the Python dispatch layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quarimo._quartets import Quartets


@dataclass
class ForestKernelData:
    """
    All forest arrays required by any quartet kernel.

    Created once at ``Forest.__init__`` after ``_pack_csr()`` completes and
    stored as ``Forest._kernel_data``.  For the CUDA backend a second instance
    is created with device-resident arrays (uploaded at construction) and stored
    as ``Forest._cuda_kernel_data``.  Both instances expose the same interface,
    so dispatch code is backend-agnostic.

    Topology arrays (CSR flat-packed)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    global_to_local : int32[n_trees, n_global_taxa]
        Maps (tree_index, global_taxon_id) → local leaf ID in that tree.
        Value is -1 when the taxon is absent from the tree.

    all_first_occ : int32[total_nodes]
        First-occurrence position in the Euler tour for each node across all
        trees, indexed by ``node_offsets[ti] + local_node_id``.

    all_root_distance : float64[total_nodes]
        Cumulative branch-length distance from each node to the root, indexed
        identically to ``all_first_occ``.

    all_euler_tour : int32[total_tour_len]
        Flattened Euler tours (DFS node sequence) for all trees, indexed by
        ``tour_offsets[ti] + position``.

    all_euler_depth : int32[total_tour_len]
        Depth of each node at the corresponding Euler tour position.

    all_sparse_table : int32[total_sp_size]
        Range-minimum-query sparse table over ``all_euler_depth``.  Layout:
        ``sp_offsets[ti] + level * sp_tour_widths[ti] + position``.

    all_log2_table : int32[total_log2_size]
        Floor-log₂ lookup table per tree, indexed by
        ``lg_offsets[ti] + length``.

    Offset arrays
    ~~~~~~~~~~~~~
    node_offsets : int64[n_trees + 1]
        CSR offsets into ``all_first_occ`` and ``all_root_distance``.

    tour_offsets : int64[n_trees + 1]
        CSR offsets into ``all_euler_tour`` and ``all_euler_depth``.

    sp_offsets : int64[n_trees + 1]
        CSR offsets into ``all_sparse_table``.

    lg_offsets : int64[n_trees + 1]
        CSR offsets into ``all_log2_table``.

    sp_tour_widths : int32[n_trees]
        Column stride of the sparse table for each tree (= Euler tour length).

    Group and polytomy arrays
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    tree_to_group_idx : int32[n_trees]
        Maps each tree index to its group index in ``Forest.unique_groups``
        (sorted alphabetically).  All trees share index 0 for list-input forests.

    polytomy_offsets : int32[n_trees + 1]
        CSR offsets into ``polytomy_nodes``.
        ``polytomy_offsets[ti+1] - polytomy_offsets[ti]`` is the number of
        polytomy-inserted internal nodes in tree *ti* (zero for binary trees).

    polytomy_nodes : int32[total_polytomy]
        Local node IDs of polytomy-inserted internal nodes, in tree order.
        Used by the hybrid CSR+tie polytomy detection in all kernels.

    Scalars
    ~~~~~~~
    n_trees : int
    n_global_taxa : int
    n_groups : int
    """

    # Topology arrays
    global_to_local: np.ndarray    # int32[n_trees, n_global_taxa]
    all_first_occ: np.ndarray      # int32[total_nodes]
    all_root_distance: np.ndarray  # float64[total_nodes]
    all_euler_tour: np.ndarray     # int32[total_tour_len]
    all_euler_depth: np.ndarray    # int32[total_tour_len]
    all_sparse_table: np.ndarray   # int32[total_sp_size]
    all_log2_table: np.ndarray     # int32[total_log2_size]

    # Offset arrays
    node_offsets: np.ndarray       # int64[n_trees + 1]
    tour_offsets: np.ndarray       # int64[n_trees + 1]
    sp_offsets: np.ndarray         # int64[n_trees + 1]
    lg_offsets: np.ndarray         # int64[n_trees + 1]
    sp_tour_widths: np.ndarray     # int32[n_trees]

    # Group and polytomy arrays
    tree_to_group_idx: np.ndarray  # int32[n_trees]
    polytomy_offsets: np.ndarray   # int32[n_trees + 1]
    polytomy_nodes: np.ndarray     # int32[total_polytomy]

    # Scalars
    n_trees: int
    n_global_taxa: int
    n_groups: int

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def cpu_common_args(self, sorted_ids: np.ndarray, n_quartets: int) -> tuple:
        """
        Full argument tuple for CPU kernel calls (counts and Steiner).

        Matches the positional parameter order of ``_quartet_counts_njit``
        and ``_quartet_steiner_njit``.  Append output arrays (``counts_out``,
        optionally ``steiner_out``) at the end before calling the kernel.

        Parameters
        ----------
        sorted_ids : int32[n_quartets, 4]
            Pre-materialised quartet IDs, sorted ascending per row.
        n_quartets : int
            Number of quartets (``sorted_ids.shape[0]``).
        """
        return (
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
            self.tree_to_group_idx,
            self.polytomy_offsets,
            self.polytomy_nodes,
        )

    def cuda_forest_args(self) -> tuple:
        """
        The 15 static forest arrays for CUDA kernel calls.

        Matches the ``# Forest data (CSR format)`` parameter block in
        ``quartet_counts_cuda_unified`` and ``quartet_steiner_cuda_unified``.
        Prepend ``(d_seed, n_seed, offset, count, rng_seed, n_global_taxa)``
        and append output arrays to form the complete kernel argument list.
        """
        return (
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
            self.tree_to_group_idx,
            self.polytomy_offsets,
            self.polytomy_nodes,
        )

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def to_device(self) -> "ForestKernelData":
        """
        Upload all arrays to the GPU and return a new ``ForestKernelData``
        with device-resident arrays.  Scalars are copied unchanged.

        Requires ``numba.cuda`` to be available.
        """
        from numba import cuda

        return ForestKernelData(
            global_to_local=cuda.to_device(self.global_to_local),
            all_first_occ=cuda.to_device(self.all_first_occ),
            all_root_distance=cuda.to_device(self.all_root_distance),
            all_euler_tour=cuda.to_device(self.all_euler_tour),
            all_euler_depth=cuda.to_device(self.all_euler_depth),
            all_sparse_table=cuda.to_device(self.all_sparse_table),
            all_log2_table=cuda.to_device(self.all_log2_table),
            node_offsets=cuda.to_device(self.node_offsets),
            tour_offsets=cuda.to_device(self.tour_offsets),
            sp_offsets=cuda.to_device(self.sp_offsets),
            lg_offsets=cuda.to_device(self.lg_offsets),
            sp_tour_widths=cuda.to_device(self.sp_tour_widths),
            tree_to_group_idx=cuda.to_device(self.tree_to_group_idx),
            polytomy_offsets=cuda.to_device(self.polytomy_offsets),
            polytomy_nodes=cuda.to_device(self.polytomy_nodes),
            n_trees=self.n_trees,
            n_global_taxa=self.n_global_taxa,
            n_groups=self.n_groups,
        )

    def device_arrays(self) -> list:
        """Return all array attributes as a flat list (for GPU cleanup)."""
        return [
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
            self.tree_to_group_idx,
            self.polytomy_offsets,
            self.polytomy_nodes,
        ]

    @property
    def upload_bytes(self) -> int:
        """Total byte size of all array attributes (CPU arrays only)."""
        return sum(a.nbytes for a in self.device_arrays())


@dataclass
class QuartetKernelArgs:
    """
    Per-call quartet generation parameters derived from a ``Quartets`` object.

    Created once at the start of ``Forest.quartet_topology()`` and passed to
    backend-specific dispatch helpers.  Avoids threading individual ``Quartets``
    attributes through multiple method calls.

    Attributes
    ----------
    seed : int32[n_seed, 4]
        Explicit seed quartets.  The first ``n_seed`` positions in the
        infinite deterministic sequence are read from this array rather than
        from the RNG.

    n_seed : int
        Number of explicit seed quartets (``seed.shape[0]``).

    rng_seed : uint32
        XorShift128 seed derived from ``Quartets.rng_seed``.  Determines the
        pseudo-random suffix of the sequence beyond ``n_seed``.

    n_quartets : int
        Total quartets to process (``len(quartets)``).

    offset : int
        Starting index in the infinite deterministic sequence.
    """

    seed: np.ndarray    # int32[n_seed, 4]
    n_seed: int
    rng_seed: np.uint32
    n_quartets: int
    offset: int

    @classmethod
    def from_quartets(cls, quartets: "Quartets") -> "QuartetKernelArgs":
        """Build from a ``Quartets`` object."""
        return cls(
            seed=np.array(quartets.seed, dtype=np.int32),
            n_seed=len(quartets.seed),
            rng_seed=quartets.rng_seed,
            n_quartets=len(quartets),
            offset=quartets.offset,
        )

    def cuda_batch_args(self, d_seed, proc_n_seed: int, batch_offset: int, bc: int) -> tuple:
        """
        Quartet-generation argument block for one CUDA batch.

        Matches the ``# Quartet generation parameters`` block at the top of
        ``quartet_counts_cuda_unified`` / ``quartet_steiner_cuda_unified``.
        Append ``n_global_taxa``, the forest args tuple, and output arrays to
        form the complete kernel argument list.

        Parameters
        ----------
        d_seed : device ndarray
            Device-resident seed array for this batch — either the original
            ``d_seed_quartets`` or a pre-generated scratch array from the
            1D generation kernel.
        proc_n_seed : int
            Effective seed count for this batch.  Pass ``self.n_seed`` when
            ``d_seed`` is the original seed array (``batch_needs_rng=False``).
            Pass ``bc`` when ``d_seed`` is a pre-generated scratch array
            (``batch_needs_rng=True``) so every thread qi < bc reads from the
            scratch array rather than falling through to the on-kernel RNG.
        batch_offset : int
            Starting absolute index in the infinite sequence for this batch.
        bc : int
            Batch count (number of quartets in this batch).
        """
        return (d_seed, proc_n_seed, batch_offset, bc, self.rng_seed)
