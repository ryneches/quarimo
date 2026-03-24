"""
_cuda_kernels.py
================
CUDA-accelerated quartet topology kernels using Numba CUDA.

This module contains ONLY numba.cuda code and should not import other project
modules to avoid import-time complications. The module exposes GPU-accelerated
kernels for quartet topology queries when a compatible CUDA GPU is available.

Device helpers (inlined by PTX compiler, no call overhead)
----------------------------------------------------------
_rmq_csr_cuda
    O(1) range minimum query over CSR-packed sparse table.

_resolve_quartet_cuda
    Map four global taxon IDs to tree-local positions.

_quartet_topology_and_rd_cuda
    Six RMQ calls + four-point condition → topology and pair-sums.

_steiner_length_cuda
    Steiner spanning length of the winning quartet topology.

_polytomy_check_cuda
    CSR polytomy scan + tie check → (found, topo, r0, r1, r2, rw).
    Called by both unified kernels; returns found=False for binary trees.

Kernels (called from _forest.py via the cuda backend)
------------------------------------------------------
generate_quartets_cuda
    1D kernel: materialise quartets from the deterministic sequence.

quartet_counts_cuda_unified
    2D kernel: topology counts, on-GPU quartet generation, per-group output.

quartet_steiner_cuda_unified
    2D kernel: topology counts + Steiner distances, per-group output.

_quartet_counts_cuda, _quartet_steiner_cuda
    Pre-materialized variants (legacy; not used by the main dispatch path).

_quartet_counts_delta_cuda
    2D kernel: signed ±1 delta updates to counts for a paralog permutation.

_counts_d2d_copy_cuda
    3D kernel: device-to-device copy of a (n_quartets, n_groups, 4) array.
    Used by device-aware ParalogOptimizer to avoid PCIe round-trips.

_compute_cuda_grid
    Helper to compute CUDA grid dimensions.

Notes
-----
- Kernel grid is 2D: x over quartets (qi), y over trees (ti).
- counts and steiner_out accumulate with cuda.atomic.add (multiple threads
  write to the same group row).
"""

import numpy as np

# This module requires numba.cuda
try:
    from numba import cuda
    import numba
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    # If CUDA is not available, this module will not be importable
    # The main module should handle this gracefully


if _CUDA_AVAILABLE:
    # ======================================================================== #
    # CUDA Kernels                                                              #
    # ======================================================================== #

    @cuda.jit(device=True)
    def _rmq_csr_cuda(l, r, sp_base, sp_stride, sparse_table, euler_depth,
                      log2_table, lg_base, tour_base, euler_tour):
        """
        O(1) RMQ over CSR-packed sparse table for a single tree (device-only).

        Parameters and return value mirror ``_rmq_csr_nb``; all arrays are
        device-resident.  See that function's docstring for details.
        """
        length = r - l + 1
        k = log2_table[lg_base + length]
        half = 1 << k
        li = sparse_table[sp_base + k * sp_stride + l]
        ri = sparse_table[sp_base + k * sp_stride + (r - half + 1)]
        if euler_depth[tour_base + ri] < euler_depth[tour_base + li]:
            lca_local = ri
        else:
            lca_local = li
        return euler_tour[tour_base + lca_local]

    # ======================================================================== #
    # RNG Device Functions for On-GPU Quartet Generation                       #
    # ======================================================================== #
    # Imported from _rng_cuda.py — uses numba.uint32/uint64 for correct
    # 32-bit arithmetic in CUDA device functions (np.uint32/uint64 can cause
    # 64-bit promotion in Numba, producing wrong bit patterns).

    from quarimo._rng_cuda import (  # noqa: E402
        init_xorshift128,
        xorshift128_next,
        sample_4_unique_cuda,
        get_quartet_at_index,
    )

    @cuda.jit(device=True)
    def _resolve_quartet_cuda(t0, t1, t2, t3, ti,
                              global_to_local,
                              node_offsets, tour_offsets, sp_offsets, lg_offsets,
                              sp_tour_widths):
        """
        Map four global taxon IDs to tree-local positions for tree *ti*.

        CUDA device counterpart of ``_resolve_quartet_nb``; see that function's
        docstring for the full return-value description and caller contract.
        All arrays are device-resident.
        """
        ln0       = global_to_local[ti, t0]
        ln1       = global_to_local[ti, t1]
        ln2       = global_to_local[ti, t2]
        ln3       = global_to_local[ti, t3]
        node_base = node_offsets[ti]
        tour_base = tour_offsets[ti]
        sp_base   = sp_offsets[ti]
        lg_base   = lg_offsets[ti]
        sp_stride = sp_tour_widths[ti]
        return ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride

    @cuda.jit(device=True)
    def _quartet_topology_and_rd_cuda(fo0, fo1, fo2, fo3,
                                       node_base, tour_base, sp_base, lg_base, sp_stride,
                                       all_root_distance,
                                       all_sparse_table, all_euler_depth,
                                       all_log2_table, all_euler_tour):
        """
        Six RMQ calls + four-point condition → topology and pair-sums.

        CUDA device counterpart of ``_quartet_topology_and_rd_nb``; see that
        function's docstring for parameter and return-value descriptions.
        All arrays are device-resident.
        """
        l = fo0; r = fo1
        if l > r: l, r = r, l
        rd01 = all_root_distance[node_base + _rmq_csr_cuda(l, r, sp_base, sp_stride,
                                                             all_sparse_table,
                                                             all_euler_depth, all_log2_table,
                                                             lg_base, tour_base, all_euler_tour)]
        l = fo0; r = fo2
        if l > r: l, r = r, l
        rd02 = all_root_distance[node_base + _rmq_csr_cuda(l, r, sp_base, sp_stride,
                                                             all_sparse_table,
                                                             all_euler_depth, all_log2_table,
                                                             lg_base, tour_base, all_euler_tour)]
        l = fo0; r = fo3
        if l > r: l, r = r, l
        rd03 = all_root_distance[node_base + _rmq_csr_cuda(l, r, sp_base, sp_stride,
                                                             all_sparse_table,
                                                             all_euler_depth, all_log2_table,
                                                             lg_base, tour_base, all_euler_tour)]
        l = fo1; r = fo2
        if l > r: l, r = r, l
        rd12 = all_root_distance[node_base + _rmq_csr_cuda(l, r, sp_base, sp_stride,
                                                             all_sparse_table,
                                                             all_euler_depth, all_log2_table,
                                                             lg_base, tour_base, all_euler_tour)]
        l = fo1; r = fo3
        if l > r: l, r = r, l
        rd13 = all_root_distance[node_base + _rmq_csr_cuda(l, r, sp_base, sp_stride,
                                                             all_sparse_table,
                                                             all_euler_depth, all_log2_table,
                                                             lg_base, tour_base, all_euler_tour)]
        l = fo2; r = fo3
        if l > r: l, r = r, l
        rd23 = all_root_distance[node_base + _rmq_csr_cuda(l, r, sp_base, sp_stride,
                                                             all_sparse_table,
                                                             all_euler_depth, all_log2_table,
                                                             lg_base, tour_base, all_euler_tour)]

        r0 = rd01 + rd23  # topology 0: (t0,t1)|(t2,t3)
        r1 = rd02 + rd13  # topology 1: (t0,t2)|(t1,t3)
        r2 = rd03 + rd12  # topology 2: (t0,t3)|(t1,t2)

        if r0 >= r1 and r0 >= r2:
            topo = 0; r_winner = r0
        elif r1 >= r0 and r1 >= r2:
            topo = 1; r_winner = r1
        else:
            topo = 2; r_winner = r2

        return topo, r0, r1, r2, r_winner

    @cuda.jit(device=True)
    def _steiner_length_cuda(ln0, ln1, ln2, ln3, node_base, r0, r1, r2, r_winner,
                              all_root_distance):
        """
        Steiner spanning length of the winning quartet topology.

        CUDA device counterpart of ``_steiner_length_nb``; see that function's
        docstring for parameter and return-value descriptions.
        ``all_root_distance`` is device-resident.
        """
        leaf_sum = (all_root_distance[node_base + ln0]
                  + all_root_distance[node_base + ln1]
                  + all_root_distance[node_base + ln2]
                  + all_root_distance[node_base + ln3])
        return leaf_sum - (r_winner + r0 + r1 + r2) * 0.5

    @cuda.jit(device=True)
    def _accumulate_steiner_cuda(qi, gi, topo, sl, mult,
                                  steiner_out, steiner_min_out,
                                  steiner_max_out, steiner_sum_sq_out):
        """
        Atomically accumulate one Steiner observation into the four stat arrays.

        CUDA device counterpart of ``_accumulate_steiner_nb``.  All updates use
        CUDA atomics because multiple (qi, ti) threads may share the same group
        row.  Counts are excluded — the caller handles ``counts`` atomics before
        this call.

        Parameters
        ----------
        qi, gi, topo : int
            Output cell indices.
        sl : float64
            Steiner spanning length for this (tree, quartet) observation.
        mult : int32
            Tree multiplicity weight.  ``steiner_out`` and ``steiner_sum_sq_out``
            are scaled by ``mult``; min/max are not.
        steiner_out : float64 device array [n_quartets, n_groups, 4]
        steiner_min_out : float64 device array [n_quartets, n_groups, 4]
        steiner_max_out : float64 device array [n_quartets, n_groups, 4]
        steiner_sum_sq_out : float64 device array [n_quartets, n_groups, 4]
        """
        cuda.atomic.add(steiner_out, (qi, gi, topo), sl * mult)
        cuda.atomic.min(steiner_min_out, (qi, gi, topo), sl)
        cuda.atomic.max(steiner_max_out, (qi, gi, topo), sl)
        cuda.atomic.add(steiner_sum_sq_out, (qi, gi, topo), sl * sl * mult)

    @cuda.jit(device=True)
    def _compute_lca_nodes_cuda(
            fo0, fo1, fo2, fo3,
            tour_base, sp_base, lg_base, sp_stride,
            all_sparse_table, all_euler_depth, all_log2_table, all_euler_tour):
        """
        Compute the six pairwise LCA local node IDs for four taxa (device-only).

        Returns topology-stable local node IDs for all six pairs — computed once
        per stored tree and reused across BL variants.  Device counterpart of
        ``_compute_lca_nodes_nb``.
        """
        if fo0 <= fo1:
            lca01 = _rmq_csr_cuda(fo0, fo1, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        else:
            lca01 = _rmq_csr_cuda(fo1, fo0, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        if fo2 <= fo3:
            lca23 = _rmq_csr_cuda(fo2, fo3, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        else:
            lca23 = _rmq_csr_cuda(fo3, fo2, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        if fo0 <= fo2:
            lca02 = _rmq_csr_cuda(fo0, fo2, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        else:
            lca02 = _rmq_csr_cuda(fo2, fo0, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        if fo1 <= fo3:
            lca13 = _rmq_csr_cuda(fo1, fo3, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        else:
            lca13 = _rmq_csr_cuda(fo3, fo1, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        if fo0 <= fo3:
            lca03 = _rmq_csr_cuda(fo0, fo3, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        else:
            lca03 = _rmq_csr_cuda(fo3, fo0, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        if fo1 <= fo2:
            lca12 = _rmq_csr_cuda(fo1, fo2, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        else:
            lca12 = _rmq_csr_cuda(fo2, fo1, sp_base, sp_stride, all_sparse_table,
                                   all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
        return lca01, lca23, lca02, lca13, lca03, lca12

    @cuda.jit(device=True)
    def _steiner_from_lca_nodes_cuda(
            ln0, ln1, ln2, ln3,
            lca01, lca23, lca02, lca13, lca03, lca12,
            rd_vbase, all_rd_variants):
        """
        Steiner length for one BL variant given precomputed LCA local node IDs
        (device-only).  Device counterpart of ``_steiner_from_lca_nodes_nb``.
        """
        r0_v = all_rd_variants[rd_vbase + lca01] + all_rd_variants[rd_vbase + lca23]
        r1_v = all_rd_variants[rd_vbase + lca02] + all_rd_variants[rd_vbase + lca13]
        r2_v = all_rd_variants[rd_vbase + lca03] + all_rd_variants[rd_vbase + lca12]
        leaf_sum = (all_rd_variants[rd_vbase + ln0]
                    + all_rd_variants[rd_vbase + ln1]
                    + all_rd_variants[rd_vbase + ln2]
                    + all_rd_variants[rd_vbase + ln3])
        if r0_v >= r1_v and r0_v >= r2_v:
            r_winner = r0_v
        elif r1_v >= r0_v and r1_v >= r2_v:
            r_winner = r1_v
        else:
            r_winner = r2_v
        return leaf_sum - (r_winner + r0_v + r1_v + r2_v) * 0.5

    @cuda.jit(device=True)
    def _polytomy_check_cuda(fo0, fo1, fo2, fo3,
                              node_base, tour_base, sp_base, lg_base, sp_stride,
                              poly_start, poly_end, polytomy_nodes,
                              all_sparse_table, all_euler_depth, all_log2_table,
                              all_euler_tour, all_root_distance):
        """
        CSR-based polytomy detection with tie check (device-only).

        Scans polytomy-inserted nodes for tree *ti* and determines whether
        the quartet is unresolvable (all three pair-sums equal) or resolves
        to a normal topology despite spanning a polytomy node.

        Returns
        -------
        found : bool
            True if a polytomy node is an LCA of any quartet pair.
            False when poly_end <= poly_start (binary tree, zero overhead)
            or when no polytomy node is any of the six LCAs.
        topo : int32
            Winning topology (0–3).  Meaningful only when found=True.
        r0, r1, r2 : float64
            Pair sums for topologies 0, 1, 2.  Meaningful only when found=True.
        rw : float64
            Pair sum of the winning topology.  Meaningful only when found=True.
        """
        if poly_end <= poly_start:
            return False, 0, 0.0, 0.0, 0.0, 0.0

        l = fo0; r = fo1
        if l > r: l, r = r, l
        lca01 = _rmq_csr_cuda(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                               all_log2_table, lg_base, tour_base, all_euler_tour)
        l = fo0; r = fo2
        if l > r: l, r = r, l
        lca02 = _rmq_csr_cuda(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                               all_log2_table, lg_base, tour_base, all_euler_tour)
        l = fo0; r = fo3
        if l > r: l, r = r, l
        lca03 = _rmq_csr_cuda(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                               all_log2_table, lg_base, tour_base, all_euler_tour)
        l = fo1; r = fo2
        if l > r: l, r = r, l
        lca12 = _rmq_csr_cuda(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                               all_log2_table, lg_base, tour_base, all_euler_tour)
        l = fo1; r = fo3
        if l > r: l, r = r, l
        lca13 = _rmq_csr_cuda(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                               all_log2_table, lg_base, tour_base, all_euler_tour)
        l = fo2; r = fo3
        if l > r: l, r = r, l
        lca23 = _rmq_csr_cuda(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth,
                               all_log2_table, lg_base, tour_base, all_euler_tour)

        for j in range(poly_start, poly_end):
            pn = polytomy_nodes[j]
            if (pn == lca01 or pn == lca02 or pn == lca03
                    or pn == lca12 or pn == lca13 or pn == lca23):
                rd01 = all_root_distance[node_base + lca01]
                rd23 = all_root_distance[node_base + lca23]
                rd02 = all_root_distance[node_base + lca02]
                rd13 = all_root_distance[node_base + lca13]
                rd03 = all_root_distance[node_base + lca03]
                rd12 = all_root_distance[node_base + lca12]
                r0 = rd01 + rd23; r1 = rd02 + rd13; r2 = rd03 + rd12
                if r0 == r1 and r1 == r2:
                    return True, 3, r0, r1, r2, r0
                elif r0 >= r1 and r0 >= r2:
                    return True, 0, r0, r1, r2, r0
                elif r1 >= r0 and r1 >= r2:
                    return True, 1, r0, r1, r2, r1
                else:
                    return True, 2, r0, r1, r2, r2

        return False, 0, 0.0, 0.0, 0.0, 0.0

    @cuda.jit
    def _quartet_counts_cuda(
            sorted_quartet_ids,
            global_to_local,
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
            n_quartets,
            n_trees,
            counts_out):
        """
        Pre-materialized counts-only quartet kernel (legacy).

        Not used by the main dispatch path in ``_forest.py``; superseded by
        ``quartet_counts_cuda_unified``.  Output shape is ``(n_quartets, 3)``
        — no per-group axis.
        """
        # 2D thread grid: (qi, ti)
        qi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Bounds check
        if qi >= n_quartets or ti >= n_trees:
            return

        # Get quartet taxa and resolve to tree-local positions
        t0 = sorted_quartet_ids[qi, 0]
        t1 = sorted_quartet_ids[qi, 1]
        t2 = sorted_quartet_ids[qi, 2]
        t3 = sorted_quartet_ids[qi, 3]
        ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
            _resolve_quartet_cuda(
                t0, t1, t2, t3, ti,
                global_to_local, node_offsets, tour_offsets, sp_offsets,
                lg_offsets, sp_tour_widths,
            )
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        fo0 = all_first_occ[node_base + ln0]
        fo1 = all_first_occ[node_base + ln1]
        fo2 = all_first_occ[node_base + ln2]
        fo3 = all_first_occ[node_base + ln3]
        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
            all_root_distance, all_sparse_table, all_euler_depth,
            all_log2_table, all_euler_tour,
        )

        # Atomic increment (multiple threads may write to same (qi, topo))
        cuda.atomic.add(counts_out, (qi, topo), 1)

    @cuda.jit
    def _quartet_steiner_cuda(
            sorted_quartet_ids,
            global_to_local,
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
            n_quartets,
            n_trees,
            counts_out,
            steiner_out):
        """
        Pre-materialized Steiner quartet kernel (legacy).

        Not used by the main dispatch path in ``_forest.py``; superseded by
        ``quartet_steiner_cuda_unified``.  Output shapes are ``(n_quartets, 3)``
        — no per-group axis.
        """
        # 2D thread grid: (qi, ti)
        qi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Bounds check
        if qi >= n_quartets or ti >= n_trees:
            return

        # Get quartet taxa and resolve to tree-local positions
        t0 = sorted_quartet_ids[qi, 0]
        t1 = sorted_quartet_ids[qi, 1]
        t2 = sorted_quartet_ids[qi, 2]
        t3 = sorted_quartet_ids[qi, 3]
        ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
            _resolve_quartet_cuda(
                t0, t1, t2, t3, ti,
                global_to_local, node_offsets, tour_offsets, sp_offsets,
                lg_offsets, sp_tour_widths,
            )
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        fo0 = all_first_occ[node_base + ln0]
        fo1 = all_first_occ[node_base + ln1]
        fo2 = all_first_occ[node_base + ln2]
        fo3 = all_first_occ[node_base + ln3]
        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
            all_root_distance, all_sparse_table, all_euler_depth,
            all_log2_table, all_euler_tour,
        )

        # Atomic increment for counts
        cuda.atomic.add(counts_out, (qi, topo), 1)

        # Compute and store Steiner distance (conflict-free write)
        steiner_out[qi, ti, topo] = _steiner_length_cuda(
            ln0, ln1, ln2, ln3, node_base, r0, r1, r2, r_winner, all_root_distance,
        )


    # ======================================================================== #
    # 1D Generation Kernel                                                    #
    # ======================================================================== #

    @cuda.jit
    def generate_quartets_cuda(
        seed_quartets,  # [n_seed, 4] int32
        n_seed,         # int - number of seed quartets
        offset,         # int - starting absolute index in the sequence
        count,          # int - number of quartets to generate
        rng_seed,       # uint32 - RNG seed
        n_taxa,         # int - namespace size
        quartets_out    # [count, 4] int32 - output device array
    ):
        """
        1D generation kernel: materialise quartets from the deterministic sequence.

        One thread per quartet. Thread qi writes the quartet at
        ``absolute_idx = offset + qi`` to ``quartets_out[qi, :]``.

        After this kernel, ``quartets_out`` can be passed to the 2D processing
        kernels as a pre-generated seed array (``n_seed=count, offset=0``),
        so each quartet is computed exactly once rather than once per tree.
        """
        qi = cuda.grid(1)
        if qi >= count:
            return
        absolute_idx = offset + qi
        a, b, c, d = get_quartet_at_index(
            absolute_idx, seed_quartets, n_seed, rng_seed, n_taxa
        )
        quartets_out[qi, 0] = a
        quartets_out[qi, 1] = b
        quartets_out[qi, 2] = c
        quartets_out[qi, 3] = d

    # ======================================================================== #
    # Unified Kernels with On-GPU Quartet Generation                          #
    # ======================================================================== #

    @cuda.jit
    def quartet_counts_cuda_unified(
        # Quartet generation parameters
        seed_quartets,      # [n_seed, 4] int32 - explicit seed quartets
        n_seed,             # int - number of seed quartets
        offset,             # int - starting index in sequence
        count,              # int - number of quartets to process
        rng_seed,           # uint32 - hash of seed for RNG
        n_global_taxa,      # int - namespace size
        # Forest data (CSR format)
        global_to_local,    # [n_trees, n_global_taxa] int32
        all_first_occ,      # [total_nodes] int32
        all_root_distance,  # [total_nodes] float64
        all_euler_tour,     # [total_tour_len] int32
        all_euler_depth,    # [total_tour_len] int32
        all_sparse_table,   # [total_sp_size] int32
        all_log2_table,     # [total_log2_size] int32
        node_offsets,       # [n_trees + 1] int64
        tour_offsets,       # [n_trees + 1] int64
        sp_offsets,         # [n_trees + 1] int64
        lg_offsets,         # [n_trees + 1] int64
        sp_tour_widths,     # [n_trees] int32
        tree_to_group_idx,  # [n_trees] int32 - maps tree to group index
        polytomy_offsets,   # [n_trees + 1] int32 - CSR offsets for polytomy nodes
        polytomy_nodes,     # [total_polytomy] int32 - local node IDs of polytomy internals
        tree_multiplicities,  # [n_trees] int32 - deduplication weights
        bl_variant_offsets,       # [n_trees + 1] int32 - CSR into bl_variant_multiplicities
        bl_variant_multiplicities,  # [total_variants] int32 - per-variant counts
        bl_node_offsets,          # [total_variants + 1] int32 - CSR into all_rd_variants
        all_rd_variants,          # [total_variant_nodes] float64 - per-variant root distances
        # Output
        counts              # [count, n_groups, 4] int32 - topology counts per group
    ):
        """
        Unified kernel: process quartets from deterministic sequence.

        For each (qi, ti) thread pair:
          absolute_idx = offset + qi
          quartet = seed_quartets[qi] if qi < n_seed else generate_random(...)
          Process that quartet for tree ti

        Grid/Block Configuration
        -------------------------
        - Grid: 2D grid — x over quartets, y over trees
        - threads_per_block: typically (16, 16) = 256 threads per block
        - Each thread processes one (quartet, tree) pair
        """
        qi, ti = cuda.grid(2)
        n_trees = node_offsets.shape[0] - 1
        if qi >= count or ti >= n_trees:
            return

        # Determine absolute index in the infinite sequence
        absolute_idx = offset + qi

        # Get quartet for this index (re-run RNG per thread; fast for seed case)
        a, b, c, d = get_quartet_at_index(
            absolute_idx, seed_quartets, n_seed, rng_seed, n_global_taxa
        )

        # Resolve global IDs to tree-local positions
        ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
            _resolve_quartet_cuda(
                a, b, c, d, ti,
                global_to_local, node_offsets, tour_offsets, sp_offsets,
                lg_offsets, sp_tour_widths,
            )
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        fo0 = all_first_occ[node_base + ln0]
        fo1 = all_first_occ[node_base + ln1]
        fo2 = all_first_occ[node_base + ln2]
        fo3 = all_first_occ[node_base + ln3]

        # CSR-based polytomy detection (zero overhead for trees without polytomies)
        poly_start = polytomy_offsets[ti]
        poly_end = polytomy_offsets[ti + 1]
        found, topo, r0, r1, r2, rw = _polytomy_check_cuda(
            fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
            poly_start, poly_end, polytomy_nodes,
            all_sparse_table, all_euler_depth, all_log2_table,
            all_euler_tour, all_root_distance,
        )
        mult = tree_multiplicities[ti]
        if found:
            cuda.atomic.add(counts, (qi, tree_to_group_idx[ti], topo), mult)
            return

        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
            all_root_distance, all_sparse_table, all_euler_depth,
            all_log2_table, all_euler_tour,
        )

        gi = tree_to_group_idx[ti]
        cuda.atomic.add(counts, (qi, gi, topo), mult)

    @cuda.jit
    def quartet_steiner_cuda_unified(
        # Quartet generation parameters
        seed_quartets,      # [n_seed, 4] int32
        n_seed,             # int
        offset,             # int
        count,              # int
        rng_seed,           # uint32
        n_global_taxa,      # int
        # Forest data
        global_to_local,    # [n_trees, n_global_taxa] int32
        all_first_occ,      # [total_nodes] int32
        all_root_distance,  # [total_nodes] float64
        all_euler_tour,     # [total_tour_len] int32
        all_euler_depth,    # [total_tour_len] int32
        all_sparse_table,   # [total_sp_size] int32
        all_log2_table,     # [total_log2_size] int32
        node_offsets,       # [n_trees + 1] int64
        tour_offsets,       # [n_trees + 1] int64
        sp_offsets,         # [n_trees + 1] int64
        lg_offsets,         # [n_trees + 1] int64
        sp_tour_widths,     # [n_trees] int32
        tree_to_group_idx,  # [n_trees] int32 - maps tree to group index
        polytomy_offsets,   # [n_trees + 1] int32 - CSR offsets for polytomy nodes
        polytomy_nodes,     # [total_polytomy] int32 - local node IDs of polytomy internals
        tree_multiplicities,  # [n_trees] int32 - deduplication weights
        bl_variant_offsets,       # [n_trees + 1] int32 - CSR into bl_variant_multiplicities
        bl_variant_multiplicities,  # [total_variants] int32 - per-variant counts
        bl_node_offsets,          # [total_variants + 1] int32 - CSR into all_rd_variants
        all_rd_variants,          # [total_variant_nodes] float64 - per-variant root distances
        # Outputs
        counts,             # [count, n_groups, 4] int32
        steiner_out,        # [count, n_groups, 4] float64 — summed Steiner
        steiner_min_out,    # [count, n_groups, 4] float64 — min Steiner (init +inf)
        steiner_max_out,    # [count, n_groups, 4] float64 — max Steiner (init -inf)
        steiner_sum_sq_out  # [count, n_groups, 4] float64 — sum of squared Steiner (init 0)
    ):
        """
        Unified kernel with Steiner distances.

        Same as ``quartet_counts_cuda_unified`` plus Steiner spanning-length
        accumulation per group.  All outputs are updated atomically; the host
        must pre-initialise them (counts/steiner_out/steiner_sum_sq_out to 0,
        steiner_min_out to +inf, steiner_max_out to -inf).
        """
        qi, ti = cuda.grid(2)
        n_trees = node_offsets.shape[0] - 1
        if qi >= count or ti >= n_trees:
            return

        # Determine absolute index and get quartet
        absolute_idx = offset + qi
        a, b, c, d = get_quartet_at_index(
            absolute_idx, seed_quartets, n_seed, rng_seed, n_global_taxa
        )

        # Resolve global IDs to tree-local positions
        ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
            _resolve_quartet_cuda(
                a, b, c, d, ti,
                global_to_local, node_offsets, tour_offsets, sp_offsets,
                lg_offsets, sp_tour_widths,
            )
        # Skip if any taxon absent — steiner_out is pre-initialised to 0 by host.
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        fo0 = all_first_occ[node_base + ln0]
        fo1 = all_first_occ[node_base + ln1]
        fo2 = all_first_occ[node_base + ln2]
        fo3 = all_first_occ[node_base + ln3]

        # CSR-based polytomy detection (zero overhead for trees without polytomies)
        poly_start = polytomy_offsets[ti]
        poly_end = polytomy_offsets[ti + 1]
        found, topo, r0, r1, r2, rw = _polytomy_check_cuda(
            fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
            poly_start, poly_end, polytomy_nodes,
            all_sparse_table, all_euler_depth, all_log2_table,
            all_euler_tour, all_root_distance,
        )
        mult = tree_multiplicities[ti]
        if not found:
            topo, r0, r1, r2, rw = _quartet_topology_and_rd_cuda(
                fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
            )

        gi = tree_to_group_idx[ti]
        cuda.atomic.add(counts, (qi, gi, topo), mult)

        # Topology is fixed for all BL variants; compute LCA nodes once, then
        # loop over BL variants for variant-specific Steiner lengths.
        lca01, lca23, lca02, lca13, lca03, lca12 = _compute_lca_nodes_cuda(
            fo0, fo1, fo2, fo3, tour_base, sp_base, lg_base, sp_stride,
            all_sparse_table, all_euler_depth, all_log2_table, all_euler_tour,
        )
        v_start = bl_variant_offsets[ti]
        v_end   = bl_variant_offsets[ti + 1]
        for vi in range(v_start, v_end):
            mult_v = bl_variant_multiplicities[vi]
            rd_vbase = bl_node_offsets[vi]
            sl_v = _steiner_from_lca_nodes_cuda(
                ln0, ln1, ln2, ln3,
                lca01, lca23, lca02, lca13, lca03, lca12,
                rd_vbase, all_rd_variants,
            )
            _accumulate_steiner_cuda(
                qi, gi, topo, sl_v, mult_v,
                steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
            )


    @cuda.jit
    def _quartet_counts_delta_cuda(
        delta_quartet_ids,          # int32[n_affected_quartets, 4]
        delta_quartet_global_idx,   # int32[n_affected_quartets]
        n_affected_quartets,        # int
        old_global_to_local,        # int32[n_trees, n_global_taxa]
        new_global_to_local,        # int32[n_trees, n_global_taxa]
        delta_tree_ids,             # int32[n_affected_trees]
        n_affected_trees,           # int
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
        tree_multiplicities,        # int32[n_trees]
        counts_out,                 # int32[n_quartets, n_groups, 4]
    ):
        """
        CUDA delta kernel: incremental ±mult updates to ``counts_out`` after a
        paralog copy-slot permutation.

        Grid is 2D: x over ``qi_local`` (affected quartet index, 0 …
        n_affected_quartets-1), y over ``t_idx`` (affected tree position, 0 …
        n_affected_trees-1).  Multiple threads sharing the same ``qi`` can
        write to the same ``counts_out[qi, gi, :]`` bin, so all updates use
        ``cuda.atomic.add``.

        For each (qi_local, t_idx) thread the kernel:

        1. Resolves quartet taxa to tree-local positions under the *old*
           copy-slot mapping.  Skips if any taxon is absent.
        2. Determines the old topology (with polytomy detection).
        3. Resolves the same quartet under the *new* mapping.  Skips if any
           taxon is absent.
        4. Determines the new topology.
        5. If topology changed, atomically decrements ``counts_out`` at the
           old topology and increments at the new one.

        ``old_global_to_local`` and ``new_global_to_local`` are small
        per-call arrays uploaded before each kernel launch.  When
        ``ParalogOptimizer`` is constructed with ``backend='cuda'`` it keeps
        ``counts_out`` device-resident and calls the kernel directly,
        bypassing the H→D/D→H round-trip that ``apply_quartet_counts_delta``
        would otherwise perform.  The structural forest arrays
        (``all_first_occ``, ``node_offsets``, etc.) are always taken from
        the pre-uploaded ``_cuda_kernel_data`` and are never re-uploaded.

        Parameters
        ----------
        delta_quartet_ids : int32[n_affected_quartets, 4]
            Sorted global taxon IDs for each affected quartet.
        delta_quartet_global_idx : int32[n_affected_quartets]
            Row index into ``counts_out`` for each affected quartet.
        n_affected_quartets : int
        old_global_to_local : int32[n_trees, n_global_taxa]
            Copy-slot → local-leaf mapping before the permutation.
        new_global_to_local : int32[n_trees, n_global_taxa]
            Copy-slot → local-leaf mapping after the permutation.
        delta_tree_ids : int32[n_affected_trees]
            Indices of trees where ≥ 2 copies of the permuted genome are present.
        n_affected_trees : int
        counts_out : int32[n_quartets, n_groups, 4]
            Modified atomically in-place.
        """
        qi_local, t_idx = cuda.grid(2)
        if qi_local >= n_affected_quartets or t_idx >= n_affected_trees:
            return

        t0 = delta_quartet_ids[qi_local, 0]
        t1 = delta_quartet_ids[qi_local, 1]
        t2 = delta_quartet_ids[qi_local, 2]
        t3 = delta_quartet_ids[qi_local, 3]
        qi = delta_quartet_global_idx[qi_local]
        ti = delta_tree_ids[t_idx]
        gi = tree_to_group_idx[ti]

        # ── Old assignment ───────────────────────────────────────────────── #
        ln0_old, ln1_old, ln2_old, ln3_old, \
            node_base, tour_base, sp_base, lg_base, sp_stride = \
            _resolve_quartet_cuda(
                t0, t1, t2, t3, ti,
                old_global_to_local, node_offsets, tour_offsets,
                sp_offsets, lg_offsets, sp_tour_widths,
            )
        if ln0_old < 0 or ln1_old < 0 or ln2_old < 0 or ln3_old < 0:
            return

        fo0 = all_first_occ[node_base + ln0_old]
        fo1 = all_first_occ[node_base + ln1_old]
        fo2 = all_first_occ[node_base + ln2_old]
        fo3 = all_first_occ[node_base + ln3_old]
        poly_start = polytomy_offsets[ti]
        poly_end   = polytomy_offsets[ti + 1]

        found, old_topo, r0, r1, r2, rw = _polytomy_check_cuda(
            fo0, fo1, fo2, fo3,
            node_base, tour_base, sp_base, lg_base, sp_stride,
            poly_start, poly_end, polytomy_nodes,
            all_sparse_table, all_euler_depth, all_log2_table,
            all_euler_tour, all_root_distance,
        )
        if not found:
            old_topo, r0, r1, r2, rw = _quartet_topology_and_rd_cuda(
                fo0, fo1, fo2, fo3,
                node_base, tour_base, sp_base, lg_base, sp_stride,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
            )

        # ── New assignment ───────────────────────────────────────────────── #
        ln0_new = new_global_to_local[ti, t0]
        ln1_new = new_global_to_local[ti, t1]
        ln2_new = new_global_to_local[ti, t2]
        ln3_new = new_global_to_local[ti, t3]
        if ln0_new < 0 or ln1_new < 0 or ln2_new < 0 or ln3_new < 0:
            return

        fo0 = all_first_occ[node_base + ln0_new]
        fo1 = all_first_occ[node_base + ln1_new]
        fo2 = all_first_occ[node_base + ln2_new]
        fo3 = all_first_occ[node_base + ln3_new]

        found, new_topo, r0, r1, r2, rw = _polytomy_check_cuda(
            fo0, fo1, fo2, fo3,
            node_base, tour_base, sp_base, lg_base, sp_stride,
            poly_start, poly_end, polytomy_nodes,
            all_sparse_table, all_euler_depth, all_log2_table,
            all_euler_tour, all_root_distance,
        )
        if not found:
            new_topo, r0, r1, r2, rw = _quartet_topology_and_rd_cuda(
                fo0, fo1, fo2, fo3,
                node_base, tour_base, sp_base, lg_base, sp_stride,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
            )

        # ── Apply signed delta ───────────────────────────────────────────── #
        if old_topo != new_topo:
            mult = tree_multiplicities[ti]
            cuda.atomic.add(counts_out, (qi, gi, old_topo), -mult)
            cuda.atomic.add(counts_out, (qi, gi, new_topo), mult)

    @cuda.jit
    def _counts_d2d_copy_cuda(src, dst):
        """
        Device-to-device copy of a 3-D int32 array.

        Grid is 3D: x over axis 0 (n_quartets), y over axis 1 (n_groups),
        z over axis 2 (4 topology slots).  Used by the device-aware
        ``ParalogOptimizer`` to create a trial copy of ``_d_counts`` without
        a round-trip through host memory.
        """
        i, j, k = cuda.grid(3)
        if i < dst.shape[0] and j < dst.shape[1] and k < dst.shape[2]:
            dst[i, j, k] = src[i, j, k]


def _compute_cuda_grid(n_quartets, n_trees, threads_per_block=(16, 16)):
    """
    Compute CUDA grid dimensions for the 2D (qi, ti) thread space.

    Parameters
    ----------
    n_quartets : int
        Number of quartets to process.
    n_trees : int
        Number of trees in collection.
    threads_per_block : tuple[int, int], default (16, 16)
        Block dimensions (x, y). Total threads per block = x * y.
        Default 16×16 = 256 threads per block (good for most GPUs).

    Returns
    -------
    blocks_per_grid : tuple[int, int]
        Grid dimensions (x, y) in blocks.
    threads_per_block : tuple[int, int]
        Block dimensions (x, y) in threads (echoed back).
        
    Examples
    --------
    >>> _compute_cuda_grid(1000, 50)
    ((63, 4), (16, 16))
    
    This creates a 63×4 grid of 16×16 blocks, giving:
    - 63 * 16 = 1008 threads in x (covers 1000 quartets)
    - 4 * 16 = 64 threads in y (covers 50 trees)
    """
    tpb_x, tpb_y = threads_per_block
    blocks_x = (n_quartets + tpb_x - 1) // tpb_x
    blocks_y = (n_trees + tpb_y - 1) // tpb_y
    return (blocks_x, blocks_y), threads_per_block
