"""
_kernels_cpu.py
===============
CPU-accelerated quartet topology kernels using Numba.

This module contains ONLY numba-accelerated code and should not import other
project modules to avoid import-time complications. The module-level functions
are JIT-compiled by numba when available, or run as pure Python when numba is
not installed.

Exported Functions
------------------
_rmq_csr_nb : njit function
    O(1) range minimum query helper for CSR-packed arrays.

_quartet_counts_njit : njit function  
    Parallel quartet topology counts (no Steiner distances).

_quartet_steiner_njit : njit function
    Parallel quartet topology counts with Steiner distances.

Notes
-----
- All functions are decorated with @njit for numba compilation
- Functions use prange for parallel execution when numba is available
- When numba is unavailable, functions run as pure Python (slower but functional)
- cache=True persists compiled binary to disk for faster subsequent runs
"""

import numpy as np

# ── Optional numba acceleration ──────────────────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):
        """Identity decorator — stands in for numba.njit when numba is absent."""
        if args and callable(args[0]):   # @njit without arguments
            return args[0]
        return lambda fn: fn             # @njit(parallel=True, ...) with arguments
    
    prange = range  # serial fallback for prange


# ======================================================================== #
# CPU Kernels                                                               #
# ======================================================================== #


@njit(cache=True)
def _rmq_csr_nb(l, r,
                sp_base, sp_stride,
                sparse_table,
                euler_depth,
                log2_table,
                lg_base, tour_base,
                euler_tour):
    """
    O(1) RMQ over CSR-packed arrays for a single tree.

    This is a helper function inlined into the quartet kernels. All index
    arguments are absolute positions in the flat buffers.

    Parameters
    ----------
    l, r         : int  
        Inclusive local tour range (l <= r).
    sp_base      : int  
        Offset of this tree's sparse table in all_sparse_table.
    sp_stride    : int  
        sp_tour_widths[ti] — column stride.
    sparse_table : int32[:]  
        all_sparse_table.
    euler_depth  : int32[:]  
        all_euler_depth.
    log2_table   : int32[:]  
        all_log2_table.
    lg_base      : int  
        Offset of this tree's log2_table.
    tour_base    : int  
        Offset of this tree's tour in all_euler_tour/depth.
    euler_tour   : int32[:]  
        all_euler_tour.

    Returns
    -------
    int  
        Local node ID of the LCA.
    """
    length = r - l + 1
    k      = log2_table[lg_base + length]
    half   = 1 << k
    li     = sparse_table[sp_base + k * sp_stride + l]
    ri     = sparse_table[sp_base + k * sp_stride + (r - half + 1)]
    if euler_depth[tour_base + ri] < euler_depth[tour_base + li]:
        lca_local = ri
    else:
        lca_local = li
    return euler_tour[tour_base + lca_local]


@njit(parallel=True, cache=True)
def _quartet_counts_njit(
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
    Numba-compiled counts-only quartet kernel.

    The outer loop over qi runs in parallel via prange; the inner loop over
    ti is sequential. No atomics are needed: each parallel thread owns its
    entire counts_out[qi, :] row.

    Parameters
    ----------
    sorted_quartet_ids : int32[n_quartets, 4]
        Quartet taxa as sorted global IDs.
    global_to_local : int32[n_trees, n_global_taxa]
        Global ID → local leaf ID mapping (-1 if absent).
    all_first_occ : int32[total_nodes]
        CSR-packed first Euler tour occurrence for each node.
    all_root_distance : float64[total_nodes]
        CSR-packed root distances.
    all_euler_tour : int32[total_tour_length]
        CSR-packed Euler tour.
    all_euler_depth : int32[total_tour_length]
        CSR-packed Euler tour depths.
    all_sparse_table : int32[total_sparse_entries]
        CSR-packed sparse tables for RMQ.
    all_log2_table : int32[total_log2_entries]
        CSR-packed log2 lookup tables.
    node_offsets : int64[n_trees+1]
        CSR offsets into node arrays.
    tour_offsets : int64[n_trees+1]
        CSR offsets into tour arrays.
    sp_offsets : int64[n_trees+1]
        CSR offsets into sparse table.
    lg_offsets : int64[n_trees+1]
        CSR offsets into log2 table.
    sp_tour_widths : int32[n_trees]
        Sparse table column strides.
    n_quartets : int
        Number of quartets to process.
    n_trees : int
        Number of trees in collection.
    counts_out : int32[n_quartets, 3]
        Output array for topology counts.
    """
    for qi in prange(n_quartets):
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]

        for ti in range(n_trees):
            ln0 = global_to_local[ti, n0]
            ln1 = global_to_local[ti, n1]
            ln2 = global_to_local[ti, n2]
            ln3 = global_to_local[ti, n3]

            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue

            nb = node_offsets[ti]
            tb = tour_offsets[ti]
            sb = sp_offsets[ti]
            lb = lg_offsets[ti]
            tw = sp_tour_widths[ti]

            # Compute 6 pairwise LCAs
            l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln1]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd01 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln2]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd02 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln3]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd03 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln2]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd12 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln3]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd13 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln2]; r = all_first_occ[nb + ln3]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd23 = all_root_distance[nb + lca]

            # Determine winning topology
            r0 = rd01 + rd23  # (n0,n1)|(n2,n3)
            r1 = rd02 + rd13  # (n0,n2)|(n1,n3)
            r2 = rd03 + rd12  # (n0,n3)|(n1,n2)

            if r0 > r1:
                topo = 0 if r0 > r2 else 2
            else:
                topo = 1 if r1 > r2 else 2

            counts_out[qi, topo] += 1


@njit(parallel=True, cache=True)
def _quartet_steiner_njit(
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
    Numba-compiled quartet kernel with Steiner distances.

    Identical to _quartet_counts_njit plus the Steiner distance calculation.
    Kept as a separate function to avoid rank mismatch issues with sentinel
    arrays.

    steiner_out[qi, ti, topo] is a conflict-free write — each parallel thread
    owns its entire (qi, :, :) slice.

    Parameters
    ----------
    Same as _quartet_counts_njit, plus:
    
    steiner_out : float64[n_quartets, n_trees, 3]
        Output array for Steiner distances.
    """
    for qi in prange(n_quartets):
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]

        for ti in range(n_trees):
            ln0 = global_to_local[ti, n0]
            ln1 = global_to_local[ti, n1]
            ln2 = global_to_local[ti, n2]
            ln3 = global_to_local[ti, n3]

            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue

            nb = node_offsets[ti]
            tb = tour_offsets[ti]
            sb = sp_offsets[ti]
            lb = lg_offsets[ti]
            tw = sp_tour_widths[ti]

            # Compute 6 pairwise LCAs (identical to counts kernel)
            l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln1]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd01 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln2]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd02 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln3]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd03 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln2]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd12 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln3]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd13 = all_root_distance[nb + lca]

            l = all_first_occ[nb + ln2]; r = all_first_occ[nb + ln3]
            if l > r: l, r = r, l
            lca = _rmq_csr_nb(l, r, sb, tw, all_sparse_table, all_euler_depth,
                              all_log2_table, lb, tb, all_euler_tour)
            rd23 = all_root_distance[nb + lca]

            # Determine winning topology (track r_winner for Steiner)
            r0 = rd01 + rd23  # (n0,n1)|(n2,n3)
            r1 = rd02 + rd13  # (n0,n2)|(n1,n3)
            r2 = rd03 + rd12  # (n0,n3)|(n1,n2)

            if r0 > r1:
                if r0 > r2:
                    topo = 0; r_winner = r0
                else:
                    topo = 2; r_winner = r2
            else:
                if r1 > r2:
                    topo = 1; r_winner = r1
                else:
                    topo = 2; r_winner = r2

            counts_out[qi, topo] += 1

            # Compute Steiner distance
            leaf_rd_sum = (all_root_distance[nb + ln0]
                         + all_root_distance[nb + ln1]
                         + all_root_distance[nb + ln2]
                         + all_root_distance[nb + ln3])
            S = leaf_rd_sum - (r_winner + r0 + r1 + r2) * 0.5
            steiner_out[qi, ti, topo] = S
