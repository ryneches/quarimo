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

_resolve_quartet_nb : njit function
    Map four global taxon IDs to per-tree local positions (helper).

_quartet_steiner_njit : njit function
    Parallel quartet topology counts with Steiner distances.

_qmetric_njit : njit function
    Parallel qmetric computation over all quartet × group-pair combinations.

Notes
-----
- All functions are decorated with @njit for numba compilation
- Functions use prange for parallel execution when numba is available
- When numba is unavailable, functions run as pure Python (slower but functional)
- cache=True persists compiled binary to disk for faster subsequent runs
"""

import math

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


@njit(cache=True)
def _resolve_quartet_nb(n0, n1, n2, n3, ti,
                        global_to_local,
                        node_offsets, tour_offsets, sp_offsets, lg_offsets,
                        sp_tour_widths):
    """
    Map four global taxon IDs to tree-local positions for tree *ti*.

    Returns a 9-tuple ``(ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw)`` where:

    ln0..ln3 : int32
        Local leaf IDs in tree *ti* (-1 if the taxon is absent).
    nb : int64
        Node-array CSR offset for tree *ti*.
    tb : int64
        Euler-tour CSR offset for tree *ti*.
    sb : int64
        Sparse-table CSR offset for tree *ti*.
    lb : int64
        Log2-table CSR offset for tree *ti*.
    tw : int32
        Sparse-table column stride (= tour length) for tree *ti*.

    The CSR offsets ``nb..tw`` are always valid; they do not depend on taxon
    presence.  The caller is responsible for checking::

        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            continue  # skip this (qi, ti) pair

    before accessing ``all_first_occ[nb + ln0]`` etc.

    This function is inlined by the Numba JIT compiler into every kernel that
    calls it, so there is no function-call overhead at runtime.
    """
    ln0 = global_to_local[ti, n0]
    ln1 = global_to_local[ti, n1]
    ln2 = global_to_local[ti, n2]
    ln3 = global_to_local[ti, n3]
    nb  = node_offsets[ti]
    tb  = tour_offsets[ti]
    sb  = sp_offsets[ti]
    lb  = lg_offsets[ti]
    tw  = sp_tour_widths[ti]
    return ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw


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
        tree_to_group_idx,
        counts_out):
    """
    Numba-compiled counts-only quartet kernel.

    The outer loop over qi runs in parallel via prange; the inner loop over
    ti is sequential. No atomics are needed: each parallel thread owns its
    entire counts_out[qi, :, :] slice.

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
    tree_to_group_idx : int32[n_trees]
        Maps each tree index to its group index.
    counts_out : int32[n_quartets, n_groups, 3]
        Output array for topology counts per group.
    """
    for qi in prange(n_quartets):
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]

        for ti in range(n_trees):
            ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw = _resolve_quartet_nb(
                n0, n1, n2, n3, ti,
                global_to_local, node_offsets, tour_offsets, sp_offsets,
                lg_offsets, sp_tour_widths,
            )
            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue

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

            if r0 >= r1 and r0 >= r2:
                topo = 0
            elif r1 >= r0 and r1 >= r2:
                topo = 1
            else:
                topo = 2

            counts_out[qi, tree_to_group_idx[ti], topo] += 1


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
        tree_to_group_idx,
        counts_out,
        steiner_out):
    """
    Numba-compiled quartet kernel with Steiner distances.

    Identical to _quartet_counts_njit plus the Steiner distance calculation.
    Kept as a separate function to avoid rank mismatch issues with sentinel
    arrays.

    counts_out[qi, gi, topo] and steiner_out[qi, gi, topo] accumulate
    per-group — conflict-free by qi (each prange thread owns its row).

    Parameters
    ----------
    Same as _quartet_counts_njit, plus:

    steiner_out : float64[n_quartets, n_groups, 3]
        Output array for summed Steiner distances per group per topology.
    """
    for qi in prange(n_quartets):
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]

        for ti in range(n_trees):
            ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw = _resolve_quartet_nb(
                n0, n1, n2, n3, ti,
                global_to_local, node_offsets, tour_offsets, sp_offsets,
                lg_offsets, sp_tour_widths,
            )
            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue

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

            if r0 >= r1 and r0 >= r2:
                topo = 0; r_winner = r0
            elif r1 >= r0 and r1 >= r2:
                topo = 1; r_winner = r1
            else:
                topo = 2; r_winner = r2

            gi = tree_to_group_idx[ti]
            counts_out[qi, gi, topo] += 1

            # Compute Steiner distance
            leaf_rd_sum = (all_root_distance[nb + ln0]
                         + all_root_distance[nb + ln1]
                         + all_root_distance[nb + ln2]
                         + all_root_distance[nb + ln3])
            S = leaf_rd_sum - (r_winner + r0 + r1 + r2) * 0.5
            steiner_out[qi, gi, topo] += S


# ======================================================================== #
# QMetric Kernel                                                            #
# ======================================================================== #

@njit(parallel=True, cache=True)
def _qmetric_njit(counts, pair_indices, n_quartets, n_pairs, out):
    """
    Compute the quartet qmetric for all quartet × group-pair combinations.

    The qmetric is an entropy-like similarity score in [-1, +1] that measures
    how consistently two groups of trees agree on the dominant quartet topology.

    Parameters
    ----------
    counts : int32[n_quartets, n_groups, 3]
        Topology count array from Forest.quartet_topology().
    pair_indices : int32[n_pairs, 2]
        Each row (g1, g2) is an ordered pair of group indices to compare.
    n_quartets : int
        Number of quartets.
    n_pairs : int
        Number of group pairs.
    out : float64[n_quartets, n_pairs]
        Output array (pre-allocated, written in place).

    Notes
    -----
    LOG3_INV = 1 / ln(3) converts natural-log entropy to base-3.

    For groups g1, g2 and quartet qi:
      - ca = counts[qi, g1, :], cb = counts[qi, g2, :]
      - Na = sum(ca), Nb = sum(cb)
      - If Na == 0 or Nb == 0: out[qi, pi] = 0.0  (undefined)
      - pa = ca / Na,  pb = cb / Nb
      - J = +1 if argmax(pa) == argmax(pb) else -1
      - q_k = pa_k * pb_k
      - l3(q) = q * log(q) * LOG3_INV  (0 when q == 0)
      - out[qi, pi] = J * (1 + l3(q_0) + l3(q_1) + l3(q_2))
    """
    LOG3_INV = 0.9102392266268374  # 1 / ln(3)
    for qi in prange(n_quartets):
        for pi in range(n_pairs):
            g1 = pair_indices[pi, 0]
            g2 = pair_indices[pi, 1]

            ca0 = counts[qi, g1, 0]
            ca1 = counts[qi, g1, 1]
            ca2 = counts[qi, g1, 2]
            cb0 = counts[qi, g2, 0]
            cb1 = counts[qi, g2, 1]
            cb2 = counts[qi, g2, 2]

            Na = ca0 + ca1 + ca2
            Nb = cb0 + cb1 + cb2

            if Na == 0 or Nb == 0:
                out[qi, pi] = 0.0
                continue

            pa0 = ca0 / Na
            pa1 = ca1 / Na
            pa2 = ca2 / Na
            pb0 = cb0 / Nb
            pb1 = cb1 / Nb
            pb2 = cb2 / Nb

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

            q0 = pa0 * pb0
            q1 = pa1 * pb1
            q2 = pa2 * pb2

            s = 0.0
            if q0 > 0.0:
                s += q0 * math.log(q0) * LOG3_INV
            if q1 > 0.0:
                s += q1 * math.log(q1) * LOG3_INV
            if q2 > 0.0:
                s += q2 * math.log(q2) * LOG3_INV

            out[qi, pi] = J * (1.0 + s)
