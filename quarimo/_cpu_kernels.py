"""
_cpu_kernels.py
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

_resolve_quartet_nb : njit function
    Map four global taxon IDs to per-tree local positions (helper).

_quartet_topology_and_rd_nb : njit function
    Six LCA calls + four-point condition → topology and pair-sums (helper).

_steiner_length_nb : njit function
    Steiner spanning length of the winning quartet topology (helper).

_quartet_counts_njit : njit function
    Parallel quartet topology counts (no Steiner distances).

_quartet_steiner_njit : njit function
    Parallel quartet topology counts with Steiner distances.

_qed_njit : njit function
    Parallel QED computation over all quartet × group-pair combinations.

Notes
-----
- Helper functions (_rmq_csr_nb, _resolve_quartet_nb, etc.) are inlined by the
  Numba JIT compiler; there is no function-call overhead at runtime.
- Parallel kernels use prange; when numba is unavailable prange falls back to range.
- cache=True persists compiled binaries to disk for faster subsequent runs.
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
    O(1) RMQ over CSR-packed sparse table for a single tree.

    Parameters
    ----------
    l, r : int
        Inclusive local tour range (l <= r).
    sp_base : int
        Offset of this tree's sparse table in all_sparse_table.
    sp_stride : int
        Column stride (= tour length for this tree).
    sparse_table : int32[:]
        CSR-packed sparse table (min-position per range/level).
    euler_depth : int32[:]
        CSR-packed Euler-tour depths.
    log2_table : int32[:]
        CSR-packed floor-log2 lookup (indexed by range length).
    lg_base : int
        Offset of this tree's log2_table slice.
    tour_base : int
        Offset of this tree's Euler-tour slice.
    euler_tour : int32[:]
        CSR-packed Euler tour (local node IDs).

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


@njit(cache=True)
def _quartet_topology_and_rd_nb(occ0, occ1, occ2, occ3,
                                 nb, tb, sb, lb, tw,
                                 all_root_distance,
                                 all_sparse_table, all_euler_depth,
                                 all_log2_table, all_euler_tour):
    """
    Six RMQ calls + four-point condition → topology and pair-sums.

    Computes all six pairwise LCA root-distances for four taxa (whose first
    Euler-tour occurrences are ``occ0..occ3``), applies the four-point
    condition to determine the winning unrooted split, and returns the topology
    index alongside all three pair-sums and the winning pair-sum.

    Parameters
    ----------
    occ0..occ3 : int
        First Euler-tour occurrences for the four taxa in tree *ti*.
        All must be valid — the caller has already checked ``ln0..ln3 >= 0``
        before computing these.
    nb, tb, sb, lb, tw : int
        CSR offsets and sparse-table stride for tree *ti*
        (from ``_resolve_quartet_nb``).
    all_root_distance : float64[:]
        CSR-packed root distances.
    all_sparse_table, all_euler_depth, all_log2_table, all_euler_tour
        CSR-packed RMQ arrays.

    Returns
    -------
    topo : int
        Winning topology index: 0 = (n0,n1)|(n2,n3),
                                1 = (n0,n2)|(n1,n3),
                                2 = (n0,n3)|(n1,n2).
    r0, r1, r2 : float64
        Pair-sums for each of the three topologies.
    r_winner : float64
        Score of the winning topology (max of r0, r1, r2).

    Notes
    -----
    Uses ``>=`` comparisons so polytomies (tied scores) are handled
    consistently.
    """
    l = occ0; r = occ1
    if l > r: l, r = r, l
    rd01 = all_root_distance[nb + _rmq_csr_nb(l, r, sb, tw, all_sparse_table,
                                               all_euler_depth, all_log2_table,
                                               lb, tb, all_euler_tour)]
    l = occ0; r = occ2
    if l > r: l, r = r, l
    rd02 = all_root_distance[nb + _rmq_csr_nb(l, r, sb, tw, all_sparse_table,
                                               all_euler_depth, all_log2_table,
                                               lb, tb, all_euler_tour)]
    l = occ0; r = occ3
    if l > r: l, r = r, l
    rd03 = all_root_distance[nb + _rmq_csr_nb(l, r, sb, tw, all_sparse_table,
                                               all_euler_depth, all_log2_table,
                                               lb, tb, all_euler_tour)]
    l = occ1; r = occ2
    if l > r: l, r = r, l
    rd12 = all_root_distance[nb + _rmq_csr_nb(l, r, sb, tw, all_sparse_table,
                                               all_euler_depth, all_log2_table,
                                               lb, tb, all_euler_tour)]
    l = occ1; r = occ3
    if l > r: l, r = r, l
    rd13 = all_root_distance[nb + _rmq_csr_nb(l, r, sb, tw, all_sparse_table,
                                               all_euler_depth, all_log2_table,
                                               lb, tb, all_euler_tour)]
    l = occ2; r = occ3
    if l > r: l, r = r, l
    rd23 = all_root_distance[nb + _rmq_csr_nb(l, r, sb, tw, all_sparse_table,
                                               all_euler_depth, all_log2_table,
                                               lb, tb, all_euler_tour)]

    r0 = rd01 + rd23  # topology 0: (n0,n1)|(n2,n3)
    r1 = rd02 + rd13  # topology 1: (n0,n2)|(n1,n3)
    r2 = rd03 + rd12  # topology 2: (n0,n3)|(n1,n2)

    if r0 >= r1 and r0 >= r2:
        topo = 0; r_winner = r0
    elif r1 >= r0 and r1 >= r2:
        topo = 1; r_winner = r1
    else:
        topo = 2; r_winner = r2

    return topo, r0, r1, r2, r_winner


@njit(cache=True)
def _steiner_length_nb(ln0, ln1, ln2, ln3, nb, r0, r1, r2, r_winner, all_root_distance):
    """
    Steiner spanning length of the winning quartet topology.

    Given the four local leaf IDs and the three pair-sums already computed by
    ``_quartet_topology_and_rd_nb``, returns the Steiner spanning length of the
    minimal subtree connecting the four taxa.

    Parameters
    ----------
    ln0..ln3 : int
        Local leaf IDs in tree *ti* (all must be >= 0; caller has checked).
    nb : int
        Node-array CSR offset for tree *ti*.
    r0, r1, r2 : float64
        Pair-sums for each of the three topologies (from
        ``_quartet_topology_and_rd_nb``).
    r_winner : float64
        Score of the winning topology (max of r0, r1, r2).
    all_root_distance : float64[:]
        CSR-packed root distances.

    Returns
    -------
    float64
        Steiner spanning length S >= 0.

    Notes
    -----
    Formula: S = Σ rd(leaf_i) − 0.5 * (r_winner + r0 + r1 + r2)
    """
    leaf_sum = (all_root_distance[nb + ln0]
              + all_root_distance[nb + ln1]
              + all_root_distance[nb + ln2]
              + all_root_distance[nb + ln3])
    return leaf_sum - (r_winner + r0 + r1 + r2) * 0.5


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

    Outer loop over qi runs in parallel via prange; inner loop over ti is
    sequential.  No atomics needed: each prange thread owns its entire
    counts_out[qi, :, :] slice.

    Parameters
    ----------
    sorted_quartet_ids : int32[n_quartets, 4]
        Quartet taxa as sorted global IDs.
    global_to_local : int32[n_trees, n_global_taxa]
        Global taxon ID → local leaf ID (-1 if absent).
    all_first_occ, all_root_distance, all_euler_tour, all_euler_depth,
    all_sparse_table, all_log2_table : CSR-packed tree arrays
        See Forest class docs for layout details.
    node_offsets, tour_offsets, sp_offsets, lg_offsets : int64[n_trees+1]
        CSR offset vectors into the corresponding flat arrays.
    sp_tour_widths : int32[n_trees]
        Sparse-table column stride for each tree.
    n_quartets, n_trees : int
    tree_to_group_idx : int32[n_trees]
        Maps each tree index to its group index.
    counts_out : int32[n_quartets, n_groups, 3]
        Accumulated in-place; pre-filled with zeros by caller.
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

            occ0 = all_first_occ[nb + ln0]
            occ1 = all_first_occ[nb + ln1]
            occ2 = all_first_occ[nb + ln2]
            occ3 = all_first_occ[nb + ln3]
            topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_nb(
                occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
            )
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

            occ0 = all_first_occ[nb + ln0]
            occ1 = all_first_occ[nb + ln1]
            occ2 = all_first_occ[nb + ln2]
            occ3 = all_first_occ[nb + ln3]
            topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_nb(
                occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
            )
            gi = tree_to_group_idx[ti]
            counts_out[qi, gi, topo] += 1
            steiner_out[qi, gi, topo] += _steiner_length_nb(
                ln0, ln1, ln2, ln3, nb, r0, r1, r2, r_winner, all_root_distance,
            )


# ======================================================================== #
# QED Kernel                                                                #
# ======================================================================== #

@njit(parallel=True, cache=True)
def _qed_njit(counts, pair_indices, n_quartets, n_pairs, out):
    """
    Compute the Quartet Ensemble Discordance (QED) for all quartet × group-pair combinations.

    QED is an entropy-like similarity score in [-1, +1] that measures how
    consistently two groups of trees agree on quartet topology.

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
