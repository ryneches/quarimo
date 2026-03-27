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
    Six LCA calls + four-point condition → topology, pair-sums, and LCA IDs (helper).

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
def _resolve_quartet_nb(t0, t1, t2, t3, ti,
                        global_to_local,
                        node_offsets, tour_offsets, sp_offsets, lg_offsets,
                        sp_tour_widths):
    """
    Map four global taxon IDs to tree-local positions for tree *ti*.

    Returns a 9-tuple
    ``(ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride)``
    where:

    ln0..ln3 : int32
        Local leaf IDs in tree *ti* (-1 if the taxon is absent).
    node_base : int64
        Node-array CSR offset for tree *ti*.
    tour_base : int64
        Euler-tour CSR offset for tree *ti*.
    sp_base : int64
        Sparse-table CSR offset for tree *ti*.
    lg_base : int64
        Log2-table CSR offset for tree *ti*.
    sp_stride : int32
        Sparse-table column stride (= tour length) for tree *ti*.

    The CSR offsets are always valid; they do not depend on taxon presence.
    The caller is responsible for checking::

        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            continue  # skip this (qi, ti) pair

    before accessing ``all_first_occ[node_base + ln0]`` etc.
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


@njit(cache=True)
def _quartet_topology_and_rd_nb(fo0, fo1, fo2, fo3,
                                 node_base, tour_base, sp_base, lg_base, sp_stride,
                                 all_root_distance,
                                 all_sparse_table, all_euler_depth,
                                 all_log2_table, all_euler_tour,
                                 poly_start, poly_end, polytomy_nodes):
    """
    Six RMQ calls + four-point condition → topology and pair-sums.

    Computes all six pairwise LCA node IDs for four taxa, checks whether any
    is a polytomy-inserted internal node (via the CSR ``polytomy_nodes`` list),
    and if so returns topo=3 immediately.  Otherwise applies the four-point
    condition on LCA root-distances to determine the winning unrooted split.

    Parameters
    ----------
    fo0..fo3 : int
        First Euler-tour occurrences for the four taxa in tree *ti*.
        All must be valid — the caller has already checked ``ln0..ln3 >= 0``
        before computing these.
    node_base, tour_base, sp_base, lg_base, sp_stride : int
        CSR offsets and sparse-table stride for tree *ti*
        (from ``_resolve_quartet_nb``).
    all_root_distance : float64[:]
        CSR-packed root distances.
    all_sparse_table, all_euler_depth, all_log2_table, all_euler_tour
        CSR-packed RMQ arrays.
    poly_start, poly_end : int
        Slice bounds into ``polytomy_nodes`` for tree *ti*
        (``polytomy_offsets[ti]`` and ``polytomy_offsets[ti+1]``).
        When ``poly_start == poly_end`` the tree has no polytomy nodes and the
        CSR check is skipped entirely (zero overhead for the common case).
    polytomy_nodes : int32[:]
        Flat CSR array of local polytomy-node IDs across all trees.

    Returns
    -------
    topo : int
        Winning topology index: 0 = (t0,t1)|(t2,t3),
                                1 = (t0,t2)|(t1,t3),
                                2 = (t0,t3)|(t1,t2),
                                3 = unresolved (polytomy-inserted LCA detected).
    r0, r1, r2 : float64
        Pair-sums for each of the three topologies; all 0.0 when topo==3.
    r_winner : float64
        Score of the winning topology; 0.0 when topo==3.
    lca01, lca23, lca02, lca13, lca03, lca12 : int
        Local node IDs of the six pairwise LCAs.
    """
    l = fo0; r = fo1
    if l > r: l, r = r, l
    lca01 = _rmq_csr_nb(l, r, sp_base, sp_stride, all_sparse_table,
                         all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
    l = fo0; r = fo2
    if l > r: l, r = r, l
    lca02 = _rmq_csr_nb(l, r, sp_base, sp_stride, all_sparse_table,
                         all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
    l = fo0; r = fo3
    if l > r: l, r = r, l
    lca03 = _rmq_csr_nb(l, r, sp_base, sp_stride, all_sparse_table,
                         all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
    l = fo1; r = fo2
    if l > r: l, r = r, l
    lca12 = _rmq_csr_nb(l, r, sp_base, sp_stride, all_sparse_table,
                         all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
    l = fo1; r = fo3
    if l > r: l, r = r, l
    lca13 = _rmq_csr_nb(l, r, sp_base, sp_stride, all_sparse_table,
                         all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)
    l = fo2; r = fo3
    if l > r: l, r = r, l
    lca23 = _rmq_csr_nb(l, r, sp_base, sp_stride, all_sparse_table,
                         all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour)

    rd01 = all_root_distance[node_base + lca01]
    rd02 = all_root_distance[node_base + lca02]
    rd03 = all_root_distance[node_base + lca03]
    rd12 = all_root_distance[node_base + lca12]
    rd13 = all_root_distance[node_base + lca13]
    rd23 = all_root_distance[node_base + lca23]

    r0 = rd01 + rd23  # topology 0: (n0,n1)|(n2,n3)
    r1 = rd02 + rd13  # topology 1: (n0,n2)|(n1,n3)
    r2 = rd03 + rd12  # topology 2: (n0,n3)|(n1,n2)

    # CSR-based polytomy detection (zero overhead for trees without polytomies).
    # The CSR check is the fast pre-filter: only trees with polytomy-inserted
    # nodes enter the loop.  The numerical tie-check (r0==r1==r2) is safe here
    # because it is restricted to trees where a polytomy node IS an LCA, so
    # any tie is caused by the zero-length sentinel branch (exact IEEE-754
    # equality), not by floating-point coincidence in real branch lengths.
    if poly_end > poly_start:
        for j in range(poly_start, poly_end):
            pn = polytomy_nodes[j]
            if (pn == lca01 or pn == lca02 or pn == lca03
                    or pn == lca12 or pn == lca13 or pn == lca23):
                if r0 == r1 and r1 == r2:
                    return np.int32(3), r0, r1, r2, r0, lca01, lca23, lca02, lca13, lca03, lca12
                break  # polytomy node found but no tie: fall through to resolved

    if r0 >= r1 and r0 >= r2:
        topo = np.int32(0); r_winner = r0
    elif r1 >= r0 and r1 >= r2:
        topo = np.int32(1); r_winner = r1
    else:
        topo = np.int32(2); r_winner = r2

    return topo, r0, r1, r2, r_winner, lca01, lca23, lca02, lca13, lca03, lca12


@njit(cache=True)
def _accumulate_steiner_nb(qi, gi, topo, sl, mult,
                            steiner_out, steiner_min_out,
                            steiner_max_out, steiner_sum_sq_out):
    """
    Accumulate one Steiner observation into the four per-cell stat arrays.

    CPU counterpart of ``_accumulate_steiner_cuda``.  Inlined by Numba; no
    function-call overhead at runtime.  Counts are intentionally excluded —
    the caller increments ``counts_out`` unconditionally before deciding
    whether to compute a Steiner length.

    Parameters
    ----------
    qi, gi, topo : int
        Output cell indices.
    sl : float64
        Steiner spanning length for this (tree, quartet) observation.
    mult : int32
        Tree multiplicity weight.  ``steiner_out`` and ``steiner_sum_sq_out``
        are scaled by ``mult``; min/max are not (identical duplicate trees
        have the same Steiner length).
    steiner_out : float64[n_quartets, n_groups, 4]
        Accumulates the weighted sum of Steiner lengths.
    steiner_min_out : float64[n_quartets, n_groups, 4]
        Accumulates the per-cell minimum; pre-filled with +inf by caller.
    steiner_max_out : float64[n_quartets, n_groups, 4]
        Accumulates the per-cell maximum; pre-filled with -inf by caller.
    steiner_sum_sq_out : float64[n_quartets, n_groups, 4]
        Accumulates the weighted sum of squared lengths (used for variance).
    """
    steiner_out[qi, gi, topo] += sl * mult
    if sl < steiner_min_out[qi, gi, topo]:
        steiner_min_out[qi, gi, topo] = sl
    if sl > steiner_max_out[qi, gi, topo]:
        steiner_max_out[qi, gi, topo] = sl
    steiner_sum_sq_out[qi, gi, topo] += sl * sl * mult


@njit(cache=True)
def _steiner_from_lca_nodes_nb(
        ln0, ln1, ln2, ln3,
        lca01, lca23, lca02, lca13, lca03, lca12,
        rd_vbase, all_rd_variants):
    """
    Steiner length for one BL variant given precomputed LCA local node IDs.

    Uses variant-specific root distances from ``all_rd_variants[rd_vbase:]``.
    The LCA node IDs are topology-stable (shared across all BL variants of the
    same stored tree); only the rd values differ between variants.

    Parameters
    ----------
    ln0..ln3 : int
        Local leaf node IDs of the four taxa.
    lca01, lca23, lca02, lca13, lca03, lca12 : int
        Pairwise LCA local node IDs.
    rd_vbase : int
        Offset into ``all_rd_variants`` for this BL variant's node data.
    all_rd_variants : float64[:]
        Flat array of per-variant root distances.

    Returns
    -------
    float64
        Steiner spanning length S >= 0.
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
        polytomy_offsets,
        polytomy_nodes,
        tree_multiplicities,
        bl_variant_offsets,
        bl_variant_multiplicities,
        bl_node_offsets,
        all_rd_variants,
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
    counts_out : int32[n_quartets, n_groups, 4]
        Accumulated in-place; pre-filled with zeros by caller.
        Last axis: k=0,1,2 resolved topologies; k=3 unresolved (polytomy).
    """
    for qi in prange(n_quartets):
        t0 = sorted_quartet_ids[qi, 0]
        t1 = sorted_quartet_ids[qi, 1]
        t2 = sorted_quartet_ids[qi, 2]
        t3 = sorted_quartet_ids[qi, 3]

        for ti in range(n_trees):
            ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
                _resolve_quartet_nb(
                    t0, t1, t2, t3, ti,
                    global_to_local, node_offsets, tour_offsets, sp_offsets,
                    lg_offsets, sp_tour_widths,
                )
            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue

            fo0 = all_first_occ[node_base + ln0]
            fo1 = all_first_occ[node_base + ln1]
            fo2 = all_first_occ[node_base + ln2]
            fo3 = all_first_occ[node_base + ln3]
            poly_start = polytomy_offsets[ti]
            poly_end = polytomy_offsets[ti + 1]
            topo, r0, r1, r2, r_winner, lca01, lca23, lca02, lca13, lca03, lca12 = \
                _quartet_topology_and_rd_nb(
                    fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
                    all_root_distance, all_sparse_table, all_euler_depth,
                    all_log2_table, all_euler_tour,
                    poly_start, poly_end, polytomy_nodes,
                )
            counts_out[qi, tree_to_group_idx[ti], topo] += tree_multiplicities[ti]


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
        polytomy_offsets,
        polytomy_nodes,
        tree_multiplicities,
        bl_variant_offsets,
        bl_variant_multiplicities,
        bl_node_offsets,
        all_rd_variants,
        counts_out,
        steiner_out,
        steiner_min_out,
        steiner_max_out,
        steiner_sum_sq_out):
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

    steiner_out : float64[n_quartets, n_groups, 4]
        Output array for summed Steiner distances per group per topology.
        Last axis: k=0,1,2 resolved topologies; k=3 unresolved (polytomy).

    steiner_min_out : float64[n_quartets, n_groups, 4]
        Output array for per-cell minimum Steiner distance.
        Pre-filled with +inf by caller; cells with count=0 keep +inf.

    steiner_max_out : float64[n_quartets, n_groups, 4]
        Output array for per-cell maximum Steiner distance.
        Pre-filled with -inf by caller; cells with count=0 keep -inf.

    steiner_sum_sq_out : float64[n_quartets, n_groups, 4]
        Output array for the sum of squared Steiner distances per cell.
        Used to compute variance = sum_sq/n - (sum/n)^2 after the kernel.
        Pre-filled with zeros by caller.
    """
    for qi in prange(n_quartets):
        t0 = sorted_quartet_ids[qi, 0]
        t1 = sorted_quartet_ids[qi, 1]
        t2 = sorted_quartet_ids[qi, 2]
        t3 = sorted_quartet_ids[qi, 3]

        for ti in range(n_trees):
            ln0, ln1, ln2, ln3, node_base, tour_base, sp_base, lg_base, sp_stride = \
                _resolve_quartet_nb(
                    t0, t1, t2, t3, ti,
                    global_to_local, node_offsets, tour_offsets, sp_offsets,
                    lg_offsets, sp_tour_widths,
                )
            if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
                continue

            fo0 = all_first_occ[node_base + ln0]
            fo1 = all_first_occ[node_base + ln1]
            fo2 = all_first_occ[node_base + ln2]
            fo3 = all_first_occ[node_base + ln3]
            poly_start = polytomy_offsets[ti]
            poly_end = polytomy_offsets[ti + 1]
            topo, r0, r1, r2, r_winner, lca01, lca23, lca02, lca13, lca03, lca12 = \
                _quartet_topology_and_rd_nb(
                    fo0, fo1, fo2, fo3, node_base, tour_base, sp_base, lg_base, sp_stride,
                    all_root_distance, all_sparse_table, all_euler_depth,
                    all_log2_table, all_euler_tour,
                    poly_start, poly_end, polytomy_nodes,
                )
            gi = tree_to_group_idx[ti]
            mult = tree_multiplicities[ti]
            counts_out[qi, gi, topo] += mult

            v_start = bl_variant_offsets[ti]
            v_end   = bl_variant_offsets[ti + 1]
            for vi in range(v_start, v_end):
                mult_v = bl_variant_multiplicities[vi]
                rd_vbase = bl_node_offsets[vi]
                sl_v = _steiner_from_lca_nodes_nb(
                    ln0, ln1, ln2, ln3,
                    lca01, lca23, lca02, lca13, lca03, lca12,
                    rd_vbase, all_rd_variants,
                )
                _accumulate_steiner_nb(
                    qi, gi, topo, sl_v, mult_v,
                    steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
                )


# ======================================================================== #
# Paralog Delta Kernel                                                      #
# ======================================================================== #


@njit(parallel=True, cache=True)
def _quartet_counts_delta_nb(
        delta_quartet_ids,
        delta_quartet_global_idx,
        n_affected_quartets,
        old_global_to_local,
        new_global_to_local,
        delta_tree_ids,
        n_affected_trees,
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
        counts_out):
    """
    Incremental update to ``counts_out`` after a paralog copy-slot
    permutation.

    For each affected (quartet, tree) pair, computes the topology under both
    the old and new copy-slot assignments.  If the topology changed, applies
    a signed ±mult update to ``counts_out`` in-place, where ``mult`` is
    ``tree_multiplicities[ti]``.

    Thread safety
    -------------
    ``prange`` is over ``qi_local`` (0 … n_affected_quartets-1).  Every value
    in ``delta_quartet_global_idx`` is unique — each thread owns a distinct
    row of ``counts_out`` — so no atomic operations are needed.

    Parameters
    ----------
    delta_quartet_ids : int32[n_affected_quartets, 4]
        Sorted global taxon IDs (t0, t1, t2, t3) for each affected quartet.
    delta_quartet_global_idx : int32[n_affected_quartets]
        Row index into ``counts_out`` for each affected quartet.
    n_affected_quartets : int
    old_global_to_local : int32[n_trees, n_global_taxa]
        Copy-slot → local-leaf mapping before the permutation.
    new_global_to_local : int32[n_trees, n_global_taxa]
        Copy-slot → local-leaf mapping after the permutation.
    delta_tree_ids : int32[n_affected_trees]
        Indices of trees where ≥ 2 copies of the permuted genome are present
        (the only trees where topology can change).
    n_affected_trees : int
    all_first_occ, all_root_distance, all_euler_tour, all_euler_depth,
    all_sparse_table, all_log2_table : CSR-packed tree arrays
    node_offsets, tour_offsets, sp_offsets, lg_offsets : int64[n_trees+1]
    sp_tour_widths : int32[n_trees]
    tree_to_group_idx : int32[n_trees]
    polytomy_offsets : int32[n_trees+1]
    polytomy_nodes : int32[total_polytomy_nodes]
    counts_out : int32[n_quartets, n_groups, 4]
        Modified in-place.  k=0,1,2 = resolved topologies; k=3 = unresolved.
    """
    for qi_local in prange(n_affected_quartets):
        t0 = delta_quartet_ids[qi_local, 0]
        t1 = delta_quartet_ids[qi_local, 1]
        t2 = delta_quartet_ids[qi_local, 2]
        t3 = delta_quartet_ids[qi_local, 3]
        qi = delta_quartet_global_idx[qi_local]

        for t_idx in range(n_affected_trees):
            ti = delta_tree_ids[t_idx]
            gi = tree_to_group_idx[ti]

            # ---- Old assignment ----------------------------------------- #
            ln0_old, ln1_old, ln2_old, ln3_old, \
                node_base, tour_base, sp_base, lg_base, sp_stride = \
                _resolve_quartet_nb(
                    t0, t1, t2, t3, ti,
                    old_global_to_local, node_offsets, tour_offsets,
                    sp_offsets, lg_offsets, sp_tour_widths,
                )
            if ln0_old < 0 or ln1_old < 0 or ln2_old < 0 or ln3_old < 0:
                continue

            fo0_old = all_first_occ[node_base + ln0_old]
            fo1_old = all_first_occ[node_base + ln1_old]
            fo2_old = all_first_occ[node_base + ln2_old]
            fo3_old = all_first_occ[node_base + ln3_old]
            poly_start = polytomy_offsets[ti]
            poly_end   = polytomy_offsets[ti + 1]
            old_topo, _, _, _, _, _, _, _, _, _, _ = _quartet_topology_and_rd_nb(
                fo0_old, fo1_old, fo2_old, fo3_old,
                node_base, tour_base, sp_base, lg_base, sp_stride,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
                poly_start, poly_end, polytomy_nodes,
            )

            # ---- New assignment ----------------------------------------- #
            ln0_new = new_global_to_local[ti, t0]
            ln1_new = new_global_to_local[ti, t1]
            ln2_new = new_global_to_local[ti, t2]
            ln3_new = new_global_to_local[ti, t3]
            if ln0_new < 0 or ln1_new < 0 or ln2_new < 0 or ln3_new < 0:
                continue

            fo0_new = all_first_occ[node_base + ln0_new]
            fo1_new = all_first_occ[node_base + ln1_new]
            fo2_new = all_first_occ[node_base + ln2_new]
            fo3_new = all_first_occ[node_base + ln3_new]
            new_topo, _, _, _, _, _, _, _, _, _, _ = _quartet_topology_and_rd_nb(
                fo0_new, fo1_new, fo2_new, fo3_new,
                node_base, tour_base, sp_base, lg_base, sp_stride,
                all_root_distance, all_sparse_table, all_euler_depth,
                all_log2_table, all_euler_tour,
                poly_start, poly_end, polytomy_nodes,
            )

            # ---- Apply signed delta ------------------------------------- #
            if old_topo != new_topo:
                mult = tree_multiplicities[ti]
                counts_out[qi, gi, old_topo] -= mult
                counts_out[qi, gi, new_topo] += mult


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
