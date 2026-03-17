"""
_mlx_kernels.py
===============
MLX Metal quartet generation and topology kernels for Apple Silicon.

Kernels
-------
``generate_quartets_mlx``
    1D generation kernel — direct equivalent of ``generate_quartets_cuda``.
    One thread per quartet; produces the deterministic XorShift128 sequence.

``quartet_counts_mlx``
    Full topology-counting kernel — counts how many trees in the forest
    support each of the three unrooted quartet topologies (k=0,1,2) or are
    unresolved (k=3) for each (quartet, group) pair.  Dispatches 1 thread
    per quartet; each thread loops sequentially over all trees (no atomics
    needed since each thread owns its entire output row).

``quartet_steiner_mlx``
    Same as ``quartet_counts_mlx`` plus accumulation of Steiner spanning
    distances per (quartet, group, topology) cell.  Because Metal does not
    support float64, root distances and Steiner outputs use float32; the
    Python wrapper converts back to float64 before returning.

Why Metal via MLX
-----------------
Apple Silicon has no CUDA support.  MLX (https://github.com/ml-explore/mlx)
provides Python-friendly access to Metal compute on M-series chips.  Because
Apple Silicon uses a Unified Memory Architecture (UMA) the ``mx.array(numpy)``
call does not copy memory — the CPU and GPU share the same physical pages —
so the "upload" cost that dominates small CUDA workloads essentially vanishes.

Type safety
-----------
All XorShift128 arithmetic in the MSL header uses native ``uint32_t`` /
``uint64_t``, eliminating the signed-vs-unsigned shift ambiguity that existed
in the Numba CUDA port (fixed in ``_rng_cuda.py``).  MSL is a C++14 dialect
with unambiguous C integer semantics.

Backend detection
-----------------
MLX availability is detected once per session by ``quarimo._backend.backends.mlx``
(a ``@cached_property`` on the ``_BackendCapabilities`` singleton).  This module
uses its own ``_MLX_AVAILABLE`` flag — set at import time — solely to guard
kernel definitions; external callers should use ``backends.mlx`` instead.
"""

from __future__ import annotations

import numpy as np

_MLX_AVAILABLE = False

try:
    import mlx.core as mx  # noqa: F401

    # Probe Metal availability: importing mlx succeeds even on machines without
    # Metal.  A trivial eval forces the Metal device to initialise.
    mx.eval(mx.array([0], dtype=mx.int32))
    _MLX_AVAILABLE = True
except Exception:
    pass


if _MLX_AVAILABLE:
    # ====================================================================== #
    # MSL helper functions                                                    #
    # ====================================================================== #
    # Must match _quartets.py::_init_rng / _sample_quartet and              #
    # _rng_cuda.py exactly.  Because MSL operands are native uint32_t the   #
    # & 0xFFFFFFFF masks used in the Python fallback are redundant — uint32  #
    # arithmetic wraps at 32 bits automatically.                             #

    _RNG_HEADER = """
#include <metal_stdlib>
using namespace metal;

// Initialise XorShift128 state from (base_seed, rng_offset).
// Matches quarimo._quartets.Quartets._init_rng and _rng_cuda.init_xorshift128.
inline void init_xorshift128(
    uint32_t base_seed,
    uint32_t rng_offset,
    thread uint32_t* state)
{
    uint64_t combined = (uint64_t)base_seed + (uint64_t)rng_offset;
    state[0] = (uint32_t)(combined & 0xFFFFFFFFu);
    state[1] = (uint32_t)(combined >> 32u);
    state[2] = 0x9e3779b9u;  // golden-ratio constant
    state[3] = 0x7f4a7c13u;  // arbitrary constant
}

// One XorShift128 step; returns state[0].
// Matches quarimo._quartets.Quartets._sample_quartet inner loop and _rng_cuda.xorshift128_next.
inline uint32_t xorshift128_next(thread uint32_t* state) {
    uint32_t t = state[3];
    uint32_t s = state[0];
    state[3] = state[2];
    state[2] = state[1];
    state[1] = s;
    t ^= (t << 11u);   // left shift: wraps at uint32 boundary
    t ^= (t >> 8u);    // logical right shift (unsigned — no arithmetic shift risk)
    state[0] = t ^ s ^ (s >> 19u);
    return state[0];
}

// Sample 4 unique taxa in [0, n_taxa) via rejection, then sort ascending.
// Matches quarimo._quartets.Quartets._sample_quartet.
inline void sample_4_unique(
    thread uint32_t* rng_state,
    int32_t n_taxa,
    thread int32_t* out)
{
    int32_t n = 0;
    while (n < 4) {
        int32_t c = (int32_t)(xorshift128_next(rng_state) % (uint32_t)n_taxa);
        bool unique = true;
        for (int32_t i = 0; i < n; i++) {
            if (out[i] == c) { unique = false; break; }
        }
        if (unique) out[n++] = c;
    }
    // Insertion sort — branch-free for 4 elements
    for (int32_t i = 1; i < 4; i++) {
        int32_t key = out[i], j = i - 1;
        while (j >= 0 && out[j] > key) { out[j + 1] = out[j]; j--; }
        out[j + 1] = key;
    }
}
"""

    # ====================================================================== #
    # Kernel body                                                             #
    # ====================================================================== #
    # Inputs (device const pointers, named to match input_names below):      #
    #   seed_quartets  int[n_seed * 4]   flat row-major int32               #
    #   n_seed_arr     int[1]            number of explicit seed quartets    #
    #   offset_arr     long[1]           starting absolute sequence index    #
    #   count_arr      int[1]            number of quartets to generate      #
    #   rng_seed_arr   uint[1]           XorShift base seed                  #
    #   n_taxa_arr     int[1]            global taxon namespace size         #
    # Output:                                                                 #
    #   quartets_out   int[count * 4]    flat row-major int32               #
    #                                                                         #
    # In MSL, mx.int32 → int, mx.int64 → long, mx.uint32 → uint.           #

    _GENERATE_QUARTETS_SOURCE = """
    uint32_t qi    = thread_position_in_grid.x;
    int32_t  count = count_arr[0];
    if ((int32_t)qi >= count) return;

    int32_t  n_seed   = n_seed_arr[0];
    long     offset   = offset_arr[0];
    uint32_t rng_seed = rng_seed_arr[0];
    int32_t  n_taxa   = n_taxa_arr[0];

    long absolute_idx = offset + (long)qi;

    int32_t a, b, c, d;
    if (absolute_idx < (long)n_seed) {
        // Return explicit seed quartet
        int32_t base = (int32_t)absolute_idx * 4;
        a = seed_quartets[base + 0];
        b = seed_quartets[base + 1];
        c = seed_quartets[base + 2];
        d = seed_quartets[base + 3];
    } else {
        // Generate via XorShift128 + rejection sampling
        uint32_t rng_state[4];
        uint32_t rng_offset = (uint32_t)(absolute_idx - (long)n_seed);
        init_xorshift128(rng_seed, rng_offset, rng_state);
        int32_t samples[4];
        sample_4_unique(rng_state, n_taxa, samples);
        a = samples[0]; b = samples[1]; c = samples[2]; d = samples[3];
    }

    int32_t out_base = (int32_t)qi * 4;
    quartets_out[out_base + 0] = a;
    quartets_out[out_base + 1] = b;
    quartets_out[out_base + 2] = c;
    quartets_out[out_base + 3] = d;
"""

    # Compile once at module import (Metal JIT — fast, sub-millisecond)
    _generate_kernel = mx.fast.metal_kernel(
        name="generate_quartets",
        input_names=["seed_quartets", "n_seed_arr", "offset_arr",
                     "count_arr", "rng_seed_arr", "n_taxa_arr"],
        output_names=["quartets_out"],
        header=_RNG_HEADER,
        source=_GENERATE_QUARTETS_SOURCE,
    )

    def generate_quartets_mlx(
        seed_quartets_np,   # int32 ndarray, shape (n_seed, 4)
        n_seed: int,
        offset: int,
        count: int,
        rng_seed: int,
        n_taxa: int,
    ):
        """
        Generate ``count`` quartets from the deterministic sequence starting at
        ``offset``, using the MLX Metal backend.

        Mirrors the interface of ``generate_quartets_cuda`` and produces
        bit-identical output for the same (seed, offset, rng_seed, n_taxa).

        Parameters
        ----------
        seed_quartets_np : int32 ndarray, shape (n_seed, 4)
            Explicit seed quartets.  Positions ``absolute_idx < n_seed`` are
            read directly from this array; positions beyond use the RNG.
        n_seed : int
        offset : int
            Starting absolute index in the infinite deterministic sequence.
        count : int
            Number of quartets to generate.
        rng_seed : int
            XorShift128 base seed (``Quartets.rng_seed``).
        n_taxa : int
            Global taxon namespace size.

        Returns
        -------
        ndarray, shape (count, 4), dtype int32
            Sorted quartet indices for each position.  On Apple Silicon (UMA)
            ``np.array(mlx_array)`` does not copy memory.
        """
        # Grid rounded up to threadgroup boundary; out-of-bounds threads
        # return early via the `if qi >= count` guard in the kernel.
        tg_size = 256
        grid_x = ((count + tg_size - 1) // tg_size) * tg_size

        out = _generate_kernel(
            inputs=[
                mx.array(seed_quartets_np.reshape(-1), dtype=mx.int32),
                mx.array([n_seed],                     dtype=mx.int32),
                mx.array([offset],                     dtype=mx.int64),
                mx.array([count],                      dtype=mx.int32),
                mx.array([int(rng_seed)],              dtype=mx.uint32),
                mx.array([n_taxa],                     dtype=mx.int32),
            ],
            output_shapes=[(count * 4,)],
            output_dtypes=[mx.int32],
            grid=(grid_x, 1, 1),
            threadgroup=(tg_size, 1, 1),
        )
        mx.eval(out[0])
        return np.array(out[0], copy=False).reshape(count, 4)

    # ====================================================================== #
    # Topology kernels                                                        #
    # ====================================================================== #
    # MSL header: O(1) RMQ helper used by both quartet_counts and            #
    # quartet_steiner kernels.  Does NOT include the RNG helpers — those     #
    # are only needed by generate_quartets_mlx.                              #

    _TOPOLOGY_HEADER = """
#include <metal_stdlib>
using namespace metal;

// O(1) RMQ lookup over a CSR-packed sparse table for one tree.
// Returns the local node ID of the LCA of any two positions l, r.
// sp_base, lg_base, tour_base are int64 offsets into the flat CSR arrays.
inline int32_t rmq_msl(
    int32_t l, int32_t r,
    long sp_base, int32_t sp_stride,
    device const int32_t* sparse_table,
    device const int32_t* euler_depth,
    device const int32_t* log2_table,
    long lg_base, long tour_base,
    device const int32_t* euler_tour)
{
    int32_t length = r - l + 1;
    int32_t k      = log2_table[lg_base + length];
    int32_t span   = 1 << k;
    int32_t li = sparse_table[sp_base + (long)k * sp_stride + l];
    int32_t ri = sparse_table[sp_base + (long)k * sp_stride + (r - span + 1)];
    int32_t lca_local =
        (euler_depth[tour_base + ri] < euler_depth[tour_base + li]) ? ri : li;
    return euler_tour[tour_base + lca_local];
}
"""

    # ---------------------------------------------------------------------- #
    # Counts kernel body                                                      #
    # One thread per quartet (qi).  Sequential inner loop over trees.        #
    # Each thread zero-initialises and owns its entire counts_out[qi] slice. #

    _QUARTET_COUNTS_BODY = """
    uint32_t qi      = thread_position_in_grid.x;
    int32_t n_q      = n_quartets_arr[0];
    if ((int32_t)qi >= n_q) return;

    int32_t n_trees  = n_trees_arr[0];
    int32_t n_groups = n_groups_arr[0];
    int32_t n_gtaxa  = n_global_taxa_arr[0];

    // Zero-initialise this thread's output slice
    int32_t out_base = (int32_t)qi * n_groups * 4;
    for (int32_t gi = 0; gi < n_groups; gi++)
        for (int32_t k = 0; k < 4; k++)
            counts_out[out_base + gi * 4 + k] = 0;

    int32_t t0 = sorted_quartet_ids[(int32_t)qi * 4 + 0];
    int32_t t1 = sorted_quartet_ids[(int32_t)qi * 4 + 1];
    int32_t t2 = sorted_quartet_ids[(int32_t)qi * 4 + 2];
    int32_t t3 = sorted_quartet_ids[(int32_t)qi * 4 + 3];

    for (int32_t ti = 0; ti < n_trees; ti++) {
        int32_t base_g2l = ti * n_gtaxa;
        int32_t ln0 = global_to_local[base_g2l + t0];
        int32_t ln1 = global_to_local[base_g2l + t1];
        int32_t ln2 = global_to_local[base_g2l + t2];
        int32_t ln3 = global_to_local[base_g2l + t3];
        if (ln0 < 0 || ln1 < 0 || ln2 < 0 || ln3 < 0) continue;

        long node_base = node_offsets[ti];
        long tour_base = tour_offsets[ti];
        long sp_base   = sp_offsets[ti];
        long lg_base   = lg_offsets[ti];
        int32_t sp_stride = sp_tour_widths[ti];

        int32_t fo0 = all_first_occ[node_base + ln0];
        int32_t fo1 = all_first_occ[node_base + ln1];
        int32_t fo2 = all_first_occ[node_base + ln2];
        int32_t fo3 = all_first_occ[node_base + ln3];

        // 6 RMQ calls for all pairwise LCAs
        int32_t l, r, tmp;
        l = fo0; r = fo1; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca01 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo0; r = fo2; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca02 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo0; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca03 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo1; r = fo2; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca12 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo1; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca13 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo2; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca23 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);

        // Four-point condition (float32 — Metal does not support float64)
        float rd01 = all_root_distance[node_base + lca01];
        float rd02 = all_root_distance[node_base + lca02];
        float rd03 = all_root_distance[node_base + lca03];
        float rd12 = all_root_distance[node_base + lca12];
        float rd13 = all_root_distance[node_base + lca13];
        float rd23 = all_root_distance[node_base + lca23];

        float r0 = rd01 + rd23;   // topology 0: (t0,t1)|(t2,t3)
        float r1 = rd02 + rd13;   // topology 1: (t0,t2)|(t1,t3)
        float r2 = rd03 + rd12;   // topology 2: (t0,t3)|(t1,t2)

        // Polytomy detection: CSR pre-filter + IEEE-754 tie check.
        // Matches _quartet_topology_and_rd_nb exactly.
        int32_t poly_start = polytomy_offsets[ti];
        int32_t poly_end   = polytomy_offsets[ti + 1];

        int32_t topo;
        if (poly_end > poly_start) {
            bool found_poly = false;
            for (int32_t j = poly_start; j < poly_end; j++) {
                int32_t pn = polytomy_nodes[j];
                if (pn == lca01 || pn == lca02 || pn == lca03 ||
                    pn == lca12 || pn == lca13 || pn == lca23) {
                    found_poly = true;
                    break;
                }
            }
            if (found_poly && r0 == r1 && r1 == r2) {
                topo = 3;
            } else {
                if      (r0 >= r1 && r0 >= r2) topo = 0;
                else if (r1 >= r0 && r1 >= r2) topo = 1;
                else                           topo = 2;
            }
        } else {
            if      (r0 >= r1 && r0 >= r2) topo = 0;
            else if (r1 >= r0 && r1 >= r2) topo = 1;
            else                           topo = 2;
        }

        int32_t gi = tree_to_group_idx[ti];
        counts_out[out_base + gi * 4 + topo] += 1;
    }
"""

    # ---------------------------------------------------------------------- #
    # Steiner kernel body                                                     #
    # Identical to counts body, plus Steiner length accumulation.            #
    # Outputs are float32 (Metal limitation); Python wrapper converts to f64. #

    _QUARTET_STEINER_BODY = """
    uint32_t qi      = thread_position_in_grid.x;
    int32_t n_q      = n_quartets_arr[0];
    if ((int32_t)qi >= n_q) return;

    int32_t n_trees  = n_trees_arr[0];
    int32_t n_groups = n_groups_arr[0];
    int32_t n_gtaxa  = n_global_taxa_arr[0];

    // Zero-initialise this thread's output slices
    int32_t out_base = (int32_t)qi * n_groups * 4;
    for (int32_t gi = 0; gi < n_groups; gi++) {
        for (int32_t k = 0; k < 4; k++) {
            int32_t idx = out_base + gi * 4 + k;
            counts_out[idx]      = 0;
            steiner_out[idx]     = 0.0f;
            steiner_min_out[idx] = INFINITY;
            steiner_max_out[idx] = -INFINITY;
            steiner_ssq_out[idx] = 0.0f;
        }
    }

    int32_t t0 = sorted_quartet_ids[(int32_t)qi * 4 + 0];
    int32_t t1 = sorted_quartet_ids[(int32_t)qi * 4 + 1];
    int32_t t2 = sorted_quartet_ids[(int32_t)qi * 4 + 2];
    int32_t t3 = sorted_quartet_ids[(int32_t)qi * 4 + 3];

    for (int32_t ti = 0; ti < n_trees; ti++) {
        int32_t base_g2l = ti * n_gtaxa;
        int32_t ln0 = global_to_local[base_g2l + t0];
        int32_t ln1 = global_to_local[base_g2l + t1];
        int32_t ln2 = global_to_local[base_g2l + t2];
        int32_t ln3 = global_to_local[base_g2l + t3];
        if (ln0 < 0 || ln1 < 0 || ln2 < 0 || ln3 < 0) continue;

        long node_base = node_offsets[ti];
        long tour_base = tour_offsets[ti];
        long sp_base   = sp_offsets[ti];
        long lg_base   = lg_offsets[ti];
        int32_t sp_stride = sp_tour_widths[ti];

        int32_t fo0 = all_first_occ[node_base + ln0];
        int32_t fo1 = all_first_occ[node_base + ln1];
        int32_t fo2 = all_first_occ[node_base + ln2];
        int32_t fo3 = all_first_occ[node_base + ln3];

        int32_t l, r, tmp;
        l = fo0; r = fo1; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca01 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo0; r = fo2; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca02 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo0; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca03 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo1; r = fo2; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca12 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo1; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca13 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
        l = fo2; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
        int32_t lca23 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);

        float rd01 = all_root_distance[node_base + lca01];
        float rd02 = all_root_distance[node_base + lca02];
        float rd03 = all_root_distance[node_base + lca03];
        float rd12 = all_root_distance[node_base + lca12];
        float rd13 = all_root_distance[node_base + lca13];
        float rd23 = all_root_distance[node_base + lca23];

        float r0 = rd01 + rd23;
        float r1 = rd02 + rd13;
        float r2 = rd03 + rd12;

        int32_t poly_start = polytomy_offsets[ti];
        int32_t poly_end   = polytomy_offsets[ti + 1];

        int32_t topo;
        float r_winner;
        if (poly_end > poly_start) {
            bool found_poly = false;
            for (int32_t j = poly_start; j < poly_end; j++) {
                int32_t pn = polytomy_nodes[j];
                if (pn == lca01 || pn == lca02 || pn == lca03 ||
                    pn == lca12 || pn == lca13 || pn == lca23) {
                    found_poly = true;
                    break;
                }
            }
            if (found_poly && r0 == r1 && r1 == r2) {
                topo = 3; r_winner = r0;
            } else {
                if      (r0 >= r1 && r0 >= r2) { topo = 0; r_winner = r0; }
                else if (r1 >= r0 && r1 >= r2) { topo = 1; r_winner = r1; }
                else                           { topo = 2; r_winner = r2; }
            }
        } else {
            if      (r0 >= r1 && r0 >= r2) { topo = 0; r_winner = r0; }
            else if (r1 >= r0 && r1 >= r2) { topo = 1; r_winner = r1; }
            else                           { topo = 2; r_winner = r2; }
        }

        // Steiner spanning length: S = sum(rd[leaf_i]) - 0.5*(r_winner+r0+r1+r2)
        float leaf_rd0 = all_root_distance[node_base + ln0];
        float leaf_rd1 = all_root_distance[node_base + ln1];
        float leaf_rd2 = all_root_distance[node_base + ln2];
        float leaf_rd3 = all_root_distance[node_base + ln3];
        float sl = (leaf_rd0 + leaf_rd1 + leaf_rd2 + leaf_rd3)
                   - 0.5f * (r_winner + r0 + r1 + r2);

        int32_t gi  = tree_to_group_idx[ti];
        int32_t idx = out_base + gi * 4 + topo;
        counts_out[idx]  += 1;
        steiner_out[idx] += sl;
        if (sl < steiner_min_out[idx]) steiner_min_out[idx] = sl;
        if (sl > steiner_max_out[idx]) steiner_max_out[idx] = sl;
        steiner_ssq_out[idx] += sl * sl;
    }
"""

    # ---------------------------------------------------------------------- #
    # Compile kernels once at import time (Metal JIT — fast, sub-ms)         #

    # Shared input names for both topology kernels
    _TOPOLOGY_INPUT_NAMES = [
        "sorted_quartet_ids",   # int32[n_quartets * 4]
        "global_to_local",      # int32[n_trees * n_global_taxa]
        "all_first_occ",        # int32[total_nodes]
        "all_root_distance",    # float32[total_nodes]  (converted from float64)
        "all_euler_tour",       # int32[total_tour_len]
        "all_euler_depth",      # int32[total_tour_len]
        "all_sparse_table",     # int32[total_sp_size]
        "all_log2_table",       # int32[total_log2_size]
        "node_offsets",         # int64[n_trees + 1]
        "tour_offsets",         # int64[n_trees + 1]
        "sp_offsets",           # int64[n_trees + 1]
        "lg_offsets",           # int64[n_trees + 1]
        "sp_tour_widths",       # int32[n_trees]
        "tree_to_group_idx",    # int32[n_trees]
        "polytomy_offsets",     # int32[n_trees + 1]
        "polytomy_nodes",       # int32[total_polytomy] (at least 1 element)
        "n_quartets_arr",       # int32[1]
        "n_trees_arr",          # int32[1]
        "n_groups_arr",         # int32[1]
        "n_global_taxa_arr",    # int32[1]
    ]

    _counts_topology_kernel = mx.fast.metal_kernel(
        name="quartet_counts",
        input_names=_TOPOLOGY_INPUT_NAMES,
        output_names=["counts_out"],
        header=_TOPOLOGY_HEADER,
        source=_QUARTET_COUNTS_BODY,
    )

    _steiner_topology_kernel = mx.fast.metal_kernel(
        name="quartet_steiner",
        input_names=_TOPOLOGY_INPUT_NAMES,
        output_names=["counts_out", "steiner_out", "steiner_min_out",
                      "steiner_max_out", "steiner_ssq_out"],
        header=_TOPOLOGY_HEADER,
        source=_QUARTET_STEINER_BODY,
    )

    def _topology_inputs(kd, sorted_ids, n_quartets, n_groups):
        """
        Build the shared MLX input list for both topology kernels.

        Converts ``all_root_distance`` from float64 to float32 (Metal
        does not support float64).  An empty ``polytomy_nodes`` array is
        replaced by a one-element sentinel ``[-1]`` so the buffer is never
        zero-length.

        Parameters
        ----------
        kd : ForestKernelData
        sorted_ids : int32[n_quartets, 4]
        n_quartets : int
        n_groups : int
        """
        poly_nodes = (
            kd.polytomy_nodes if len(kd.polytomy_nodes) > 0
            else np.array([-1], dtype=np.int32)
        )
        return [
            mx.array(sorted_ids.reshape(-1),             dtype=mx.int32),
            mx.array(kd.global_to_local.reshape(-1),     dtype=mx.int32),
            mx.array(kd.all_first_occ,                   dtype=mx.int32),
            mx.array(kd.all_root_distance.astype("f4"),  dtype=mx.float32),
            mx.array(kd.all_euler_tour,                  dtype=mx.int32),
            mx.array(kd.all_euler_depth,                 dtype=mx.int32),
            mx.array(kd.all_sparse_table,               dtype=mx.int32),
            mx.array(kd.all_log2_table,                  dtype=mx.int32),
            mx.array(kd.node_offsets,                    dtype=mx.int64),
            mx.array(kd.tour_offsets,                    dtype=mx.int64),
            mx.array(kd.sp_offsets,                      dtype=mx.int64),
            mx.array(kd.lg_offsets,                      dtype=mx.int64),
            mx.array(kd.sp_tour_widths,                  dtype=mx.int32),
            mx.array(kd.tree_to_group_idx,               dtype=mx.int32),
            mx.array(kd.polytomy_offsets,                dtype=mx.int32),
            mx.array(poly_nodes,                         dtype=mx.int32),
            mx.array([n_quartets],                       dtype=mx.int32),
            mx.array([kd.n_trees],                       dtype=mx.int32),
            mx.array([n_groups],                         dtype=mx.int32),
            mx.array([kd.n_global_taxa],                 dtype=mx.int32),
        ]

    def quartet_counts_mlx(kd, sorted_ids, n_quartets, n_groups):
        """
        Count quartet topologies for all (quartet, group) pairs using Metal.

        One Metal thread per quartet; sequential inner loop over trees.
        No atomics needed — each thread owns its entire ``counts_out[qi]``
        slice.

        Parameters
        ----------
        kd : ForestKernelData
            Forest kernel data (CPU arrays; not device-uploaded).
        sorted_ids : int32[n_quartets, 4]
            Quartet taxa as sorted global IDs.
        n_quartets : int
        n_groups : int

        Returns
        -------
        counts_out : int32[n_quartets, n_groups, 4]
            Topology counts.  Last axis: k=0,1,2 resolved; k=3 unresolved.
        """
        tg_size = 256
        grid_x = ((n_quartets + tg_size - 1) // tg_size) * tg_size

        out = _counts_topology_kernel(
            inputs=_topology_inputs(kd, sorted_ids, n_quartets, n_groups),
            output_shapes=[(n_quartets * n_groups * 4,)],
            output_dtypes=[mx.int32],
            grid=(grid_x, 1, 1),
            threadgroup=(tg_size, 1, 1),
        )
        mx.eval(out[0])
        return np.array(out[0], copy=False).reshape(n_quartets, n_groups, 4)

    # ====================================================================== #
    # Delta kernel                                                            #
    # ====================================================================== #
    # 1 thread per affected quartet (qi_local); sequential inner loop over   #
    # affected trees.  No atomics needed — each thread owns its entire       #
    # output row.  On Apple Silicon (UMA) the mx.array() wrapping of numpy  #
    # host arrays has negligible cost, so no device-resident state is needed #
    # and the kernel is simply called once per evaluate_swap / apply_swap.   #
    #                                                                         #
    # Output layout: only the n_affected_quartets rows, not the full array.  #
    # The Python wrapper scatter-assigns them back into the host counts array #
    # (counts_out[affected_qi] = delta_rows).                                #

    # Shared topology-resolution helper used by the delta kernel only.
    # Appended to _TOPOLOGY_HEADER so that rmq_msl is in scope.
    _RESOLVE_TOPO_HELPER = """
// Resolve the quartet topology for one (quartet, tree) pair.
// Returns k=0,1,2 for the three resolved topologies or k=3 for a polytomy.
inline int32_t resolve_topo_msl(
    int32_t ln0, int32_t ln1, int32_t ln2, int32_t ln3,
    long node_base, long tour_base, long sp_base, long lg_base, int32_t sp_stride,
    int32_t poly_start, int32_t poly_end,
    device const int32_t* all_first_occ,
    device const float*   all_root_distance,
    device const int32_t* all_euler_tour,
    device const int32_t* all_euler_depth,
    device const int32_t* all_sparse_table,
    device const int32_t* all_log2_table,
    device const int32_t* polytomy_nodes)
{
    int32_t fo0 = all_first_occ[node_base + ln0];
    int32_t fo1 = all_first_occ[node_base + ln1];
    int32_t fo2 = all_first_occ[node_base + ln2];
    int32_t fo3 = all_first_occ[node_base + ln3];

    int32_t l, r, tmp;
    l = fo0; r = fo1; if (l > r) { tmp = l; l = r; r = tmp; }
    int32_t lca01 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
    l = fo0; r = fo2; if (l > r) { tmp = l; l = r; r = tmp; }
    int32_t lca02 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
    l = fo0; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
    int32_t lca03 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
    l = fo1; r = fo2; if (l > r) { tmp = l; l = r; r = tmp; }
    int32_t lca12 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
    l = fo1; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
    int32_t lca13 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);
    l = fo2; r = fo3; if (l > r) { tmp = l; l = r; r = tmp; }
    int32_t lca23 = rmq_msl(l, r, sp_base, sp_stride, all_sparse_table, all_euler_depth, all_log2_table, lg_base, tour_base, all_euler_tour);

    float rd01 = all_root_distance[node_base + lca01];
    float rd02 = all_root_distance[node_base + lca02];
    float rd03 = all_root_distance[node_base + lca03];
    float rd12 = all_root_distance[node_base + lca12];
    float rd13 = all_root_distance[node_base + lca13];
    float rd23 = all_root_distance[node_base + lca23];

    float r0 = rd01 + rd23;
    float r1 = rd02 + rd13;
    float r2 = rd03 + rd12;

    if (poly_end > poly_start) {
        bool found_poly = false;
        for (int32_t j = poly_start; j < poly_end; j++) {
            int32_t pn = polytomy_nodes[j];
            if (pn == lca01 || pn == lca02 || pn == lca03 ||
                pn == lca12 || pn == lca13 || pn == lca23) {
                found_poly = true;
                break;
            }
        }
        if (found_poly && r0 == r1 && r1 == r2) return 3;
    }
    if (r0 >= r1 && r0 >= r2) return 0;
    if (r1 >= r0 && r1 >= r2) return 1;
    return 2;
}
"""

    _DELTA_HEADER = _TOPOLOGY_HEADER + _RESOLVE_TOPO_HELPER

    _QUARTET_DELTA_BODY = """
    uint32_t qi_local       = thread_position_in_grid.x;
    int32_t  n_affected     = n_affected_arr[0];
    if ((int32_t)qi_local >= n_affected) return;

    int32_t n_groups         = n_groups_arr[0];
    int32_t n_gtaxa          = n_global_taxa_arr[0];
    int32_t n_affected_trees = n_affected_trees_arr[0];

    int32_t t0 = delta_quartet_ids[(int32_t)qi_local * 4 + 0];
    int32_t t1 = delta_quartet_ids[(int32_t)qi_local * 4 + 1];
    int32_t t2 = delta_quartet_ids[(int32_t)qi_local * 4 + 2];
    int32_t t3 = delta_quartet_ids[(int32_t)qi_local * 4 + 3];
    int32_t qi = delta_quartet_global_idx[(int32_t)qi_local];

    // Copy this quartet's counts row from counts_in → counts_out
    int32_t qi_base_in  = qi            * n_groups * 4;
    int32_t qi_base_out = (int32_t)qi_local * n_groups * 4;
    for (int32_t g = 0; g < n_groups; g++)
        for (int32_t k = 0; k < 4; k++)
            counts_out[qi_base_out + g * 4 + k] = counts_in[qi_base_in + g * 4 + k];

    // Apply per-tree signed deltas
    for (int32_t t_idx = 0; t_idx < n_affected_trees; t_idx++) {
        int32_t ti       = delta_tree_ids[t_idx];
        int32_t gi       = tree_to_group_idx[ti];
        int32_t base_g2l = ti * n_gtaxa;

        int32_t ln0_old = old_global_to_local[base_g2l + t0];
        int32_t ln1_old = old_global_to_local[base_g2l + t1];
        int32_t ln2_old = old_global_to_local[base_g2l + t2];
        int32_t ln3_old = old_global_to_local[base_g2l + t3];
        if (ln0_old < 0 || ln1_old < 0 || ln2_old < 0 || ln3_old < 0) continue;

        long    node_base = node_offsets[ti];
        long    tour_base = tour_offsets[ti];
        long    sp_base   = sp_offsets[ti];
        long    lg_base   = lg_offsets[ti];
        int32_t sp_stride = sp_tour_widths[ti];
        int32_t poly_start = polytomy_offsets[ti];
        int32_t poly_end   = polytomy_offsets[ti + 1];

        int32_t old_topo = resolve_topo_msl(
            ln0_old, ln1_old, ln2_old, ln3_old,
            node_base, tour_base, sp_base, lg_base, sp_stride,
            poly_start, poly_end,
            all_first_occ, all_root_distance, all_euler_tour,
            all_euler_depth, all_sparse_table, all_log2_table, polytomy_nodes);

        int32_t ln0_new = new_global_to_local[base_g2l + t0];
        int32_t ln1_new = new_global_to_local[base_g2l + t1];
        int32_t ln2_new = new_global_to_local[base_g2l + t2];
        int32_t ln3_new = new_global_to_local[base_g2l + t3];
        if (ln0_new < 0 || ln1_new < 0 || ln2_new < 0 || ln3_new < 0) continue;

        int32_t new_topo = resolve_topo_msl(
            ln0_new, ln1_new, ln2_new, ln3_new,
            node_base, tour_base, sp_base, lg_base, sp_stride,
            poly_start, poly_end,
            all_first_occ, all_root_distance, all_euler_tour,
            all_euler_depth, all_sparse_table, all_log2_table, polytomy_nodes);

        if (old_topo != new_topo) {
            counts_out[qi_base_out + gi * 4 + old_topo] -= 1;
            counts_out[qi_base_out + gi * 4 + new_topo] += 1;
        }
    }
"""

    _DELTA_INPUT_NAMES = [
        "delta_quartet_ids",        # int32[n_affected * 4]
        "delta_quartet_global_idx", # int32[n_affected]
        "old_global_to_local",      # int32[n_trees * n_global_taxa]
        "new_global_to_local",      # int32[n_trees * n_global_taxa]
        "delta_tree_ids",           # int32[n_affected_trees]
        "counts_in",                # int32[n_quartets * n_groups * 4]
        "all_first_occ",            # int32[total_nodes]
        "all_root_distance",        # float32[total_nodes]
        "all_euler_tour",           # int32[total_tour_len]
        "all_euler_depth",          # int32[total_tour_len]
        "all_sparse_table",         # int32[total_sp_size]
        "all_log2_table",           # int32[total_log2_size]
        "node_offsets",             # int64[n_trees + 1]
        "tour_offsets",             # int64[n_trees + 1]
        "sp_offsets",               # int64[n_trees + 1]
        "lg_offsets",               # int64[n_trees + 1]
        "sp_tour_widths",           # int32[n_trees]
        "tree_to_group_idx",        # int32[n_trees]
        "polytomy_offsets",         # int32[n_trees + 1]
        "polytomy_nodes",           # int32[total_polytomy] (at least 1 element)
        "n_affected_arr",           # int32[1]
        "n_affected_trees_arr",     # int32[1]
        "n_groups_arr",             # int32[1]
        "n_global_taxa_arr",        # int32[1]
    ]

    _delta_kernel = mx.fast.metal_kernel(
        name="quartet_counts_delta",
        input_names=_DELTA_INPUT_NAMES,
        output_names=["counts_out"],
        header=_DELTA_HEADER,
        source=_QUARTET_DELTA_BODY,
    )

    def quartet_counts_delta_mlx(
        kd,
        affected_taxa,          # int32[n_affected, 4]
        affected_qi,            # int32[n_affected]
        old_global_to_local,    # int32[n_trees, n_global_taxa]
        new_global_to_local,    # int32[n_trees, n_global_taxa]
        affected_tree_ids,      # int32[n_affected_trees]
        counts_out,             # int32[n_quartets, n_groups, 4] — modified in-place
        n_groups: int,
    ) -> None:
        """
        Apply a paralog copy-slot permutation to ``counts_out`` in-place using Metal.

        Dispatches one Metal thread per affected quartet; each thread loops
        sequentially over the affected trees and applies signed ±1 updates.
        No atomics are needed because each thread owns its entire output row.

        On Apple Silicon (UMA), wrapping numpy arrays as ``mx.array`` has
        negligible cost, so no device-resident state is maintained between
        calls.  The kernel outputs only the affected rows; they are scattered
        back into ``counts_out`` via index assignment.

        Parameters
        ----------
        kd : ForestKernelData
            Forest structural arrays (CPU arrays, wrapped by MLX at call time).
        affected_taxa : int32[n_affected, 4]
            Sorted global taxon IDs for each affected quartet.
        affected_qi : int32[n_affected]
            Row indices into ``counts_out`` for each affected quartet.
        old_global_to_local : int32[n_trees, n_global_taxa]
            Copy-slot → local-leaf mapping before the permutation.
        new_global_to_local : int32[n_trees, n_global_taxa]
            Copy-slot → local-leaf mapping after the permutation.
        affected_tree_ids : int32[n_affected_trees]
            Indices of trees where ≥ 2 copies of the genome are present.
        counts_out : int32[n_quartets, n_groups, 4]
            Modified in-place via scatter assignment of the affected rows.
        n_groups : int
        """
        n_affected = len(affected_qi)
        n_affected_trees = len(affected_tree_ids)
        if n_affected == 0 or n_affected_trees == 0:
            return

        poly_nodes = (
            kd.polytomy_nodes if len(kd.polytomy_nodes) > 0
            else np.array([-1], dtype=np.int32)
        )

        tg_size = 256
        grid_x = ((n_affected + tg_size - 1) // tg_size) * tg_size

        out = _delta_kernel(
            inputs=[
                mx.array(affected_taxa.reshape(-1),          dtype=mx.int32),
                mx.array(affected_qi,                        dtype=mx.int32),
                mx.array(old_global_to_local.reshape(-1),    dtype=mx.int32),
                mx.array(new_global_to_local.reshape(-1),    dtype=mx.int32),
                mx.array(affected_tree_ids,                  dtype=mx.int32),
                mx.array(counts_out.reshape(-1),             dtype=mx.int32),
                mx.array(kd.all_first_occ,                   dtype=mx.int32),
                mx.array(kd.all_root_distance.astype("f4"),  dtype=mx.float32),
                mx.array(kd.all_euler_tour,                  dtype=mx.int32),
                mx.array(kd.all_euler_depth,                 dtype=mx.int32),
                mx.array(kd.all_sparse_table,                dtype=mx.int32),
                mx.array(kd.all_log2_table,                  dtype=mx.int32),
                mx.array(kd.node_offsets,                    dtype=mx.int64),
                mx.array(kd.tour_offsets,                    dtype=mx.int64),
                mx.array(kd.sp_offsets,                      dtype=mx.int64),
                mx.array(kd.lg_offsets,                      dtype=mx.int64),
                mx.array(kd.sp_tour_widths,                  dtype=mx.int32),
                mx.array(kd.tree_to_group_idx,               dtype=mx.int32),
                mx.array(kd.polytomy_offsets,                dtype=mx.int32),
                mx.array(poly_nodes,                         dtype=mx.int32),
                mx.array([n_affected],                       dtype=mx.int32),
                mx.array([n_affected_trees],                 dtype=mx.int32),
                mx.array([n_groups],                         dtype=mx.int32),
                mx.array([kd.n_global_taxa],                 dtype=mx.int32),
            ],
            output_shapes=[(n_affected * n_groups * 4,)],
            output_dtypes=[mx.int32],
            grid=(grid_x, 1, 1),
            threadgroup=(tg_size, 1, 1),
        )
        mx.eval(out[0])

        # Scatter the affected rows back into counts_out in-place
        delta_rows = np.array(out[0], copy=False).reshape(n_affected, n_groups, 4)
        counts_out[affected_qi] = delta_rows

    def quartet_steiner_mlx(kd, sorted_ids, n_quartets, n_groups):
        """
        Count quartet topologies and accumulate Steiner distances using Metal.

        Steiner accumulation runs in float32 (Metal limitation).  The Python
        wrapper returns float64 arrays for compatibility with the rest of the
        pipeline; the caller handles any precision differences.

        Parameters
        ----------
        kd : ForestKernelData
        sorted_ids : int32[n_quartets, 4]
        n_quartets : int
        n_groups : int

        Returns
        -------
        counts_out : int32[n_quartets, n_groups, 4]
        steiner_out : float64[n_quartets, n_groups, 4]
            Sum of Steiner lengths per cell.
        steiner_min_out : float64[n_quartets, n_groups, 4]
            Per-cell minimum Steiner length; +inf for empty cells (count==0).
        steiner_max_out : float64[n_quartets, n_groups, 4]
            Per-cell maximum Steiner length; -inf for empty cells (count==0).
        steiner_sum_sq_out : float64[n_quartets, n_groups, 4]
            Sum of squared Steiner lengths per cell (for variance).
        """
        shape = (n_quartets * n_groups * 4,)
        tg_size = 256
        grid_x = ((n_quartets + tg_size - 1) // tg_size) * tg_size

        out = _steiner_topology_kernel(
            inputs=_topology_inputs(kd, sorted_ids, n_quartets, n_groups),
            output_shapes=[shape, shape, shape, shape, shape],
            output_dtypes=[mx.int32, mx.float32, mx.float32, mx.float32, mx.float32],
            grid=(grid_x, 1, 1),
            threadgroup=(tg_size, 1, 1),
        )
        mx.eval(*out)

        def _f64(arr):
            return np.array(arr, copy=False).reshape(n_quartets, n_groups, 4).astype(np.float64)

        return (
            np.array(out[0], copy=False).reshape(n_quartets, n_groups, 4),
            _f64(out[1]),
            _f64(out[2]),
            _f64(out[3]),
            _f64(out[4]),
        )
