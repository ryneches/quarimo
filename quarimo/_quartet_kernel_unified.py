"""
Unified GPU kernel for quartet topology counting with on-GPU generation.

This kernel handles both explicit seed quartets and random generation
in a single unified implementation.
"""

from numba import cuda
import numba
import numpy as np

# Import RNG device functions
from ._rng_cuda import (
    init_xorshift128,
    xorshift128_next,
    sample_4_unique_cuda,
    get_quartet_at_index
)


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
    node_offsets,       # [n_trees + 1] int64
    all_parent,         # [total_nodes] int32
    all_depth,          # [total_nodes] int32
    all_first_occ,      # [total_nodes] int32
    tour_offsets,       # [n_trees + 1] int64
    all_euler_tour,     # [total_tour_len] int32
    all_euler_depth,    # [total_tour_len] int32
    sp_offsets,         # [n_trees + 1] int64
    all_sparse_table,   # [total_sp_size] int32
    sp_log_widths,      # [n_trees] int32
    # Output
    counts              # [count, 3] int32 - topology counts
):
    """
    Unified kernel: process quartets from deterministic sequence.
    
    For each local index i in [0, count):
      absolute_idx = offset + i
      if absolute_idx < n_seed:
          quartet = seed_quartets[absolute_idx]  # Explicit
      else:
          quartet = generate_random(...)         # Random
      
      Process quartet across all trees
    
    Grid/Block Configuration
    -------------------------
    - Grid: 1D grid over quartets
    - Threads per block: typically 256
    - Each thread processes one quartet across all trees
    """
    local_idx = cuda.grid(1)
    if local_idx >= count:
        return
    
    # Determine absolute index in the infinite sequence
    absolute_idx = offset + local_idx
    
    # Get quartet for this index
    a, b, c, d = get_quartet_at_index(
        absolute_idx,
        seed_quartets,
        n_seed,
        rng_seed,
        n_global_taxa
    )
    
    # Get number of trees
    n_trees = node_offsets.shape[0] - 1
    
    # Process this quartet across all trees
    for tree_idx in range(n_trees):
        # Get local IDs for this tree
        local_a = global_to_local[tree_idx, a]
        local_b = global_to_local[tree_idx, b]
        local_c = global_to_local[tree_idx, c]
        local_d = global_to_local[tree_idx, d]
        
        # Check if all 4 taxa present (-1 means absent)
        if local_a == -1 or local_b == -1 or local_c == -1 or local_d == -1:
            continue
        
        # Get offsets for this tree's data
        node_start = node_offsets[tree_idx]
        tour_start = tour_offsets[tree_idx]
        sp_start = sp_offsets[tree_idx]
        sp_log_width = sp_log_widths[tree_idx]
        
        # Get first occurrences in Euler tour
        occ_a = all_first_occ[node_start + local_a]
        occ_b = all_first_occ[node_start + local_b]
        occ_c = all_first_occ[node_start + local_c]
        occ_d = all_first_occ[node_start + local_d]
        
        # Find LCAs using RMQ on Euler tour depths
        # LCA(a,b)
        left_ab = min(occ_a, occ_b)
        right_ab = max(occ_a, occ_b)
        lca_ab_idx = _rmq_csr_cuda(
            left_ab, right_ab,
            all_euler_tour, all_euler_depth, all_sparse_table,
            tour_start, sp_start, sp_log_width
        )
        lca_ab_node = all_euler_tour[tour_start + lca_ab_idx]
        lca_ab_depth = all_depth[node_start + lca_ab_node]
        
        # LCA(c,d)
        left_cd = min(occ_c, occ_d)
        right_cd = max(occ_c, occ_d)
        lca_cd_idx = _rmq_csr_cuda(
            left_cd, right_cd,
            all_euler_tour, all_euler_depth, all_sparse_table,
            tour_start, sp_start, sp_log_width
        )
        lca_cd_node = all_euler_tour[tour_start + lca_cd_idx]
        lca_cd_depth = all_depth[node_start + lca_cd_node]
        
        # LCA(a,c)
        left_ac = min(occ_a, occ_c)
        right_ac = max(occ_a, occ_c)
        lca_ac_idx = _rmq_csr_cuda(
            left_ac, right_ac,
            all_euler_tour, all_euler_depth, all_sparse_table,
            tour_start, sp_start, sp_log_width
        )
        lca_ac_node = all_euler_tour[tour_start + lca_ac_idx]
        lca_ac_depth = all_depth[node_start + lca_ac_node]
        
        # Determine topology by comparing LCA depths
        # Topology 0: (AB|CD) - lca(A,B) and lca(C,D) are deepest
        # Topology 1: (AC|BD) - lca(A,C) and lca(B,D) are deepest
        # Topology 2: (AD|BC) - lca(A,D) and lca(B,C) are deepest
        
        # For efficiency, check which pair has shallowest LCA
        # The topology is the one where the split pair has shallow LCA
        if lca_ab_depth <= lca_cd_depth and lca_ab_depth <= lca_ac_depth:
            # lca(AB) is shallowest → topology (AC|BD) or (AD|BC)
            if lca_cd_depth < lca_ac_depth:
                topology = 1  # (AC|BD)
            else:
                topology = 2  # (AD|BC)
        elif lca_cd_depth <= lca_ab_depth and lca_cd_depth <= lca_ac_depth:
            # lca(CD) is shallowest → topology (AB|CD) or (AD|BC)
            if lca_ab_depth < lca_ac_depth:
                topology = 0  # (AB|CD)
            else:
                topology = 2  # (AD|BC)
        else:
            # lca(AC) is shallowest → topology (AB|CD) or (AC|BD)
            if lca_ab_depth < lca_cd_depth:
                topology = 0  # (AB|CD)
            else:
                topology = 1  # (AC|BD)
        
        # Accumulate count
        cuda.atomic.add(counts, (local_idx, topology), 1)


@cuda.jit(device=True)
def _rmq_csr_cuda(
    left, right,
    all_euler_tour, all_euler_depth, all_sparse_table,
    tour_start, sp_start, sp_log_width
):
    """
    Range minimum query on Euler tour depths using sparse table.
    
    Parameters
    ----------
    left, right : int
        Range indices (relative to this tree's tour)
    all_euler_tour : array
        Concatenated Euler tours
    all_euler_depth : array
        Concatenated Euler depths
    all_sparse_table : array
        Concatenated sparse tables
    tour_start : int
        Offset into Euler tour arrays
    sp_start : int
        Offset into sparse table array
    sp_log_width : int
        Log width for this tree's sparse table
    
    Returns
    -------
    int
        Index of minimum depth element (relative to this tree's tour)
    """
    if left == right:
        return left
    
    # Compute log2(right - left)
    length = right - left
    k = 0
    temp = length
    while temp > 1:
        temp >>= 1
        k += 1
    
    # Get indices in sparse table
    # SP[i,k] is at offset: i * sp_log_width + k
    idx_left = sp_start + left * sp_log_width + k
    idx_right = sp_start + (right - (1 << k)) * sp_log_width + k
    
    min_left = all_sparse_table[idx_left]
    min_right = all_sparse_table[idx_right]
    
    # Return index of minimum depth
    depth_left = all_euler_depth[tour_start + min_left]
    depth_right = all_euler_depth[tour_start + min_right]
    
    if depth_left <= depth_right:
        return min_left
    else:
        return min_right
