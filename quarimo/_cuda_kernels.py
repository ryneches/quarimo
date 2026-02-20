"""
_kernels_cuda.py
================
CUDA-accelerated quartet topology kernels using Numba CUDA.

This module contains ONLY numba.cuda code and should not import other project
modules to avoid import-time complications. The module exposes GPU-accelerated
kernels for quartet topology queries when a compatible CUDA GPU is available.

Exported Functions
------------------
_rmq_csr_cuda : cuda.jit device function
    O(1) range minimum query helper for CUDA kernels (device-only).

_quartet_counts_cuda : cuda.jit kernel
    GPU-parallel quartet topology counts (no Steiner distances).

_quartet_steiner_cuda : cuda.jit kernel
    GPU-parallel quartet topology counts with Steiner distances.

_compute_cuda_grid : function
    Helper to compute CUDA grid dimensions.

Notes
-----
- All kernel functions use @cuda.jit decoration
- Device functions use @cuda.jit(device=True)
- 2D thread grid (qi, ti) - each thread processes one (quartet, tree) pair
- Atomic operations used for counts_out (multiple threads may write same cell)
- steiner_out writes are conflict-free (one thread per (qi, ti) pair)
"""

import numpy as np

# This module requires numba.cuda
try:
    from numba import cuda
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
        Device function for RMQ — inlined into CUDA kernels.
        
        This is a GPU device function that performs O(1) range minimum query
        over CSR-packed sparse table. It cannot be called from CPU code.
        
        Parameters
        ----------
        l, r         : int  
            Inclusive local tour range (l <= r).
        sp_base      : int  
            Offset of this tree's sparse table.
        sp_stride    : int  
            Column stride for sparse table.
        sparse_table : device array
            all_sparse_table on GPU.
        euler_depth  : device array
            all_euler_depth on GPU.
        log2_table   : device array
            all_log2_table on GPU.
        lg_base      : int  
            Offset of this tree's log2_table.
        tour_base    : int  
            Offset of this tree's tour.
        euler_tour   : device array
            all_euler_tour on GPU.
        
        Returns
        -------
        int  
            Local node ID of the LCA.
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
        CUDA counts-only quartet kernel.
        
        Each thread processes one (quartet, tree) pair. The 2D thread grid
        has qi (quartet index) on the x-axis and ti (tree index) on the y-axis.
        
        Atomic operations are used for counts_out since multiple threads
        may increment the same (qi, topology) cell.
        
        Parameters
        ----------
        sorted_quartet_ids : int32[n_quartets, 4] on device
            Quartet taxa as sorted global IDs.
        global_to_local : int32[n_trees, n_global_taxa] on device
            Global ID → local leaf ID mapping (-1 if absent).
        all_first_occ : int32[total_nodes] on device
            CSR-packed first Euler tour occurrence.
        all_root_distance : float64[total_nodes] on device
            CSR-packed root distances.
        all_euler_tour : int32[total_tour_length] on device
            CSR-packed Euler tour.
        all_euler_depth : int32[total_tour_length] on device
            CSR-packed Euler tour depths.
        all_sparse_table : int32[total_sparse_entries] on device
            CSR-packed sparse tables for RMQ.
        all_log2_table : int32[total_log2_entries] on device
            CSR-packed log2 lookup tables.
        node_offsets : int64[n_trees+1] on device
            CSR offsets into node arrays.
        tour_offsets : int64[n_trees+1] on device
            CSR offsets into tour arrays.
        sp_offsets : int64[n_trees+1] on device
            CSR offsets into sparse table.
        lg_offsets : int64[n_trees+1] on device
            CSR offsets into log2 table.
        sp_tour_widths : int32[n_trees] on device
            Sparse table column strides.
        n_quartets : int
            Number of quartets to process.
        n_trees : int
            Number of trees in collection.
        counts_out : int32[n_quartets, 3] on device
            Output array for topology counts.
        """
        # 2D thread grid: (qi, ti)
        qi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Bounds check
        if qi >= n_quartets or ti >= n_trees:
            return

        # Get quartet taxa
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]

        # Map to local leaf IDs
        ln0 = global_to_local[ti, n0]
        ln1 = global_to_local[ti, n1]
        ln2 = global_to_local[ti, n2]
        ln3 = global_to_local[ti, n3]

        # Skip if any taxon is absent
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        # Get CSR offsets for this tree
        nb = node_offsets[ti]
        tb = tour_offsets[ti]
        sb = sp_offsets[ti]
        lb = lg_offsets[ti]
        tw = sp_tour_widths[ti]

        # Compute 6 pairwise LCAs
        l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln1]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd01 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln2]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd02 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln3]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd03 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln2]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd12 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln3]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd13 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln2]; r = all_first_occ[nb + ln3]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
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
        CUDA quartet kernel with Steiner distances.
        
        Identical to _quartet_counts_cuda plus Steiner distance calculation.
        Each thread computes one (quartet, tree) pair.
        
        The steiner_out write is conflict-free since each thread writes to
        its unique (qi, ti, topo) location.
        
        Parameters
        ----------
        Same as _quartet_counts_cuda, plus:
        
        steiner_out : float64[n_quartets, n_trees, 3] on device
            Output array for Steiner distances.
        """
        # 2D thread grid: (qi, ti)
        qi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ti = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Bounds check
        if qi >= n_quartets or ti >= n_trees:
            return

        # Get quartet taxa
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]

        # Map to local leaf IDs
        ln0 = global_to_local[ti, n0]
        ln1 = global_to_local[ti, n1]
        ln2 = global_to_local[ti, n2]
        ln3 = global_to_local[ti, n3]

        # Skip if any taxon is absent
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        # Get CSR offsets for this tree
        nb = node_offsets[ti]
        tb = tour_offsets[ti]
        sb = sp_offsets[ti]
        lb = lg_offsets[ti]
        tw = sp_tour_widths[ti]

        # Compute 6 pairwise LCAs (identical to counts kernel)
        l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln1]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd01 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln2]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd02 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln0]; r = all_first_occ[nb + ln3]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd03 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln2]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd12 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln1]; r = all_first_occ[nb + ln3]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                            all_log2_table, lb, tb, all_euler_tour)
        rd13 = all_root_distance[nb + lca]

        l = all_first_occ[nb + ln2]; r = all_first_occ[nb + ln3]
        if l > r: l, r = r, l
        lca = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
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

        # Atomic increment for counts
        cuda.atomic.add(counts_out, (qi, topo), 1)

        # Compute and store Steiner distance (conflict-free write)
        leaf_rd_sum = (all_root_distance[nb + ln0]
                     + all_root_distance[nb + ln1]
                     + all_root_distance[nb + ln2]
                     + all_root_distance[nb + ln3])
        S = leaf_rd_sum - (r_winner + r0 + r1 + r2) * 0.5
        steiner_out[qi, ti, topo] = S


def _compute_cuda_grid(n_quartets, n_trees, threads_per_block=(16, 16)):
    """
    Compute CUDA grid dimensions for the 2D (qi, ti) thread space.
    
    This helper function calculates the number of blocks needed in each
    dimension to cover all quartets and trees.

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
