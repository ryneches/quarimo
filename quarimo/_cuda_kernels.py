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

    # ======================================================================== #
    # RNG Device Functions for On-GPU Quartet Generation                       #
    # ======================================================================== #

    @cuda.jit(device=True)
    def init_xorshift128(base_seed, offset, state):
        """
        Initialize XorShift128 RNG state.
        
        Parameters
        ----------
        base_seed : uint32
            Base seed from hash of seed quartets
        offset : int
            Offset into random sequence (absolute_idx - n_seed)
        state : array of 4 uint32
            Output: initialized RNG state
        
        Notes
        -----
        Must match _quartets.py::_init_rng() exactly.
        """
        # Combine seed and offset
        combined = np.uint64(base_seed) + np.uint64(offset)
        
        # Initialize 4-word state
        state[0] = np.uint32(combined & 0xFFFFFFFF)
        state[1] = np.uint32((combined >> 32) & 0xFFFFFFFF)
        state[2] = np.uint32(0x9e3779b9)  # Golden ratio
        state[3] = np.uint32(0x7f4a7c13)  # Arbitrary constant

    @cuda.jit(device=True)
    def xorshift128_next(state):
        """
        Generate next random number with XorShift128.
        
        Parameters
        ----------
        state : array of 4 uint32
            RNG state (modified in place)
        
        Returns
        -------
        uint32
            Random number
        
        Notes
        -----
        Must match _quartets.py::_sample_quartet() XorShift step exactly.
        """
        t = state[3]
        s = state[0]
        
        # Rotate state
        state[3] = state[2]
        state[2] = state[1]
        state[1] = s
        
        # XorShift operations
        t ^= (t << np.uint32(11))
        t ^= (t >> np.uint32(8))
        state[0] = t ^ s ^ (s >> np.uint32(19))
        
        return state[0]

    @cuda.jit(device=True)
    def sample_4_unique_cuda(rng_state, n_taxa):
        """
        Sample 4 unique taxa indices using XorShift128.
        
        Parameters
        ----------
        rng_state : array of 4 uint32
            XorShift128 state (modified in place)
        n_taxa : int
            Namespace size
        
        Returns
        -------
        tuple of 4 ints
            Sampled indices (a, b, c, d) where a < b < c < d
        
        Notes
        -----
        Must match _quartets.py::_sample_quartet() exactly.
        Uses rejection sampling to ensure uniqueness.
        """
        # Use local array for samples
        samples = cuda.local.array(4, np.int32)
        n_samples = 0
        
        # Sample with rejection until we have 4 unique values
        while n_samples < 4:
            # Generate candidate
            rand_val = xorshift128_next(rng_state)
            candidate = np.int32(rand_val % n_taxa)
            
            # Check uniqueness
            is_unique = True
            for i in range(n_samples):
                if samples[i] == candidate:
                    is_unique = False
                    break
            
            if is_unique:
                samples[n_samples] = candidate
                n_samples += 1
        
        # Sort in-place using simple insertion sort (fast for 4 elements)
        for i in range(1, 4):
            key = samples[i]
            j = i - 1
            while j >= 0 and samples[j] > key:
                samples[j + 1] = samples[j]
                j -= 1
            samples[j + 1] = key
        
        return samples[0], samples[1], samples[2], samples[3]

    @cuda.jit(device=True)
    def get_quartet_at_index(
        absolute_idx,
        seed_quartets,
        n_seed,
        rng_seed,
        n_taxa
    ):
        """
        Get quartet at absolute index in the deterministic sequence.
        
        Parameters
        ----------
        absolute_idx : int
            Index in the infinite sequence
        seed_quartets : array, shape (n_seed, 4)
            Explicit seed quartets
        n_seed : int
            Number of seed quartets
        rng_seed : uint32
            Hash of seed quartets for RNG initialization
        n_taxa : int
            Namespace size for random generation
        
        Returns
        -------
        tuple of 4 ints
            Quartet (a, b, c, d) where a < b < c < d
        """
        if absolute_idx < n_seed:
            # Return seed quartet
            a = seed_quartets[absolute_idx, 0]
            b = seed_quartets[absolute_idx, 1]
            c = seed_quartets[absolute_idx, 2]
            d = seed_quartets[absolute_idx, 3]
            return a, b, c, d
        else:
            # Generate random quartet
            rng_state = cuda.local.array(4, np.uint32)
            init_xorshift128(rng_seed, absolute_idx - n_seed, rng_state)
            return sample_4_unique_cuda(rng_state, n_taxa)

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
            lg_start = lg_offsets[tree_idx]
            sp_stride = sp_tour_widths[tree_idx]
            
            # Get first occurrences in Euler tour
            occ_a = all_first_occ[node_start + local_a]
            occ_b = all_first_occ[node_start + local_b]
            occ_c = all_first_occ[node_start + local_c]
            occ_d = all_first_occ[node_start + local_d]
            
            # Find LCAs using RMQ
            left_ab = min(occ_a, occ_b)
            right_ab = max(occ_a, occ_b)
            lca_ab = _rmq_csr_cuda(
                left_ab, right_ab,
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )

            left_cd = min(occ_c, occ_d)
            right_cd = max(occ_c, occ_d)
            lca_cd = _rmq_csr_cuda(
                left_cd, right_cd,
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )

            left_ac = min(occ_a, occ_c)
            right_ac = max(occ_a, occ_c)
            lca_ac = _rmq_csr_cuda(
                left_ac, right_ac,
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            
            # Get root distances for LCAs
            rd_ab = all_root_distance[node_start + lca_ab]
            rd_cd = all_root_distance[node_start + lca_cd]
            rd_ac = all_root_distance[node_start + lca_ac]
            
            # Determine topology by comparing root distances
            # Topology 0: (AB|CD), Topology 1: (AC|BD), Topology 2: (AD|BC)
            if rd_ab >= rd_cd and rd_ab >= rd_ac:
                topology = 0
            elif rd_cd >= rd_ab and rd_cd >= rd_ac:
                topology = 1
            else:
                topology = 2
            
            # Accumulate count
            cuda.atomic.add(counts, (local_idx, topology), 1)

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
        # Outputs
        counts,             # [count, 3] int32
        steiner_out         # [count, n_trees, 3] float64
    ):
        """
        Unified kernel with Steiner distances.
        
        Same as quartet_counts_cuda_unified but also computes Steiner
        spanning lengths for winning topologies.
        """
        local_idx = cuda.grid(1)
        if local_idx >= count:
            return
        
        # Determine absolute index and get quartet
        absolute_idx = offset + local_idx
        a, b, c, d = get_quartet_at_index(
            absolute_idx, seed_quartets, n_seed, rng_seed, n_global_taxa
        )
        
        n_trees = node_offsets.shape[0] - 1
        
        # Process across all trees
        for tree_idx in range(n_trees):
            # Get local IDs
            local_a = global_to_local[tree_idx, a]
            local_b = global_to_local[tree_idx, b]
            local_c = global_to_local[tree_idx, c]
            local_d = global_to_local[tree_idx, d]
            
            # Check presence
            if local_a == -1 or local_b == -1 or local_c == -1 or local_d == -1:
                continue
            
            # Get offsets
            node_start = node_offsets[tree_idx]
            tour_start = tour_offsets[tree_idx]
            sp_start = sp_offsets[tree_idx]
            lg_start = lg_offsets[tree_idx]
            sp_stride = sp_tour_widths[tree_idx]
            
            # Get occurrences
            occ_a = all_first_occ[node_start + local_a]
            occ_b = all_first_occ[node_start + local_b]
            occ_c = all_first_occ[node_start + local_c]
            occ_d = all_first_occ[node_start + local_d]
            
            # Find all 6 LCAs for Steiner computation
            lca_ab = _rmq_csr_cuda(
                min(occ_a, occ_b), max(occ_a, occ_b),
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            lca_cd = _rmq_csr_cuda(
                min(occ_c, occ_d), max(occ_c, occ_d),
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            lca_ac = _rmq_csr_cuda(
                min(occ_a, occ_c), max(occ_a, occ_c),
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            lca_bd = _rmq_csr_cuda(
                min(occ_b, occ_d), max(occ_b, occ_d),
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            lca_ad = _rmq_csr_cuda(
                min(occ_a, occ_d), max(occ_a, occ_d),
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            lca_bc = _rmq_csr_cuda(
                min(occ_b, occ_c), max(occ_b, occ_c),
                sp_start, sp_stride, all_sparse_table, all_euler_depth,
                all_log2_table, lg_start, tour_start, all_euler_tour
            )
            
            # Get root distances
            rd_ab = all_root_distance[node_start + lca_ab]
            rd_cd = all_root_distance[node_start + lca_cd]
            rd_ac = all_root_distance[node_start + lca_ac]
            rd_bd = all_root_distance[node_start + lca_bd]
            rd_ad = all_root_distance[node_start + lca_ad]
            rd_bc = all_root_distance[node_start + lca_bc]
            
            # Determine topology
            if rd_ab >= rd_cd and rd_ab >= rd_ac:
                topology = 0
            elif rd_cd >= rd_ab and rd_cd >= rd_ac:
                topology = 1
            else:
                topology = 2
            
            # Compute Steiner distance for winning topology
            rd_a = all_root_distance[node_start + local_a]
            rd_b = all_root_distance[node_start + local_b]
            rd_c = all_root_distance[node_start + local_c]
            rd_d = all_root_distance[node_start + local_d]
            
            leaf_sum = rd_a + rd_b + rd_c + rd_d
            r0 = rd_ab + rd_cd
            r1 = rd_ac + rd_bd
            r2 = rd_ad + rd_bc
            
            if topology == 0:
                r_winner = r0
            elif topology == 1:
                r_winner = r1
            else:
                r_winner = r2
            
            steiner = leaf_sum - (r_winner + r0 + r1 + r2) / 2.0
            
            # Store results
            cuda.atomic.add(counts, (local_idx, topology), 1)
            steiner_out[local_idx, tree_idx, topology] = steiner


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
