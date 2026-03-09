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

    @cuda.jit(device=True)
    def _resolve_quartet_cuda(n0, n1, n2, n3, ti,
                              global_to_local,
                              node_offsets, tour_offsets, sp_offsets, lg_offsets,
                              sp_tour_widths):
        """
        Map four global taxon IDs to tree-local positions for tree *ti*.

        CUDA device counterpart of ``_resolve_quartet_nb``; see that function's
        docstring for the full return-value description and caller contract.
        All arrays are device-resident.
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

    @cuda.jit(device=True)
    def _quartet_topology_and_rd_cuda(occ0, occ1, occ2, occ3,
                                       nb, tb, sb, lb, tw,
                                       all_root_distance,
                                       all_sparse_table, all_euler_depth,
                                       all_log2_table, all_euler_tour):
        """
        Six RMQ calls + four-point condition → topology and pair-sums.

        CUDA device counterpart of ``_quartet_topology_and_rd_nb``; see that
        function's docstring for parameter and return-value descriptions.
        All arrays are device-resident.
        """
        l = occ0; r = occ1
        if l > r: l, r = r, l
        rd01 = all_root_distance[nb + _rmq_csr_cuda(l, r, sb, tw, all_sparse_table,
                                                      all_euler_depth, all_log2_table,
                                                      lb, tb, all_euler_tour)]
        l = occ0; r = occ2
        if l > r: l, r = r, l
        rd02 = all_root_distance[nb + _rmq_csr_cuda(l, r, sb, tw, all_sparse_table,
                                                      all_euler_depth, all_log2_table,
                                                      lb, tb, all_euler_tour)]
        l = occ0; r = occ3
        if l > r: l, r = r, l
        rd03 = all_root_distance[nb + _rmq_csr_cuda(l, r, sb, tw, all_sparse_table,
                                                      all_euler_depth, all_log2_table,
                                                      lb, tb, all_euler_tour)]
        l = occ1; r = occ2
        if l > r: l, r = r, l
        rd12 = all_root_distance[nb + _rmq_csr_cuda(l, r, sb, tw, all_sparse_table,
                                                      all_euler_depth, all_log2_table,
                                                      lb, tb, all_euler_tour)]
        l = occ1; r = occ3
        if l > r: l, r = r, l
        rd13 = all_root_distance[nb + _rmq_csr_cuda(l, r, sb, tw, all_sparse_table,
                                                      all_euler_depth, all_log2_table,
                                                      lb, tb, all_euler_tour)]
        l = occ2; r = occ3
        if l > r: l, r = r, l
        rd23 = all_root_distance[nb + _rmq_csr_cuda(l, r, sb, tw, all_sparse_table,
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

    @cuda.jit(device=True)
    def _steiner_length_cuda(ln0, ln1, ln2, ln3, nb, r0, r1, r2, r_winner, all_root_distance):
        """
        Steiner spanning length of the winning quartet topology.

        CUDA device counterpart of ``_steiner_length_nb``; see that function's
        docstring for parameter and return-value descriptions.
        ``all_root_distance`` is device-resident.
        """
        leaf_sum = (all_root_distance[nb + ln0]
                  + all_root_distance[nb + ln1]
                  + all_root_distance[nb + ln2]
                  + all_root_distance[nb + ln3])
        return leaf_sum - (r_winner + r0 + r1 + r2) * 0.5

    @cuda.jit(device=True)
    def _accumulate_steiner_cuda(qi, gi, topo, sl,
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
        steiner_out : float64 device array [n_quartets, n_groups, 4]
        steiner_min_out : float64 device array [n_quartets, n_groups, 4]
        steiner_max_out : float64 device array [n_quartets, n_groups, 4]
        steiner_sum_sq_out : float64 device array [n_quartets, n_groups, 4]
        """
        cuda.atomic.add(steiner_out, (qi, gi, topo), sl)
        cuda.atomic.min(steiner_min_out, (qi, gi, topo), sl)
        cuda.atomic.max(steiner_max_out, (qi, gi, topo), sl)
        cuda.atomic.add(steiner_sum_sq_out, (qi, gi, topo), sl * sl)

    @cuda.jit(device=True)
    def _polytomy_check_cuda(occ0, occ1, occ2, occ3,
                              nb, tb, sb, lb, tw,
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

        l = occ0; r = occ1
        if l > r: l, r = r, l
        lca01 = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                               all_log2_table, lb, tb, all_euler_tour)
        l = occ0; r = occ2
        if l > r: l, r = r, l
        lca02 = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                               all_log2_table, lb, tb, all_euler_tour)
        l = occ0; r = occ3
        if l > r: l, r = r, l
        lca03 = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                               all_log2_table, lb, tb, all_euler_tour)
        l = occ1; r = occ2
        if l > r: l, r = r, l
        lca12 = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                               all_log2_table, lb, tb, all_euler_tour)
        l = occ1; r = occ3
        if l > r: l, r = r, l
        lca13 = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                               all_log2_table, lb, tb, all_euler_tour)
        l = occ2; r = occ3
        if l > r: l, r = r, l
        lca23 = _rmq_csr_cuda(l, r, sb, tw, all_sparse_table, all_euler_depth,
                               all_log2_table, lb, tb, all_euler_tour)

        for j in range(poly_start, poly_end):
            pn = polytomy_nodes[j]
            if (pn == lca01 or pn == lca02 or pn == lca03
                    or pn == lca12 or pn == lca13 or pn == lca23):
                rd01 = all_root_distance[nb + lca01]; rd23 = all_root_distance[nb + lca23]
                rd02 = all_root_distance[nb + lca02]; rd13 = all_root_distance[nb + lca13]
                rd03 = all_root_distance[nb + lca03]; rd12 = all_root_distance[nb + lca12]
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
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]
        ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw = _resolve_quartet_cuda(
            n0, n1, n2, n3, ti,
            global_to_local, node_offsets, tour_offsets, sp_offsets,
            lg_offsets, sp_tour_widths,
        )
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        occ0 = all_first_occ[nb + ln0]
        occ1 = all_first_occ[nb + ln1]
        occ2 = all_first_occ[nb + ln2]
        occ3 = all_first_occ[nb + ln3]
        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
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
        n0 = sorted_quartet_ids[qi, 0]
        n1 = sorted_quartet_ids[qi, 1]
        n2 = sorted_quartet_ids[qi, 2]
        n3 = sorted_quartet_ids[qi, 3]
        ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw = _resolve_quartet_cuda(
            n0, n1, n2, n3, ti,
            global_to_local, node_offsets, tour_offsets, sp_offsets,
            lg_offsets, sp_tour_widths,
        )
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        occ0 = all_first_occ[nb + ln0]
        occ1 = all_first_occ[nb + ln1]
        occ2 = all_first_occ[nb + ln2]
        occ3 = all_first_occ[nb + ln3]
        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
            all_root_distance, all_sparse_table, all_euler_depth,
            all_log2_table, all_euler_tour,
        )

        # Atomic increment for counts
        cuda.atomic.add(counts_out, (qi, topo), 1)

        # Compute and store Steiner distance (conflict-free write)
        steiner_out[qi, ti, topo] = _steiner_length_cuda(
            ln0, ln1, ln2, ln3, nb, r0, r1, r2, r_winner, all_root_distance,
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
        ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw = _resolve_quartet_cuda(
            a, b, c, d, ti,
            global_to_local, node_offsets, tour_offsets, sp_offsets,
            lg_offsets, sp_tour_widths,
        )
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        occ0 = all_first_occ[nb + ln0]
        occ1 = all_first_occ[nb + ln1]
        occ2 = all_first_occ[nb + ln2]
        occ3 = all_first_occ[nb + ln3]

        # CSR-based polytomy detection (zero overhead for trees without polytomies)
        poly_start = polytomy_offsets[ti]
        poly_end = polytomy_offsets[ti + 1]
        found, topo, r0, r1, r2, rw = _polytomy_check_cuda(
            occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
            poly_start, poly_end, polytomy_nodes,
            all_sparse_table, all_euler_depth, all_log2_table,
            all_euler_tour, all_root_distance,
        )
        if found:
            cuda.atomic.add(counts, (qi, tree_to_group_idx[ti], topo), 1)
            return

        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
            all_root_distance, all_sparse_table, all_euler_depth,
            all_log2_table, all_euler_tour,
        )

        gi = tree_to_group_idx[ti]
        cuda.atomic.add(counts, (qi, gi, topo), 1)

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
        ln0, ln1, ln2, ln3, nb, tb, sb, lb, tw = _resolve_quartet_cuda(
            a, b, c, d, ti,
            global_to_local, node_offsets, tour_offsets, sp_offsets,
            lg_offsets, sp_tour_widths,
        )
        # Skip if any taxon absent — steiner_out is pre-initialised to 0 by host.
        if ln0 < 0 or ln1 < 0 or ln2 < 0 or ln3 < 0:
            return

        occ0 = all_first_occ[nb + ln0]
        occ1 = all_first_occ[nb + ln1]
        occ2 = all_first_occ[nb + ln2]
        occ3 = all_first_occ[nb + ln3]

        # CSR-based polytomy detection (zero overhead for trees without polytomies)
        poly_start = polytomy_offsets[ti]
        poly_end = polytomy_offsets[ti + 1]
        found, topo, r0, r1, r2, rw = _polytomy_check_cuda(
            occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
            poly_start, poly_end, polytomy_nodes,
            all_sparse_table, all_euler_depth, all_log2_table,
            all_euler_tour, all_root_distance,
        )
        if found:
            gi = tree_to_group_idx[ti]
            sl = _steiner_length_cuda(ln0, ln1, ln2, ln3, nb, r0, r1, r2, rw, all_root_distance)
            cuda.atomic.add(counts, (qi, gi, topo), 1)
            _accumulate_steiner_cuda(
                qi, gi, topo, sl,
                steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
            )
            return

        topo, r0, r1, r2, r_winner = _quartet_topology_and_rd_cuda(
            occ0, occ1, occ2, occ3, nb, tb, sb, lb, tw,
            all_root_distance, all_sparse_table, all_euler_depth,
            all_log2_table, all_euler_tour,
        )

        gi = tree_to_group_idx[ti]
        sl = _steiner_length_cuda(ln0, ln1, ln2, ln3, nb, r0, r1, r2, r_winner, all_root_distance)
        cuda.atomic.add(counts, (qi, gi, topo), 1)
        _accumulate_steiner_cuda(
            qi, gi, topo, sl,
            steiner_out, steiner_min_out, steiner_max_out, steiner_sum_sq_out,
        )


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
