"""
GPU device functions for random quartet generation.

These functions implement XorShift128 RNG and quartet sampling on GPU.
Must match the CPU implementation in _quartets.py exactly.

[MORTON_SCHED] functions are gated behind the morton_order flag in the
dispatch layer.  To remove the Morton path: delete everything tagged
[MORTON_SCHED] in this file and in _cuda_kernels.py / _forest.py.
To remove the standard path: delete [STD_SCHED] tagged code and rename
the Morton variants to drop the _morton suffix.
"""

from numba import cuda
import numba


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
    combined = numba.uint64(base_seed) + numba.uint64(offset)
    
    # Initialize 4-word state
    state[0] = numba.uint32(combined & 0xFFFFFFFF)
    state[1] = numba.uint32((combined >> 32) & 0xFFFFFFFF)
    state[2] = numba.uint32(0x9e3779b9)  # Golden ratio
    state[3] = numba.uint32(0x7f4a7c13)  # Arbitrary constant


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
    t = numba.uint32(state[3])
    s = numba.uint32(state[0])

    # Rotate state
    state[3] = state[2]
    state[2] = state[1]
    state[1] = s

    # XorShift operations — explicit uint32 casts ensure logical (not arithmetic) right shifts
    t = numba.uint32(t ^ numba.uint32(t << numba.uint32(11)))
    t = numba.uint32(t ^ numba.uint32(t >> numba.uint32(8)))
    state[0] = numba.uint32(t ^ s ^ numba.uint32(s >> numba.uint32(19)))

    return numba.uint32(state[0])


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
    samples = cuda.local.array(4, numba.int32)
    n_samples = 0
    
    # Sample with rejection until we have 4 unique values
    while n_samples < 4:
        # Generate candidate
        rand_val = xorshift128_next(rng_state)
        candidate = numba.int32(rand_val % n_taxa)
        
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
        # [STD_SCHED] Return seed quartet directly.
        a = seed_quartets[absolute_idx, 0]
        b = seed_quartets[absolute_idx, 1]
        c = seed_quartets[absolute_idx, 2]
        d = seed_quartets[absolute_idx, 3]
        return a, b, c, d
    else:
        # [STD_SCHED] Generate random quartet from XorShift sequence.
        rng_state = cuda.local.array(4, numba.uint32)
        init_xorshift128(rng_seed, absolute_idx - n_seed, rng_state)
        return sample_4_unique_cuda(rng_state, n_taxa)


# ============================================================================ #
# [MORTON_SCHED] Morton-ordered quartet generation                             #
# ============================================================================ #
#
# Two-level structure:
#   Level 1 — Morton block counter (deterministic, advancing with qi):
#       block_idx = qi // M  (M = quartets per block, passed by caller)
#       Decodes to 4 DFS-subrange prefixes, one per quartet dimension.
#       All quartets in the same block access the same DFS subregion of each
#       tree's sparse table, keeping it warm in L2 across the block.
#
#   Level 2 — within-block XorShift (fast, random):
#       within_idx = qi % M
#       XorShift128 seeded from (rng_seed, block_idx, within_idx) generates
#       the low-bit DFS offset within each dimension's subrange.
#
# Tag: [MORTON_SCHED].  Remove this entire section to revert to [STD_SCHED].


@cuda.jit(device=True)
def _morton_extract_dim(block_idx, dim, n_bits):
    """
    [MORTON_SCHED] Extract the n_bits-bit prefix for one quartet dimension
    from a 4D Morton block index.

    A 4D Morton code interleaves 4 sequences of bits: bits at positions
    4k+dim (k=0,1,...,n_bits-1) all belong to dimension ``dim``.  This
    function gathers those bits into a contiguous integer.

    Parameters
    ----------
    block_idx : int
        Morton block index (at most 4*n_bits bits wide).
    dim : int
        Dimension index 0–3.
    n_bits : int
        Number of bits per dimension (block depth).

    Returns
    -------
    int
        The n_bits-bit DFS prefix for the requested dimension.
    """
    prefix = numba.uint32(0)
    for k in range(n_bits):
        bit = numba.uint32((block_idx >> numba.uint32(4 * k + dim)) & numba.uint32(1))
        prefix |= numba.uint32(bit << numba.uint32(k))
    return prefix


@cuda.jit(device=True)
def get_quartet_morton_at_index(
    qi,
    M,
    rng_seed,
    gid_sorted_by_dfs,
    n_taxa,
    n_bits_per_dim,
):
    """
    [MORTON_SCHED] Generate the quartet at position qi in the Morton-ordered
    sequence.

    Parameters
    ----------
    qi : int
        Global quartet index within this kernel launch (0 … count-1).
    M : int
        Quartets per Morton block.  block_idx = qi // M.
    rng_seed : uint32
        Base XorShift seed (same as the standard path).
    gid_sorted_by_dfs : int32 device array [n_taxa]
        Maps DFS rank i → global taxon ID.  Built once at Forest init.
    n_taxa : int
        Total number of global taxa.
    n_bits_per_dim : int
        Morton block depth: each dimension gets n_bits_per_dim high bits
        from the block index.  block_width = n_taxa >> n_bits_per_dim.

    Returns
    -------
    a, b, c, d : int32
        Sorted global taxon IDs (a < b < c < d).

    Notes
    -----
    - block_width = n_taxa >> n_bits_per_dim (integer shift, may be 0 for very
      small forests — caller should guard with n_bits_per_dim <= log2(n_taxa)//4).
    - Rejection-sampling handles collisions (two dimensions landing on the same
      GID) and edge effects at the last block (subrange narrower than block_width).
    - All integer operations use explicit numba.uint32 casts to guarantee
      logical right shifts (matches the discipline in xorshift128_next).
    """
    block_idx  = numba.uint32(qi // M)
    within_idx = numba.uint32(qi % M)

    block_width = numba.uint32(n_taxa >> numba.uint32(n_bits_per_dim))
    if block_width < numba.uint32(1):
        block_width = numba.uint32(1)

    # Extract per-dimension DFS subrange starts from the Morton block index.
    prefix0 = _morton_extract_dim(block_idx, 0, n_bits_per_dim)
    prefix1 = _morton_extract_dim(block_idx, 1, n_bits_per_dim)
    prefix2 = _morton_extract_dim(block_idx, 2, n_bits_per_dim)
    prefix3 = _morton_extract_dim(block_idx, 3, n_bits_per_dim)

    start0 = numba.uint32(prefix0 * block_width)
    start1 = numba.uint32(prefix1 * block_width)
    start2 = numba.uint32(prefix2 * block_width)
    start3 = numba.uint32(prefix3 * block_width)

    # Clamp each subrange to [0, n_taxa).
    end0 = numba.uint32(min(start0 + block_width, numba.uint32(n_taxa)))
    end1 = numba.uint32(min(start1 + block_width, numba.uint32(n_taxa)))
    end2 = numba.uint32(min(start2 + block_width, numba.uint32(n_taxa)))
    end3 = numba.uint32(min(start3 + block_width, numba.uint32(n_taxa)))

    # Seed within-block XorShift from (rng_seed, block_idx, within_idx) so
    # that two threads in the same block but different within-block positions
    # produce independent sequences.
    rng_state = cuda.local.array(4, numba.uint32)
    combined = numba.uint64(rng_seed) ^ (numba.uint64(block_idx) << 32) ^ numba.uint64(within_idx)
    rng_state[0] = numba.uint32(combined & numba.uint64(0xFFFFFFFF))
    rng_state[1] = numba.uint32((combined >> numba.uint64(32)) & numba.uint64(0xFFFFFFFF))
    rng_state[2] = numba.uint32(0x9e3779b9)
    rng_state[3] = numba.uint32(0x7f4a7c13)

    # Sample one DFS index per dimension, reject collisions.
    samples = cuda.local.array(4, numba.int32)
    n_samples = 0
    while n_samples < 4:
        # Pick dimension and its subrange.
        if n_samples == 0:
            width = end0 - start0
            start = start0
        elif n_samples == 1:
            width = end1 - start1
            start = start1
        elif n_samples == 2:
            width = end2 - start2
            start = start2
        else:
            width = end3 - start3
            start = start3

        if width == numba.uint32(0):
            # Empty subrange — fall back to any position in [0, n_taxa).
            rand_val = xorshift128_next(rng_state)
            dfs_idx = numba.int32(rand_val % numba.uint32(n_taxa))
        else:
            rand_val = xorshift128_next(rng_state)
            dfs_idx = numba.int32(start + rand_val % width)

        gid = gid_sorted_by_dfs[dfs_idx]

        # Uniqueness check across all accepted samples so far.
        is_unique = True
        for i in range(n_samples):
            if samples[i] == gid:
                is_unique = False
                break

        if is_unique:
            samples[n_samples] = gid
            n_samples += 1

    # Insertion sort (optimal for 4 elements).
    for i in range(1, 4):
        key = samples[i]
        j = i - 1
        while j >= 0 and samples[j] > key:
            samples[j + 1] = samples[j]
            j -= 1
        samples[j + 1] = key

    return samples[0], samples[1], samples[2], samples[3]
