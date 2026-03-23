"""
GPU device functions for random quartet generation.

These functions implement XorShift128 RNG and quartet sampling on GPU.
Must match the CPU implementation in _quartets.py exactly.
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
        a = seed_quartets[absolute_idx, 0]
        b = seed_quartets[absolute_idx, 1]
        c = seed_quartets[absolute_idx, 2]
        d = seed_quartets[absolute_idx, 3]
        return a, b, c, d
    else:
        rng_state = cuda.local.array(4, numba.uint32)
        init_xorshift128(rng_seed, absolute_idx - n_seed, rng_state)
        return sample_4_unique_cuda(rng_state, n_taxa)

