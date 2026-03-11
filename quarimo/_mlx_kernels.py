"""
_mlx_kernels.py
===============
MLX Metal quartet generation kernel for Apple Silicon.

Prototype scope
---------------
Exposes only ``generate_quartets_mlx`` — the direct equivalent of the CUDA
``generate_quartets_cuda`` 1D kernel.  Full ``quartet_counts`` /
``quartet_steiner`` Metal kernels are future work; the dispatch path in
``_forest.py`` is not yet wired to this backend.

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
``check_mlx_available()`` is called by ``_backend.py`` and is safe to call
on any platform — it imports MLX, runs a trivial Metal operation, and returns
False if either step fails.
"""

from __future__ import annotations

_MLX_AVAILABLE = False

try:
    import mlx.core as mx  # noqa: F401

    # Probe Metal availability: importing mlx succeeds even on machines without
    # Metal.  A trivial eval forces the Metal device to initialise.
    mx.eval(mx.array([0], dtype=mx.int32))
    _MLX_AVAILABLE = True
except Exception:
    pass


def check_mlx_available() -> bool:
    """Return True if MLX with Metal GPU compute is available on this machine."""
    return _MLX_AVAILABLE


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
        import numpy as np

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
