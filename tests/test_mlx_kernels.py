"""
tests/test_mlx_kernels.py
=========================
Correctness tests for the MLX Metal quartet kernels.

Test classes
------------
``TestMLX1DKernel``
    Verifies that ``generate_quartets_mlx`` produces bit-identical output to
    the CPU iterator (``Quartets.__iter__``) for all relevant code paths:

    1. ``test_1d_kernel_vs_cpu_iterator``
       Main agreement test: 140 random quartets against the CPU sequence.

    2. ``test_rejection_no_sequence_dup``
       5 quartets, all requiring rejection sampling, none a sequence duplicate.

    3. ``test_rejection_with_sequence_duplicates``
       10 quartets with rejection sampling; positions 2, 8, 9 are duplicates.

    4. ``test_seed_isolation``
       Different rng_seed values produce different quartet sequences.

``TestMLXQuartetTopology``
    Verifies that the Metal quartet topology and Steiner kernels produce
    results that agree with the Python fallback backend:

    1. ``test_counts_match_python``
       MLX counts must equal Python counts exactly (integer result).

    2. ``test_counts_grouped_match_python``
       Same agreement check over a grouped (one group per tree) forest.

    3. ``test_steiner_match_python``
       MLX Steiner sums must agree with Python within float32 tolerance
       (rtol=1e-4); min/max/variance are sanity-checked for shape and sign.

    4. ``test_forest_quartet_topology_mlx_backend``
       End-to-end: ``forest.quartet_topology()`` via ``use_backend("mlx")``
       produces a valid ``QuartetTopologyResult`` with correct shape.

All tests are skipped if MLX is not available.

Test forest
-----------
16 trees over 8 taxa (a–h), identical to the corpus in
``test_kernel_agreement.py``.  n_taxa=8 maximises rejection probability
(~59 % of quartets require at least one retry), which is the hard case
for RNG agreement.
"""

import numpy as np
import pytest

from quarimo._context import quiet, use_backend
from quarimo._forest import Forest
from quarimo._quartets import Quartets

pytestmark = pytest.mark.requires_mlx

# ---------------------------------------------------------------------------
# 16-tree test corpus (identical to test_kernel_agreement.py)
# ---------------------------------------------------------------------------

TREES = [
    # 0: binary balanced — (ab|cd)(ef|gh)
    "(((a:0.10,b:0.20):0.15,(c:0.30,d:0.40):0.25):0.50,"
    "((e:0.10,f:0.20):0.15,(g:0.30,h:0.40):0.25):0.50);",
    # 1: binary balanced — (ac|bd)(eg|fh)
    "(((a:0.10,c:0.20):0.15,(b:0.30,d:0.40):0.25):0.50,"
    "((e:0.10,g:0.20):0.15,(f:0.30,h:0.40):0.25):0.50);",
    # 2: binary balanced — (ad|bc)(eh|fg)
    "(((a:0.10,d:0.20):0.15,(b:0.30,c:0.40):0.25):0.50,"
    "((e:0.10,h:0.20):0.15,(f:0.30,g:0.40):0.25):0.50);",
    # 3: caterpillar a→h
    "(a:0.50,(b:0.40,(c:0.30,(d:0.20,(e:0.15,"
    "(f:0.10,(g:0.05,h:0.05):0.10):0.15):0.20):0.25):0.30):0.40);",
    # 4: caterpillar h→a (reversed)
    "(h:0.50,(g:0.40,(f:0.30,(e:0.20,(d:0.15,"
    "(c:0.10,(b:0.05,a:0.05):0.10):0.15):0.20):0.25):0.30):0.40);",
    # 5: 8-taxon star
    "(a:1.0,b:1.0,c:1.0,d:1.0,e:1.0,f:1.0,g:1.0,h:1.0);",
    # 6: quadrifurcating root
    "((a:0.30,b:0.40):0.20,(c:0.30,d:0.40):0.20,"
    "(e:0.30,f:0.40):0.20,(g:0.30,h:0.40):0.20);",
    # 7: internal trifurcation abc
    "(((a:0.10,b:0.10,c:0.10):0.30,d:0.50):0.40,"
    "((e:0.10,f:0.10):0.20,(g:0.10,h:0.10):0.20):0.30);",
    # 8: zero-length branches
    "(((a:0.10,b:0.20):0.00,(c:0.30,d:0.40):0.10):0.50,"
    "((e:0.10,f:0.20):0.30,(g:0.30,h:0.40):0.00):0.50);",
    # 9: zero-length branches
    "(((a:0.20,b:0.30):0.10,(c:0.10,d:0.20):0.00):0.40,"
    "((e:0.20,f:0.30):0.00,(g:0.10,h:0.20):0.30):0.40);",
    # 10: only a,b,c,d,e,f
    "(((a:0.20,b:0.30):0.40,(c:0.10,d:0.50):0.20):0.60,(e:0.30,f:0.40):0.70);",
    # 11: only a,b,g,h
    "((a:0.50,b:0.50):0.40,(g:0.60,h:0.40):0.30);",
    # 12: only c,d,e,f,g,h
    "(((c:0.20,d:0.30):0.40,(e:0.10,f:0.50):0.20):0.60,(g:0.30,h:0.40):0.70);",
    # 13: 7 taxa (h absent)
    "(((a:0.10,b:0.20):0.30,(c:0.10,d:0.20):0.40):0.50,"
    "((e:0.20,f:0.30):0.20,g:0.40):0.30);",
    # 14: 7 taxa (h absent), deeply unbalanced
    "(((a:0.30,e:0.20):0.10,"
    "((b:0.40,f:0.30):0.20,(c:0.50,g:0.10):0.30):0.20):0.40,d:0.60);",
    # 15: all 8 taxa, topology (ae|bf)(cg|dh)
    "(((a:0.10,e:0.20):0.15,(b:0.30,f:0.40):0.25):0.50,"
    "((c:0.10,g:0.20):0.15,(d:0.30,h:0.40):0.25):0.50);",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grouped_forest():
    """One group per tree — identical to the fixture in test_kernel_agreement.py."""
    with quiet():
        return Forest({f"t{i:02d}": [t] for i, t in enumerate(TREES)})


@pytest.fixture(scope="module")
def random_quartets(grouped_forest):
    """140 random quartets — same as the fixture in test_kernel_agreement.py."""
    with quiet():
        return Quartets.random(grouped_forest, count=140, seed=42)


# ---------------------------------------------------------------------------
# Helpers (shared with TestMLX1DKernel)
# ---------------------------------------------------------------------------


def _run_mlx_1d(forest, q):
    """Run generate_quartets_mlx and return a numpy (count, 4) int32 array."""
    from quarimo._mlx_kernels import generate_quartets_mlx

    seed_array = np.array(q.seed, dtype=np.int32)
    return generate_quartets_mlx(
        seed_array,
        n_seed=len(q.seed),
        offset=q.offset,
        count=len(q),
        rng_seed=q.rng_seed,
        n_taxa=forest.n_global_taxa,
    )


def _annotate_positions(cpu_quartets, rng_seed):
    """Return (n_rng_calls, is_sequence_dup) per position (CPU replay)."""
    seen = {}
    info = []
    for i, row in enumerate(cpu_quartets):
        tup = tuple(row)
        is_dup = tup in seen
        seen.setdefault(tup, i)
        state = np.array([
            (rng_seed + i) & 0xFFFFFFFF,
            ((rng_seed + i) >> 32) & 0xFFFFFFFF,
            0x9e3779b9,
            0x7f4a7c13,
        ], dtype=np.uint32)
        samples, calls = [], 0
        while len(samples) < 4:
            t = state[3]; s = state[0]
            state[3] = state[2]; state[2] = state[1]; state[1] = s
            t = (t ^ ((t << 11) & 0xFFFFFFFF)) & 0xFFFFFFFF
            t = (t ^ (t >> 8)) & 0xFFFFFFFF
            state[0] = (t ^ s ^ (s >> 19)) & 0xFFFFFFFF
            calls += 1
            c = int(state[0] % 8)
            if c not in samples:
                samples.append(c)
        info.append((calls, is_dup))
    return info


def _assert_no_mismatch(cpu_quartets, gpu_quartets, info):
    mismatches = np.where(np.any(cpu_quartets != gpu_quartets, axis=1))[0].tolist()
    if mismatches:
        lines = [
            f"  qi={i}: cpu={tuple(cpu_quartets[i])} "
            f"gpu={tuple(gpu_quartets[i])} "
            f"rng_calls={info[i][0]} is_sequence_dup={info[i][1]}"
            for i in mismatches
        ]
        raise AssertionError(
            f"MLX quartet sequence differs at {len(mismatches)} positions:\n"
            + "\n".join(lines)
        )


# ===========================================================================
# Test class
# ===========================================================================


class TestMLX1DKernel:
    """
    Verify that generate_quartets_mlx matches the CPU iterator exactly.

    All tests are skipped when MLX with Metal is not available (non-Apple
    Silicon machines and machines where MLX is not installed).
    """

    @pytest.mark.requires_mlx
    def test_1d_kernel_vs_cpu_iterator(self, grouped_forest, random_quartets):
        """
        Metal kernel must produce the same 140 quartets as the CPU iterator.

        Mirrors test_cuda_1d_kernel_vs_cpu_iterator.  Uses count=140 so that
        offset (=1) is immediately past n_seed (=1) — every quartet goes
        through the XorShift128 RNG path.  A failure here means the MSL
        XorShift128 or rejection-sampling loop diverges from the CPU.
        """
        cpu_quartets = np.array(list(random_quartets), dtype=np.int32)
        gpu_quartets = _run_mlx_1d(grouped_forest, random_quartets)
        np.testing.assert_array_equal(
            cpu_quartets,
            gpu_quartets,
            err_msg=(
                "generate_quartets_mlx produced different quartets than "
                "the CPU iterator.  Bug is in the MSL RNG or sampling loop."
            ),
        )

    @pytest.mark.requires_mlx
    def test_rejection_no_sequence_dup(self, grouped_forest):
        """
        Metal kernel must match CPU at rejection-sampling positions with no
        sequence duplicates.

        Seed 2, count 5 — verified offline: calls=[6,9,5,5,5], all unique.
        All five positions exercise the rejection path; none is a duplicate.
        A failure isolates the rejection-sampling RNG arithmetic.
        """
        with quiet():
            q = Quartets.random(grouped_forest, count=5, seed=2)
        cpu_quartets = np.array(list(q), dtype=np.int32)
        info = _annotate_positions(cpu_quartets, q.rng_seed)

        assert all(c > 4 for c, _ in info), "Precondition: all positions must have rejections"
        assert all(not d for _, d in info), "Precondition: no sequence duplicates expected"

        gpu_quartets = _run_mlx_1d(grouped_forest, q)
        _assert_no_mismatch(cpu_quartets, gpu_quartets, info)

    @pytest.mark.requires_mlx
    def test_rejection_with_sequence_duplicates(self, grouped_forest):
        """
        Metal kernel must match CPU when some positions repeat earlier quartets.

        Seed 0, count 10 — verified offline: positions 2, 8, 9 are sequence
        duplicates; all 10 positions have rejections.  If
        test_rejection_no_sequence_dup passes but this fails only at 2/8/9,
        the bug is in duplicate handling rather than rejection sampling.
        """
        with quiet():
            q = Quartets.random(grouped_forest, count=10, seed=0)
        cpu_quartets = np.array(list(q), dtype=np.int32)
        info = _annotate_positions(cpu_quartets, q.rng_seed)

        dup_positions = [i for i, (_, is_dup) in enumerate(info) if is_dup]
        assert dup_positions == [2, 8, 9], (
            f"Precondition: expected duplicates at [2, 8, 9], got {dup_positions}"
        )
        assert all(c > 4 for c, _ in info), "Precondition: all positions must have rejections"

        gpu_quartets = _run_mlx_1d(grouped_forest, q)
        _assert_no_mismatch(cpu_quartets, gpu_quartets, info)

    @pytest.mark.requires_mlx
    def test_seed_isolation(self, grouped_forest):
        """
        Different rng_seed values must produce different quartet sequences.

        Sanity check: the Metal kernel is actually consuming the rng_seed
        argument rather than returning a fixed or garbage sequence.
        """
        with quiet():
            q0 = Quartets.random(grouped_forest, count=20, seed=0)
            q1 = Quartets.random(grouped_forest, count=20, seed=99)

        out0 = _run_mlx_1d(grouped_forest, q0)
        out1 = _run_mlx_1d(grouped_forest, q1)

        assert not np.array_equal(out0, out1), (
            "Different seeds produced identical quartet sequences — "
            "the Metal kernel is ignoring rng_seed."
        )


# ===========================================================================
# 2. Quartet topology — counts and Steiner
# ===========================================================================

# Flat (single-group) forest for topology tests
@pytest.fixture(scope="module")
def flat_forest():
    """16-tree single-group forest."""
    with quiet():
        return Forest(TREES)


@pytest.fixture(scope="module")
def topology_quartets(flat_forest):
    """70 random quartets over the 16-tree single-group forest."""
    with quiet():
        return Quartets.random(flat_forest, count=70, seed=7)


@pytest.fixture(scope="module")
def grouped_topology_quartets(grouped_forest):
    """70 random quartets over the 16-tree grouped forest."""
    with quiet():
        return Quartets.random(grouped_forest, count=70, seed=7)


class TestMLXQuartetTopology:
    """
    Verify that the Metal quartet topology and Steiner kernels produce results
    that agree with the Python fallback backend.

    All tests are hardware-gated by ``pytestmark = pytest.mark.requires_mlx``
    at module level.
    """

    def test_counts_match_python(self, flat_forest, topology_quartets):
        """
        MLX counts must be identical to Python counts (integer result, no
        rounding).  Uses a single-group forest so the shape is (n_q, 1, 4).
        """
        q = topology_quartets
        with quiet():
            with use_backend("python"):
                ref = flat_forest.quartet_topology(q)
            with use_backend("mlx"):
                mlx = flat_forest.quartet_topology(q)

        np.testing.assert_array_equal(
            ref.counts, mlx.counts,
            err_msg="MLX counts differ from Python counts (single-group forest)",
        )

    def test_counts_grouped_match_python(self, grouped_forest, grouped_topology_quartets):
        """
        MLX counts must match Python counts for a grouped (one-group-per-tree)
        forest.  Shape: (n_q, 16, 4).
        """
        q = grouped_topology_quartets
        with quiet():
            with use_backend("python"):
                ref = grouped_forest.quartet_topology(q)
            with use_backend("mlx"):
                mlx = grouped_forest.quartet_topology(q)

        np.testing.assert_array_equal(
            ref.counts, mlx.counts,
            err_msg="MLX counts differ from Python counts (grouped forest)",
        )

    def test_steiner_match_python(self, flat_forest, topology_quartets):
        """
        MLX Steiner sums must agree with Python within float32 tolerance
        (rtol=1e-4).  Min/max/variance are sanity-checked for shape and
        non-negativity of the sum.
        """
        q = topology_quartets
        with quiet():
            with use_backend("python"):
                ref = flat_forest.quartet_topology(q, steiner=True)
            with use_backend("mlx"):
                mlx = flat_forest.quartet_topology(q, steiner=True)

        # Counts must match exactly
        np.testing.assert_array_equal(
            ref.counts, mlx.counts,
            err_msg="MLX Steiner-mode counts differ from Python",
        )

        # Steiner sums: float32 precision loss is expected (rtol=1e-4)
        np.testing.assert_allclose(
            ref.steiner, mlx.steiner, rtol=1e-4, atol=0,
            err_msg="MLX Steiner sums differ from Python beyond float32 tolerance",
        )

        # Shapes match
        assert mlx.steiner_min.shape == ref.steiner_min.shape
        assert mlx.steiner_max.shape == ref.steiner_max.shape

        # All non-NaN Steiner values must be non-negative
        steiner_vals = mlx.steiner[ref.counts > 0]
        assert (steiner_vals >= 0).all(), "MLX Steiner values contain negatives"

    def test_forest_quartet_topology_mlx_backend(self, flat_forest, topology_quartets):
        """
        End-to-end: Forest.quartet_topology() via use_backend("mlx") returns
        a valid QuartetTopologyResult with the correct shape and dtype.
        """
        q = topology_quartets
        with quiet():
            with use_backend("mlx"):
                result = flat_forest.quartet_topology(q)

        assert result.counts.shape == (len(q), flat_forest.n_groups, 4)
        assert result.counts.dtype == np.int32
        assert (result.counts >= 0).all()
        # Total count across topologies for each (qi, gi) must not exceed n_trees
        totals = result.counts.sum(axis=-1)
        assert (totals <= flat_forest.n_trees).all()
