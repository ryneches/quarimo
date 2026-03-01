"""
tests/test_kernel_agreement.py
==============================

Cross-validation between quarimo backends and against independent pairwise
branch-distance computations, plus cross-validation against SuchTree.

Validation layers
-----------------
1. Backend agreement  (TestBackendAgreement)
   Run quartet_topology() with every available backend pair (python vs
   cpu-parallel, python vs cuda) and assert bit-identical counts and
   allclose Steiner distances.

2. Four-point condition  (TestFourPointCondition)
   For every (quartet, tree) pair where all four taxa are present, the
   topology reported by the kernel must minimise the appropriate pairwise-
   distance pair-sum.  Formally, for sorted taxa (n0,n1,n2,n3) define

       s0 = d(n0,n1) + d(n2,n3)   # topo 0: (n0n1)|(n2n3)
       s1 = d(n0,n2) + d(n1,n3)   # topo 1: (n0n2)|(n1n3)
       s2 = d(n0,n3) + d(n1,n2)   # topo 2: (n0n3)|(n1n2)

   The winning topology k must satisfy s[k] == min(s0, s1, s2).  For binary
   trees the minimum is strict and unique; for polytomies the minimum may be
   tied—the kernel's tie-breaking is arbitrary, but it must still not violate
   the condition.

3. Steiner formula  (TestSteinerVsPairwise)
   The Steiner spanning length satisfies

       S = ( max(s0,s1,s2) + min(s0,s1,s2) ) / 2

   where s0, s1, s2 are the same pair-sums used for topology detection (and
   derived independently from branch_distance).  This formula holds for both
   binary trees and polytomies, and is derived from the quarimo closed-form
   Steiner expression.

4. SuchTree cross-validation  (TestSuchTreeAgreement)
   For every (quartet, tree) pair where the winning topology is unambiguous
   (pair-sums are not tied), quarimo and SuchTree must report the same
   topology.  Tied pairs are handled separately.

   SuchTree represents topologies as frozensets of frozensets:
       frozenset({frozenset({name_i, name_j}), frozenset({name_k, name_l})})
   quarimo uses k ∈ {0, 1, 2} for sorted taxa (n0,n1,n2,n3):
       k=0 → (n0n1)|(n2n3),  k=1 → (n0n2)|(n1n3),  k=2 → (n0n3)|(n1n2)
   _st_topo_to_k() performs the bijection between the two representations.

5. SuchTree polytomy cross-validation  (TestSuchTreePolytomyAgreement)
   [pytest.mark.polytomy — skip with -m 'not polytomy']
   For tied (quartet, tree) pairs, both quarimo and SuchTree must choose a
   topology whose pair-sum equals the minimum.  Exact agreement between the
   two implementations is NOT required: tie-breaking is implementation-defined
   and will be re-examined when quarimo's polytomy handling changes.

Test forest
-----------
16 trees over 8 taxa (a–h), giving C(8,4) = 70 quartets, 16 groups × 70
quartets = 1 120 (quartet, tree) pairs.

Trees 0–2  : binary balanced, three distinct pairings         — binary, resolved
Trees 3–4  : caterpillar ladderized both ways                 — binary, resolved
Trees 5    : 8-taxon star                                     — total polytomy
Tree  6    : quadrifurcating root                             — soft polytomy
Tree  7    : internal trifurcation (a,b,c joined at one node) — soft polytomy
Trees 8–9  : zero-length internal branches                    — binary, but
                                                                some LCA depths
                                                                coincide
Trees 10–12: trees with 4–6 of the 8 taxa                    — absent-taxon cases
Trees 13–14: 7-taxon trees (h absent), different topologies  — partial overlap
Tree  15   : all 8 taxa, topology ae|bf / cg|dh              — additional binary
"""

import itertools

import numpy as np
import pytest

from quarimo._backend import get_available_backends
from quarimo._context import quiet, silent_benchmark
from quarimo._forest import Forest
from quarimo._quartets import Quartets

# ---------------------------------------------------------------------------
# 16-tree test corpus
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
    # 5: 8-taxon star — all quartet pair-sums equal
    "(a:1.0,b:1.0,c:1.0,d:1.0,e:1.0,f:1.0,g:1.0,h:1.0);",
    # 6: quadrifurcating root — cross-group quartet pair-sums all tied
    "((a:0.30,b:0.40):0.20,(c:0.30,d:0.40):0.20,"
    "(e:0.30,f:0.40):0.20,(g:0.30,h:0.40):0.20);",
    # 7: internal trifurcation abc — quartets (a,b,c,X) are tied
    "(((a:0.10,b:0.10,c:0.10):0.30,d:0.50):0.40,"
    "((e:0.10,f:0.10):0.20,(g:0.10,h:0.10):0.20):0.30);",
    # 8: zero-length branch between (ab) node and its sibling,
    #    and between (gh) node and its parent — induces equal LCA depths
    "(((a:0.10,b:0.20):0.00,(c:0.30,d:0.40):0.10):0.50,"
    "((e:0.10,f:0.20):0.30,(g:0.30,h:0.40):0.00):0.50);",
    # 9: zero-length branch between (cd) node and its sibling,
    #    and between (ef) node and its sibling
    "(((a:0.20,b:0.30):0.10,(c:0.10,d:0.20):0.00):0.40,"
    "((e:0.20,f:0.30):0.00,(g:0.10,h:0.20):0.30):0.40);",
    # 10: only a,b,c,d,e,f (g,h absent)
    "(((a:0.20,b:0.30):0.40,(c:0.10,d:0.50):0.20):0.60,(e:0.30,f:0.40):0.70);",
    # 11: only a,b,g,h (c,d,e,f absent)
    "((a:0.50,b:0.50):0.40,(g:0.60,h:0.40):0.30);",
    # 12: only c,d,e,f,g,h (a,b absent)
    "(((c:0.20,d:0.30):0.40,(e:0.10,f:0.50):0.20):0.60,(g:0.30,h:0.40):0.70);",
    # 13: 7 taxa (h absent), caterpillar-like
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
# Pytest markers
# ---------------------------------------------------------------------------

_AVAILABLE = get_available_backends()

cpu_parallel_skip = pytest.mark.skipif(
    "cpu-parallel" not in _AVAILABLE,
    reason="cpu-parallel backend not available",
)
cuda_skip = pytest.mark.skipif(
    "cuda" not in _AVAILABLE,
    reason="cuda backend not available",
)

try:
    from SuchTree import SuchTree as _SuchTree  # noqa: F401

    _SUCHTREE_AVAILABLE = True
except ImportError:
    _SUCHTREE_AVAILABLE = False

suchtree_skip = pytest.mark.skipif(
    not _SUCHTREE_AVAILABLE,
    reason="SuchTree not installed",
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grouped_forest():
    """One group per tree, so counts[qi, gi, :] reflects a single tree."""
    with quiet():
        return Forest({f"t{i:02d}": [t] for i, t in enumerate(TREES)})


@pytest.fixture(scope="module")
def all_quartets(grouped_forest):
    """All C(8,4) = 70 quartets over the 8-taxon global namespace."""
    taxa = sorted(grouped_forest.global_names)
    return Quartets.from_list(grouped_forest, list(itertools.combinations(taxa, 4)))


@pytest.fixture(scope="module")
def suchforest():
    """List of SuchTree objects, one per NEWICK string in TREES."""
    if not _SUCHTREE_AVAILABLE:
        pytest.skip("SuchTree not installed")
    from SuchTree import SuchTree

    return [SuchTree(newick) for newick in TREES]


@pytest.fixture(scope="module")
def pairwise(grouped_forest):
    """
    Precomputed branch distances for all taxon pairs × all trees.

    Returns dict[(gi, gj)] -> ndarray shape (n_trees,), gi < gj are global IDs.
    NaN for trees where either taxon is absent.
    """
    names = grouped_forest.global_names  # gid → name
    n = len(names)
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            D[(i, j)] = grouped_forest.branch_distance(names[i], names[j])
    return D


# ---------------------------------------------------------------------------
# Helper: topology-independent Steiner formula
# ---------------------------------------------------------------------------


def _steiner_and_sums(pairwise, n0, n1, n2, n3, ti):
    """
    Steiner length from branch distances using the closed-form expression:

        S = ( max(s0, s1, s2) + min(s0, s1, s2) ) / 2

    Also returns the three pair-sums s0, s1, s2.
    Returns (None, None, None, None) when any taxon is absent in tree ti
    (one of the pairwise distances is NaN).
    """

    def d(i, j):
        lo, hi = (i, j) if i < j else (j, i)
        return float(pairwise[(lo, hi)][ti])

    d01 = d(n0, n1); d23 = d(n2, n3)
    d02 = d(n0, n2); d13 = d(n1, n3)
    d03 = d(n0, n3); d12 = d(n1, n2)

    if any(np.isnan(x) for x in (d01, d23, d02, d13, d03, d12)):
        return None, None, None, None

    s0 = d01 + d23
    s1 = d02 + d13
    s2 = d03 + d12
    steiner = (max(s0, s1, s2) + min(s0, s1, s2)) / 2.0
    return steiner, s0, s1, s2


# ===========================================================================
# Test classes
# ===========================================================================


class TestBackendAgreement:
    """All backends produce bit-identical counts and allclose Steiner values."""

    @cpu_parallel_skip
    def test_counts_python_vs_cpu(self, grouped_forest, all_quartets):
        with silent_benchmark("python"):
            counts_py = grouped_forest.quartet_topology(all_quartets)
        with silent_benchmark("cpu-parallel"):
            counts_cpu = grouped_forest.quartet_topology(all_quartets)
        np.testing.assert_array_equal(counts_py, counts_cpu)

    @cpu_parallel_skip
    def test_steiner_python_vs_cpu(self, grouped_forest, all_quartets):
        with silent_benchmark("python"):
            _, steiner_py = grouped_forest.quartet_topology(all_quartets, steiner=True)
        with silent_benchmark("cpu-parallel"):
            _, steiner_cpu = grouped_forest.quartet_topology(all_quartets, steiner=True)
        np.testing.assert_allclose(steiner_py, steiner_cpu, rtol=1e-10, atol=1e-12)

    @cpu_parallel_skip
    def test_counts_consistent_across_steiner_modes_cpu(self, grouped_forest, all_quartets):
        """Counts from counts-only and steiner=True must be identical (cpu-parallel)."""
        with silent_benchmark("cpu-parallel"):
            counts_only = grouped_forest.quartet_topology(all_quartets, steiner=False)
            counts_with_steiner, _ = grouped_forest.quartet_topology(
                all_quartets, steiner=True
            )
        np.testing.assert_array_equal(counts_only, counts_with_steiner)

    @cuda_skip
    def test_counts_python_vs_cuda(self, grouped_forest, all_quartets):
        with silent_benchmark("python"):
            counts_py = grouped_forest.quartet_topology(all_quartets)
        with silent_benchmark("cuda"):
            counts_cuda = grouped_forest.quartet_topology(all_quartets)
        np.testing.assert_array_equal(counts_py, counts_cuda)

    @cuda_skip
    def test_steiner_python_vs_cuda(self, grouped_forest, all_quartets):
        with silent_benchmark("python"):
            _, steiner_py = grouped_forest.quartet_topology(all_quartets, steiner=True)
        with silent_benchmark("cuda"):
            _, steiner_cuda = grouped_forest.quartet_topology(all_quartets, steiner=True)
        np.testing.assert_allclose(steiner_py, steiner_cuda, rtol=1e-8, atol=1e-10)

    def test_counts_consistent_across_steiner_modes_python(self, grouped_forest, all_quartets):
        """Counts from counts-only and steiner=True must be identical (python)."""
        with silent_benchmark("python"):
            counts_only = grouped_forest.quartet_topology(all_quartets, steiner=False)
            counts_with_steiner, _ = grouped_forest.quartet_topology(
                all_quartets, steiner=True
            )
        np.testing.assert_array_equal(counts_only, counts_with_steiner)


class TestFourPointCondition:
    """Detected topology minimises the pairwise-distance pair-sum."""

    def _check(self, grouped_forest, all_quartets, pairwise, backend):
        with silent_benchmark(backend):
            counts = grouped_forest.quartet_topology(all_quartets)

        n_groups = grouped_forest.n_groups
        violations = []

        for qi, quartet in enumerate(all_quartets):
            n0, n1, n2, n3 = quartet
            for gi in range(n_groups):
                if counts[qi, gi].sum() == 0:
                    continue  # all taxa absent in this tree

                k_win = int(counts[qi, gi].argmax())
                _, s0, s1, s2 = _steiner_and_sums(pairwise, n0, n1, n2, n3, gi)
                if s0 is None:
                    continue  # absent taxon (shouldn't happen when count>0, but guard anyway)

                s = (s0, s1, s2)
                if not np.isclose(s[k_win], min(s), rtol=1e-9, atol=1e-12):
                    violations.append(
                        f"qi={qi} gi={gi} topo={k_win} "
                        f"s={[f'{x:.6f}' for x in s]} "
                        f"s_win={s[k_win]:.6f} s_min={min(s):.6f}"
                    )

        assert not violations, (
            f"{len(violations)} four-point violations [{backend}]:\n"
            + "\n".join(violations[:10])
        )

    def test_four_point_python(self, grouped_forest, all_quartets, pairwise):
        self._check(grouped_forest, all_quartets, pairwise, "python")

    @cpu_parallel_skip
    def test_four_point_cpu(self, grouped_forest, all_quartets, pairwise):
        self._check(grouped_forest, all_quartets, pairwise, "cpu-parallel")

    @cuda_skip
    def test_four_point_cuda(self, grouped_forest, all_quartets, pairwise):
        self._check(grouped_forest, all_quartets, pairwise, "cuda")


class TestSteinerVsPairwise:
    """Per-group Steiner values match the pairwise-distance formula."""

    def _check(self, grouped_forest, all_quartets, pairwise, backend):
        with silent_benchmark(backend):
            counts, steiner = grouped_forest.quartet_topology(all_quartets, steiner=True)

        n_groups = grouped_forest.n_groups
        mismatches = []

        for qi, quartet in enumerate(all_quartets):
            n0, n1, n2, n3 = quartet
            for gi in range(n_groups):
                if counts[qi, gi].sum() == 0:
                    continue

                k_win = int(counts[qi, gi].argmax())
                s_expected, s0, s1, s2 = _steiner_and_sums(pairwise, n0, n1, n2, n3, gi)
                if s_expected is None:
                    continue

                s_actual = float(steiner[qi, gi, k_win])
                if not np.isclose(s_actual, s_expected, rtol=1e-9, atol=1e-12):
                    mismatches.append(
                        f"qi={qi} gi={gi} topo={k_win} "
                        f"expected={s_expected:.10f} actual={s_actual:.10f} "
                        f"s=({s0:.6f},{s1:.6f},{s2:.6f})"
                    )

        assert not mismatches, (
            f"{len(mismatches)} Steiner mismatches [{backend}]:\n"
            + "\n".join(mismatches[:10])
        )

    def test_steiner_vs_pairwise_python(self, grouped_forest, all_quartets, pairwise):
        self._check(grouped_forest, all_quartets, pairwise, "python")

    @cpu_parallel_skip
    def test_steiner_vs_pairwise_cpu(self, grouped_forest, all_quartets, pairwise):
        self._check(grouped_forest, all_quartets, pairwise, "cpu-parallel")

    @cuda_skip
    def test_steiner_vs_pairwise_cuda(self, grouped_forest, all_quartets, pairwise):
        self._check(grouped_forest, all_quartets, pairwise, "cuda")

    def test_steiner_non_negative(self, grouped_forest, all_quartets):
        with silent_benchmark("python"):
            _, steiner = grouped_forest.quartet_topology(all_quartets, steiner=True)
        assert (steiner >= 0.0).all(), "Negative Steiner distance found"

    def test_steiner_zero_iff_count_zero(self, grouped_forest, all_quartets):
        """steiner[qi, gi, k] > 0 iff counts[qi, gi, k] > 0."""
        with silent_benchmark("python"):
            counts, steiner = grouped_forest.quartet_topology(all_quartets, steiner=True)
        has_count = counts > 0
        has_steiner = steiner > 0.0
        assert np.array_equal(has_count, has_steiner), (
            "Mismatch between count>0 and steiner>0 masks"
        )


class TestPairwiseDistances:
    """Sanity checks for branch_distance."""

    def test_symmetric(self, grouped_forest):
        names = grouped_forest.global_names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d_ij = grouped_forest.branch_distance(names[i], names[j])
                d_ji = grouped_forest.branch_distance(names[j], names[i])
                np.testing.assert_allclose(d_ij, d_ji, rtol=1e-12,
                                            err_msg=f"asymmetric {names[i]},{names[j]}")

    def test_nan_for_absent_taxon(self, grouped_forest):
        """Trees 10 (t10) covers {a,b,c,d,e,f} only — g and h must be NaN."""
        gi = grouped_forest.unique_groups.index("t10")
        for absent in ("g", "h"):
            d = grouped_forest.branch_distance("a", absent)
            assert np.isnan(d[gi]), (
                f"Expected NaN for absent taxon '{absent}' in t10, got {d[gi]}"
            )

    def test_positive_for_present_distinct_taxa(self, grouped_forest):
        """All-8-taxa trees (t00–t09) must have positive pairwise distances."""
        names = grouped_forest.global_names
        full_groups = [f"t{i:02d}" for i in range(10)]
        for g_name in full_groups:
            gi = grouped_forest.unique_groups.index(g_name)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    d = float(grouped_forest.branch_distance(names[i], names[j])[gi])
                    assert d > 0, (
                        f"Expected d>0 for ({names[i]},{names[j]}) in {g_name}, got {d}"
                    )

    def test_triangle_inequality(self, grouped_forest):
        """d(a,c) <= d(a,b) + d(b,c) for all triplets in trees 0-9."""
        names = grouped_forest.global_names
        n = len(names)
        full_groups = [f"t{i:02d}" for i in range(10)]
        violations = []
        for g_name in full_groups:
            gi = grouped_forest.unique_groups.index(g_name)
            for i, j, k in itertools.combinations(range(n), 3):
                dij = float(grouped_forest.branch_distance(names[i], names[j])[gi])
                djk = float(grouped_forest.branch_distance(names[j], names[k])[gi])
                dik = float(grouped_forest.branch_distance(names[i], names[k])[gi])
                # Check all three triangle inequalities
                for da, db, dc, la, lb, lc in [
                    (dij, djk, dik, names[i], names[j], names[k]),
                    (dij, dik, djk, names[i], names[k], names[j]),
                    (djk, dik, dij, names[j], names[k], names[i]),
                ]:
                    if da + db < dc - 1e-10:
                        violations.append(
                            f"{g_name}: d({la},{lc})={dc:.6f} > "
                            f"d({la},{lb})+d({lb},{lc})={da+db:.6f}"
                        )
        assert not violations, (
            f"{len(violations)} triangle-inequality violations:\n"
            + "\n".join(violations[:5])
        )


# ---------------------------------------------------------------------------
# SuchTree topology translation
# ---------------------------------------------------------------------------

# The three unrooted topologies for sorted taxa (n0, n1, n2, n3), expressed
# as frozensets of frozensets — SuchTree's native representation.
_TOPO_BUILDERS = [
    lambda ns: frozenset({frozenset({ns[0], ns[1]}), frozenset({ns[2], ns[3]})}),
    lambda ns: frozenset({frozenset({ns[0], ns[2]}), frozenset({ns[1], ns[3]})}),
    lambda ns: frozenset({frozenset({ns[0], ns[3]}), frozenset({ns[1], ns[2]})}),
]


def _st_topo_to_k(st_topo, names):
    """
    Convert a SuchTree frozenset topology to quarimo's k index (0, 1, or 2).

    Parameters
    ----------
    st_topo : frozenset
        SuchTree topology: frozenset of two frozensets of two taxon names.
    names : list[str]
        The four taxon names in sorted global-ID order [name0, name1, name2, name3].

    Returns
    -------
    int
        quarimo topology index: 0, 1, or 2.

    Raises
    ------
    ValueError
        If st_topo does not match any of the three expected topologies.
    """
    for k, build in enumerate(_TOPO_BUILDERS):
        if st_topo == build(names):
            return k
    raise ValueError(f"Unrecognized SuchTree topology {st_topo} for names {names}")


# ===========================================================================
# SuchTree cross-validation
# ===========================================================================


class TestSuchTreeAgreement:
    """
    Exact topology agreement between quarimo and SuchTree for all (quartet, tree)
    pairs where the winning topology is unambiguous (pair-sums are not tied).

    Tied pairs are covered by TestSuchTreePolytomyAgreement.  Tests use the
    Python backend; add cpu_parallel_skip / cuda_skip decorators to extend to
    other backends without changing the _check() logic.
    """

    @suchtree_skip
    def test_topology_python(self, grouped_forest, all_quartets, pairwise, suchforest):
        self._check("python", grouped_forest, all_quartets, pairwise, suchforest)

    @suchtree_skip
    @cpu_parallel_skip
    def test_topology_cpu(self, grouped_forest, all_quartets, pairwise, suchforest):
        self._check("cpu-parallel", grouped_forest, all_quartets, pairwise, suchforest)

    @suchtree_skip
    @cuda_skip
    def test_topology_cuda(self, grouped_forest, all_quartets, pairwise, suchforest):
        self._check("cuda", grouped_forest, all_quartets, pairwise, suchforest)

    def _check(self, backend, grouped_forest, all_quartets, pairwise, suchforest):
        with silent_benchmark(backend):
            counts = grouped_forest.quartet_topology(all_quartets)

        global_names = grouped_forest.global_names
        n_groups = grouped_forest.n_groups
        quartet_list = list(all_quartets)
        tol = 1e-9
        mismatches = []

        for gi in range(n_groups):
            st = suchforest[gi]
            st_taxa = set(st.leaves.keys())

            # Partition quartets: present (all 4 taxa in tree) vs absent.
            present = []  # list of (qi, names)
            for qi, (n0, n1, n2, n3) in enumerate(quartet_list):
                names = [global_names[n] for n in (n0, n1, n2, n3)]
                if all(name in st_taxa for name in names):
                    present.append((qi, names))
                else:
                    assert counts[qi, gi].sum() == 0, (
                        f"Expected 0 count for qi={qi} gi={gi} (absent taxa), "
                        f"got {counts[qi, gi]}"
                    )

            if not present:
                continue

            # Batch all present quartets in a single SuchTree call.
            st_topos = st.quartet_topologies_by_name([nm for _, nm in present])

            for (qi, names), st_topo in zip(present, st_topos):
                n0, n1, n2, n3 = quartet_list[qi]
                _, s0, s1, s2 = _steiner_and_sums(pairwise, n0, n1, n2, n3, gi)
                s_vals = (s0, s1, s2)
                s_min = min(s_vals)

                # Skip tied pairs — they belong to TestSuchTreePolytomyAgreement.
                n_tied = sum(1 for s in s_vals if abs(s - s_min) < tol)
                if n_tied > 1:
                    continue

                quarimo_k = int(counts[qi, gi].argmax())
                st_k = _st_topo_to_k(st_topo, names)

                if quarimo_k != st_k:
                    mismatches.append(
                        f"qi={qi} gi={gi} quarimo_k={quarimo_k} st_k={st_k} "
                        f"names={names} s=({s0:.6f},{s1:.6f},{s2:.6f})"
                    )

        assert not mismatches, (
            f"{len(mismatches)} topology mismatches [{backend}]:\n"
            + "\n".join(mismatches[:10])
        )


@pytest.mark.polytomy
class TestSuchTreePolytomyAgreement:
    """
    For polytomous (tied) quartet-tree pairs, both quarimo and SuchTree must
    select a topology whose pair-sum equals the minimum.  Exact agreement
    between the two implementations is NOT required: tie-breaking is
    implementation-defined and will be re-examined when quarimo's polytomy
    handling changes.

    Skip this class with ``-m 'not polytomy'``; run only with ``-m polytomy``.
    """

    @suchtree_skip
    def test_polytomy_topology_python(
        self, grouped_forest, all_quartets, pairwise, suchforest
    ):
        self._check("python", grouped_forest, all_quartets, pairwise, suchforest)

    @suchtree_skip
    @cpu_parallel_skip
    def test_polytomy_topology_cpu(
        self, grouped_forest, all_quartets, pairwise, suchforest
    ):
        self._check("cpu-parallel", grouped_forest, all_quartets, pairwise, suchforest)

    @suchtree_skip
    @cuda_skip
    def test_polytomy_topology_cuda(
        self, grouped_forest, all_quartets, pairwise, suchforest
    ):
        self._check("cuda", grouped_forest, all_quartets, pairwise, suchforest)

    def _check(self, backend, grouped_forest, all_quartets, pairwise, suchforest):
        with silent_benchmark(backend):
            counts = grouped_forest.quartet_topology(all_quartets)

        global_names = grouped_forest.global_names
        n_groups = grouped_forest.n_groups
        quartet_list = list(all_quartets)
        tol = 1e-9
        quarimo_invalid = []
        suchtree_invalid = []

        for gi in range(n_groups):
            st = suchforest[gi]
            st_taxa = set(st.leaves.keys())

            # Collect only tied quartet-tree pairs.
            tied = []  # list of (qi, names, valid_ks)
            for qi, (n0, n1, n2, n3) in enumerate(quartet_list):
                names = [global_names[n] for n in (n0, n1, n2, n3)]
                if not all(name in st_taxa for name in names):
                    continue
                _, s0, s1, s2 = _steiner_and_sums(pairwise, n0, n1, n2, n3, gi)
                s_vals = (s0, s1, s2)
                s_min = min(s_vals)
                valid_ks = frozenset(k for k, s in enumerate(s_vals) if abs(s - s_min) < tol)
                if len(valid_ks) > 1:
                    tied.append((qi, names, valid_ks))

            if not tied:
                continue

            # Batch SuchTree call for this tree.
            st_topos = st.quartet_topologies_by_name([nm for _, nm, _ in tied])

            for (qi, names, valid_ks), st_topo in zip(tied, st_topos):
                quarimo_k = int(counts[qi, gi].argmax())
                if quarimo_k not in valid_ks:
                    quarimo_invalid.append(
                        f"qi={qi} gi={gi} quarimo_k={quarimo_k} "
                        f"valid={set(valid_ks)} names={names}"
                    )

                st_k = _st_topo_to_k(st_topo, names)
                if st_k not in valid_ks:
                    suchtree_invalid.append(
                        f"qi={qi} gi={gi} st_k={st_k} "
                        f"valid={set(valid_ks)} names={names}"
                    )

        # Separate asserts so both failure sets are visible independently.
        assert not quarimo_invalid, (
            f"quarimo chose invalid topology for {len(quarimo_invalid)} tied "
            f"pairs [{backend}]:\n" + "\n".join(quarimo_invalid[:10])
        )
        assert not suchtree_invalid, (
            f"SuchTree chose invalid topology for {len(suchtree_invalid)} tied pairs:\n"
            + "\n".join(suchtree_invalid[:10])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
