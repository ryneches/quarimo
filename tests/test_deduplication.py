"""
test_deduplication.py
======================
Tests for automatic deduplication of identical trees during Forest construction.

Case A: trees with identical topology AND identical branch lengths are merged
into a single stored representative; ``_tree_multiplicities`` records the weight.
Public ``n_trees`` always reflects the number of input trees loaded by the user.

Correctness criterion: a Forest constructed from k copies of the same tree
must produce the same counts (scaled by k) as a Forest with one copy.
"""

import numpy as np

from quarimo import Forest, Quartets

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TREE_A = "((A:0.1,B:0.2):0.5,(C:0.3,D:0.4):0.6);"   # balanced 4-leaf
TREE_B = "((A:0.1,C:0.2):0.5,(B:0.3,D:0.4):0.6);"   # different topology
TREE_A_DIFF_BL = "((A:0.9,B:0.1):0.2,(C:0.7,D:0.5):0.3);"  # same topo, diff BL
TREE_A_DIFF_SUPPORT = "((A:0.1,B:0.2)0.99:0.5,(C:0.3,D:0.4)0.50:0.6);"  # diff support only


# ---------------------------------------------------------------------------
# tree_hash tests
# ---------------------------------------------------------------------------

class TestTreeHash:
    def test_same_tree_same_hash(self):
        from quarimo._tree import Tree
        t1 = Tree(TREE_A)
        t2 = Tree(TREE_A)
        assert t1.tree_hash() == t2.tree_hash()

    def test_different_topology_different_hash(self):
        from quarimo._tree import Tree
        t1 = Tree(TREE_A)
        t2 = Tree(TREE_B)
        assert t1.tree_hash() != t2.tree_hash()

    def test_different_branch_lengths_different_hash(self):
        from quarimo._tree import Tree
        t1 = Tree(TREE_A)
        t2 = Tree(TREE_A_DIFF_BL)
        assert t1.tree_hash() != t2.tree_hash()

    def test_different_support_same_hash(self):
        """Bootstrap support values are excluded — different support → same hash."""
        from quarimo._tree import Tree
        t1 = Tree(TREE_A)
        t2 = Tree(TREE_A_DIFF_SUPPORT)
        assert t1.tree_hash() == t2.tree_hash()

    def test_returns_bytes(self):
        from quarimo._tree import Tree
        t = Tree(TREE_A)
        assert isinstance(t.tree_hash(), bytes)
        assert len(t.tree_hash()) == 32  # SHA-256 digest length


# ---------------------------------------------------------------------------
# Forest deduplication metadata tests
# ---------------------------------------------------------------------------

class TestDeduplicationMetadata:
    def test_no_duplicates_multiplicity_all_ones(self):
        forest = Forest([TREE_A, TREE_B])
        assert forest.n_trees == 2           # public: input count
        assert forest._n_stored_trees == 2   # private: stored count
        np.testing.assert_array_equal(forest._tree_multiplicities, [1, 1])

    def test_two_copies_deduplicated(self):
        forest = Forest([TREE_A, TREE_A])
        assert forest.n_trees == 2           # public: input count unchanged
        assert forest._n_stored_trees == 1   # private: 1 unique stored
        np.testing.assert_array_equal(forest._tree_multiplicities, [2])

    def test_three_copies_one_unique(self):
        forest = Forest([TREE_A, TREE_A, TREE_A])
        assert forest.n_trees == 3
        assert forest._n_stored_trees == 1
        np.testing.assert_array_equal(forest._tree_multiplicities, [3])

    def test_mixed_deduplication(self):
        # TREE_A appears twice, TREE_B once
        forest = Forest([TREE_A, TREE_B, TREE_A])
        assert forest.n_trees == 3
        assert forest._n_stored_trees == 2

        # First occurrence wins; TREE_A is at index 0, TREE_B at index 1
        # (order preserved: first occurrence of each unique tree)
        total = int(forest._tree_multiplicities.sum())
        assert total == 3
        assert forest._tree_multiplicities[0] == 2  # TREE_A x2
        assert forest._tree_multiplicities[1] == 1  # TREE_B x1

    def test_different_bl_not_deduplicated(self):
        """Trees with different branch lengths must NOT be merged."""
        forest = Forest([TREE_A, TREE_A_DIFF_BL])
        assert forest.n_trees == 2
        assert forest._n_stored_trees == 2
        np.testing.assert_array_equal(forest._tree_multiplicities, [1, 1])

    def test_different_support_deduplicated(self):
        """Trees differing only in support values ARE merged (support excluded from hash)."""
        forest = Forest([TREE_A, TREE_A_DIFF_SUPPORT])
        assert forest.n_trees == 2
        assert forest._n_stored_trees == 1
        np.testing.assert_array_equal(forest._tree_multiplicities, [2])

    def test_multiplicities_dtype(self):
        forest = Forest([TREE_A, TREE_B])
        assert forest._tree_multiplicities.dtype == np.int32

    def test_multiplicities_sum_equals_n_input(self):
        forest = Forest([TREE_A, TREE_B, TREE_A])
        assert int(forest._tree_multiplicities.sum()) == forest.n_trees


# ---------------------------------------------------------------------------
# Cross-group deduplication: trees in different groups must NOT be merged
# ---------------------------------------------------------------------------

class TestCrossGroupDeduplication:
    def test_same_tree_different_groups_not_merged(self):
        forest = Forest({"group1": [TREE_A], "group2": [TREE_A]})
        # Both groups have TREE_A; they must not be merged
        assert forest.n_trees == 2
        assert forest._n_stored_trees == 2
        np.testing.assert_array_equal(forest._tree_multiplicities, [1, 1])

    def test_same_tree_same_group_merged(self):
        forest = Forest({"group1": [TREE_A, TREE_A], "group2": [TREE_B]})
        assert forest.n_trees == 3           # public: 3 input trees
        assert forest._n_stored_trees == 2   # private: 2 unique stored
        total = int(forest._tree_multiplicities.sum())
        assert total == 3  # 2 + 1


# ---------------------------------------------------------------------------
# Count correctness: k copies should give k× the counts
# ---------------------------------------------------------------------------

class TestCountCorrectness:
    def _q(self, forest, n=20):
        return Quartets.random(forest, count=n, seed=42)

    def _q_for(self, forest, ref_q):
        """Create a Quartets for `forest` that covers the same sequence window as `ref_q`."""
        return Quartets(forest, seed=ref_q.seed, offset=ref_q.offset, count=len(ref_q))

    def test_counts_scale_with_multiplicity(self):
        """Forest([T, T, T]) counts should be 3× Forest([T]) counts."""
        f1 = Forest([TREE_A])
        f3 = Forest([TREE_A, TREE_A, TREE_A])
        q1 = self._q(f1)
        q3 = self._q_for(f3, q1)

        r1 = f1.quartet_topology(q1)
        r3 = f3.quartet_topology(q3)

        np.testing.assert_array_equal(r3.counts, r1.counts * 3)

    def test_counts_mixed_deduplicated_vs_expanded(self):
        """Forest([A, A, B]) counts should equal 2*counts_A + counts_B."""
        f_deduped = Forest([TREE_A, TREE_A, TREE_B])
        f_a = Forest([TREE_A])
        f_b = Forest([TREE_B])

        q = self._q(f_deduped)
        qa = self._q_for(f_a, q)
        qb = self._q_for(f_b, q)

        r = f_deduped.quartet_topology(q)
        ra = f_a.quartet_topology(qa)
        rb = f_b.quartet_topology(qb)

        # r_deduped should equal 2*ra + rb (element-wise)
        np.testing.assert_array_equal(r.counts, ra.counts * 2 + rb.counts)

    def test_steiner_scales_with_multiplicity(self):
        """Steiner sum should scale by multiplicity; min/max should be unchanged."""
        f1 = Forest([TREE_A])
        f2 = Forest([TREE_A, TREE_A])

        q1 = self._q(f1)
        q2 = self._q_for(f2, q1)

        r1 = f1.quartet_topology(q1, steiner=True)
        r2 = f2.quartet_topology(q2, steiner=True)

        # Steiner sum scales by 2
        np.testing.assert_allclose(r2.steiner, r1.steiner * 2)

        # Counts scale by 2
        np.testing.assert_array_equal(r2.counts, r1.counts * 2)

        # Min and max are unchanged (identical trees have same Steiner length)
        mask = r1.counts > 0
        np.testing.assert_allclose(r2.steiner_min[mask], r1.steiner_min[mask])
        np.testing.assert_allclose(r2.steiner_max[mask], r1.steiner_max[mask])

    def test_variance_correct_with_multiplicity(self):
        """Population variance should be identical regardless of deduplication."""
        f1 = Forest([TREE_A, TREE_B])
        f2 = Forest([TREE_A, TREE_B, TREE_A, TREE_B])

        q1 = self._q(f1)
        q2 = self._q_for(f2, q1)

        r1 = f1.quartet_topology(q1, steiner=True)
        r2 = f2.quartet_topology(q2, steiner=True)

        # Variance should be the same (same population, just counted twice)
        mask = r1.counts > 0
        np.testing.assert_allclose(
            r2.steiner_var[mask], r1.steiner_var[mask], atol=1e-10,
        )


# ---------------------------------------------------------------------------
# CSR integrity after deduplication
# ---------------------------------------------------------------------------

class TestCSRIntegrityAfterDeduplication:
    def test_csr_offsets_consistent(self):
        forest = Forest([TREE_A, TREE_A, TREE_B])
        # Input: 3 trees; stored (unique): 2 trees
        assert forest.n_trees == 3               # public: input count
        assert forest._n_stored_trees == 2       # private: stored count
        assert len(forest.node_offsets) == forest._n_stored_trees + 1
        assert len(forest.tour_offsets) == forest._n_stored_trees + 1
        assert forest.global_to_local.shape[0] == forest._n_stored_trees

    def test_tree_multiplicities_in_kernel_data(self):
        forest = Forest([TREE_A, TREE_A, TREE_B])
        kd = forest._kernel_data
        assert hasattr(kd, "tree_multiplicities")
        np.testing.assert_array_equal(kd.tree_multiplicities, forest._tree_multiplicities)
