"""
tests/test_consensus_graph.py
=============================
Tests for quarimo._consensus_graph: the mutable bipartition graph used by the
prune-only quartet consensus algorithm.

Tree fixtures use ``zero_eps=0`` to prevent zero-length collapse at construction
so that graph structure tests can use topology-only NEWICK strings.  The
zero-length collapse behaviour is tested separately with explicit edge lengths.

Tree topologies after root normalization
-----------------------------------------
4-leaf balanced ``((A,B),(C,D));`` with ``zero_eps=0``:
  2-child root → re-rooted at (A,B) subtree.
  New root (node 1) children = [A, B, (C,D)].
  active_branch_ids has exactly 1 edge: the (C,D) branch.
  n_qsupp = pair_total([1,1]) * pair_total([1,1]) = 1.

8-leaf balanced ``(((A,B),(C,D)),((E,F),(G,H)));`` with ``zero_eps=0``:
  2-child root → re-rooted at (ABCD) subtree.
  New root children = [(A,B), (C,D), (E,F,G,H)].
  active_branch_ids has exactly 5 edges: AB, CD, EF, GH, EFGH.
  n_qsupp(EFGH) = pair_total([2,2]) * pair_total([2,2]) = 16.
  n_qsupp(AB)   = pair_total([2,4]) * pair_total([1,1]) = 8.
"""

import random

import pytest

from quarimo._consensus_graph import (
    ConsensusGraph,
    _component_size,
    _pair_sampling_tables,
    _pair_total,
    _parse_newick,
    _quartet_topo_index,
    _strip_newick_comments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NAMES4 = ["A", "B", "C", "D"]
NAMES8 = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _graph4():
    """((A,B),(C,D)); with 4-leaf global namespace, no zero-length collapse."""
    return ConsensusGraph("((A,B),(C,D));", NAMES4, zero_eps=0)


def _graph8():
    """(((A,B),(C,D)),((E,F),(G,H))); 8-leaf, no zero-length collapse."""
    return ConsensusGraph("(((A,B),(C,D)),((E,F),(G,H)));", NAMES8, zero_eps=0)



# ---------------------------------------------------------------------------
# _strip_newick_comments
# ---------------------------------------------------------------------------

class TestStripComments:
    def test_no_comments(self):
        assert _strip_newick_comments("((A,B),(C,D));") == "((A,B),(C,D));"

    def test_single_comment(self):
        assert _strip_newick_comments("(A[&label=1],B);") == "(A,B);"

    def test_multiple_comments(self):
        assert _strip_newick_comments("(A[x],B[y],C);") == "(A,B,C);"

    def test_nested_brackets(self):
        assert _strip_newick_comments("(A[outer[inner]],B);") == "(A,B);"

    def test_empty_string(self):
        assert _strip_newick_comments("") == ""

    def test_only_comment(self):
        assert _strip_newick_comments("[comment]") == ""


# ---------------------------------------------------------------------------
# _parse_newick
# ---------------------------------------------------------------------------

class TestParseNewick:
    def test_leaf_only(self):
        d = _parse_newick("Taxon")
        assert d["name"] == "Taxon"
        assert d["children"] == []
        assert d["length"] == 0.0

    def test_two_leaves(self):
        d = _parse_newick("(A,B)")
        assert len(d["children"]) == 2
        assert d["children"][0]["name"] == "A"
        assert d["children"][1]["name"] == "B"

    def test_branch_lengths(self):
        d = _parse_newick("(A:0.1,B:0.2):0.5")
        assert d["children"][0]["length"] == pytest.approx(0.1)
        assert d["children"][1]["length"] == pytest.approx(0.2)
        assert d["length"] == pytest.approx(0.5)

    def test_bootstrap_label(self):
        d = _parse_newick("((A,B)95,(C,D)87);")
        assert d["children"][0]["name"] == "95"
        assert d["children"][1]["name"] == "87"

    def test_quoted_label(self):
        d = _parse_newick("('Homo sapiens',Pan);")
        assert d["children"][0]["name"] == "Homo sapiens"
        assert d["children"][1]["name"] == "Pan"

    def test_escaped_single_quote(self):
        d = _parse_newick("('A''B',C);")
        assert d["children"][0]["name"] == "A'B"

    def test_semicolon_stripped(self):
        d = _parse_newick("(A,B);")
        assert len(d["children"]) == 2  # no crash, semicolon handled

    def test_three_children(self):
        d = _parse_newick("(A,B,C);")
        assert len(d["children"]) == 3

    def test_nested(self):
        d = _parse_newick("((A,B),(C,D));")
        assert len(d["children"]) == 2
        assert len(d["children"][0]["children"]) == 2


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

class TestComponentSize:
    def test_single_range(self):
        assert _component_size(((0, 3),)) == 4

    def test_two_ranges(self):
        assert _component_size(((0, 2), (5, 7))) == 6

    def test_single_leaf(self):
        assert _component_size(((5, 5),)) == 1

    def test_length_one_gap(self):
        assert _component_size(((0, 0), (2, 2))) == 2


class TestPairTotal:
    def test_single_component_no_pairs(self):
        assert _pair_total([5]) == 0

    def test_two_equal(self):
        assert _pair_total([2, 2]) == 4

    def test_two_unequal(self):
        # (8^2 - 9 - 25) / 2 = (64 - 34) / 2 = 15
        assert _pair_total([3, 5]) == 15

    def test_three_components(self):
        # S=3, sum(si^2)=3 → (9-3)/2=3
        assert _pair_total([1, 1, 1]) == 3

    def test_empty(self):
        assert _pair_total([]) == 0

    def test_consistency_with_formula(self):
        sizes = [2, 3, 4, 5]
        s = sum(sizes)
        expected = (s * s - sum(x * x for x in sizes)) // 2
        assert _pair_total(sizes) == expected


class TestPairSamplingTables:
    def test_single_component_no_pairs(self):
        pairs, weights, cum = _pair_sampling_tables([5])
        assert pairs == []
        assert weights == []
        assert len(cum) == 0

    def test_two_components(self):
        pairs, weights, cum = _pair_sampling_tables([2, 3])
        assert pairs == [(0, 1)]
        assert weights == [6]
        assert int(cum[-1]) == 6

    def test_three_components_pair_count(self):
        pairs, _, _ = _pair_sampling_tables([1, 2, 3])
        assert len(pairs) == 3  # C(3,2)

    def test_weight_equals_product(self):
        pairs, weights, _ = _pair_sampling_tables([2, 3, 5])
        expected = {(0, 1): 6, (0, 2): 10, (1, 2): 15}
        for (i, j), w in zip(pairs, weights):
            assert w == expected[(i, j)]

    def test_cumulative_monotone(self):
        _, _, cum = _pair_sampling_tables([3, 4, 5])
        for i in range(len(cum) - 1):
            assert cum[i] <= cum[i + 1]

    def test_cumulative_total_equals_sum_of_weights(self):
        _, weights, cum = _pair_sampling_tables([2, 3, 4])
        assert int(cum[-1]) == sum(weights)


class TestQuartetTopoIndex:
    def test_topo0_ab_side_u(self):
        # a=0,b=1 on side_u; c=2,d=3 on side_v.
        # sorted: t0=0,t1=1,t2=2,t3=3; ab={0,1}={t0,t1} → 0
        assert _quartet_topo_index(0, 1, 2, 3) == 0

    def test_topo0_reversed_sides(self):
        # a=2,b=3 on side_u; c=0,d=1 on side_v.
        # ab={2,3}={t2,t3} → 0
        assert _quartet_topo_index(2, 3, 0, 1) == 0

    def test_topo1_ac_bd(self):
        # a=0,b=2; sorted: 0,1,2,3; {0,2}={t0,t2} → 1
        assert _quartet_topo_index(0, 2, 1, 3) == 1

    def test_topo1_reversed(self):
        # ab={1,3}={t1,t3} → 1
        assert _quartet_topo_index(1, 3, 0, 2) == 1

    def test_topo2_ad_bc(self):
        # a=0,b=3; ab={0,3}: not {t0,t1}, not {t0,t2}, not {t2,t3}, not {t1,t3} → 2
        assert _quartet_topo_index(0, 3, 1, 2) == 2

    def test_topo2_bc_ad(self):
        # ab={1,2}: none of the topo-0 or topo-1 pairs → 2
        assert _quartet_topo_index(1, 2, 0, 3) == 2

    def test_result_in_range(self):
        for a, b, c, d in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
            assert _quartet_topo_index(a, b, c, d) in (0, 1, 2)

    def test_symmetry_within_side(self):
        # Swapping a and b (same side) gives same topology
        assert _quartet_topo_index(1, 0, 2, 3) == _quartet_topo_index(0, 1, 2, 3)
        assert _quartet_topo_index(0, 1, 3, 2) == _quartet_topo_index(0, 1, 2, 3)


# ---------------------------------------------------------------------------
# ConsensusGraph construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_n_leaves_4(self):
        assert _graph4()._n_leaves == 4

    def test_n_leaves_8(self):
        assert _graph8()._n_leaves == 8

    def test_leaf_gids_distinct(self):
        g = _graph4()
        assert sorted(g._leaf_gids) == [0, 1, 2, 3]

    def test_leaf_count_8(self):
        g = _graph8()
        assert sorted(g._leaf_gids) == list(range(8))

    def test_too_few_taxa_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            ConsensusGraph("(A,(B,C));", ["A", "B", "C"], zero_eps=0)

    def test_leaf_not_in_namespace_raises(self):
        with pytest.raises(ValueError, match="not found in forest namespace"):
            ConsensusGraph("((A,B),(C,X));", NAMES4, zero_eps=0)

    def test_extra_namespace_taxa_raises(self):
        # Reference has 4 leaves but namespace has 5 → mismatch
        with pytest.raises(ValueError, match="does not match forest namespace"):
            ConsensusGraph("((A,B),(C,D));", NAMES4 + ["E"], zero_eps=0)

    def test_root_normalized_to_three_plus_children(self):
        """2-child root is rerooted so active root has ≥ 3 children."""
        g = _graph4()
        root = g._nodes[g._root]
        assert len(root.children) >= 3

    def test_three_child_root_unchanged(self):
        """3-child root needs no normalization."""
        g = ConsensusGraph("(A,B,(C,D));", NAMES4, zero_eps=0)
        root = g._nodes[g._root]
        assert len(root.children) == 3

    def test_zero_length_internal_collapsed(self):
        """An internal branch with length ≤ zero_eps is collapsed at construction."""
        # (C,D) sub-tree has length 0.0; default zero_eps=1e-12 collapses it
        g = ConsensusGraph("((A,B),(C,D):0.0);", NAMES4)
        # After collapse, root should have 4 direct children (all leaves)
        root = g._nodes[g._root]
        assert len(root.children) == 4

    def test_positive_length_not_collapsed(self):
        """An internal branch with positive length is not collapsed."""
        g = ConsensusGraph("((A,B),(C,D):0.5);", NAMES4)
        assert len(g.active_branch_ids) >= 1

    def test_zero_eps_zero_disables_collapse(self):
        """zero_eps=0 means no branches are auto-collapsed."""
        # All-zero lengths but zero_eps=0 → branches preserved
        g = ConsensusGraph("((A,B),(C,D));", NAMES4, zero_eps=0)
        assert len(g.active_branch_ids) == 1

    def test_root_interval_covers_all_leaves(self):
        g = _graph8()
        root = g._nodes[g._root]
        assert root.start == 0
        assert root.end == g._n_leaves - 1

    def test_child_intervals_nested(self):
        """Every edge's child interval is a sub-interval of its parent's."""
        g = _graph8()
        for edge in g._edges:
            if not edge.active:
                continue
            parent = g._nodes[edge.parent]
            child = g._nodes[edge.child]
            assert child.start >= parent.start
            assert child.end <= parent.end

    def test_leaf_intervals_are_singletons(self):
        """Every leaf node has start == end."""
        g = _graph8()
        for node in g._nodes:
            if node.taxon_gid != -1:  # leaf
                assert node.start == node.end


# ---------------------------------------------------------------------------
# active_branch_ids
# ---------------------------------------------------------------------------

class TestActiveBranchIds:
    def test_4leaf_one_internal_branch(self):
        assert len(_graph4().active_branch_ids) == 1

    def test_8leaf_five_internal_branches(self):
        assert len(_graph8().active_branch_ids) == 5

    def test_star_has_no_internal_branches(self):
        g = ConsensusGraph("(A,B,C,D);", NAMES4, zero_eps=0)
        assert g.active_branch_ids == set()

    def test_after_prune_count_decreases_by_one(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        g.prune(bid)
        assert len(g.active_branch_ids) == 4

    def test_all_returned_ids_are_active_internal(self):
        g = _graph8()
        for bid in g.active_branch_ids:
            e = g._edges[bid]
            assert e.active
            assert e.is_internal

    def test_ids_are_valid_edge_indices(self):
        g = _graph8()
        for bid in g.active_branch_ids:
            assert 0 <= bid < len(g._edges)


# ---------------------------------------------------------------------------
# n_qsupp
# ---------------------------------------------------------------------------

class TestNQsupp:
    def test_4leaf_single_branch(self):
        """
        After normalization, root children = [A, B, (C,D)].
        CD branch: side_u = [A(1), B(1)], side_v = [C(1), D(1)].
        n_qsupp = pair_total([1,1]) * pair_total([1,1]) = 1 * 1 = 1.
        """
        g = _graph4()
        (bid,) = g.active_branch_ids
        assert g.n_qsupp(bid) == 1

    def test_8leaf_efgh_branch(self):
        """
        EFGH branch (child covers DFS positions 4-7):
          side_u = [AB(2), CD(2)] → pair_total=4
          side_v = [EF(2), GH(2)] → pair_total=4
          n_qsupp = 16
        """
        g = _graph8()
        # Find EFGH branch: child.start=4, child.end=7
        efgh_bid = None
        for bid in g.active_branch_ids:
            child = g._nodes[g._edges[bid].child]
            if child.start == 4 and child.end == 7:
                efgh_bid = bid
                break
        assert efgh_bid is not None, "EFGH branch not found"
        assert g.n_qsupp(efgh_bid) == 16

    def test_8leaf_ab_branch(self):
        """
        AB branch (child covers positions 0-1):
          side_u = [CD(2), EFGH(4)] → pair_total=8
          side_v = [A(1), B(1)]     → pair_total=1
          n_qsupp = 8
        """
        g = _graph8()
        ab_bid = None
        for bid in g.active_branch_ids:
            child = g._nodes[g._edges[bid].child]
            if child.start == 0 and child.end == 1:
                ab_bid = bid
                break
        assert ab_bid is not None, "AB branch not found"
        assert g.n_qsupp(ab_bid) == 8

    def test_all_active_branches_positive(self):
        """Every scoreable branch has n_qsupp > 0."""
        for bid in _graph8().active_branch_ids:
            assert _graph8().n_qsupp(bid) > 0

    def test_larger_tree_positive(self):
        """Caterpillar-style tree with 5 taxa has positive n_qsupp on all branches."""
        g = ConsensusGraph("(A,(B,(C,(D,E))));", ["A", "B", "C", "D", "E"], zero_eps=0)
        assert len(g.active_branch_ids) > 0
        for bid in g.active_branch_ids:
            assert g.n_qsupp(bid) > 0


# ---------------------------------------------------------------------------
# sample_quartets
# ---------------------------------------------------------------------------

class TestSampleQuartets:
    def test_n_zero_returns_empty(self):
        g = _graph4()
        (bid,) = g.active_branch_ids
        assert g.sample_quartets(bid, 0, random.Random(0)) == []

    def test_returns_exactly_n_results(self):
        g = _graph4()
        (bid,) = g.active_branch_ids
        assert len(g.sample_quartets(bid, 10, random.Random(42))) == 10

    def test_quartet_is_4tuple(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        for quartet, _ in g.sample_quartets(bid, 20, random.Random(7)):
            assert len(quartet) == 4

    def test_topo_in_range(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        for _, topo in g.sample_quartets(bid, 20, random.Random(7)):
            assert topo in (0, 1, 2)

    def test_gids_sorted_within_quartet(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        for quartet, _ in g.sample_quartets(bid, 20, random.Random(1)):
            assert list(quartet) == sorted(quartet)

    def test_no_duplicate_taxa_within_quartet(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        for quartet, _ in g.sample_quartets(bid, 50, random.Random(99)):
            assert len(set(quartet)) == 4

    def test_gids_in_valid_range(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        for quartet, _ in g.sample_quartets(bid, 30, random.Random(5)):
            for gid in quartet:
                assert 0 <= gid < g._n_leaves

    def test_4leaf_only_one_possible_quartet(self):
        """
        The (C,D) branch of the 4-leaf tree has exactly one quartet:
        {A,B} | {C,D} → (0,1,2,3) with topology 0.
        """
        g = _graph4()
        (bid,) = g.active_branch_ids
        for quartet, topo in g.sample_quartets(bid, 30, random.Random(0)):
            assert quartet == (0, 1, 2, 3)
            assert topo == 0

    def test_reproducible_with_same_seed(self):
        g = _graph8()
        bid = next(iter(sorted(g.active_branch_ids)))
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        assert g.sample_quartets(bid, 10, rng_a) == g.sample_quartets(bid, 10, rng_b)

    def test_different_seeds_differ(self):
        g = _graph8()
        bid = next(iter(sorted(g.active_branch_ids)))
        r1 = g.sample_quartets(bid, 50, random.Random(0))
        r2 = g.sample_quartets(bid, 50, random.Random(99999))
        assert r1 != r2


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

class TestPrune:
    def test_4leaf_prune_removes_only_branch(self):
        g = _graph4()
        (bid,) = g.active_branch_ids
        g.prune(bid)
        assert g.active_branch_ids == set()

    def test_returns_set(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        dirty = g.prune(bid)
        assert isinstance(dirty, set)

    def test_dirty_is_subset_of_remaining_active(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        dirty = g.prune(bid)
        assert dirty <= g.active_branch_ids

    def test_pruned_branch_no_longer_active(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        g.prune(bid)
        assert bid not in g.active_branch_ids

    def test_prune_inactive_raises(self):
        g = _graph4()
        (bid,) = g.active_branch_ids
        g.prune(bid)
        with pytest.raises(AssertionError, match="already inactive"):
            g.prune(bid)

    def test_prune_leaf_edge_raises(self):
        g = _graph4()
        leaf_eids = [e.edge_id for e in g._edges if e.active and not e.is_internal]
        assert leaf_eids, "Expected at least one active leaf edge"
        with pytest.raises(AssertionError, match="leads to a leaf"):
            g.prune(leaf_eids[0])

    def test_active_count_decreases_by_one(self):
        g = _graph8()
        before = len(g.active_branch_ids)
        bid = next(iter(g.active_branch_ids))
        g.prune(bid)
        assert len(g.active_branch_ids) == before - 1

    def test_root_interval_unchanged_after_prune(self):
        """Root [start, end] must always be [0, n_leaves-1]."""
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        g.prune(bid)
        root = g._nodes[g._root]
        assert root.start == 0
        assert root.end == g._n_leaves - 1

    def test_neighbor_n_qsupp_callable_after_prune(self):
        """Dirty neighbors of a pruned branch remain scoreable."""
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        dirty = g.prune(bid)
        for dbid in dirty:
            n = g.n_qsupp(dbid)
            assert isinstance(n, int) and n >= 0

    def test_prune_all_branches(self):
        """Prune all 5 branches of the 8-leaf tree; graph becomes a star."""
        g = _graph8()
        while g.active_branch_ids:
            bid = next(iter(g.active_branch_ids))
            g.prune(bid)
        assert g.active_branch_ids == set()


# ---------------------------------------------------------------------------
# to_newick
# ---------------------------------------------------------------------------

class TestToNewick:
    def test_ends_with_semicolon(self):
        assert _graph4().to_newick().endswith(";")

    def test_contains_all_4_taxa(self):
        nwk = _graph4().to_newick()
        for name in NAMES4:
            assert name in nwk

    def test_contains_all_8_taxa(self):
        nwk = _graph8().to_newick()
        for name in NAMES8:
            assert name in nwk

    def test_each_taxon_appears_once(self):
        nwk = _graph8().to_newick()
        for name in NAMES8:
            assert nwk.count(name) == 1

    def test_4leaf_after_prune_is_star(self):
        """After pruning, the single inner subtree is flattened → no nested parens."""
        g = _graph4()
        (bid,) = g.active_branch_ids
        g.prune(bid)
        nwk = g.to_newick()
        # Strip outermost parens and semicolon
        inner = nwk.strip().rstrip(";").strip("()")
        assert "(" not in inner

    def test_each_taxon_still_appears_once_after_prune(self):
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        g.prune(bid)
        nwk = g.to_newick()
        for name in NAMES8:
            assert nwk.count(name) == 1

    def test_parseable_after_prune(self):
        """to_newick output can be re-parsed by our NEWICK parser."""
        g = _graph8()
        bid = next(iter(g.active_branch_ids))
        g.prune(bid)
        nwk = g.to_newick()
        parsed = _parse_newick(nwk)
        assert len(parsed["children"]) > 0

    def test_all_branches_pruned_gives_star(self):
        g = _graph8()
        while g.active_branch_ids:
            g.prune(next(iter(g.active_branch_ids)))
        nwk = g.to_newick()
        for name in NAMES8:
            assert nwk.count(name) == 1
        # No nested parens: all taxa are direct root children
        inner = nwk.strip().rstrip(";").strip("()")
        assert "(" not in inner


# ---------------------------------------------------------------------------
# Bipartition context
# ---------------------------------------------------------------------------

class TestBipartitionContext:
    def test_leaves_covered_by_both_sides(self):
        """For every active branch, u_size + v_size == n_leaves."""
        g = _graph8()
        for bid in g.active_branch_ids:
            edge = g._edges[bid]
            side_u, side_v = g._bipartition_context(edge)
            u_size = sum(_component_size(c) for c in side_u)
            v_size = sum(_component_size(c) for c in side_v)
            assert u_size + v_size == g._n_leaves

    def test_root_adjacent_branch_no_complement(self):
        """
        For a root-adjacent branch, side_u consists only of siblings
        (no parent-complement component, since parent IS the root).
        Verified by: u_size == n_leaves - child_subtree_size.
        """
        g = _graph8()
        for bid in g.active_branch_ids:
            edge = g._edges[bid]
            if g._nodes[edge.parent].parent == -1:  # root-adjacent
                child = g._nodes[edge.child]
                child_size = child.end - child.start + 1
                side_u, _ = g._bipartition_context(edge)
                u_size = sum(_component_size(c) for c in side_u)
                assert u_size == g._n_leaves - child_size
                break

    def test_non_root_branch_has_complement_component(self):
        """
        Non-root branches include a complement component in side_u.
        EF and GH branches (parent = EFGH node) are non-root-adjacent.
        """
        g = _graph8()
        non_root_bids = [
            bid for bid in g.active_branch_ids
            if g._nodes[g._edges[bid].parent].parent != -1
        ]
        assert non_root_bids, "Expected at least one non-root-adjacent branch"
        for bid in non_root_bids:
            edge = g._edges[bid]
            side_u, _ = g._bipartition_context(edge)
            # At minimum one component (the complement)
            assert len(side_u) >= 1

    def test_side_v_matches_child_subtree(self):
        """side_v components collectively cover exactly the child's [start, end]."""
        g = _graph8()
        for bid in g.active_branch_ids:
            edge = g._edges[bid]
            _, side_v = g._bipartition_context(edge)
            child = g._nodes[edge.child]
            v_size = sum(_component_size(c) for c in side_v)
            child_size = child.end - child.start + 1
            assert v_size == child_size
