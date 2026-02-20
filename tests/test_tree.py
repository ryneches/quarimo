"""
tests/test_tree.py
========================
Pytest test suite for the Tree class.

Tree fixtures
-------------
Four reference trees are loaded from .tree files in tests/trees/:

  balanced_4leaf.tree
      ((A:0.1,B:0.2)0.95:0.5,(C:0.3,D:0.4)0.87:0.6);

      Node IDs (parse_newick left-to-right leaf order, then post-order internals):
        A=0  B=1  C=2  D=3  AB=4  CD=5  root=6

      root_distance: A=0.6 B=0.7 C=0.9 D=1.0 AB=0.5 CD=0.6 root=0.0

  two_leaf.tree
      (Alpha:1.0,Beta:2.0)0.99;

      Node IDs: Alpha=0  Beta=1  root=2

  caterpillar_5leaf.tree
      (A:1,(B:1,(C:1,(D:1,E:1):1):1):1);

      Node IDs: A=0 B=1 C=2 D=3 E=4  DE=5 CDE=6 BCDE=7 root=8

      root_distance: A=1 B=2 C=3 D=4 E=4
                     DE=3 CDE=2 BCDE=1 root=0

  asymmetric_4leaf.tree  [regression tree for four-point condition bug fix]
      ((A:1,(B:1,C:1):1):1,D:1);

      Node IDs: A=0  B=1  C=2  D=3  BC=4  ABC=5  root=6

      root_distance: A=2 B=3 C=3 D=1 BC=2 ABC=1 root=0

      B and C are sisters → correct split is ALWAYS (B,C)|(A,D) = (0,3,1,2).
      The old DFS-adjacent-pairs algorithm returned (A,B)|(C,D) = (0,1,2,3)
      for this tree because D's late DFS position places it non-adjacent to
      its correct partner A.  The four-point condition fix is correct here.
"""

import os
import math
import itertools
import pytest
import numpy as np

# Locate the tree files relative to this test file so the tests can be
# run from any working directory.
_TREES_DIR = os.path.join(os.path.dirname(__file__), "trees")

# Add parent directory to path so Tree can be imported regardless of
# whether the package has been installed.
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quarimo._tree import Tree


# ======================================================================== #
# Helper                                                                    #
# ======================================================================== #


def load_tree(filename: str) -> Tree:
    """
    Load a NEWICK string from *filename* (inside tests/trees/) and return a
    fully constructed Tree.  Strips trailing whitespace and newlines.
    """
    path = os.path.join(_TREES_DIR, filename)
    with open(path) as fh:
        newick = fh.read().strip()
    return Tree(newick)


def taxa_names(tree: Tree) -> list:
    """Return the non-empty names from *tree* (leaf names only)."""
    return [name for name in tree.names if name != ""]


# ======================================================================== #
# Fixtures                                                                  #
# ======================================================================== #


@pytest.fixture(scope="module")
def balanced():
    """4-leaf balanced tree: ((A:0.1,B:0.2)0.95:0.5,(C:0.3,D:0.4)0.87:0.6)"""
    return load_tree("balanced_4leaf.tree")


@pytest.fixture(scope="module")
def two_leaf():
    """2-leaf tree: (Alpha:1.0,Beta:2.0)0.99"""
    return load_tree("two_leaf.tree")


@pytest.fixture(scope="module")
def caterpillar():
    """5-leaf caterpillar: (A:1,(B:1,(C:1,(D:1,E:1):1):1):1)"""
    return load_tree("caterpillar_5leaf.tree")


@pytest.fixture(scope="module")
def asymmetric():
    """4-leaf asymmetric tree: ((A:1,(B:1,C:1):1):1,D:1)
    B and C are sisters; correct split for {A,B,C,D} is (B,C)|(A,D).
    This tree was used to expose the DFS-adjacent-pairs bug.
    """
    return load_tree("asymmetric_4leaf.tree")


# ======================================================================== #
# 0. Regression — four-point condition bug fix                             #
#                                                                           #
# The old _quartet_split_core sorted nodes by first_occurrence and claimed #
# that "adjacent pairs" in DFS order always give the correct split.  This  #
# is FALSE for unbalanced quartets where one pair's LCA equals the overall #
# LCA of all four nodes.  These tests verify the fix is in place.          #
# ======================================================================== #


class TestQuartetFourPointRegressions:
    """
    Regression tests that specifically target the class of inputs where the
    old DFS-adjacent-pairs algorithm returned the wrong topology.

    Failure signature of the old bug
    ---------------------------------
    For a quartet where the correct split is (P,Q)|(R,S) and LCA(R,S) equals
    the overall LCA (i.e. R and S span the entire tree), the DFS-sorted order
    placed P and Q non-adjacently, producing (P,R)|(Q,S) or (P,S)|(Q,R) —
    neither of which groups the sisters P,Q together.

    The asymmetric_4leaf tree ((A,(B,C)),D) is the minimal example:
      · B and C are sisters  → must appear in the same pair
      · LCA(A,D) = root      → old algorithm's "adjacent pairs" split (A,B)|(C,D)
      · Correct split        → (B,C)|(A,D) = canonical (0,3,1,2)
    """

    def test_sisters_grouped_by_id(self, asymmetric):
        """B(1) and C(2) must always be in the same pair."""
        result = asymmetric.quartet_split(0, 1, 2, 3)
        p0, p1, q0, q1 = result
        pair1 = frozenset({p0, p1})
        pair2 = frozenset({q0, q1})
        assert frozenset({1, 2}) in {pair1, pair2}, (
            f"Sisters B(1) and C(2) are in different pairs: "
            f"{{{p0},{p1}}} | {{{q0},{q1}}}"
        )

    def test_sisters_grouped_by_name(self, asymmetric):
        """Named query: B and C must always be in the same pair."""
        p0, p1, q0, q1 = asymmetric.quartet_split("A", "B", "C", "D")
        bc_together = ({asymmetric.names[p0], asymmetric.names[p1]} == {"B", "C"}) or (
            {asymmetric.names[q0], asymmetric.names[q1]} == {"B", "C"}
        )
        assert bc_together, (
            f"Sisters B,C split across pairs: "
            f"{asymmetric.names[p0]},{asymmetric.names[p1]} | "
            f"{asymmetric.names[q0]},{asymmetric.names[q1]}"
        )

    def test_canonical_split(self, asymmetric):
        """Exact canonical split: (A,D)|(B,C) → (0,3,1,2)."""
        assert asymmetric.quartet_split(0, 1, 2, 3) == (0, 3, 1, 2)

    def test_canonical_split_by_name(self, asymmetric):
        assert asymmetric.quartet_split("A", "B", "C", "D") == (0, 3, 1, 2)

    def test_old_bug_result_is_rejected(self, asymmetric):
        """The old wrong answer (0,1,2,3) — (A,B)|(C,D) — must NOT be returned."""
        result = asymmetric.quartet_split(0, 1, 2, 3)
        assert result != (0, 1, 2, 3), (
            "Got (0,1,2,3) = (A,B)|(C,D), which is the known-wrong answer "
            "produced by the old DFS-adjacent-pairs algorithm."
        )

    def test_all_24_permutations_give_correct_split(self, asymmetric):
        """Input order must not affect the result."""
        expected = (0, 3, 1, 2)
        for perm in itertools.permutations([0, 1, 2, 3]):
            result = asymmetric.quartet_split(*perm)
            assert result == expected, (
                f"Permutation {perm} returned {result}, expected {expected}"
            )

    def test_frozenset_topology(self, asymmetric):
        expected = frozenset({frozenset({0, 3}), frozenset({1, 2})})
        assert asymmetric.quartet_topology(0, 1, 2, 3) == expected

    def test_topo_index_for_asymmetric(self, asymmetric):
        """
        Sorted IDs: n0=0,n1=1,n2=2,n3=3.
        Split (0,3)|(1,2): n0=0 partners with n3=3 → topo index 2.
        """
        split = asymmetric.quartet_split(0, 1, 2, 3)
        ti = Tree.quartet_topo_index(*split, 0, 1, 2, 3)
        assert ti == 2, f"Expected topo index 2, got {ti}"

    def test_steiner_length(self, asymmetric):
        """
        All branch lengths = 1.  The Steiner tree for {A,B,C,D} spans:
          A-ABC + ABC-root + root-D + ABC-BC + BC-B + BC-C = 6 edges.
        """
        _, _, _, _, st = asymmetric.quartet_split(0, 1, 2, 3, return_steiner=True)
        assert abs(st - 6.0) < 1e-10, f"Expected Steiner=6.0, got {st}"

    @pytest.mark.parametrize(
        "tree_name,nodes,correct_pair,wrong_pair",
        [
            # asymmetric_4leaf: B,C sisters → must be grouped; old bug split A,B
            (
                "asymmetric_4leaf.tree",
                [0, 1, 2, 3],
                frozenset({1, 2}),
                frozenset({0, 1}),
            ),
            # balanced_4leaf: A,B sisters AND C,D sisters → both pairs correct
            ("balanced_4leaf.tree", [0, 1, 2, 3], frozenset({0, 1}), None),
        ],
    )
    def test_sisters_are_grouped_parametric(
        self, tree_name, nodes, correct_pair, wrong_pair
    ):
        """Sisters must be in the same pair; the known-wrong pair must not appear."""
        tree = load_tree(tree_name)
        p0, p1, q0, q1 = tree.quartet_split(*nodes)
        pair1 = frozenset({p0, p1})
        pair2 = frozenset({q0, q1})
        assert correct_pair in {pair1, pair2}, (
            f"{tree_name}: expected {correct_pair} as one pair, got {pair1} | {pair2}"
        )
        if wrong_pair is not None:
            assert wrong_pair not in {pair1, pair2}, (
                f"{tree_name}: wrong pair {wrong_pair} should not appear, "
                f"got {pair1} | {pair2}"
            )


# ======================================================================== #
# 1. Scalar properties                                                      #
# ======================================================================== #


class TestScalarProperties:
    def test_balanced_n_nodes(self, balanced):
        assert balanced.n_nodes == 7

    def test_balanced_n_leaves(self, balanced):
        assert balanced.n_leaves == 4

    def test_balanced_root(self, balanced):
        assert balanced.root == 6

    def test_balanced_max_depth(self, balanced):
        assert balanced.max_depth == 2

    def test_two_leaf_n_nodes(self, two_leaf):
        assert two_leaf.n_nodes == 3

    def test_two_leaf_n_leaves(self, two_leaf):
        assert two_leaf.n_leaves == 2

    def test_two_leaf_root(self, two_leaf):
        assert two_leaf.root == 2

    def test_two_leaf_max_depth(self, two_leaf):
        assert two_leaf.max_depth == 1

    def test_caterpillar_n_nodes(self, caterpillar):
        assert caterpillar.n_nodes == 9

    def test_caterpillar_n_leaves(self, caterpillar):
        assert caterpillar.n_leaves == 5

    def test_caterpillar_root(self, caterpillar):
        assert caterpillar.root == 8

    def test_caterpillar_max_depth(self, caterpillar):
        assert caterpillar.max_depth == 4


# ======================================================================== #
# 2. Tree structure arrays (parse_newick)                                   #
# ======================================================================== #


class TestTreeStructure:
    def test_names_leaves(self, balanced):
        assert taxa_names(balanced) == ["A", "B", "C", "D"]

    def test_names_internals_empty(self, balanced):
        for i in range(balanced.n_leaves, balanced.n_nodes):
            assert balanced.names[i] == ""

    def test_parent_array(self, balanced):
        assert list(balanced.parent) == [4, 4, 5, 5, 6, 6, -1]

    def test_left_child_array(self, balanced):
        assert list(balanced.left_child) == [-1, -1, -1, -1, 0, 2, 4]

    def test_right_child_array(self, balanced):
        assert list(balanced.right_child) == [-1, -1, -1, -1, 1, 3, 5]

    def test_leaf_distances(self, balanced):
        assert abs(balanced.distance[0] - 0.1) < 1e-12  # A
        assert abs(balanced.distance[1] - 0.2) < 1e-12  # B
        assert abs(balanced.distance[2] - 0.3) < 1e-12  # C
        assert abs(balanced.distance[3] - 0.4) < 1e-12  # D

    def test_internal_distances(self, balanced):
        assert abs(balanced.distance[4] - 0.5) < 1e-12  # AB
        assert abs(balanced.distance[5] - 0.6) < 1e-12  # CD

    def test_root_distance_sentinel(self, balanced):
        assert balanced.distance[balanced.root] == -1.0

    def test_support_internal(self, balanced):
        assert abs(balanced.support[4] - 0.95) < 1e-12
        assert abs(balanced.support[5] - 0.87) < 1e-12

    def test_support_root_sentinel(self, balanced):
        assert balanced.support[balanced.root] == -1.0

    def test_support_leaf_sentinel(self, balanced):
        for i in range(balanced.n_leaves):
            assert balanced.support[i] == -1.0

    def test_leaf_children_sentinel(self, balanced):
        for i in range(balanced.n_leaves):
            assert balanced.left_child[i] == -1
            assert balanced.right_child[i] == -1

    def test_root_parent_sentinel(self, balanced):
        assert balanced.parent[balanced.root] == -1

    @pytest.mark.parametrize(
        "tree_name,expected_leaves",
        [
            ("balanced_4leaf.tree", ["A", "B", "C", "D"]),
            ("two_leaf.tree", ["Alpha", "Beta"]),
            ("caterpillar_5leaf.tree", ["A", "B", "C", "D", "E"]),
        ],
    )
    def test_leaf_names_across_trees(self, tree_name, expected_leaves):
        tree = load_tree(tree_name)
        assert taxa_names(tree) == expected_leaves


# ======================================================================== #
# 3. LCA structures (Euler tour, sparse table)                              #
# ======================================================================== #


class TestLCAStructures:
    def test_euler_tour_length(self, balanced):
        assert len(balanced.euler_tour) == 2 * balanced.n_nodes - 1

    def test_euler_tour_values(self, balanced):
        assert list(balanced.euler_tour) == [6, 4, 0, 4, 1, 4, 6, 5, 2, 5, 3, 5, 6]

    def test_euler_depth_values(self, balanced):
        assert list(balanced.euler_depth) == [0, 1, 2, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0]

    def test_depth_array(self, balanced):
        assert list(balanced.depth) == [2, 2, 2, 2, 1, 1, 0]

    def test_root_distance_leaves(self, balanced):
        assert abs(balanced.root_distance[0] - 0.6) < 1e-12  # A
        assert abs(balanced.root_distance[1] - 0.7) < 1e-12  # B
        assert abs(balanced.root_distance[2] - 0.9) < 1e-12  # C
        assert abs(balanced.root_distance[3] - 1.0) < 1e-12  # D

    def test_root_distance_internals(self, balanced):
        assert abs(balanced.root_distance[4] - 0.5) < 1e-12  # AB
        assert abs(balanced.root_distance[5] - 0.6) < 1e-12  # CD
        assert balanced.root_distance[6] == 0.0  # root

    def test_first_occurrence(self, balanced):
        fo = balanced.first_occurrence
        assert fo[6] == 0
        assert fo[4] == 1
        assert fo[0] == 2
        assert fo[1] == 4
        assert fo[5] == 7
        assert fo[2] == 8
        assert fo[3] == 10

    def test_sparse_table_level0_identity(self, balanced):
        assert list(balanced.sparse_table[0]) == list(range(13))

    def test_log2_table_spot_checks(self, balanced):
        lg = balanced.log2_table
        assert lg[1] == 0
        assert lg[2] == 1
        assert lg[3] == 1
        assert lg[4] == 2
        assert lg[7] == 2
        assert lg[8] == 3

    def test_sparse_table_shape(self, balanced):
        tour_len = 2 * balanced.n_nodes - 1
        LOG = int(math.floor(math.log2(tour_len))) + 1
        assert balanced.sparse_table.shape == (LOG, tour_len)


# ======================================================================== #
# 4. LCA pairwise queries                                                   #
# ======================================================================== #


class TestLCA:
    @pytest.mark.parametrize(
        "u,v,expected",
        [
            (0, 1, 4),  # LCA(A,B)   = AB
            (2, 3, 5),  # LCA(C,D)   = CD
            (0, 2, 6),  # LCA(A,C)   = root
            (0, 3, 6),  # LCA(A,D)   = root
            (1, 2, 6),  # LCA(B,C)   = root
            (1, 3, 6),  # LCA(B,D)   = root
            (4, 5, 6),  # LCA(AB,CD) = root
            (0, 4, 4),  # LCA(A,AB)  = AB
            (1, 4, 4),  # LCA(B,AB)  = AB
            (2, 5, 5),  # LCA(C,CD)  = CD
            (3, 5, 5),  # LCA(D,CD)  = CD
            (0, 6, 6),  # LCA(A,root)= root
            (4, 6, 6),  # LCA(AB,root)=root
        ],
    )
    def test_lca_by_id(self, balanced, u, v, expected):
        assert balanced.lca(u, v) == expected

    @pytest.mark.parametrize(
        "u,v,expected",
        [
            (0, 0, 0),  # reflexive: leaf
            (4, 4, 4),  # reflexive: internal
            (6, 6, 6),  # reflexive: root
        ],
    )
    def test_lca_reflexive(self, balanced, u, v, expected):
        assert balanced.lca(u, v) == expected

    @pytest.mark.parametrize(
        "u,v,expected",
        [
            ("A", "B", 4),
            ("C", "D", 5),
            ("A", "C", 6),
            ("B", "D", 6),
        ],
    )
    def test_lca_by_name(self, balanced, u, v, expected):
        assert balanced.lca(u, v) == expected

    def test_lca_name_reflexive(self, balanced):
        assert balanced.lca("A", "A") == 0

    def test_lca_mixed_int_str(self, balanced):
        assert balanced.lca(0, "B") == 4
        assert balanced.lca("A", 1) == 4

    def test_lca_symmetry(self, balanced):
        for u, v in itertools.combinations(range(balanced.n_nodes), 2):
            assert balanced.lca(u, v) == balanced.lca(v, u)

    def test_lca_unknown_name_raises(self, balanced):
        with pytest.raises(KeyError):
            balanced.lca("X", "A")

    def test_lca_two_leaf(self, two_leaf):
        assert two_leaf.lca(0, 1) == 2
        assert two_leaf.lca("Alpha", "Beta") == 2
        assert two_leaf.lca("Alpha", "Alpha") == 0

    @pytest.mark.parametrize(
        "u,v,expected",
        [
            ("D", "E", 5),
            ("C", "D", 6),
            ("B", "C", 7),
            ("A", "E", 8),
            ("A", "B", 8),
        ],
    )
    def test_lca_caterpillar(self, caterpillar, u, v, expected):
        assert caterpillar.lca(u, v) == expected


# ======================================================================== #
# 5. Branch distance                                                        #
# ======================================================================== #

EPS = 1e-10


class TestBranchDistance:
    @pytest.mark.parametrize(
        "u,v,expected_dist,expected_lca",
        [
            (0, 1, 0.3, 4),  # dist(A,B)   = 0.1+0.2
            (2, 3, 0.7, 5),  # dist(C,D)   = 0.3+0.4
            (0, 2, 1.5, 6),  # dist(A,C)   = 0.6+0.9
            (0, 3, 1.6, 6),
            (1, 2, 1.6, 6),
            (1, 3, 1.7, 6),
            (0, 4, 0.1, 4),  # dist(A,AB)  = branch A
            (4, 5, 1.1, 6),  # dist(AB,CD) = 0.5+0.6
            (0, 6, 0.6, 6),  # dist(A,root)= rd[A]
            (0, 0, 0.0, 0),  # same node
        ],
    )
    def test_branch_distance_with_lca(
        self, balanced, u, v, expected_dist, expected_lca
    ):
        dist, lca_id = balanced.branch_distance(u, v, return_lca=True)
        assert abs(dist - expected_dist) < EPS
        assert lca_id == expected_lca

    def test_branch_distance_without_lca(self, balanced):
        dist = balanced.branch_distance(0, 1)
        assert abs(dist - 0.3) < EPS

    def test_branch_distance_symmetry(self, balanced):
        for u, v in itertools.combinations(range(balanced.n_nodes), 2):
            assert (
                abs(balanced.branch_distance(u, v) - balanced.branch_distance(v, u))
                < EPS
            )

    def test_branch_distance_by_name(self, balanced):
        dist, lca_id = balanced.branch_distance("A", "B", return_lca=True)
        assert abs(dist - 0.3) < EPS
        assert lca_id == 4

    def test_branch_distance_mixed(self, balanced):
        assert abs(balanced.branch_distance(0, "B") - 0.3) < EPS

    def test_branch_distance_unknown_name(self, balanced):
        with pytest.raises(KeyError):
            balanced.branch_distance("X", "A")

    def test_branch_distance_two_leaf(self, two_leaf):
        dist, lca_id = two_leaf.branch_distance("Alpha", "Beta", return_lca=True)
        assert abs(dist - 3.0) < EPS
        assert lca_id == 2

    @pytest.mark.parametrize(
        "u,v,expected",
        [
            ("D", "E", 2.0),
            ("C", "D", 3.0),
            ("B", "E", 4.0),
            ("A", "E", 5.0),
            ("A", "B", 3.0),
        ],
    )
    def test_branch_distance_caterpillar(self, caterpillar, u, v, expected):
        assert abs(caterpillar.branch_distance(u, v) - expected) < EPS


# ======================================================================== #
# 6. multi_lca — LCA of N nodes                                             #
# ======================================================================== #


class TestMultiLCA:
    @pytest.mark.parametrize(
        "nodes,expected_lca",
        [
            ([0], 0),
            ([6], 6),
            ([0, 1], 4),
            ([2, 3], 5),
            ([0, 2], 6),
            ([0, 1, 2], 6),
            ([0, 1, 2, 3], 6),
            ([0, 1, 4], 4),  # ancestor already in set
            ([2, 3, 5], 5),  # ancestor already in set
            ([4, 5], 6),
            ([0, 6], 6),
            ([0, 0], 0),  # duplicate
            ([0, 0, 1], 4),  # duplicate + sibling
        ],
    )
    def test_multi_lca_by_id(self, balanced, nodes, expected_lca):
        assert balanced.multi_lca(nodes) == expected_lca

    @pytest.mark.parametrize(
        "nodes,expected_lca",
        [
            (["A", "B"], 4),
            (["A", "C"], 6),
            (["A", "B", "C", "D"], 6),
        ],
    )
    def test_multi_lca_by_name(self, balanced, nodes, expected_lca):
        assert balanced.multi_lca(nodes) == expected_lca

    def test_multi_lca_mixed(self, balanced):
        assert balanced.multi_lca([0, "B"]) == 4

    def test_multi_lca_empty_raises(self, balanced):
        with pytest.raises(ValueError):
            balanced.multi_lca([])

    def test_multi_lca_unknown_name_raises(self, balanced):
        with pytest.raises(KeyError):
            balanced.multi_lca(["A", "X"])

    def test_multi_lca_order_independent(self, balanced):
        """All permutations of {A,B,C,D} must yield the same LCA."""
        ref = balanced.multi_lca([0, 1, 2, 3])
        for perm in itertools.permutations([0, 1, 2, 3]):
            assert balanced.multi_lca(list(perm)) == ref

    def test_multi_lca_two_leaf(self, two_leaf):
        assert two_leaf.multi_lca([0, 1]) == 2

    @pytest.mark.parametrize(
        "nodes,expected_lca",
        [
            (["D", "E"], 5),
            (["C", "D", "E"], 6),
            (["B", "C", "D", "E"], 7),
            (["A", "B"], 8),
            (list("ABCDE"), 8),
        ],
    )
    def test_multi_lca_caterpillar(self, caterpillar, nodes, expected_lca):
        assert caterpillar.multi_lca(nodes) == expected_lca


# ======================================================================== #
# 7. Steiner lengths                                                        #
# ======================================================================== #


class TestSteinerLength:
    """
    All expected Steiner lengths computed via:
        Steiner(S) = ½ · Σᵢ dist(sorted[i], sorted[(i+1) % n])
    where nodes are sorted by first_occurrence.
    """

    @pytest.mark.parametrize(
        "nodes,expected_lca,expected_st",
        [
            ([0], 0, 0.0),  # single node
            ([0, 0], 0, 0.0),  # duplicate
            ([0, 1], 4, 0.3),  # {A,B}
            ([2, 3], 5, 0.7),  # {C,D}
            ([0, 2], 6, 1.5),  # {A,C}
            ([0, 1, 2], 6, 1.7),  # {A,B,C}
            ([0, 1, 2, 3], 6, 2.1),  # all leaves
            ([0, 1, 4], 4, 0.3),  # {A,B,AB}: ancestor in set
            ([2, 3, 5], 5, 0.7),  # {C,D,CD}: ancestor in set
            ([4, 5], 6, 1.1),  # {AB,CD}
            ([0, 6], 6, 0.6),  # {A,root}
            ([0, 0, 1], 4, 0.3),  # {A,A,B} with duplicate
        ],
    )
    def test_steiner_by_id(self, balanced, nodes, expected_lca, expected_st):
        lca_id, st = balanced.multi_lca(nodes, return_steiner=True)
        assert lca_id == expected_lca
        assert abs(st - expected_st) < EPS

    @pytest.mark.parametrize(
        "nodes,expected_st",
        [
            (["A", "B"], 0.3),
            (["A", "B", "C", "D"], 2.1),
        ],
    )
    def test_steiner_by_name(self, balanced, nodes, expected_st):
        _, st = balanced.multi_lca(nodes, return_steiner=True)
        assert abs(st - expected_st) < EPS

    def test_steiner_order_independent(self, balanced):
        """All permutations of {A,B,C,D} must give the same Steiner length."""
        _, ref_st = balanced.multi_lca([0, 1, 2, 3], return_steiner=True)
        for perm in itertools.permutations([0, 1, 2, 3]):
            _, st = balanced.multi_lca(list(perm), return_steiner=True)
            assert abs(st - ref_st) < EPS

    def test_steiner_two_leaf(self, two_leaf):
        _, st = two_leaf.multi_lca([0, 1], return_steiner=True)
        assert abs(st - 3.0) < EPS

    @pytest.mark.parametrize(
        "nodes,expected_st",
        [
            (["D", "E"], 2.0),
            (["C", "D", "E"], 4.0),
            (["B", "C", "D", "E"], 6.0),
            (list("ABCDE"), 8.0),
            (["A", "E"], 5.0),
            (["B", "D"], 4.0),
        ],
    )
    def test_steiner_caterpillar(self, caterpillar, nodes, expected_st):
        _, st = caterpillar.multi_lca(nodes, return_steiner=True)
        assert abs(st - expected_st) < EPS


# ======================================================================== #
# 8. Quartet split (bare-metal)                                             #
# ======================================================================== #


class TestQuartetSplit:
    def test_all_permutations_give_same_split(self, balanced):
        expected = (0, 1, 2, 3)
        for perm in itertools.permutations([0, 1, 2, 3]):
            assert balanced.quartet_split(*perm) == expected

    def test_split_by_name(self, balanced):
        assert balanced.quartet_split("A", "B", "C", "D") == (0, 1, 2, 3)

    def test_split_mixed_int_str(self, balanced):
        assert balanced.quartet_split(0, "B", 2, "D") == (0, 1, 2, 3)

    def test_split_without_steiner(self, balanced):
        result = balanced.quartet_split(0, 1, 2, 3)
        assert result == (0, 1, 2, 3)

    def test_split_with_steiner(self, balanced):
        result = balanced.quartet_split(0, 1, 2, 3, return_steiner=True)
        assert len(result) == 5
        p0, p1, q0, q1, st = result
        assert (p0, p1, q0, q1) == (0, 1, 2, 3)
        assert abs(st - 2.1) < EPS

    def test_split_duplicate_raises(self, balanced):
        with pytest.raises(ValueError):
            balanced.quartet_split("A", "A", "C", "D")

    def test_split_unknown_name_raises(self, balanced):
        with pytest.raises(KeyError):
            balanced.quartet_split("A", "X", "C", "D")

    @pytest.mark.parametrize(
        "nodes,expected_split,expected_st",
        [
            (["A", "B", "C", "D"], (0, 1, 2, 3), 7.0),
            (["A", "B", "C", "E"], (0, 1, 2, 4), 7.0),
            (["A", "B", "D", "E"], (0, 1, 3, 4), 7.0),
            (["A", "C", "D", "E"], (0, 2, 3, 4), 7.0),
            (["B", "C", "D", "E"], (1, 2, 3, 4), 6.0),
        ],
    )
    def test_split_caterpillar(self, caterpillar, nodes, expected_split, expected_st):
        p0, p1, q0, q1, st = caterpillar.quartet_split(*nodes, return_steiner=True)
        assert (p0, p1, q0, q1) == expected_split
        assert abs(st - expected_st) < EPS


# ======================================================================== #
# 9. Topology index (static method)                                         #
# ======================================================================== #


class TestQuartetTopoIndex:
    """
    Quartet topology index encodes which of the three possible splits
    a quartet takes relative to four node IDs sorted as n0<n1<n2<n3:

      topo 0  ↔  (n0,n1) | (n2,n3)
      topo 1  ↔  (n0,n2) | (n1,n3)
      topo 2  ↔  (n0,n3) | (n1,n2)

    Test quartets use the balanced tree (nodes A=0,B=1,C=2,D=3,AB=4,CD=5,root=6).
    The correct splits below are verified by the four-point condition, NOT by
    the old DFS-adjacent-pairs algorithm.  The old tests for topo_1 and topo_2
    used quartets whose correct splits the old buggy algorithm coincidentally
    ordered so as to produce distinct topo indices — those expected splits were
    themselves wrong.  The new tests use quartets whose correct topologies are
    unambiguous.

    Topo 0 — {A,B,C,D} = {0,1,2,3}
        Correct split: (A,B)|(C,D) = (0,1)|(2,3).
        n0=0 partners n1=1 → topo 0.

    Topo 1 — {A,C,AB,CD} = {0,2,4,5}
        Correct split: (A,AB)|(C,CD) = (0,4)|(2,5) — A is inside AB's clade;
        C is inside CD's clade; the balanced-tree internal edge (AB-root) and
        (CD-root) each keep a leaf with its ancestor.
        Sorted IDs: n0=0,n1=2,n2=4,n3=5.
        n0=0 is in pair {0,4}: partners n2=4 → topo 1.

    Topo 2 — {A,C,D,AB} = {0,2,3,4}
        Correct split: (A,AB)|(C,D) = (0,4)|(2,3) — A stays with its direct
        ancestor AB; C and D share their internal node CD.
        Sorted IDs: n0=0,n1=2,n2=3,n3=4.
        n0=0 is in pair {0,4}: partners n3=4 → topo 2.
    """

    def test_topo_0_leaf_quartet(self, balanced):
        """Leaf quartet {A,B,C,D}: split (A,B)|(C,D) → topo 0."""
        s = balanced.quartet_split(0, 1, 2, 3)
        assert s == (0, 1, 2, 3)
        ti = Tree.quartet_topo_index(*s, 0, 1, 2, 3)
        assert ti == 0

    def test_topo_1_leaf_plus_ancestors(self, balanced):
        """
        {A,C,AB,CD} = {0,2,4,5}: split (A,AB)|(C,CD) = (0,4,2,5) → topo 1.
        n0=0 partners n2=4 in sorted order 0<2<4<5.
        """
        s = balanced.quartet_split(0, 2, 4, 5)
        assert s == (0, 4, 2, 5), f"Expected (0,4,2,5), got {s}"
        ti = Tree.quartet_topo_index(*s, 0, 2, 4, 5)
        assert ti == 1

    def test_topo_2_leaf_plus_ancestor(self, balanced):
        """
        {A,C,D,AB} = {0,2,3,4}: split (A,AB)|(C,D) = (0,4,2,3) → topo 2.
        n0=0 partners n3=4 in sorted order 0<2<3<4.
        """
        s = balanced.quartet_split(0, 2, 3, 4)
        assert s == (0, 4, 2, 3), f"Expected (0,4,2,3), got {s}"
        ti = Tree.quartet_topo_index(*s, 0, 2, 3, 4)
        assert ti == 2

    def test_topo_2_asymmetric_tree(self, asymmetric):
        """
        {A,B,C,D} in asymmetric tree ((A,(B,C)),D):
        B and C are sisters → split (B,C)|(A,D) = canonical (0,3,1,2).
        Sorted IDs: n0=0,n1=1,n2=2,n3=3.  n0=0 partners n3=3 → topo 2.
        """
        s = asymmetric.quartet_split(0, 1, 2, 3)
        assert s == (0, 3, 1, 2)
        ti = Tree.quartet_topo_index(*s, 0, 1, 2, 3)
        assert ti == 2

    def test_three_distinct_topo_indices(self, balanced):
        """One quartet per topo value; all three must be distinct."""
        s0 = balanced.quartet_split(0, 1, 2, 3)  # topo 0
        s1 = balanced.quartet_split(0, 2, 4, 5)  # topo 1
        s2 = balanced.quartet_split(0, 2, 3, 4)  # topo 2
        ti0 = Tree.quartet_topo_index(*s0, 0, 1, 2, 3)
        ti1 = Tree.quartet_topo_index(*s1, 0, 2, 4, 5)
        ti2 = Tree.quartet_topo_index(*s2, 0, 2, 3, 4)
        assert ti0 == 0
        assert ti1 == 1
        assert ti2 == 2
        assert len({ti0, ti1, ti2}) == 3

    def test_topo_index_is_static(self):
        """quartet_topo_index can be called without an instance."""
        ti = Tree.quartet_topo_index(0, 1, 2, 3, 0, 1, 2, 3)
        assert ti == 0


# ======================================================================== #
# 10. Quartet topology (frozenset wrapper)                                  #
# ======================================================================== #


class TestQuartetTopology:
    def test_frozenset_structure(self, balanced):
        topo = balanced.quartet_topology(0, 1, 2, 3)
        assert topo == frozenset({frozenset({0, 1}), frozenset({2, 3})})

    def test_frozenset_with_steiner(self, balanced):
        topo, st = balanced.quartet_topology(0, 1, 2, 3, return_steiner=True)
        assert topo == frozenset({frozenset({0, 1}), frozenset({2, 3})})
        assert abs(st - 2.1) < EPS

    def test_frozenset_by_name(self, balanced):
        topo = balanced.quartet_topology("A", "B", "C", "D")
        assert topo == frozenset({frozenset({"A", "B"}), frozenset({"C", "D"})})

    def test_frozenset_all_permutations(self, balanced):
        expected = frozenset({frozenset({0, 1}), frozenset({2, 3})})
        for perm in itertools.permutations([0, 1, 2, 3]):
            assert balanced.quartet_topology(*perm) == expected

    def test_frozenset_is_hashable(self, balanced):
        """frozenset of frozensets can be used as a dict key or set element."""
        topo = balanced.quartet_topology(0, 1, 2, 3)
        d = {topo: "balanced"}
        assert d[topo] == "balanced"


# ======================================================================== #
# 11. Name index caching                                                    #
# ======================================================================== #


class TestNameIndex:
    def test_name_index_built_lazily(self):
        """_name_index is None until a name-based query is made."""
        tree = load_tree("balanced_4leaf.tree")
        assert tree._name_index is None
        tree.lca("A", "B")
        assert tree._name_index is not None

    def test_name_index_not_rebuilt(self, balanced):
        """A second name query reuses the cached index."""
        balanced.lca("A", "B")  # ensure it is built
        id_before = id(balanced._name_index)
        balanced.lca("C", "D")
        assert id(balanced._name_index) == id_before

    def test_name_index_covers_all_leaves(self, balanced):
        balanced.lca("A", "B")  # trigger build
        for name in taxa_names(balanced):
            assert name in balanced._name_index


# ======================================================================== #
# 12. Private static _rmq (kernel unit test)                                #
# ======================================================================== #


class TestRMQKernel:
    """
    Test the private _rmq static method directly to verify the core kernel.
    These tests mirror the sparse-table structure tests but at the level of
    individual range queries.
    """

    def test_rmq_single_element(self, balanced):
        idx = Tree._rmq(
            2, 2, balanced.sparse_table, balanced.euler_depth, balanced.log2_table
        )
        assert idx == 2

    def test_rmq_full_tour(self, balanced):
        tour_len = len(balanced.euler_tour)
        idx = Tree._rmq(
            0,
            tour_len - 1,
            balanced.sparse_table,
            balanced.euler_depth,
            balanced.log2_table,
        )
        # Minimum depth over the full tour is 0 (root)
        assert balanced.euler_depth[idx] == 0
        assert balanced.euler_tour[idx] == balanced.root

    @pytest.mark.parametrize(
        "fo_u,fo_v,expected_lca",
        [
            # fo: A=2, B=4, AB=1, C=8, D=10, CD=7, root=0
            (2, 4, 4),  # range [2,4]: min-depth node between A and B → AB
            (8, 10, 5),  # range [8,10]: min-depth between C and D    → CD
            (2, 10, 6),  # range [2,10]: min-depth between A and D    → root
        ],
    )
    def test_rmq_known_ranges(self, balanced, fo_u, fo_v, expected_lca):
        idx = Tree._rmq(
            fo_u, fo_v, balanced.sparse_table, balanced.euler_depth, balanced.log2_table
        )
        assert balanced.euler_tour[idx] == expected_lca


# ======================================================================== #
# 13. Multifurcation resolution                                             #
# ======================================================================== #


class TestMultifurcationResolution:
    """
    Tests for _resolve_multifurcations() and the multifurcation-aware parser.

    Three multifurcating test trees:
      trifurcating_root.tree     — 15-leaf real-world tree with trifurcating root
      trifurcating_internal.tree — ((A,B,C):1,D:1) — internal trifurcation
      star_4leaf.tree            — (A,B,C,D) — star/unresolved topology
    """

    def test_warning_emitted_on_trifurcating_root(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            load_tree("trifurcating_root.tree")
        assert (
            len([r for r in caplog.records if "bifurcating" in r.getMessage().lower()])
            == 1
        )

    def test_no_warning_on_bifurcating_tree(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            load_tree("balanced_4leaf.tree")
        bifurcation_warnings = [
            r for r in caplog.records if "bifurcating" in r.getMessage().lower()
        ]
        assert len(bifurcation_warnings) == 0

    def test_trifurcating_root_n_leaves(self):
        t = load_tree("trifurcating_root.tree")
        assert t.n_leaves == 15

    def test_trifurcating_root_n_nodes(self):
        t = load_tree("trifurcating_root.tree")
        assert t.n_nodes == 29  # 2*15 - 1

    def test_trifurcating_root_distances_nonzero(self):
        """All leaves must have positive root_distance after resolution."""
        t = load_tree("trifurcating_root.tree")
        for i in range(t.n_leaves):
            assert t.root_distance[i] > 0, (
                f"Leaf {i} ({t.names[i]}) has root_distance=0"
            )

    def test_trifurcating_root_permutation_invariance(self):
        """The original bug: all 24 permutations must give the same topology."""
        t = load_tree("trifurcating_root.tree")
        names = ["Ccas", "Ocav", "Ohet", "Oche"]
        topologies = set()
        for perm in itertools.permutations(names):
            topologies.add(t.quartet_topology(*perm))
        assert len(topologies) == 1, (
            f"Got {len(topologies)} distinct topologies across permutations: {topologies}"
        )

    def test_trifurcating_root_sisters_grouped(self):
        """Oche and Ohet are sisters in the tree — they must share a pair."""
        t = load_tree("trifurcating_root.tree")
        topo = t.quartet_topology("Oche", "Ohet", "Ocav", "Ccas")
        assert frozenset({"Oche", "Ohet"}) in topo, (
            f"Sisters Oche/Ohet not grouped together in {topo}"
        )

    def test_trifurcating_internal_n_leaves(self):
        """((A,B,C):1,D:1) has 4 leaves."""
        t = load_tree("trifurcating_internal.tree")
        assert t.n_leaves == 4

    def test_trifurcating_internal_n_nodes(self):
        t = load_tree("trifurcating_internal.tree")
        assert t.n_nodes == 7  # 2*4 - 1

    def test_trifurcating_internal_topology(self):
        """
        ((A,B,C):1,D:1) after resolution: A,B,C are more closely related
        to each other than any is to D.  The only valid quartet split for
        {A,B,C,D} is one that groups two of {A,B,C} against D + the third.
        D must NOT be grouped with any of the {A,B,C} triplet as a sister pair
        while another two of {A,B,C} form the other pair.  In other words,
        the pair containing D must also contain exactly one of {A,B,C}.
        """
        t = load_tree("trifurcating_internal.tree")
        topo = t.quartet_topology("A", "B", "C", "D")
        # D must be paired with exactly one of {A,B,C}
        d_pair = next(p for p in topo if "D" in p)
        abc_pair = next(p for p in topo if "D" not in p)
        assert len(d_pair) == 2
        assert len(abc_pair) == 2
        assert len(abc_pair & {"A", "B", "C"}) == 2, (
            f"Expected 2 of {{A,B,C}} in one pair, got topo={topo}"
        )

    def test_star_4leaf_resolves(self):
        """(A,B,C,D) star tree resolves without error."""
        t = load_tree("star_4leaf.tree")
        assert t.n_leaves == 4
        assert t.n_nodes == 7
        # Any topology is valid; just check it returns without error
        topo = t.quartet_topology("A", "B", "C", "D")
        assert len(topo) == 2
        for pair in topo:
            assert len(pair) == 2

    def test_resolve_multifurcations_simple(self, caplog):
        """Unit test _resolve_multifurcations on a simple trifurcation."""
        import logging

        result = Tree._resolve_multifurcations("(A:1,B:2,C:3)")
        # Should be parseable as a bifurcating tree
        with caplog.at_level(logging.WARNING):
            t = Tree(result)
        bifurcation_warnings = [
            r for r in caplog.records if "bifurcating" in r.getMessage().lower()
        ]
        assert len(bifurcation_warnings) == 0, "Result is still multifurcating"
        assert t.n_leaves == 3
        assert t.n_nodes == 5

    def test_resolve_multifurcations_star(self, caplog):
        """(A,B,C,D) → strictly bifurcating 4-leaf tree."""
        import logging

        result = Tree._resolve_multifurcations("(A:1,B:1,C:1,D:1)")
        with caplog.at_level(logging.WARNING):
            t = Tree(result)
        bifurcation_warnings = [
            r for r in caplog.records if "bifurcating" in r.getMessage().lower()
        ]
        assert len(bifurcation_warnings) == 0
        assert t.n_leaves == 4
        assert t.n_nodes == 7


# ======================================================================== #
# 14. quartet_topology name/ID return type                                  #
# ======================================================================== #


class TestQuartetTopologyReturnType:
    """
    When all four inputs are strings, quartet_topology() must return
    frozensets of strings.  When any input is an integer, it must return
    frozensets of integers.
    """

    def test_all_names_returns_name_frozensets(self, balanced):
        topo = balanced.quartet_topology("A", "B", "C", "D")
        for pair in topo:
            for elem in pair:
                assert isinstance(elem, str), (
                    f"Expected str, got {type(elem).__name__}: {elem!r}"
                )

    def test_all_ids_returns_int_frozensets(self, balanced):
        topo = balanced.quartet_topology(0, 1, 2, 3)
        for pair in topo:
            for elem in pair:
                assert isinstance(elem, int), (
                    f"Expected int, got {type(elem).__name__}: {elem!r}"
                )

    def test_name_frozensets_correct_values(self, balanced):
        topo = balanced.quartet_topology("A", "B", "C", "D")
        assert topo == frozenset({frozenset({"A", "B"}), frozenset({"C", "D"})})

    def test_id_frozensets_correct_values(self, balanced):
        topo = balanced.quartet_topology(0, 1, 2, 3)
        assert topo == frozenset({frozenset({0, 1}), frozenset({2, 3})})

    def test_name_and_id_topologies_consistent(self, balanced):
        """Name-based and ID-based results should encode the same split."""
        topo_names = balanced.quartet_topology("A", "B", "C", "D")
        topo_ids = balanced.quartet_topology(0, 1, 2, 3)
        # Convert name topo to IDs for comparison
        ni = balanced._name_index if balanced._name_index else {}
        balanced._build_name_index()
        ni = balanced._name_index
        topo_names_as_ids = frozenset(
            frozenset(ni[n] for n in pair) for pair in topo_names
        )
        assert topo_names_as_ids == topo_ids

    def test_mixed_input_returns_int_frozensets(self, balanced):
        """Mixed int/str input → integer frozensets."""
        topo = balanced.quartet_topology(0, "B", 2, "D")
        for pair in topo:
            for elem in pair:
                assert isinstance(elem, int), (
                    f"Expected int for mixed input, got {type(elem).__name__}"
                )

    @pytest.mark.parametrize("perm", list(itertools.permutations(["A", "B", "C", "D"])))
    def test_name_topology_permutation_invariant(self, balanced, perm):
        """String-input topology is invariant under all 24 permutations."""
        expected = frozenset({frozenset({"A", "B"}), frozenset({"C", "D"})})
        assert balanced.quartet_topology(*perm) == expected

    def test_asymmetric_names_correct(self, asymmetric):
        """((A,(B,C)),D): name-based result groups B,C together."""
        topo = asymmetric.quartet_topology("A", "B", "C", "D")
        assert frozenset({"B", "C"}) in topo, f"Expected B,C grouped, got {topo}"
        assert topo == frozenset({frozenset({"A", "D"}), frozenset({"B", "C"})})

    def test_steiner_return_type_names(self, balanced):
        topo, st = balanced.quartet_topology("A", "B", "C", "D", return_steiner=True)
        assert isinstance(st, float)
        for pair in topo:
            for elem in pair:
                assert isinstance(elem, str)

    def test_steiner_return_type_ids(self, balanced):
        topo, st = balanced.quartet_topology(0, 1, 2, 3, return_steiner=True)
        assert isinstance(st, float)
        for pair in topo:
            for elem in pair:
                assert isinstance(elem, int)
