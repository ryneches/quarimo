"""
tests/test_paralog_csr.py
=========================
Tests for Step 2: paralog CSR table and assignment table.

Covers:
- paralog_leaf_offsets / paralog_leaf_nodes CSR consistency
- paralog_assignments shape and contents
- Absence sentinel (-1) for copy slots that exceed a tree's actual count
- Non-paralog forest: empty sentinel arrays
- Multi-genome: two independent paralog groups
"""

import pytest
import numpy as np

from quarimo import Forest


# ── Fixtures ─────────────────────────────────────────────────────────────────

# Two trees: tree0 has one copy of G; tree1 has two copies of G.
FOREST_TREES = [
    "(G1:1,(X:1,Y:1):1);",
    "((G1:1,G2:1):1,(X:1,Y:1):1);",
]
TAXON_MAP = {"G1": "G", "G2": "G"}

# Single tree with two copies of genome G and two singletons
SINGLE_TREE_2COPY = ["((G1:1,G2:1):1,(X:1,Y:1):1);"]
SINGLE_MAP_2COPY = {"G1": "G", "G2": "G"}

# Two independent paralog genomes (H1/H2 → H, K1/K2 → K) in one tree
DUAL_PARALOG_TREES = ["((H1:1,H2:1):1,(K1:1,K2:1):1);"]
DUAL_MAP = {"H1": "H", "H2": "H", "K1": "K", "K2": "K"}

# No paralogs
NO_PARALOG = ["((A:1,B:1):1,(C:1,D:1):1);"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def tree_leaves_for(f: Forest, li: int, ti: int) -> list:
    """Return local leaf IDs for paralog genome li in tree ti."""
    NT = f.n_trees
    base = li * (NT + 1)
    start = int(f.paralog_leaf_offsets[base + ti])
    end = int(f.paralog_leaf_offsets[base + ti + 1])
    return f.paralog_leaf_nodes[start:end].tolist()


# ── CSR consistency ───────────────────────────────────────────────────────────


class TestLeafCSRConsistency:
    def test_offsets_length(self):
        """paralog_leaf_offsets length == n_paralog_genomes * (n_trees + 1)."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        n_pg = len(f.paralog_genome_names)
        assert len(f.paralog_leaf_offsets) == n_pg * (f.n_trees + 1)

    def test_offsets_non_decreasing(self):
        """CSR offsets are non-decreasing within each genome's sub-array."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        NT = f.n_trees
        for li in range(len(f.paralog_genome_names)):
            sub = f.paralog_leaf_offsets[li * (NT + 1) : (li + 1) * (NT + 1)]
            assert np.all(np.diff(sub) >= 0)

    def test_total_leaf_count_sums_correctly(self):
        """Sum of per-tree leaf counts for each genome == len(paralog_leaf_nodes) slice."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        NT = f.n_trees
        # For genome G: tree0 has 1 copy, tree1 has 2 copies → total 3 leaves
        li = f.paralog_genome_names.index("G")
        base = li * (NT + 1)
        n_leaves = int(f.paralog_leaf_offsets[base + NT])  # last offset = total
        per_tree = [
            int(f.paralog_leaf_offsets[base + ti + 1]) - int(f.paralog_leaf_offsets[base + ti])
            for ti in range(NT)
        ]
        assert sum(per_tree) == n_leaves

    def test_tree0_one_leaf(self):
        """tree0 has only one G copy → exactly 1 leaf in CSR."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        leaves = tree_leaves_for(f, li, ti=0)
        assert len(leaves) == 1

    def test_tree1_two_leaves(self):
        """tree1 has two G copies → exactly 2 leaves in CSR."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        leaves = tree_leaves_for(f, li, ti=1)
        assert len(leaves) == 2

    def test_leaf_ids_are_valid(self):
        """All leaf IDs in paralog_leaf_nodes are valid local leaf IDs."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        # leaf IDs must be in [0, n_leaves-1] for their respective trees
        for lid in f.paralog_leaf_nodes.tolist():
            assert lid >= 0  # no sentinel -1 in leaf_nodes

    def test_no_duplicate_leaves_per_tree(self):
        """Within one tree, no leaf ID appears twice in paralog_leaf_nodes."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        for ti in range(f.n_trees):
            leaves = tree_leaves_for(f, li, ti)
            assert len(leaves) == len(set(leaves)), f"Duplicate in tree {ti}: {leaves}"


# ── Assignment table ──────────────────────────────────────────────────────────


class TestAssignmentTable:
    def test_shape(self):
        """paralog_assignments has shape (n_paralog_genomes, max_copies, n_trees)."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        n_pg = len(f.paralog_genome_names)
        # G has max 2 copies → max_copies = 2
        assert f.paralog_assignments.shape == (n_pg, 2, f.n_trees)

    def test_copy0_present_both_trees(self):
        """copy0 of G is assigned in both trees."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        assert f.paralog_assignments[li, 0, 0] >= 0, "G_copy0 absent in tree0"
        assert f.paralog_assignments[li, 0, 1] >= 0, "G_copy0 absent in tree1"

    def test_copy1_absent_tree0(self):
        """copy1 of G is absent (-1) in tree0 (only 1 G copy there)."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        assert f.paralog_assignments[li, 1, 0] == -1, "G_copy1 should be -1 in tree0"

    def test_copy1_present_tree1(self):
        """copy1 of G is assigned in tree1 (2 G copies there)."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        assert f.paralog_assignments[li, 1, 1] >= 0, "G_copy1 absent in tree1"

    def test_assignments_agree_with_global_to_local(self):
        """paralog_assignments[li, ci, ti] == global_to_local[ti, gid_for_ci]."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        li = f.paralog_genome_names.index("G")
        for ci in range(2):
            gid = int(f.paralog_copy_global_ids[int(f.paralog_copy_offsets[li]) + ci])
            for ti in range(f.n_trees):
                expected = int(f.global_to_local[ti, gid])
                actual = int(f.paralog_assignments[li, ci, ti])
                assert actual == expected, (
                    f"Mismatch at li={li} ci={ci} ti={ti}: "
                    f"assignment={actual} global_to_local={expected}"
                )

    def test_assignment_coverage(self):
        """Every leaf in paralog_leaf_nodes appears in paralog_assignments for its tree."""
        f = Forest(SINGLE_TREE_2COPY, taxon_map=SINGLE_MAP_2COPY)
        li = f.paralog_genome_names.index("G")
        ti = 0
        leaves_from_csr = set(tree_leaves_for(f, li, ti))
        leaves_from_assignments = {
            int(f.paralog_assignments[li, ci, ti])
            for ci in range(f.paralog_assignments.shape[1])
            if f.paralog_assignments[li, ci, ti] >= 0
        }
        assert leaves_from_csr == leaves_from_assignments

    def test_dtype(self):
        """paralog_assignments has dtype int32."""
        f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
        assert f.paralog_assignments.dtype == np.int32


# ── Dual paralog genomes ──────────────────────────────────────────────────────


class TestDualParalogGenomes:
    def test_two_genomes_detected(self):
        """Two independent paralog genomes are both listed."""
        f = Forest(DUAL_PARALOG_TREES, taxon_map=DUAL_MAP)
        assert set(f.paralog_genome_names) == {"H", "K"}

    def test_each_genome_two_copies(self):
        """Each genome has max_copies = 2."""
        f = Forest(DUAL_PARALOG_TREES, taxon_map=DUAL_MAP)
        assert f.paralog_assignments.shape[1] == 2

    def test_leaf_sets_disjoint(self):
        """The leaf sets for H and K are disjoint in the single tree."""
        f = Forest(DUAL_PARALOG_TREES, taxon_map=DUAL_MAP)
        li_h = f.paralog_genome_names.index("H")
        li_k = f.paralog_genome_names.index("K")
        h_leaves = set(tree_leaves_for(f, li_h, ti=0))
        k_leaves = set(tree_leaves_for(f, li_k, ti=0))
        assert h_leaves.isdisjoint(k_leaves), f"Overlap: {h_leaves & k_leaves}"

    def test_all_leaves_covered(self):
        """Together H and K leaves account for all 4 leaves in the tree."""
        f = Forest(DUAL_PARALOG_TREES, taxon_map=DUAL_MAP)
        li_h = f.paralog_genome_names.index("H")
        li_k = f.paralog_genome_names.index("K")
        combined = set(tree_leaves_for(f, li_h, ti=0)) | set(tree_leaves_for(f, li_k, ti=0))
        # Tree has 4 leaves (H1, H2, K1, K2 after remapping all become paralog groups)
        assert len(combined) == 4

    def test_offsets_length_dual(self):
        """offsets length == 2 * (n_trees + 1) for two paralog genomes."""
        f = Forest(DUAL_PARALOG_TREES, taxon_map=DUAL_MAP)
        assert len(f.paralog_leaf_offsets) == 2 * (f.n_trees + 1)


# ── No-paralog sentinel ───────────────────────────────────────────────────────


class TestNoParalogSentinels:
    def test_leaf_offsets_empty(self):
        f = Forest(NO_PARALOG)
        assert len(f.paralog_leaf_offsets) == 0

    def test_leaf_nodes_empty(self):
        f = Forest(NO_PARALOG)
        assert len(f.paralog_leaf_nodes) == 0

    def test_assignments_empty_first_dim(self):
        f = Forest(NO_PARALOG)
        assert f.paralog_assignments.shape[0] == 0
