"""
tests/test_forest.py
====================================
Pytest test suite for Forest.

Test data
---------
Trees are loaded from tests/trees/basic_collection.trees, which contains
three NEWICK strings (one per line):

  Tree 0 — 4-leaf balanced: ((A:0.1,B:0.2)0.95:0.5,(C:0.3,D:0.4)0.87:0.6);
  Tree 1 — 5-leaf caterpillar: (A:1,(B:1,(C:1,(D:1,E:1):1):1):1);
  Tree 2 — 4-leaf two-clade: ((A:0.5,C:0.5):0.3,(X:0.4,Y:0.4):0.3);

Global taxon namespace (7 taxa, sorted):
  gid  0=A  1=B  2=C  3=D  4=E  5=X  6=Y

global_to_local matrix (row=tree, col=gid, -1=absent):
       A   B   C   D   E   X   Y
  t0 [ 0   1   2   3  -1  -1  -1 ]
  t1 [ 0   1   2   3   4  -1  -1 ]
  t2 [ 0  -1   1  -1  -1   2   3 ]

Expected layout dimensions:
  node_offsets  = [0, 7, 16, 23]
  tour_offsets  = [0, 13, 30, 43]
  sp_offsets    = [0, 52, 137, 189]
  lg_offsets    = [0, 14, 32, 46]
  leaf_offsets  = [0, 4, 9, 13]

Expected properties and distances are loaded from CSV files in tests/data/:
  basic_namespace.csv          — scalar properties and offset vectors
  basic_global_to_local.csv    — per-(tree,taxon) local IDs
  basic_branch_distances.csv   — expected pairwise distances per tree
"""

import os
import sys
import csv
import math
import itertools

import pytest
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
_TREES_DIR = os.path.join(_HERE, "trees")
_DATA_DIR = os.path.join(_HERE, "data")

sys.path.insert(0, _ROOT)

from quarimo._forest import Forest
from quarimo._tree import Tree


# ======================================================================== #
# Helpers                                                                   #
# ======================================================================== #


def load_newick_file(filename: str) -> list:
    """Read a multi-NEWICK file (one tree per line) from tests/trees/."""
    path = os.path.join(_TREES_DIR, filename)
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_namespace_csv(filename: str) -> dict:
    """Load key→value pairs from a namespace CSV into a plain dict."""
    path = os.path.join(_DATA_DIR, filename)
    d = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d[row["property"]] = row["value"]
    return d


def load_global_to_local_csv(filename: str) -> dict:
    """Return dict keyed by (tree_index:int, global_id:int) → local_id:int."""
    path = os.path.join(_DATA_DIR, filename)
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[(int(row["tree_index"]), int(row["global_id"]))] = int(row["local_id"])
    return out


def load_branch_distances_csv(filename: str) -> dict:
    """Return dict keyed by (taxon_a, taxon_b, tree_index:int) → float|NaN."""
    path = os.path.join(_DATA_DIR, filename)
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["taxon_a"], row["taxon_b"], int(row["tree_index"]))
            val = row["expected_distance"]
            out[key] = float("nan") if val == "" else float(val)
    return out


# ======================================================================== #
# Fixtures                                                                  #
# ======================================================================== #


@pytest.fixture(scope="module")
def basic_collection():
    """Three-tree collection with partially overlapping taxa sets."""
    newicks = load_newick_file("basic_collection.trees")
    return Forest(newicks)


@pytest.fixture(scope="module")
def basic_namespace():
    return load_namespace_csv("basic_namespace.csv")


@pytest.fixture(scope="module")
def basic_g2l():
    return load_global_to_local_csv("basic_global_to_local.csv")


@pytest.fixture(scope="module")
def basic_distances():
    return load_branch_distances_csv("basic_branch_distances.csv")


# ======================================================================== #
# 1. Construction and scalar properties                                     #
# ======================================================================== #


class TestConstruction:
    def test_n_trees(self, basic_collection, basic_namespace):
        assert basic_collection.n_trees == int(basic_namespace["n_trees"])

    def test_n_global_taxa(self, basic_collection, basic_namespace):
        assert basic_collection.n_global_taxa == int(basic_namespace["n_global_taxa"])

    def test_global_names_sorted(self, basic_collection):
        assert basic_collection.global_names == sorted(basic_collection.global_names)

    def test_global_names_values(self, basic_collection, basic_namespace):
        expected = basic_namespace["global_taxa"].split(",")
        assert basic_collection.global_names == expected

    def test_per_tree_n_nodes(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees):
            assert basic_collection.per_tree_n_nodes[i] == int(
                basic_namespace[f"n_nodes_{i}"]
            )

    def test_per_tree_n_leaves(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees):
            assert basic_collection.per_tree_n_leaves[i] == int(
                basic_namespace[f"n_leaves_{i}"]
            )

    def test_per_tree_roots(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees):
            assert basic_collection.per_tree_roots[i] == int(
                basic_namespace[f"root_{i}"]
            )

    def test_per_tree_max_depth(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees):
            assert basic_collection.per_tree_max_depth[i] == int(
                basic_namespace[f"max_depth_{i}"]
            )

    def test_multifurcating_tree_gets_warning(self, caplog):
        """Test that multifurcating trees emit a consolidated WARNING."""
        import logging

        newick = "(A:1,B:1,C:1);"
        with caplog.at_level(logging.WARNING):
            Forest([newick])
        # Should have exactly one warning about multifurcation
        multifurcation_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "multifurcation" in r.getMessage().lower()
        ]
        assert len(multifurcation_warnings) == 1
        assert "1 tree" in multifurcation_warnings[0].getMessage()

    def test_multiple_multifurcating_trees_consolidated_warning(self, caplog):
        """Multiple multifurcating trees produce ONE consolidated warning."""
        import logging

        newicks = [
            "(A:1,B:1,C:1);",  # multifurcation
            "((A:1,B:1):1,C:1);",  # bifurcating
            "(X:1,Y:1,Z:1,W:1);",  # multifurcation (4-way)
        ]
        with caplog.at_level(logging.WARNING):
            Forest(newicks)
        multifurcation_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "multifurcation" in r.getMessage().lower()
        ]
        # Should have exactly ONE warning, not 2
        assert len(multifurcation_warnings) == 1
        # Should mention "2 trees"
        assert "2 trees" in multifurcation_warnings[0].getMessage()

    def test_single_tree_collection(self):
        c = Forest(["((A:1,B:1):1,(C:1,D:1):1);"])
        assert c.n_trees == 1
        assert c.n_global_taxa == 4


# ======================================================================== #
# 2. CSR layout — offset vectors                                            #
# ======================================================================== #


class TestCSROffsets:
    def test_node_offsets_values(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees + 1):
            assert basic_collection.node_offsets[i] == int(
                basic_namespace[f"node_offset_{i}"]
            )

    def test_tour_offsets_values(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees + 1):
            assert basic_collection.tour_offsets[i] == int(
                basic_namespace[f"tour_offset_{i}"]
            )

    def test_sp_offsets_values(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees + 1):
            assert basic_collection.sp_offsets[i] == int(
                basic_namespace[f"sp_offset_{i}"]
            )

    def test_lg_offsets_values(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees + 1):
            assert basic_collection.lg_offsets[i] == int(
                basic_namespace[f"lg_offset_{i}"]
            )

    def test_leaf_offsets_values(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees + 1):
            assert basic_collection.leaf_offsets[i] == int(
                basic_namespace[f"leaf_offset_{i}"]
            )

    def test_sp_log_widths(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees):
            assert basic_collection.sp_log_widths[i] == int(
                basic_namespace[f"sp_log_width_{i}"]
            )

    def test_sp_tour_widths(self, basic_collection, basic_namespace):
        for i in range(basic_collection.n_trees):
            assert basic_collection.sp_tour_widths[i] == int(
                basic_namespace[f"sp_tour_width_{i}"]
            )

    def test_node_offsets_monotone(self, basic_collection):
        offsets = basic_collection.node_offsets
        assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))

    def test_tour_offsets_monotone(self, basic_collection):
        offsets = basic_collection.tour_offsets
        assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))

    def test_sp_offsets_monotone(self, basic_collection):
        offsets = basic_collection.sp_offsets
        assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))

    def test_total_nodes_consistent(self, basic_collection, basic_namespace):
        assert basic_collection.node_offsets[-1] == int(basic_namespace["total_nodes"])
        assert basic_collection.node_offsets[-1] == len(basic_collection.all_parent)

    def test_total_tour_consistent(self, basic_collection, basic_namespace):
        assert basic_collection.tour_offsets[-1] == int(
            basic_namespace["total_tour_len"]
        )
        assert basic_collection.tour_offsets[-1] == len(basic_collection.all_euler_tour)

    def test_total_sp_consistent(self, basic_collection, basic_namespace):
        assert basic_collection.sp_offsets[-1] == int(
            basic_namespace["total_sp_entries"]
        )
        assert basic_collection.sp_offsets[-1] == len(basic_collection.all_sparse_table)

    def test_total_lg_consistent(self, basic_collection, basic_namespace):
        assert basic_collection.lg_offsets[-1] == int(
            basic_namespace["total_lg_entries"]
        )
        assert basic_collection.lg_offsets[-1] == len(basic_collection.all_log2_table)

    def test_sp_offsets_match_log_times_tour(self, basic_collection):
        """sp_offsets differences must equal sp_log_widths * sp_tour_widths."""
        for i in range(basic_collection.n_trees):
            expected = int(basic_collection.sp_log_widths[i]) * int(
                basic_collection.sp_tour_widths[i]
            )
            actual = int(
                basic_collection.sp_offsets[i + 1] - basic_collection.sp_offsets[i]
            )
            assert actual == expected, (
                f"Tree {i}: sp size {actual} != {expected} "
                f"(LOG={basic_collection.sp_log_widths[i]}, "
                f"tour_len={basic_collection.sp_tour_widths[i]})"
            )

    def test_tour_len_is_2n_minus_1(self, basic_collection):
        """Euler tour length must be 2*n_nodes - 1 for each tree."""
        for i in range(basic_collection.n_trees):
            n = int(basic_collection.per_tree_n_nodes[i])
            tour_len = int(basic_collection.sp_tour_widths[i])
            assert tour_len == 2 * n - 1, (
                f"Tree {i}: tour_len={tour_len}, expected {2 * n - 1}"
            )


# ======================================================================== #
# 3. Array dtypes and shapes                                                #
# ======================================================================== #


class TestArrayDtypes:
    @pytest.mark.parametrize(
        "attr",
        [
            "all_parent",
            "all_left_child",
            "all_right_child",
            "all_depth",
            "all_first_occ",
            "all_euler_tour",
            "all_euler_depth",
            "all_sparse_table",
            "all_log2_table",
        ],
    )
    def test_int_arrays_dtype(self, basic_collection, attr):
        arr = getattr(basic_collection, attr)
        assert arr.dtype == np.int32, f"{attr}: expected int32, got {arr.dtype}"

    @pytest.mark.parametrize(
        "attr",
        [
            "all_distance",
            "all_support",
            "all_root_distance",
        ],
    )
    def test_float_arrays_dtype(self, basic_collection, attr):
        arr = getattr(basic_collection, attr)
        assert arr.dtype == np.float64, f"{attr}: expected float64, got {arr.dtype}"

    @pytest.mark.parametrize(
        "attr",
        [
            "node_offsets",
            "tour_offsets",
            "sp_offsets",
            "lg_offsets",
            "leaf_offsets",
        ],
    )
    def test_offset_arrays_dtype(self, basic_collection, attr):
        arr = getattr(basic_collection, attr)
        assert arr.dtype == np.int64, f"{attr}: expected int64, got {arr.dtype}"

    def test_global_to_local_dtype(self, basic_collection):
        assert basic_collection.global_to_local.dtype == np.int32

    def test_global_to_local_shape(self, basic_collection):
        assert basic_collection.global_to_local.shape == (
            basic_collection.n_trees,
            basic_collection.n_global_taxa,
        )

    def test_local_to_global_dtype(self, basic_collection):
        assert basic_collection.local_to_global.dtype == np.int32

    def test_local_to_global_length(self, basic_collection, basic_namespace):
        assert len(basic_collection.local_to_global) == int(
            basic_namespace["total_leaves"]
        )

    def test_taxa_present_dtype(self, basic_collection):
        assert basic_collection.taxa_present.dtype == bool

    def test_taxa_present_shape(self, basic_collection):
        assert basic_collection.taxa_present.shape == (
            basic_collection.n_trees,
            basic_collection.n_global_taxa,
        )

    def test_all_arrays_contiguous(self, basic_collection):
        """All flat packed arrays must be C-contiguous for GPU transfer."""
        for attr in [
            "all_parent",
            "all_distance",
            "all_root_distance",
            "all_euler_tour",
            "all_euler_depth",
            "all_sparse_table",
            "all_log2_table",
            "all_first_occ",
        ]:
            arr = getattr(basic_collection, attr)
            assert arr.flags["C_CONTIGUOUS"], f"{attr} is not C-contiguous"


# ======================================================================== #
# 4. Global namespace mapping                                               #
# ======================================================================== #


class TestGlobalNamespace:
    def test_global_to_local_values(self, basic_collection, basic_g2l):
        """Every (tree, gid) cell matches the CSV ground truth."""
        g2l = basic_collection.global_to_local
        for (ti, gid), expected_local in basic_g2l.items():
            assert int(g2l[ti, gid]) == expected_local, (
                f"global_to_local[{ti},{gid}] = {g2l[ti, gid]}, expected {expected_local}"
            )

    def test_global_to_local_absent_is_minus_one(self, basic_collection):
        """Absent taxa must be -1, not 0 (which is a valid leaf ID)."""
        g2l = basic_collection.global_to_local
        # E (gid=4) absent in tree 0 and tree 2
        assert g2l[0, 4] == -1  # E absent from tree 0
        assert g2l[2, 4] == -1  # E absent from tree 2
        assert g2l[1, 4] == 4  # E present in tree 1 as local ID 4

    def test_taxa_present_matches_g2l(self, basic_collection):
        g2l = basic_collection.global_to_local
        tp = basic_collection.taxa_present
        assert np.all(tp == (g2l >= 0))

    def test_local_to_global_round_trips(self, basic_collection):
        """For every leaf, local→global→local must be identity."""
        g2l = basic_collection.global_to_local
        l2g = basic_collection.local_to_global
        lo = basic_collection.leaf_offsets
        for ti in range(basic_collection.n_trees):
            n_leaves = int(basic_collection.per_tree_n_leaves[ti])
            base = int(lo[ti])
            for local_id in range(n_leaves):
                gid = int(l2g[base + local_id])
                assert gid >= 0, f"Tree {ti} leaf {local_id}: l2g returned -1"
                assert int(g2l[ti, gid]) == local_id, (
                    f"Round-trip fail: tree {ti} leaf {local_id} "
                    f"→ gid {gid} → local {g2l[ti, gid]}"
                )

    def test_global_id_lookup_by_name(self, basic_collection):
        """_resolve_global should work for all known taxa."""
        for gid, name in enumerate(basic_collection.global_names):
            assert basic_collection._resolve_global(name) == gid

    def test_global_id_lookup_by_int(self, basic_collection):
        for gid in range(basic_collection.n_global_taxa):
            assert basic_collection._resolve_global(gid) == gid

    def test_global_id_unknown_name_raises(self, basic_collection):
        with pytest.raises(KeyError):
            basic_collection._resolve_global("Zzz_unknown_taxon")

    def test_presence_matrix_correct_row_sums(self, basic_collection):
        """Each row sum = number of taxa in that tree."""
        expected = [4, 5, 4]  # n_leaves per tree
        for ti, exp in enumerate(expected):
            assert int(basic_collection.taxa_present[ti].sum()) == exp, (
                f"Tree {ti}: presence sum {basic_collection.taxa_present[ti].sum()}, expected {exp}"
            )

    def test_presence_matrix_correct_col_sums(self, basic_collection):
        """Column sums = number of trees each taxon appears in."""
        # A:3, B:2, C:3, D:2, E:1, X:1, Y:1
        expected = {"A": 3, "B": 2, "C": 3, "D": 2, "E": 1, "X": 1, "Y": 1}
        for name, exp_count in expected.items():
            gid = basic_collection._resolve_global(name)
            actual = int(basic_collection.taxa_present[:, gid].sum())
            assert actual == exp_count, (
                f"Taxon {name} (gid={gid}): present in {actual} trees, expected {exp_count}"
            )

    @pytest.mark.parametrize(
        "ti,name,expected_local",
        [
            (0, "A", 0),
            (0, "B", 1),
            (0, "C", 2),
            (0, "D", 3),
            (1, "A", 0),
            (1, "E", 4),
            (2, "A", 0),
            (2, "C", 1),
            (2, "X", 2),
            (2, "Y", 3),
        ],
    )
    def test_known_local_ids(self, basic_collection, ti, name, expected_local):
        gid = basic_collection._resolve_global(name)
        assert int(basic_collection.global_to_local[ti, gid]) == expected_local


# ======================================================================== #
# 5. CSR data integrity — per-tree slices match PhyloTree                  #
# ======================================================================== #


class TestCSRDataIntegrity:
    """
    Slice each tree out of the flat arrays using the offset vectors and
    compare with the corresponding PhyloTree instance arrays.
    """

    def _tree_slice(self, csr, attr, ti):
        """Return the slice of a node-indexed flat array for tree ti."""
        a = int(csr.node_offsets[ti])
        b = int(csr.node_offsets[ti + 1])
        return getattr(csr, attr)[a:b]

    def _tour_slice(self, csr, attr, ti):
        a = int(csr.tour_offsets[ti])
        b = int(csr.tour_offsets[ti + 1])
        return getattr(csr, attr)[a:b]

    @pytest.mark.parametrize(
        "attr",
        [
            "all_parent",
            "all_left_child",
            "all_right_child",
            "all_depth",
            "all_first_occ",
        ],
    )
    def test_int_node_arrays_match_tree(self, basic_collection, attr):
        tree_attr = attr.replace("all_", "")  # e.g. all_parent → parent
        special = {
            "all_first_occ": "first_occurrence",
        }
        tree_attr = special.get(attr, tree_attr)
        for ti, tree in enumerate(basic_collection._trees):
            expected = getattr(tree, tree_attr)
            actual = self._tree_slice(basic_collection, attr, ti)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{attr} tree {ti}")

    @pytest.mark.parametrize(
        "attr",
        [
            "all_distance",
            "all_support",
            "all_root_distance",
        ],
    )
    def test_float_node_arrays_match_tree(self, basic_collection, attr):
        tree_attr = attr.replace("all_", "")
        for ti, tree in enumerate(basic_collection._trees):
            expected = getattr(tree, tree_attr)
            actual = self._tree_slice(basic_collection, attr, ti)
            np.testing.assert_array_almost_equal(
                actual, expected, decimal=12, err_msg=f"{attr} tree {ti}"
            )

    @pytest.mark.parametrize("attr", ["all_euler_tour", "all_euler_depth"])
    def test_tour_arrays_match_tree(self, basic_collection, attr):
        tree_attr = attr.replace("all_", "")
        special = {"all_euler_tour": "euler_tour", "all_euler_depth": "euler_depth"}
        tree_attr = special.get(attr, tree_attr)
        for ti, tree in enumerate(basic_collection._trees):
            expected = getattr(tree, tree_attr)
            actual = self._tour_slice(basic_collection, attr, ti)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{attr} tree {ti}")

    def test_sparse_table_slice_matches_tree(self, basic_collection):
        for ti, tree in enumerate(basic_collection._trees):
            sp_base = int(basic_collection.sp_offsets[ti])
            LOG = int(basic_collection.sp_log_widths[ti])
            tlen = int(basic_collection.sp_tour_widths[ti])
            expected = tree.sparse_table.ravel()
            actual = basic_collection.all_sparse_table[sp_base : sp_base + LOG * tlen]
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"sparse_table tree {ti}"
            )

    def test_log2_table_slice_matches_tree(self, basic_collection):
        for ti, tree in enumerate(basic_collection._trees):
            a = int(basic_collection.lg_offsets[ti])
            b = int(basic_collection.lg_offsets[ti + 1])
            actual = basic_collection.all_log2_table[a:b]
            expected = tree.log2_table
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"log2_table tree {ti}"
            )

    def test_root_node_is_correct(self, basic_collection):
        for ti, tree in enumerate(basic_collection._trees):
            node_base = int(basic_collection.node_offsets[ti])
            root_local = int(basic_collection.per_tree_roots[ti])
            assert int(basic_collection.all_parent[node_base + root_local]) == -1, (
                f"Tree {ti} root node does not have parent=-1"
            )

    def test_root_distance_root_is_zero(self, basic_collection):
        for ti, tree in enumerate(basic_collection._trees):
            node_base = int(basic_collection.node_offsets[ti])
            root_local = int(basic_collection.per_tree_roots[ti])
            rd = float(basic_collection.all_root_distance[node_base + root_local])
            assert rd == 0.0, f"Tree {ti} root_distance at root = {rd}"


# ======================================================================== #
# 6. branch_distance                                                        #
# ======================================================================== #

EPS = 1e-10


class TestBranchDistance:
    def test_return_type(self, basic_collection):
        d = basic_collection.branch_distance("A", "B")
        assert isinstance(d, np.ndarray)
        assert d.dtype == np.float64
        assert d.shape == (basic_collection.n_trees,)

    def test_all_distances_against_csv(self, basic_collection, basic_distances):
        """Every (taxon_a, taxon_b, tree) entry in the CSV must match."""
        seen = set()
        for (ta, tb, ti), expected in basic_distances.items():
            key = (ta, tb) if ta < tb else (tb, ta)
            if key in seen:
                continue
            seen.add(key)
            d = basic_collection.branch_distance(ta, tb)
            actual = float(d[ti])
            if math.isnan(expected):
                assert math.isnan(actual), (
                    f"dist({ta},{tb}) tree {ti}: expected NaN, got {actual}"
                )
            else:
                assert abs(actual - expected) < EPS, (
                    f"dist({ta},{tb}) tree {ti}: got {actual}, expected {expected}"
                )

    def test_absent_taxon_is_nan(self, basic_collection):
        """E is absent from trees 0 and 2 → NaN there."""
        d = basic_collection.branch_distance("A", "E")
        assert math.isnan(d[0]), "A,E tree 0 should be NaN (E absent)"
        assert not math.isnan(d[1]), "A,E tree 1 should not be NaN"
        assert math.isnan(d[2]), "A,E tree 2 should be NaN (E absent)"

    def test_taxa_not_sharing_any_tree_all_nan(self, basic_collection):
        """B and X share no tree → all NaN."""
        d = basic_collection.branch_distance("B", "X")
        assert all(math.isnan(v) for v in d), f"B,X expected all NaN, got {d}"

    def test_same_taxon_distance_is_zero(self, basic_collection):
        """Distance from a taxon to itself is 0 where present, NaN where absent."""
        d = basic_collection.branch_distance("E", "E")
        assert d[1] == 0.0
        assert math.isnan(d[0])
        assert math.isnan(d[2])

    def test_symmetry(self, basic_collection):
        """branch_distance(a,b) == branch_distance(b,a) for all trees."""
        for ta, tb in [("A", "B"), ("A", "C"), ("C", "D"), ("D", "E"), ("X", "Y")]:
            d1 = basic_collection.branch_distance(ta, tb)
            d2 = basic_collection.branch_distance(tb, ta)
            for ti in range(basic_collection.n_trees):
                v1, v2 = float(d1[ti]), float(d2[ti])
                if math.isnan(v1):
                    assert math.isnan(v2), (
                        f"Symmetry NaN mismatch ({ta},{tb}) tree {ti}"
                    )
                else:
                    assert abs(v1 - v2) < EPS, (
                        f"Asymmetry: dist({ta},{tb})={v1} != dist({tb},{ta})={v2} tree {ti}"
                    )

    def test_by_global_id(self, basic_collection):
        """Passing integer global IDs should give the same result as names."""
        ga = basic_collection._resolve_global("A")
        gb = basic_collection._resolve_global("B")
        d_names = basic_collection.branch_distance("A", "B")
        d_ids = basic_collection.branch_distance(ga, gb)
        np.testing.assert_array_equal(d_names, d_ids)

    def test_unknown_name_raises(self, basic_collection):
        with pytest.raises(KeyError):
            basic_collection.branch_distance("A", "Zzz_unknown")

    def test_matches_individual_phylotree(self, basic_collection):
        """
        branch_distance results must match PhyloTree.branch_distance() called
        on each individual tree.
        """
        for ti, tree in enumerate(basic_collection._trees):
            if tree._name_index is None:
                tree._build_name_index()
            leaves = [tree.names[i] for i in range(tree.n_leaves)]
            for j in range(len(leaves)):
                for k in range(j + 1, len(leaves)):
                    ta, tb = leaves[j], leaves[k]
                    expected = tree.branch_distance(ta, tb)
                    actual = float(basic_collection.branch_distance(ta, tb)[ti])
                    assert abs(actual - expected) < EPS, (
                        f"Tree {ti} ({ta},{tb}): collection={actual}, "
                        f"PhyloTree={expected}"
                    )

    @pytest.mark.parametrize(
        "ta,tb,ti,expected",
        [
            ("A", "B", 0, 0.3),  # balanced tree: A→AB(0.1) + B→AB(0.2)
            ("C", "D", 0, 0.7),  # balanced tree: C→CD(0.3) + D→CD(0.4)
            ("A", "C", 0, 1.5),  # balanced tree: A→root(0.6) + C→root(0.9)
            ("D", "E", 1, 2.0),  # caterpillar: D and E are sisters
            ("A", "E", 1, 5.0),  # caterpillar: A→root(1) + E→root(4)
            ("X", "Y", 2, 0.8),  # two-clade tree: X→XY(0.4) + Y→XY(0.4)
            ("A", "X", 2, 1.5),  # two-clade tree: A→root(0.8) + X→root(0.7)
            ("A", "C", 2, 1.0),  # two-clade tree: A and C are sisters
        ],
    )
    def test_known_values(self, basic_collection, ta, tb, ti, expected):
        d = basic_collection.branch_distance(ta, tb)
        assert abs(float(d[ti]) - expected) < EPS, (
            f"dist({ta},{tb}) tree {ti}: got {d[ti]}, expected {expected}"
        )


# ======================================================================== #
# 7. _rmq_csr kernel unit tests                                             #
# ======================================================================== #


class TestRMQCSR:
    """
    Test the _rmq_csr static method directly using the flat arrays of the
    basic_collection fixture.  Expected LCA nodes are verified against
    PhyloTree.lca() from the individual trees.
    """

    def _csr_lca(self, csr, ti, ta, tb):
        """Run _rmq_csr for two taxa in tree ti; return local LCA node ID."""
        ga = csr._resolve_global(ta)
        gb = csr._resolve_global(tb)
        la = int(csr.global_to_local[ti, ga])
        lb = int(csr.global_to_local[ti, gb])
        nb = int(csr.node_offsets[ti])
        tb_ = int(csr.tour_offsets[ti])
        sb = int(csr.sp_offsets[ti])
        lb_ = int(csr.lg_offsets[ti])
        tw = int(csr.sp_tour_widths[ti])

        l = int(csr.all_first_occ[nb + la])
        r = int(csr.all_first_occ[nb + lb])
        if l > r:
            l, r = r, l

        return Forest._rmq_csr(
            l,
            r,
            sb,
            tw,
            csr.all_sparse_table,
            csr.all_euler_depth,
            csr.all_log2_table,
            lb_,
            tb_,
            csr.all_euler_tour,
        )

    @pytest.mark.parametrize(
        "ti,ta,tb,expected_local_lca",
        [
            (0, "A", "B", 4),  # balanced: LCA(A,B)=AB node, local ID 4
            (0, "C", "D", 5),  # balanced: LCA(C,D)=CD node, local ID 5
            (0, "A", "C", 6),  # balanced: LCA(A,C)=root, local ID 6
            (1, "D", "E", 5),  # caterpillar: LCA(D,E)=DE node, local ID 5
            (1, "A", "E", 8),  # caterpillar: LCA(A,E)=root, local ID 8
            (2, "X", "Y", 5),  # two-clade: LCA(X,Y)=XY node, local ID 5
            (2, "A", "X", 6),  # two-clade: LCA(A,X)=root, local ID 6
        ],
    )
    def test_rmq_csr_lca(self, basic_collection, ti, ta, tb, expected_local_lca):
        actual = self._csr_lca(basic_collection, ti, ta, tb)
        assert actual == expected_local_lca, (
            f"_rmq_csr tree {ti} LCA({ta},{tb}): got {actual}, expected {expected_local_lca}"
        )

    def test_rmq_csr_matches_phylotree_lca(self, basic_collection):
        """_rmq_csr must match PhyloTree.lca() for all within-tree leaf pairs."""
        for ti, tree in enumerate(basic_collection._trees):
            if tree._name_index is None:
                tree._build_name_index()
            leaves = [tree.names[i] for i in range(tree.n_leaves)]
            for j in range(len(leaves)):
                for k in range(j + 1, len(leaves)):
                    ta, tb = leaves[j], leaves[k]
                    expected = tree.lca(ta, tb)
                    actual = self._csr_lca(basic_collection, ti, ta, tb)
                    assert actual == expected, (
                        f"Tree {ti} LCA({ta},{tb}): _rmq_csr={actual}, "
                        f"PhyloTree.lca={expected}"
                    )


# ======================================================================== #
# 8. quartet_topology                                                        #
# ======================================================================== #


class TestQuartetTopology:
    """
    Tests for Forest.quartet_topology().

    The method accepts an iterable of 4-tuples and always returns arrays
    with a leading quartet axis, regardless of how many quartets are queried.
    Single-quartet use:  c.quartet_topology([['A','B','C','D']])
    Bulk use:            c.quartet_topology([q1, q2, q3, ...])

    Topology encoding (global-ID-sorted order n0 < n1 < n2 < n3):
        index 0: (n0, n1) | (n2, n3)
        index 1: (n0, n2) | (n1, n3)
        index 2: (n0, n3) | (n1, n2)
    """

    # ── fixtures and helpers ─────────────────────────────────────────────── #

    @pytest.fixture
    def mixed_collection(self):
        """5-tree collection spanning all three topologies plus a caterpillar."""
        return Forest(
            [
                "((A:1,B:1):1,(C:1,D:1):1);",  # topo 0 for ABCD
                "((A:1,C:1):1,(B:1,D:1):1);",  # topo 1 for ABCD
                "((A:1,D:1):1,(B:1,C:1):1);",  # topo 2 for ABCD
                "((A:0.3,B:0.7):0.5,(C:0.2,D:0.9):0.4);",  # topo 0, unequal
                "(A:1,(B:1,(C:1,(D:1,E:1):1):1):1);",  # caterpillar ABCDE
            ]
        )

    BULK_QUARTETS = [("A", "B", "C", "D"), ("A", "B", "C", "E"), ("A", "B", "D", "E")]

    @staticmethod
    def _expected_steiner(nwk, a="A", b="B", c="C", d="D"):
        """Steiner length from a verified single-tree PhyloTree instance."""
        t = Tree(nwk)
        t._build_name_index()
        _, s = t.quartet_topology(a, b, c, d, return_steiner=True)
        return s

    # ── return types and shapes ─────────────────────────────────────────── #

    def test_single_quartet_counts_shape_and_dtype(self, basic_collection):
        counts = basic_collection.quartet_topology([("A", "B", "C", "D")])
        assert isinstance(counts, np.ndarray)
        assert counts.shape == (1, 3)
        assert counts.dtype == np.int32

    def test_bulk_counts_shape_and_dtype(self, mixed_collection):
        counts = mixed_collection.quartet_topology(self.BULK_QUARTETS)
        assert isinstance(counts, np.ndarray)
        assert counts.shape == (3, 3)
        assert counts.dtype == np.int32

    def test_steiner_returns_tuple(self, mixed_collection):
        result = mixed_collection.quartet_topology(self.BULK_QUARTETS, steiner=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_steiner_counts_shape_and_dtype(self, mixed_collection):
        counts, _ = mixed_collection.quartet_topology(self.BULK_QUARTETS, steiner=True)
        assert counts.shape == (3, 3)
        assert counts.dtype == np.int32

    def test_steiner_distances_shape_and_dtype(self, mixed_collection):
        _, dists = mixed_collection.quartet_topology(self.BULK_QUARTETS, steiner=True)
        assert dists.shape == (3, mixed_collection.n_trees, 3)
        assert dists.dtype == np.float64

    def test_single_quartet_steiner_shapes(self, basic_collection):
        counts, dists = basic_collection.quartet_topology(
            [("A", "B", "C", "D")], steiner=True
        )
        assert counts.shape == (1, 3)
        assert dists.shape == (1, basic_collection.n_trees, 3)

    # ── steiner=False correctness ────────────────────────────────────────── #

    def test_counts_sum_equals_trees_with_all_taxa(self, basic_collection):
        """counts[qi].sum() == number of trees where all four taxa are present."""
        counts = basic_collection.quartet_topology([("A", "B", "C", "D")])
        # basic_collection: A,B,C,D in trees 0 and 1; tree 2 lacks D
        assert int(counts[0].sum()) == 2

    def test_known_topology_distribution(self):
        c = Forest(
            [
                "((A:1,B:1):1,(C:1,D:1):1);",  # topo 0
                "((A:1,C:1):1,(B:1,D:1):1);",  # topo 1
                "((A:1,D:1):1,(B:1,C:1):1);",  # topo 2
                "((A:1,B:1):1,(C:1,D:1):1);",  # topo 0
                "((A:1,C:1):1,(B:1,D:1):1);",  # topo 1
            ]
        )
        counts = c.quartet_topology([("A", "B", "C", "D")])
        assert counts[0, 0] == 2
        assert counts[0, 1] == 2
        assert counts[0, 2] == 1

    def test_absent_taxon_excluded(self):
        c = Forest(
            [
                "((A:1,B:1):1,(C:1,D:1):1);",  # has A,B,C,D
                "((A:1,X:1):1,(B:1,C:1):1);",  # has A,B,C,X — no D
                "((A:1,B:1):1,(C:1,D:1):1);",  # has A,B,C,D
            ]
        )
        counts = c.quartet_topology([("A", "B", "C", "D")])
        assert int(counts[0].sum()) == 2

    def test_unknown_taxon_raises(self, basic_collection):
        with pytest.raises(KeyError):
            basic_collection.quartet_topology([("A", "B", "C", "Unknown")])

    def test_counts_100_trees(self):
        newicks = []
        for i in range(100):
            if i % 3 == 0:
                newicks.append("((A:1,B:1):1,(C:1,D:1):1);")
            elif i % 3 == 1:
                newicks.append("((A:1,C:1):1,(B:1,D:1):1);")
            else:
                newicks.append("((A:1,D:1):1,(B:1,C:1):1);")
        c = Forest(newicks)
        counts = c.quartet_topology([("A", "B", "C", "D")])
        assert counts[0, 0] == 34
        assert counts[0, 1] == 33
        assert counts[0, 2] == 33
        assert int(counts[0].sum()) == 100

    def test_counts_match_individual_phylotree_calls(self, basic_collection):
        """Counts must agree with per-tree PhyloTree.quartet_topology()."""
        quartet = ("A", "B", "C", "D")
        counts = basic_collection.quartet_topology([quartet])
        n0, n1, n2, n3 = sorted(quartet)
        topo_map = {
            frozenset({frozenset({n0, n1}), frozenset({n2, n3})}): 0,
            frozenset({frozenset({n0, n2}), frozenset({n1, n3})}): 1,
            frozenset({frozenset({n0, n3}), frozenset({n1, n2})}): 2,
        }
        individual = np.zeros(3, dtype=np.int32)
        for tree in basic_collection._trees:
            if tree._name_index is None:
                tree._build_name_index()
            if not all(x in tree._name_index for x in quartet):
                continue
            individual[topo_map[tree.quartet_topology(*quartet)]] += 1
        np.testing.assert_array_equal(counts[0], individual)

    def test_permutation_invariance(self, basic_collection):
        ref = basic_collection.quartet_topology([("A", "B", "C", "D")])
        for perm in itertools.permutations(("A", "B", "C", "D")):
            r = basic_collection.quartet_topology([perm])
            np.testing.assert_array_equal(
                r, ref, err_msg=f"counts differ for perm {perm}"
            )

    def test_global_id_input(self, basic_collection):
        ga = basic_collection._resolve_global("A")
        gb = basic_collection._resolve_global("B")
        gc = basic_collection._resolve_global("C")
        gd = basic_collection._resolve_global("D")
        np.testing.assert_array_equal(
            basic_collection.quartet_topology([(ga, gb, gc, gd)]),
            basic_collection.quartet_topology([("A", "B", "C", "D")]),
        )

    # ── bulk: multiple quartets ───────────────────────────────────────────── #

    def test_bulk_rows_match_individual_calls(self, mixed_collection):
        """Each row of a bulk call matches the equivalent single-quartet call."""
        counts_bulk = mixed_collection.quartet_topology(self.BULK_QUARTETS)
        for qi, q in enumerate(self.BULK_QUARTETS):
            np.testing.assert_array_equal(
                counts_bulk[qi],
                mixed_collection.quartet_topology([q])[0],
                err_msg=f"counts mismatch for quartet {q}",
            )

    def test_large_bulk_all_5_leaf_quartets(self):
        """All C(5,4)=5 quartets: bulk call matches individual calls exactly."""
        newicks = [
            "((A:1,B:1):1,(C:1,D:1):1);",
            "((A:1,C:1):1,(B:1,D:1):1);",
            "(A:0.5,(B:0.5,(C:0.5,(D:0.5,E:0.5):0.5):0.5):0.5);",
        ]
        c = Forest(newicks)
        all_quartets = list(itertools.combinations(("A", "B", "C", "D", "E"), 4))
        counts_bulk, dists_bulk = c.quartet_topology(all_quartets, steiner=True)
        for qi, q in enumerate(all_quartets):
            c_single, d_single = c.quartet_topology([q], steiner=True)
            np.testing.assert_array_equal(
                counts_bulk[qi], c_single[0], err_msg=f"counts mismatch for {q}"
            )
            np.testing.assert_array_almost_equal(
                dists_bulk[qi],
                d_single[0],
                decimal=10,
                err_msg=f"Steiner mismatch for {q}",
            )

    # ── steiner=True correctness ─────────────────────────────────────────── #

    def test_steiner_counts_agree_with_counts_only(self, mixed_collection):
        counts_only = mixed_collection.quartet_topology(self.BULK_QUARTETS)
        counts_s, _ = mixed_collection.quartet_topology(
            self.BULK_QUARTETS, steiner=True
        )
        np.testing.assert_array_equal(counts_s, counts_only)

    def test_steiner_at_most_one_nonzero_per_qi_ti_row(self, mixed_collection):
        """Each (qi, ti) row has at most one non-zero Steiner entry."""
        _, dists = mixed_collection.quartet_topology(self.BULK_QUARTETS, steiner=True)
        for qi in range(len(self.BULK_QUARTETS)):
            for ti in range(mixed_collection.n_trees):
                nz = int((dists[qi, ti] > 0).sum())
                assert nz <= 1, f"qi={qi} ti={ti}: expected ≤1 non-zero, got {nz}"

    def test_steiner_nonzero_in_winning_topology_column(self):
        """Non-zero Steiner sits in the column matching the winning topology."""
        c = Forest(
            [
                "((A:1,B:1):1,(C:1,D:1):1);",  # topo 0
                "((A:1,C:1):1,(B:1,D:1):1);",  # topo 1
                "((A:1,D:1):1,(B:1,C:1):1);",  # topo 2
            ]
        )
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        # dists shape (1, 3, 3): qi=0, ti=0..2
        assert dists[0, 0, 0] > 0 and dists[0, 0, 1] == 0 and dists[0, 0, 2] == 0
        assert dists[0, 1, 0] == 0 and dists[0, 1, 1] > 0 and dists[0, 1, 2] == 0
        assert dists[0, 2, 0] == 0 and dists[0, 2, 1] == 0 and dists[0, 2, 2] > 0

    def test_steiner_all_values_non_negative(self, mixed_collection):
        _, dists = mixed_collection.quartet_topology(self.BULK_QUARTETS, steiner=True)
        assert np.all(dists >= 0.0)

    def test_steiner_column_sum_over_count_gives_mean(self):
        """dists[qi, :, k].sum() / counts[qi, k] == mean Steiner for topo k."""
        c = Forest(
            [
                "((A:1,B:1):1,(C:1,D:1):1);",  # topo 0, S=6
                "((A:2,B:2):2,(C:2,D:2):2);",  # topo 0, S=12
                "((A:1,C:1):1,(B:1,D:1):1);",  # topo 1, S=6
            ]
        )
        counts, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        np.testing.assert_allclose(dists[0, :, 0].sum() / counts[0, 0], 9.0, rtol=1e-10)
        np.testing.assert_allclose(dists[0, :, 1].sum() / counts[0, 1], 6.0, rtol=1e-10)

    def test_steiner_values_match_phylotree_balanced_equal(self):
        nwk = "((A:1,B:1):1,(C:1,D:1):1);"
        c = Forest([nwk])
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        expected = self._expected_steiner(nwk)
        got = float(dists[0, 0, np.argmax(dists[0, 0])])
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_steiner_values_match_phylotree_balanced_unequal(self):
        nwk = "((A:0.3,B:0.7):0.5,(C:0.2,D:0.9):0.4);"
        c = Forest([nwk])
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        expected = self._expected_steiner(nwk)
        got = float(dists[0, 0, np.argmax(dists[0, 0])])
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_steiner_values_match_phylotree_topo1(self):
        nwk = "((A:1.1,C:0.6):0.8,(B:0.4,D:1.3):0.2);"
        c = Forest([nwk])
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        expected = self._expected_steiner(nwk)
        got = float(dists[0, 0, np.argmax(dists[0, 0])])
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_steiner_values_match_phylotree_topo2(self):
        nwk = "((A:1,D:1):1,(B:1,C:1):1);"
        c = Forest([nwk])
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        expected = self._expected_steiner(nwk)
        got = float(dists[0, 0, np.argmax(dists[0, 0])])
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_steiner_values_match_phylotree_caterpillar(self):
        nwk = "(A:1,(B:1,(C:1,(D:1,E:1):1):1):1);"
        c = Forest([nwk])
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        expected = self._expected_steiner(nwk)
        got = float(dists[0, 0, np.argmax(dists[0, 0])])
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_steiner_values_match_phylotree_deep_caterpillar(self):
        nwk = "(A:0.1,(B:0.2,(C:0.3,(D:0.4,E:0.5):0.6):0.7):0.8);"
        c = Forest([nwk])
        _, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        expected = self._expected_steiner(nwk)
        got = float(dists[0, 0, np.argmax(dists[0, 0])])
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_steiner_multi_tree_all_match_phylotree(self, mixed_collection):
        """All trees in a multi-tree collection match per-tree PhyloTree."""
        _, dists = mixed_collection.quartet_topology(
            [("A", "B", "C", "D")], steiner=True
        )
        for ti, tree in enumerate(mixed_collection._trees):
            if tree._name_index is None:
                tree._build_name_index()
            if not all(x in tree._name_index for x in "ABCD"):
                continue
            _, expected_S = tree.quartet_topology(
                "A", "B", "C", "D", return_steiner=True
            )
            got_S = float(dists[0, ti, np.argmax(dists[0, ti])])
            np.testing.assert_allclose(
                got_S, expected_S, rtol=1e-10, err_msg=f"Steiner mismatch for ti={ti}"
            )

    def test_steiner_permutation_invariance(self, mixed_collection):
        ref_c, ref_d = mixed_collection.quartet_topology(
            [("A", "B", "C", "D")], steiner=True
        )
        for perm in itertools.permutations(("A", "B", "C", "D")):
            pc, pd = mixed_collection.quartet_topology([perm], steiner=True)
            np.testing.assert_array_equal(
                pc, ref_c, err_msg=f"counts differ for perm {perm}"
            )
            np.testing.assert_array_almost_equal(
                pd, ref_d, err_msg=f"Steiner differs for perm {perm}"
            )

    # ── absent taxon handling ─────────────────────────────────────────────── #

    def test_absent_taxon_steiner_rows_are_zero(self, mixed_collection):
        """Trees lacking a taxon contribute 0.0 to the Steiner array."""
        _, dists = mixed_collection.quartet_topology(
            [("A", "B", "C", "E")], steiner=True
        )
        # E absent from trees 0-3; tree 4 (caterpillar) has it
        for ti in range(4):
            assert np.all(dists[0, ti] == 0.0), f"ti={ti} should be all zero"
        assert dists[0, 4].sum() > 0

    def test_zero_count_topology_steiner_column_all_zero(self):
        """For a topology with no support, its entire Steiner column is 0.0."""
        c = Forest(["((A:1,B:1):1,(C:1,D:1):1);"] * 5)
        counts, dists = c.quartet_topology([("A", "B", "C", "D")], steiner=True)
        assert counts[0, 1] == 0 and counts[0, 2] == 0
        assert np.all(dists[0, :, 1] == 0.0)
        assert np.all(dists[0, :, 2] == 0.0)
        assert np.all(dists[0, :, 0] > 0.0)

    def test_never_cooccurring_quartet_returns_zero_counts(self):
        """All taxa in namespace but never all four in the same tree → zeros."""
        c = Forest(
            [
                "((A:1,B:1):1,C:1);",  # A,B,C — no D
                "((A:1,B:1):1,D:1);",  # A,B,D — no C
            ]
        )
        counts = c.quartet_topology([("A", "B", "C", "D")])
        assert np.all(counts == 0)

    # ── input flexibility ────────────────────────────────────────────────── #

    def test_generator_input(self, mixed_collection):
        def gen():
            yield ("A", "B", "C", "D")
            yield ("A", "B", "C", "E")

        counts = mixed_collection.quartet_topology(gen())
        assert counts.shape == (2, 3)

    def test_empty_input_counts_shape(self, mixed_collection):
        counts = mixed_collection.quartet_topology([])
        assert counts.shape == (0, 3)
        assert counts.dtype == np.int32

    def test_empty_input_steiner_shapes(self, mixed_collection):
        counts, dists = mixed_collection.quartet_topology([], steiner=True)
        assert counts.shape == (0, 3)
        assert dists.shape == (0, mixed_collection.n_trees, 3)

    def test_unknown_taxon_in_second_quartet_raises(self, mixed_collection):
        """Validation is eager — any bad name fails the whole call."""
        with pytest.raises(KeyError):
            mixed_collection.quartet_topology(
                [
                    ("A", "B", "C", "D"),
                    ("A", "B", "C", "UNKNOWN"),
                ]
            )
