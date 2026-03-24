"""
tests/test_forest_groups.py
===========================================
Tests for tree group labels functionality in PhyloTreeCollection.
"""

import pytest
import numpy as np
from pathlib import Path
from quarimo._forest import Forest, _jaccard
from quarimo._quartets import Quartets
from quarimo._results import QuartetTopologyResult

TREES_DIR = Path(__file__).parent / "trees"

# Single-tree files (one NEWICK per file)
ASYMMETRIC = TREES_DIR / "asymmetric_4leaf.tree"   # ((A:1,(B:1,C:1):1):1,D:1);
BALANCED   = TREES_DIR / "balanced_4leaf.tree"     # ((A:0.1,B:0.2)..., (C:0.3,D:0.4)...);

# Multi-tree file (3 NEWICK on separate lines, taxa A-E and X,Y)
COLLECTION = TREES_DIR / "basic_collection.trees"  # 3 trees


class TestForestInputNormalization:
    """Tests covering all supported Forest input forms and their combinations."""

    # ── single multiline string ─────────────────────────────────────────── #

    def test_multiline_string_creates_single_group(self):
        """A newline-separated block of NEWICK trees is treated as one group."""
        block = (
            "((A:1,B:1):1,(C:1,D:1):1);\n"
            "((A:1,C:1):1,(B:1,D:1):1);\n"
            "((A:1,D:1):1,(B:1,C:1):1);\n"
        )
        f = Forest(block)
        assert f.n_trees == 3
        assert f.n_groups == 1

    def test_multiline_string_blank_lines_ignored(self):
        """Blank lines and comment lines in a block are silently skipped."""
        block = (
            "\n"
            "((A:1,B:1):1,(C:1,D:1):1);\n"
            "\n"
            "((A:1,C:1):1,(B:1,D:1):1);\n"
            "\n"
        )
        f = Forest(block)
        assert f.n_trees == 2

    # ── single NEWICK string ────────────────────────────────────────────── #

    def test_single_newick_string(self):
        """A string starting with '(' is treated as a single tree."""
        f = Forest("((A:1,B:1):1,(C:1,D:1):1);")
        assert f.n_trees == 1
        assert f.n_groups == 1

    # ── str / Path file paths ───────────────────────────────────────────── #

    def test_path_object_to_single_tree_file(self):
        """A Path to a single-tree file produces a 1-tree forest."""
        f = Forest(ASYMMETRIC)
        assert f.n_trees == 1
        assert f.n_groups == 1

    def test_str_path_to_single_tree_file(self):
        """A string that resolves to a file is read as a NEWICK file."""
        f = Forest(str(ASYMMETRIC))
        assert f.n_trees == 1

    def test_path_to_multitree_file(self):
        """A Path to a multi-tree file loads all trees in the file."""
        f = Forest(COLLECTION)
        assert f.n_trees == 3
        assert f.n_groups == 1

    # ── list inputs ─────────────────────────────────────────────────────── #

    def test_list_of_path_objects_single_tree_files(self):
        """A list of Paths to single-tree files loads one tree per file."""
        f = Forest([ASYMMETRIC, BALANCED])
        assert f.n_trees == 2
        assert f.n_groups == 1

    def test_list_of_str_paths_single_tree_files(self):
        """A list of string paths to single-tree files."""
        f = Forest([str(ASYMMETRIC), str(BALANCED)])
        assert f.n_trees == 2

    def test_list_of_paths_to_multitree_files(self):
        """Each Path in the list may point to a multi-tree file; trees are concatenated."""
        f = Forest([COLLECTION, COLLECTION])
        assert f.n_trees == 6  # 3 trees × 2 files (6 loaded, duplicates deduped internally)
        assert f.n_groups == 1

    def test_list_mixing_strings_and_paths(self):
        """A list may mix inline NEWICK strings and file Paths."""
        inline = "((A:1,B:1):1,(C:1,D:1):1);"
        f = Forest([inline, ASYMMETRIC])
        assert f.n_trees == 2

    # ── dict with multiline string values ───────────────────────────────── #

    def test_dict_multiline_string_value(self):
        """Dict values may be multiline strings instead of lists of strings."""
        block = "((A:1,B:1):1,(C:1,D:1):1);\n((A:1,C:1):1,(B:1,D:1):1);\n"
        f = Forest({"grp": block})
        assert f.n_trees == 2
        assert f.n_groups == 1
        assert f.unique_groups == ["grp"]

    def test_dict_multiline_string_value_multiple_groups(self):
        """Multiple groups, each given as a multiline string."""
        block_a = "((A:1,B:1):1,(C:1,D:1):1);\n((A:1,C:1):1,(B:1,D:1):1);\n"
        block_b = "((A:1,D:1):1,(B:1,C:1):1);\n"
        f = Forest({"A": block_a, "B": block_b})
        assert f.n_groups == 2
        assert f.n_trees == 3
        assert f.unique_groups == ["A", "B"]

    # ── dict with Path values ────────────────────────────────────────────── #

    def test_dict_single_path_value(self):
        """A dict value may be a single Path to a single-tree file."""
        f = Forest({"grp": ASYMMETRIC})
        assert f.n_trees == 1
        assert f.unique_groups == ["grp"]

    def test_dict_path_to_multitree_file(self):
        """A dict value may be a Path to a multi-tree file."""
        f = Forest({"grp": COLLECTION})
        assert f.n_trees == 3
        assert f.unique_groups == ["grp"]

    def test_dict_multiple_groups_path_values(self):
        """Multiple groups, each given as a Path to a (possibly multi-tree) file."""
        f = Forest({"short": ASYMMETRIC, "long": COLLECTION})
        assert f.n_groups == 2
        assert f.n_trees == 4  # 1 + 3
        assert f.unique_groups == ["long", "short"]  # sorted

    # ── dict with list-of-Paths values ──────────────────────────────────── #

    def test_dict_list_of_paths_single_tree_files(self):
        """A dict value may be a list of Paths to individual single-tree files."""
        f = Forest({"grp": [ASYMMETRIC, BALANCED]})
        assert f.n_trees == 2
        assert f.unique_groups == ["grp"]

    def test_dict_list_of_paths_multitree_files(self):
        """A dict value may be a list of Paths each containing multiple trees."""
        f = Forest({"grp": [COLLECTION, COLLECTION]})
        assert f.n_trees == 6  # 6 loaded, duplicates deduped internally

    def test_dict_multiple_groups_list_of_paths(self):
        """Multiple groups, each given as a list of Paths."""
        f = Forest({
            "A": [ASYMMETRIC, BALANCED],
            "B": [COLLECTION],
        })
        assert f.n_groups == 2
        assert f.n_trees == 5  # 2 + 3
        assert f.unique_groups == ["A", "B"]

    def test_dict_mixed_value_types(self):
        """Dict values may mix inline strings, Paths, and multiline blocks."""
        inline = "((A:1,B:1):1,(C:1,D:1):1);"
        f = Forest({
            "inline": inline,
            "file":   ASYMMETRIC,
            "multi":  [ASYMMETRIC, BALANCED],
        })
        assert f.n_groups == 3
        assert f.n_trees == 4

    # ── error cases ─────────────────────────────────────────────────────── #

    def test_multitree_string_not_accepted_as_single_tree(self):
        """A string containing multiple semicolons is rejected by validate_newick."""
        from quarimo._utils import validate_newick
        with pytest.raises(ValueError, match="semicolons"):
            validate_newick("((A:1,B:1):1,(C:1,D:1):1);((A:1,C:1):1,(B:1,D:1):1);")

    def test_nonexistent_file_raises(self):
        """A Path that does not exist raises ValueError."""
        with pytest.raises(ValueError, match="cannot read file"):
            Forest(Path("/nonexistent/path/to/trees.nwk"))

    def test_dict_empty_list_value_raises(self):
        """A dict group with an empty list raises ValueError."""
        with pytest.raises(ValueError, match="Group 'B' is empty"):
            Forest({"A": ["((A:1,B:1):1,(C:1,D:1):1);"], "B": []})

    def test_unbalanced_parens_raises(self):
        """A NEWICK string with unbalanced parentheses raises ValueError."""
        from quarimo._utils import validate_newick
        with pytest.raises(ValueError, match="parentheses"):
            validate_newick("((A:1,B:1):1,(C:1,D:1):1;")  # missing closing paren


class TestGroupLabels:
    """Tests for group label initialization and metadata."""

    def test_dict_input_multiple_groups(self):
        """Test explicit group labels via dict input."""
        groups = {
            "species_A": [
                "((A:1,B:1):1,(C:1,D:1):1);",
                "((A:1,C:1):1,(B:1,D:1):1);",
            ],
            "species_B": [
                "((A:1,D:1):1,(B:1,C:1):1);",
            ],
            "species_C": [
                "((E:1,F:1):1,(G:1,H:1):1);",
            ],
        }

        c = Forest(groups)

        # Check basic attributes
        assert c.n_groups == 3
        assert c.unique_groups == ["species_A", "species_B", "species_C"]
        assert len(c.group_labels) == 4  # Total trees

        # Check group labels are correct
        assert c.group_labels == ["species_A", "species_A", "species_B", "species_C"]

        # Check group_to_tree_indices
        np.testing.assert_array_equal(c.group_to_tree_indices["species_A"], [0, 1])
        np.testing.assert_array_equal(c.group_to_tree_indices["species_B"], [2])
        np.testing.assert_array_equal(c.group_to_tree_indices["species_C"], [3])

        # Check tree_to_group_idx
        expected_tree_to_group = np.array([0, 0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(c.tree_to_group_idx, expected_tree_to_group)

        # Check group_offsets (CSR)
        expected_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
        np.testing.assert_array_equal(c.group_offsets, expected_offsets)

    def test_list_input_auto_labeled(self):
        """Test auto-labeling for list input."""
        trees = [
            "((A:1,B:1):1,(C:1,D:1):1);",
            "((A:1,C:1):1,(B:1,D:1):1);",
        ]

        c = Forest(trees)

        # Check basic attributes
        assert c.n_groups == 1
        assert len(c.unique_groups) == 1

        # Check label format (10 hex characters)
        label = c.unique_groups[0]
        assert len(label) == 10
        assert all(ch in "0123456789abcdef" for ch in label)

        # Check all trees have same label
        assert c.group_labels == [label, label]

    def test_auto_label_determinism(self):
        """Test that same input produces same label."""
        trees = [
            "((A:1,B:1):1,(C:1,D:1):1);",
            "((A:1,C:1):1,(B:1,D:1):1);",
        ]

        c1 = Forest(trees)
        c2 = Forest(trees)

        # Same input should produce same label
        assert c1.unique_groups[0] == c2.unique_groups[0]

    def test_auto_label_different_for_different_input(self):
        """Test that different inputs produce different labels."""
        trees1 = ["((A:1,B:1):1,(C:1,D:1):1);"]
        trees2 = ["((A:1,C:1):1,(B:1,D:1):1);"]

        c1 = Forest(trees1)
        c2 = Forest(trees2)

        # Different input should produce different labels
        assert c1.unique_groups[0] != c2.unique_groups[0]

    def test_empty_group_raises_error(self):
        """Test that empty groups raise ValueError."""
        groups = {
            "A": ["((A:1,B:1):1,(C:1,D:1):1);"],
            "B": [],  # Empty!
        }

        with pytest.raises(ValueError, match="Group 'B' is empty"):
            Forest(groups)

    def test_single_group_dict(self):
        """Test dict with single group."""
        groups = {
            "my_group": [
                "((A:1,B:1):1,(C:1,D:1):1);",
                "((A:1,C:1):1,(B:1,D:1):1);",
            ]
        }

        c = Forest(groups)

        assert c.n_groups == 1
        assert c.unique_groups == ["my_group"]

    def test_invalid_input_type(self):
        """Test that unsupported input types raise TypeError, and unrecognised
        strings raise ValueError (strings are a valid input form — they are
        treated as a NEWICK tree, a multiline block, or a file path)."""
        # Bare integers are not a supported input type
        with pytest.raises(TypeError, match="must be dict, list, tuple, str, or Path"):
            Forest(12345)

        # A string that is not a valid NEWICK tree and not an existing file
        # raises ValueError, not TypeError
        with pytest.raises(ValueError):
            Forest("not a valid newick string")


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculations."""

    def test_jaccard_helper_function(self):
        """Test the _jaccard helper function."""
        # Perfect overlap
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

        # No overlap
        assert _jaccard({1, 2}, {3, 4}) == 0.0

        # Partial overlap
        assert _jaccard({1, 2, 3}, {2, 3, 4}) == 0.5  # 2 shared / 4 total

        # Empty sets
        assert _jaccard(set(), set()) == 0.0

        # One empty
        assert _jaccard({1, 2}, set()) == 0.0

    def test_jaccard_within_group_perfect_overlap(self):
        """Test within-group Jaccard when all trees have same taxa."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,c:1):1,(b:1,d:1):1);",
                "((a:1,d:1):1,(b:1,c:1):1);",
            ]
        }

        c = Forest(groups)

        # All trees have same taxa {a,b,c,d} → Jaccard = 1.0
        # (This would be verified via log capture in practice)
        pass

    def test_jaccard_within_group_partial_overlap(self):
        """Test within-group Jaccard with varying taxa."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",  # {a,b,c,d}
                "((a:1,b:1):1,(c:1,e:1):1);",  # {a,b,c,e}
            ]
        }

        c = Forest(groups)

        # Manual calculation:
        # Tree 0: {a,b,c,d}
        # Tree 1: {a,b,c,e}
        # Intersection: {a,b,c} = 3
        # Union: {a,b,c,d,e} = 5
        # Jaccard = 3/5 = 0.6
        # (Would be verified via log capture)
        pass

    def test_jaccard_between_groups_disjoint(self):
        """Test between-group Jaccard with no shared taxa."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((e:1,f:1):1,(g:1,h:1):1);"],
        }

        c = Forest(groups)

        # Group A: {a,b,c,d}
        # Group B: {e,f,g,h}
        # Intersection: {} = 0
        # Union: {a,b,c,d,e,f,g,h} = 8
        # Jaccard = 0/8 = 0.0
        pass

    def test_jaccard_between_groups_partial_overlap(self):
        """Test between-group Jaccard with some shared taxa."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((c:1,d:1):1,(e:1,f:1):1);"],
        }

        c = Forest(groups)

        # Group A: {a,b,c,d}
        # Group B: {c,d,e,f}
        # Intersection: {c,d} = 2
        # Union: {a,b,c,d,e,f} = 6
        # Jaccard = 2/6 = 0.333
        pass


class TestGroupedQuartetTopology:
    """Tests for per-group quartet topology output."""

    def test_counts_shape(self):
        """counts.shape == (n_quartets, n_groups, 4) for a 2-group forest."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,c:1):1,(b:1,d:1):1);",
            ],
            "B": [
                "((a:1,d:1):1,(b:1,c:1):1);",
            ],
        }
        c = Forest(groups)
        quartets = [("a", "b", "c", "d")]
        result = c.quartet_topology(Quartets.from_list(c, quartets))
        assert result.counts.shape == (1, 2, 4)
        assert result.counts.dtype == np.int32

    def test_per_group_counts_correct(self):
        """Each group accumulates its own topology votes."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",  # topo 0
                "((a:1,c:1):1,(b:1,d:1):1);",  # topo 1
            ],
            "B": [
                "((a:1,d:1):1,(b:1,c:1):1);",  # topo 2
            ],
        }
        c = Forest(groups)
        # unique_groups is sorted: ["A", "B"] → group_A_idx=0, group_B_idx=1
        result = c.quartet_topology(Quartets.from_list(c, [("a", "b", "c", "d")]))
        assert result.counts[0, 0, 0] == 1  # Group A: 1 vote for topo 0
        assert result.counts[0, 0, 1] == 1  # Group A: 1 vote for topo 1
        assert result.counts[0, 0, 2] == 0  # Group A: 0 votes for topo 2
        assert result.counts[0, 1, 0] == 0  # Group B: 0 votes for topo 0
        assert result.counts[0, 1, 1] == 0  # Group B: 0 votes for topo 1
        assert result.counts[0, 1, 2] == 1  # Group B: 1 vote for topo 2

    def test_group_counts_sum_to_total_tree_count(self):
        """Sum of counts across all groups == total trees with all 4 taxa present."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,c:1):1,(b:1,d:1):1);",
            ],
            "B": [
                "((a:1,d:1):1,(b:1,c:1):1);",
            ],
        }
        c = Forest(groups)
        result = c.quartet_topology(Quartets.from_list(c, [("a", "b", "c", "d")]))
        assert int(result.counts[0].sum()) == 3  # 3 trees total, all have a,b,c,d

    def test_steiner_shape(self):
        """steiner.shape == (n_quartets, n_groups, 4)."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        c = Forest(groups)
        result = c.quartet_topology(
            Quartets.from_list(c, [("a", "b", "c", "d")]), steiner=True
        )
        assert result.counts.shape == (1, 2, 4)
        assert result.steiner.shape == (1, 2, 4)
        assert result.steiner.dtype == np.float64

    def test_steiner_values_correct(self):
        """steiner[qi, gi, topo] == sum of Steiner values for group gi, topo k."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",  # topo 0
                "((a:1,b:1):1,(c:1,d:1):1);",  # topo 0
            ],
            "B": [
                "((a:1,c:1):1,(b:1,d:1):1);",  # topo 1
            ],
        }
        c = Forest(groups)
        result = c.quartet_topology(
            Quartets.from_list(c, [("a", "b", "c", "d")]), steiner=True
        )
        # Group A (idx 0): 2 trees both topo 0 — steiner[0,0,0] should be non-zero
        assert result.steiner[0, 0, 0] > 0.0
        assert result.steiner[0, 0, 1] == 0.0
        assert result.steiner[0, 0, 2] == 0.0
        # Group B (idx 1): 1 tree topo 1 — steiner[0,1,1] should be non-zero
        assert result.steiner[0, 1, 0] == 0.0
        assert result.steiner[0, 1, 1] > 0.0
        assert result.steiner[0, 1, 2] == 0.0

    def test_counts_only_works_without_steiner(self):
        """counts-only mode returns shape (n_quartets, n_groups, 4)."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        c = Forest(groups)
        result = c.quartet_topology(
            Quartets.from_list(c, [("a", "b", "c", "d")]), steiner=False
        )
        assert result.counts.shape == (1, 2, 4)
        assert isinstance(result, QuartetTopologyResult)


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing API."""

    def test_list_input_behaves_like_old_api(self):
        """Test that wrapping a list in Quartets.from_list() maintains old behavior."""
        trees = ["((A:1,B:1):1,(C:1,D:1):1);", "((A:1,C:1):1,(B:1,D:1):1);"]

        c = Forest(trees)

        # Forest attributes
        assert c.n_trees == 2
        assert c.n_global_taxa == 4
        assert "A" in c.global_names

        # Group attributes
        assert hasattr(c, "n_groups")
        assert hasattr(c, "group_labels")

        # Functionality via new Quartets interface
        result = c.quartet_topology(Quartets.from_list(c, [("A", "B", "C", "D")]))
        assert result.counts.shape == (1, 1, 4)

    def test_quartet_topology_unchanged(self):
        """Test that quartet_topology works correctly via Quartets.from_list."""
        trees = ["((A:1,B:1):1,(C:1,D:1):1);"] * 3
        c = Forest(trees)

        # Counts-only mode
        result = c.quartet_topology(Quartets.from_list(c, [("A", "B", "C", "D")]))
        assert result.counts.shape == (1, 1, 4)
        assert result.counts.sum() == 3  # All 3 trees

        # Steiner mode
        result_s = c.quartet_topology(
            Quartets.from_list(c, [("A", "B", "C", "D")]), steiner=True
        )
        assert result_s.counts.shape == (1, 1, 4)
        assert result_s.steiner.shape == (1, 1, 4)


class TestQuartetQED:
    """Tests for Forest.qed."""

    # ── Shared fixtures ─────────────────────────────────────────────────── #

    @pytest.fixture
    def two_group_forest(self):
        """2-group forest with a single quartet (a,b,c,d).

        Group A (idx 0): 2 trees both voting topology 0 — (ab)|(cd).
        Group B (idx 1): 2 trees both voting topology 1 — (ac)|(bd).
        """
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,b:1):1,(c:1,d:1):1);",
            ],
            "B": [
                "((a:1,c:1):1,(b:1,d:1):1);",
                "((a:1,c:1):1,(b:1,d:1):1);",
            ],
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        return forest, counts

    @pytest.fixture
    def single_group_forest(self):
        """1-group forest; default group_pairs is empty."""
        trees = ["((a:1,b:1):1,(c:1,d:1):1);"] * 3
        forest = Forest(trees)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        return forest, counts

    # ── Shape and dtype ──────────────────────────────────────────────────── #

    def test_output_shape_two_groups(self, two_group_forest):
        """Shape is (n_quartets, n_pairs) = (1, 1) for 2 groups, 1 quartet."""
        forest, counts = two_group_forest
        scores = forest.qed(counts)
        assert scores.shape == (1, 1)
        assert scores.dtype == np.float64

    def test_output_shape_single_group(self, single_group_forest):
        """Single group → 0 pairs → shape (n_quartets, 0)."""
        forest, counts = single_group_forest
        scores = forest.qed(counts)
        assert scores.shape == (1, 0)

    def test_output_shape_bulk_quartets(self):
        """Shape is (n_quartets, n_pairs) for multiple quartets."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);", "((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,c:1):1,(b:1,d:1):1);", "((a:1,d:1):1,(b:1,c:1):1);"],
        }
        forest = Forest(groups)
        quartets = [("a", "b", "c", "d"), ("a", "b", "c", "d")]  # 2 quartets
        q = Quartets.from_list(forest, quartets)
        counts = forest.quartet_topology(q)
        scores = forest.qed(counts)
        assert scores.shape == (2, 1)

    # ── Correctness ──────────────────────────────────────────────────────── #

    def test_perfect_agreement_is_plus_one(self, two_group_forest):
        """Both groups identical → score == +1.0."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"] * 4,
            "B": ["((a:1,b:1):1,(c:1,d:1):1);"] * 4,
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        scores = forest.qed(counts)
        assert abs(scores[0, 0] - 1.0) < 1e-12

    def test_perfect_disagreement_is_minus_one(self, two_group_forest):
        """Groups favour completely different topologies → score == -1.0."""
        forest, counts = two_group_forest
        scores = forest.qed(counts)
        assert abs(scores[0, 0] - (-1.0)) < 1e-12

    def test_missing_taxa_gives_zero(self):
        """Quartet absent in one group → score == 0.0."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((e:1,f:1):1,(g:1,h:1):1);"],  # disjoint taxa
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        result = forest.quartet_topology(q)
        # Group B has count 0 for this quartet
        assert result.counts[0, 1].sum() == 0
        scores = forest.qed(result)
        assert scores[0, 0] == 0.0

    def test_score_in_range(self):
        """All scores must lie in [-1, +1]."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,c:1):1,(b:1,d:1):1);",
                "((a:1,d:1):1,(b:1,c:1):1);",
            ],
            "B": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,c:1):1,(b:1,d:1):1);",
            ],
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        scores = forest.qed(counts)
        assert np.all(scores >= -1.0 - 1e-12)
        assert np.all(scores <= 1.0 + 1e-12)

    def test_same_dominant_topology_gives_positive_score(self):
        """Agreeing dominant topology → positive score even if distributions differ."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"] * 4,   # all topo 0
            "B": [
                "((a:1,b:1):1,(c:1,d:1):1);",           # topo 0
                "((a:1,c:1):1,(b:1,d:1):1);",           # topo 1
            ],
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        scores = forest.qed(counts)
        assert scores[0, 0] > 0.0

    # ── Three-group forest — all pairs ─────────────────────────────────────

    def test_three_groups_default_pairs(self):
        """3 groups → 3 pairs (0,1), (0,2), (1,2) by default."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "C": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        scores = forest.qed(counts)
        assert scores.shape == (1, 3)
        # Groups A and B are identical → pair (0,1) should be +1
        assert abs(scores[0, 0] - 1.0) < 1e-12
        # Groups A/B vs C disagree → pairs (0,2) and (1,2) should be -1
        assert abs(scores[0, 1] - (-1.0)) < 1e-12
        assert abs(scores[0, 2] - (-1.0)) < 1e-12

    # ── Custom group_pairs ───────────────────────────────────────────────── #

    def test_custom_group_pairs(self, two_group_forest):
        """Custom group_pairs selects specific comparisons."""
        forest, counts = two_group_forest
        gp = np.array([[1, 0]], dtype=np.int32)  # reversed order
        scores = forest.qed(counts, group_pairs=gp)
        assert scores.shape == (1, 1)
        # Reversed pair is the same comparison — score should be the same
        scores_default = forest.qed(counts)
        assert abs(scores[0, 0] - scores_default[0, 0]) < 1e-12

    def test_custom_group_pairs_same_group(self, two_group_forest):
        """Comparing a group to itself gives +1."""
        forest, counts = two_group_forest
        gp = np.array([[0, 0]], dtype=np.int32)
        scores = forest.qed(counts, group_pairs=gp)
        assert abs(scores[0, 0] - 1.0) < 1e-12

    # ── Validation errors ────────────────────────────────────────────────── #

    def test_wrong_shape_raises(self, two_group_forest):
        """Wrong counts shape raises ValueError."""
        forest, counts = two_group_forest
        bad_counts = np.zeros((1, 3), dtype=np.int32)  # missing n_groups axis
        with pytest.raises(ValueError, match="shape"):
            forest.qed(bad_counts)

    def test_out_of_range_group_pair_raises(self, two_group_forest):
        """Out-of-range group index raises ValueError."""
        forest, counts = two_group_forest
        gp = np.array([[0, 99]], dtype=np.int32)
        with pytest.raises(ValueError, match="indices"):
            forest.qed(counts, group_pairs=gp)

    # ── to_frame() ───────────────────────────────────────────────────────── #

    def test_to_frame_long_shape(self, two_group_forest):
        """Long form has n_quartets * n_pairs rows and expected columns."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        result = forest.qed(counts)
        df = result.to_frame("long")
        # quartet_idx, a, b, c, d, group_a, group_b, qed
        assert df.shape == (1, 8)
        assert list(df.columns) == [
            "quartet_idx", "a", "b", "c", "d", "group_a", "group_b", "qed"
        ]

    def test_to_frame_wide_shape(self, two_group_forest):
        """Wide form has n_quartets rows and one QED column per pair."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        result = forest.qed(counts)
        df = result.to_frame("wide")
        # quartet_idx, a, b, c, d, A_vs_B
        assert df.shape == (1, 6)
        assert df.columns[0] == "quartet_idx"
        assert "A_vs_B" in df.columns

    def test_to_frame_long_join_on_quartet_idx(self, two_group_forest):
        """Long QED and long topology join 1-to-(n_groups×3) on quartet_idx."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        qed_df = forest.qed(counts).to_frame("long")
        topo_df = counts.to_frame("long")
        joined = qed_df.join(topo_df, on="quartet_idx", how="left")
        # QED and topology columns both present
        assert "qed" in joined.columns
        assert "group" in joined.columns
        assert "topology" in joined.columns
        assert "count" in joined.columns
        # 1 QED row × (n_groups=2 × n_topologies=4) topology rows = 8
        assert joined.shape[0] == qed_df.shape[0] * forest.n_groups * 4

    def test_to_frame_wide_join_on_quartet_idx(self, two_group_forest):
        """Wide QED and wide topology join 1-to-1 on quartet_idx."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        qed_df = forest.qed(counts).to_frame("wide")
        topo_df = counts.to_frame("wide")
        joined = qed_df.join(topo_df, on="quartet_idx", how="left")
        assert joined.shape[0] == qed_df.shape[0]
        assert "A_vs_B" in joined.columns
        assert "A_t0" in joined.columns

    def test_to_frame_long_group_labels(self, two_group_forest):
        """group_a and group_b columns contain the correct group names."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        df = forest.qed(counts).to_frame("long")
        assert df["group_a"].to_list() == ["A"]
        assert df["group_b"].to_list() == ["B"]

    def test_to_frame_long_three_groups(self):
        """Long form for 3 groups produces n_quartets * 3 rows (one per pair)."""
        pytest.importorskip("polars")
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "C": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        df = forest.qed(counts).to_frame("long")
        assert df.shape[0] == 3  # 1 quartet * 3 pairs

    def test_to_frame_wide_three_groups(self):
        """Wide form for 3 groups has 3 QED columns."""
        pytest.importorskip("polars")
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "C": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        forest = Forest(groups)
        q = Quartets.from_list(forest, [("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        df = forest.qed(counts).to_frame("wide")
        qed_cols = [c for c in df.columns if "_vs_" in c]
        assert len(qed_cols) == 3

    def test_to_frame_invalid_form(self, two_group_forest):
        """Invalid form string raises ValueError."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        with pytest.raises(ValueError, match="form"):
            forest.qed(counts).to_frame("diagonal")

    def test_to_frame_raw_array_raises(self, two_group_forest):
        """to_frame() raises RuntimeError when built from raw ndarray."""
        pytest.importorskip("polars")
        forest, counts = two_group_forest
        result = forest.qed(counts.counts)  # raw ndarray — no metadata
        with pytest.raises(RuntimeError, match="metadata"):
            result.to_frame()

    # ── deduplicate parameter ─────────────────────────────────────────────── #

    @pytest.fixture
    def duplicate_quartet_forest(self):
        """2-group forest with the same quartet queried twice (explicit duplicates)."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);", "((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,c:1):1,(b:1,d:1):1);", "((a:1,c:1):1,(b:1,d:1):1);"],
        }
        forest = Forest(groups)
        # Same quartet listed twice → qi=0 and qi=1 are identical
        q = Quartets.from_list(forest, [("a", "b", "c", "d"), ("a", "b", "c", "d")])
        counts = forest.quartet_topology(q)
        return forest, counts

    def test_topology_deduplicate_true_long_removes_duplicate_rows(
        self, duplicate_quartet_forest
    ):
        """QuartetTopologyResult.to_frame('long', deduplicate=True) drops identical rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        df_dedup = counts.to_frame("long", deduplicate=True)
        df_raw = counts.to_frame("long", deduplicate=False)
        # Raw has 2 quartets × 2 groups × 4 topologies = 16 rows; dedup collapses to 8
        assert df_raw.shape[0] == 16
        assert df_dedup.shape[0] == 8

    def test_topology_deduplicate_true_wide_removes_duplicate_rows(
        self, duplicate_quartet_forest
    ):
        """QuartetTopologyResult.to_frame('wide', deduplicate=True) drops identical rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        df_dedup = counts.to_frame("wide", deduplicate=True)
        df_raw = counts.to_frame("wide", deduplicate=False)
        assert df_raw.shape[0] == 2   # two identical rows
        assert df_dedup.shape[0] == 1  # collapsed to one

    def test_topology_deduplicate_false_long_preserves_all_rows(
        self, duplicate_quartet_forest
    ):
        """QuartetTopologyResult.to_frame('long', deduplicate=False) keeps all rows."""
        pytest.importorskip("polars")
        _, counts = duplicate_quartet_forest
        df = counts.to_frame("long", deduplicate=False)
        assert df.shape[0] == 16  # 2 quartets × 2 groups × 4 topologies

    def test_topology_deduplicate_false_wide_preserves_all_rows(
        self, duplicate_quartet_forest
    ):
        """QuartetTopologyResult.to_frame('wide', deduplicate=False) keeps all rows."""
        pytest.importorskip("polars")
        _, counts = duplicate_quartet_forest
        df = counts.to_frame("wide", deduplicate=False)
        assert df.shape[0] == 2  # two identical rows

    def test_qed_deduplicate_true_long_removes_duplicate_rows(
        self, duplicate_quartet_forest
    ):
        """QEDResult.to_frame('long', deduplicate=True) drops identical rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        result = forest.qed(counts)
        df_dedup = result.to_frame("long", deduplicate=True)
        df_raw = result.to_frame("long", deduplicate=False)
        # Raw: 2 quartets × 1 pair = 2 rows; dedup collapses to 1
        assert df_raw.shape[0] == 2
        assert df_dedup.shape[0] == 1

    def test_qed_deduplicate_true_wide_removes_duplicate_rows(
        self, duplicate_quartet_forest
    ):
        """QEDResult.to_frame('wide', deduplicate=True) drops identical rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        result = forest.qed(counts)
        df_dedup = result.to_frame("wide", deduplicate=True)
        df_raw = result.to_frame("wide", deduplicate=False)
        assert df_raw.shape[0] == 2
        assert df_dedup.shape[0] == 1

    def test_qed_deduplicate_false_long_preserves_all_rows(
        self, duplicate_quartet_forest
    ):
        """QEDResult.to_frame('long', deduplicate=False) keeps all rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        df = forest.qed(counts).to_frame("long", deduplicate=False)
        assert df.shape[0] == 2  # 2 quartets × 1 pair

    def test_qed_deduplicate_false_wide_preserves_all_rows(
        self, duplicate_quartet_forest
    ):
        """QEDResult.to_frame('wide', deduplicate=False) keeps all rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        df = forest.qed(counts).to_frame("wide", deduplicate=False)
        assert df.shape[0] == 2

    def test_join_works_after_deduplication(self, duplicate_quartet_forest):
        """With duplicates present, deduplicate=True (default) enables a clean join."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        qed_df = forest.qed(counts).to_frame("wide")      # deduplicate=True
        topo_df = counts.to_frame("wide")                  # deduplicate=True
        joined = qed_df.join(topo_df, on="quartet_idx", how="left")
        # Both sides have 1 row after dedup → 1-to-1 join
        assert joined.shape[0] == 1

    def test_join_produces_cartesian_product_without_deduplication(
        self, duplicate_quartet_forest
    ):
        """With deduplicate=False, duplicate quartet_idx values cause extra join rows."""
        pytest.importorskip("polars")
        forest, counts = duplicate_quartet_forest
        qed_df = forest.qed(counts).to_frame("wide", deduplicate=False)
        topo_df = counts.to_frame("wide", deduplicate=False)
        joined = qed_df.join(topo_df, on="quartet_idx", how="left")
        # 2 qed rows × 2 matching topo rows each = 4 rows
        assert joined.shape[0] == 4


class TestGroupOffsets:
    """Tests for group_offsets CSR array."""

    def test_group_offsets_construction(self):
        """Test that group_offsets array is correct and reflects input tree counts."""
        groups = {
            "A": [
                "((a:1,b:1):1,(c:1,d:1):1);",
                "((a:1,b:1):1,(c:1,d:1):1);",
            ],
            "B": [
                "((e:1,f:1):1,(g:1,h:1):1);",
                "((e:1,f:1):1,(g:1,h:1):1);",
                "((e:1,f:1):1,(g:1,h:1):1);",
            ],
            "C": ["((i:1,j:1):1,(k:1,l:1):1);"],
        }

        c = Forest(groups)

        # Expected: [0, 2, 5, 6]
        expected = np.array([0, 2, 5, 6], dtype=np.int64)
        np.testing.assert_array_equal(c.group_offsets, expected)

        # Verify consistency
        assert c.group_offsets[-1] == c.n_trees

        # Verify group ranges
        assert c.group_offsets[1] - c.group_offsets[0] == 2  # Group A: 2 trees
        assert c.group_offsets[2] - c.group_offsets[1] == 3  # Group B: 3 trees
        assert c.group_offsets[3] - c.group_offsets[2] == 1  # Group C: 1 tree

    def test_group_offsets_single_group(self):
        """Test group_offsets with single group reflects input tree count."""
        trees = ["((A:1,B:1):1,(C:1,D:1):1);"] * 5
        c = Forest(trees)

        expected = np.array([0, 5], dtype=np.int64)
        np.testing.assert_array_equal(c.group_offsets, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
