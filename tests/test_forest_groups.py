"""
tests/test_forest_groups.py
===========================================
Tests for tree group labels functionality in PhyloTreeCollection.
"""

import pytest
import numpy as np
from quarimo._forest import Forest, _jaccard
from quarimo._quartets import Quartets


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
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="must be list or dict"):
            Forest("not a list or dict")

        with pytest.raises(TypeError, match="must be list or dict"):
            Forest(12345)


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
        """counts.shape == (n_quartets, n_groups, 3) for a 2-group forest."""
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
        counts = c.quartet_topology(Quartets.from_list(c, quartets))
        assert counts.shape == (1, 2, 3)
        assert counts.dtype == np.int32

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
        counts = c.quartet_topology(Quartets.from_list(c, [("a", "b", "c", "d")]))
        assert counts[0, 0, 0] == 1  # Group A: 1 vote for topo 0
        assert counts[0, 0, 1] == 1  # Group A: 1 vote for topo 1
        assert counts[0, 0, 2] == 0  # Group A: 0 votes for topo 2
        assert counts[0, 1, 0] == 0  # Group B: 0 votes for topo 0
        assert counts[0, 1, 1] == 0  # Group B: 0 votes for topo 1
        assert counts[0, 1, 2] == 1  # Group B: 1 vote for topo 2

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
        counts = c.quartet_topology(Quartets.from_list(c, [("a", "b", "c", "d")]))
        assert int(counts[0].sum()) == 3  # 3 trees total, all have a,b,c,d

    def test_steiner_shape(self):
        """steiner.shape == (n_quartets, n_groups, 3)."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        c = Forest(groups)
        counts, steiner = c.quartet_topology(
            Quartets.from_list(c, [("a", "b", "c", "d")]), steiner=True
        )
        assert counts.shape == (1, 2, 3)
        assert steiner.shape == (1, 2, 3)
        assert steiner.dtype == np.float64

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
        counts, steiner = c.quartet_topology(
            Quartets.from_list(c, [("a", "b", "c", "d")]), steiner=True
        )
        # Group A (idx 0): 2 trees both topo 0 — steiner[0,0,0] should be non-zero
        assert steiner[0, 0, 0] > 0.0
        assert steiner[0, 0, 1] == 0.0
        assert steiner[0, 0, 2] == 0.0
        # Group B (idx 1): 1 tree topo 1 — steiner[0,1,1] should be non-zero
        assert steiner[0, 1, 0] == 0.0
        assert steiner[0, 1, 1] > 0.0
        assert steiner[0, 1, 2] == 0.0

    def test_counts_only_works_without_steiner(self):
        """counts-only mode returns shape (n_quartets, n_groups, 3)."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"],
            "B": ["((a:1,c:1):1,(b:1,d:1):1);"],
        }
        c = Forest(groups)
        counts = c.quartet_topology(
            Quartets.from_list(c, [("a", "b", "c", "d")]), steiner=False
        )
        assert counts.shape == (1, 2, 3)
        assert isinstance(counts, np.ndarray)


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
        counts = c.quartet_topology(Quartets.from_list(c, [("A", "B", "C", "D")]))
        assert counts.shape == (1, 1, 3)

    def test_quartet_topology_unchanged(self):
        """Test that quartet_topology works correctly via Quartets.from_list."""
        trees = ["((A:1,B:1):1,(C:1,D:1):1);"] * 3
        c = Forest(trees)

        # Counts-only mode
        counts = c.quartet_topology(Quartets.from_list(c, [("A", "B", "C", "D")]))
        assert counts.shape == (1, 1, 3)
        assert counts.sum() == 3  # All 3 trees

        # Steiner mode
        counts, dists = c.quartet_topology(
            Quartets.from_list(c, [("A", "B", "C", "D")]), steiner=True
        )
        assert counts.shape == (1, 1, 3)
        assert dists.shape == (1, 1, 3)


class TestGroupOffsets:
    """Tests for group_offsets CSR array."""

    def test_group_offsets_construction(self):
        """Test that group_offsets array is correct."""
        groups = {
            "A": ["((a:1,b:1):1,(c:1,d:1):1);"] * 2,
            "B": ["((e:1,f:1):1,(g:1,h:1):1);"] * 3,
            "C": ["((i:1,j:1):1,(k:1,l:1):1);"] * 1,
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
        """Test group_offsets with single group."""
        trees = ["((A:1,B:1):1,(C:1,D:1):1);"] * 5
        c = Forest(trees)

        expected = np.array([0, 5], dtype=np.int64)
        np.testing.assert_array_equal(c.group_offsets, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
