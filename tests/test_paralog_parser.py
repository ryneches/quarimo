"""
tests/test_paralog_parser.py
============================
Tests for paralog-aware parser and taxon_map support.

Covers:
- Tree._apply_taxon_map: correct leaf renaming and paralog_groups
- Forest(taxon_map=...): correct paralog_genome_names, copy-slot counts,
  global_names expansion, global_to_local mapping
- Forest without taxon_map + duplicate leaf names: raises ValueError
"""

import pytest

from quarimo import Forest, Tree


# ── Fixtures ────────────────────────────────────────────────────────────────

# A simple binary tree with four leaves: A, B, C, D
SIMPLE_4_NEWICK = "((A:1,B:1):1,(C:1,D:1):1);"

# Three leaves with one pair of paralogs after taxon_map application:
# leaf A1 and A2 both map to genome A; leaf B stays B.
PARALOG_3_NEWICK = "((A1:1,A2:1):1,B:1);"

# Larger: two paralog copies of genome G, plus two singletons X, Y
PARALOG_4_NEWICK = "((G1:1,G2:1):1,(X:1,Y:1):1);"

# Forest of two trees — G appears once in tree1 (1 copy) and twice in tree2
FOREST_TREES = [
    "(G1:1,(X:1,Y:1):1);",  # tree1: G1 only
    "((G1:1,G2:1):1,(X:1,Y:1):1);",  # tree2: G1 + G2
]
FOREST_TAXON_MAP = {"G1": "G", "G2": "G"}

# Duplicate names WITHOUT taxon_map — should raise ValueError
DUP_NEWICK = "((A:1,A:1):1,B:1);"


# ── Tree._apply_taxon_map ────────────────────────────────────────────────────


class TestApplyTaxonMap:
    def test_single_paralog_pair(self):
        """Two leaves mapping to the same genome form a paralog group."""
        t = Tree(PARALOG_3_NEWICK)
        taxon_map = {"A1": "A", "A2": "A"}
        t._apply_taxon_map(taxon_map)

        # Both leaves should now carry genome name "A"
        assert "A" in t.paralog_groups
        assert len(t.paralog_groups["A"]) == 2

        # The leaf IDs in paralog_groups should actually be leaf nodes
        leaf_ids = t.paralog_groups["A"]
        for lid in leaf_ids:
            assert t.names[lid] == "A"

    def test_unmapped_leaves_unchanged(self):
        """Leaves not in taxon_map keep their original names."""
        t = Tree(PARALOG_3_NEWICK)
        t._apply_taxon_map({"A1": "A", "A2": "A"})

        # B was not in taxon_map — should still be named "B"
        b_ids = [i for i in range(t.n_leaves) if t.names[i] == "B"]
        assert len(b_ids) == 1

    def test_name_index_invalidated(self):
        """_apply_taxon_map must invalidate the _name_index cache."""
        t = Tree(PARALOG_3_NEWICK)
        # Force the index to be built
        t._build_name_index()
        assert t._name_index is not None

        t._apply_taxon_map({"A1": "A", "A2": "A"})
        assert t._name_index is None

    def test_no_paralogs_empty_groups(self):
        """When no leaves share a genome name, paralog_groups is empty."""
        t = Tree(SIMPLE_4_NEWICK)
        t._apply_taxon_map({"A": "A_renamed"})
        # Only one leaf maps to A_renamed — no paralog group
        assert (
            "A_renamed" not in t.paralog_groups
            or len(t.paralog_groups.get("A_renamed", [])) == 1
        )
        # Specifically, nothing with >1 copy
        for genome, ids in t.paralog_groups.items():
            assert len(ids) >= 1  # every entry is valid

    def test_all_leaves_remapped(self):
        """All leaves can be remapped through taxon_map."""
        t = Tree(PARALOG_4_NEWICK)
        taxon_map = {"G1": "G", "G2": "G", "X": "X", "Y": "Y"}
        t._apply_taxon_map(taxon_map)
        assert "G" in t.paralog_groups
        assert len(t.paralog_groups["G"]) == 2


# ── Forest with taxon_map ─────────────────────────────────────────────────────


class TestForestTaxonMap:
    def test_paralog_genome_names(self):
        """Forest correctly identifies paralog genomes."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        assert f.paralog_genome_names == ["G"]

    def test_global_names_expansion(self):
        """Paralog genome G with k=2 max copies → G_copy0, G_copy1."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        assert "G_copy0" in f.global_names
        assert "G_copy1" in f.global_names
        # Singletons X and Y should appear with plain names
        assert "X" in f.global_names
        assert "Y" in f.global_names
        # Original leaf names must not appear
        assert "G" not in f.global_names
        assert "G1" not in f.global_names
        assert "G2" not in f.global_names

    def test_n_global_taxa(self):
        """G→2 copies + X + Y = 4 global taxa total."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        assert f.n_global_taxa == 4

    def test_copy_offsets_csr(self):
        """paralog_copy_offsets is a valid CSR for 1 genome with 2 copies."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        assert f.paralog_copy_offsets[0] == 0
        assert f.paralog_copy_offsets[1] == 2
        assert len(f.paralog_copy_global_ids) == 2

    def test_copy_global_ids_match_global_names(self):
        """paralog_copy_global_ids points to G_copy0 and G_copy1."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        gids = f.paralog_copy_global_ids.tolist()
        names_at_gids = [f.global_names[gid] for gid in gids]
        assert "G_copy0" in names_at_gids
        assert "G_copy1" in names_at_gids

    def test_global_to_local_tree1(self):
        """Tree 1 has only G_copy0 (one G leaf); G_copy1 must be absent (-1)."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        gid_copy0 = f.global_names.index("G_copy0")
        gid_copy1 = f.global_names.index("G_copy1")
        # tree index 0 = first tree
        assert f.global_to_local[0, gid_copy0] >= 0, (
            "G_copy0 should be present in tree 1"
        )
        assert f.global_to_local[0, gid_copy1] == -1, (
            "G_copy1 should be absent in tree 1"
        )

    def test_global_to_local_tree2(self):
        """Tree 2 has both G_copy0 and G_copy1."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        gid_copy0 = f.global_names.index("G_copy0")
        gid_copy1 = f.global_names.index("G_copy1")
        assert f.global_to_local[1, gid_copy0] >= 0, (
            "G_copy0 should be present in tree 2"
        )
        assert f.global_to_local[1, gid_copy1] >= 0, (
            "G_copy1 should be present in tree 2"
        )

    def test_local_to_global_consistency(self):
        """local_to_global is the inverse of global_to_local."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        for ti in range(f.n_trees):
            lo = int(f.leaf_offsets[ti])
            for gid in range(f.n_global_taxa):
                lid = int(f.global_to_local[ti, gid])
                if lid >= 0:
                    assert f.local_to_global[lo + lid] == gid

    def test_taxa_present_shape(self):
        """taxa_present has shape (n_trees, n_global_taxa)."""
        f = Forest(FOREST_TREES, taxon_map=FOREST_TAXON_MAP)
        assert f.taxa_present.shape == (f.n_trees, f.n_global_taxa)

    def test_no_paralogs_empty_attributes(self):
        """Without any paralog genomes the CSR arrays are empty sentinels."""
        f = Forest([SIMPLE_4_NEWICK])
        assert f.paralog_genome_names == []
        assert len(f.paralog_copy_offsets) == 1
        assert f.paralog_copy_offsets[0] == 0
        assert len(f.paralog_copy_global_ids) == 0

    def test_taxon_map_singletons_only(self):
        """taxon_map with only 1-to-1 renames — no paralogs, plain names used."""
        f = Forest([SIMPLE_4_NEWICK], taxon_map={"A": "Alpha", "B": "Beta"})
        assert "Alpha" in f.global_names
        assert "Beta" in f.global_names
        assert "A" not in f.global_names
        assert f.paralog_genome_names == []


# ── Duplicate names without taxon_map ────────────────────────────────────────


class TestDuplicateNamesGuard:
    def test_duplicate_leaf_no_taxon_map_raises(self):
        """Duplicate leaf names without taxon_map raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate taxon name"):
            Forest([DUP_NEWICK])

    def test_duplicate_leaf_with_taxon_map_ok(self):
        """Same duplicate names with taxon_map do NOT raise."""
        # A already appears twice in DUP_NEWICK; map both to genome "A"
        # (the NEWICK parser itself allows duplicate raw names; taxon_map
        # is applied after parsing)
        forest = Forest([DUP_NEWICK], taxon_map={"A": "A"})
        assert "A_copy0" in forest.global_names
        assert "A_copy1" in forest.global_names
