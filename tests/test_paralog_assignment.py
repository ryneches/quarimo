"""
tests/test_paralog_assignment.py
================================
Tests for Step 3: ParalogData class, inverted quartet index, and
apply_permutation.

Covers:
- build_paralog_data: correct taxon_quartet_offsets / taxon_quartet_ids
- taxon_quartet_offsets consistency: every quartet involving a paralog gid
  appears in the index for that gid
- apply_permutation: produces correct trial_global_to_local
- apply_permutation: identity permutation leaves global_to_local unchanged
- apply_permutation: affected_tree_ids contains exactly the trees with ≥2 copies
- apply_permutation: affected quartet set matches what we expect
- Forest.build_paralog_data: ValueError when no paralog genomes
"""

import pytest
import numpy as np

from quarimo import Forest, Quartets, ParalogData, build_paralog_data


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Two trees: tree0 has 1 G copy, tree1 has 2 G copies + singletons X, Y
FOREST_TREES = [
    "(G1:1,(X:1,Y:1):1);",
    "((G1:1,G2:1):1,(X:1,Y:1):1);",
]
TAXON_MAP = {"G1": "G", "G2": "G"}

NO_PARALOG = ["((A:1,B:1):1,(C:1,D:1):1);"]


def make_forest_and_quartets(seed=None, count=None):
    """Return (Forest, Quartets) for the two-tree paralog fixture."""
    f = Forest(FOREST_TREES, taxon_map=TAXON_MAP)
    # Use explicit seed quartets if count is small enough to be exhaustive
    if seed is None:
        # Use all 4-choose-4 = 1 possible quartet from 4 global taxa:
        # G_copy0, G_copy1, X, Y  (gids 0,1,2,3 depending on sort)
        # Build explicitly with all taxa in sorted global order
        all_taxa = list(range(f.n_global_taxa))  # [0,1,2,3]
        seed = [tuple(all_taxa)]
        count = count or 1
    q = Quartets(f, seed=seed, offset=0, count=count)
    return f, q


# ── build_paralog_data ────────────────────────────────────────────────────────


class TestBuildParalogData:
    def test_returns_paralog_data_instance(self):
        f, q = make_forest_and_quartets()
        pd = build_paralog_data(f, q)
        assert isinstance(pd, ParalogData)

    def test_genome_names(self):
        f, q = make_forest_and_quartets()
        pd = build_paralog_data(f, q)
        assert pd.genome_names == ["G"]
        assert pd.n_paralog_genomes == 1

    def test_quartet_taxa_shape(self):
        f, q = make_forest_and_quartets(count=1)
        pd = build_paralog_data(f, q)
        assert pd.quartet_taxa.shape == (1, 4)
        assert pd.quartet_taxa.dtype == np.int32

    def test_quartet_taxa_values(self):
        """quartet_taxa[qi] matches the taxa returned by iterating Quartets."""
        f, q = make_forest_and_quartets(count=1)
        pd = build_paralog_data(f, q)
        expected = list(q)[0]
        actual = tuple(int(x) for x in pd.quartet_taxa[0])
        assert set(actual) == set(expected)

    def test_taxon_quartet_offsets_length(self):
        f, q = make_forest_and_quartets()
        pd = build_paralog_data(f, q)
        assert len(pd.taxon_quartet_offsets) == f.n_global_taxa + 1

    def test_taxon_quartet_offsets_non_decreasing(self):
        f, q = make_forest_and_quartets()
        pd = build_paralog_data(f, q)
        assert np.all(np.diff(pd.taxon_quartet_offsets) >= 0)

    def test_total_entries_equals_4_times_n_quartets(self):
        """Each quartet contributes 4 taxon entries to the inverted index."""
        f, q = make_forest_and_quartets(count=1)
        pd = build_paralog_data(f, q)
        total = int(pd.taxon_quartet_offsets[-1])
        assert total == 4 * q.count

    def test_every_taxon_in_quartet_is_indexed(self):
        """Every gid in a quartet appears in the inverted index for that gid."""
        f, q = make_forest_and_quartets(count=1)
        pd = build_paralog_data(f, q)
        for qi, quartet in enumerate(q):
            for gid in quartet:
                start = int(pd.taxon_quartet_offsets[gid])
                end = int(pd.taxon_quartet_offsets[gid + 1])
                indexed_qis = pd.taxon_quartet_ids[start:end].tolist()
                assert qi in indexed_qis, f"qi={qi} not in index for gid={gid}"

    def test_paralog_gid_in_every_quartet(self):
        """
        The paralog copy global IDs (G_copy0, G_copy1) appear in the index
        for any quartet that uses them.
        """
        f, q = make_forest_and_quartets(count=1)
        pd = build_paralog_data(f, q)
        # Find gids for G_copy0 and G_copy1
        li = pd.genome_names.index("G")
        gid_copy0 = int(pd.copy_global_ids[int(pd.copy_offsets[li])])
        gid_copy1 = int(pd.copy_global_ids[int(pd.copy_offsets[li]) + 1])

        for gid in (gid_copy0, gid_copy1):
            start = int(pd.taxon_quartet_offsets[gid])
            end = int(pd.taxon_quartet_offsets[gid + 1])
            # The single quartet uses all 4 taxa including both G copies
            assert end - start == 1, f"Expected 1 entry for gid {gid}, got {end-start}"

    def test_forest_build_paralog_data_method(self):
        """Forest.build_paralog_data delegates to build_paralog_data."""
        f, q = make_forest_and_quartets()
        pd = f.build_paralog_data(q)
        assert isinstance(pd, ParalogData)

    def test_forest_build_paralog_data_no_paralogs_raises(self):
        """Forest.build_paralog_data raises ValueError for non-paralog forest."""
        f = Forest(NO_PARALOG)
        q = Quartets(f, seed=[(0, 1, 2, 3)], offset=0, count=1)
        with pytest.raises(ValueError, match="no paralog genomes"):
            f.build_paralog_data(q)


# ── apply_permutation ─────────────────────────────────────────────────────────


class TestApplyPermutation:
    def _setup(self):
        f, q = make_forest_and_quartets(count=1)
        pd = build_paralog_data(f, q)
        return f, pd

    def test_identity_unchanged(self):
        """Identity permutation leaves global_to_local unchanged."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        k = int(pd.copy_offsets[li + 1]) - int(pd.copy_offsets[li])
        perm = np.arange(k, dtype=np.int32)
        trial, _, _, _ = pd.apply_permutation(li, perm, f.global_to_local)
        np.testing.assert_array_equal(trial, f.global_to_local)

    def test_swap_changes_paralog_entries(self):
        """[1, 0] permutation swaps the two copy-slot leaf assignments."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        trial, _, _, _ = pd.apply_permutation(li, perm, f.global_to_local)

        gid0 = int(pd.copy_global_ids[int(pd.copy_offsets[li])])
        gid1 = int(pd.copy_global_ids[int(pd.copy_offsets[li]) + 1])

        # In tree1 (ti=1) both copies are present — their leaves should swap
        old_l0_t1 = int(f.global_to_local[1, gid0])
        old_l1_t1 = int(f.global_to_local[1, gid1])
        new_l0_t1 = int(trial[1, gid0])
        new_l1_t1 = int(trial[1, gid1])
        assert new_l0_t1 == old_l1_t1, "G_copy0 should now hold G_copy1's old leaf"
        assert new_l1_t1 == old_l0_t1, "G_copy1 should now hold G_copy0's old leaf"

    def test_swap_does_not_change_singletons(self):
        """Non-paralog taxa in global_to_local are untouched by permutation."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        trial, _, _, _ = pd.apply_permutation(li, perm, f.global_to_local)

        gid_x = f.global_names.index("X")
        gid_y = f.global_names.index("Y")
        np.testing.assert_array_equal(trial[:, gid_x], f.global_to_local[:, gid_x])
        np.testing.assert_array_equal(trial[:, gid_y], f.global_to_local[:, gid_y])

    def test_affected_tree_ids(self):
        """
        Only tree1 (index 1) has ≥2 copies of G — so it is the only
        affected tree for any non-identity permutation.
        """
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        _, affected_trees, _, _ = pd.apply_permutation(li, perm, f.global_to_local)
        assert 1 in affected_trees.tolist(), "tree1 should be affected"
        assert 0 not in affected_trees.tolist(), "tree0 has only 1 copy — not affected"

    def test_identity_affected_tree_ids(self):
        """
        Identity permutation still returns tree1 as affected (it HAS ≥2
        copies — it's the caller's responsibility to skip no-op permutations).
        """
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        k = int(pd.copy_offsets[li + 1]) - int(pd.copy_offsets[li])
        perm = np.arange(k, dtype=np.int32)
        _, affected_trees, _, _ = pd.apply_permutation(li, perm, f.global_to_local)
        assert 1 in affected_trees.tolist()

    def test_affected_quartet_qi_nonempty(self):
        """The single quartet involves both G copies — it is affected."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        _, _, _, affected_qi = pd.apply_permutation(li, perm, f.global_to_local)
        assert len(affected_qi) == 1
        assert int(affected_qi[0]) == 0

    def test_affected_quartet_taxa_shape(self):
        """affected_quartet_taxa has shape (n_affected, 4)."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        _, _, affected_taxa, affected_qi = pd.apply_permutation(li, perm, f.global_to_local)
        assert affected_taxa.shape == (len(affected_qi), 4)

    def test_double_swap_is_identity(self):
        """Applying [1,0] twice returns to the original global_to_local."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        trial1, _, _, _ = pd.apply_permutation(li, perm, f.global_to_local)
        trial2, _, _, _ = pd.apply_permutation(li, perm, trial1)
        np.testing.assert_array_equal(trial2, f.global_to_local)

    def test_does_not_mutate_original(self):
        """apply_permutation must not modify global_to_local in place."""
        f, pd = self._setup()
        li = pd.genome_names.index("G")
        perm = np.array([1, 0], dtype=np.int32)
        original = f.global_to_local.copy()
        pd.apply_permutation(li, perm, f.global_to_local)
        np.testing.assert_array_equal(f.global_to_local, original)
