"""
Tests for the ``polytomy_strategy`` parameter on :class:`Tree` and :class:`Forest`.

Three strategies
----------------
'multifurcation'   (default)  — true multifurcations → polytomies;
                                zero-length branches → real branches + warning.
'zero-length-branch'          — zero-length internal branches → polytomies;
                                true multifurcations → resolved + warning.
'both'                        — both forms → polytomies; no warnings.
"""
import logging
import pytest
from quarimo import Forest, Tree
from quarimo._quartets import Quartets


# ── Shared NEWICK fixtures ───────────────────────────────────────────────── #

# True multifurcation: (A, B, C) at root level
_MULTIFURC_NWK = "((A:1,B:1,C:1):1,D:1);"

# Binary tree with a zero-length internal branch (A+B clade has 0 branch to root)
_ZERO_LEN_INTERNAL_NWK = "((A:1,B:1):0.0,(C:1,D:1):1.0);"

# Both internal branches zero-length: scores fully tie under 'zero-length-branch',
# triggering polytomy (unresolvable) detection.
_ZERO_LEN_BOTH_INTERNAL_NWK = "((A:1,B:1):0.0,(C:1,D:1):0.0);"

# Binary tree where the zero-length branch is on a leaf
_ZERO_LEN_LEAF_NWK = "(A:0.0,(B:1,C:1):1.0);"

# Fully resolved binary tree — no zero-length, no multifurcation
_CLEAN_NWK = "((A:1,B:1):1,(C:1,D:1):1);"


# ============================================================================
# Tree-level tests
# ============================================================================


class TestTreePolytomyStrategy:
    """Tree.__init__ validates polytomy_strategy and stores had_multifurcations."""

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="polytomy_strategy"):
            Tree(_CLEAN_NWK, polytomy_strategy="wrong")

    def test_default_strategy_is_multifurcation(self):
        t = Tree(_CLEAN_NWK)
        # No polytomies, no zeros — trivial check that default still works
        assert t.polytomy_node_ids == []
        assert t.n_zero_length_branches == 0
        assert not t.had_multifurcations

    # ── 'multifurcation' strategy ─────────────────────────────────────────

    def test_multifurcation_strategy_resolves_multifurc(self):
        t = Tree(_MULTIFURC_NWK, polytomy_strategy="multifurcation")
        assert t.had_multifurcations
        assert len(t.polytomy_node_ids) > 0, "inserted node must be recorded"

    def test_multifurcation_strategy_zero_len_not_polytomy(self):
        """Under 'multifurcation', zero-length internal branches are NOT polytomies."""
        t = Tree(_ZERO_LEN_INTERNAL_NWK, polytomy_strategy="multifurcation")
        assert not t.had_multifurcations
        assert t.polytomy_node_ids == []
        assert t.n_zero_length_branches > 0

    # ── 'zero-length-branch' strategy ─────────────────────────────────────

    def test_zero_len_strategy_absorbs_zero_len_branch(self):
        """Under 'zero-length-branch', internal zero-length branches become polytomies."""
        t = Tree(_ZERO_LEN_INTERNAL_NWK, polytomy_strategy="zero-length-branch")
        assert len(t.polytomy_node_ids) > 0, "zero-length internal must be in polytomy_ids"
        assert t.n_zero_length_branches == 0

    def test_zero_len_strategy_still_resolves_multifurc(self):
        """Under 'zero-length-branch', true multifurcations are still binarized."""
        t = Tree(_MULTIFURC_NWK, polytomy_strategy="zero-length-branch")
        assert t.had_multifurcations
        assert len(t.polytomy_node_ids) > 0

    def test_zero_len_strategy_clean_tree_has_no_polytomies(self):
        t = Tree(_CLEAN_NWK, polytomy_strategy="zero-length-branch")
        assert t.polytomy_node_ids == []
        assert t.n_zero_length_branches == 0
        assert not t.had_multifurcations

    def test_zero_len_leaf_branch_not_added_to_polytomy_ids(self):
        """Leaf zero-length branches have no LCA effect; they are silently absorbed."""
        t = Tree(_ZERO_LEN_LEAF_NWK, polytomy_strategy="zero-length-branch")
        # The leaf (A:0) itself is not an internal node, so it won't appear in
        # polytomy_node_ids (only internal nodes can be LCAs).
        assert t.n_zero_length_branches == 0
        # polytomy_node_ids may be empty (no internal zero-length branch)
        for nid in t.polytomy_node_ids:
            assert nid >= t.n_leaves, "polytomy ids must be internal nodes"

    # ── 'both' strategy ───────────────────────────────────────────────────

    def test_both_strategy_absorbs_zero_len_branch(self):
        t = Tree(_ZERO_LEN_INTERNAL_NWK, polytomy_strategy="both")
        assert len(t.polytomy_node_ids) > 0
        assert t.n_zero_length_branches == 0

    def test_both_strategy_resolves_multifurc(self):
        t = Tree(_MULTIFURC_NWK, polytomy_strategy="both")
        assert t.had_multifurcations
        assert len(t.polytomy_node_ids) > 0

    def test_both_strategy_clean_tree(self):
        t = Tree(_CLEAN_NWK, polytomy_strategy="both")
        assert t.polytomy_node_ids == []
        assert t.n_zero_length_branches == 0


# ============================================================================
# Forest-level tests
# ============================================================================


class TestForestPolytomyStrategy:
    """Forest.__init__ validates strategy and emits the correct warnings."""

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="polytomy_strategy"):
            Forest([_CLEAN_NWK], polytomy_strategy="oops")

    def test_default_no_warning_for_clean_trees(self, caplog):
        # Use WARNING level on the quarimo logger; don't use quiet() since that
        # suppresses the very warnings we want to check for (or their absence).
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest([_CLEAN_NWK, _CLEAN_NWK])
        assert "zero-length" not in caplog.text.lower()
        assert "multifurcation" not in caplog.text.lower()

    # ── 'multifurcation' ─────────────────────────────────────────────────

    def test_multifurcation_emits_zero_length_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest([_ZERO_LEN_INTERNAL_NWK], polytomy_strategy="multifurcation")
        assert "zero-length" in caplog.text.lower()

    def test_multifurcation_no_warning_for_multifurc(self, caplog):
        """True multifurcations are resolved silently under 'multifurcation'."""
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest([_MULTIFURC_NWK], polytomy_strategy="multifurcation")
        # The word "multifurcation" should only appear in the polytomy
        # statistics INFO line, not in any WARNING.
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("multifurcation" in r.message.lower() for r in warning_records)

    # ── 'zero-length-branch' ─────────────────────────────────────────────

    def test_zero_len_strategy_no_warning_for_zero_len(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest([_ZERO_LEN_INTERNAL_NWK], polytomy_strategy="zero-length-branch")
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("zero-length" in r.message.lower() for r in warning_records)

    def test_zero_len_strategy_warns_for_multifurc(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest([_MULTIFURC_NWK], polytomy_strategy="zero-length-branch")
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("multifurcation" in r.message.lower() for r in warning_records)

    def test_zero_len_strategy_no_warning_when_no_multifurc(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest([_ZERO_LEN_INTERNAL_NWK], polytomy_strategy="zero-length-branch")
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("multifurcation" in r.message.lower() for r in warning_records)

    # ── 'both' ───────────────────────────────────────────────────────────

    def test_both_strategy_no_warnings(self, caplog):
        with caplog.at_level(logging.WARNING, logger="quarimo._logging"):
            Forest(
                [_MULTIFURC_NWK, _ZERO_LEN_INTERNAL_NWK],
                polytomy_strategy="both",
            )
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any(
            "zero-length" in r.message.lower() or "multifurcation" in r.message.lower()
            for r in warning_records
        )

    # ── Polytomy CSR is correct ───────────────────────────────────────────

    def test_zero_len_branch_appears_in_polytomy_csr(self):
        """Under 'zero-length-branch', the zero-length internal node ends up
        in the polytomy CSR arrays that the kernels use."""
        default = Forest([_ZERO_LEN_INTERNAL_NWK], polytomy_strategy="multifurcation")
        zlb = Forest([_ZERO_LEN_INTERNAL_NWK], polytomy_strategy="zero-length-branch")
        # default: no polytomy nodes
        assert int(default.polytomy_offsets[-1]) == 0
        # zero-length-branch: polytomy node was registered
        assert int(zlb.polytomy_offsets[-1]) > 0

    def test_quartet_unresolved_under_zero_len_strategy(self):
        """
        When both internal branches are zero-length, the four-point-condition
        scores tie (r0 == r1 == r2 == 0) and all internal nodes are registered
        as polytomy nodes.  The CSR pre-filter fires and the quartet is
        unresolvable (k=3).

        Under the default 'multifurcation' strategy, the same tree has no
        polytomy nodes in the CSR, so the tie does not trigger k=3 and the
        quartet is assigned some resolved topology.
        """
        zlb = Forest([_ZERO_LEN_BOTH_INTERNAL_NWK], polytomy_strategy="zero-length-branch")
        q = Quartets.from_list(zlb, [("A", "B", "C", "D")])
        counts_zlb = zlb.quartet_topology(q).counts
        assert counts_zlb[0, 0, 3] > 0, "expected unresolved under zero-length-branch"

    def test_quartet_resolved_under_multifurcation_strategy(self):
        """Under the default 'multifurcation' strategy, zero-length branches
        are treated as real branches and the CSR polytomy check never fires,
        so the same fully-tied quartet is forced into a resolved topology."""
        f = Forest([_ZERO_LEN_BOTH_INTERNAL_NWK], polytomy_strategy="multifurcation")
        q = Quartets.from_list(f, [("A", "B", "C", "D")])
        counts = f.quartet_topology(q).counts
        assert counts[0, 0, 3] == 0, "expected resolved topology under multifurcation"
        assert counts[0, 0, :3].sum() > 0
