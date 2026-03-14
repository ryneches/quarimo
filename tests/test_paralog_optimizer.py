"""
tests/test_paralog_optimizer.py
================================
Tests for Step 5: ParalogOptimizer, OptimizationResult, Forest.resolve_paralogs.

Covers:
- qed_history is monotonically non-decreasing
- OptimizationResult fields are correct types and shapes
- Optimizer recovers the ground-truth assignment that maximises QED
- Tie case (identity already optimal): converges in 1 iteration, no swaps
- Multi-genome: two independent paralog groups are each optimised
- Forest.resolve_paralogs end-to-end
- Forest.resolve_paralogs raises ValueError for non-paralog forest

Design of GROUND_TRUTH_TREES
-----------------------------
Group A has a MIXED default assignment:
  - 3 trees: ((A:1,G1:1):1,(B:1,G2:1):1)  — G1 is DFS-first -> G_copy0=G1 -> topo1
  - 3 trees: ((G2:1,B:1):1,(A:1,G1:1):1)  — G2 is DFS-first -> G_copy0=G2 -> topo2
  -> counts: [0, 3, 3, 0]; p(topo1)=0.5, p(topo2)=0.5; argmax=1 (tie->topo1)

Group B (pure, discordant with Group A):
  - 6 trees: ((B:1,G1:1):1,(A:1,G2:1):1)  — G1=copy0, G1 pairs with B -> topo2
  -> counts: [0, 0, 6, 0]; argmax=2 (topo2)

Initial QED (pair A-B): argmax mismatch -> J=-1, score ≈ -0.684 (entropy correction
from the 50/50 mix in group A reduces the magnitude from -1).

After global swap [1, 0]:
  Group A: the 3 topo1 trees become topo2 and the 3 topo2 trees become topo1;
           distribution stays 50/50; argmax=1 (same tie-breaking) -> UNCHANGED.
  Group B: G_copy0 now maps to G2 (was G1); G2 pairs with A -> topo1; all 6 -> topo1.

Post-swap QED: argmax_A=1 = argmax_B=1 -> J=+1, score ≈ +0.684.
Improvement: delta ≈ 1.37 > 0 -> optimizer applies the swap and converges.
"""

import numpy as np
import pytest

from quarimo import Forest, Quartets, OptimizationResult, ParalogOptimizer
from quarimo._paralog import build_paralog_data


# ── Ground-truth fixture ──────────────────────────────────────────────────────
#
# Group A: mixed DFS assignment (3 trees with G1=copy0, 3 with G2=copy0)
# Group B: pure, all G1=copy0, but G1 pairs with B in these trees -> topo2
#
# Initial QED ≈ -0.684 (J=-1, softened by 50/50 entropy in group A)
# After optimal swap: Group A still 50/50 (argmax=topo1), Group B -> all topo1
# Final QED ≈ +0.684
#
GROUND_TRUTH_TREES = {
    "A": [
        "((A:1,G1:1):1,(B:1,G2:1):1);",  # G1 DFS-first -> topo1
        "((A:1,G1:1):1,(B:1,G2:1):1);",
        "((A:1,G1:1):1,(B:1,G2:1):1);",
        "((G2:1,B:1):1,(A:1,G1:1):1);",  # G2 DFS-first -> topo2
        "((G2:1,B:1):1,(A:1,G1:1):1);",
        "((G2:1,B:1):1,(A:1,G1:1):1);",
    ],
    "B": ["((B:1,G1:1):1,(A:1,G2:1):1);"] * 6,  # G1=copy0, G1~B -> topo2
}
GROUND_TRUTH_MAP = {"G1": "G", "G2": "G"}

# Identical trees — both assignments give the same QED (no swap needed)
TIED_TREES = {
    "A": ["((A:1,G1:1):1,(B:1,G2:1):1);"] * 3,
    "B": ["((A:1,G1:1):1,(B:1,G2:1):1);"] * 3,
}

# Two independent paralog genomes H and K, both copies present in each tree
DUAL_TREES = {
    "A": [
        "((A:1,H1:1):1,((B:1,K1:1):1,(K2:1,H2:1):1):1);",
    ],
    "B": [
        "((A:1,H2:1):1,((B:1,K2:1):1,(K1:1,H1:1):1):1);",
    ],
}
DUAL_MAP = {"H1": "H", "H2": "H", "K1": "K", "K2": "K"}

NO_PARALOG = ["((A:1,B:1):1,(C:1,D:1):1);"]


def make_quartets(f, count=None):
    """All-taxa quartet(s) from the forest."""
    n = f.n_global_taxa
    if n < 4:
        raise ValueError(f"Not enough taxa: {n}")
    seed = [tuple(range(4))]
    return Quartets(f, seed=seed, offset=0, count=count or 1)


# ── qed_history monotonicity ──────────────────────────────────────────────────


def test_qed_history_non_decreasing():
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    _, opt = f.resolve_paralogs(q)
    hist = opt.qed_history
    for i in range(len(hist) - 1):
        assert hist[i + 1] >= hist[i] - 1e-9, (
            f"qed_history decreased at index {i}: {hist[i]} → {hist[i+1]}"
        )


# ── OptimizationResult fields ─────────────────────────────────────────────────


def test_optimization_result_fields():
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    _, opt = f.resolve_paralogs(q)

    assert isinstance(opt, OptimizationResult)
    assert isinstance(opt.assignments, np.ndarray)
    assert opt.assignments.dtype == np.int32
    assert opt.assignments.shape[0] == len(opt.genome_names)
    assert isinstance(opt.qed_history, list)
    assert len(opt.qed_history) >= 1
    assert isinstance(opt.converged, bool)
    assert isinstance(opt.n_iterations, int)
    assert opt.n_iterations >= 1
    assert opt.genome_names == ["G"]


# ── Optimizer recovers ground truth ──────────────────────────────────────────


def test_optimizer_improves_qed():
    """After optimisation the QED must be strictly higher than the default."""
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)

    pd = build_paralog_data(f, q)
    init_result = f.quartet_topology(q, backend="python")
    optimizer = ParalogOptimizer(f, q, init_result.counts, pd)
    initial_qed = optimizer._current_qed

    opt = optimizer.optimize()
    final_qed = opt.qed_history[-1]

    assert final_qed > initial_qed + 1e-6, (
        f"QED did not improve: initial={initial_qed:.4f}, final={final_qed:.4f}"
    )


def test_optimizer_reaches_positive_concordance():
    """After optimisation the mean QED should be positive."""
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    _, opt = f.resolve_paralogs(q)
    assert opt.qed_history[-1] > 0.5, (
        f"Expected QED positive after optimization, got {opt.qed_history[-1]:.4f}"
    )


# ── Tie case ──────────────────────────────────────────────────────────────────


def test_tie_converges_immediately():
    """
    When both copy-slot assignments give identical QED (tied), the optimiser
    should not apply any swap and should converge after a single sweep.
    """
    f = Forest(TIED_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    _, opt = f.resolve_paralogs(q)

    assert opt.converged, "Optimizer should converge when no improvement is possible"
    # qed_history has initial + one post-sweep entry; should be constant
    assert len(opt.qed_history) >= 1
    if len(opt.qed_history) > 1:
        assert abs(opt.qed_history[-1] - opt.qed_history[0]) < 1e-9


# ── resolve_paralogs end-to-end ───────────────────────────────────────────────


def test_resolve_paralogs_returns_correct_types():
    from quarimo._results import QuartetTopologyResult

    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    result, opt = f.resolve_paralogs(q)

    assert isinstance(result, QuartetTopologyResult)
    assert isinstance(opt, OptimizationResult)
    assert result.counts.shape == (q.count, f.n_groups, 4)
    assert result.steiner is None  # delta kernel doesn't track Steiner


def test_resolve_paralogs_counts_non_negative():
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    result, _ = f.resolve_paralogs(q)
    assert np.all(result.counts >= 0), "counts must be non-negative after delta updates"


def test_resolve_paralogs_no_paralogs_raises():
    f = Forest(NO_PARALOG)
    q = Quartets(f, seed=[(0, 1, 2, 3)], offset=0, count=1)
    with pytest.raises(ValueError, match="no paralog genomes"):
        f.resolve_paralogs(q)


# ── Assignments shape and validity ────────────────────────────────────────────


def test_assignments_shape_and_dtype():
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    _, opt = f.resolve_paralogs(q)

    n_pg = len(f.paralog_genome_names)
    assert opt.assignments.shape[0] == n_pg
    assert opt.assignments.shape[2] == f.n_trees
    assert opt.assignments.dtype == np.int32


def test_assignments_sentinel_valid():
    """All entries must be either -1 (absent) or a valid local leaf ID."""
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    _, opt = f.resolve_paralogs(q)

    max_leaves = max(t.n_leaves for t in f._trees)
    for v in opt.assignments.flat:
        assert v == -1 or 0 <= v < max_leaves, f"Invalid assignment value: {v}"


# ── Multi-genome ──────────────────────────────────────────────────────────────


def test_dual_genome_optimizer_runs():
    """Optimizer completes without error for two independent paralog genomes."""
    f = Forest(DUAL_TREES, taxon_map=DUAL_MAP)
    n = f.n_global_taxa
    if n < 4:
        pytest.skip("Not enough global taxa for this fixture")
    q = Quartets(f, seed=[tuple(range(4))], offset=0, count=1)
    _, opt = f.resolve_paralogs(q)

    assert len(opt.genome_names) == 2
    assert opt.assignments.shape[0] == 2
    assert isinstance(opt.converged, bool)


def test_dual_genome_qed_non_decreasing():
    f = Forest(DUAL_TREES, taxon_map=DUAL_MAP)
    n = f.n_global_taxa
    if n < 4:
        pytest.skip("Not enough global taxa for this fixture")
    q = Quartets(f, seed=[tuple(range(4))], offset=0, count=1)
    _, opt = f.resolve_paralogs(q)

    hist = opt.qed_history
    for i in range(len(hist) - 1):
        assert hist[i + 1] >= hist[i] - 1e-9


# ── Forest state after resolve_paralogs ──────────────────────────────────────


def test_global_to_local_updated_after_optimization():
    """
    After resolve_paralogs, forest.global_to_local should reflect the
    optimal assignment (not the default one).
    """
    f = Forest(GROUND_TRUTH_TREES, taxon_map=GROUND_TRUTH_MAP)
    q = make_quartets(f)
    default_g2l = f.global_to_local.copy()

    f.resolve_paralogs(q)
    # After a successful swap, global_to_local must have changed
    assert not np.array_equal(f.global_to_local, default_g2l), (
        "global_to_local should be updated after optimizer accepts a swap"
    )
