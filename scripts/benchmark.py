import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import time
    import random
    import math
    import polars as pl
    from dendropy.simulate import treesim
    from quarimo import Forest, Quartets, quiet

    return Forest, Quartets, math, pl, quiet, random, time, treesim


@app.cell
def _(mo):
    mo.md(r"""
    # Quartet Kernel Benchmark — GB10 Grace Blackwell Reference Case

    This notebook measures `Forest.quartet_topology()` throughput across
    (n_trees, n_quartets) space using synthetic tree ensembles, and projects
    the cost of adding per-group Steiner statistics (min, max, variance) to
    the kernel.

    **Hardware reference: NVIDIA GB10 Grace Blackwell (DGX Spark)**

    | Parameter | Value |
    |---|---|
    | CUDA cores | 6,144 (96 SM × 64) |
    | Max resident threads | 196,608 |
    | L2 cache | ~96 MB (single Blackwell die) |
    | Memory bandwidth | 273 GB/s (LPDDR5x unified) |
    | Unified memory | 128 GB |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Parameters

    `N_TAXA = 1_000` runs quickly for development.
    Set to `10_000` for the full-scale design-target benchmark;
    tree generation will be proportionally slower.
    """)
    return


@app.cell
def _():
    N_TAXA           = 1_000   # design target: 10_000
    N_GROUPS         = 2
    REPEATS          = 3       # best-of-N timing repetitions per cell
    BACKEND          = "best"  # 'best' | 'cpu-parallel' | 'cuda'
    N_TREES_SWEEP    = [10, 30, 100, 300, 1_000]
    N_QUARTETS_SWEEP = [1_000, 10_000, 100_000]
    return BACKEND, N_GROUPS, N_QUARTETS_SWEEP, N_TAXA, N_TREES_SWEEP, REPEATS


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-Tree Memory Footprint

    For a binary tree with $n$ taxa: nodes $= 2n-1$, Euler tour $\approx 4n$,
    sparse table levels $= \lfloor\log_2(4n)\rfloor + 1$.
    The **sparse table (int32)** dominates at ~77% of total per-tree device memory.
    """)
    return


@app.cell
def _(N_TAXA, math, pl):
    _tour  = 4 * N_TAXA - 3
    _lg    = math.floor(math.log2(_tour)) + 1
    _nodes = 2 * N_TAXA - 1

    _components = [
        ("sparse_table",    "int32",   _lg * _tour,  _lg * _tour * 4),
        ("euler_tour",      "int32",   _tour,         _tour * 4),
        ("euler_depth",     "int32",   _tour,         _tour * 4),
        ("root_distance",   "float64", _nodes,        _nodes * 8),
        ("first_occ",       "int32",   _nodes,        _nodes * 4),
        ("log2_table",      "int32",   _tour + 1,     (_tour + 1) * 4),
        ("global_to_local", "int32",   N_TAXA,        N_TAXA * 4),
    ]
    memory_df = pl.DataFrame({
        "array":   [c[0] for c in _components],
        "dtype":   [c[1] for c in _components],
        "entries": [c[2] for c in _components],
        "bytes":   [c[3] for c in _components],
    })
    bytes_per_tree = sum(c[3] for c in _components)
    mb_per_tree    = bytes_per_tree / 1_000_000
    memory_df
    return bytes_per_tree, mb_per_tree


@app.cell
def _(N_TAXA, bytes_per_tree, mb_per_tree, mo):
    _l2_thresh = int(96_000_000       / bytes_per_tree)
    _dram_cap  = int(128_000_000_000  / bytes_per_tree)
    mo.md(
        f"""
        At **{N_TAXA:,} taxa**: **{mb_per_tree:.2f} MB** per tree.

        | Regime | Tree count | Bandwidth |
        |---|---|---|
        | L2-resident  | ≤ {_l2_thresh} trees | ~10 TB/s (GB10 L2) |
        | DRAM-resident | > {_l2_thresh} trees | 273 GB/s (LPDDR5x) |
        | Memory ceiling | {_dram_cap:,} trees | — |

        The L2 → DRAM transition is a ~40× bandwidth drop.
        At the design target (10,000 taxa, ~3.25 MB/tree) the L2 threshold is ≈ 29 trees.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Synthetic Ensemble Generation

    `make_base_tree` generates one Yule-process topology via
    `treesim.birth_death_tree` (birth rate = 1, death rate = 0).
    `make_forest` builds the ensemble by resampling branch lengths from
    Exp(1) for each tree, reusing the base topology.  For a throughput
    benchmark this is sufficient: LCA computation time depends on tree
    structure (n_nodes, tour length, sparse table size), not topology.
    """)
    return


@app.cell
def _(random, treesim):
    def make_base_tree(n_taxa, seed=0):
        rng = random.Random(seed)
        return treesim.birth_death_tree(
            birth_rate=1.0,
            death_rate=0.0,
            num_extant_tips=n_taxa,
            rng=rng,
        )

    return (make_base_tree,)


@app.cell
def _(Forest, N_GROUPS, N_TAXA, make_base_tree, quiet, random):
    def make_forest(n_trees, n_taxa=N_TAXA, n_groups=N_GROUPS, seed=0):
        rng  = random.Random(seed + 1)
        base = make_base_tree(n_taxa, seed)

        def one_newick():
            for node in base.preorder_node_iter():
                if node.edge_length is not None:
                    node.edge_length = max(1e-6, rng.expovariate(1.0))
            return base.as_string(schema="newick").strip()

        trees_per_group = max(1, n_trees // n_groups)
        groups = {
            f"g{i}": [one_newick() for _ in range(trees_per_group)]
            for i in range(n_groups)
        }
        with quiet():
            return Forest(groups)

    return (make_forest,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Backend Warm-Up

    The first call to a JIT-compiled backend triggers kernel compilation
    (30–60 s for Numba; similar for CUDA).  This cell runs a minimal query
    on a small forest to absorb that cost before the timed sweep.
    """)
    return


@app.cell
def _(BACKEND, Quartets, make_forest):
    _wf = make_forest(n_trees=4, n_taxa=100)
    _wq = Quartets.random(_wf, count=200, seed=1)
    _wf.quartet_topology(_wq, steiner=True, backend=BACKEND)
    print(f"warm-up complete  (backend={BACKEND!r})")
    return


@app.cell
def _(Quartets, REPEATS, time):
    def bench(forest, n_quartets, steiner=False, backend="best", repeats=REPEATS):
        """Return the minimum wall time (s) over `repeats` timed runs."""
        q = Quartets.random(forest, count=n_quartets, seed=42)
        forest.quartet_topology(q, steiner=steiner, backend=backend)   # warm up config
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            forest.quartet_topology(q, steiner=steiner, backend=backend)
            ts.append(time.perf_counter() - t0)
        return min(ts)

    return (bench,)


@app.cell
def _(mo):
    mo.md(r"""
    ## (n_trees, n_quartets) Parameter Sweep

    Total kernel thread-operations $= n_\text{quartets} \times n_\text{trees}$
    scales compute; only $n_\text{trees}$ scales memory footprint.
    GPU saturation on GB10 requires ≥ 196,608 active threads.

    Each (n_trees, n_quartets) point is timed twice: `steiner=False`
    (counts only) and `steiner=True` (counts + summed Steiner lengths).
    The ratio between them isolates the cost of one extra accumulation
    per thread — the unit we multiply to project min/max/variance overhead.
    """)
    return


@app.cell
def _(
    BACKEND,
    N_GROUPS,
    N_QUARTETS_SWEEP,
    N_TAXA,
    N_TREES_SWEEP,
    bench,
    make_forest,
    pl,
):
    _rows = []
    for _nt in N_TREES_SWEEP:
        _f = make_forest(_nt)
        for _nq in N_QUARTETS_SWEEP:
            _total = _nq * _f.n_trees
            _tc    = bench(_f, _nq, steiner=False, backend=BACKEND)
            _ts    = bench(_f, _nq, steiner=True,  backend=BACKEND)
            _rows.append({
                "n_taxa":                N_TAXA,
                "n_trees":               _f.n_trees,
                "n_groups":              N_GROUPS,
                "n_quartets":            _nq,
                "backend":               BACKEND,
                "total_threads":         _total,
                "counts_wall_s":         _tc,
                "steiner_wall_s":        _ts,
                "counts_threads_per_s":  _total / _tc,
                "steiner_threads_per_s": _total / _ts,
                "steiner_overhead":      _ts / _tc,
            })
    results = pl.DataFrame(_rows)
    results
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Steiner Overhead as a Proxy for Accumulation Cost

    `steiner_overhead = steiner_wall / counts_wall` measures the relative
    cost of one additional accumulation per (qi, ti) thread — a float
    multiply and a write (or atomic add on CUDA).

    Adding min, max, and sum_sq introduces three more accumulations of
    similar character, bringing the total to four above counts-only.
    Projecting linearly from the measured steiner ratio:

    $$\text{stats\_overhead} \approx 1 + 4 \times (\text{steiner\_overhead} - 1)$$

    This is a lower bound in the atomic-contention regime (CUDA, many trees
    per group) and a reasonable estimate in the arithmetic-bound regime
    (CPU-parallel).
    """)
    return


@app.cell
def _(pl, results):
    stats_projection = (
        results
        .with_columns(
            (1.0 + 4.0 * (pl.col("steiner_overhead") - 1.0))
            .alias("stats_overhead_projected")
        )
        .select([
            "n_trees",
            "n_quartets",
            "total_threads",
            "counts_threads_per_s",
            "steiner_overhead",
            "stats_overhead_projected",
        ])
        .sort(["n_trees", "n_quartets"])
    )
    stats_projection
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Performance Extremes

    **Best case** — L2-resident, saturated, dense overlap.
    n_trees ≤ L2 threshold, n_quartets ≥ 100K, all taxa present in all
    trees.  Throughput approaches the L2 bandwidth limit (~10 TB/s on GB10).
    Statistics additions are nearly free here; the bottleneck is L2
    bandwidth, not atomic serialisation.

    **Worst case A — atomic contention.**
    Few quartets, many trees per group (e.g., n_quartets = 200,
    n_trees = 10,000, n_groups = 2): ~5,000 threads serialise on each of
    the 200 × 2 × 3 = 1,200 output cells.  Adding min/max/variance
    multiplies the atomic queue length by ~4×.

    **Worst case B — DRAM-saturated.**
    n_trees beyond the L2 threshold; every RMQ fetch goes to 273 GB/s DRAM.
    Wall time scales nearly linearly with n_trees in this regime.
    Statistics additions add proportionally to the steiner overhead
    observed here.

    **Worst case C — sparse taxon overlap.**
    A large fraction of threads exit early at the `global_to_local` check,
    occupying thread slots without contributing to output.  Useful
    throughput (contributing thread-ops / s) drops by the absence rate.
    """)
    return


if __name__ == "__main__":
    app.run()
