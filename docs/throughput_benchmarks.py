import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quartet Topology Throughput Benchmarks

    This notebook aggregates `pytest-benchmark` JSON output files produced by
    `tests/bench_throughput.py` and visualises how quartet topology throughput
    (quartets per second, **calculation phase only**) scales across three axes.
    All timing comes from the `⏱ t_calc` phase logged by `quartet_topology()`,
    so device-load and host-to-device copy overhead are excluded.

    | Plot | Sweep | Fixed | Varying | What it measures |
    |---|---|---|---|---|
    | 1 | `fixed_forest`      | total trees | group count | per-group accumulation overhead |
    | 2 | `fixed_groups`      | group count | total trees | kernel throughput vs. forest depth |
    | 3 | `fixed_trees`       | tree count  | leaves/tree | O(1) LCA hypothesis; cache-spill threshold (random trees) |
    | 4 | `correlated_trees`  | tree count  | leaves/tree | same as Plot 3 but on a correlated NNI ensemble |

    Each plot draws one line per (backend, machine) combination.  Color encodes
    backend; line style encodes machine.  Legend entries include hostname, CPU
    architecture, and GPU name where applicable.

    The **summary table** at the bottom shows implied memory bandwidth (GB/s)
    estimated from quartet throughput × estimated bytes read per (quartet, tree)
    pair, alongside the theoretical peak bandwidth for the detected hardware.

    ## Generating benchmark data

    ```bash
    # Run all sweeps from the project root:
    pytest tests/bench_throughput.py \
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json

    # Or run individual sweeps:
    pytest tests/bench_throughput.py::TestThroughputFixedForest \
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    pytest tests/bench_throughput.py::TestThroughputFixedGroups \
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    pytest tests/bench_throughput.py::TestThroughputFixedTrees \
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    pytest tests/bench_throughput.py::TestThroughputCorrelatedTrees \
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    ```

    Drop any number of `.json` files into `docs/benchmark_results/` and this
    notebook aggregates them automatically.  Results from different machines
    appear as separate line styles so hardware comparisons are readable on a
    single plot.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import pathlib
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import polars as pl

    return json, mo, pathlib, pl, plt, re


@app.cell
def _(mo):
    json_dir_input = mo.ui.text(
        value="docs/benchmark_results",
        label="Directory containing benchmark JSON files",
        full_width=True,
    )
    json_dir_input
    return (json_dir_input,)


@app.cell
def _(json, json_dir_input, mo, pathlib, pl, re):
    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _iter_json_objects(path: pathlib.Path):
        """Yield every top-level JSON object from a file.

        pytest-benchmark can append multiple objects to the same file when run
        repeatedly, producing technically invalid JSON.  This parser handles it.
        """
        content = path.read_text()
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content):
            chunk = content[idx:].lstrip()
            if not chunk:
                break
            skip = len(content[idx:]) - len(chunk)
            try:
                obj, end = decoder.raw_decode(chunk)
                yield obj
                idx += skip + end
            except json.JSONDecodeError:
                break

    def _machine_label(mi: dict) -> str:
        """Short human-readable label for a machine_info dict."""
        cpu = mi.get("cpu", {}) if isinstance(mi.get("cpu"), dict) else {}
        brand = cpu.get("brand_raw", "")
        brand = re.sub(r"\s+@.*$", "", brand)
        brand = re.sub(r"Intel\(R\) Core\(TM\) ", "", brand)
        brand = re.sub(r"\s+CPU$", "", brand, flags=re.I)
        node = mi.get("node", "unknown")
        arch = mi.get("machine", "")
        system = mi.get("system", "")
        parts = [node]
        if brand:
            parts.append(brand)
        elif arch:
            parts.append(f"{system}/{arch}")
        return " / ".join(parts)

    # ── Hardware performance tables ────────────────────────────────────────────
    # Both tables use longest-match-first substring matching on gpu_name.
    # Sources: NVIDIA product pages, Apple silicon specs, Anandtech/WikiChip.

    # Peak memory bandwidth (GB/s)
    _HW_PEAK_BW = [  # (substring, peak_GB_s)
        ("A100 SXM",    2_000),  ("A100",         2_000),
        ("H100 SXM",    3_350),  ("H100 PCIe",    2_000),  ("H100",   2_000),
        ("RTX 4090",    1_008),  ("RTX 3090",       936),
        ("RTX 3080 Ti",   912),  ("RTX 3080",       760),
        ("RTX 3070 Ti",   608),  ("RTX 3070",       448),
        ("V100",           900),
        ("M4 Ultra",       819),  ("M4 Max",         546),  ("M4",     273),
        ("M3 Ultra",       800),  ("M3 Max",         400),  ("M3",     150),
        ("M2 Ultra",       800),  ("M2 Max",         400),  ("M2",     100),
        ("M1 Ultra",       800),  ("M1 Max",         400),  ("M1",      68),
    ]

    # Peak compute throughput (TOPS) for integer/FP32 shader ops (non-tensor-core).
    # NVIDIA: INT32 ≈ FP32 throughput on Ampere/Hopper; Apple: FP32 GPU throughput.
    _HW_PEAK_TOPS = [  # (substring, peak_TOPS)
        ("A100 SXM",    19.5),  ("A100",          19.5),
        ("H100 SXM",    30.0),  ("H100 PCIe",     24.0),  ("H100",   24.0),
        ("RTX 4090",    21.0),  ("RTX 3090",      17.9),
        ("RTX 3080 Ti", 16.1),  ("RTX 3080",      12.5),
        ("RTX 3070 Ti", 10.7),  ("RTX 3070",       7.8),
        ("V100",         7.8),
        ("M4 Ultra",    32.0),  ("M4 Max",        16.0),  ("M4",      8.0),
        ("M3 Ultra",    28.0),  ("M3 Max",        14.2),  ("M3",      7.1),
        ("M2 Ultra",    27.2),  ("M2 Max",        13.6),  ("M2",      6.8),
        ("M1 Ultra",    20.0),  ("M1 Max",        10.0),  ("M1",      5.0),
    ]

    def _lookup_peak_bw(gpu_name) -> float | None:
        if not gpu_name:
            return None
        g = str(gpu_name)
        for substr, bw in _HW_PEAK_BW:
            if substr.lower() in g.lower():
                return float(bw)
        return None

    def _lookup_peak_tops(gpu_name) -> float | None:
        if not gpu_name:
            return None
        g = str(gpu_name)
        for substr, tops in _HW_PEAK_TOPS:
            if substr.lower() in g.lower():
                return float(tops)
        return None

    def _is_throughput_bench(fullname: str) -> bool:
        return "TestThroughputFixed" in fullname or "TestThroughputCorrelated" in fullname

    # ── Load and flatten all JSON files ──────────────────────────────────────

    d = pathlib.Path(json_dir_input.value)
    rows = []
    warnings_list = []
    n_files = 0

    if d.exists():
        for fpath in sorted(d.glob("*.json")):
            try:
                for obj in _iter_json_objects(fpath):
                    machine = _machine_label(obj.get("machine_info", {}))
                    for b in obj.get("benchmarks", []):
                        fullname = b.get("fullname", b.get("name", ""))
                        if not _is_throughput_bench(fullname):
                            continue
                        ei    = b.get("extra_info", {})
                        mi    = obj.get("machine_info", {})
                        stats = b.get("stats", {})
                        rows.append(
                            {
                                "machine":           machine,
                                "arch":              mi.get("machine", ""),
                                "file":              fpath.name,
                                "name":              b.get("name", ""),
                                "fullname":          fullname,
                                "sweep":             ei.get("sweep", ""),
                                "backend":           ei.get("backend", "unknown"),
                                "gpu_name":          ei.get("gpu_name"),
                                "n_trees":           ei.get("n_trees"),
                                "n_groups":          ei.get("n_groups"),
                                "n_leaves":          ei.get("n_leaves"),
                                "n_quartets":        ei.get("n_quartets"),
                                "steiner":           ei.get("steiner", False),
                                "correlated":        ei.get("correlated", False),
                                "t_device_load":     ei.get("t_device_load"),
                                "t_query_load":      ei.get("t_query_load"),
                                "t_calc":            ei.get("t_calc"),
                                "t_retrieve":        ei.get("t_retrieve"),
                                "quartets_per_second": ei.get("quartets_per_second"),
                                "mean_s":            stats.get("mean"),
                                "stddev_s":          stats.get("stddev"),
                            }
                        )
                n_files += 1
            except Exception as exc:
                warnings_list.append(f"`{fpath.name}`: {exc}")

    # ── Deduplicate: last write wins for (machine, name) pairs ───────────────
    seen: dict[tuple, dict] = {}
    for _r in rows:
        seen[(_r["machine"], _r["name"])] = _r
    rows = list(seen.values())

    # ── Build polars DataFrame ────────────────────────────────────────────────
    if rows:
        df = pl.DataFrame(rows, schema_overrides={"gpu_name": pl.Utf8, "arch": pl.Utf8})
    else:
        df = pl.DataFrame(
            schema={
                "machine": pl.Utf8,
                "arch": pl.Utf8,
                "file": pl.Utf8,
                "name": pl.Utf8,
                "fullname": pl.Utf8,
                "sweep": pl.Utf8,
                "backend": pl.Utf8,
                "gpu_name": pl.Utf8,
                "n_trees": pl.Int64,
                "n_groups": pl.Int64,
                "n_leaves": pl.Int64,
                "n_quartets": pl.Int64,
                "steiner": pl.Boolean,
                "correlated": pl.Boolean,
                "t_device_load": pl.Float64,
                "t_query_load": pl.Float64,
                "t_calc": pl.Float64,
                "t_retrieve": pl.Float64,
                "quartets_per_second": pl.Float64,
                "mean_s": pl.Float64,
                "stddev_s": pl.Float64,
            }
        )

    # ── Roofline derived columns ──────────────────────────────────────────────
    # Per-(quartet, tree) cost estimates for the roofline model.
    #
    # _BYTES_PER_PAIR: estimated bytes read from device memory assuming no cache:
    #   global_to_local (16 B), all_first_occ (16 B), all_root_distance (64 B),
    #   all_euler_depth (32 B), all_sparse_table (32 B), all_log2_table (16 B),
    #   polytomy CSR reads (8 B) ≈ 184 bytes.
    #
    # _OPS_PER_PAIR: estimated integer/FP arithmetic ops (array index arithmetic,
    #   comparisons, LCA range-min computations, four-point sums) ≈ 48 ops.
    #   These are non-memory ALU operations; memory reads are counted above.
    _BYTES_PER_PAIR = 184
    _OPS_PER_PAIR   = 48

    if len(df) > 0:
        df = df.with_columns([
            (
                pl.col("n_quartets").cast(pl.Float64)
                * pl.col("n_trees").cast(pl.Float64)
                * pl.lit(_BYTES_PER_PAIR)
                / pl.col("t_calc")
                / pl.lit(1e9)
            ).alias("implied_bw_GBs"),
            # implied_compute_Tops: ALU throughput required if _OPS_PER_PAIR ops
            # per pair and all ops are executed (no cache / pipelining benefit).
            (
                pl.col("quartets_per_second")
                * pl.col("n_trees").cast(pl.Float64)
                * pl.lit(_OPS_PER_PAIR)
                / pl.lit(1e12)
            ).alias("implied_compute_Tops"),
            pl.col("gpu_name").map_elements(
                _lookup_peak_bw, return_dtype=pl.Float64
            ).alias("peak_bw_GBs"),
            pl.col("gpu_name").map_elements(
                _lookup_peak_tops, return_dtype=pl.Float64
            ).alias("peak_tops_Tops"),
        ])
        df = df.with_columns([
            (pl.col("implied_bw_GBs") / pl.col("peak_bw_GBs") * 100.0)
            .alias("bw_util_pct"),
            (pl.col("implied_compute_Tops") / pl.col("peak_tops_Tops") * 100.0)
            .alias("compute_util_pct"),
        ])
    else:
        df = df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("implied_bw_GBs"),
            pl.lit(None, dtype=pl.Float64).alias("implied_compute_Tops"),
            pl.lit(None, dtype=pl.Float64).alias("peak_bw_GBs"),
            pl.lit(None, dtype=pl.Float64).alias("peak_tops_Tops"),
            pl.lit(None, dtype=pl.Float64).alias("bw_util_pct"),
            pl.lit(None, dtype=pl.Float64).alias("compute_util_pct"),
        ])

    machines = sorted(df["machine"].unique().to_list()) if len(df) > 0 else []

    # ── Status callout ────────────────────────────────────────────────────────
    if not d.exists():
        _status = mo.callout(
            mo.md(
                f"Directory `{d}` not found.  "
                "Create it and run the benchmarks as shown above."
            ),
            kind="warn",
        )
    elif n_files == 0:
        _status = mo.callout(
            mo.md(
                f"No `.json` files found in `{d}`.  "
                "Run the benchmarks and save the output there."
            ),
            kind="warn",
        )
    else:
        _lines = [
            f"Loaded **{len(df)}** benchmark entries from **{n_files}** file(s) "
            f"across **{len(machines)}** machine(s).",
        ]
        if warnings_list:
            _lines += ["", "**Warnings:**"] + [f"- {w}" for w in warnings_list]
        _status = mo.callout(
            mo.md("\n".join(_lines)),
            kind="success" if not warnings_list else "warn",
        )

    _status
    return df, machines


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot 1 — Fixed forest size, varying group count

    Total tree count is fixed; group count varies.  Each line shows
    **quartets per second** (`steiner=False`, calc phase) as one large forest
    is progressively sub-divided into more groups.

    Because the total work (`n_quartets × n_trees`) is constant across the
    x-axis, a flat line means the kernel has zero per-group overhead.  Any
    throughput decline with increasing group count reflects accumulation
    overhead in the kernel's per-group inner loop.  The magnitude of this
    decline is expected to differ between GPU and CPU backends.
    """)
    return


@app.cell
def _(df, machines, mo, pl, plt):
    def _plot_fixed_forest():
        _BACKEND_ORDER = ["python", "cpu-parallel", "cuda", "mlx"]
        _BACKEND_COLORS = {
            "python":       "#4C72B0",
            "cpu-parallel": "#DD8452",
            "cuda":         "#55A868",
            "mlx":          "#C44E52",
        }
        _BACKEND_LABELS = {
            "python":       "Python",
            "cpu-parallel": "CPU-parallel (Numba)",
            "cuda":         "CUDA",
            "mlx":          "MLX (Metal)",
        }
        _MACHINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

        if len(df) == 0:
            return mo.callout(mo.md("No benchmark data available."), kind="neutral")

        sub = (
            df.filter(
                (pl.col("sweep") == "fixed_forest")
                & (pl.col("steiner") == False)  # noqa: E712
                & pl.col("quartets_per_second").is_not_null()
            )
            .sort("n_groups")
        )

        if len(sub) == 0:
            return mo.callout(
                mo.md(
                    "No `fixed_forest` / `steiner=False` rows found.  "
                    "Run `TestThroughputFixedForest` and save the JSON."
                ),
                kind="neutral",
            )

        backends_present = [b for b in _BACKEND_ORDER if b in sub["backend"].to_list()]

        fig, ax = plt.subplots(figsize=(8, 5))
        n_lines = 0

        for mi, machine in enumerate(machines):
            ls = _MACHINE_STYLES[mi % len(_MACHINE_STYLES)]
            for backend in backends_present:
                seg = (
                    sub.filter(
                        (pl.col("machine") == machine)
                        & (pl.col("backend") == backend)
                    )
                    .sort("n_groups")
                )
                if len(seg) == 0:
                    continue
                xs = seg["n_groups"].to_list()
                ys = seg["quartets_per_second"].to_list()
                hostname = machine.split("/")[0].strip()
                arch_v = seg["arch"].drop_nulls().head(1).to_list()
                arch = arch_v[0] if arch_v else ""
                gpu_v = seg["gpu_name"].drop_nulls().head(1).to_list()
                gpu = gpu_v[0] if gpu_v else None
                hw_parts = [hostname]
                if arch:
                    hw_parts.append(arch)
                if gpu:
                    hw_parts.append(gpu)
                label = (
                    f"{_BACKEND_LABELS.get(backend, backend)}\n"
                    f"({' / '.join(hw_parts)})"
                )
                ax.plot(
                    xs, ys,
                    color=_BACKEND_COLORS.get(backend, "grey"),
                    linestyle=ls,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )
                n_lines += 1

        ax.set_xlabel("Number of tree groups", fontsize=10)
        ax.set_ylabel("Quartets / second  (calc phase)", fontsize=10)
        ax.set_title(
            f"Throughput vs. group count  "
            f"(fixed forest: {df.filter(pl.col('sweep') == 'fixed_forest')['n_trees'].max()} trees, "
            f"steiner=False)",
            fontsize=10,
        )
        ax.set_xticks(sorted(sub["n_groups"].unique().to_list()))
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=max(1, min(n_lines, 3)),
            fontsize=8,
            framealpha=0.8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.semilogy()
        fig.tight_layout()
        return fig

    _plot_fixed_forest()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot 2 — Fixed group count, varying forest size

    Group count is fixed; total tree count varies.  Each line shows
    **quartets per second** (`steiner=False`, calc phase) as the forest grows
    deeper (more trees per group), with the group structure held constant.

    All sweep points start above the GPU L2-cache escape threshold so
    throughput reflects HBM bandwidth rather than cache capacity.
    CPU-parallel throughput should grow roughly linearly until the available
    core count is saturated.  GPU backends should show steeper initial growth
    followed by a plateau at peak device occupancy.
    """)
    return


@app.cell
def _(df, machines, mo, pl, plt):
    def _plot_fixed_groups():
        _BACKEND_ORDER = ["python", "cpu-parallel", "cuda", "mlx"]
        _BACKEND_COLORS = {
            "python":       "#4C72B0",
            "cpu-parallel": "#DD8452",
            "cuda":         "#55A868",
            "mlx":          "#C44E52",
        }
        _BACKEND_LABELS = {
            "python":       "Python",
            "cpu-parallel": "CPU-parallel (Numba)",
            "cuda":         "CUDA",
            "mlx":          "MLX (Metal)",
        }
        _MACHINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

        if len(df) == 0:
            return mo.callout(mo.md("No benchmark data available."), kind="neutral")

        sub = (
            df.filter(
                (pl.col("sweep") == "fixed_groups")
                & (pl.col("steiner") == False)  # noqa: E712
                & pl.col("quartets_per_second").is_not_null()
            )
            .sort("n_trees")
        )

        if len(sub) == 0:
            return mo.callout(
                mo.md(
                    "No `fixed_groups` / `steiner=False` rows found.  "
                    "Run `TestThroughputFixedGroups` and save the JSON."
                ),
                kind="neutral",
            )

        backends_present = [b for b in _BACKEND_ORDER if b in sub["backend"].to_list()]

        fig, ax = plt.subplots(figsize=(8, 5))
        n_lines = 0

        for mi, machine in enumerate(machines):
            ls = _MACHINE_STYLES[mi % len(_MACHINE_STYLES)]
            for backend in backends_present:
                seg = (
                    sub.filter(
                        (pl.col("machine") == machine)
                        & (pl.col("backend") == backend)
                    )
                    .sort("n_trees")
                )
                if len(seg) == 0:
                    continue
                xs = seg["n_trees"].to_list()
                ys = seg["quartets_per_second"].to_list()
                hostname = machine.split("/")[0].strip()
                arch_v = seg["arch"].drop_nulls().head(1).to_list()
                arch = arch_v[0] if arch_v else ""
                gpu_v = seg["gpu_name"].drop_nulls().head(1).to_list()
                gpu = gpu_v[0] if gpu_v else None
                hw_parts = [hostname]
                if arch:
                    hw_parts.append(arch)
                if gpu:
                    hw_parts.append(gpu)
                label = (
                    f"{_BACKEND_LABELS.get(backend, backend)}\n"
                    f"({' / '.join(hw_parts)})"
                )
                ax.plot(
                    xs, ys,
                    color=_BACKEND_COLORS.get(backend, "grey"),
                    linestyle=ls,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )
                n_lines += 1

        ax.set_xlabel("Total number of trees", fontsize=10)
        ax.set_ylabel("Quartets / second  (calc phase)", fontsize=10)
        ax.set_title(
            f"Throughput vs. forest size  "
            f"(fixed groups: {df.filter(pl.col('sweep') == 'fixed_groups')['n_groups'].max()}, "
            f"steiner=False)",
            fontsize=10,
        )
        ax.set_xticks(sorted(sub["n_trees"].unique().to_list()))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=max(1, min(n_lines, 3)),
            fontsize=8,
            framealpha=0.8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.semilogy()
        fig.tight_layout()
        return fig

    _plot_fixed_groups()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot 3 — Fixed tree count, varying leaf count

    Tree count is fixed; leaf count per tree varies.  Each line shows
    **quartets per second** (`steiner=False`, calc phase) as trees grow
    larger, with the total number of trees held constant.

    The O(1) LCA algorithm (sparse table RMQ) predicts **flat throughput**
    across this sweep — each quartet topology query requires exactly four
    sparse-table lookups regardless of n_leaves.  A throughput drop at larger
    leaf counts indicates that the sparse table has grown large enough to spill
    out of the GPU's L2 cache, making the kernel memory-latency-bound rather
    than compute-bound.  Per-tree sparse table size is O(n log n) in n_leaves,
    so cache spill is expected well before the largest sizes in this sweep.
    Note that forest construction time (dominated by sparse-table build) grows
    with leaf count but is excluded from all timing via `t_device_load`.
    """)
    return


@app.cell
def _(df, machines, mo, pl, plt):
    def _plot_fixed_trees():
        _BACKEND_ORDER = ["python", "cpu-parallel", "cuda", "mlx"]
        _BACKEND_COLORS = {
            "python":       "#4C72B0",
            "cpu-parallel": "#DD8452",
            "cuda":         "#55A868",
            "mlx":          "#C44E52",
        }
        _BACKEND_LABELS = {
            "python":       "Python",
            "cpu-parallel": "CPU-parallel (Numba)",
            "cuda":         "CUDA",
            "mlx":          "MLX (Metal)",
        }
        _MACHINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

        if len(df) == 0:
            return mo.callout(mo.md("No benchmark data available."), kind="neutral")

        sub = (
            df.filter(
                (pl.col("sweep") == "fixed_trees")
                & (pl.col("steiner") == False)  # noqa: E712
                & pl.col("quartets_per_second").is_not_null()
            )
            .sort("n_leaves")
        )

        if len(sub) == 0:
            return mo.callout(
                mo.md(
                    "No `fixed_trees` / `steiner=False` rows found.  "
                    "Run `TestThroughputFixedTrees` and save the JSON."
                ),
                kind="neutral",
            )

        backends_present = [b for b in _BACKEND_ORDER if b in sub["backend"].to_list()]

        fig, ax = plt.subplots(figsize=(8, 5))
        n_lines = 0

        for mi, machine in enumerate(machines):
            ls = _MACHINE_STYLES[mi % len(_MACHINE_STYLES)]
            for backend in backends_present:
                seg = (
                    sub.filter(
                        (pl.col("machine") == machine)
                        & (pl.col("backend") == backend)
                    )
                    .sort("n_leaves")
                )
                if len(seg) == 0:
                    continue
                xs = seg["n_leaves"].to_list()
                ys = seg["quartets_per_second"].to_list()
                hostname = machine.split("/")[0].strip()
                arch_v = seg["arch"].drop_nulls().head(1).to_list()
                arch = arch_v[0] if arch_v else ""
                gpu_v = seg["gpu_name"].drop_nulls().head(1).to_list()
                gpu = gpu_v[0] if gpu_v else None
                hw_parts = [hostname]
                if arch:
                    hw_parts.append(arch)
                if gpu:
                    hw_parts.append(gpu)
                label = (
                    f"{_BACKEND_LABELS.get(backend, backend)}\n"
                    f"({' / '.join(hw_parts)})"
                )
                ax.plot(
                    xs, ys,
                    color=_BACKEND_COLORS.get(backend, "grey"),
                    linestyle=ls,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )
                n_lines += 1

        n_trees_val = df.filter(pl.col("sweep") == "fixed_trees")["n_trees"].max()
        ax.set_xlabel("Leaves per tree", fontsize=10)
        ax.set_ylabel("Quartets / second  (calc phase)", fontsize=10)
        ax.set_title(
            f"Throughput vs. leaf count  "
            f"(fixed trees: {n_trees_val}, steiner=False)",
            fontsize=10,
        )
        ax.set_xticks(sorted(sub["n_leaves"].unique().to_list()))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=max(1, min(n_lines, 3)),
            fontsize=8,
            framealpha=0.8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.semilogy()
        fig.tight_layout()
        return fig

    _plot_fixed_trees()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot 4 — Correlated trees (bootstrap NNI ensemble)

    Same axis as Plot 3 but trees are drawn from a five-template NNI ensemble
    rather than independently shuffled leaf orders.  All trees share a common
    balanced-binary reference topology; only a few NNI edits per template
    perturb it.  DFS order is therefore correlated across trees.

    Compare these curves against Plot 3 to see whether tree topology correlation
    affects the cache-spill profile: if the correlated ensemble shows higher
    throughput at large leaf counts, the kernel benefits from inter-tree locality
    in the random access patterns.
    """)
    return


@app.cell
def _(df, machines, mo, pl, plt):
    def _plot_correlated_trees():
        _BACKEND_ORDER = ["python", "cpu-parallel", "cuda", "mlx"]
        _BACKEND_COLORS = {
            "python":       "#4C72B0",
            "cpu-parallel": "#DD8452",
            "cuda":         "#55A868",
            "mlx":          "#C44E52",
        }
        _BACKEND_LABELS = {
            "python":       "Python",
            "cpu-parallel": "CPU-parallel (Numba)",
            "cuda":         "CUDA",
            "mlx":          "MLX (Metal)",
        }
        _MACHINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

        if len(df) == 0:
            return mo.callout(mo.md("No benchmark data available."), kind="neutral")

        sub = (
            df.filter(
                (pl.col("sweep") == "correlated_trees")
                & (pl.col("steiner") == False)  # noqa: E712
                & pl.col("quartets_per_second").is_not_null()
            )
            .sort("n_leaves")
        )

        if len(sub) == 0:
            return mo.callout(
                mo.md(
                    "No `correlated_trees` / `steiner=False` rows found.  "
                    "Run `TestThroughputCorrelatedTrees` and save the JSON."
                ),
                kind="neutral",
            )

        backends_present = [b for b in _BACKEND_ORDER if b in sub["backend"].to_list()]

        fig, ax = plt.subplots(figsize=(8, 5))
        n_lines = 0

        for mi, machine in enumerate(machines):
            ls = _MACHINE_STYLES[mi % len(_MACHINE_STYLES)]
            for backend in backends_present:
                seg = (
                    sub.filter(
                        (pl.col("machine") == machine)
                        & (pl.col("backend") == backend)
                    )
                    .sort("n_leaves")
                )
                if len(seg) == 0:
                    continue
                xs = seg["n_leaves"].to_list()
                ys = seg["quartets_per_second"].to_list()
                hostname = machine.split("/")[0].strip()
                arch_v = seg["arch"].drop_nulls().head(1).to_list()
                arch = arch_v[0] if arch_v else ""
                gpu_v = seg["gpu_name"].drop_nulls().head(1).to_list()
                gpu = gpu_v[0] if gpu_v else None
                hw_parts = [hostname]
                if arch:
                    hw_parts.append(arch)
                if gpu:
                    hw_parts.append(gpu)
                label = (
                    f"{_BACKEND_LABELS.get(backend, backend)}\n"
                    f"({' / '.join(hw_parts)})"
                )
                ax.plot(
                    xs, ys,
                    color=_BACKEND_COLORS.get(backend, "grey"),
                    linestyle=ls,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )
                n_lines += 1

        n_trees_val = df.filter(pl.col("sweep") == "correlated_trees")["n_trees"].max()
        ax.set_xlabel("Leaves per tree", fontsize=10)
        ax.set_ylabel("Quartets / second  (calc phase)", fontsize=10)
        ax.set_title(
            f"Throughput vs. leaf count — correlated NNI ensemble  "
            f"(fixed trees: {n_trees_val}, steiner=False)",
            fontsize=10,
        )
        ax.set_xticks(sorted(sub["n_leaves"].unique().to_list()))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=max(1, min(n_lines, 3)),
            fontsize=8,
            framealpha=0.8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.semilogy()
        fig.tight_layout()
        return fig

    _plot_correlated_trees()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Hardware specifications

Used for roofline-model bandwidth and compute utilization estimates.

| Architecture | Chip | Peak BW (GB/s) | Peak compute (TOPS) | Notes |
|---|---|---:|---:|---|
| NVIDIA H100 SXM | H100 SXM5 80 GB | 3,350 | 30.0 | HBM3; INT32/FP32 non-tensor-core |
| NVIDIA H100 PCIe | H100 PCIe 80 GB | 2,000 | 24.0 | HBM2e |
| NVIDIA A100 SXM | A100 SXM4 80 GB | 2,000 | 19.5 | HBM2e |
| NVIDIA A100 PCIe | A100 PCIe 80 GB | 2,000 | 19.5 | HBM2e |
| NVIDIA V100 SXM | V100 SXM2 32 GB | 900 | 7.8 | HBM2 |
| NVIDIA RTX 4090 | GeForce RTX 4090 | 1,008 | 21.0 | GDDR6X |
| NVIDIA RTX 3090 | GeForce RTX 3090 | 936 | 17.9 | GDDR6X |
| NVIDIA RTX 3080 Ti | GeForce RTX 3080 Ti | 912 | 16.1 | GDDR6X |
| NVIDIA RTX 3080 | GeForce RTX 3080 | 760 | 12.5 | GDDR6X |
| NVIDIA RTX 3070 Ti | GeForce RTX 3070 Ti | 608 | 10.7 | GDDR6X |
| NVIDIA RTX 3070 | GeForce RTX 3070 | 448 | 7.8 | GDDR6 |
| Apple M4 Ultra | M4 Ultra | 819 | 32.0 | unified memory; GPU FP32 |
| Apple M4 Max | M4 Max | 546 | 16.0 | unified memory |
| Apple M4 | M4 | 273 | 8.0 | unified memory |
| Apple M3 Ultra | M3 Ultra | 800 | 28.0 | unified memory |
| Apple M3 Max | M3 Max | 400 | 14.2 | unified memory |
| Apple M3 | M3 | 150 | 7.1 | unified memory |
| Apple M2 Ultra | M2 Ultra | 800 | 27.2 | unified memory |
| Apple M2 Max | M2 Max | 400 | 13.6 | unified memory |
| Apple M2 | M2 | 100 | 6.8 | unified memory |
| Apple M1 Ultra | M1 Ultra | 800 | 20.0 | unified memory |
| Apple M1 Max | M1 Max | 400 | 10.0 | unified memory |
| Apple M1 | M1 | 68 | 5.0 | unified memory |

**Roofline model** — two per-(quartet, tree) cost constants drive the analysis:

- **Memory**: 184 B (global\_to\_local 16 B, all\_first\_occ 16 B, all\_root\_distance 64 B,
  all\_euler\_depth 32 B, all\_sparse\_table 32 B, all\_log2\_table 16 B, polytomy CSR 8 B),
  assuming no L2/L1 cache reuse.
- **Compute**: 48 ops (LCA range-min queries, four-point-condition sums and comparisons,
  array index arithmetic), assuming all ops hit ALUs without pipelining gaps.

The arithmetic intensity is 48 ops / 184 B ≈ 0.26 FLOPS/B.
The roofline ridge point for A100 is 19.5 TOPS / 2000 GB/s ≈ 9.75 FLOPS/B.
Since 0.26 < 9.75, the quartet kernel is **theoretically memory-bandwidth-bound** on all
listed hardware.  In practice, L2/L1 cache effects move the effective operating point;
see the summary table footnote.
    """)
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    def _summary_table():
        if len(df) == 0:
            return mo.md("*(no data)*")

        summary = (
            df.filter(pl.col("quartets_per_second").is_not_null())
            .group_by(["machine", "backend", "sweep", "steiner", "correlated"])
            .agg(
                pl.col("quartets_per_second").mean().alias("mean_qps"),
                pl.col("quartets_per_second").min().alias("min_qps"),
                pl.col("quartets_per_second").max().alias("max_qps"),
                pl.col("n_trees").n_unique().alias("n_sizes"),
                pl.col("implied_bw_GBs").mean().alias("mean_bw_GBs"),
                pl.col("peak_bw_GBs").first().alias("peak_bw_GBs"),
                pl.col("implied_compute_Tops").mean().alias("mean_compute_Tops"),
                pl.col("peak_tops_Tops").first().alias("peak_tops_Tops"),
            )
            # Derive both utilization metrics from the aggregated means.  This
            # ensures the displayed "impl. BW" and "BW util%" columns are always
            # consistent with each other (and likewise for compute).
            .with_columns([
                (pl.col("mean_bw_GBs") / pl.col("peak_bw_GBs") * 100.0)
                .alias("_bw_pct"),
                (pl.col("mean_compute_Tops") / pl.col("peak_tops_Tops") * 100.0)
                .alias("_compute_pct"),
            ])
            .sort(["sweep", "backend", "machine", "steiner", "correlated"])
        )

        def _fmt_util(pct, flag_cache: bool = False) -> str:
            """Format a utilization % with optional cache-limited flag."""
            if pct is None:
                return "—"
            if flag_cache and pct > 100.0:
                return f">100%‡"
            return f"{min(pct, 100.0):.1f}%"

        def _bound_label(bw_pct, compute_pct) -> str:
            """Roofline bound classification."""
            if bw_pct is None or compute_pct is None:
                return "—"
            if bw_pct > 100.0:
                # Kernel exceeds the no-cache BW limit — L2/L1 cache is helping;
                # can't determine true bound without profiling.
                return "cache†"
            if bw_pct >= compute_pct:
                return "BW"
            return "compute"

        _header = (
            "| sweep | backend | machine | steiner | trees "
            "| mean Q/s | min Q/s | max Q/s | n sizes "
            "| BW (GB/s) | peak BW | BW util% "
            "| compute (TOPS) | peak TOPS | compute% | bound |"
        )
        _sep = (
            "|---|---|---|---|---|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---|"
        )
        _tbody = []
        _has_cache_note = False
        for row in summary.to_dicts():
            steiner_str = "✓" if row["steiner"] else "✗"
            trees_str = "correlated" if row["correlated"] else "random"
            bw_pct = row["_bw_pct"]
            compute_pct = row["_compute_pct"]

            bw_impl = (
                f"{row['mean_bw_GBs']:.1f}" if row["mean_bw_GBs"] is not None else "—"
            )
            bw_peak = (
                f"{row['peak_bw_GBs']:.0f}" if row["peak_bw_GBs"] is not None else "—"
            )
            bw_util = _fmt_util(bw_pct, flag_cache=True)
            if bw_pct is not None and bw_pct > 100.0:
                _has_cache_note = True

            compute_impl = (
                f"{row['mean_compute_Tops']:.2f}"
                if row["mean_compute_Tops"] is not None else "—"
            )
            compute_peak = (
                f"{row['peak_tops_Tops']:.1f}"
                if row["peak_tops_Tops"] is not None else "—"
            )
            compute_util = _fmt_util(compute_pct)
            bound = _bound_label(bw_pct, compute_pct)

            _tbody.append(
                f"| {row['sweep']} | {row['backend']} "
                f"| {row['machine'].split('/')[0].strip()} "
                f"| {steiner_str} "
                f"| {trees_str} "
                f"| {row['mean_qps']:,.0f} "
                f"| {row['min_qps']:,.0f} "
                f"| {row['max_qps']:,.0f} "
                f"| {row['n_sizes']} "
                f"| {bw_impl} "
                f"| {bw_peak} "
                f"| {bw_util} "
                f"| {compute_impl} "
                f"| {compute_peak} "
                f"| {compute_util} "
                f"| {bound} |"
            )

        footnotes = []
        if _has_cache_note:
            footnotes.append(
                "‡ The no-cache bandwidth estimate exceeds hardware peak.  "
                "L2/L1 cache is absorbing a significant fraction of memory traffic; "
                "the kernel is cache/latency-limited in this configuration.  "
                "Actual DRAM bandwidth utilization is < 100%."
            )
        footnotes.append(
            "† 'cache' bound = no-cache BW estimate > hardware peak BW "
            "(L2/L1 cache-limited).  "
            "'BW' = bandwidth-limited in the roofline model.  "
            "'compute' = compute-limited.  "
            "Arithmetic intensity for this kernel ≈ 0.26 FLOPS/B; "
            "all listed hardware has a ridge point > 5 FLOPS/B, so the kernel "
            "is theoretically BW-bound in uncached operation."
        )
        footnote_md = "\n\n" + "  \n".join(footnotes) if footnotes else ""
        return mo.md("\n".join([_header, _sep] + _tbody) + footnote_md)

    mo.vstack([mo.md("## Summary table"), _summary_table()])
    return


if __name__ == "__main__":
    app.run()
