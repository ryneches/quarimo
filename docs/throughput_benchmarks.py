import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quartet Topology Throughput Benchmarks

    This notebook aggregates `pytest-benchmark` JSON output files produced by
    `tests/bench_throughput.py` and visualises how quartet topology throughput
    (quartets per second, **calculation phase only**) scales across two axes:

    1. **Fixed forest size** — 1 000 trees total, 1–10 tree groups.
       Shows how per-group accumulation overhead grows with group count.

    2. **Fixed group count** — 5 groups, 100–1 000 total trees.
       Shows how the kernel scales with forest depth.

    Each plot draws one line per backend.  All timing values come from the
    `⏱ t_calc` phase logged by `quartet_topology()` — not the total wall time —
    so device-load and host-to-device copy overhead are excluded.

    ## Generating benchmark data

    ```bash
    # Run from the project root (substitute a short machine label).
    pytest tests/bench_throughput.py -m "not large_scale" \\
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    ```

    Drop any number of `.json` files into `docs/benchmark_results/` and this
    notebook aggregates them automatically.  Results from different machines
    (different backends available) appear as separate line styles.
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

    return json, mo, mticker, pathlib, pl, plt, re


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

    def _is_throughput_bench(fullname: str) -> bool:
        return "TestThroughputFixed" in fullname

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
                        ei   = b.get("extra_info", {})
                        stats = b.get("stats", {})
                        rows.append(
                            {
                                "machine":           machine,
                                "file":              fpath.name,
                                "name":              b.get("name", ""),
                                "fullname":          fullname,
                                "sweep":             ei.get("sweep", ""),
                                "backend":           ei.get("backend", "unknown"),
                                "n_trees":           ei.get("n_trees"),
                                "n_groups":          ei.get("n_groups"),
                                "n_leaves":          ei.get("n_leaves"),
                                "n_quartets":        ei.get("n_quartets"),
                                "steiner":           ei.get("steiner", False),
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
        df = pl.DataFrame(rows)
    else:
        df = pl.DataFrame(
            schema={
                "machine": pl.Utf8,
                "file": pl.Utf8,
                "name": pl.Utf8,
                "fullname": pl.Utf8,
                "sweep": pl.Utf8,
                "backend": pl.Utf8,
                "n_trees": pl.Int64,
                "n_groups": pl.Int64,
                "n_leaves": pl.Int64,
                "n_quartets": pl.Int64,
                "steiner": pl.Boolean,
                "t_device_load": pl.Float64,
                "t_query_load": pl.Float64,
                "t_calc": pl.Float64,
                "t_retrieve": pl.Float64,
                "quartets_per_second": pl.Float64,
                "mean_s": pl.Float64,
                "stddev_s": pl.Float64,
            }
        )

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

    Each line shows how **quartets per second** (calculation phase only,
    `steiner=False`) changes as the 1 000-tree forest is split into 1–10
    groups.  Separate lines per backend; separate line styles per machine.

    Expected behaviour: throughput should be roughly constant (the total
    work — `n_quartets × n_trees` — is fixed), with a slight overhead
    increase for larger group counts due to the extra per-group accumulation
    in the kernel's inner loop.
    """)
    return


@app.cell
def _(df, machines, mo, plt):
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
                label = _BACKEND_LABELS.get(backend, backend)
                if len(machines) > 1:
                    label = f"{label}\n({machine.split('/')[0].strip()})"
                ax.plot(
                    xs, ys,
                    color=_BACKEND_COLORS.get(backend, "grey"),
                    linestyle=ls,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )

        ax.set_xlabel("Number of tree groups", fontsize=10)
        ax.set_ylabel("Quartets / second  (calc phase)", fontsize=10)
        ax.set_title(
            f"Throughput vs. group count  "
            f"(fixed forest: {df.filter(pl.col('sweep') == 'fixed_forest')['n_trees'].max()} trees, "
            f"steiner=False)",
            fontsize=10,
        )
        ax.set_xticks(sorted(sub["n_groups"].unique().to_list()))
        ax.legend(fontsize=8, framealpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    _plot_fixed_forest()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot 2 — Fixed group count, varying forest size

    Each line shows how **quartets per second** (calculation phase only,
    `steiner=False`) grows as the total number of trees increases, with the
    group count held fixed at 5.

    Expected behaviour: CPU-parallel throughput should increase roughly
    linearly until the available core count is saturated.  GPU backends
    should show steeper initial growth followed by a plateau at peak
    device occupancy.
    """)
    return


@app.cell
def _(df, machines, mo, plt):
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
                label = _BACKEND_LABELS.get(backend, backend)
                if len(machines) > 1:
                    label = f"{label}\n({machine.split('/')[0].strip()})"
                ax.plot(
                    xs, ys,
                    color=_BACKEND_COLORS.get(backend, "grey"),
                    linestyle=ls,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )

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
        ax.legend(fontsize=8, framealpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    _plot_fixed_groups()
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    def _summary_table():
        if len(df) == 0:
            return mo.md("*(no data)*")

        summary = (
            df.filter(pl.col("quartets_per_second").is_not_null())
            .group_by(["machine", "backend", "sweep", "steiner"])
            .agg(
                pl.col("quartets_per_second").mean().alias("mean_qps"),
                pl.col("quartets_per_second").min().alias("min_qps"),
                pl.col("quartets_per_second").max().alias("max_qps"),
                pl.col("n_trees").n_unique().alias("n_sizes"),
            )
            .sort(["sweep", "backend", "machine", "steiner"])
        )

        _header = (
            "| sweep | backend | machine | steiner "
            "| mean Q/s | min Q/s | max Q/s | n sizes |"
        )
        _sep = "|---|---|---|---|---:|---:|---:|---:|"
        _tbody = []
        for row in summary.to_dicts():
            steiner_str = "✓" if row["steiner"] else "✗"
            _tbody.append(
                f"| {row['sweep']} | {row['backend']} "
                f"| {row['machine'].split('/')[0].strip()} "
                f"| {steiner_str} "
                f"| {row['mean_qps']:,.0f} "
                f"| {row['min_qps']:,.0f} "
                f"| {row['max_qps']:,.0f} "
                f"| {row['n_sizes']} |"
            )
        return mo.md("\n".join([_header, _sep] + _tbody))

    mo.vstack([mo.md("## Summary table"), _summary_table()])
    return


if __name__ == "__main__":
    app.run()
