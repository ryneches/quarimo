import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Paralog Benchmark Analysis

    This notebook aggregates `pytest-benchmark` JSON output files produced by
    `tests/bench_paralog.py` and visualises two aspects of the paralog resolver:

    1. **Performance matrix** (`TestBenchResolveParalogs`) — throughput in
       (quartet × tree) pairs per second across the full condition space
       *(paralog frequency × copies per genome × background discordance)*.
       One panel per backend; multiple machines are overlaid for direct
       hardware comparison.

    2. **Optimizer convergence** (`TestStressResolveParalogs`) — per-sweep
       QED trajectories for each *(paralog frequency × copies per genome)*
       cell, showing how quickly and how much the coordinate-descent
       assignment optimizer improves mean QED.

    ## Generating benchmark data

    ```bash
    # Run from the project root.  Substitute $(hostname) or a short arch label.
    pytest tests/bench_paralog.py -m "not large_scale" \
        --benchmark-json=docs/benchmark_results/paralog_$(hostname)_$(date +%Y%m%d).json

    # Apple Silicon / CI runner — same command, different host name
    pytest tests/bench_paralog.py -m "not large_scale" \
        --benchmark-json=docs/benchmark_results/paralog_apple_$(date +%Y%m%d).json

    # Large-scale tier (requires Numba)
    pytest tests/bench_paralog.py -m "large_scale" \
        --benchmark-json=docs/benchmark_results/paralog_large_$(hostname)_$(date +%Y%m%d).json
    ```

    Drop any number of `.json` files into `docs/benchmark_results/` and
    this notebook will aggregate them automatically.
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
    import numpy as np  # noqa: F401 — used in plotting cells

    return json, mo, mticker, pathlib, plt, re


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
def _(json, json_dir_input, mo, pathlib, re):
    def _iter_json_objects(path: pathlib.Path):
        """Yield every top-level JSON object from a file.

        pytest-benchmark can append multiple objects to the same file when
        run repeatedly, producing technically invalid JSON (two top-level
        objects). This parser handles that gracefully.
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
        # Strip clock speed and trailing "CPU" from Intel/AMD brand strings
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
        return "\n".join(parts)

    def _classify(fullname: str) -> str:
        """Return 'bench', 'stress', or 'other' for a benchmark fullname."""
        if "TestBenchResolveParalogs" in fullname:
            return "bench"
        if "TestStressResolveParalogs" in fullname:
            return "stress"
        return "other"

    # ── Load all JSON files ────────────────────────────────────────────────── #

    d = pathlib.Path(json_dir_input.value)

    bench_rows = []   # TestBenchResolveParalogs entries
    stress_rows = []  # TestStressResolveParalogs entries
    warnings = []
    n_files = 0

    if d.exists():
        files = sorted(d.glob("*.json"))
        for fpath in files:
            try:
                for obj in _iter_json_objects(fpath):
                    machine = _machine_label(obj.get("machine_info", {}))
                    for b in obj.get("benchmarks", []):
                        fullname = b.get("fullname", b.get("name", ""))
                        kind = _classify(fullname)
                        if kind == "other":
                            continue
                        ei = b.get("extra_info", {})
                        stats = b.get("stats", {})
                        row = dict(
                            machine=machine,
                            file=fpath.name,
                            name=b.get("name", ""),
                            fullname=fullname,
                            backend=ei.get("backend", "unknown"),
                            npg=ei.get("n_paralog_genomes"),
                            cpg=ei.get("copies_per_genome"),
                            discordance=ei.get("discordance", ""),
                            n_trees=ei.get("n_trees"),
                            n_quartets=ei.get("n_quartets"),
                            cells=ei.get("cells"),
                            mean_s=stats.get("mean"),
                            qed_initial=ei.get("qed_initial"),
                            qed_final=ei.get("qed_final"),
                            qed_history=ei.get("qed_history", []),
                            n_optimizer_iters=ei.get("n_optimizer_iters"),
                        )
                        if kind == "bench":
                            bench_rows.append(row)
                        else:
                            stress_rows.append(row)
                n_files += 1
            except Exception as exc:
                warnings.append(f"`{fpath.name}`: {exc}")

    # Deduplicate: if the same (machine, name) appears in multiple files
    # (appended runs), keep the most recent occurrence.
    def _dedup(rows):
        seen = {}
        for r in rows:
            key = (r["machine"], r["name"])
            seen[key] = r   # last write wins
        return list(seen.values())

    bench_rows = _dedup(bench_rows)
    stress_rows = _dedup(stress_rows)

    machines = sorted({r["machine"] for r in bench_rows + stress_rows})

    # ── Status callout ─────────────────────────────────────────────────────── #

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
            f"Loaded **{len(bench_rows)}** bench + **{len(stress_rows)}** stress "
            f"entries from **{n_files}** file(s) across "
            f"**{len(machines)}** machine(s).",
        ]
        if warnings:
            _lines += ["", "**Warnings:**"] + [f"- {w}" for w in warnings]
        _status = mo.callout(mo.md("\n".join(_lines)), kind="success" if not warnings else "warn")

    _status
    return bench_rows, stress_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance matrix

    Throughput (million quartet×tree pairs per second) across the full
    condition space.  **Rows** = architecture (machine); **columns** =
    backend, ordered slowest → fastest (Python, CPU-parallel, CUDA, MLX).
    Blank panels indicate backends not available on that machine.

    Within each panel, the x-axis groups the three background-discordance
    modes; bar colour encodes paralog frequency (*npg*); hatching
    distinguishes **k = 2** (solid) from **k = 3** (hatched).  All panels
    share the same Y-axis scale so throughput is directly comparable across
    backends and architectures.
    """)
    return


@app.cell
def _(bench_rows, mo, mticker, plt):
    def _():
        # ── Constants ─────────────────────────────────────────────────────────── #

        _DISC_ORDER  = ["concordant", "mixed", "discordant"]
        _DISC_LABELS = {"concordant": "Concordant", "mixed": "Mixed", "discordant": "Discordant"}
        _NPG_ORDER   = [1, 3, 5]
        _CPG_ORDER   = [2, 3]
        _CPG_HATCH   = {2: "", 3: "///"}
        _NPG_COLORS  = ["#4C72B0", "#DD8452", "#55A868"]   # blue / orange / green

        # Backends shown left→right (reverse priority: slowest first)
        _BACKEND_ORDER  = ["python", "cpu-parallel", "cuda", "mlx"]
        _BACKEND_LABELS = {
            "python":       "Python",
            "cpu-parallel": "CPU-parallel\n(Numba)",
            "cuda":         "CUDA",
            "mlx":          "MLX (Metal)",
        }

        def _throughput(row):
            if row["mean_s"] and row["mean_s"] > 0 and row["cells"]:
                return row["cells"] / row["mean_s"] / 1e6
            return None

        if not bench_rows:
            return mo.callout(
                mo.md("No `TestBenchResolveParalogs` data available yet."),
                kind="neutral",
            )

        # ── Grid dimensions ───────────────────────────────────────────────────── #

        _machines = sorted({r["machine"] for r in bench_rows})
        _backends = [b for b in _BACKEND_ORDER
                     if any(r["backend"] == b for r in bench_rows)]

        _n_rows = len(_machines)
        _n_cols = len(_backends)

        # Pre-index data: {(machine, backend): {(disc, npg, cpg): throughput}}
        _cell_data = {}
        for _r in bench_rows:
            _tp = _throughput(_r)
            if _tp is None:
                continue
            _key = (_r["machine"], _r["backend"])
            _cell_data.setdefault(_key, {})[
                (_r["discordance"], _r["npg"], _r["cpg"])
            ] = _tp

        # Global Y maximum for shared axis
        _all_tp = [v for d in _cell_data.values() for v in d.values()]
        _y_max = max(_all_tp) * 1.15 if _all_tp else 1.0

        # ── Figure ────────────────────────────────────────────────────────────── #

        _fig, _axes = plt.subplots(
            _n_rows, _n_cols,
            figsize=(4.5 * _n_cols, 3.8 * _n_rows),
            sharey=True,
            squeeze=False,
        )

        _n_npg   = len(_NPG_ORDER)
        _n_cpg   = len(_CPG_ORDER)
        _gw      = 0.8                          # total group width
        _bar_w   = _gw / (_n_npg * _n_cpg)

        for _ri, machine in enumerate(_machines):
            for _ci, backend in enumerate(_backends):
                ax = _axes[_ri, _ci]
                _data = _cell_data.get((machine, backend), {})

                if not _data:
                    ax.set_visible(False)
                    continue

                # Draw bars
                for _di, disc in enumerate(_DISC_ORDER):
                    for _ni, npg in enumerate(_NPG_ORDER):
                        for _ki, cpg in enumerate(_CPG_ORDER):
                            _bar_idx = _ni * _n_cpg + _ki
                            _x = _di - _gw / 2 + _bar_w * (_bar_idx + 0.5)
                            _val = _data.get((disc, npg, cpg))
                            if _val is not None:
                                ax.bar(
                                    _x, _val,
                                    width=_bar_w * 0.9,
                                    color=_NPG_COLORS[_ni],
                                    hatch=_CPG_HATCH[cpg],
                                    edgecolor="white",
                                    linewidth=0.5,
                                )

                ax.set_xticks(range(len(_DISC_ORDER)))
                ax.set_xticklabels(
                    [_DISC_LABELS[d] for d in _DISC_ORDER], fontsize=8
                )
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                ax.set_ylim(0, _y_max)
                ax.tick_params(labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", linestyle="--", alpha=0.4)

                # Y-axis tick labels — leftmost column only
                if _ci > 0:
                    ax.tick_params(labelleft=False)

                # Row label (machine name) — leftmost column only
                if _ci == 0:
                    ax.set_ylabel(
                        machine.replace("\n", "  "),
                        fontsize=8, labelpad=6,
                    )

        # Shared Y-axis label
        _fig.supylabel("Throughput (M pairs / s)", fontsize=9, x=0.01)

        # Legend: placed above the grid, centered
        from matplotlib.patches import Patch as _Patch
        _legend_handles = (
            [_Patch(facecolor=_NPG_COLORS[i], label=f"npg = {npg}")
             for i, npg in enumerate(_NPG_ORDER)]
            + [_Patch(facecolor="grey", hatch="",    label="k = 2 (solid)"),
               _Patch(facecolor="grey", hatch="///", label="k = 3 (hatched)")]
        )
        _fig.legend(
            handles=_legend_handles,
            fontsize=8,
            ncol=len(_legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            framealpha=0.8,
        )

        _fig.tight_layout()

        # Column headers via fig.text at figure-level coordinates.
        # Must come AFTER tight_layout() so that axis positions are finalised.
        # Using ax.set_title() on the first visible row would cause labels to
        # "fall down" whenever the top row has a blank cell for that column.
        _y_top = max(
            _axes[_ri, _ci].get_position().y1
            for _ri in range(_n_rows)
            for _ci in range(_n_cols)
            if _axes[_ri, _ci].get_visible()
        )
        for _ci, backend in enumerate(_backends):
            _xs = [
                (_axes[_ri, _ci].get_position().x0
                 + _axes[_ri, _ci].get_position().x1) / 2
                for _ri in range(_n_rows)
                if _axes[_ri, _ci].get_visible()
            ]
            if _xs:
                _fig.text(
                    _xs[0], _y_top + 0.01,
                    _BACKEND_LABELS.get(backend, backend),
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                )

        _fig.suptitle(
            "resolve_paralogs() throughput — performance matrix",
            fontsize=11, y=_y_top + 0.07,
        )
        return _fig


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimizer convergence

    Each panel shows the QED trajectory for one *(paralog frequency × copies
    per genome)* cell.  The x-axis is the sweep index (0 = initial assignment
    from DFS order; each subsequent point is after one full coordinate-descent
    sweep).  The y-axis is mean QED across all quartet × group-pair
    combinations.

    A flat trajectory (converged after sweep 0) means the initial DFS
    assignment was already at a local optimum for this random ensemble.
    An upward step indicates the optimizer found a copy-slot permutation that
    increased concordance.
    """)
    return


@app.cell
def _(mo, plt, stress_rows):
    def _():
        _NPG_ORDER_S = [1, 3, 5]
        _CPG_ORDER_S = [2, 3]
        _NPG_LABELS = {1: "npg = 1  (pf_low ≈ 6 %)", 3: "npg = 3  (pf_med ≈ 17 %)", 5: "npg = 5  (pf_hi ≈ 25 %)"}
        _CPG_LABELS = {2: "k = 2", 3: "k = 3"}

        # Color: one per machine; line style: cpg
        _MARKERS = {2: "o", 3: "s"}

        if not stress_rows:
            _qed_fig = mo.callout(
                mo.md("No `TestStressResolveParalogs` data available yet."),
                kind="neutral",
            )
        else:
            _machines_s = sorted({r["machine"] for r in stress_rows})
            # One color per machine
            _cmap = plt.get_cmap("tab10")
            _mach_color = {m: _cmap(i) for i, m in enumerate(_machines_s)}

            _n_rows = len(_NPG_ORDER_S)
            _n_cols = len(_CPG_ORDER_S)
            _fig_s, _axes_s = plt.subplots(
                _n_rows, _n_cols,
                figsize=(5 * _n_cols, 3.5 * _n_rows),
                sharex=False,
                squeeze=False,
            )

            for _ri, npg in enumerate(_NPG_ORDER_S):
                for _ci, cpg in enumerate(_CPG_ORDER_S):
                    ax = _axes_s[_ri, _ci]

                    _cell_rows = [
                        r for r in stress_rows
                        if r["npg"] == npg and r["cpg"] == cpg
                    ]

                    if not _cell_rows:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, color="grey")
                    else:
                        for r in _cell_rows:
                            hist = r["qed_history"]
                            if not hist:
                                continue
                            xs = list(range(len(hist)))
                            color = _mach_color.get(r["machine"], "black")
                            ax.plot(
                                xs, hist,
                                color=color,
                                linewidth=1.8,
                                marker=_MARKERS[cpg],
                                markersize=5,
                                label=r["machine"].split("\n")[0],
                            )
                            # Mark initial and final QED
                            ax.axhline(hist[0], color=color, linewidth=0.5,
                                       linestyle=":", alpha=0.5)

                    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.3)
                    ax.set_xlabel("Sweep", fontsize=8)
                    ax.set_ylabel("Mean QED", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(axis="y", linestyle="--", alpha=0.3)
                    ax.set_title(
                        f"{_NPG_LABELS[npg]},  {_CPG_LABELS[cpg]}",
                        fontsize=8,
                    )

                    # x-axis: integer ticks only
                    _max_sweeps = max(
                        (len(r["qed_history"]) - 1 for r in _cell_rows if r["qed_history"]),
                        default=0,
                    )
                    ax.set_xticks(range(_max_sweeps + 1))
                    ax.xaxis.set_tick_params(labelsize=7)

                    if len(_machines_s) > 1 and _ri == 0 and _ci == _n_cols - 1:
                        ax.legend(fontsize=6, loc="lower right", framealpha=0.7)

            _fig_s.suptitle(
                "resolve_paralogs() optimizer convergence — QED history",
                fontsize=11, y=1.01,
            )
            _fig_s.tight_layout()
            _qed_fig = _fig_s
        return _qed_fig


    _()
    return


@app.cell(hide_code=True)
def _(mo, stress_rows):
    def _():
        # Summary table of convergence statistics
        if not stress_rows:
            _tbl = mo.md("*(no data)*")
        else:
            _header = "| npg | k | machine | sweeps | converged | QED initial | QED final | Δ QED |"
            _sep    = "|-----|---|---------|--------|-----------|-------------|-----------|-------|"
            _tbody = []
            for r in sorted(stress_rows, key=lambda x: (x["npg"], x["cpg"], x["machine"])):
                hist = r["qed_history"]
                if not hist:
                    continue
                delta = hist[-1] - hist[0]
                arrow = "↑" if delta > 1e-6 else ("=" if abs(delta) < 1e-6 else "↓")
                _tbody.append(
                    f"| {r['npg']} | {r['cpg']} | {r['machine'].split(chr(10))[0]} "
                    f"| {r['n_optimizer_iters']} | {'✓' if r.get('n_optimizer_iters','') != '' else '?'} "
                    f"| {hist[0]:+.4f} | {hist[-1]:+.4f} | {arrow}{abs(delta):.4f} |"
                )
            _tbl = mo.md("\n".join([_header, _sep] + _tbody))
        return mo.vstack([mo.md("### Convergence summary"), _tbl])


    _()
    return


if __name__ == "__main__":
    app.run()
