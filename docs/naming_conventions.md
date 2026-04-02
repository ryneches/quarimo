# Quarimo Naming Conventions

This document describes the naming conventions used throughout the quarimo
codebase. It exists so that contributors can understand existing code quickly
and design new features consistently. Rules R1–R8 apply to all new code; any
departures in existing code are legacy and should be corrected when the
surrounding code is touched.

---

## R1 — Loop index shorthand

Use two-character lowercase shorthand where the first letter names the entity
and `i` signals "index":

| Shorthand | Meaning |
|---|---|
| `qi` | Quartet index (position in the current batch) |
| `ti` | Tree index |
| `gi` | Group index |
| `pi` | Group-pair index |
| `li` | Paralog-genome index |
| `ci` | Copy-slot index within a paralog genome |

For generic nested loops use `i`, `j`, `k`. Never use `idx` as a standalone
variable name — it is too generic to be meaningful in context.

---

## R2 — Quartet taxon IDs at each layer

Quartet taxa have different names depending on how close to the metal the code
is:

| Layer | Names | Meaning |
|---|---|---|
| User-facing API | `a, b, c, d` | Named taxa (strings or ints) |
| Kernel internals — global IDs | `t0, t1, t2, t3` | Sorted global taxon IDs (int32) |
| Kernel internals — local IDs | `ln0, ln1, ln2, ln3` | Local leaf node IDs within one tree (int32) |

Never use `n0..n3` — the `n` prefix reads as "node", which collides with the
codebase's extensive use of "node" for tree-internal node IDs. `t` = taxon.

---

## R3 — First-occurrence abbreviation

The LCA algorithm requires the **first** occurrence of each leaf in the Euler
tour. This qualifier must be preserved in names at every level:

| Scope | Name |
|---|---|
| `Tree` class attribute | `first_occurrence` |
| `Forest` flat CSR array | `all_first_occ` |
| Kernel local variables | `fo0, fo1, fo2, fo3` |

The word `occ` alone is not used. `fo` is always short for `first_occ`.

---

## R4 — CSR pair naming

All CSR structures consist of an offsets array and a data array, named:

```
[concept]_offsets   int32 or int64, shape (n + 1,)   — start/end indices
[concept]_nodes     int32, shape (total,)              — flat data
```

The canonical example in the codebase is `polytomy_offsets` /
`polytomy_nodes`. New CSR structures must follow this pattern.

**Documented exception:** The tree-structure arrays (`all_euler_tour`,
`all_root_distance`, etc.) have their own `*_offsets` vectors
(`tour_offsets`, `node_offsets`, etc.) but no matching `*_nodes` sibling —
because the data lives in separately-named semantic arrays whose names convey
the content. This exception is not extended to new structures.

**`node_offsets` note:** Despite the name, `node_offsets` is a CSR offset into
the per-node data arrays (`all_first_occ`, `all_root_distance`), not into a
node-adjacency structure. The name is established and consistent; read it as
"offsets for node-data arrays".

---

## R5 — Kernel function suffixes

Kernel functions are suffixed by backend:

| Suffix | Backend |
|---|---|
| `_nb` | Numba JIT (CPU parallel) |
| `_cuda` | CUDA (Numba CUDA device functions) |
| `_mlx` | Metal Shading Language (Apple Silicon) |
| *(none)* | Pure Python fallback — lives in `_forest.py` |

The suffix applies to both top-level kernels and helper device functions
(e.g., `_rmq_csr_nb`, `_resolve_quartet_cuda`). Consistent suffixing makes it
immediately clear which backend a function belongs to and which other functions
are its counterparts.

---

## R6 — Offset/stride variable names in kernel call chains

When a tree's CSR offsets and sparse-table stride are extracted for use inside
a kernel, they are always named:

| Name | Meaning |
|---|---|
| `node_base` | CSR offset into per-node arrays (`all_first_occ`, `all_root_distance`) |
| `tour_base` | CSR offset into tour arrays (`all_euler_tour`, `all_euler_depth`) |
| `sp_base` | CSR offset into the sparse table (`all_sparse_table`) |
| `lg_base` | CSR offset into the log2 table (`all_log2_table`) |
| `sp_stride` | Sparse table column stride (= Euler tour length for this tree) |

These five names are used uniformly across parameter lists, return tuples, and
local variables throughout all three kernel backends. They are the same names
used in `_rmq_csr_nb`'s parameter list — there is no further abbreviation at
call sites.

The historical abbreviations `nb, tb, sb, lb, tw` should not be used in new
code and are being removed from existing code.

---

## R7 — Output accumulator arguments

Kernel output arrays always carry the `_out` suffix:

```python
counts_out          int32 [n_quartets, n_groups, 4]
steiner_out         float64 [n_quartets, n_groups, 4]
steiner_min_out     float64 [n_quartets, n_groups, 4]
steiner_max_out     float64 [n_quartets, n_groups, 4]
steiner_sum_sq_out  float64 [n_quartets, n_groups, 4]
```

The delta kernel writes signed updates into `counts_out` in-place; no separate
`delta_counts_out` array is needed.

---

## R8 — Dimension scalars

All scalar dimension arguments use the `n_` prefix:

```
n_quartets       n_trees        n_groups        n_global_taxa
n_paralog_genomes               n_copies        n_affected
n_seed           n_pairs
```

Never use a bare `N` or `count` as a dimension argument name.

---

## Additional documented conventions (not rules, but recorded patterns)

**`r0, r1, r2` — four-point-condition pair-sum scores**

In kernel code, `r0`, `r1`, `r2` are the three pair-sums used to determine
quartet topology:

```
r0 = rd[LCA(t0,t1)] + rd[LCA(t2,t3)]   →  topology 0: (t0,t1)|(t2,t3)
r1 = rd[LCA(t0,t2)] + rd[LCA(t1,t3)]   →  topology 1: (t0,t2)|(t1,t3)
r2 = rd[LCA(t0,t3)] + rd[LCA(t1,t2)]   →  topology 2: (t0,t3)|(t1,t2)
```

`r_winner` = max(r0, r1, r2). `topo` ∈ {0, 1, 2, 3} where 3 = unresolved
(polytomy). These names appear identically in the CPU, CUDA, and MLX backends.

**`tree_to_group_idx` vs `group_to_tree_indices`**

`tree_to_group_idx` is a dense int32 array of length `n_trees`: one group
index per tree. `group_to_tree_indices` is a dict mapping group names to lists
of tree indices. The asymmetry of `idx` (singular) vs `indices` (plural)
reflects the asymmetry of the data.

**`ForestKernelData` / `QuartetKernelArgs`**

`ForestKernelData` holds the packed, read-only tree arrays (the infrastructure).
`QuartetKernelArgs` holds the quartet specification (what to compute). The two
are deliberately separate: the forest data is built once at construction time;
the quartet args vary per call.

**`all_` prefix on flat-packed tree arrays**

Arrays named `all_*` are flat CSR-packed versions of per-tree data. They are
always indexed via the corresponding `*_offsets` and `*_base` variables. The
prefix signals "all trees, concatenated".

---

## Notation, structure and formatting (N1–N9)

These rules govern how mathematics and technical content are written across
different contexts: Markdown documentation, Python code, docstrings, and
notebooks. The core principle is that the rendering environment determines
the appropriate notation: use the richest notation the renderer supports,
and no more.

---

### N1 — LaTeX math in Markdown

In contexts that will be processed as Markdown — `.md` files and notebook
Markdown cells — equations must be formatted using LaTeX math mode. Write
inline quantities as `$...$` and standalone equations as `$$...$$`. The
target renderer is MathJax, so standard LaTeX math commands are available.

---

### N2 — No Unicode math in documentation

Unicode mathematical symbols (e.g., `×`, `≥`, `Σ`, `δ̂`) must not appear in
documentation source files. Use LaTeX math mode instead (N1). Typography and
symbol choices should be governed by the stylesheet and renderer, not
embedded in the source.

This rule does not apply to plain-text table separators, arrows used as
prose connectives (`→`), or similar layout characters that carry no
mathematical meaning.

---

### N3 — Unicode symbols in code

Unicode symbols and special characters may be used in code only where they
serve a clear purpose that cannot be served by plain ASCII, and only in
output-facing contexts: logging messages, notebook cell output, and formatted
results. Examples that are acceptable:

- `δ̂=` and `p̂=` in a formatted string passed to `mo.md()`.
- `↑` / `↓` in a console progress line to indicate direction of a trend.

Unicode must not appear in variable names, import aliases, or any context
where it would require a reader to type or search for the character.

---

### N4 — Mathematics in docstrings and comments

Complex mathematical notation should generally not appear in Python
docstrings or inline comments. Prefer clear English explanations that
describe the algorithm in terms of the variable names involved, formatted
`like_this`.

When mathematical notation is clearly more appropriate than English — for
example, when defining a non-trivial closed-form expression — the docstring
should reference the corresponding section in the `docs/` directory rather
than reproducing the notation inline:

```python
def quartet_score(p_hat, n_qsupp, n_trees):
    """
    Compute the delta_hat score for a single branch.

    The scoring formula is defined in
    :doc:`/docs/quartet_consensus_scoring`.
    The key parameters are ``p_hat`` (estimated disagreement rate),
    ``n_qsupp`` (number of straddling quartets), and ``n_trees``.
    """
```

The corresponding Markdown document should include a reciprocal link back
to the API reference so that both directions resolve correctly after
processing with mkdocs and mkdocstrings.

---

### N5 — Inline code formatting in Markdown prose

Variable names, function names, class names, and parameter names appearing
in prose — in `.md` files, notebook Markdown cells, or docstrings — must be
formatted with backticks: `` `like_this` ``. This includes names that are
also used in LaTeX math (write both: "the resolved fraction
`total_resolved / n_relevant` approximates $1 - \hat p$").

---

### N6 — LaTeX–Python name correspondence

When LaTeX math refers to a quantity that has a Python variable name, the
LaTeX should follow these conventions so that the correspondence is
unambiguous:

| Python name pattern | LaTeX convention | Example |
|---|---|---|
| `n_foo` (dimension scalar) | `$n_\text{foo}$` | `n_trees` → $n_\text{trees}$ |
| `foo_hat` (estimate) | `$\hat\foo$` or `$\hat{f}$` | `delta_hat` → $\hat\delta$, `p_hat` → $\hat p$ |
| `foo_bar` (compound) | `$\text{foo\_bar}$` if no natural symbol | `total_resolved` → $\text{total\_resolved}$ |

Multi-character subscripts always use `\text{}`. Single-character subscripts
may use the bare subscript form.

---

### N7 — Display vs inline math

- **Inline** `$...$`: quantities and short expressions embedded in running
  prose ("the score is $\hat\delta = N_\text{qsupp} \cdot n_\text{trees}
  \cdot (2\hat p - 1)$").
- **Display** `$$...$$`: standalone equations that are the primary subject of
  a paragraph, or any expression too long to read inline.
- Numbered equations are not used. Cross-references to equations are made
  by quoting the surrounding prose, not by equation number.

---

### N8 — Notebook cell organisation

A `hide_code=True` Markdown cell introduces a concept, section, or
design decision. The immediately following visible code cell implements it.
The Markdown cell explains *why*; the code explains *what*.

Markdown cells must not restate what the code already says. A Markdown cell
that says "the following cell defines `_score_active_branches`" adds no
information — write instead about the design rationale, the algorithm, or
the known limitations.

---

### N9 — Docstring style

All docstrings use **NumPy style** parameter and returns sections:

```python
def foo(x, y):
    """
    One-line summary.

    Longer description if needed.

    Parameters
    ----------
    x : int
        Description of x.
    y : float
        Description of y.

    Returns
    -------
    float
        Description of the return value.
    """
```

Type annotations in the function signature take precedence over types
repeated in the docstring; when annotations are present, the type field in
the docstring may be omitted.
