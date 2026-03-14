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
