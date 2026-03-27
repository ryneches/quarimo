"""
_consensus_graph.py
===================
Mutable bipartition graph for the prune-only consensus algorithm.

The core data structure is a DFS-interval tree: every subtree's leaves
occupy a contiguous position range ``[start, end]`` in the DFS leaf
ordering fixed at construction.  This makes bipartition representation
O(1) and keeps interval invariants valid across arbitrary pruning
operations — pruning re-parents grandchildren, but never moves leaves,
so every node's ``[start, end]`` remains correct.

Quartet-support counting and sampling use cross-component pairs: two
taxa drawn from *different* sub-components on each side of a branch.
These are the maximally discriminating quartets for a given bipartition
because they test the branch against its immediate neighbours rather
than against sub-branches it already contains.

Usage
-----
::

    from quarimo import Forest
    from quarimo._consensus_graph import ConsensusGraph

    forest = Forest(bootstrap_newicks)
    graph  = ConsensusGraph(reference_newick, forest.global_names)

    for bid in sorted(graph.active_branch_ids):
        print(bid, graph.n_qsupp(bid))

    dirty = graph.prune(some_branch_id)
    print(graph.to_newick())
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ======================================================================
# Internal data records
# ======================================================================

@dataclass
class _Node:
    node_id: int
    taxon_gid: int      # global taxon ID for leaves; -1 for internal nodes
    parent: int         # parent node_id; -1 for the root
    children: list      # list[int] — mutable; updated in-place by prune()
    start: int = -1     # DFS leaf interval start (inclusive)
    end: int = -1       # DFS leaf interval end (inclusive)


@dataclass
class _Edge:
    edge_id: int
    parent: int         # parent node_id
    child: int          # child node_id
    length: float
    is_internal: bool   # True iff child is an internal node
    active: bool = True


# ======================================================================
# NEWICK parser (tokeniser-based)
# ======================================================================

# Tokeniser: quoted strings (with '' escaping), punctuation, unquoted tokens.
# Bracket comments are stripped by _strip_newick_comments before tokenising.
_NEWICK_TOK = re.compile(r"'(?:[^']|'')*'|[(),:;]|[^(),:;'\s]+")


def _strip_newick_comments(s: str) -> str:
    """
    Remove bracketed NHX/BEAST/FigTree comments ``[...]`` from a NEWICK string.

    Uses a depth counter to correctly handle nested brackets such as
    ``[outer[inner]]``.
    """
    result: list = []
    depth = 0
    for ch in s:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        elif depth == 0:
            result.append(ch)
    return ''.join(result)


def _parse_newick(s: str) -> dict:
    """
    Parse a NEWICK string into a nested dict structure.

    Returns a dict ``{'name': str, 'length': float, 'children': [...]}``.
    Leaf nodes have empty ``children`` lists.  Internal-node labels
    (e.g., bootstrap supports) are captured in ``name`` but not used by
    ``ConsensusGraph``.

    Handles: standard NEWICK, branch lengths, bootstrap labels,
    single-quoted taxon names, and bracketed comments.

    Uses a regex tokeniser rather than character-by-character parsing;
    this is ~5× faster for large NEWICK strings (> 10 000 characters).
    """
    s = _strip_newick_comments(s).strip().rstrip(';').strip()
    tokens = _NEWICK_TOK.findall(s)
    pos = [0]

    def tok() -> str:
        return tokens[pos[0]] if pos[0] < len(tokens) else ''

    def advance() -> str:
        t = tokens[pos[0]]; pos[0] += 1; return t

    def maybe(ch: str) -> bool:
        if tok() == ch:
            pos[0] += 1; return True
        return False

    def read_label() -> str:
        t = tok()
        if t in ('', '(', ')', ',', ':', ';'):
            return ''
        t = advance()
        if t.startswith("'") and t.endswith("'"):
            return t[1:-1].replace("''", "'")
        return t

    def read_length() -> float:
        if tok() != ':':
            return 0.0
        pos[0] += 1
        try:
            return float(advance())
        except (ValueError, IndexError):
            return 0.0

    def parse_node() -> dict:
        children = []
        if tok() == '(':
            pos[0] += 1
            children.append(parse_node())
            while tok() == ',':
                pos[0] += 1
                children.append(parse_node())
            maybe(')')
        name   = read_label()
        length = read_length()
        return {'name': name, 'children': children, 'length': length}

    return parse_node()


# ======================================================================
# ConsensusGraph
# ======================================================================

class ConsensusGraph:
    """
    Mutable bipartition graph for a single reference tree.

    Constructed from a NEWICK string and the global taxon namespace of
    the bootstrap ``Forest`` that will be queried.  All leaf labels in
    the NEWICK must appear in ``global_names``.

    The DFS leaf positions assigned at construction are permanent.  Every
    subtree occupies a contiguous ``[start, end]`` interval throughout
    the lifetime of the object, even after pruning, because pruning
    re-parents subtrees without moving any leaves.

    Parameters
    ----------
    newick : str
        Reference tree in NEWICK format.  The leaf label set must exactly
        match ``global_names``.
    global_names : sequence of str
        Taxon names from ``Forest.global_names``.  Determines the mapping
        from leaf label to global taxon ID (position in this list).
    zero_eps : float
        Internal branches with length ≤ ``zero_eps`` are collapsed on
        construction (they represent pre-existing polytomies in the
        reference tree).

    Attributes
    ----------
    active_branch_ids : set of int
        Edge IDs of all currently active internal branches.
    """

    def __init__(
        self,
        newick: str,
        global_names: Sequence[str],
        zero_eps: float = 1e-12,
    ) -> None:
        self._global_names: list = list(global_names)
        self._name_to_gid: dict = {name: i for i, name in enumerate(global_names)}

        # Node and edge arrays — indexed by node_id / edge_id.
        self._nodes: list = []
        self._edges: list = []

        # Leaf GID at each DFS position (fixed at construction).
        self._leaf_gids: list = []

        # Maps child node_id -> edge_id (the edge connecting it to its parent).
        # Active nodes only; deleted when a node is pruned out.
        self._node_to_edge: dict = {}

        parsed = _parse_newick(newick)
        self._root: int = self._build_graph(parsed, root_parent=-1)
        self._n_leaves: int = len(self._leaf_gids)

        # Numpy view of leaf GIDs — used for vectorised quartet sampling.
        self._leaf_gids_arr: np.ndarray = np.array(self._leaf_gids, dtype=np.int32)

        if self._n_leaves < 4:
            raise ValueError(
                f"ConsensusGraph requires at least 4 taxa; got {self._n_leaves}"
            )

        # Validate: leaf set must exactly match the forest namespace.
        leaf_names = {self._global_names[g] for g in self._leaf_gids}
        forest_names = set(global_names)
        if leaf_names != forest_names:
            extra   = leaf_names - forest_names
            missing = forest_names - leaf_names
            parts = []
            if extra:
                parts.append(f"labels not in forest namespace: {sorted(extra)}")
            if missing:
                parts.append(f"forest taxa absent from reference tree: {sorted(missing)}")
            raise ValueError(
                "Reference tree leaf set does not match forest namespace.  "
                + "  ".join(parts)
            )

        # Collapse pre-existing zero-length internal branches (input polytomies).
        # Iterate over a snapshot of edge IDs so mutations during the loop are safe.
        if zero_eps > 0:
            for eid in range(len(self._edges)):
                e = self._edges[eid]
                if e.active and e.is_internal and e.length <= zero_eps:
                    self._prune_internal(eid)

        # Normalise: if root has exactly 2 children after the collapse, the
        # rooted representation encodes a basal bifurcation that does not exist
        # in the unrooted tree.  Re-root at an internal child so the root has
        # ≥ 3 children; this ensures n_qsupp > 0 for root-adjacent branches.
        self._normalize_root()

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    def _build_graph(self, root_dict: dict, root_parent: int) -> int:
        """
        Iterative pre-order / post-order DFS to build ``_nodes`` and
        ``_edges`` from a parsed NEWICK dict.

        Leaf positions in ``_leaf_gids`` are assigned in pre-order
        (left-to-right DFS traversal).  Parent intervals
        ``[node.start, node.end]`` are set in post-order after all
        descendants have been processed.

        Returns the ``node_id`` of the root node.

        The iterative implementation avoids Python's default recursion
        limit (1 000 frames), which caterpillar / pectinate trees with
        thousands of taxa would otherwise exceed.
        """
        root_id: int = -1
        # Each stack entry is either:
        #   ('enter', node_dict, parent_id)  — allocate node, recurse children
        #   ('fix',   node_id)               — set [start, end] from children
        stack: list = [('enter', root_dict, root_parent)]

        while stack:
            tag, *args = stack.pop()

            if tag == 'fix':
                nid = args[0]
                ch = self._nodes[nid].children
                if ch:
                    self._nodes[nid].start = min(self._nodes[c].start for c in ch)
                    self._nodes[nid].end   = max(self._nodes[c].end   for c in ch)
                continue

            nd, par_id = args
            node_id = len(self._nodes)
            if root_id == -1:
                root_id = node_id

            if not nd['children']:   # leaf
                name = nd['name']
                if not name:
                    raise ValueError("Leaf node has no taxon label in NEWICK string")
                if name not in self._name_to_gid:
                    raise ValueError(
                        f"Leaf '{name}' in reference NEWICK not found in forest namespace"
                    )
                gid = self._name_to_gid[name]
                pos = len(self._leaf_gids)
                self._leaf_gids.append(gid)
                self._nodes.append(
                    _Node(node_id=node_id, taxon_gid=gid, parent=par_id,
                          children=[], start=pos, end=pos)
                )
                if par_id != -1:
                    self._connect_to_parent(node_id, par_id, nd['length'], is_internal=False)
                continue

            # Internal node — allocate placeholder, push fix + children.
            self._nodes.append(
                _Node(node_id=node_id, taxon_gid=-1, parent=par_id, children=[])
            )
            if par_id != -1:
                self._connect_to_parent(node_id, par_id, nd['length'], is_internal=True)

            # Push fix BEFORE children so it runs AFTER all descendants finish.
            stack.append(('fix', node_id))
            # Reverse so leftmost child is popped (processed) first.
            for child_dict in reversed(nd['children']):
                stack.append(('enter', child_dict, node_id))

        return root_id

    def _connect_to_parent(
        self,
        child_id: int,
        parent_id: int,
        length: float,
        is_internal: bool,
    ) -> None:
        """Create an edge and register child in parent's children list."""
        edge_id = len(self._edges)
        self._edges.append(_Edge(
            edge_id=edge_id,
            parent=parent_id,
            child=child_id,
            length=length,
            is_internal=is_internal,
        ))
        self._nodes[parent_id].children.append(child_id)
        self._node_to_edge[child_id] = edge_id

    def _normalize_root(self) -> None:
        """
        Collapse the basal bifurcation when the root has exactly 2 children.

        An unrooted binary tree has no basal bifurcation; when the NEWICK
        root happens to have 2 children, the representation encodes one
        that does not exist.  Re-rooting at an internal child gives the
        root ≥ 3 children and restores valid n_qsupp for all root-adjacent
        branches.

        Invariants maintained:
        - Leaf positions are unchanged.
        - All node intervals remain correct (grandchildren are sub-intervals
          of the new root's interval, which covers all leaves).
        - The deactivated root→new_root edge is removed from ``_node_to_edge``.
        """
        root = self._nodes[self._root]
        if len(root.children) != 2:
            return

        c0, c1 = root.children

        # Prefer an internal child as the new root.
        if self._nodes[c0].taxon_gid == -1:
            new_root_id, other_id = c0, c1
        elif self._nodes[c1].taxon_gid == -1:
            new_root_id, other_id = c1, c0
        else:
            return  # Both children are leaves: 2-leaf tree, no branches to score.

        old_root_id = self._root

        # The edge old_root → other is re-parented to new_root.
        other_edge_id = self._node_to_edge[other_id]
        self._edges[other_edge_id].parent = new_root_id
        self._nodes[other_id].parent = new_root_id
        self._nodes[new_root_id].children.append(other_id)

        # Deactivate old_root → new_root edge; new_root has no parent edge.
        nr_edge_id = self._node_to_edge[new_root_id]
        self._edges[nr_edge_id].active = False
        del self._node_to_edge[new_root_id]

        # Promote new_root.
        self._nodes[new_root_id].parent = -1
        self._nodes[new_root_id].start  = 0
        self._nodes[new_root_id].end    = self._n_leaves - 1
        self._root = new_root_id

        # Isolate old root so it is invisible to to_newick() traversals.
        self._nodes[old_root_id].children = []

    # ------------------------------------------------------------------ #
    # Public query interface                                               #
    # ------------------------------------------------------------------ #

    @property
    def active_branch_ids(self) -> set:
        """Set of edge IDs for all currently active internal branches."""
        return {e.edge_id for e in self._edges if e.active and e.is_internal}

    def n_qsupp(self, branch_id: int) -> int:
        """
        Count maximally discriminating quartets for *branch_id*.

        Returns the number of distinct quartets (a, b, c, d) where a, b
        come from different sub-components on side_u and c, d come from
        different sub-components on side_v.  These are the quartets that
        directly test this bipartition against its immediate neighbours.

        Returns 0 if the branch cannot be scored (degenerate topology).
        """
        edge = self._edges[branch_id]
        side_u, side_v = self._bipartition_context(edge)
        u_sizes = [_component_size(c) for c in side_u]
        v_sizes = [_component_size(c) for c in side_v]
        return _pair_total(u_sizes) * _pair_total(v_sizes)

    def sample_quartets(
        self,
        branch_id: int,
        n: int,
        rng: random.Random,
    ) -> list:
        """
        Sample *n* quartets uniformly from the support set of *branch_id*.

        Returns a list of ``(quartet_gids, expected_topo)`` pairs where
        ``quartet_gids`` is a sorted 4-tuple of global taxon IDs and
        ``expected_topo`` is the quarimo topology index (0/1/2) consistent
        with this branch's bipartition.

        Topology convention for sorted (t0 < t1 < t2 < t3):
          0 → t0 t1 | t2 t3
          1 → t0 t2 | t1 t3
          2 → t0 t3 | t1 t2

        Implementation note
        -------------------
        The inner sampling loop is fully vectorised with numpy: all *n*
        component-pair draws and leaf lookups are computed in a single
        ``np.random.Generator.integers()`` call per role (a, b, c, d).
        Topology indices are resolved with element-wise comparisons on
        the sorted (t0..t3) array.  This is ~4× faster than the
        equivalent Python loop for typical batch sizes (n = 64–512).

        Parameters
        ----------
        branch_id : int
        n : int
            Number of samples.  Must be > 0.
        rng : random.Random
            Seeded with ``rng.getrandbits(64)`` to initialise an internal
            ``numpy.random.Generator``.
        """
        if n <= 0:
            return []

        edge = self._edges[branch_id]
        side_u, side_v = self._bipartition_context(edge)

        u_sizes = [_component_size(c) for c in side_u]
        v_sizes = [_component_size(c) for c in side_v]

        u_pairs, _, u_cum = _pair_sampling_tables(u_sizes)
        v_pairs, _, v_cum = _pair_sampling_tables(v_sizes)

        if not u_pairs or not v_pairs:
            raise ValueError(
                f"Branch {branch_id} has no quartet support (n_qsupp == 0); "
                "cannot sample quartets"
            )

        np_rng = np.random.default_rng(rng.getrandbits(64))

        # --- Select cross-component pairs (existing vectorised code) ----
        u_idx = np.searchsorted(
            u_cum, np_rng.integers(0, int(u_cum[-1]), size=n), side='right'
        )
        v_idx = np.searchsorted(
            v_cum, np_rng.integers(0, int(v_cum[-1]), size=n), side='right'
        )

        u_pa = np.asarray(u_pairs, dtype=np.int32)   # (n_upairs, 2)
        v_pa = np.asarray(v_pairs, dtype=np.int32)   # (n_vpairs, 2)

        ui = u_pa[u_idx, 0];  uj = u_pa[u_idx, 1]   # component indices (n,)
        vi = v_pa[v_idx, 0];  vj = v_pa[v_idx, 1]

        # --- Vectorised leaf sampling -----------------------------------
        # _prep_components encodes each component as (start0, span0, start1, span1)
        # where span1 == 0 for single-interval components (the common case).
        # _vsample_fast draws all n leaves for one role in one integers() call.
        su = _prep_components(side_u)
        sv = _prep_components(side_v)

        a = self._leaf_gids_arr[_vsample_fast(*su, ui, np_rng)]
        b = self._leaf_gids_arr[_vsample_fast(*su, uj, np_rng)]
        c = self._leaf_gids_arr[_vsample_fast(*sv, vi, np_rng)]
        d = self._leaf_gids_arr[_vsample_fast(*sv, vj, np_rng)]

        # --- Sort taxa and compute topology index -----------------------
        # Sort each row of [a, b, c, d] → (t0, t1, t2, t3) in ascending order.
        taxa = np.stack([a, b, c, d], axis=1)
        taxa.sort(axis=1)
        t0, t1, t2, t3 = taxa[:, 0], taxa[:, 1], taxa[:, 2], taxa[:, 3]

        # The bipartition is ab | cd.  Find which of the three sorted
        # unrooted splits it corresponds to:
        #   0: t0t1 | t2t3   1: t0t2 | t1t3   2: t0t3 | t1t2
        ab_eq_01 = ((a == t0) & (b == t1)) | ((a == t1) & (b == t0))
        ab_eq_23 = ((a == t2) & (b == t3)) | ((a == t3) & (b == t2))
        ab_eq_02 = ((a == t0) & (b == t2)) | ((a == t2) & (b == t0))
        ab_eq_13 = ((a == t1) & (b == t3)) | ((a == t3) & (b == t1))
        topo = np.where(
            ab_eq_01 | ab_eq_23, np.int32(0),
            np.where(ab_eq_02 | ab_eq_13, np.int32(1), np.int32(2))
        )

        # --- Convert to list of (4-tuple, int) pairs --------------------
        # taxa.tolist() is a fast C operation; map(tuple, ...) avoids a
        # Python loop over indices.
        return list(zip(map(tuple, taxa.tolist()), topo.tolist()))

    def prune(self, branch_id: int) -> set:
        """
        Collapse an internal branch, converting its resolution to a polytomy.

        Re-parents the child node's children to the child's parent, then
        deactivates the edge and child node.  The parent's DFS interval
        does not change because no leaf positions are modified.

        Returns the set of branch IDs of active internal edges adjacent to
        the pruned edge (they should be re-scored after this change).

        Raises
        ------
        AssertionError
            If *branch_id* is already inactive or leads to a leaf.
        """
        return self._prune_internal(branch_id)

    def to_newick(self) -> str:
        """
        Return the current tree topology as a NEWICK string (no branch lengths).

        Leaf labels are the global taxon names from the forest namespace
        supplied at construction.  The tree reflects all pruning operations
        applied since construction.

        Uses an iterative post-order traversal to avoid Python's recursion
        limit for deep (pectinate / caterpillar) trees.

        Stack encoding: positive integer ``nid`` = "leave" phase (write
        result for node ``nid``); negative integer ``~nid`` = "enter" phase
        (push leave + recurse into children).  Using plain integers avoids
        the tuple-creation overhead of a two-element ``(phase, nid)`` stack.
        """
        buf: list = [''] * len(self._nodes)
        stack: list = [~self._root]   # ~nid encodes "enter nid"

        while stack:
            v = stack.pop()
            if v >= 0:              # leave phase
                node = self._nodes[v]
                if not node.children:
                    buf[v] = self._global_names[node.taxon_gid]
                else:
                    buf[v] = '(' + ','.join(buf[c] for c in node.children) + ')'
            else:                   # enter phase: v == ~nid
                nid = ~v
                stack.append(nid)  # push leave
                for c in reversed(self._nodes[nid].children):
                    stack.append(~c)

        return buf[self._root] + ";"

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _prune_internal(self, branch_id: int) -> set:
        """Perform the prune and return the dirty neighbor set."""
        edge = self._edges[branch_id]
        assert edge.active,      f"Branch {branch_id} is already inactive"
        assert edge.is_internal, f"Branch {branch_id} leads to a leaf — cannot prune"

        dirty = self._neighbor_branch_ids(edge)

        parent_id = edge.parent
        child_id  = edge.child
        parent    = self._nodes[parent_id]
        child     = self._nodes[child_id]

        # Detach child from parent.
        parent.children = [c for c in parent.children if c != child_id]

        # Re-parent grandchildren to parent (interval invariant is preserved).
        for grand_id in list(child.children):
            grand_edge_id = self._node_to_edge[grand_id]
            self._edges[grand_edge_id].parent = parent_id
            self._nodes[grand_id].parent      = parent_id
            parent.children.append(grand_id)

        # Deactivate edge and isolate child.
        edge.active = False
        if child_id in self._node_to_edge:
            del self._node_to_edge[child_id]
        child.children = []

        return dirty

    def _bipartition_context(self, edge: _Edge):
        """
        Compute side_u and side_v component lists for *edge*.

        Each component is a tuple of ``(start, end)`` interval pairs.
        Leaf sub-components have a single pair; the parent's complement
        (when parent is not the root) may have two pairs.

        Returns ``(side_u, side_v)``.
        """
        parent = self._nodes[edge.parent]
        child  = self._nodes[edge.child]

        # side_v: one component per active child of the child node.
        side_v = []
        for grand_id in child.children:
            g = self._nodes[grand_id]
            side_v.append(((g.start, g.end),))

        # side_u: active siblings of this edge + parent's complement (if not root).
        side_u = []

        if parent.parent != -1:
            # Two-piece complement of parent's interval.
            complement = []
            if parent.start > 0:
                complement.append((0, parent.start - 1))
            if parent.end < self._n_leaves - 1:
                complement.append((parent.end + 1, self._n_leaves - 1))
            if complement:
                side_u.append(tuple(complement))

        for sib_id in parent.children:
            if sib_id == child.node_id:
                continue
            s = self._nodes[sib_id]
            side_u.append(((s.start, s.end),))

        return side_u, side_v

    def _neighbor_branch_ids(self, edge: _Edge) -> set:
        """Return branch IDs of active internal edges adjacent to *edge*."""
        parent = self._nodes[edge.parent]
        child  = self._nodes[edge.child]
        neighbors: set = set()

        # Parent's own parent edge (edge above the parent node).
        if parent.parent != -1:
            pe_id = self._node_to_edge.get(parent.node_id)
            if pe_id is not None:
                pe = self._edges[pe_id]
                if pe.active and pe.is_internal:
                    neighbors.add(pe.edge_id)

        # Active internal siblings.
        for sib_id in parent.children:
            if sib_id == child.node_id:
                continue
            se_id = self._node_to_edge.get(sib_id)
            if se_id is not None:
                se = self._edges[se_id]
                if se.active and se.is_internal:
                    neighbors.add(se.edge_id)

        # Active internal grandchildren (child's children).
        for grand_id in child.children:
            ge_id = self._node_to_edge.get(grand_id)
            if ge_id is not None:
                ge = self._edges[ge_id]
                if ge.active and ge.is_internal:
                    neighbors.add(ge.edge_id)

        return neighbors


# ======================================================================
# Stand-alone helpers (also used by _consensus.py)
# ======================================================================

def _component_size(component: tuple) -> int:
    """Total number of leaf positions covered by *component*."""
    return sum(end - start + 1 for start, end in component)


def _pair_total(sizes: list) -> int:
    """
    Cross-component pair count: ``(S² − Σ sᵢ²) / 2``.

    Counts unordered pairs of leaves drawn from *different* components.
    Equal to C(S, 2) minus the within-component pairs.
    """
    s = sum(sizes)
    return (s * s - sum(x * x for x in sizes)) // 2


def _pair_sampling_tables(sizes: list) -> tuple:
    """
    Build weighted pair-sampling tables for a list of component sizes.

    Returns ``(pairs, weights, cumulative_weights)`` where:

    * ``pairs[i]`` = ``(component_index_a, component_index_b)`` with ``a < b``
    * ``weights[i]`` = ``sizes[a] × sizes[b]``
    * ``cumulative_weights`` = ``np.cumsum(weights, dtype=int64)``

    Sampling an index proportional to cumulative weight and mapping it
    through ``pairs`` gives a uniformly random cross-component leaf pair.
    """
    pairs: list = []
    weights: list = []
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            pairs.append((i, j))
            weights.append(sizes[i] * sizes[j])
    cum = np.cumsum(weights, dtype=np.int64) if weights else np.zeros(0, dtype=np.int64)
    return pairs, weights, cum


def _quartet_topo_index(a: int, b: int, c: int, d: int) -> int:
    """
    Topology index for the quartet ``{a, b, c, d}`` resolved as ``ab | cd``.

    Given that ``a, b`` were sampled from side_u and ``c, d`` from side_v
    of a bipartition, returns the quarimo topology index (0/1/2) for the
    ab|cd split after sorting the four global taxon IDs.

    Quarimo convention for sorted ``(t0 < t1 < t2 < t3)``:

      0 → t0 t1 | t2 t3
      1 → t0 t2 | t1 t3
      2 → t0 t3 | t1 t2
    """
    t0, t1, t2, t3 = sorted((a, b, c, d))
    ab = frozenset((a, b))
    if ab == frozenset((t0, t1)) or ab == frozenset((t2, t3)):
        return 0
    if ab == frozenset((t0, t2)) or ab == frozenset((t1, t3)):
        return 1
    return 2


# ======================================================================
# Vectorised sampling helpers (used only by ConsensusGraph.sample_quartets)
# ======================================================================

def _prep_components(comps: list) -> tuple:
    """
    Encode a list of components as four parallel int32 arrays for
    vectorised sampling.

    Each component ``comps[i]`` is a tuple of 1 or 2 ``(start, end)``
    DFS position intervals:

    * Single-interval ``((s0, e0),)``: all leaves are in ``[s0, e0]``.
    * Two-interval ``((s0, e0), (s1, e1))``: the complement component
      spans two disjoint ranges (only appears for the parent's complement
      in ``_bipartition_context``).

    Returns ``(starts0, spans0, starts1, spans1)`` where:

    * ``starts0[i]`` = start of the first interval of component ``i``
    * ``spans0[i]``  = width of the first interval (= e0 − s0 + 1)
    * ``starts1[i]`` = start of the second interval (0 if absent)
    * ``spans1[i]``  = width of the second interval (0 if absent)

    Total component size = ``spans0[i] + spans1[i]``.
    """
    k = len(comps)
    starts0 = np.empty(k, dtype=np.int32)
    spans0  = np.empty(k, dtype=np.int32)
    starts1 = np.zeros(k, dtype=np.int32)
    spans1  = np.zeros(k, dtype=np.int32)
    for i, c in enumerate(comps):
        s0, e0 = c[0]
        starts0[i] = s0
        spans0[i]  = e0 - s0 + 1
        if len(c) > 1:
            s1, e1     = c[1]
            starts1[i] = s1
            spans1[i]  = e1 - s1 + 1
    return starts0, spans0, starts1, spans1


def _vsample_fast(
    starts0: np.ndarray,
    spans0:  np.ndarray,
    starts1: np.ndarray,
    spans1:  np.ndarray,
    comp_idx: np.ndarray,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample one DFS leaf position per query using a single numpy
    ``integers()`` call.

    For each query ``k``, the target component is ``comp_idx[k]``.
    The component's total size is ``(spans0 + spans1)[comp_idx[k]]``
    (``spans1 == 0`` for single-interval components).  A uniform offset
    ``r`` is drawn in ``[0, total_size)``; if ``r < spans0[comp_idx[k]]``
    the leaf comes from the first interval, otherwise from the second.

    Returns an int32 array of DFS positions (indices into
    ``ConsensusGraph._leaf_gids_arr``), shape ``(n,)``.
    """
    total = (spans0 + spans1)[comp_idx]          # per-query component sizes (n,)
    r     = np_rng.integers(0, total)            # one draw per query
    in_first = r < spans0[comp_idx]
    return np.where(
        in_first,
        starts0[comp_idx] + r,
        starts1[comp_idx] + r - spans0[comp_idx],
    )
