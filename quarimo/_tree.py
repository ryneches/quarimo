"""
_tree.py
========
A single phylogenetic tree represented as a set of parallel numpy arrays,
with O(1) LCA lookups via a sparse-table Range Minimum Query structure built
on the Eulerian tour.

Public API
----------
  Tree(newick_string)
      Constructor.  Parses the NEWICK string and builds all data structures.

  .lca(u, v)
  .multi_lca(nodes, return_steiner=False)
  .branch_distance(u, v, return_lca=False)
  .quartet_split(a, b, c, d, return_steiner=False)
  .quartet_topo_index(p0, p1, q0, q1, n0, n1, n2, n3)   [staticmethod]
  .quartet_topology(a, b, c, d, return_steiner=False)

Inheritance notes
-----------------
This class is designed to be inherited by a tree-collection class (not yet
implemented).  Key design choices that support this:

* All tree data lives in flat numpy arrays attached directly to ``self``.
  A collection subclass can store per-tree arrays as rows in a 2-D batch
  array (where array lengths permit) or as a list of 1-D arrays.
* No Python-level cross-instance state.  The name-index cache is private to
  each instance (``self._name_index``).
* No mutable class-level state; every attribute is set in ``__init__``.

GPU / numba notes
-----------------
Migration to numba requires that computational kernels accept only numpy
arrays and plain Python scalars — no ``self`` references.  This is already
the case for ``_rmq`` and ``_quartet_split_core``, which are ``@staticmethod``
functions with explicit array-parameter signatures.  When adapting:

1. Decorate ``_rmq`` and ``_quartet_split_core`` with ``@numba.njit``.
2. The public instance methods remain as Python wrappers that extract the
   relevant ``self.*`` arrays and forward them to the JIT-compiled kernels.
3. The name-index resolution (``_resolve_node``, ``_build_name_index``) stays
   in pure Python — GPU kernels work on integer IDs only; name lookup is done
   once in the host wrapper before the kernel call.
"""

import math
import numpy as np


class Tree:
    """
    A rooted, strictly bifurcating phylogenetic tree with O(1) LCA queries.

    Attributes (all read-only after construction)
    ----------------------------------------------
    n_nodes   : int     Total number of nodes (2 * n_leaves - 1).
    n_leaves  : int     Number of leaf (taxon) nodes.
    root      : int     Node ID of the root (always n_nodes - 1).
    max_depth : int     Maximum node depth (edge count from root).
    names     : list[str]  Taxon name for each node; '' for internal nodes.

    Arrays — tree structure
    -----------------------
    parent      : int32  [n_nodes]   Parent ID; -1 for root.
    distance    : float64[n_nodes]   Branch length to parent; -1.0 for root.
    support     : float64[n_nodes]   Branch support to parent; -1.0 sentinel.
    left_child  : int32  [n_nodes]   Left child ID; -1 for leaves.
    right_child : int32  [n_nodes]   Right child ID; -1 for leaves.

    Arrays — LCA / Euler tour
    -------------------------
    depth            : int32  [n_nodes]       Edge depth from root.
    root_distance    : float64[n_nodes]       Cumulative branch length from root.
    euler_tour       : int32  [2n-1]          Euler tour node IDs.
    euler_depth      : int32  [2n-1]          Depth at each tour position.
    first_occurrence : int32  [n_nodes]       First tour index for each node.
    sparse_table     : int32  [LOG, 2n-1]     Sparse table (stores tour indices).
    log2_table       : int32  [2n]            floor(log2(i)) for i ∈ [0, 2n-1].
    """

    # ================================================================== #
    # Construction                                                         #
    # ================================================================== #

    def __init__(self, newick_string: str) -> None:
        """
        Parse *newick_string* and build all tree and LCA data structures.

        Parameters
        ----------
        newick_string : str
            A valid NEWICK-formatted tree string (trailing ';' optional).
        """
        self._parse_newick(newick_string)
        self._build_lca_structures()

        # ---- Derived scalar properties (convenient for callers and  ---- #
        # ---- future subclasses that query shape without array access) ---- #
        self.n_nodes: int = int(self.parent.shape[0])
        self.n_leaves: int = (self.n_nodes + 1) // 2
        self.root: int = self.n_nodes - 1  # parse_newick invariant
        self.max_depth: int = int(np.max(self.depth))

        # Name index: built lazily on first name-based query.
        self._name_index: dict = None  # type: ignore[assignment]

    # ================================================================== #
    # Public methods                                                       #
    # ================================================================== #

    def lca(self, u, v) -> int:
        """
        Return the node ID of the Lowest Common Ancestor of *u* and *v*.

        Parameters
        ----------
        u, v : int | str   Node IDs or taxon names (resolved independently).

        Returns
        -------
        int   Node ID of the LCA.

        Complexity
        ----------
        O(1) per call (after the O(n) name-index is built on first name use).
        """
        u_id = self._resolve_node(u)
        v_id = self._resolve_node(v)

        if u_id == v_id:
            return u_id

        l = int(self.first_occurrence[u_id])
        r = int(self.first_occurrence[v_id])
        if l > r:
            l, r = r, l

        idx = Tree._rmq(l, r, self.sparse_table, self.euler_depth, self.log2_table)
        return int(self.euler_tour[idx])

    def multi_lca(self, nodes, return_steiner: bool = False):
        """
        Return the LCA of all nodes in *nodes* and, optionally, the total
        branch length of the minimal Steiner subtree connecting them.

        The LCA of N nodes requires a single RMQ over the span
        [min fo, max fo] of the nodes' first-occurrence positions.

        The Steiner length uses the DFS-cycle identity:
            Steiner(S) = ½ · Σᵢ dist(v_{i}, v_{(i+1) mod n})
        where the nodes are sorted by first_occurrence.

        Parameters
        ----------
        nodes          : sequence of (int | str)   Length ≥ 1; duplicates OK.
        return_steiner : bool   If True return (lca_id, steiner_length).

        Returns
        -------
        int              LCA node ID (when return_steiner is False).
        (int, float)     (lca_id, steiner_length) (when return_steiner is True).

        Raises
        ------
        ValueError   if *nodes* is empty.
        KeyError     if a name is not found.

        Complexity
        ----------
        O(n log n) sort + O(n) RMQ queries.
        """
        n = len(nodes)
        if n == 0:
            raise ValueError("nodes must contain at least one element.")

        node_ids = np.empty(n, dtype=np.int32)
        for i in range(n):
            node_ids[i] = self._resolve_node(nodes[i])

        if n == 1:
            lca_id = int(node_ids[0])
            return (lca_id, 0.0) if return_steiner else lca_id

        fo_values = np.empty(n, dtype=np.int32)
        for i in range(n):
            fo_values[i] = int(self.first_occurrence[node_ids[i]])

        order = np.argsort(fo_values, kind="stable")
        sorted_ids = node_ids[order]
        sorted_fo = fo_values[order]

        l = int(sorted_fo[0])
        r = int(sorted_fo[n - 1])
        lca_idx = Tree._rmq(l, r, self.sparse_table, self.euler_depth, self.log2_table)
        lca_id = int(self.euler_tour[lca_idx])

        if not return_steiner:
            return lca_id

        total = 0.0
        for i in range(n):
            u_id = int(sorted_ids[i])
            v_id = int(sorted_ids[(i + 1) % n])

            fu = int(self.first_occurrence[u_id])
            fv = int(self.first_occurrence[v_id])
            if fu > fv:
                fu, fv = fv, fu

            pair_lca_idx = Tree._rmq(
                fu, fv, self.sparse_table, self.euler_depth, self.log2_table
            )
            pair_lca_id = int(self.euler_tour[pair_lca_idx])
            total += (
                float(self.root_distance[u_id])
                + float(self.root_distance[v_id])
                - 2.0 * float(self.root_distance[pair_lca_id])
            )

        return lca_id, total * 0.5

    def branch_distance(self, u, v, return_lca: bool = False):
        """
        Return the total branch length between nodes *u* and *v*.

        Uses the identity:
            dist(u, v) = root_distance[u] + root_distance[v]
                         − 2 × root_distance[LCA(u, v)]

        Parameters
        ----------
        u, v       : int | str   Node IDs or taxon names.
        return_lca : bool        If True return (distance, lca_id).

        Returns
        -------
        float              Branch distance (when return_lca is False).
        (float, int)       (distance, lca_id) (when return_lca is True).

        Complexity
        ----------
        O(1) per call.
        """
        u_id = self._resolve_node(u)
        v_id = self._resolve_node(v)

        if u_id == v_id:
            return (0.0, u_id) if return_lca else 0.0

        l = int(self.first_occurrence[u_id])
        r = int(self.first_occurrence[v_id])
        if l > r:
            l, r = r, l

        lca_idx = Tree._rmq(l, r, self.sparse_table, self.euler_depth, self.log2_table)
        lca_id = int(self.euler_tour[lca_idx])

        dist = (
            float(self.root_distance[u_id])
            + float(self.root_distance[v_id])
            - 2.0 * float(self.root_distance[lca_id])
        )

        return (dist, lca_id) if return_lca else dist

    def quartet_split(self, a, b, c, d, return_steiner: bool = False):
        """
        Return the unrooted quartet topology of four *distinct* nodes as a
        canonical 4-int split tuple (p0, p1, q0, q1).

        The split satisfies p0 < p1, q0 < q1, and p0 < q0 (fully sorted
        canonical form).  This representation maps directly to a C struct:
            typedef struct { int32_t p0, p1, q0, q1; } QuartetSplit;

        The topology is derived from a single 4-element sort by
        first_occurrence — no RMQ call is needed for the topology itself.
        Optionally, the Steiner tree length is computed via 4 O(1) RMQ calls.

        Parameters
        ----------
        a, b, c, d     : int | str   Four distinct node IDs or taxon names.
        return_steiner : bool        If True return (p0, p1, q0, q1, steiner).

        Returns
        -------
        (int, int, int, int)              Canonical split (when return_steiner False).
        (int, int, int, int, float)       Split + Steiner length.

        Raises
        ------
        ValueError   if any two of the four nodes are identical.
        KeyError     if a name is not found.

        Complexity
        ----------
        O(1) topology; O(1) Steiner length (4 RMQ queries).
        """
        a_id = self._resolve_node(a)
        b_id = self._resolve_node(b)
        c_id = self._resolve_node(c)
        d_id = self._resolve_node(d)

        if (
            a_id == b_id
            or a_id == c_id
            or a_id == d_id
            or b_id == c_id
            or b_id == d_id
            or c_id == d_id
        ):
            raise ValueError(
                "All four nodes must be distinct; "
                f"received IDs ({a_id}, {b_id}, {c_id}, {d_id})."
            )

        return Tree._quartet_split_core(
            a_id,
            b_id,
            c_id,
            d_id,
            return_steiner,
            self.first_occurrence,
            self.root_distance,
            self.sparse_table,
            self.euler_depth,
            self.log2_table,
            self.euler_tour,
        )

    @staticmethod
    def quartet_topo_index(
        p0: int, p1: int, q0: int, q1: int, n0: int, n1: int, n2: int, n3: int
    ) -> int:
        """
        Convert a canonical split (p0, p1, q0, q1) to a compact topology
        index in {0, 1, 2} relative to the four nodes sorted by ID as
        n0 < n1 < n2 < n3.

          topo 0  ↔  (n0, n1) | (n2, n3)
          topo 1  ↔  (n0, n2) | (n1, n3)
          topo 2  ↔  (n0, n3) | (n1, n2)

        Only 2 bits are required; in C/Rust this is ``uint8_t topo;``.

        Parameters
        ----------
        p0, p1, q0, q1 : int   Canonical split from quartet_split().
        n0, n1, n2, n3 : int   The four node IDs sorted ascending.

        Returns
        -------
        int  — 0, 1, or 2.

        Notes
        -----
        This is a ``@staticmethod`` because it operates on plain integers only
        and requires no instance state.  It can be extracted and JIT-compiled
        with numba without modification.
        """
        # p0 == n0 by the canonical form's p0 < q0 guarantee, so n0's partner
        # is always p1.
        n0_partner = p1
        if n0_partner == n1:
            return 0
        if n0_partner == n2:
            return 1
        return 2  # n0_partner == n3

    def quartet_topology(self, a, b, c, d, return_steiner: bool = False):
        """
        Return the unrooted quartet topology as a ``frozenset`` of
        ``frozenset``s.

        This is a thin Python wrapper around ``quartet_split`` that converts
        the low-level (p0, p1, q0, q1) tuple into the canonical Pythonic form
        used in set-based phylogenetic algorithms.

        The element type of the inner frozensets mirrors the input type:
          - All four inputs are ``str``  →  inner frozensets contain taxon names.
          - Otherwise                    →  inner frozensets contain integer node IDs.

        Parameters
        ----------
        a, b, c, d     : int | str   Four distinct node IDs or taxon names.
        return_steiner : bool        If True return (frozenset, steiner_length).

        Returns
        -------
        frozenset{frozenset{str,str}, frozenset{str,str}}   when all inputs are str.
        frozenset{frozenset{int,int}, frozenset{int,int}}   otherwise.
        (frozenset{…}, float)  — when return_steiner is True.
        """
        use_names = all(isinstance(x, str) for x in (a, b, c, d))

        result = self.quartet_split(a, b, c, d, return_steiner=return_steiner)

        if return_steiner:
            p0, p1, q0, q1, steiner = result
        else:
            p0, p1, q0, q1 = result

        if use_names:
            topo = frozenset(
                {
                    frozenset({self.names[p0], self.names[p1]}),
                    frozenset({self.names[q0], self.names[q1]}),
                }
            )
        else:
            topo = frozenset({frozenset({p0, p1}), frozenset({q0, q1})})

        return (topo, steiner) if return_steiner else topo

    # ================================================================== #
    # Private instance methods                                             #
    # ================================================================== #

    def _parse_newick(self, newick_string: str) -> None:
        """
        **Private.**  Parse *newick_string* and populate the tree-structure
        arrays as instance attributes.

        Two-pass algorithm
        ------------------
        Pass 1  Count commas → derive exact array sizes (n_leaves, n_nodes).
        Pass 2  Iterative, stack-based character scan; no recursion.

        Node-ID conventions (set once; never change):
          Leaves   : 0 … n_leaves-1       (left-to-right in NEWICK string)
          Internal : n_leaves … n_nodes-2 (post-order)
          Root     : n_nodes-1            (invariant used throughout the class)

        Populates
        ---------
        self.names, self.parent, self.distance, self.support,
        self.left_child, self.right_child

        Notes
        -----
        Designed for straightforward translation to Cython/C/Rust:
        all control flow uses indexed loops and explicit character comparisons.
        """
        s = newick_string.strip()
        n_chars = len(s)
        if n_chars > 0 and s[n_chars - 1] == ";":
            n_chars -= 1

        # ---- Pass 1: count commas and open parens ------------------- #
        # For a strictly bifurcating rooted tree with L leaves:
        #   n_commas = L - 1  (always true for any rooted tree)
        #   n_parens = L - 1  (one internal node per internal edge)
        # Multifurcation adds extra commas without extra parens, so
        # n_parens < n_commas reliably signals multifurcation.
        n_commas = 0
        n_parens = 0
        for k in range(n_chars):
            c = s[k]
            if c == ",":
                n_commas += 1
            elif c == "(":
                n_parens += 1

        n_leaves = n_commas + 1

        if n_parens < n_commas:
            import logging

            logger = logging.getLogger(__name__)
            n_extra = n_commas - n_parens
            logger.warning(
                f"Input tree is not strictly bifurcating: {n_parens} internal "
                f"nodes for {n_leaves} leaves (expected {n_commas}). "
                f"{n_extra} multifurcation(s) will be resolved into zero-length "
                f"bifurcations. The order of splitting is arbitrary."
            )
            s = Tree._resolve_multifurcations(s[:n_chars])
            n_chars = len(s)
            if n_chars > 0 and s[n_chars - 1] == ";":
                n_chars -= 1
            n_commas = 0
            for k in range(n_chars):
                if s[k] == ",":
                    n_commas += 1
            n_leaves = n_commas + 1

        n_nodes = 2 * n_leaves - 1

        # ---- Allocate arrays ---------------------------------------- #
        parent = np.full(n_nodes, -1, dtype=np.int32)
        distance = np.full(n_nodes, -1.0, dtype=np.float64)
        support = np.full(n_nodes, -1.0, dtype=np.float64)
        left_child = np.full(n_nodes, -1, dtype=np.int32)
        right_child = np.full(n_nodes, -1, dtype=np.int32)
        names = [""] * n_nodes

        # ---- Pass 2: iterative stack-based parse -------------------- #
        OPEN_PAREN = -2
        stack_node = [0] * n_nodes
        stack_top = -1

        leaf_id = 0
        internal_id = n_leaves

        i = 0
        while i < n_chars:
            c = s[i]

            if c == " " or c == "\t" or c == "\n" or c == "\r":
                i += 1
                continue

            if c == "(":
                stack_top += 1
                stack_node[stack_top] = OPEN_PAREN
                i += 1
                continue

            if c == ",":
                i += 1
                continue

            if c == ")":
                i += 1
                right = stack_node[stack_top]
                stack_top -= 1
                left = stack_node[stack_top]
                stack_top -= 1
                stack_top -= 1  # discard OPEN_PAREN

                node_id = internal_id
                internal_id += 1

                left_child[node_id] = left
                right_child[node_id] = right
                parent[left] = node_id
                parent[right] = node_id

                while i < n_chars and (s[i] == " " or s[i] == "\t"):
                    i += 1

                if (
                    i < n_chars
                    and s[i] != ":"
                    and s[i] != ","
                    and s[i] != ")"
                    and s[i] != ";"
                ):
                    j = i
                    while (
                        j < n_chars
                        and s[j] != ":"
                        and s[j] != ","
                        and s[j] != ")"
                        and s[j] != ";"
                        and s[j] != " "
                        and s[j] != "\t"
                    ):
                        j += 1
                    if j > i:
                        support[node_id] = float(s[i:j])
                    i = j

                while i < n_chars and (s[i] == " " or s[i] == "\t"):
                    i += 1

                if i < n_chars and s[i] == ":":
                    i += 1
                    while i < n_chars and (s[i] == " " or s[i] == "\t"):
                        i += 1
                    j = i
                    while (
                        j < n_chars
                        and s[j] != ","
                        and s[j] != ")"
                        and s[j] != ";"
                        and s[j] != " "
                        and s[j] != "\t"
                    ):
                        j += 1
                    distance[node_id] = float(s[i:j])
                    i = j

                stack_top += 1
                stack_node[stack_top] = node_id
                continue

            # Leaf
            j = i
            while (
                j < n_chars
                and s[j] != ":"
                and s[j] != ","
                and s[j] != ")"
                and s[j] != ";"
                and s[j] != " "
                and s[j] != "\t"
            ):
                j += 1

            node_id = leaf_id
            leaf_id += 1
            names[node_id] = s[i:j]
            i = j

            while i < n_chars and (s[i] == " " or s[i] == "\t"):
                i += 1

            if i < n_chars and s[i] == ":":
                i += 1
                while i < n_chars and (s[i] == " " or s[i] == "\t"):
                    i += 1
                j = i
                while (
                    j < n_chars
                    and s[j] != ","
                    and s[j] != ")"
                    and s[j] != ";"
                    and s[j] != " "
                    and s[j] != "\t"
                ):
                    j += 1
                distance[node_id] = float(s[i:j])
                i = j

            stack_top += 1
            stack_node[stack_top] = node_id

        self.names = names
        self.parent = parent
        self.distance = distance
        self.support = support
        self.left_child = left_child
        self.right_child = right_child

    def _build_lca_structures(self) -> None:
        """
        **Private.**  Build the Euler-tour arrays and sparse table needed for
        O(1) LCA queries.

        Iterative Euler tour
        --------------------
        A phase-coded stack drives the DFS without recursion:
          phase 0  First entry: compute depth/root_distance; append to tour.
          phase 1  Return from left child: append node again.
          phase 2  Return from right child: append node again.

        Sparse table
        ------------
        ``sparse_table[k, i]`` stores the tour index j ∈ [i, i+2^k-1] where
        ``euler_depth[j]`` is minimised.  Built level-by-level using NumPy
        element-wise operations (maps to a C inner loop).

        Populates
        ---------
        self.depth, self.root_distance, self.euler_tour, self.euler_depth,
        self.first_occurrence, self.sparse_table, self.log2_table
        """
        n_nodes = int(self.parent.shape[0])
        tour_len = 2 * n_nodes - 1
        root = n_nodes - 1

        depth = np.zeros(n_nodes, dtype=np.int32)
        root_distance = np.zeros(n_nodes, dtype=np.float64)
        euler_tour = np.zeros(tour_len, dtype=np.int32)
        euler_depth = np.zeros(tour_len, dtype=np.int32)
        first_occurrence = np.full(n_nodes, -1, dtype=np.int32)

        # Euler tour via phase-coded stack
        max_stack = 4 * n_nodes
        stack_node = np.zeros(max_stack, dtype=np.int32)
        stack_phase = np.zeros(max_stack, dtype=np.int32)
        stack_top = 0
        stack_node[0] = root
        stack_phase[0] = 0
        tour_pos = 0

        parent = self.parent
        distance = self.distance
        left_child = self.left_child
        right_child = self.right_child

        while stack_top >= 0:
            node = int(stack_node[stack_top])
            phase = int(stack_phase[stack_top])
            stack_top -= 1

            if phase == 0:
                p = int(parent[node])
                if p == -1:
                    depth[node] = 0
                    root_distance[node] = 0.0
                else:
                    depth[node] = depth[p] + 1
                    root_distance[node] = root_distance[p] + distance[node]

                euler_tour[tour_pos] = node
                euler_depth[tour_pos] = depth[node]
                first_occurrence[node] = tour_pos
                tour_pos += 1

                lc = int(left_child[node])
                if lc != -1:
                    rc = int(right_child[node])
                    stack_top += 1
                    stack_node[stack_top] = node
                    stack_phase[stack_top] = 2
                    stack_top += 1
                    stack_node[stack_top] = rc
                    stack_phase[stack_top] = 0
                    stack_top += 1
                    stack_node[stack_top] = node
                    stack_phase[stack_top] = 1
                    stack_top += 1
                    stack_node[stack_top] = lc
                    stack_phase[stack_top] = 0
            else:
                euler_tour[tour_pos] = node
                euler_depth[tour_pos] = depth[node]
                tour_pos += 1

        # Sparse table
        LOG = int(math.floor(math.log2(tour_len))) + 1 if tour_len > 1 else 1
        sparse_table = np.zeros((LOG, tour_len), dtype=np.int32)
        sparse_table[0] = np.arange(tour_len, dtype=np.int32)

        for k in range(1, LOG):
            half = 1 << (k - 1)
            valid = tour_len - half
            left_pos = sparse_table[k - 1, :valid]
            right_pos = sparse_table[k - 1, half:tour_len]
            left_depths = euler_depth[left_pos]
            right_depths = euler_depth[right_pos]
            sparse_table[k, :valid] = np.where(
                right_depths < left_depths, right_pos, left_pos
            )
            sparse_table[k, valid:] = sparse_table[k - 1, valid:]

        # floor(log2) lookup table
        log2_table = np.zeros(tour_len + 1, dtype=np.int32)
        for i in range(2, tour_len + 1):
            log2_table[i] = log2_table[i >> 1] + 1

        self.depth = depth
        self.root_distance = root_distance
        self.euler_tour = euler_tour
        self.euler_depth = euler_depth
        self.first_occurrence = first_occurrence
        self.sparse_table = sparse_table
        self.log2_table = log2_table

    def _resolve_node(self, node) -> int:
        """
        **Private.**  Return the integer node ID for *node*.

        If *node* is already an integer (or numpy integer), it is returned
        as a plain Python ``int``.  If it is a ``str``, the name index is
        built lazily and the result is looked up.

        Raises
        ------
        KeyError   if *node* is a string not present in the tree.

        Notes
        -----
        The name-index cache is intentionally kept as a plain Python dict
        (``self._name_index``) so it is isolated per instance.  GPU/numba
        callers should perform all name resolution in Python before passing
        integer node IDs to the kernel.
        """
        if isinstance(node, (int, np.integer)):
            return int(node)
        if self._name_index is None:
            self._build_name_index()
        if node not in self._name_index:
            raise KeyError(f"No node with name '{node}' found in tree.")
        return self._name_index[node]

    def _build_name_index(self) -> None:
        """
        **Private.**  Build and cache ``self._name_index``: a dict mapping
        each non-empty node name to its integer node ID.

        Called at most once per instance (lazily from ``_resolve_node``).

        Raises
        ------
        ValueError   if duplicate taxon names are found.
        """
        idx = {}
        for node_id in range(len(self.names)):
            name = self.names[node_id]
            if name != "":
                if name in idx:
                    raise ValueError(
                        f"Duplicate node name '{name}' at IDs "
                        f"{idx[name]} and {node_id}."
                    )
                idx[name] = node_id
        self._name_index = idx

    # ================================================================== #
    # Private static methods (pure computational kernels)                  #
    #                                                                      #
    # These methods take only numpy arrays and plain integers — no self.  #
    # They are the primary candidates for @numba.njit decoration when     #
    # migrating to GPU: decorate and call from the public instance        #
    # methods which pass the relevant self.* arrays explicitly.            #
    # ================================================================== #

    @staticmethod
    def _resolve_multifurcations(s: str) -> str:
        """
        **Private static.**  Rewrite a NEWICK string so that every internal
        node has exactly two children.

        Any node with k > 2 children is converted to a left-to-right cascade
        of (k - 1) binary nodes whose added parent branches have length 0.0:

            (A, B, C, D)  →  (((A, B):0.0, C):0.0, D)

        The order of merging is arbitrary (first-two-first); only the
        unrooted topology of the original tree is preserved, not the relative
        order of the extra zero-length branches.

        Parameters
        ----------
        s : str
            NEWICK string with the trailing ';' already stripped.

        Returns
        -------
        str
            Strictly bifurcating NEWICK string with a trailing ';'.

        Notes
        -----
        Uses a single character-by-character pass with an explicit list stack
        (no recursion).  The suffix after each closing ')' — an optional
        support value followed by an optional ':length' — is read inline and
        appended verbatim to the node string, preserving all original branch
        lengths and support values for the non-added nodes.
        """
        n = len(s)
        stack = [[]]  # stack of completed-child lists; index 0 = root level
        buf = []  # character buffer for the current leaf/token

        i = 0
        while i < n:
            c = s[i]

            if c in " \t\r\n":
                i += 1
                continue

            if c == "(":
                # Flush any stray buffer content (should not occur in valid NEWICK)
                if buf:
                    stack[-1].append("".join(buf))
                    buf = []
                stack.append([])  # open a new child-list
                i += 1

            elif c == ",":
                if buf:
                    stack[-1].append("".join(buf))
                    buf = []
                i += 1

            elif c == ")":
                # Flush last child of this group
                if buf:
                    stack[-1].append("".join(buf))
                    buf = []

                children = stack.pop()

                # Binarize: merge first two children repeatedly until binary
                while len(children) > 2:
                    left = children.pop(0)
                    right = children.pop(0)
                    children.insert(0, f"({left},{right}):0.0")

                node_str = "(" + ",".join(children) + ")"
                i += 1

                # ── Read optional suffix: support_value and/or :length ── #
                # Skip whitespace
                while i < n and s[i] in " \t":
                    i += 1
                # Support value: non-special chars before ':' or delimiter
                if i < n and s[i] not in ":,);\t\n\r ":
                    j = i
                    while j < n and s[j] not in ":,);\t\n\r ":
                        j += 1
                    node_str += s[i:j]
                    i = j
                # Skip whitespace
                while i < n and s[i] in " \t":
                    i += 1
                # :length
                if i < n and s[i] == ":":
                    node_str += ":"
                    i += 1
                    while i < n and s[i] in " \t":
                        i += 1
                    j = i
                    while j < n and s[j] not in ",);\t\n\r ":
                        j += 1
                    node_str += s[i:j]
                    i = j

                stack[-1].append(node_str)

            elif c == ";":
                if buf:
                    stack[-1].append("".join(buf))
                    buf = []
                break

            else:
                buf.append(c)
                i += 1

        # Flush any trailing token
        if buf:
            stack[-1].append("".join(buf))

        # Root level should hold exactly one item (the full tree string)
        root_children = stack[0]
        if len(root_children) == 1:
            return root_children[0] + ";"
        # Safety: if somehow the root was never wrapped, wrap it now
        while len(root_children) > 2:
            left = root_children.pop(0)
            right = root_children.pop(0)
            root_children.insert(0, f"({left},{right}):0.0")
        return "(" + ",".join(root_children) + ");"

    @staticmethod
    def _rmq(l: int, r: int, sparse_table, euler_depth, log2_table) -> int:
        """
        **Private static.**  O(1) Range Minimum Query on ``euler_depth``.

        Returns the tour index ``i`` in ``[l, r]`` where ``euler_depth[i]``
        is minimised (left-biased on ties for deterministic results).

        Parameters
        ----------
        l, r          : int   Inclusive range endpoints; caller ensures l ≤ r.
        sparse_table  : int32 array shape (LOG, tour_len)
        euler_depth   : int32 array shape (tour_len,)
        log2_table    : int32 array shape (tour_len + 1,)

        Returns
        -------
        int   Tour index of the minimum-depth element.

        Notes
        -----
        This is a ``@staticmethod`` with an explicit array-parameter
        signature so it can be extracted and decorated with ``@numba.njit``
        without modification.  In C:
            int32_t rmq(int l, int r, int32_t *sp, int32_t *ed, int32_t *lg)
        """
        length = r - l + 1
        k = int(log2_table[length])
        half = 1 << k
        li = int(sparse_table[k, l])
        ri = int(sparse_table[k, r - half + 1])
        if int(euler_depth[ri]) < int(euler_depth[li]):
            return ri
        return li

    @staticmethod
    def _quartet_split_core(
        a_id: int,
        b_id: int,
        c_id: int,
        d_id: int,
        return_steiner: bool,
        first_occurrence,
        root_distance,
        sparse_table,
        euler_depth,
        log2_table,
        euler_tour,
    ):
        """
        **Private static.**  Core quartet topology and Steiner length kernel.

        Determines the correct unrooted quartet topology via the **four-point
        condition**, then optionally computes the Steiner length.

        Parameters
        ----------
        a_id … d_id    : int     Resolved integer node IDs (all distinct).
        return_steiner : bool
        first_occurrence, root_distance, sparse_table,
        euler_depth, log2_table, euler_tour
                       : numpy arrays from the tree instance.

        Returns
        -------
        (p0, p1, q0, q1)           or
        (p0, p1, q0, q1, steiner)

        Notes
        -----
        This is a ``@staticmethod`` with explicit array parameters — it can
        be extracted and JIT-compiled with ``@numba.njit`` without any
        modification to its body.

        Topology via the four-point condition
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        For any 4 nodes {a,b,c,d} in a tree there are three possible splits.
        Define the *score* of split (p,q)|(r,s) as:

            score = root_distance[LCA(p,q)] + root_distance[LCA(r,s)]

        By the four-point condition, the score of the correct split is strictly
        greater than the scores of both wrong splits (which are equal to each
        other).  Comparing just two of the three scores is sufficient:

            r1 = rd[LCA(a,b)] + rd[LCA(c,d)]   # split (a,b)|(c,d)
            r2 = rd[LCA(a,c)] + rd[LCA(b,d)]   # split (a,c)|(b,d)

            r1 > r2  →  (a,b)|(c,d) is correct
            r2 > r1  →  (a,c)|(b,d) is correct
            r1 = r2  →  (a,d)|(b,c) is correct  (third split wins by exclusion)

        This requires exactly 4 RMQ calls for topology, with equality
        between r1 and r2 guaranteed to be exact (both equal 2·rd[overall LCA]).

        *** Previous implementation note ***
        An earlier version sorted nodes by first_occurrence and returned the
        "adjacent pairs" as the split.  That approach is correct only when
        BOTH pairs' LCAs are proper descendants of the overall LCA (balanced
        quartets).  For unbalanced quartets — where one pair's LCA equals
        the overall LCA — the DFS-adjacent pairs yield the *wrong* split
        roughly half the time (sister taxa are separated).  The four-point
        condition has no such restriction and is always correct.

        Steiner length
        ~~~~~~~~~~~~~~
        The half-cycle formula requires consecutive distances in DFS
        (first_occurrence) order.  This is independent of which pairs form
        the topology split, so the DFS sort is retained for Steiner only.
        The Steiner computation uses 4 additional RMQ calls.

        Complexity
        ~~~~~~~~~~
        Topology: 4 RMQ calls.  Steiner (when requested): 4 additional RMQ
        calls.  All O(1) per quartet.
        """
        fo = first_occurrence
        rd = root_distance

        # ------------------------------------------------------------------ #
        # Topology: four-point condition (4 RMQ calls)                        #
        # ------------------------------------------------------------------ #
        # LCA(a,b)
        l = int(fo[a_id])
        r = int(fo[b_id])
        if l > r:
            l, r = r, l
        idx = Tree._rmq(l, r, sparse_table, euler_depth, log2_table)
        rd_lca_ab = float(rd[int(euler_tour[idx])])

        # LCA(c,d)
        l = int(fo[c_id])
        r = int(fo[d_id])
        if l > r:
            l, r = r, l
        idx = Tree._rmq(l, r, sparse_table, euler_depth, log2_table)
        rd_lca_cd = float(rd[int(euler_tour[idx])])

        # LCA(a,c)
        l = int(fo[a_id])
        r = int(fo[c_id])
        if l > r:
            l, r = r, l
        idx = Tree._rmq(l, r, sparse_table, euler_depth, log2_table)
        rd_lca_ac = float(rd[int(euler_tour[idx])])

        # LCA(b,d)
        l = int(fo[b_id])
        r = int(fo[d_id])
        if l > r:
            l, r = r, l
        idx = Tree._rmq(l, r, sparse_table, euler_depth, log2_table)
        rd_lca_bd = float(rd[int(euler_tour[idx])])

        r1 = rd_lca_ab + rd_lca_cd  # score for (a,b)|(c,d)
        r2 = rd_lca_ac + rd_lca_bd  # score for (a,c)|(b,d)

        # r1 == r2 iff both equal 2·rd[overall_LCA]; third split wins exactly.
        if r1 > r2:
            left0, left1, right0, right1 = a_id, b_id, c_id, d_id
        elif r2 > r1:
            left0, left1, right0, right1 = a_id, c_id, b_id, d_id
        else:  # r1 == r2 → (a,d)|(b,c) is the correct split
            left0, left1, right0, right1 = a_id, d_id, b_id, c_id

        # Canonicalise: sort within pairs, then sort the two pairs
        if left0 > left1:
            left0, left1 = left1, left0
        if right0 > right1:
            right0, right1 = right1, right0
        if left0 > right0 or (left0 == right0 and left1 > right1):
            left0, left1, right0, right1 = right0, right1, left0, left1

        p0 = left0
        p1 = left1
        q0 = right0
        q1 = right1

        if not return_steiner:
            return p0, p1, q0, q1

        # ------------------------------------------------------------------ #
        # Steiner length: half-sum of 4 consecutive DFS-order distances.      #
        # Sort nodes by first_occurrence for the DFS cycle.                   #
        # ------------------------------------------------------------------ #
        node_ids = [a_id, b_id, c_id, d_id]
        fo_vals = [int(fo[a_id]), int(fo[b_id]), int(fo[c_id]), int(fo[d_id])]

        # 4-element insertion sort
        for i in range(1, 4):
            key_fo = fo_vals[i]
            key_id = node_ids[i]
            j = i - 1
            while j >= 0 and fo_vals[j] > key_fo:
                fo_vals[j + 1] = fo_vals[j]
                node_ids[j + 1] = node_ids[j]
                j -= 1
            fo_vals[j + 1] = key_fo
            node_ids[j + 1] = key_id

        total = 0.0
        for i in range(4):
            u_id = node_ids[i]
            v_id = node_ids[(i + 1) & 3]

            fu = int(fo[u_id])
            fv = int(fo[v_id])
            if fu > fv:
                fu, fv = fv, fu

            lca_idx = Tree._rmq(fu, fv, sparse_table, euler_depth, log2_table)
            lca_id = int(euler_tour[lca_idx])
            total += float(rd[u_id]) + float(rd[v_id]) - 2.0 * float(rd[lca_id])

        return p0, p1, q0, q1, total * 0.5
