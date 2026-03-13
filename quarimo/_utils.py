"""
_utils.py
=========
General-purpose utility functions for quarimo.

These are standalone functions that don't depend on the main classes
and could be useful in multiple contexts.
"""

from pathlib import Path
from typing import Set, TypeVar, Union


T = TypeVar('T')


def jaccard_similarity(set_a: Set[T], set_b: Set[T]) -> float:
    """
    Compute Jaccard similarity coefficient between two sets.
    
    The Jaccard similarity is the size of the intersection divided by the
    size of the union of the two sets. It ranges from 0 (completely disjoint)
    to 1 (identical sets).
    
    Parameters
    ----------
    set_a, set_b : Set[T]
        Two sets to compare. Can contain any hashable type.
    
    Returns
    -------
    float
        Jaccard similarity in [0, 1].
        Returns 0.0 if both sets are empty (union size is 0).
    
    Examples
    --------
    >>> jaccard_similarity({1, 2, 3}, {1, 2, 3})
    1.0
    
    >>> jaccard_similarity({1, 2}, {3, 4})
    0.0
    
    >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
    0.5
    
    >>> jaccard_similarity(set(), set())
    0.0
    
    >>> jaccard_similarity({'A', 'B'}, {'B', 'C', 'D'})
    0.25
    
    Notes
    -----
    This function is used internally for computing taxa overlap between
    tree groups, but is exposed as a public utility since it's generally
    useful for set comparison.
    
    The implementation handles edge cases:
    - Empty sets: Returns 0.0 (convention: empty sets are maximally dissimilar)
    - One empty set: Returns 0.0 (no overlap possible)
    - Identical sets: Returns 1.0 (complete overlap)
    
    Mathematical definition:
        J(A, B) = |A ∩ B| / |A ∪ B|
    
    where |X| denotes the cardinality (size) of set X.
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# Backward compatibility alias
_jaccard = jaccard_similarity


def validate_quartet(quartet: tuple) -> bool:
    """
    Validate that a quartet specification is well-formed.
    
    A valid quartet must:
    1. Be a tuple or list
    2. Have exactly 4 elements
    3. All elements must be distinct
    
    Parameters
    ----------
    quartet : tuple
        Quartet specification (taxon_a, taxon_b, taxon_c, taxon_d).
        Elements can be strings (taxon names) or integers (taxon IDs).
    
    Returns
    -------
    bool
        True if quartet is valid, False otherwise.
    
    Examples
    --------
    >>> validate_quartet(('A', 'B', 'C', 'D'))
    True
    
    >>> validate_quartet((0, 1, 2, 3))
    True
    
    >>> validate_quartet(('A', 'B', 'C'))  # Too few
    False
    
    >>> validate_quartet(('A', 'B', 'C', 'D', 'E'))  # Too many
    False
    
    >>> validate_quartet(('A', 'A', 'C', 'D'))  # Duplicate
    False
    
    >>> validate_quartet(['A', 'B', 'C', 'D'])  # List works too
    True
    """
    # Check type
    if not isinstance(quartet, (tuple, list)):
        return False
    
    # Check size
    if len(quartet) != 4:
        return False
    
    # Check uniqueness
    if len(set(quartet)) != 4:
        return False
    
    return True


def format_newick(newick: str) -> str:
    """
    Format a NEWICK string for consistent representation.

    Ensures the NEWICK string:
    - Ends with a semicolon
    - Has no leading/trailing whitespace

    Parameters
    ----------
    newick : str
        NEWICK string to format.

    Returns
    -------
    str
        Formatted NEWICK string.

    Examples
    --------
    >>> format_newick('((A:1,B:1):1,(C:1,D:1):1)')
    '((A:1,B:1):1,(C:1,D:1):1);'

    >>> format_newick('  ((A:1,B:1):1);  ')
    '((A:1,B:1):1);'

    >>> format_newick('((A:1,B:1):1);')
    '((A:1,B:1):1);'
    """
    newick = newick.strip()
    if not newick.endswith(';'):
        newick += ';'
    return newick


# ---------------------------------------------------------------------------
# NEWICK input validation and normalization
# ---------------------------------------------------------------------------


def validate_newick(s: str, context: str = "") -> str:
    """
    Check whether a string is structurally plausible as a single NEWICK tree.

    NEWICK is an informal family of formats used by many phylogenetics tools,
    and quarimo implements one dialect of it.  This function performs lightweight
    structural checks on a single-tree string; it does *not* guarantee that
    quarimo's parser will accept every string that passes these checks.

    Parameters
    ----------
    s : str
        A single NEWICK tree string (must not contain embedded newlines).
    context : str, optional
        A short description of where the string came from (e.g. a file name or
        a dict key), included in error messages to aid diagnosis.

    Returns
    -------
    str
        The stripped input string, unchanged.

    Raises
    ------
    ValueError
        If the string is empty, appears to contain multiple trees, does not end
        with a semicolon after stripping, or has unbalanced parentheses.
    """
    prefix = f"{context}: " if context else ""
    s = s.strip()

    if not s:
        raise ValueError(f"{prefix}empty string where a NEWICK tree was expected")

    n_semi = s.count(";")
    if n_semi > 1:
        raise ValueError(
            f"{prefix}found {n_semi} semicolons in a single tree string — "
            "this may be a multi-tree block; use normalize_input() to split it first"
        )

    if n_semi == 0 or not s.endswith(";"):
        raise ValueError(
            f"{prefix}NEWICK tree string does not end with ';' — "
            "quarimo expects each tree to be terminated by a semicolon"
        )

    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(
                    f"{prefix}unmatched ')' in NEWICK string — "
                    "the parentheses do not balance"
                )
    if depth != 0:
        raise ValueError(
            f"{prefix}unmatched '(' in NEWICK string — "
            "the parentheses do not balance"
        )

    return s


def _split_newick_block(text: str, context: str = "") -> list:
    """
    Split a multi-tree text block into individual NEWICK strings.

    Each non-empty line (after stripping whitespace and comment lines beginning
    with '#') is treated as one tree.  Lines are validated with
    ``validate_newick`` before being returned.

    Parameters
    ----------
    text : str
        Block of text, typically read from a file.
    context : str, optional
        Description of the source for error messages.

    Returns
    -------
    list[str]
        Validated single-tree NEWICK strings.
    """
    trees = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line_ctx = f"{context} line {i}" if context else f"line {i}"
        trees.append(validate_newick(line, context=line_ctx))
    if not trees:
        raise ValueError(
            f"{context + ': ' if context else ''}no NEWICK trees found in text block"
        )
    return trees


def _load_newick_file(path: Path, context: str = "") -> list:
    """
    Read a file and return a list of NEWICK tree strings.

    The file may contain one tree per line (e.g. IQ-TREE ``.ufboot`` bootstrap
    files) or a single tree on one line.  Blank lines and lines beginning with
    ``#`` are ignored.

    Parameters
    ----------
    path : Path
        Path to the file to read.
    context : str, optional
        Description for error messages; defaults to the file path string.

    Returns
    -------
    list[str]
        Validated single-tree NEWICK strings.
    """
    ctx = context or str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{ctx}: cannot read file — {exc}") from exc
    return _split_newick_block(text, context=ctx)


def _normalize_atom(item: Union[str, Path], context: str = "") -> list:
    """
    Normalize a single item (str or Path) to a list of NEWICK strings.

    Dispatch rules for *str* input:
    - Contains ``'\\n'``: treat as a multi-tree text block.
    - Starts with ``'('``: treat as a single NEWICK string.
    - Resolves to an existing file path: read as a NEWICK file.
    - Otherwise: attempt ``validate_newick`` and return as a single-element list.

    ``pathlib.Path`` objects are always treated as file paths.

    Parameters
    ----------
    item : str or Path
        One element from a group's value list, or the group value itself.
    context : str, optional
        Description for error messages.

    Returns
    -------
    list[str]
        One or more validated NEWICK strings.
    """
    if isinstance(item, Path):
        return _load_newick_file(item, context=context or str(item))

    # str dispatch
    if "\n" in item:
        return _split_newick_block(item, context=context)

    if item.lstrip().startswith("("):
        return [validate_newick(item, context=context)]

    p = Path(item)
    if p.exists():
        return _load_newick_file(p, context=context or str(p))

    # Last resort: try to validate as-is (handles trees without leading '(', e.g. "A;")
    return [validate_newick(item, context=context)]


def _normalize_group_value(value, group_name: str) -> list:
    """
    Normalize the value for one group to a flat list of NEWICK strings.

    Accepted forms:
    - A single ``str`` or ``Path`` — see ``_normalize_atom``.
    - A ``list`` or ``tuple`` of ``str`` / ``Path`` — each element is
      normalized with ``_normalize_atom`` and the results are concatenated.

    Parameters
    ----------
    value : str, Path, list, or tuple
        The trees for one group.
    group_name : str
        Used in error messages.

    Returns
    -------
    list[str]
        Flat list of validated NEWICK strings.
    """
    ctx = f"group '{group_name}'"
    if isinstance(value, (str, Path)):
        return _normalize_atom(value, context=ctx)
    if isinstance(value, (list, tuple)):
        trees = []
        for i, item in enumerate(value):
            item_ctx = f"{ctx}[{i}]"
            trees.extend(_normalize_atom(item, context=item_ctx))
        if not trees:
            raise ValueError(f"Group '{group_name}' is empty")
        return trees
    raise TypeError(
        f"{ctx}: expected str, Path, list, or tuple; "
        f"got {type(value).__name__}"
    )


def normalize_input(newick_input) -> dict:
    """
    Normalize any supported input form to ``dict[str, list[str]]``.

    This is the single entry point for all input normalization in quarimo.
    It accepts the same forms that ``Forest.__init__`` accepts, plus several
    convenience forms:

    ==================  =====================================================
    Input type          Interpretation
    ==================  =====================================================
    ``dict``            Keys become group labels; each value is normalized
                        with ``_normalize_group_value``.
    ``list`` / ``tuple``  A single unnamed group; each element is normalized
                        with ``_normalize_atom``.
    ``str``             A single unnamed group containing one or more trees:
                        multi-line → split; starts with ``(`` → single tree;
                        valid file path → read file.
    ``Path``            A single unnamed group; file is read and split.
    ==================  =====================================================

    For ``list`` / ``str`` / ``Path`` input, the group label is derived
    deterministically from the content (a short BLAKE2b hex digest), matching
    the existing ``Forest.__init__`` behaviour.

    Parameters
    ----------
    newick_input : dict, list, tuple, str, or Path
        Trees to load.

    Returns
    -------
    dict[str, list[str]]
        Mapping of group label → list of validated NEWICK strings.

    Raises
    ------
    TypeError
        If the top-level type is not one of the accepted forms.
    ValueError
        If any individual tree string fails structural validation.
    """
    import hashlib

    if isinstance(newick_input, dict):
        return {k: _normalize_group_value(v, group_name=k) for k, v in newick_input.items()}

    # Normalize to a flat list first
    if isinstance(newick_input, (str, Path)):
        flat = _normalize_atom(newick_input, context="input")
    elif isinstance(newick_input, (list, tuple)):
        flat = []
        for i, item in enumerate(newick_input):
            flat.extend(_normalize_atom(item, context=f"input[{i}]"))
    else:
        raise TypeError(
            f"newick_input must be dict, list, tuple, str, or Path; "
            f"got {type(newick_input).__name__}"
        )

    # Derive a deterministic label from content (mirrors Forest.__init__ behaviour)
    combined = "".join(sorted(flat))
    h = hashlib.blake2b(combined.encode("utf-8"), digest_size=5)
    label = h.hexdigest()
    return {label: flat}
