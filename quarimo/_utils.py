"""
_utils.py
=========
General-purpose utility functions for phylo_tree_collection.

These are standalone functions that don't depend on the main classes
and could be useful in multiple contexts.
"""

from typing import Set, TypeVar


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
