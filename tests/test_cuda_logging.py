"""
test_cuda_logging.py
====================
Validates that expected logging messages are emitted when using the CUDA
backend.  All tests in this module are skipped on machines without CUDA.
"""

import io
import logging

import pytest

from quarimo._forest import Forest
from quarimo._quartets import Quartets

pytestmark = pytest.mark.requires_cuda

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_NEWICKS = [
    "((A:1,B:1):1,(C:1,D:1):1);",
    "((A:1,C:1):1,(B:1,D:1):1);",
    "((A:1,D:1):1,(B:1,C:1):1);",
] * 10  # 30 trees

_QUARTETS = [("A", "B", "C", "D"), ("A", "B", "D", "C")]

_EXPECTED_CUDA_MESSAGES = [
    "Transferring data to GPU device:",
    "Tree data:",
    "global_to_local:",
    "CSR packed arrays:",
    "Query data:",
    "sorted_quartet_ids:",
    "Total H→D transfer:",
    "Launching CUDA kernel:",
    "Grid:",
    "Total threads:",
    "Transferring results from GPU device:",
    "counts_out:",
    "Total D→H transfer:",
]

_STEINER_MESSAGES = [
    "Output arrays:",
    "steiner_out",
    "float64",
]


@pytest.fixture(scope="module")
def forest_and_quartets():
    c = Forest(_NEWICKS)
    q = Quartets.from_list(c, _QUARTETS)
    return c, q


def _capture_cuda_log(forest, quartets, **kwargs):
    """Run quartet_topology with CUDA backend and return captured log text."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("phylo_tree_collection")
    logger.addHandler(handler)
    try:
        forest.quartet_topology(quartets, backend="cuda", **kwargs)
    finally:
        logger.removeHandler(handler)
    return buf.getvalue()


class TestCUDALogging:
    def test_counts_mode_logs_expected_messages(self, forest_and_quartets):
        """Counts-only CUDA call must log H→D transfer, kernel launch, D→H transfer."""
        c, q = forest_and_quartets
        logs = _capture_cuda_log(c, q)
        missing = [msg for msg in _EXPECTED_CUDA_MESSAGES if msg not in logs]
        assert not missing, f"Missing log messages: {missing}\n\nFull log:\n{logs}"

    def test_steiner_mode_logs_output_arrays(self, forest_and_quartets):
        """Steiner CUDA call must additionally log steiner_out array info."""
        c, q = forest_and_quartets
        logs = _capture_cuda_log(c, q, steiner=True)
        missing = [msg for msg in _STEINER_MESSAGES if msg not in logs]
        assert not missing, (
            f"Missing Steiner log messages: {missing}\n\nFull log:\n{logs}"
        )
