"""
tests/test_logging.py
=====================
Smoke tests for quarimo._logging functions, and integration tests verifying
that quartet_topology() emits the expected log records for every available
backend.

Test classes
------------
TestLogBackendAvailability
    Verifies that log_backend_availability() emits INFO records, does not
    escalate above INFO, and covers every backend name returned by
    backends.status().

TestQuartetTopologyLogging
    Parametrised over every backend that quartet_topology() actually
    dispatches to (python, cpu-parallel, cuda — mlx excluded until the
    quartet kernel is implemented).  For each backend, both counts-only and
    steiner=True modes are exercised.

    Universal assertion (all backends, both modes):
        The execution-mode banner contains the backend name and the mode
        string ("counts-only" or "Steiner").

    CUDA-specific additional assertions:
        The per-batch kernel-launch line and the D→H transfer summary are
        present.

TestUseKernelFallback
    Verifies the graceful-fallback behaviour of use_kernel():
    - An unregistered kernel name triggers a WARNING on any machine.
    - On Apple Silicon, requesting mlx for quartet_topology falls back to
      cpu-parallel (or python) with a WARNING, and the call still succeeds.
"""

import logging

import pytest

from quarimo._backend import backends
from quarimo._context import quiet, use_backend, use_kernel
from quarimo._forest import Forest
from quarimo._logging import log_backend_availability
from quarimo._quartets import Quartets

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_NEWICKS = [
    "((A:1,B:1):1,(C:1,D:1):1);",
    "((A:1,C:1):1,(B:1,D:1):1);",
    "((A:1,D:1):1,(B:1,C:1):1);",
] * 10  # 30 trees

_QUARTETS = [("A", "B", "C", "D"), ("A", "B", "D", "C")]

# Backends that quartet_topology() dispatches to.  MLX is omitted until the
# full quartet kernel (counts + Steiner) is implemented for Metal.
_QUARTET_TOPOLOGY_BACKENDS = [
    b for b in backends.available() if b in {"python", "cpu-parallel", "cuda"}
]

# Fragment of the universal execution-mode banner emitted by quartet_topology()
# for counts-only and Steiner modes respectively.
_BANNER_COUNTS = "quartet_topology(counts-only"
_BANNER_STEINER = "quartet_topology(Steiner"

# Log fragments expected only from the CUDA backend.
_CUDA_COUNTS_FRAGMENTS = ["launching counts kernel", "D→H total"]
_CUDA_STEINER_FRAGMENTS = ["launching Steiner kernel", "D→H total"]

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forest_and_quartets():
    """30-tree forest and two quartets, constructed silently."""
    with quiet():
        c = Forest(_NEWICKS)
        q = Quartets.from_list(c, _QUARTETS)
    return c, q


# ===========================================================================
# 1. log_backend_availability
# ===========================================================================


class TestLogBackendAvailability:
    """Smoke tests for log_backend_availability()."""

    def test_emits_info_records(self, caplog):
        """log_backend_availability must emit at least one INFO record."""
        with caplog.at_level(logging.INFO, logger="quarimo._logging"):
            log_backend_availability(backends)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info_records, "Expected at least one INFO record"

    def test_no_records_above_info(self, caplog):
        """log_backend_availability must not emit WARNING or higher."""
        with caplog.at_level(logging.DEBUG, logger="quarimo._logging"):
            log_backend_availability(backends)

        noisy = [r for r in caplog.records if r.levelno > logging.INFO]
        assert not noisy, f"Unexpected high-level records: {noisy}"

    def test_output_mentions_all_backends(self, caplog):
        """Every backend name from backends.status() appears in the log output."""
        with caplog.at_level(logging.INFO, logger="quarimo._logging"):
            log_backend_availability(backends)

        full_text = "\n".join(r.getMessage() for r in caplog.records)
        for name, _ in backends.status():
            assert name in full_text, f"Backend '{name}' missing from log output"

    def test_output_mentions_best(self, caplog):
        """The best backend name must appear in the log output."""
        with caplog.at_level(logging.INFO, logger="quarimo._logging"):
            log_backend_availability(backends)

        full_text = "\n".join(r.getMessage() for r in caplog.records)
        assert backends.best() in full_text


# ===========================================================================
# 2. quartet_topology logging — per backend
# ===========================================================================


class TestQuartetTopologyLogging:
    """
    Verify that quartet_topology() emits the expected INFO records for every
    backend that supports it.

    The parametrize list (_QUARTET_TOPOLOGY_BACKENDS) is built from
    backends.available() at collection time, so only hardware-present backends
    are exercised.
    """

    @pytest.mark.parametrize("backend", _QUARTET_TOPOLOGY_BACKENDS)
    def test_counts_mode_logs_banner(self, forest_and_quartets, backend, caplog):
        """counts-only mode must log the execution banner with backend name and mode."""
        c, q = forest_and_quartets
        with caplog.at_level(logging.INFO, logger="quarimo"):
            with use_backend(backend):
                c.quartet_topology(q)

        full_text = "\n".join(r.getMessage() for r in caplog.records)
        assert _BANNER_COUNTS in full_text, (
            f"[{backend}] Missing banner fragment {_BANNER_COUNTS!r}\n\n{full_text}"
        )
        assert backend in full_text, (
            f"[{backend}] Backend name absent from log output\n\n{full_text}"
        )

    @pytest.mark.parametrize("backend", _QUARTET_TOPOLOGY_BACKENDS)
    def test_steiner_mode_logs_banner(self, forest_and_quartets, backend, caplog):
        """steiner=True mode must log the execution banner with backend name and mode."""
        c, q = forest_and_quartets
        with caplog.at_level(logging.INFO, logger="quarimo"):
            with use_backend(backend):
                c.quartet_topology(q, steiner=True)

        full_text = "\n".join(r.getMessage() for r in caplog.records)
        assert _BANNER_STEINER in full_text, (
            f"[{backend}] Missing banner fragment {_BANNER_STEINER!r}\n\n{full_text}"
        )
        assert backend in full_text, (
            f"[{backend}] Backend name absent from log output\n\n{full_text}"
        )

    @pytest.mark.requires_cuda
    def test_cuda_counts_mode_logs_kernel_and_transfer(
        self, forest_and_quartets, caplog
    ):
        """CUDA counts-only mode must log the kernel launch and D→H transfer."""
        c, q = forest_and_quartets
        with caplog.at_level(logging.INFO, logger="quarimo"):
            with use_backend("cuda"):
                c.quartet_topology(q)

        full_text = "\n".join(r.getMessage() for r in caplog.records)
        for fragment in _CUDA_COUNTS_FRAGMENTS:
            assert fragment in full_text, (
                f"[cuda] Missing fragment {fragment!r}\n\n{full_text}"
            )

    @pytest.mark.requires_cuda
    def test_cuda_steiner_mode_logs_kernel_and_transfer(
        self, forest_and_quartets, caplog
    ):
        """CUDA steiner=True mode must log the Steiner kernel launch and D→H transfer."""
        c, q = forest_and_quartets
        with caplog.at_level(logging.INFO, logger="quarimo"):
            with use_backend("cuda"):
                c.quartet_topology(q, steiner=True)

        full_text = "\n".join(r.getMessage() for r in caplog.records)
        for fragment in _CUDA_STEINER_FRAGMENTS:
            assert fragment in full_text, (
                f"[cuda] Missing fragment {fragment!r}\n\n{full_text}"
            )


# ===========================================================================
# 3. use_kernel fallback behaviour
# ===========================================================================


class TestUseKernelFallback:
    """
    Verify that use_kernel() falls back gracefully when the requested backend
    has no implementation for the given kernel, emitting a WARNING each time.
    """

    def test_unregistered_kernel_emits_warning(self, caplog):
        """
        Any backend requested for an unregistered kernel name falls back to
        'python' (the hardcoded last-resort) and emits a WARNING.

        This test works on every machine because it does not require any
        optional hardware — it exercises the fallback logic directly.
        """
        with caplog.at_level(logging.WARNING, logger="quarimo._context"):
            with use_kernel("unregistered_kernel", "python") as b:
                actual = b

        assert actual == "python"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected a WARNING for missing kernel"
        assert "unregistered_kernel" in warnings[0].getMessage()

    @pytest.mark.requires_mlx
    def test_mlx_quartet_topology_falls_back_and_warns(self, forest_and_quartets, caplog):
        """
        On Apple Silicon, requesting mlx for quartet_topology falls back to
        the next-best available backend, emits a WARNING, and the call still
        produces a valid result.
        """
        c, q = forest_and_quartets
        with caplog.at_level(logging.WARNING, logger="quarimo._context"):
            with use_backend("mlx"):
                result = c.quartet_topology(q)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected a WARNING for mlx fallback"
        msg = warnings[0].getMessage()
        assert "mlx" in msg
        assert "quartet_topology" in msg
        # The result must still be a valid QuartetTopologyResult
        assert result.counts.shape[0] == len(q)
