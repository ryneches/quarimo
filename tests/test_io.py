"""
tests/test_io.py
================
Tests for quarimo._io.NewickFile — automatic compression detection and
decompression for NEWICK input files.

All format-round-trip tests compress the same source file (basic_collection.trees,
3 NEWICK trees) and verify that the decompressed content is byte-for-byte identical
to the original.

Bytestream input tests verify that the constructor accepts raw ``bytes`` and binary
file-like objects in addition to file paths, so that archive-extracted content can
be passed directly without an intermediate write to disk.
"""

import bz2
import builtins
import gzip
import io
import lzma
from pathlib import Path

import pytest

from quarimo import Forest, NewickFile

# ---------------------------------------------------------------------------
# Source data shared across all tests
# ---------------------------------------------------------------------------

_TREES_FILE = Path(__file__).parent / "trees" / "basic_collection.trees"
_PLAIN_BYTES = _TREES_FILE.read_bytes()
_PLAIN_TEXT = _PLAIN_BYTES.decode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures — one per compression format
# ---------------------------------------------------------------------------


@pytest.fixture
def plain_file(tmp_path):
    p = tmp_path / "trees.nwk"
    p.write_bytes(_PLAIN_BYTES)
    return p


@pytest.fixture
def gz_file(tmp_path):
    p = tmp_path / "trees.nwk.gz"
    p.write_bytes(gzip.compress(_PLAIN_BYTES))
    return p


@pytest.fixture
def bz2_file(tmp_path):
    p = tmp_path / "trees.nwk.bz2"
    p.write_bytes(bz2.compress(_PLAIN_BYTES))
    return p


@pytest.fixture
def xz_file(tmp_path):
    p = tmp_path / "trees.nwk.xz"
    p.write_bytes(lzma.compress(_PLAIN_BYTES))
    return p


@pytest.fixture
def zstd_file(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    p = tmp_path / "trees.nwk.zst"
    p.write_bytes(zstandard.compress(_PLAIN_BYTES))
    return p


# ---------------------------------------------------------------------------
# detect() — magic byte identification
# ---------------------------------------------------------------------------


def test_detect_text(plain_file):
    assert NewickFile.detect(plain_file) == "text"


def test_detect_gzip(gz_file):
    assert NewickFile.detect(gz_file) == "gzip"


def test_detect_bzip2(bz2_file):
    assert NewickFile.detect(bz2_file) == "bzip2"


def test_detect_xz(xz_file):
    assert NewickFile.detect(xz_file) == "xz"


@pytest.mark.requires_zstd
def test_detect_zstd(zstd_file):
    assert NewickFile.detect(zstd_file) == "zstd"


def test_detect_ignores_extension(tmp_path):
    """A gzip file with a .txt extension is still detected as gzip."""
    p = tmp_path / "misleading_name.txt"
    p.write_bytes(gzip.compress(_PLAIN_BYTES))
    assert NewickFile.detect(p) == "gzip"


def test_detect_str_path(plain_file):
    """detect() accepts a plain string path, not just Path objects."""
    assert NewickFile.detect(str(plain_file)) == "text"


# ---------------------------------------------------------------------------
# Read round-trips — content must match original
# ---------------------------------------------------------------------------


def test_read_text(plain_file):
    with NewickFile(plain_file) as f:
        assert f.read() == _PLAIN_TEXT


def test_read_gzip(gz_file):
    with NewickFile(gz_file) as f:
        assert f.read() == _PLAIN_TEXT


def test_read_bzip2(bz2_file):
    with NewickFile(bz2_file) as f:
        assert f.read() == _PLAIN_TEXT


def test_read_xz(xz_file):
    with NewickFile(xz_file) as f:
        assert f.read() == _PLAIN_TEXT


@pytest.mark.requires_zstd
def test_read_zstd(zstd_file):
    with NewickFile(zstd_file) as f:
        assert f.read() == _PLAIN_TEXT


# ---------------------------------------------------------------------------
# File-like object behaviour
# ---------------------------------------------------------------------------


def test_read_iterable(gz_file):
    """The returned object must support line-by-line iteration."""
    with NewickFile(gz_file) as f:
        lines_iter = [line.rstrip("\n") for line in f]
    lines_split = [l for l in _PLAIN_TEXT.splitlines()]
    assert lines_iter == lines_split


def test_read_readline(gz_file):
    """readline() must return one line at a time."""
    with NewickFile(gz_file) as f:
        first = f.readline()
    assert first == _PLAIN_TEXT.splitlines(keepends=True)[0]


def test_read_seek(gz_file):
    """seek(0) must allow re-reading from the start."""
    with NewickFile(gz_file) as f:
        first_read = f.read()
        f.seek(0)
        second_read = f.read()
    assert first_read == second_read


def test_returns_stringio(plain_file):
    """__enter__ must return an io.StringIO instance."""
    with NewickFile(plain_file) as f:
        assert isinstance(f, io.StringIO)


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------


def test_context_manager_closes(plain_file):
    """The StringIO must be closed after the with block exits normally."""
    with NewickFile(plain_file) as f:
        sio = f
    assert sio.closed


def test_context_manager_closes_on_exception(plain_file):
    """The StringIO must be closed even if the body raises."""
    nf = NewickFile(plain_file)
    with pytest.raises(RuntimeError):
        with nf as f:
            sio = f
            raise RuntimeError("intentional")
    assert sio.closed


def test_str_path_accepted(tmp_path):
    """NewickFile must accept a plain string path."""
    p = tmp_path / "trees.nwk"
    p.write_bytes(_PLAIN_BYTES)
    with NewickFile(str(p)) as f:
        assert f.read() == _PLAIN_TEXT


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_invalid_utf8_plain(tmp_path):
    """A plain (non-compressed) file with invalid UTF-8 raises ValueError."""
    p = tmp_path / "bad.nwk"
    p.write_bytes(b"\xff\xfe invalid utf-8 here")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        with NewickFile(p) as f:
            f.read()


def test_invalid_utf8_after_decompression(tmp_path):
    """A valid gzip file whose payload is not UTF-8 raises ValueError."""
    p = tmp_path / "bad.nwk.gz"
    p.write_bytes(gzip.compress(b"\xff\xfe not utf-8"))
    with pytest.raises(ValueError, match="not valid UTF-8"):
        with NewickFile(p) as f:
            f.read()


def test_truncated_gzip(tmp_path):
    """A truncated gzip file (valid magic, bad payload) raises ValueError."""
    p = tmp_path / "truncated.gz"
    p.write_bytes(b"\x1f\x8b" + b"\x00" * 4)  # magic + garbage
    with pytest.raises(ValueError, match="invalid gzip"):
        with NewickFile(p) as f:
            f.read()


def test_truncated_xz(tmp_path):
    """A truncated xz file raises ValueError."""
    p = tmp_path / "truncated.xz"
    p.write_bytes(b"\xfd7zXZ\x00" + b"\x00" * 4)
    with pytest.raises(ValueError, match="invalid xz"):
        with NewickFile(p) as f:
            f.read()


def test_unrecognized_binary(tmp_path):
    """Binary content with no matching magic and invalid UTF-8 raises ValueError."""
    p = tmp_path / "mystery.bin"
    p.write_bytes(bytes(range(256)))  # all byte values, clearly not UTF-8
    with pytest.raises(ValueError, match="not valid UTF-8"):
        with NewickFile(p) as f:
            f.read()


def test_zstd_missing_import(tmp_path, monkeypatch):
    """When zstandard is absent, a zstd file raises ImportError with install hint."""
    p = tmp_path / "fake.zst"
    # Write valid zstd magic; decompression won't be reached before the ImportError
    p.write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 16)

    real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "zstandard":
            raise ImportError("No module named 'zstandard'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    with pytest.raises(ImportError, match="quarimo\\[zstd\\]"):
        with NewickFile(p) as f:
            f.read()


# ---------------------------------------------------------------------------
# Integration — Forest accepts compressed files end-to-end
# ---------------------------------------------------------------------------


def test_forest_reads_gzip(gz_file):
    """Forest can load a gzip-compressed multi-tree file via NewickFile."""
    with NewickFile(gz_file) as f:
        forest = Forest(f.read())
    assert forest.n_trees == 3


def test_forest_reads_bzip2(bz2_file):
    with NewickFile(bz2_file) as f:
        forest = Forest(f.read())
    assert forest.n_trees == 3


def test_forest_reads_xz(xz_file):
    with NewickFile(xz_file) as f:
        forest = Forest(f.read())
    assert forest.n_trees == 3


@pytest.mark.requires_zstd
def test_forest_reads_zstd(zstd_file):
    with NewickFile(zstd_file) as f:
        forest = Forest(f.read())
    assert forest.n_trees == 3


def test_load_newick_file_reads_gzip(gz_file):
    """Forest() with a path to a gz file works end-to-end via _load_newick_file."""
    forest = Forest(gz_file)
    assert forest.n_trees == 3


def test_load_newick_file_reads_xz(xz_file):
    forest = Forest(xz_file)
    assert forest.n_trees == 3


# ---------------------------------------------------------------------------
# Bytestream input — bytes and binary file-like objects
# ---------------------------------------------------------------------------


def test_bytes_input_plain():
    """NewickFile accepts raw bytes for a plain (uncompressed) source."""
    with NewickFile(_PLAIN_BYTES) as f:
        assert f.read() == _PLAIN_TEXT


def test_bytes_input_gzip():
    """NewickFile accepts gzip-compressed bytes."""
    compressed = gzip.compress(_PLAIN_BYTES)
    with NewickFile(compressed) as f:
        assert f.read() == _PLAIN_TEXT


def test_bytes_input_xz():
    """NewickFile accepts xz-compressed bytes."""
    compressed = lzma.compress(_PLAIN_BYTES)
    with NewickFile(compressed) as f:
        assert f.read() == _PLAIN_TEXT


def test_bytes_input_bzip2():
    """NewickFile accepts bzip2-compressed bytes."""
    compressed = bz2.compress(_PLAIN_BYTES)
    with NewickFile(compressed) as f:
        assert f.read() == _PLAIN_TEXT


@pytest.mark.requires_zstd
def test_bytes_input_zstd():
    """NewickFile accepts zstd-compressed bytes."""
    zstandard = pytest.importorskip("zstandard")
    compressed = zstandard.compress(_PLAIN_BYTES)
    with NewickFile(compressed) as f:
        assert f.read() == _PLAIN_TEXT


def test_stream_input_plain():
    """NewickFile accepts a binary file-like object (BytesIO)."""
    stream = io.BytesIO(_PLAIN_BYTES)
    with NewickFile(stream) as f:
        assert f.read() == _PLAIN_TEXT


def test_stream_input_gzip():
    """NewickFile accepts a binary stream containing gzip-compressed data."""
    stream = io.BytesIO(gzip.compress(_PLAIN_BYTES))
    with NewickFile(stream) as f:
        assert f.read() == _PLAIN_TEXT


def test_stream_input_label():
    """label kwarg overrides the default error-message label for stream sources."""
    stream = io.BytesIO(gzip.compress(_PLAIN_BYTES))
    nf = NewickFile(stream, label="my-archive/member.gz")
    with nf as f:
        assert f.read() == _PLAIN_TEXT


def test_bytes_input_label():
    """label kwarg overrides the default '<bytes>' label."""
    nf = NewickFile(_PLAIN_BYTES, label="in-memory-buffer")
    with nf as f:
        assert f.read() == _PLAIN_TEXT


def test_bytes_invalid_utf8_raises():
    """Bytes with invalid UTF-8 payload raises ValueError."""
    with pytest.raises(ValueError, match="not valid UTF-8"):
        with NewickFile(b"\xff\xfe invalid utf-8") as f:
            f.read()


def test_stream_invalid_utf8_raises():
    """Stream with invalid UTF-8 payload raises ValueError."""
    with pytest.raises(ValueError, match="not valid UTF-8"):
        with NewickFile(io.BytesIO(b"\xff\xfe invalid utf-8")) as f:
            f.read()


def test_detect_bytes():
    """detect() accepts raw bytes."""
    assert NewickFile.detect(gzip.compress(_PLAIN_BYTES)) == "gzip"
    assert NewickFile.detect(_PLAIN_BYTES) == "text"


def test_detect_stream():
    """detect() accepts a binary file-like object, consuming only the header bytes."""
    stream = io.BytesIO(gzip.compress(_PLAIN_BYTES))
    assert NewickFile.detect(stream) == "gzip"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_public_api_import():
    """NewickFile must be importable directly from quarimo."""
    from quarimo import NewickFile as NF  # noqa: PLC0415

    assert NF is NewickFile
