"""
quarimo/_io.py
==============
File I/O helpers with automatic compression detection.

NewickFile
----------
Context manager that detects compression format from magic bytes and returns a
text-mode ``io.StringIO`` object suitable for reading NEWICK data.

The constructor accepts a file path, raw bytes, or any binary-readable object
(e.g. a member extracted from a ``tarfile`` archive):

- ``NewickFile("run.ufboot.gz")``           — file path
- ``NewickFile(Path("run.ufboot.gz"))``      — pathlib.Path
- ``NewickFile(raw_bytes)``                  — already-read bytes
- ``NewickFile(tar.extractfile(member))``    — binary file-like object

Supported formats
~~~~~~~~~~~~~~~~~
- Plain UTF-8 text  (no magic bytes matched)
- gzip              (magic: ``\\x1f\\x8b``)
- bzip2             (magic: ``BZh``)
- xz / lzma        (magic: ``\\xfd7zXZ\\x00``)
- zstd              (magic: ``\\x28\\xb5\\x2f\\xfd``) — requires ``pip install quarimo[zstd]``

Detection reads the first 6 bytes of the bytestream; filename extensions are ignored.
"""

import bz2
import gzip
import io
import lzma
from pathlib import Path
from typing import BinaryIO, Union

# Accepted source types for the constructor and detect()
_Source = Union[str, Path, bytes, BinaryIO]


class NewickFile:
    """Context manager for reading NEWICK data with automatic decompression.

    Detects compression format from magic bytes (not filename extension) and
    returns a readable, iterable :class:`io.StringIO` object.

    Parameters
    ----------
    source : str, Path, bytes, or binary file-like object
        The NEWICK source.  Accepted forms:

        - A file path (``str`` or :class:`~pathlib.Path`) — read from disk.
        - Raw ``bytes`` — already-read content, possibly compressed.
        - A binary-readable object (anything with a ``.read()`` method that
          returns ``bytes``) — for example, a member extracted from a
          :mod:`tarfile` archive.

    label : str, optional
        Short description of the source used in error messages.  Defaults to
        the file path string, ``"<bytes>"``, or the stream's ``.name``
        attribute (if present).

    Examples
    --------
    Read a gzip-compressed bootstrap file produced by IQ-TREE:

    >>> with NewickFile("run.ufboot.gz") as f:
    ...     text = f.read()

    Iterate line-by-line through an xz-compressed multi-tree file:

    >>> with NewickFile("run.ufboot.xz") as f:
    ...     for line in f:
    ...         process(line.strip())

    Pass directly to Forest (Forest accepts a NEWICK string):

    >>> from quarimo import Forest, NewickFile
    >>> with NewickFile("run.ufboot.gz") as f:
    ...     forest = Forest(f.read())

    Load a gzip-compressed member from a tar.gz archive:

    >>> import tarfile
    >>> with tarfile.open("bootstrap-genetrees.tar.gz", "r:gz") as tar:
    ...     fobj = tar.extractfile("allgenes/intron-10002/RAxML_bootstrap.all.gz")
    ...     with NewickFile(fobj, label="intron-10002") as f:
    ...         forest = Forest(f.read())
    """

    # (format_name, magic_prefix) — checked in order; first match wins.
    # 'xz' prefix is 6 bytes (longest), so _HEADER_BYTES = 6 covers all.
    _SIGNATURES: list[tuple[str, bytes]] = [
        ("gzip",  b"\x1f\x8b"),
        ("bzip2", b"BZh"),
        ("xz",    b"\xfd7zXZ\x00"),
        ("zstd",  b"\x28\xb5\x2f\xfd"),
    ]
    _HEADER_BYTES: int = 6  # length of longest signature

    def __init__(self, source: _Source, *, label: str | None = None) -> None:
        self._source = source
        self._label = label
        self._sio: io.StringIO | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls, source: _Source) -> str:
        """Return the compression format name, or ``'text'`` for plain data.

        Reads only the first :attr:`_HEADER_BYTES` bytes; filename extensions
        are not considered.

        Parameters
        ----------
        source : str, Path, bytes, or binary file-like object
            The data to inspect.  For file-like objects, the first
            :attr:`_HEADER_BYTES` bytes are consumed from the current position.

        Returns
        -------
        str
            One of ``'gzip'``, ``'bzip2'``, ``'xz'``, ``'zstd'``, or
            ``'text'``.
        """
        if isinstance(source, bytes):
            return cls._detect_bytes(source)
        if isinstance(source, (str, Path)):
            with open(source, "rb") as fh:
                header = fh.read(cls._HEADER_BYTES)
            return cls._detect_bytes(header)
        # binary file-like object
        return cls._detect_bytes(source.read(cls._HEADER_BYTES))

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> io.StringIO:
        raw, label = self._read_raw()
        fmt = self._detect_bytes(raw)
        text = self._decode(raw, fmt, label)
        self._sio = io.StringIO(text)
        return self._sio

    def __exit__(self, *_: object) -> None:
        if self._sio is not None:
            self._sio.close()
            self._sio = None

    # ------------------------------------------------------------------
    # Public class methods
    # ------------------------------------------------------------------

    @classmethod
    def decompress_bytes(cls, data: bytes, source: str = "<bytes>") -> str:
        """Detect compression from *data*, decompress, and return UTF-8 text.

        Convenience wrapper for callers that already hold the raw bytes in
        memory and do not need a context manager.

        Parameters
        ----------
        data : bytes
            Raw (possibly compressed) bytes.
        source : str, optional
            Label used in error messages (e.g. the archive member name).

        Returns
        -------
        str
            Decoded UTF-8 text.
        """
        fmt = cls._detect_bytes(data)
        return cls._decode(data, fmt, source)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_raw(self) -> tuple[bytes, str]:
        """Return ``(raw_bytes, label)`` from whichever source was provided."""
        src = self._source
        if isinstance(src, bytes):
            return src, self._label or "<bytes>"
        if isinstance(src, (str, Path)):
            path = Path(src)
            label = self._label or str(path)
            return path.read_bytes(), label
        # binary file-like object
        label = self._label or getattr(src, "name", "<stream>")
        return src.read(), label

    @classmethod
    def _detect_bytes(cls, data: bytes) -> str:
        """Detect format from a byte string (uses the leading bytes only)."""
        for name, sig in cls._SIGNATURES:
            if data.startswith(sig):
                return name
        return "text"

    @classmethod
    def _decompress(cls, raw: bytes, fmt: str, source: str) -> bytes:
        if fmt == "text":
            return raw
        if fmt == "gzip":
            try:
                return gzip.decompress(raw)
            except (gzip.BadGzipFile, EOFError, OSError) as exc:
                raise ValueError(
                    f"{source}: invalid gzip data — {exc}"
                ) from exc
        if fmt == "bzip2":
            try:
                return bz2.decompress(raw)
            except (OSError, EOFError) as exc:
                raise ValueError(
                    f"{source}: invalid bzip2 data — {exc}"
                ) from exc
        if fmt == "xz":
            try:
                return lzma.decompress(raw)
            except (lzma.LZMAError, EOFError) as exc:
                raise ValueError(
                    f"{source}: invalid xz/lzma data — {exc}"
                ) from exc
        if fmt == "zstd":
            try:
                import zstandard  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    f"{source}: zstd-compressed file detected but 'zstandard' is not "
                    "installed. Install it with: pip install quarimo[zstd]"
                ) from exc
            try:
                return zstandard.decompress(raw)
            except Exception as exc:
                raise ValueError(
                    f"{source}: invalid zstd data — {exc}"
                ) from exc
        raise ValueError(f"unknown format: {fmt!r}")  # unreachable

    @classmethod
    def _decode(cls, raw: bytes, fmt: str, source: str) -> str:
        data = cls._decompress(raw, fmt, source)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source}: content is not valid UTF-8 (detected format: {fmt!r})"
            ) from exc
