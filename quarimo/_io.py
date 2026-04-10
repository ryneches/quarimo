"""
quarimo/_io.py
==============
File I/O helpers with automatic compression detection.

NewickFile
----------
Context manager that detects compression format from file magic bytes and
returns a text-mode ``io.StringIO`` object suitable for reading NEWICK data.

Supported formats
~~~~~~~~~~~~~~~~~
- Plain UTF-8 text  (no magic bytes matched)
- gzip              (magic: ``\\x1f\\x8b``)
- bzip2             (magic: ``BZh``)
- xz / lzma        (magic: ``\\xfd7zXZ\\x00``)
- zstd              (magic: ``\\x28\\xb5\\x2f\\xfd``) — requires ``pip install quarimo[zstd]``

Detection reads the first 6 bytes of the file; filename extensions are ignored.
"""

import bz2
import gzip
import io
import lzma
from pathlib import Path


class NewickFile:
    """Context manager for reading NEWICK files with automatic decompression.

    Detects compression format from magic bytes (not filename extension) and
    returns a readable, iterable :class:`io.StringIO` object.

    Parameters
    ----------
    path : str or Path
        Path to the file to open.

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

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._sio: io.StringIO | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls, path: str | Path) -> str:
        """Return the compression format name, or ``'text'`` for plain files.

        Reads only the first :attr:`_HEADER_BYTES` bytes of the file;
        filename extensions are not considered.

        Parameters
        ----------
        path : str or Path
            File to inspect.

        Returns
        -------
        str
            One of ``'gzip'``, ``'bzip2'``, ``'xz'``, ``'zstd'``, or
            ``'text'``.
        """
        with open(path, "rb") as fh:
            header = fh.read(cls._HEADER_BYTES)
        return cls._detect_bytes(header)

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> io.StringIO:
        raw = self._path.read_bytes()
        fmt = self._detect_bytes(raw)
        text = self._decode(raw, fmt, str(self._path))
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

        Useful when compressed content has already been read into memory —
        for example, a member extracted from a ``tarfile`` archive — where
        no file path exists to pass to the :class:`NewickFile` constructor.

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

        Examples
        --------
        Load every gene-family bootstrap file from a tar.gz archive into a
        grouped :class:`~quarimo.Forest`:

        >>> import tarfile
        >>> from pathlib import PurePosixPath
        >>> from quarimo import Forest, NewickFile
        >>>
        >>> trees = {}
        >>> with tarfile.open("bootstrap-genetrees.tar.gz", "r:gz") as tar:
        ...     for member in tar.getmembers():
        ...         parts = PurePosixPath(member.name).parts
        ...         if len(parts) != 3:
        ...             continue
        ...         locus = parts[1]
        ...         fobj = tar.extractfile(member)
        ...         if fobj is None:
        ...             continue
        ...         text = NewickFile.decompress_bytes(fobj.read(), source=member.name)
        ...         trees[locus] = text.strip().splitlines()
        >>> forest = Forest(trees)
        """
        fmt = cls._detect_bytes(data)
        return cls._decode(data, fmt, source)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
