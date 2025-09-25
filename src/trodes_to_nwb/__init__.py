import contextlib

with contextlib.suppress(ImportError):
    from ._version import __version__
