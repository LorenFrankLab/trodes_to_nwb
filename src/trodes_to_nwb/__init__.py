"""Package for converting SpikeGadgets .rec files to NWB 2.0+ format.

This package provides tools to convert electrophysiology data, position tracking,
video files, DIO events, and behavioral metadata from Trodes recording systems
into standardized NWB format for DANDI archive compatibility.
"""

import contextlib

with contextlib.suppress(ImportError):
    from ._version import __version__
