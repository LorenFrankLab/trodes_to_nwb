"""Boundary / multi-file assembly tests for ``RecFileDataChunkIterator._get_data``.

These tests drive the *real* ``_get_data`` index logic and the *real* hdmf
``GenericDataChunkIterator`` buffer/chunk machinery, but substitute a lightweight
fake ``neo_io`` that returns a deterministic "ramp" (the value at global sample
index ``g`` is ``g`` on every channel). That makes any dropped or misaligned
sample directly observable, while letting us control the file sizes and the
buffer alignment cheaply (a real ``.rec`` would need hundreds of millions of
samples to exercise the same buffer edges with the default buffer size).

Background (#171): the old assembly loop used a strict terminal condition
``while i < time_index[-1]`` (the last *index*, not the stop). For ordinary
buffer-aligned, boundary-spanning reads it tiled correctly, but a buffer
selection that was a single sample, or one whose final sample was the first
sample of the next file, was not swept up by a chunk's ``+1`` and the loop
exited early -- ``_get_data`` returned too few rows and the HDF5 write raised
(``need at least one array to concatenate`` for the single-sample case, or a
broadcast error for the boundary case). The loop now reads the contiguous
``[start, stop)`` range file-by-file, so every config below must reassemble the
ramp exactly; the last four configs are the alignments that used to crash.
"""

import numpy as np
import pytest
from hdmf.data_utils import GenericDataChunkIterator

from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator

N_CHANNEL = 4


class _RampNeoIO:
    """Minimal fake neo_io: returns ramp data (global sample index) per channel.

    Mimics neo's ``get_analogsignal_chunk`` clipping ``i_stop`` to the file size.
    """

    def __init__(self, n_time: int, offset: int):
        self.n_time = int(n_time)
        self.offset = int(offset)
        # _get_data reads neo_io[0].header[...] before the `and self.is_analog`
        # short-circuit, so a non-ECU stream id must be present.
        self.header = {"signal_streams": [{"id": "trodes"}]}

    def get_analogsignal_chunk(
        self, *, block_index, seg_index, i_start, i_stop, stream_index, channel_ids
    ):
        i_start = max(0, int(i_start))
        i_stop = min(int(i_stop), self.n_time)  # neo clips to available samples
        g = self.offset + np.arange(i_start, i_stop)  # global indices
        return np.tile(g.reshape(-1, 1), (1, len(channel_ids))).astype(np.int16)


def _make_iterator(n_time, chunk_t, buffer_t):
    """Construct a real RecFileDataChunkIterator wired to ramp fakes.

    ``__init__`` is bypassed (it parses real .rec headers); we set only the
    attributes ``_get_data`` / ``_get_maxshape`` / ``_get_dtype`` use, then run
    the genuine ``GenericDataChunkIterator`` setup.
    """
    it = RecFileDataChunkIterator.__new__(RecFileDataChunkIterator)
    it.is_analog = False
    it.conversion = 1.0
    it.block_index = 0
    it.seg_index = 0
    it.stream_index = 0
    it.n_channel = N_CHANNEL
    it.n_multiplexed_channel = 0
    it.nwb_hw_channel_order = np.arange(N_CHANNEL)
    it.n_time = list(n_time)
    starts = np.append(0, np.cumsum(n_time))[:-1]
    it.neo_io = [_RampNeoIO(nt, off) for nt, off in zip(n_time, starts)]
    it.timestamps = np.arange(sum(n_time), dtype=float)
    GenericDataChunkIterator.__init__(
        it, chunk_shape=(chunk_t, N_CHANNEL), buffer_shape=(buffer_t, N_CHANNEL)
    )
    return it


def _assemble(n_time, chunk_t, buffer_t):
    """Iterate the DCI and write each chunk into a preallocated array.

    This mirrors exactly how the HDF5 backend consumes a GenericDataChunkIterator
    (``dataset[chunk.selection] = chunk.data``), so a wrong-shaped chunk raises
    here just as it would during a real conversion. Returns channel 0 of the
    assembled array, which should equal ``arange(total)``.
    """
    total = sum(n_time)
    it = _make_iterator(n_time, chunk_t, buffer_t)
    arr = np.full((total, N_CHANNEL), -1, dtype=np.int16)
    for chunk in it:
        arr[chunk.selection] = chunk.data
    return arr[:, 0]


# Correctness must hold for every config. The last four configs are the
# alignments that regressed before #171 was fixed (single-sample final buffer,
# a selection ending one sample past a file boundary, and a buffer spanning into
# a new file) -- they used to crash; they must now reassemble exactly.
_CONFIGS = [
    pytest.param([100, 80, 120], 50, 50, id="three_files_tiled"),
    pytest.param([100, 100, 95], 64, 64, id="tail_remainder_not_one"),
    pytest.param([8, 8, 8], 6, 6, id="three_equal_files"),
    pytest.param([300], 64, 64, id="single_file"),
    pytest.param([100, 100, 101], 100, 100, id="single_sample_last_buffer"),
    pytest.param([201], 100, 100, id="single_file_one_sample_tail"),
    pytest.param([101, 99], 51, 102, id="buffer_ends_one_past_boundary"),
    pytest.param([7, 5, 9, 3, 11], 5, 5, id="many_small_files_small_buffer"),
]


@pytest.mark.parametrize("n_time,chunk_t,buffer_t", _CONFIGS)
def test_get_data_reassembles_full_ramp(n_time, chunk_t, buffer_t):
    """Every sample must be read exactly once, in order, across file boundaries."""
    total = sum(n_time)
    col0 = _assemble(n_time, chunk_t, buffer_t)
    np.testing.assert_array_equal(col0, np.arange(total, dtype=np.int16))
