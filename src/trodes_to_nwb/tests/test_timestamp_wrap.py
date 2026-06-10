"""uint32 Trodes-timestamp wrap handling in get_regressed_systime (#169).

The Trodes per-packet timestamp is a uint32 sample counter that rolls over after
2**32 samples (~39.77 h at 30 kHz). Casting it to float and regressing system
time on it used to produce silently corrupt (non-monotonic) timestamps once a
recording crossed the wrap. These tests cover the unwrap helper, the full-file
fit across a wrap, and -- the subtle part -- that a post-wrap *partial* iterator
(which inherits the full-file fit) lands on the same global time axis.
"""

import numpy as np
import pytest

from trodes_to_nwb.spike_gadgets_raw_io import (
    UINT32_WRAP,
    SpikeGadgetsRawIO,
    _unwrap_uint32,
)

FS = 30000.0  # Hz


# --------------------------------------------------------------------------- #
# _unwrap_uint32
# --------------------------------------------------------------------------- #
def test_unwrap_no_wrap_is_identity():
    v = np.array([10, 11, 12, 20], dtype=np.uint32)
    np.testing.assert_array_equal(_unwrap_uint32(v), [10, 11, 12, 20])


def test_unwrap_single_wrap():
    v = np.array([UINT32_WRAP - 2, UINT32_WRAP - 1, 0, 1], dtype=np.uint32)
    np.testing.assert_array_equal(
        _unwrap_uint32(v),
        [UINT32_WRAP - 2, UINT32_WRAP - 1, UINT32_WRAP, UINT32_WRAP + 1],
    )


def test_unwrap_multiple_wraps_is_monotonic():
    # a monotone counter with a large per-sample step so it wraps several times
    # within a small array (sample 5 crosses 2**32, sample 9 crosses 2*2**32)
    true = np.arange(12, dtype=np.int64) * 10**9
    wrapped = (true % UINT32_WRAP).astype(np.uint32)
    out = _unwrap_uint32(wrapped)
    assert np.all(np.diff(out) > 0)
    np.testing.assert_array_equal(out, true)


@pytest.mark.parametrize("v", [np.array([], dtype=np.uint32), np.array([7], dtype=np.uint32)])
def test_unwrap_short_arrays(v):
    np.testing.assert_array_equal(_unwrap_uint32(v), v.astype(np.int64))


# --------------------------------------------------------------------------- #
# get_regressed_systime through a wrap
# --------------------------------------------------------------------------- #
def _make_io(raw, systime_ns, global_offset=0, params=None):
    """Build a bare SpikeGadgetsRawIO wired to fixed timestamp/sysclock arrays."""
    io = SpikeGadgetsRawIO.__new__(SpikeGadgetsRawIO)
    io.regressed_systime_parameters = {} if params is None else params
    io._global_sample_offset = global_offset

    def _ts(i_start, i_stop):
        stop = len(raw) if i_stop is None else i_stop
        return raw[i_start:stop]

    def _sys(i_start, i_stop):
        stop = len(systime_ns) if i_stop is None else i_stop
        return systime_ns[i_start:stop]

    io.get_analogsignal_timestamps = _ts
    io.get_sys_clock = _sys
    return io


def _wrapping_session(n=1000, wrap_at=500):
    """A recording whose uint32 counter wraps at sample `wrap_at`."""
    true = (UINT32_WRAP - wrap_at) + np.arange(n, dtype=np.int64)  # global monotone
    raw = (true % UINT32_WRAP).astype(np.uint32)
    systime_ns = true.astype(np.float64) * 1e9 / FS
    expected_seconds = true / FS
    return raw, systime_ns, expected_seconds


def test_full_file_regression_recovers_monotonic_time_across_wrap():
    raw, systime_ns, expected = _wrapping_session()
    io = _make_io(raw, systime_ns)

    out = io.get_regressed_systime(0, None)

    assert np.all(np.diff(out) > 0), "regressed time must be monotonic across the wrap"
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)
    # a wrap was recorded for partial iterators to use
    assert list(io.regressed_systime_parameters["wrap_sample_indices"]) == [500]


def test_partial_after_wrap_lands_on_same_axis_as_full():
    raw, systime_ns, expected = _wrapping_session()
    full = _make_io(raw, systime_ns)
    full.get_regressed_systime(0, None)  # fit; populates wrap_sample_indices

    # a partial iterator covering a post-wrap sub-range [600, 800) inherits the
    # fitted params and reads only its (small, post-wrap) raw counter values
    start, stop = 600, 800
    partial = _make_io(
        raw[start:stop],
        systime_ns[start:stop],  # unused (params already set) but keep aligned
        global_offset=start,
        params=full.regressed_systime_parameters,
    )

    out = partial.get_regressed_systime(0, None)

    np.testing.assert_allclose(out, expected[start:stop], rtol=0, atol=1e-6)


def test_non_wrapping_session_is_unchanged():
    # sanity: without a wrap, behaviour matches a plain float regression
    n = 500
    true = 1_000 + np.arange(n, dtype=np.int64)
    raw = true.astype(np.uint32)
    systime_ns = true.astype(np.float64) * 1e9 / FS
    io = _make_io(raw, systime_ns)

    out = io.get_regressed_systime(0, None)

    np.testing.assert_allclose(out, true / FS, rtol=0, atol=1e-9)
    assert list(io.regressed_systime_parameters["wrap_sample_indices"]) == []
