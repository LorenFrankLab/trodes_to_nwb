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
    SpikeGadgetsRawIOPartial,
    _unwrap_uint32,
)
from trodes_to_nwb.tests.utils import data_path

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


@pytest.mark.parametrize(
    "v", [np.array([], dtype=np.uint32), np.array([7], dtype=np.uint32)]
)
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
    # get_regressed_systime drives its streaming fit from _raw_memmap.shape[0]
    # (#47); for a real IO that equals the counter length, so mirror it here.
    io._raw_memmap = np.empty((len(raw), 0), dtype=np.uint8)

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


def _synthetic_drop_before_wrap_io(n=500, drops=(50, 60, 70, 80, 90), wrap_at=300):
    """A real SpikeGadgetsRawIO whose memmap is replaced with synthetic packets:
    a uint32 counter with single dropped packets (diff==2) *before* a wrap, plus a
    matching int64 sysclock. Built off a real .rec so the partial constructor (which
    reads the file header and copies parsed attributes) works unchanged."""
    io = SpikeGadgetsRawIO(filename=str(data_path / "20230622_sample_01_a1.rec"))
    io.parse_header()
    ts_byte, sys_byte, pkt = (
        io._timestamp_byte,
        io.sysClock_byte,
        io._raw_memmap.shape[1],
    )
    drops = np.asarray(drops)
    extra = np.zeros(n, dtype=np.int64)
    for d in drops:
        extra[d + 1 :] += 1  # a single dropped packet -> a +1 gap (counter diff 2)
    unwrapped = np.arange(n, dtype=np.int64) + extra
    unwrapped_global = (UINT32_WRAP - int(unwrapped[wrap_at])) + unwrapped
    raw_counter = (unwrapped_global % UINT32_WRAP).astype("<u4")
    sysclock = (1.7e18 + (1e9 / FS) * unwrapped_global).astype("<i8")

    raw = np.zeros((n, pkt), dtype=np.uint8)
    raw[:, ts_byte : ts_byte + 4] = raw_counter.view(np.uint8).reshape(n, 4)
    raw[:, sys_byte : sys_byte + 8] = sysclock.view(np.uint8).reshape(n, 8)

    io._raw_memmap = raw
    io.interpolate_dropped_packets = True
    io.interpolate_index = None
    io.regressed_systime_parameters = {}
    io._global_sample_offset = 0
    return io


def test_partial_offset_translates_to_post_interpolation_axis():
    # A dropped-packet insertion BEFORE a uint32 wrap shifts the wrap onto the
    # post-interpolation axis. A post-wrap partial inherits those (post-interp)
    # wrap indices but its start_index is a pre-interpolation index; if the offset
    # is not translated, prior_wraps is undercounted and the partial lands ~39.77 h
    # off (one missed wrap). (#47 -- pre-existing interpolation x split x wrap bug.)
    io = _synthetic_drop_before_wrap_io()
    full_ts = io.get_regressed_systime(0, None)
    assert np.all(np.diff(full_ts) > 0)  # monotonic across the drops and the wrap
    # the wrap sits one sample later than its pre-interp index 300 (one insertion
    # of the 5 drops is... all 5 are before it -> +5)
    assert list(io.regressed_systime_parameters["wrap_sample_indices"]) == [305]

    start, stop = 302, 400  # a post-wrap partial; no drops in [start, stop)
    partial = SpikeGadgetsRawIOPartial(io, start_index=start, stop_index=stop)
    partial_ts = partial.get_regressed_systime(0, None)

    # those pre-interp samples sit at these post-interp positions in the full file
    post_offset = int(np.searchsorted(io._raw_memmap.mapped_index, start, side="left"))
    np.testing.assert_array_equal(
        partial_ts, full_ts[post_offset : post_offset + (stop - start)]
    )


def test_partial_interpolates_drop_at_split_boundary():
    # If a single dropped packet is represented by raw[stop - 1] -> raw[stop]
    # with diff==2, a partial-local np.diff over raw[start:stop] cannot see it.
    # The virtual packet belongs to the previous partial because full-file
    # interpolation inserts it beside raw[stop - 1].
    io = _synthetic_drop_before_wrap_io(n=400, drops=(179,), wrap_at=300)
    full_ts = io.get_regressed_systime(0, None)

    first = SpikeGadgetsRawIOPartial(io, start_index=0, stop_index=180)
    second = SpikeGadgetsRawIOPartial(io, start_index=180, stop_index=360)

    assert list(first.interpolate_index) == [179]
    assert list(second.interpolate_index) == []
    assert first._raw_memmap.shape[0] == 181
    assert second._raw_memmap.shape[0] == 180

    partial_ts = np.concatenate(
        [
            first.get_regressed_systime(0, None),
            second.get_regressed_systime(0, None),
        ]
    )
    expected_stop = int(np.searchsorted(io._raw_memmap.mapped_index, 360, side="left"))
    np.testing.assert_allclose(partial_ts, full_ts[:expected_stop], rtol=0, atol=1e-6)


def test_partial_detects_boundary_drop_without_full_interpolation_map():
    # Non-sysclock split construction may create partials before the full file
    # has resolved interpolation. The partial still needs one packet of lookahead
    # to see a dropped packet at its final raw sample.
    io = _synthetic_drop_before_wrap_io(n=400, drops=(179,), wrap_at=300)

    first = SpikeGadgetsRawIOPartial(io, start_index=0, stop_index=180)
    second = SpikeGadgetsRawIOPartial(io, start_index=180, stop_index=360)

    assert list(first.interpolate_index) == [179]
    assert list(second.interpolate_index) == []
    assert first._raw_memmap.shape[0] == 181
    assert second._raw_memmap.shape[0] == 180


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


def _fit_then_partial(raw, systime_ns, start, stop):
    """Fit on the full recording, then read a [start, stop) partial that inherits
    the fit (mirrors how split iterators are created)."""
    full = _make_io(raw, systime_ns)
    full.get_regressed_systime(0, None)  # fit; populates wrap_sample_indices
    partial = _make_io(
        raw[start:stop],
        systime_ns[start:stop],  # unused once params are set, kept aligned
        global_offset=start,
        params=full.regressed_systime_parameters,
    )
    return partial.get_regressed_systime(0, None)


def test_partial_straddling_the_wrap():
    # a partial that *contains* the wrap: prior_wraps == 0, but the in-slice
    # _unwrap_uint32 must detect the jump.
    raw, systime_ns, expected = _wrapping_session()  # wrap at sample 500
    out = _fit_then_partial(raw, systime_ns, 450, 650)
    np.testing.assert_allclose(out, expected[450:650], rtol=0, atol=1e-6)


def test_partial_starting_exactly_on_the_wrap():
    # start == wrap_sample_index (500): the single case that distinguishes
    # searchsorted side="right" (correct) from side="left" (off by one wrap).
    raw, systime_ns, expected = _wrapping_session()
    out = _fit_then_partial(raw, systime_ns, 500, 700)
    np.testing.assert_allclose(out, expected[500:700], rtol=0, atol=1e-6)


def _multi_wrap_session(n=300):
    # large per-sample step so the counter wraps several times within n samples
    step = UINT32_WRAP // 100
    true = np.arange(n, dtype=np.int64) * step
    raw = (true % UINT32_WRAP).astype(np.uint32)
    systime_ns = true.astype(np.float64) * 1e9 / FS
    return raw, systime_ns, true / FS


def test_multi_wrap_full_and_partial():
    raw, systime_ns, expected = _multi_wrap_session()  # ~2 wraps within the file
    full = _make_io(raw, systime_ns)
    out_full = full.get_regressed_systime(0, None)
    assert np.all(np.diff(out_full) > 0)
    np.testing.assert_allclose(out_full, expected, rtol=0, atol=1e-6)
    assert len(full.regressed_systime_parameters["wrap_sample_indices"]) >= 2

    # a partial after several wraps (prior_wraps == 2) must land on the same axis
    out = _fit_then_partial(raw, systime_ns, 250, 300)
    np.testing.assert_allclose(out, expected[250:300], rtol=0, atol=1e-6)
