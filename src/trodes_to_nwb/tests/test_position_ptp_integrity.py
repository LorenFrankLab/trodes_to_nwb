"""PTP position-timestamp robustness for ``_get_position_timestamps_ptp`` (#172).

Covers three failure modes that the PTP path previously handled poorly:
- (c) no acquisition pause in the first 100 frames -> used to raise IndexError
      and abort the whole conversion; should now keep all frames;
- (b) non-monotonic PTP timestamps -> used to be written silently; should warn;
- (d) trailing NaN positions (video runs past tracking) -> used to be written as
      valid-timestamped NaN; should be dropped.
"""

import logging

import numpy as np
import pandas as pd

from trodes_to_nwb.convert_position import _get_position_timestamps_ptp

FS = 30.0  # camera frames/s
EPOCH_2023 = 1.7e9  # seconds, ~2023 (avoids the <2000 warning path)
LOGGER = logging.getLogger("convert")


def _ptp_df(seconds, xloc=None, yloc=None):
    """Build the merged frame the PTP helper expects (HWTimestamp in ns)."""
    n = len(seconds)
    data = {
        "HWframeCount": np.arange(n),
        "HWTimestamp": (np.asarray(seconds) * 1e9).astype(np.int64),
        "video_frame_ind": np.arange(n),
        "non_repeat_timestamp_labels": np.ones(n, dtype=int),
    }
    if xloc is not None:
        data["xloc"] = np.asarray(xloc, dtype=float)
    if yloc is not None:
        data["yloc"] = np.asarray(yloc, dtype=float)
    return pd.DataFrame(data)


def test_no_pause_keeps_all_frames_without_raising():
    seconds = EPOCH_2023 + np.arange(200) / FS  # uniform, no acquisition pause
    out = _get_position_timestamps_ptp(_ptp_df(seconds), LOGGER)
    assert len(out) == 200  # would previously have raised IndexError


def test_pause_is_still_removed_when_present():
    seconds = EPOCH_2023 + np.arange(200) / FS
    seconds[10:] += 0.6  # a 0.6 s pause between frames 9 and 10 (within 0.4-2.0 s)
    out = _get_position_timestamps_ptp(_ptp_df(seconds), LOGGER)
    assert len(out) == 200 - 10


def test_non_monotonic_timestamps_warn(caplog):
    seconds = EPOCH_2023 + np.arange(50) / FS
    seconds[30] -= 10.0  # backward jump
    with caplog.at_level(logging.WARNING, logger="convert"):
        _get_position_timestamps_ptp(_ptp_df(seconds), LOGGER)
    assert any("strictly increasing" in r.message.lower() for r in caplog.records)


def test_trailing_nan_positions_are_dropped(caplog):
    n = 20
    seconds = EPOCH_2023 + np.arange(n) / FS  # no pause -> all frames retained
    xloc = np.arange(n, dtype=float)
    yloc = np.arange(n, dtype=float)
    xloc[-2:] = np.nan  # camera ran two frames past position tracking
    yloc[-2:] = np.nan
    with caplog.at_level(logging.WARNING, logger="convert"):
        out = _get_position_timestamps_ptp(_ptp_df(seconds, xloc, yloc), LOGGER)
    assert int(out[["xloc", "yloc"]].isna().sum().sum()) == 0
    assert len(out) == n - 2
    assert any("no matching" in r.message.lower() for r in caplog.records)


def test_partial_nan_rows_are_preserved():
    # Only fully-unmatched rows (every position column NaN) are dropped. A row
    # where one column is NaN but another is valid must be kept, so a real
    # single-LED dropout (or interior gap) is not silently elided.
    n = 20
    seconds = EPOCH_2023 + np.arange(n) / FS
    xloc = np.arange(n, dtype=float)
    yloc = np.arange(n, dtype=float)
    xloc[5] = np.nan  # interior, single-column NaN -> row is NOT fully unmatched
    xloc[-1] = np.nan
    out = _get_position_timestamps_ptp(_ptp_df(seconds, xloc, yloc), LOGGER)
    assert len(out) == n  # nothing dropped; yloc still present on those rows
