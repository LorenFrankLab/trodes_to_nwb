"""Tests for adding parsed StateScript events to an NWB file (issue #115)."""

import numpy as np
import pandas as pd
from hdmf.common.table import DynamicTable
from pynwb import NWBHDF5IO

from trodes_to_nwb import convert_dios, convert_rec_header, convert_yaml
from trodes_to_nwb.convert_statescript import (
    StateScriptLogProcessor,
    add_statescript,
    estimate_statescript_time_offset,
)
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path

SAMPLE_LOG = data_path / "20230622_sample_01_a1.stateScriptLog"
SAMPLE_REC = data_path / "20230622_sample_01_a1.rec"
SAMPLE_METADATA = data_path / "20230622_sample_metadata.yml"


def _session_df():
    """Minimal session file-info table referencing the sample StateScript log."""
    return pd.DataFrame(
        [
            {
                "epoch": 1,
                "file_extension": ".stateScriptLog",
                "full_path": str(SAMPLE_LOG),
            }
        ]
    )


def _nwb_with_dios():
    metadata, _ = convert_yaml.load_metadata(SAMPLE_METADATA, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    convert_dios.add_dios(nwbfile, [str(SAMPLE_REC)], metadata)
    return nwbfile, metadata


# --- estimator -------------------------------------------------------------


def test_estimate_offset_recovers_known_offset():
    log_times = np.array([0.0, 0.5, 1.3, 2.9, 5.0])
    delta = 1000.25
    reference = log_times + delta
    estimated = estimate_statescript_time_offset(log_times, reference, min_matches=3)
    assert estimated is not None
    assert abs(estimated - delta) < 1e-6


def test_estimate_offset_returns_none_without_match():
    log_times = np.array([0.0, 0.5, 1.3, 2.9, 5.0])
    reference = np.array([100.0, 200.0, 300.0])  # unrelated, no consistent offset
    assert estimate_statescript_time_offset(log_times, reference) is None


def test_estimate_offset_empty_inputs():
    assert estimate_statescript_time_offset(np.array([]), np.array([1.0])) is None
    assert estimate_statescript_time_offset(np.array([1.0]), np.array([])) is None


def test_estimate_offset_on_real_sample_matches_rec_dios():
    """The StateScript DIO events are the same physical events as the rec's Din2
    changes, offset by a constant. The estimator should recover that offset and
    align the events to the recorded DIO times within a few milliseconds."""
    from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO

    raw_io = SpikeGadgetsRawIO(filename=str(SAMPLE_REC))
    raw_io.parse_header()
    din2_times, _ = raw_io.get_digitalsignal("ECU_digital", "ECU_Din2")

    processor = StateScriptLogProcessor.from_file(SAMPLE_LOG)
    df = processor.get_events_dataframe(apply_offset=False)
    log_dio_times = df.loc[
        df["type"] == "ts_int_int", "trodes_timestamp_sec"
    ].to_numpy()

    offset = estimate_statescript_time_offset(log_dio_times, din2_times, tolerance=0.02)
    assert offset is not None

    # After applying the offset, each StateScript DIO event should sit within a few
    # ms of an actual recorded Din2 change.
    aligned = np.sort(log_dio_times + offset)
    din2_sorted = np.sort(din2_times)
    residuals = [
        abs(t - din2_sorted[np.argmin(np.abs(din2_sorted - t))]) for t in aligned
    ]
    assert np.median(residuals) < 0.005  # < 5 ms


# --- writer ---------------------------------------------------------------


def test_add_statescript_builds_table_without_dios():
    """Without DIO alignment the raw event table is still written."""
    metadata, _ = convert_yaml.load_metadata(SAMPLE_METADATA, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    add_statescript(nwbfile, _session_df(), align_to_dios=False)

    assert "behavior" in nwbfile.processing
    table = nwbfile.processing["behavior"]["statescript_events"]
    assert isinstance(table, DynamicTable)

    n_expected = len(
        StateScriptLogProcessor.from_file(SAMPLE_LOG).get_events_dataframe(
            apply_offset=False
        )
    )
    assert len(table) == n_expected

    for column in (
        "epoch",
        "trodes_timestamp",
        "trodes_timestamp_sec",
        "timestamp_sync",
        "type",
        "active_DIO_inputs",
        "active_DIO_outputs",
    ):
        assert column in table.colnames

    table_df = table.to_dataframe()
    assert set(table_df["epoch"]) == {1}
    # No DIO alignment requested, so timestamp_sync is entirely NaN.
    assert table_df["timestamp_sync"].isna().all()


def test_add_statescript_aligns_to_dios():
    """With DIO events present, timestamp_sync is filled and lands in the rec's
    DIO time range."""
    nwbfile, _ = _nwb_with_dios()
    add_statescript(nwbfile, _session_df())

    table_df = nwbfile.processing["behavior"]["statescript_events"].to_dataframe()
    dio_event_rows = table_df[table_df["type"] == "ts_int_int"]
    assert dio_event_rows["timestamp_sync"].notna().all()

    # Aligned DIO-event times must fall within the recorded DIO time span.
    ref = convert_dios_reference_span(nwbfile)
    synced = dio_event_rows["timestamp_sync"].to_numpy()
    assert synced.min() >= ref[0] - 0.05
    assert synced.max() <= ref[1] + 0.05


def convert_dios_reference_span(nwbfile):
    events = nwbfile.processing["behavior"]["behavioral_events"]
    all_times = np.concatenate(
        [np.asarray(ts.timestamps[:]) for ts in events.time_series.values()]
    )
    return float(all_times.min()), float(all_times.max())


def test_add_statescript_persists_to_nwb(tmp_path):
    """The table (with ragged DIO columns) survives a write/read round-trip."""
    nwbfile, _ = _nwb_with_dios()
    add_statescript(nwbfile, _session_df())

    nwb_path = tmp_path / "statescript_roundtrip.nwb"
    with NWBHDF5IO(nwb_path, mode="w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(nwb_path, mode="r") as io:
        read_nwb = io.read()
        table = read_nwb.processing["behavior"]["statescript_events"]
        table_df = table.to_dataframe()
        assert len(table_df) > 0
        # ragged column round-trips as per-row lists/arrays
        a_row = table_df.iloc[0]
        assert hasattr(a_row["active_DIO_inputs"], "__len__")


def test_add_statescript_no_log_is_noop():
    metadata, _ = convert_yaml.load_metadata(SAMPLE_METADATA, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    empty_df = pd.DataFrame(
        [{"epoch": 1, "file_extension": ".rec", "full_path": "x.rec"}]
    )
    add_statescript(nwbfile, empty_df)
    assert "behavior" not in nwbfile.processing


# --- estimator robustness (review follow-up) -------------------------------


def test_estimate_offset_rejects_dense_random_references():
    """A dense, unrelated reference set must NOT yield a confident (wrong) offset.

    Reproduces the silent-failure risk: many reference points spaced near the
    tolerance let a spurious offset match every event by chance. The estimator
    must return None rather than a plausible-but-wrong alignment."""
    rng = np.random.default_rng(0)
    log_times = np.sort(rng.uniform(0, 600, size=50))
    # ~8000 references over the same window => spacing ~0.075s (< 4x tolerance)
    dense_reference = np.sort(rng.uniform(0, 600, size=8000))
    assert (
        estimate_statescript_time_offset(log_times, dense_reference, tolerance=0.02)
        is None
    )


def test_estimate_offset_found_even_with_dense_references():
    """The hardening must not break true positives: a genuine constant offset
    embedded in a dense reference set is still recovered."""
    rng = np.random.default_rng(1)
    log_times = np.sort(rng.uniform(0, 600, size=40))
    delta = 12345.0
    dense_reference = np.sort(
        np.concatenate([log_times + delta, rng.uniform(0, 600, size=2000) + delta])
    )
    estimated = estimate_statescript_time_offset(
        log_times, dense_reference, tolerance=0.02
    )
    assert estimated is not None
    assert abs(estimated - delta) < 1e-6


def test_estimate_offset_below_fraction_threshold_returns_none():
    # Only 3 of 10 events can share a consistent offset -> below min_fraction=0.5.
    matched = np.array([0.0, 1.0, 2.0])
    log_times = np.concatenate(
        [matched, np.array([50.5, 61.7, 72.9, 83.3, 94.1, 105.6, 116.2])]
    )
    reference = matched + 1000.0
    assert estimate_statescript_time_offset(log_times, reference) is None


# --- writer robustness (review follow-up) ----------------------------------


def test_add_statescript_multi_epoch():
    """Two epochs are concatenated into one table with a correct epoch column."""
    log2 = data_path / "20230622_sample_02_a1.stateScriptLog"
    session_df = pd.DataFrame(
        [
            {
                "epoch": 1,
                "file_extension": ".stateScriptLog",
                "full_path": str(SAMPLE_LOG),
            },
            {"epoch": 2, "file_extension": ".stateScriptLog", "full_path": str(log2)},
        ]
    )
    metadata, _ = convert_yaml.load_metadata(SAMPLE_METADATA, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    add_statescript(nwbfile, session_df, align_to_dios=False)

    table_df = nwbfile.processing["behavior"]["statescript_events"].to_dataframe()
    assert set(table_df["epoch"]) == {1, 2}
    n1 = len(
        StateScriptLogProcessor.from_file(SAMPLE_LOG).get_events_dataframe(
            apply_offset=False
        )
    )
    n2 = len(
        StateScriptLogProcessor.from_file(log2).get_events_dataframe(apply_offset=False)
    )
    assert len(table_df) == n1 + n2
    # epoch-1 rows precede epoch-2 rows (concatenation order preserved)
    epochs = table_df["epoch"].to_numpy()
    assert np.all(np.diff(epochs) >= 0)


def test_add_statescript_alignment_failure_leaves_nan_and_warns(monkeypatch, caplog):
    """DIOs present but no confident match -> timestamp_sync NaN + a warning, and
    raw Trodes timestamps are still stored."""
    import logging

    nwbfile, _ = _nwb_with_dios()
    monkeypatch.setattr(
        "trodes_to_nwb.convert_statescript.estimate_statescript_time_offset",
        lambda *args, **kwargs: None,
    )
    with caplog.at_level(logging.WARNING, logger="convert"):
        add_statescript(nwbfile, _session_df())

    table_df = nwbfile.processing["behavior"]["statescript_events"].to_dataframe()
    assert table_df["timestamp_sync"].isna().all()
    assert table_df["trodes_timestamp_sec"].notna().any()  # raw times kept
    assert "could not confidently align" in caplog.text


def test_add_statescript_comment_only_log_does_not_abort(tmp_path):
    """A comment-only / empty log must be skipped, not crash the conversion, even
    when DIO alignment is requested (regression: KeyError on empty events)."""
    empty_log = tmp_path / "20230622_sample_09_a1.stateScriptLog"
    empty_log.write_text("# only a comment\n\n# another comment\n")

    nwbfile, _ = _nwb_with_dios()
    session_df = pd.DataFrame(
        [
            {
                "epoch": 1,
                "file_extension": ".stateScriptLog",
                "full_path": str(SAMPLE_LOG),
            },
            {
                "epoch": 9,
                "file_extension": ".stateScriptLog",
                "full_path": str(empty_log),
            },
        ]
    )
    add_statescript(nwbfile, session_df)  # must not raise

    table_df = nwbfile.processing["behavior"]["statescript_events"].to_dataframe()
    # only the real epoch made it in; the comment-only epoch was skipped
    assert set(table_df["epoch"]) == {1}


def test_add_statescript_missing_file_skips_epoch(caplog):
    """An unreadable/missing log is logged and skipped, not fatal."""
    import logging

    nwbfile, _ = _nwb_with_dios()
    session_df = pd.DataFrame(
        [
            {
                "epoch": 1,
                "file_extension": ".stateScriptLog",
                "full_path": str(SAMPLE_LOG),
            },
            {
                "epoch": 9,
                "file_extension": ".stateScriptLog",
                "full_path": "/nonexistent/does_not_exist.stateScriptLog",
            },
        ]
    )
    with caplog.at_level(logging.ERROR, logger="convert"):
        add_statescript(nwbfile, session_df)  # must not raise

    table_df = nwbfile.processing["behavior"]["statescript_events"].to_dataframe()
    assert set(table_df["epoch"]) == {1}
    assert "failed to read/parse" in caplog.text
