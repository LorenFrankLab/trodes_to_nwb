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
