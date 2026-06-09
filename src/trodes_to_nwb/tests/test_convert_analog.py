from datetime import datetime, timezone
import logging
import os
import re
import shutil
import types

import h5py
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
import pynwb
from pynwb import NWBFile
import pytest

from trodes_to_nwb import convert_analog, convert_rec_header, convert_yaml
from trodes_to_nwb.convert_analog import (
    SENSOR_TYPE_CONFIG,
    _categorize_sensor_channels,
    _resolve_sensor_unit,
    add_analog_data,
    update_analog_data,
)
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.tests.utils import data_path


class _FakeRecDCI:
    """Minimal stand-in for RecFileDataChunkIterator over synthetic analog data.

    Exposes just the interface ``add_analog_data`` and
    ``_AnalogChannelSubsetIterator`` rely on: a combined int16 array whose
    columns are the ECU analog channels followed by the multiplexed headstage
    channels, plus ``timestamps``, ``neo_io``, ``_get_maxshape`` and
    ``_get_data``.
    """

    def __init__(self, combined_data, multiplexed_ids, timestamps):
        self._data = combined_data
        self.timestamps = timestamps
        self.neo_io = [
            types.SimpleNamespace(
                multiplexed_channel_xml=dict.fromkeys(multiplexed_ids)
            )
        ]

    @property
    def maxshape(self):
        return self._data.shape

    def _get_maxshape(self):
        return self._data.shape

    def _get_data(self, selection):
        return self._data[selection[0], selection[1]]


def _make_minimal_nwbfile():
    return NWBFile(
        session_description="test",
        identifier="test",
        session_start_time=datetime(2023, 6, 22, tzinfo=timezone.utc),
    )


def _patch_analog_source(
    monkeypatch, ecu_ids, multiplexed_ids, combined_data, timestamps
):
    """Patch add_analog_data's rec-file reads to serve synthetic data."""
    monkeypatch.setattr(
        convert_analog, "_get_ecu_analog_channel_ids", lambda path: list(ecu_ids)
    )
    fake = _FakeRecDCI(combined_data, multiplexed_ids, timestamps)
    monkeypatch.setattr(
        convert_analog, "RecFileDataChunkIterator", lambda *a, **k: fake
    )
    return fake


def _materialize(dci):
    """Read a GenericDataChunkIterator fully into a dense array."""
    out = np.zeros(dci._get_maxshape(), dtype=dci._get_dtype())
    for chunk in dci:
        out[chunk.selection] = chunk.data
    return out


def test_add_analog_data():
    """Integration test (requires downloaded .rec / reference .nwb fixtures).

    Verifies the new acquisition layout and that recombining the per-sensor
    raw int16 streams reproduces the reference combined analog stream exactly.
    """
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_to_nwb_file = data_path / "20230622_155936.nwb"  # comparison file
    rec_header = convert_rec_header.read_header(rec_file)
    # make file with data
    nwbfile = convert_yaml.initialize_nwb(metadata, rec_header)
    add_analog_data(nwbfile, [rec_file], metadata=metadata)

    # New layout: per-sensor TimeSeries in acquisition, no combined processing stream.
    assert "analog" not in nwbfile.processing
    assert len(nwbfile.acquisition) > 0
    if "accelerometer" in nwbfile.acquisition:
        assert nwbfile.acquisition["accelerometer"].unit == "g"
        assert nwbfile.acquisition["accelerometer"].conversion == 0.000061
    if "gyroscope" in nwbfile.acquisition:
        assert nwbfile.acquisition["gyroscope"].unit == "d/s"
        assert nwbfile.acquisition["gyroscope"].conversion == 0.061

    # save file
    filename = "test_add_analog.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)
    try:
        with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            assert "analog" not in read_nwbfile.processing

            # Map every channel back to its stored (raw int16) column.
            new_by_channel = {}
            for ts in read_nwbfile.acquisition.values():
                channel_names = ts.description.split(": ", 1)[1].split(", ")
                for col, channel in enumerate(channel_names):
                    new_by_channel[channel] = ts.data[:, col]

            with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
                old_nwbfile = io2.read()
                old_ts = old_nwbfile.processing["analog"]["analog"]["analog"]
                old_id_order = old_ts.description.split("   ")[:-1]

                # same channels are present
                assert set(new_by_channel) == set(old_id_order)
                # raw values match the reference, per channel, across all timepoints
                for col, channel in enumerate(old_id_order):
                    assert (new_by_channel[channel] == old_ts.data[:, col]).all()
    finally:
        os.remove(filename)


def test_add_analog_data_writes_sensor_acquisitions(monkeypatch):
    """Synthetic: sensors land in acquisition, scaled via conversion, lazily."""
    ecu_ids = ["ECU_Ain1", "ECU_Ain2"]
    mux_ids = [
        "Headstage_AccelX",
        "Headstage_AccelY",
        "Headstage_AccelZ",
        "Headstage_GyroX",
        "Headstage_GyroY",
        "Headstage_GyroZ",
    ]
    all_ids = ecu_ids + mux_ids
    n_time = 100
    combined = np.arange(n_time * len(all_ids), dtype=np.int16).reshape(
        n_time, len(all_ids)
    )
    timestamps = np.arange(n_time, dtype=float)
    _patch_analog_source(monkeypatch, ecu_ids, mux_ids, combined, timestamps)

    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])

    # replaced, not added alongside, the combined processing stream
    assert "analog" not in nwbfile.processing
    assert set(nwbfile.acquisition) == {"analog_input", "accelerometer", "gyroscope"}

    accel = nwbfile.acquisition["accelerometer"]
    assert accel.unit == "g"
    assert accel.conversion == 0.000061
    assert nwbfile.acquisition["gyroscope"].conversion == 0.061
    assert nwbfile.acquisition["analog_input"].conversion == 1.0

    # lazy: data backed by H5DataIO over an iterator, not a dense ndarray
    assert isinstance(accel.data, H5DataIO)

    # parity: stored raw int16 (no pre-scaling) reproduces the source columns
    for ts in nwbfile.acquisition.values():
        channel_names = ts.description.split(": ", 1)[1].split(", ")
        materialized = _materialize(ts.data.data)
        for col, channel in enumerate(channel_names):
            assert (materialized[:, col] == combined[:, all_ids.index(channel)]).all()


def test_add_analog_data_stream_spans_full_source_length(monkeypatch):
    """Synthetic: the acquisition stream covers the whole source, not a prefix.

    Guards the assembly side of multi-file handling: ``add_analog_data`` must
    size each stream from the shared iterator's full ``_get_maxshape()`` rather
    than reading a single file. (The real cross-file stitching lives in
    ``RecFileDataChunkIterator`` and is covered by the integration tests.)
    """
    ecu_ids = ["ECU_Ain1"]
    n_time = 250
    combined = np.arange(n_time, dtype=np.int16).reshape(n_time, 1)
    _patch_analog_source(
        monkeypatch, ecu_ids, [], combined, np.arange(n_time, dtype=float)
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["a.rec", "b.rec"])
    materialized = _materialize(nwbfile.acquisition["analog_input"].data.data)
    assert materialized.shape[0] == n_time


def test_add_analog_data_multifile_longer_than_single(monkeypatch):
    """Integration: a two-file conversion is strictly longer than one file.

    Requires downloaded .rec fixtures. Directly guards against reading only the
    first rec file (the truncation failure mode), exercising the real
    RecFileDataChunkIterator cross-file read.
    """
    rec1 = data_path / "20230622_sample_01_a1.rec"
    rec2 = data_path / "20230622_sample_02_a1.rec"
    metadata, _ = convert_yaml.load_metadata(
        data_path / "20230622_sample_metadata.yml", []
    )
    rec_header = convert_rec_header.read_header(rec1)

    nwb_single = convert_yaml.initialize_nwb(metadata, rec_header)
    add_analog_data(nwb_single, [rec1], metadata=metadata)
    nwb_multi = convert_yaml.initialize_nwb(metadata, rec_header)
    add_analog_data(nwb_multi, [rec1, rec2], metadata=metadata)

    stream_name = next(iter(nwb_single.acquisition))
    len_single = nwb_single.acquisition[stream_name].data.data._get_maxshape()[0]
    len_multi = nwb_multi.acquisition[stream_name].data.data._get_maxshape()[0]
    assert len_multi > len_single


def test_sensor_unit_metadata_override(monkeypatch):
    """Synthetic: metadata overrides the unit label only, not the conversion."""
    ecu_ids = ["ECU_Ain1"]
    combined = np.zeros((10, 1), dtype=np.int16)
    _patch_analog_source(monkeypatch, ecu_ids, [], combined, np.arange(10, dtype=float))
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(
        nwbfile, ["fake.rec"], metadata={"sensor_units": {"analog_input": "V"}}
    )
    analog_input = nwbfile.acquisition["analog_input"]
    assert analog_input.unit == "V"
    assert analog_input.conversion == 1.0


def test_other_channels_logged(monkeypatch, caplog):
    """Synthetic: unrecognized channels become an 'other' stream and are warned."""
    ecu_ids = ["ECU_Ain1", "Mystery_Chan"]
    combined = np.zeros((5, 2), dtype=np.int16)
    _patch_analog_source(monkeypatch, ecu_ids, [], combined, np.arange(5, dtype=float))
    nwbfile = _make_minimal_nwbfile()
    with caplog.at_level(logging.WARNING, logger="convert"):
        add_analog_data(nwbfile, ["fake.rec"])
    assert "other" in nwbfile.acquisition
    assert "Mystery_Chan" in caplog.text


def test_analog_subset_iterator_parity():
    """The subset iterator yields the requested columns (non-contiguous, reordered)."""
    n_time, n_cols = 100, 5
    data = np.arange(n_time * n_cols, dtype=np.int16).reshape(n_time, n_cols)
    fake = _FakeRecDCI(data, [], np.arange(n_time, dtype=float))
    columns = [3, 1, 4]  # non-contiguous and reordered
    iterator = convert_analog._AnalogChannelSubsetIterator(fake, columns)
    assert iterator._get_maxshape() == (n_time, len(columns))
    assert iterator._get_dtype() == np.dtype("int16")
    assert (_materialize(iterator) == data[:, columns]).all()


def test_analog_subset_iterator_rejects_bad_columns():
    """Out-of-range column indices fail loudly at construction."""
    data = np.zeros((10, 3), dtype=np.int16)
    fake = _FakeRecDCI(data, [], np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="out of range"):
        convert_analog._AnalogChannelSubsetIterator(fake, [0, 5])


def test_add_analog_data_no_channels_returns(monkeypatch, caplog):
    """No analog channels: logs and returns without adding acquisitions."""
    combined = np.zeros((5, 0), dtype=np.int16)
    _patch_analog_source(monkeypatch, [], [], combined, np.arange(5, dtype=float))
    nwbfile = _make_minimal_nwbfile()
    with caplog.at_level(logging.INFO, logger="convert"):
        add_analog_data(nwbfile, ["fake.rec"])
    assert len(nwbfile.acquisition) == 0
    assert "No analog channels found" in caplog.text


def test_magnetometer_and_other_units(monkeypatch):
    """Magnetometer and 'other' streams carry conversion 1.0 / unit 'unspecified'."""
    ecu_ids = ["Mystery_Chan"]
    mux_ids = ["Headstage_MagX", "Headstage_MagY", "Headstage_MagZ"]
    combined = np.zeros((10, len(ecu_ids) + len(mux_ids)), dtype=np.int16)
    _patch_analog_source(
        monkeypatch, ecu_ids, mux_ids, combined, np.arange(10, dtype=float)
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])
    for name in ("magnetometer", "other"):
        assert nwbfile.acquisition[name].conversion == 1.0
        assert nwbfile.acquisition[name].unit == "unspecified"


def test_sensor_streams_share_one_timestamps(monkeypatch):
    """Streams after the first link to the first stream's timestamps (stored once)."""
    ecu_ids = ["ECU_Ain1"]
    mux_ids = ["Headstage_AccelX"]
    combined = np.zeros((10, 2), dtype=np.int16)
    _patch_analog_source(
        monkeypatch, ecu_ids, mux_ids, combined, np.arange(10, dtype=float)
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])
    streams = list(nwbfile.acquisition.values())
    assert len(streams) >= 2
    # every stream resolves to the same timestamps array (stored once, linked),
    # and at least one stream links rather than owning a second copy
    timestamp_arrays = [ts.timestamps for ts in streams]
    assert all(arr is timestamp_arrays[0] for arr in timestamp_arrays)
    assert any(ts.timestamp_link for ts in streams)


def test_unknown_sensor_units_key_warns(monkeypatch, caplog):
    """A misspelled sensor_units key is warned about, not silently ignored."""
    ecu_ids = ["ECU_Ain1"]
    combined = np.zeros((5, 1), dtype=np.int16)
    _patch_analog_source(monkeypatch, ecu_ids, [], combined, np.arange(5, dtype=float))
    nwbfile = _make_minimal_nwbfile()
    with caplog.at_level(logging.WARNING, logger="convert"):
        add_analog_data(
            nwbfile, ["fake.rec"], metadata={"sensor_units": {"accel": "g"}}
        )
    assert "accel" in caplog.text
    assert "unrecognized" in caplog.text.lower()


def test_add_analog_data_short_session_writes(monkeypatch, tmp_path):
    """A sub-chunk-length session writes and reads back with raw int16 intact."""
    ecu_ids = ["ECU_Ain1", "ECU_Ain2"]
    n_time = 50  # far below DEFAULT_CHUNK_TIME_DIM
    combined = np.arange(n_time * 2, dtype=np.int16).reshape(n_time, 2)
    _patch_analog_source(
        monkeypatch, ecu_ids, [], combined, np.arange(n_time, dtype=float)
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])
    path = tmp_path / "short.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(path, "r") as io:
        read = io.read()
        analog_input = read.acquisition["analog_input"]
        assert analog_input.data.shape == (n_time, 2)
        assert (analog_input.data[:] == combined).all()


def test_update_analog_data_rejects_new_layout(monkeypatch, tmp_path):
    """update_analog_data fails clearly on files without the legacy analog stream."""
    monkeypatch.setattr(convert_analog, "_get_ecu_analog_channel_ids", lambda path: [])
    nwbfile = _make_minimal_nwbfile()
    path = tmp_path / "new_layout.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    with pytest.raises(ValueError, match="legacy combined analog stream"):
        update_analog_data(str(path), ["fake.rec"])


def _write_legacy_combined_analog(nwbfile, rec_file_path):
    """Write the pre-sensor-separation combined analog stream.

    Reproduces the legacy ``processing["analog"]["analog"]["analog"]`` layout
    (a single combined TimeSeries with ``unit="-1"``) that production no longer
    writes, so the legacy-repair path of ``update_analog_data`` can be tested
    against a file in that layout.
    """
    analog_channel_ids = convert_analog._get_ecu_analog_channel_ids(rec_file_path[0])
    rec_dci = RecFileDataChunkIterator(
        rec_file_path,
        nwb_hw_channel_order=analog_channel_ids,
        stream_id="ECU_analog",
        is_analog=True,
    )
    analog_channel_ids.extend(rec_dci.neo_io[0].multiplexed_channel_xml.keys())
    data_io = H5DataIO(
        rec_dci,
        chunks=(
            convert_analog.DEFAULT_CHUNK_TIME_DIM,
            min(len(analog_channel_ids), convert_analog.DEFAULT_CHUNK_MAX_CHANNEL_DIM),
        ),
    )
    nwbfile.create_processing_module(
        name="analog", description="Contains all analog data"
    )
    analog_events = pynwb.behavior.BehavioralEvents(name="analog")
    analog_events.add_timeseries(
        pynwb.TimeSeries(
            name="analog",
            description="   ".join(analog_channel_ids) + "   ",
            data=data_io,
            timestamps=rec_dci.timestamps,
            unit="-1",
        )
    )
    nwbfile.processing["analog"].add(analog_events)


def test_update_analog_data():
    """Test that update_analog_data correctly overwrites data in an existing NWB file."""
    rec_files = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]

    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    rec_header = convert_rec_header.read_header(rec_files[0])

    # make a file in the legacy combined-analog layout that update_analog_data repairs
    nwbfile = convert_yaml.initialize_nwb(metadata, rec_header)
    _write_legacy_combined_analog(nwbfile, rec_files)

    # save file
    ref_filename = "correctly_added_analog.nwb"
    with pynwb.NWBHDF5IO(ref_filename, "w") as io:
        io.write(nwbfile)

    # Copy the reference NWB file so we don't modify the original
    buggy_filename = "test_update_analog_buggy.nwb"
    shutil.copy(ref_filename, buggy_filename)

    # Zero out the analog data in the copy to simulate the pre-fix (buggy) state
    with h5py.File(buggy_filename, "r+") as f:
        analog_hdf5_path = "processing/analog/analog/analog/data"
        f[analog_hdf5_path][...] = np.zeros_like(f[analog_hdf5_path][()])

    # Confirm data was zeroed out
    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        buggy_nwbfile = io.read()
        buggy_data = buggy_nwbfile.processing["analog"]["analog"]["analog"].data[:]
    assert (buggy_data == 0).all(), "Buggy data should be all zeros before update"

    # Run the update function (timestamps default to those already in the NWB file)
    update_analog_data(buggy_filename, rec_files)

    print("buggy file name: \n", buggy_filename)
    with pynwb.NWBHDF5IO(ref_filename, "r", load_namespaces=True) as io:
        correct_nwbfile = io.read()
        correct_data = correct_nwbfile.processing["analog"]["analog"]["analog"].data[:]

    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        updated_nwbfile = io.read()
        updated_data = updated_nwbfile.processing["analog"]["analog"]["analog"].data[:]

    # Map channel indices from the updated file into the correct file's ordering
    assert correct_data.shape == updated_data.shape
    # compare one non-zero multiplexed channel across all timepoints
    test_index = 14
    assert (correct_data[:, test_index] == updated_data[:, test_index]).all()

    # cleanup
    os.remove(buggy_filename)
    os.remove(ref_filename)


def test_selection_of_multiplexed_data():
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_header = convert_rec_header.read_header(rec_file)
    hconf = rec_header.find("HardwareConfiguration")
    ecu_conf = None
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    analog_channel_ids = []
    for channel in ecu_conf:
        if channel.attrib["dataType"] == "analog":
            analog_channel_ids.append(channel.attrib["id"])
    assert (len(analog_channel_ids)) == 12
    rec_dci = RecFileDataChunkIterator(
        [rec_file],
        nwb_hw_channel_order=analog_channel_ids,
        stream_index=2,
        is_analog=True,
    )
    assert len(rec_dci.neo_io[0].multiplexed_channel_xml.keys()) == 10
    slice_ind = [(0, 4), (0, 30), (1, 15), (5, 15), (20, 25)]
    expected_channels = [4, 22, 14, 10, 2]
    for ind, expected in zip(slice_ind, expected_channels, strict=True):
        data = rec_dci._get_data(
            (
                slice(0, 100, None),
                slice(ind[0], ind[1], None),
            )
        )
        assert data.shape[1] == expected


def test_categorize_all_sensor_types():
    channels = [
        "Headstage_AccelX",
        "Headstage_AccelY",
        "Headstage_AccelZ",
        "Headstage_GyroX",
        "Headstage_GyroY",
        "Headstage_GyroZ",
        "Headstage_MagX",
        "Headstage_MagY",
        "Headstage_MagZ",
        "ECU_Ain1",
        "Controller_Ain2",
        "Foo_Bar",
    ]
    groups = _categorize_sensor_channels(channels)
    assert groups["accelerometer"] == [
        "Headstage_AccelX",
        "Headstage_AccelY",
        "Headstage_AccelZ",
    ]
    assert groups["gyroscope"] == [
        "Headstage_GyroX",
        "Headstage_GyroY",
        "Headstage_GyroZ",
    ]
    assert groups["magnetometer"] == [
        "Headstage_MagX",
        "Headstage_MagY",
        "Headstage_MagZ",
    ]
    assert groups["analog_input"] == ["ECU_Ain1", "Controller_Ain2"]
    assert groups["other"] == ["Foo_Bar"]


def test_categorize_preserves_input_order():
    channels = ["Headstage_AccelY", "Headstage_AccelX", "Headstage_AccelZ"]
    groups = _categorize_sensor_channels(channels)
    # Order follows the input, not a sorted order.
    assert groups["accelerometer"] == channels


def test_categorize_patterns_anchored():
    channels = [
        "Headstage_AccelXfoo",
        "Headstage_Accel",
        "xHeadstage_AccelX",
        "ECU_Ain10",
    ]
    groups = _categorize_sensor_channels(channels)
    assert "accelerometer" not in groups
    assert groups["analog_input"] == ["ECU_Ain10"]
    assert sorted(groups["other"]) == [
        "Headstage_Accel",
        "Headstage_AccelXfoo",
        "xHeadstage_AccelX",
    ]


def test_categorize_empty_returns_empty():
    assert _categorize_sensor_channels([]) == {}


def test_categorize_no_other_when_all_known():
    groups = _categorize_sensor_channels(["Headstage_AccelX", "ECU_Ain1"])
    assert "other" not in groups


def test_sensor_config_conversion_unit_consistency():
    assert SENSOR_TYPE_CONFIG["accelerometer"]["conversion"] == 0.000061
    assert SENSOR_TYPE_CONFIG["accelerometer"]["unit"] == "g"
    assert SENSOR_TYPE_CONFIG["gyroscope"]["conversion"] == 0.061
    assert SENSOR_TYPE_CONFIG["gyroscope"]["unit"] == "d/s"
    for sensor_type, config in SENSOR_TYPE_CONFIG.items():
        for key in ("pattern", "conversion", "unit", "description"):
            assert key in config, f"Missing {key} in {sensor_type} config"
        re.compile(config["pattern"])  # raises re.error if invalid


def test_resolve_unit_default():
    assert _resolve_sensor_unit("accelerometer", "g", None) == "g"
    assert _resolve_sensor_unit("accelerometer", "g", {}) == "g"
    assert _resolve_sensor_unit("accelerometer", "g", {"sensor_units": {}}) == "g"


def test_resolve_unit_override():
    metadata = {"sensor_units": {"analog_input": "V"}}
    assert _resolve_sensor_unit("analog_input", "unspecified", metadata) == "V"
