from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import logging
import os
import re
import shutil

import h5py
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
import pynwb
from pynwb import NWBFile
import pytest

from trodes_to_nwb import convert_analog, convert_rec_header, convert_yaml
from trodes_to_nwb.convert_analog import (
    SENSOR_TYPE_CONFIG,
    SensorConfig,
    _categorize_sensor_channels,
    _resolve_sensor_unit,
    _unique_acquisition_name,
    add_analog_data,
    update_analog_data,
)
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO
from trodes_to_nwb.tests.utils import data_path


class _FakeNeoIO:
    """Stand-in for a SpikeGadgetsRawIO, serving synthetic multiplexed data.

    ``decimated`` maps a tuple of channel names (one sensor group) to the
    ``(data, update_indices)`` that ``get_analogsignal_multiplexed_decimated``
    should return. Groups not present are treated as disabled (no samples).

    ``schedule`` optionally maps each channel name to its
    ``(interleavedDataIDByte, interleavedDataIDBit)`` pair, used by
    ``group_multiplexed_channels_by_schedule``. When omitted, all channels are
    treated as sharing one schedule (a single group), matching the common case.
    """

    def __init__(self, multiplexed_ids, decimated, schedule=None):
        self.multiplexed_channel_xml = dict.fromkeys(multiplexed_ids)
        self._decimated = decimated
        self._schedule = schedule

    def get_analogsignal_multiplexed_decimated(self, channel_names):
        key = tuple(channel_names)
        if key in self._decimated:
            return self._decimated[key]
        return np.empty((0, len(channel_names)), dtype=np.int16), np.array(
            [], dtype=int
        )

    def group_multiplexed_channels_by_schedule(self, channel_names):
        if self._schedule is None:
            return [list(channel_names)]
        groups, order = {}, []
        for name in channel_names:
            key = self._schedule[name]
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(name)
        return [groups[key] for key in order]


class _FakeRecDCI:
    """Minimal stand-in for RecFileDataChunkIterator over synthetic analog data.

    Exposes just the interface ``add_analog_data`` and
    ``_AnalogChannelSubsetIterator`` rely on: a combined int16 array whose
    columns are the ECU analog channels followed by the multiplexed headstage
    channels (the latter are read via ``neo_io`` decimation, not ``_get_data``),
    plus ``timestamps``, ``n_time``, ``neo_io``, ``maxshape`` and ``_get_data``.
    """

    def __init__(
        self, combined_data, multiplexed_ids, timestamps, decimated=None, schedule=None
    ):
        self._data = combined_data
        self.timestamps = timestamps
        self.n_time = [combined_data.shape[0]]
        self.neo_io = [_FakeNeoIO(multiplexed_ids, decimated or {}, schedule=schedule)]

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
    monkeypatch,
    ecu_ids,
    multiplexed_ids,
    combined_data,
    timestamps,
    decimated=None,
    schedule=None,
):
    """Patch add_analog_data's rec-file reads to serve synthetic data.

    Patches both seams so the same harness works whether or not ECU analog
    channels are present: ``RecFileDataChunkIterator`` (ECU path) and
    ``_open_headstage_only_sources`` (no-ECU path) both resolve to the same
    synthetic ``_FakeRecDCI``.
    """
    monkeypatch.setattr(
        convert_analog, "_get_ecu_analog_channel_ids", lambda path: list(ecu_ids)
    )
    fake = _FakeRecDCI(
        combined_data, multiplexed_ids, timestamps, decimated, schedule=schedule
    )
    monkeypatch.setattr(
        convert_analog, "RecFileDataChunkIterator", lambda *a, **k: fake
    )
    monkeypatch.setattr(
        convert_analog,
        "_open_headstage_only_sources",
        lambda paths, ts: (fake.neo_io, fake.n_time, fake.timestamps),
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

    ECU analog inputs stay at the full acquisition rate and match the reference
    combined stream channel-for-channel; headstage IMU sensors are decimated to
    their true ~100 Hz rate with explicit timestamps.
    """
    metadata, _ = convert_yaml.load_metadata(
        data_path / "20230622_sample_metadata.yml", []
    )
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_to_nwb_file = data_path / "20230622_155936.nwb"  # reference (old layout)
    rec_header = convert_rec_header.read_header(rec_file)
    nwbfile = convert_yaml.initialize_nwb(metadata, rec_header)
    add_analog_data(nwbfile, [rec_file], metadata=metadata)

    assert "analog" not in nwbfile.processing
    imu_names = {"accelerometer", "gyroscope", "magnetometer"}

    # Headstage IMU: present with physical units, decimated to ~100 Hz.
    accel = nwbfile.acquisition["accelerometer"]
    gyro = nwbfile.acquisition["gyroscope"]
    assert accel.unit == "m/s^2"
    assert accel.conversion == pytest.approx(0.000061 * 9.80665)  # g/LSB -> m/s^2
    assert gyro.unit == "rad/s"
    assert gyro.conversion == pytest.approx(0.061 * np.pi / 180)  # deg/s/LSB -> rad/s
    assert "magnetometer" not in nwbfile.acquisition  # disabled in this fixture
    for ts in (accel, gyro):
        t = ts.timestamps[:]
        assert np.all(np.diff(t) > 0)  # strictly increasing
        rate = 1.0 / np.median(np.diff(t))
        assert 90.0 < rate < 110.0  # true sensor rate, not the 30 kHz held rate

    # IMU values match the reader's decimated output exactly (raw int16).
    io = SpikeGadgetsRawIO(filename=str(rec_file))
    io.parse_header()
    for name, channels in (
        ("accelerometer", ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]),
        ("gyroscope", ["Headstage_GyroX", "Headstage_GyroY", "Headstage_GyroZ"]),
    ):
        expected, _ = io.get_analogsignal_multiplexed_decimated(channels)
        assert (nwbfile.acquisition[name].data[:] == expected).all()

    # ECU analog inputs: full-rate, match the reference combined stream per channel.
    filename = "test_add_analog.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io_w:
        io_w.write(nwbfile)
    try:
        with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io_r:
            read_nwbfile = io_r.read()
            new_by_channel = {}
            for nm, ts in read_nwbfile.acquisition.items():
                if nm in imu_names:
                    continue
                # contract: add_analog_data writes "<description>: ch1, ch2, ..."
                channels = ts.description.split(": ", 1)[1].split(", ")
                for col, channel in enumerate(channels):
                    new_by_channel[channel] = ts.data[:, col]

            with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
                old_ts = io2.read().processing["analog"]["analog"]["analog"]
                old_id_order = old_ts.description.split("   ")[:-1]
                ecu_channels = [c for c in old_id_order if c in new_by_channel]
                # every full-rate ECU channel is present and matches exactly
                assert ecu_channels  # sanity: some ECU channels exist
                for col, channel in enumerate(old_id_order):
                    if channel in new_by_channel:
                        assert (new_by_channel[channel] == old_ts.data[:, col]).all()
    finally:
        os.remove(filename)


def test_add_analog_data_writes_sensor_acquisitions(monkeypatch):
    """Synthetic: ECU stays lazy/full-rate; IMU is decimated with own timestamps."""
    ecu_ids = ["ECU_Ain1", "ECU_Ain2"]
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    gyro = ["Headstage_GyroX", "Headstage_GyroY", "Headstage_GyroZ"]
    mux_ids = accel + gyro
    n_time = 100
    combined = np.arange(
        n_time * (len(ecu_ids) + len(mux_ids)), dtype=np.int16
    ).reshape(n_time, -1)
    timestamps = np.arange(n_time, dtype=float)
    # accel and gyro update on interleaved (different) packets
    accel_idx, gyro_idx = np.array([10, 40, 70]), np.array([20, 50, 80])
    accel_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int16)
    gyro_data = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.int16)
    decimated = {
        tuple(accel): (accel_data, accel_idx),
        tuple(gyro): (gyro_data, gyro_idx),
    }
    _patch_analog_source(monkeypatch, ecu_ids, mux_ids, combined, timestamps, decimated)

    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])

    assert "analog" not in nwbfile.processing
    assert set(nwbfile.acquisition) == {"analog_input", "accelerometer", "gyroscope"}

    # IMU: decimated (materialized), raw int16 + its own true-rate timestamps
    acc = nwbfile.acquisition["accelerometer"]
    assert acc.unit == "m/s^2"
    assert acc.conversion == pytest.approx(0.000061 * 9.80665)
    assert isinstance(acc.data, np.ndarray)  # decimated, not lazy H5DataIO
    assert (acc.data == accel_data).all()
    assert (acc.timestamps[:] == timestamps[accel_idx]).all()
    g = nwbfile.acquisition["gyroscope"]
    assert g.unit == "rad/s"
    assert g.conversion == pytest.approx(0.061 * np.pi / 180)
    assert (g.data == gyro_data).all()
    assert (g.timestamps[:] == timestamps[gyro_idx]).all()

    # ECU: lazy/full-rate, raw int16 parity with the source columns
    ecu = nwbfile.acquisition["analog_input"]
    assert ecu.conversion == 1.0
    assert isinstance(ecu.data, H5DataIO)
    materialized = _materialize(ecu.data.data)
    assert (materialized == combined[:, : len(ecu_ids)]).all()


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


def test_add_analog_data_multifile_longer_than_single():
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

    # ECU full-rate stream spans both files
    ecu_name = next(
        n for n in nwb_single.acquisition if n not in ("accelerometer", "gyroscope")
    )
    len_single = nwb_single.acquisition[ecu_name].data.data._get_maxshape()[0]
    len_multi = nwb_multi.acquisition[ecu_name].data.data._get_maxshape()[0]
    assert len_multi > len_single
    # the decimated IMU also spans both files, with timestamps strictly increasing
    # across the file boundary (guards the file_start offset + concatenation)
    acc_single = nwb_single.acquisition["accelerometer"]
    acc_multi = nwb_multi.acquisition["accelerometer"]
    assert acc_multi.data.shape[0] > acc_single.data.shape[0]
    assert np.all(np.diff(acc_multi.timestamps[:]) > 0)


def test_add_analog_data_multifile_decimation_synthetic(monkeypatch):
    """Synthetic 2-file: decimated IMU concatenates with globally-offset timestamps.

    The real fixtures cover this end-to-end, but the synthetic harness pins the
    file_start offset arithmetic deterministically: per-file local update indices
    must be mapped onto the shared (concatenated) timestamps via cumsum(n_time).
    """
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    n0, n1 = 100, 80
    timestamps = np.arange(n0 + n1, dtype=float) * 0.001
    d0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int16)
    i0 = np.array([10, 40, 70])
    d1 = np.array([[4, 4, 4], [5, 5, 5]], dtype=np.int16)
    i1 = np.array([5, 35])

    class _MultiFileRecDCI:
        n_time = [n0, n1]
        neo_io = [
            _FakeNeoIO(accel, {tuple(accel): (d0, i0)}),
            _FakeNeoIO(accel, {tuple(accel): (d1, i1)}),
        ]

    fake = _MultiFileRecDCI()
    fake.timestamps = timestamps
    monkeypatch.setattr(convert_analog, "_get_ecu_analog_channel_ids", lambda path: [])
    # no ECU channels -> headstage-only path; serve the synthetic readers directly
    monkeypatch.setattr(
        convert_analog,
        "_open_headstage_only_sources",
        lambda paths, ts: (fake.neo_io, fake.n_time, fake.timestamps),
    )

    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["a.rec", "b.rec"])

    acc = nwbfile.acquisition["accelerometer"]
    # data concatenated in file order
    assert (acc.data == np.concatenate([d0, d1])).all()
    # second file's local indices offset by n0 onto the shared timeline
    expected_idx = np.concatenate([i0, i1 + n0])
    assert (acc.timestamps[:] == timestamps[expected_idx]).all()
    assert np.all(np.diff(acc.timestamps[:]) > 0)


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
    mag = ["Headstage_MagX", "Headstage_MagY", "Headstage_MagZ"]
    combined = np.zeros((10, len(ecu_ids) + len(mag)), dtype=np.int16)
    decimated = {tuple(mag): (np.ones((3, 3), dtype=np.int16), np.array([1, 4, 7]))}
    _patch_analog_source(
        monkeypatch, ecu_ids, mag, combined, np.arange(10, dtype=float), decimated
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])
    for name in ("magnetometer", "other"):
        assert nwbfile.acquisition[name].conversion == 1.0
        assert nwbfile.acquisition[name].unit == "unspecified"


def test_disabled_sensor_omitted_and_logged(monkeypatch, caplog):
    """A headstage sensor that never updates is omitted with a WARNING."""
    ecu_ids = ["ECU_Ain1"]
    mag = ["Headstage_MagX", "Headstage_MagY", "Headstage_MagZ"]
    combined = np.zeros((10, len(ecu_ids) + len(mag)), dtype=np.int16)
    # no decimated entry for mag -> the fake reports it as disabled (no samples)
    _patch_analog_source(
        monkeypatch, ecu_ids, mag, combined, np.arange(10, dtype=float)
    )
    nwbfile = _make_minimal_nwbfile()
    with caplog.at_level(logging.WARNING, logger="convert"):
        add_analog_data(nwbfile, ["fake.rec"])
    assert "magnetometer" not in nwbfile.acquisition
    assert "no sampled data (disabled)" in caplog.text


def test_controller_analog_input_is_separate_decimated_stream(monkeypatch):
    """Controller_Ain* rides the multiplexed stream -> its own decimated stream,
    distinct from the full-rate ECU 'analog_input'."""
    ecu_ids = ["ECU_Ain1"]  # full-rate ECU -> "analog_input"
    controller = ["Controller_Ain1", "Controller_Ain2"]  # multiplexed -> decimated
    combined = np.zeros((20, len(ecu_ids) + len(controller)), dtype=np.int16)
    ctrl_data = np.array([[7, 8], [9, 10]], dtype=np.int16)
    ctrl_idx = np.array([3, 12])
    decimated = {tuple(controller): (ctrl_data, ctrl_idx)}
    timestamps = np.arange(20, dtype=float)
    _patch_analog_source(
        monkeypatch, ecu_ids, controller, combined, timestamps, decimated
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])

    assert "analog_input" in nwbfile.acquisition  # ECU, full-rate (lazy)
    assert isinstance(nwbfile.acquisition["analog_input"].data, H5DataIO)
    cai = nwbfile.acquisition["controller_analog_input"]  # multiplexed, decimated
    assert isinstance(cai.data, np.ndarray)
    assert (cai.data == ctrl_data).all()
    assert (cai.timestamps[:] == timestamps[ctrl_idx]).all()


def test_ecu_streams_share_timestamps_imu_independent(monkeypatch):
    """ECU streams link to one shared timestamps array; IMU has its own."""
    ecu_ids = ["ECU_Ain1", "Mystery_Chan"]  # -> analog_input + other, both full-rate
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    combined = np.zeros((10, len(ecu_ids) + len(accel)), dtype=np.int16)
    accel_idx = np.array([2, 5, 8])
    decimated = {tuple(accel): (np.zeros((3, 3), dtype=np.int16), accel_idx)}
    timestamps = np.arange(10, dtype=float)
    _patch_analog_source(monkeypatch, ecu_ids, accel, combined, timestamps, decimated)
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["fake.rec"])

    ai = nwbfile.acquisition["analog_input"]
    other = nwbfile.acquisition["other"]
    # the two ECU streams resolve to the same timestamps array (stored once)
    assert ai.timestamps is other.timestamps
    assert any(s.timestamp_link for s in (ai, other))
    # the IMU stream is on its own (decimated) timebase
    acc = nwbfile.acquisition["accelerometer"]
    assert acc.timestamps is not ai.timestamps
    assert (acc.timestamps[:] == timestamps[accel_idx]).all()


def test_get_decimated_multiplexed_real_data():
    """Reader returns true-rate IMU samples; empty for disabled; raises cross-sensor."""
    io = SpikeGadgetsRawIO(filename=str(data_path / "20230622_sample_01_a1.rec"))
    io.parse_header()
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    data, idx = io.get_analogsignal_multiplexed_decimated(accel)
    assert data.shape == (idx.size, 3) and idx.size > 0
    assert 90.0 < 30000.0 / np.median(np.diff(idx)) < 110.0  # ~100 Hz @ 30 kHz
    # value oracle: decimated bytes equal the trusted held method's values at the
    # update packets (catches byte-assembly / offset regressions in the new reader)
    held = io.get_analogsignal_multiplexed()  # (n_packet, n_mux), held full-rate
    mux_ids = list(io.multiplexed_channel_xml.keys())
    accel_cols = [mux_ids.index(c) for c in accel]
    assert np.array_equal(data, held[np.ix_(idx, accel_cols)])
    # disabled sensor -> empty
    mag = ["Headstage_MagX", "Headstage_MagY", "Headstage_MagZ"]
    mag_data, mag_idx = io.get_analogsignal_multiplexed_decimated(mag)
    assert mag_data.shape == (0, 3) and mag_idx.size == 0
    # channels from different sensors don't share an update schedule
    with pytest.raises(ValueError, match="share an update schedule"):
        io.get_analogsignal_multiplexed_decimated(
            ["Headstage_AccelX", "Headstage_GyroX"]
        )


def test_group_multiplexed_channels_by_schedule_real_data():
    """Co-scheduled axes group together; different sensors split into groups."""
    io = SpikeGadgetsRawIO(filename=str(data_path / "20230622_sample_01_a1.rec"))
    io.parse_header()
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    # all accel axes share (interleavedDataIDByte, Bit) -> a single group
    assert io.group_multiplexed_channels_by_schedule(accel) == [accel]
    # accel and gyro update on different bits -> two groups, input order preserved
    assert io.group_multiplexed_channels_by_schedule(
        ["Headstage_AccelX", "Headstage_GyroX"]
    ) == [["Headstage_AccelX"], ["Headstage_GyroX"]]


def test_get_ecu_analog_channel_ids_no_ecu_returns_empty(monkeypatch):
    """A recording with no ECU device yields [] instead of raising on None."""
    from xml.etree import ElementTree as ET

    root = ET.fromstring(
        "<Configuration><HardwareConfiguration>"
        '<Device name="headstageSensor">'
        '<Channel id="AccelX" dataType="analog" startByte="2"'
        ' interleavedDataIDByte="0" interleavedDataIDBit="3"/>'
        "</Device>"
        "</HardwareConfiguration></Configuration>"
    )
    monkeypatch.setattr(
        convert_analog.convert_rec_header, "read_header", lambda path: root
    )
    assert convert_analog._get_ecu_analog_channel_ids("nonexistent.rec") == []


def test_get_analog_channel_names_no_ecu_returns_empty():
    """The public helper is consistent with _get_ecu_analog_channel_ids on no-ECU."""
    from xml.etree import ElementTree as ET

    header = ET.fromstring(
        "<Configuration><HardwareConfiguration>"
        '<Device name="headstageSensor">'
        '<Channel id="AccelX" dataType="analog" startByte="2"'
        ' interleavedDataIDByte="0" interleavedDataIDBit="3"/>'
        "</Device>"
        "</HardwareConfiguration></Configuration>"
    )
    assert convert_analog.get_analog_channel_names(header) == []


def test_ecu_read_does_not_materialize_multiplexed():
    """Reading the physical ECU columns must never build the multiplexed array.

    This is the memory regression guard: with include_multiplexed=False the whole
    -file sample-and-held multiplexed array (get_analogsignal_multiplexed) must not
    be touched when materializing the ECU acquisition streams.
    """
    rec = str(data_path / "20230622_sample_01_a1.rec")
    ecu_ids = convert_analog._get_ecu_analog_channel_ids(rec)
    rec_dci = RecFileDataChunkIterator(
        [rec],
        nwb_hw_channel_order=ecu_ids,
        stream_id="ECU_analog",
        is_analog=True,
        include_multiplexed=False,
    )
    # the iterator exposes only the physical ECU columns (no appended multiplexed)
    assert rec_dci.maxshape[1] == len(ecu_ids)

    def _boom(*args, **kwargs):
        raise AssertionError("get_analogsignal_multiplexed was materialized")

    for io in rec_dci.neo_io:
        io.get_analogsignal_multiplexed = _boom
    subset = convert_analog._AnalogChannelSubsetIterator(
        rec_dci, list(range(len(ecu_ids)))
    )
    materialized = _materialize(subset)  # full read of all ECU columns
    assert materialized.shape == (rec_dci.maxshape[0], len(ecu_ids))


def test_include_multiplexed_true_appends_mux_columns():
    """The legacy default still appends the multiplexed channels (combined layout)."""
    rec = str(data_path / "20230622_sample_01_a1.rec")
    ecu_ids = convert_analog._get_ecu_analog_channel_ids(rec)
    rec_dci = RecFileDataChunkIterator(
        [rec],
        nwb_hw_channel_order=ecu_ids,
        stream_id="ECU_analog",
        is_analog=True,
    )
    n_mux = len(rec_dci.neo_io[0].multiplexed_channel_xml)
    assert n_mux > 0
    assert rec_dci.maxshape[1] == len(ecu_ids) + n_mux


def test_no_ecu_headstage_only_writes_streams(monkeypatch):
    """Headstage-only recordings (no ECU) still write the decimated sensor streams."""
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    gyro = ["Headstage_GyroX", "Headstage_GyroY", "Headstage_GyroZ"]
    mux_ids = accel + gyro
    n_time = 60
    combined = np.zeros((n_time, 0), dtype=np.int16)  # no ECU columns
    timestamps = np.arange(n_time, dtype=float)
    accel_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    gyro_data = np.array([[7, 8, 9]], dtype=np.int16)
    decimated = {
        tuple(accel): (accel_data, np.array([10, 30])),
        tuple(gyro): (gyro_data, np.array([20])),
    }
    _patch_analog_source(monkeypatch, [], mux_ids, combined, timestamps, decimated)
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["headstage_only.rec"])

    # no ECU streams, but both IMU sensors are written with their own timebases
    assert "analog_input" not in nwbfile.acquisition
    assert (nwbfile.acquisition["accelerometer"].data[:] == accel_data).all()
    assert (
        nwbfile.acquisition["accelerometer"].timestamps[:] == timestamps[[10, 30]]
    ).all()
    assert (nwbfile.acquisition["gyroscope"].data[:] == gyro_data).all()
    assert (nwbfile.acquisition["gyroscope"].timestamps[:] == timestamps[[20]]).all()


def test_mixed_update_schedule_splits_into_streams(monkeypatch):
    """A sensor whose axes update on different schedules splits into separate streams.

    Rather than raising (the reader requires one shared schedule per call), the
    channels are partitioned by (interleavedDataIDByte, interleavedDataIDBit) and
    each co-sampled group is written as its own acquisition TimeSeries.
    """
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    n_time = 40
    combined = np.zeros((n_time, 0), dtype=np.int16)
    timestamps = np.arange(n_time, dtype=float)
    # AccelX/Y share one schedule; AccelZ updates on a different bit
    schedule = {
        "Headstage_AccelX": (0, 3),
        "Headstage_AccelY": (0, 3),
        "Headstage_AccelZ": (0, 4),
    }
    xy_data = np.array([[1, 2], [3, 4]], dtype=np.int16)
    z_data = np.array([[5], [6], [7]], dtype=np.int16)
    decimated = {
        ("Headstage_AccelX", "Headstage_AccelY"): (xy_data, np.array([5, 25])),
        ("Headstage_AccelZ",): (z_data, np.array([8, 18, 28])),
    }
    _patch_analog_source(
        monkeypatch, [], accel, combined, timestamps, decimated, schedule=schedule
    )
    nwbfile = _make_minimal_nwbfile()
    add_analog_data(nwbfile, ["headstage_only.rec"])

    # two streams written (not one, and no ValueError), disambiguated by name
    assert "accelerometer" in nwbfile.acquisition
    assert "accelerometer_2" in nwbfile.acquisition
    xy = nwbfile.acquisition["accelerometer"]
    z = nwbfile.acquisition["accelerometer_2"]
    assert (xy.data[:] == xy_data).all()
    assert (z.data[:] == z_data).all()
    # each group rides its own (distinct) decimated timebase
    assert (xy.timestamps[:] == timestamps[[5, 25]]).all()
    assert (z.timestamps[:] == timestamps[[8, 18, 28]]).all()
    # the two groups list their own channels in the descriptions
    assert "Headstage_AccelX, Headstage_AccelY" in xy.description
    assert "Headstage_AccelZ" in z.description


def test_accelerometer_conversion_recovers_gravity():
    """Physical check that the accelerometer SI conversion is correct.

    An accelerometer always senses gravity, so the magnitude of the 3-axis
    reading must be ~1 g (9.80665 m/s^2). After applying the stored conversion,
    the median |acceleration| over the session should land near standard gravity
    (within tolerance for sensor bias and the animal's own motion). A wrong
    conversion factor (e.g. forgetting the g->m/s^2 step, or an LSB error) would
    push this far from 1 g.
    """
    io = SpikeGadgetsRawIO(filename=str(data_path / "20230622_sample_01_a1.rec"))
    io.parse_header()
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    raw, _ = io.get_analogsignal_multiplexed_decimated(accel)
    physical = raw.astype(float) * SENSOR_TYPE_CONFIG["accelerometer"].conversion
    magnitude = np.sqrt((physical**2).sum(axis=1))  # m/s^2
    median_g = np.median(magnitude) / 9.80665
    # measured ~1.04 g on this fixture; band catches gross (factor) conversion errors
    assert 0.8 < median_g < 1.25, f"|accel| median {median_g:.3f} g not near gravity"


def test_old_and_new_layouts_agree_at_aligned_timepoints():
    """Convert the same rec file the old (combined, held) and new (per-sensor,
    decimated) way; the IMU values at the timepoints they share are identical.

    The old combined stream stored raw int16 with no scaling (unit='-1'); the new
    stream stores the same raw int16 plus a conversion to SI. So at the kept
    timepoints the raw counts match exactly, and the new physical value is the old
    raw count times the documented conversion.
    """
    rec = data_path / "20230622_sample_01_a1.rec"
    metadata, _ = convert_yaml.load_metadata(
        data_path / "20230622_sample_metadata.yml", []
    )
    header = convert_rec_header.read_header(rec)
    new_nwb = convert_yaml.initialize_nwb(metadata, header)
    add_analog_data(new_nwb, [rec], metadata=metadata)
    accel_ts = new_nwb.acquisition["accelerometer"]
    accel = ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]

    io = SpikeGadgetsRawIO(filename=str(rec))
    io.parse_header()
    held = io.get_analogsignal_multiplexed()  # old way: full-rate sample-and-held
    mux_ids = list(io.multiplexed_channel_xml.keys())
    cols = [mux_ids.index(c) for c in accel]
    _, idx = io.get_analogsignal_multiplexed_decimated(accel)

    # raw counts are preserved at the aligned (kept) timepoints: the new stream is
    # the same data as the old, relocated/decimated, plus an SI conversion factor.
    # (That the conversion is physically correct is verified separately by
    # test_accelerometer_conversion_recovers_gravity.)
    new_raw = accel_ts.data[:]
    old_raw_at_aligned = held[np.ix_(idx, cols)]
    assert np.array_equal(new_raw, old_raw_at_aligned)
    assert accel_ts.conversion == pytest.approx(0.000061 * 9.80665)


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


def test_update_analog_data(tmp_path):
    """update_analog_data restores the analog data in a legacy-layout NWB file."""
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

    ref_filename = str(tmp_path / "correctly_added_analog.nwb")
    with pynwb.NWBHDF5IO(ref_filename, "w") as io:
        io.write(nwbfile)

    # copy the reference and zero its analog data to simulate the pre-fix state
    buggy_filename = str(tmp_path / "test_update_analog_buggy.nwb")
    shutil.copy(ref_filename, buggy_filename)
    with h5py.File(buggy_filename, "r+") as f:
        analog_hdf5_path = "processing/analog/analog/analog/data"
        f[analog_hdf5_path][...] = np.zeros_like(f[analog_hdf5_path][()])
    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        buggy_data = io.read().processing["analog"]["analog"]["analog"].data[:]
    assert (buggy_data == 0).all(), "Buggy data should be all zeros before update"

    # run the repair (timestamps default to those already in the NWB file)
    update_analog_data(buggy_filename, rec_files)

    with pynwb.NWBHDF5IO(ref_filename, "r", load_namespaces=True) as io:
        correct_data = io.read().processing["analog"]["analog"]["analog"].data[:]
    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        updated_data = io.read().processing["analog"]["analog"]["analog"].data[:]

    # the repaired file matches the correct file on every channel and timepoint
    assert updated_data.shape == correct_data.shape
    assert np.array_equal(updated_data, correct_data)


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
        "ECU_Aout1",
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
    # ECU and controller analog inputs are distinct sensor types (different
    # sources/sampling), so "analog_input" is unambiguously the ECU stream.
    assert groups["analog_input"] == ["ECU_Ain1"]
    assert groups["analog_output"] == ["ECU_Aout1"]
    assert groups["controller_analog_input"] == ["Controller_Ain2"]
    assert groups["other"] == ["Foo_Bar"]


def test_categorize_accepts_bare_imu_names():
    """Trodes' bare headstageSensor channel ids categorize like the prefixed ones."""
    channels = [
        "AccelX",
        "AccelY",
        "AccelZ",
        "GyroX",
        "GyroY",
        "GyroZ",
        "MagX",
        "MagY",
        "MagZ",
    ]
    groups = _categorize_sensor_channels(channels)
    assert groups["accelerometer"] == ["AccelX", "AccelY", "AccelZ"]
    assert groups["gyroscope"] == ["GyroX", "GyroY", "GyroZ"]
    assert groups["magnetometer"] == ["MagX", "MagY", "MagZ"]
    assert "other" not in groups


def test_categorize_ecu_analog_output_not_other():
    """ECU analog outputs are a known category, not warned-on 'other' channels."""
    groups = _categorize_sensor_channels(["ECU_Aout1", "ECU_Aout2"])
    assert groups["analog_output"] == ["ECU_Aout1", "ECU_Aout2"]
    assert "other" not in groups


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
    # IMU stored in SI units (NWB convention): conversion maps raw int16 -> SI
    assert SENSOR_TYPE_CONFIG["accelerometer"].unit == "m/s^2"
    assert SENSOR_TYPE_CONFIG["accelerometer"].conversion == pytest.approx(
        0.000061 * 9.80665
    )
    assert SENSOR_TYPE_CONFIG["gyroscope"].unit == "rad/s"
    assert SENSOR_TYPE_CONFIG["gyroscope"].conversion == pytest.approx(
        0.061 * np.pi / 180
    )
    for sensor_type, config in SENSOR_TYPE_CONFIG.items():
        assert config.pattern is not None, f"{sensor_type} must have a pattern"
        re.compile(config.pattern)  # raises re.error if invalid


def test_sensor_config_is_immutable():
    with pytest.raises(FrozenInstanceError):
        SENSOR_TYPE_CONFIG["accelerometer"].conversion = 1.0


@pytest.mark.parametrize(
    "bad",
    [
        {"conversion": 0.0},
        {"conversion": float("nan")},
        {"conversion": float("inf")},
        {"unit": ""},
    ],
)
def test_sensor_config_rejects_invalid(bad):
    """__post_init__ rejects a non-finite/zero conversion or an empty unit."""
    kwargs = {"conversion": 1.0, "unit": "g", "description": "d", **bad}
    with pytest.raises(ValueError):
        SensorConfig(**kwargs)


def test_unique_acquisition_name_dedups_and_warns(caplog):
    """A colliding acquisition name is suffixed and warned, not silently dropped."""
    logger = logging.getLogger("convert")
    nwbfile = _make_minimal_nwbfile()
    # first use of a name is returned unchanged
    assert _unique_acquisition_name(nwbfile, "analog_input", logger) == "analog_input"
    nwbfile.add_acquisition(
        pynwb.TimeSeries(name="analog_input", data=[0], unit="unspecified", rate=1.0)
    )
    with caplog.at_level(logging.WARNING, logger="convert"):
        assert (
            _unique_acquisition_name(nwbfile, "analog_input", logger)
            == "analog_input_2"
        )
    assert "already exists" in caplog.text
    # a third collision increments the suffix
    nwbfile.add_acquisition(
        pynwb.TimeSeries(name="analog_input_2", data=[0], unit="unspecified", rate=1.0)
    )
    assert _unique_acquisition_name(nwbfile, "analog_input", logger) == "analog_input_3"


def test_resolve_unit_default():
    assert _resolve_sensor_unit("accelerometer", "g", None) == "g"
    assert _resolve_sensor_unit("accelerometer", "g", {}) == "g"
    assert _resolve_sensor_unit("accelerometer", "g", {"sensor_units": {}}) == "g"


def test_resolve_unit_override():
    metadata = {"sensor_units": {"analog_input": "V"}}
    assert _resolve_sensor_unit("analog_input", "unspecified", metadata) == "V"
