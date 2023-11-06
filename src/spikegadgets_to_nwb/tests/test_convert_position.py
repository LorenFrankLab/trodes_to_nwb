import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pynwb import NWBHDF5IO, TimeSeries
from trodes_to_nwb.convert_dios import add_dios
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.convert_intervals import add_epochs, add_sample_count
from trodes_to_nwb.convert_position import (
    add_position,
    correct_timestamps_for_camera_to_mcu_lag,
    detect_repeat_timestamps,
    detect_trodes_time_repeats_or_frame_jumps,
    estimate_camera_time_from_mcu_time,
    estimate_camera_to_mcu_lag,
    find_acquisition_timing_pause,
    find_camera_dio_channel,
    find_large_frame_jumps,
    get_framerate,
    parse_dtype,
    read_trodes_datafile,
    remove_acquisition_timing_pause_non_ptp,
    wrapped_digitize,
)
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.tests.utils import data_path

from trodes_to_nwb import convert, convert_rec_header, convert_yaml


def test_wrapped_digitize():
    x = np.array([4, 5, 6, 0, 1, 2])
    bins = np.array([4, 5, 6, 0, 1, 2])
    expected = np.array([0, 1, 2, 3, 4, 5])
    assert np.array_equal(wrapped_digitize(x, bins), expected)
    # test case no wrapping
    assert np.array_equal(wrapped_digitize(expected, expected), expected)


def test_parse_dtype_standard():
    fieldstr = "<field1 uint32><field2 int32><field3 4*float64>"
    dtype = parse_dtype(fieldstr)
    expected_dtype = np.dtype(
        [("field1", np.uint32, 1), ("field2", np.int32, 1), ("field3", np.float64, 4)]
    )
    assert dtype == expected_dtype


def test_parse_dtype_no_repeat():
    fieldstr = "<field1 uint32><field2 int32>"
    dtype = parse_dtype(fieldstr)
    expected_dtype = np.dtype([("field1", np.uint32, 1), ("field2", np.int32, 1)])
    assert dtype == expected_dtype


def test_parse_dtype_invalid_dtype():
    fieldstr = "<time nonexisttype>"
    with pytest.raises(AttributeError):
        parse_dtype(fieldstr)


def test_parse_dtype_inverted_order():
    fieldstr = "<field1 uint32><field2 float64*4>"
    dtype = parse_dtype(fieldstr)
    expected_dtype = np.dtype([("field1", np.uint32, 1), ("field2", np.float64, 4)])
    assert dtype == expected_dtype


def test_read_trodes_datafile_correct_settings(tmp_path):
    filename = tmp_path / "test_file.bin"
    content = "<Start settings>\nClock rate: 30000\nfields: <field1 uint32><field2 int32>\n<End settings>\n"
    data = [1, 2, 3, 4]
    with open(filename, "wb") as file:
        file.write(content.encode())
        file.write(np.array(data, dtype=np.uint32).tobytes())

    result = read_trodes_datafile(filename)
    assert result["clock rate"] == "30000"

    expected_data = pd.DataFrame(result["data"])
    assert expected_data["field1"].dtype == np.uint32
    assert expected_data["field2"].dtype == np.int32
    assert np.array_equal(expected_data.field1, np.array([1, 3], dtype=np.uint32))
    assert np.array_equal(expected_data.field2, np.array([2, 4], dtype=np.uint32))


def test_read_trodes_datafile_incorrect_settings(tmp_path):
    filename = tmp_path / "incorrect_test_file.bin"
    content = "Incorrect content\n"
    with open(filename, "wb") as file:
        file.write(content.encode())

    with pytest.raises(Exception, match="Settings format not supported"):
        read_trodes_datafile(filename)


def test_read_trodes_datafile_missing_fields(tmp_path):
    filename = tmp_path / "missing_fields_test_file.bin"
    content = "<Start settings>\n<End settings>\n"
    data = [1.0, 2.0, 3.0]
    with open(filename, "wb") as file:
        file.write(content.encode())
        file.write(np.array(data, dtype=np.float64).tobytes())

    result = read_trodes_datafile(filename)
    expected_data = np.array(data, dtype=np.float64)
    assert np.array_equal(result["data"], expected_data)


def test_find_large_frame_jumps():
    convert.setup_logger("convert", "testing.log")
    frame_count = np.array([5, 10, 30, 40, 70])
    jumps = find_large_frame_jumps(frame_count, min_frame_jump=15)
    assert np.array_equal(jumps, [False, False, True, False, True])


def test_detect_repeat_timestamps():
    timestamps = np.array([1, 2, 2, 3, 3, 3, 4])
    repeats = detect_repeat_timestamps(timestamps)
    assert np.array_equal(repeats, [False, False, True, False, True, True, False])


def test_detect_trodes_time_repeats_or_frame_jumps():
    convert.setup_logger("convert", "testing.log")
    trodes_time = np.array([1, 2, 2, 3, 4, 5])
    frame_count = np.array([0, 10, 20, 30, 40, 1000])
    (
        _,
        non_repeat_timestamp_labels_id,
    ) = detect_trodes_time_repeats_or_frame_jumps(trodes_time, frame_count)
    assert non_repeat_timestamp_labels_id.size == 1
    assert np.array_equal(non_repeat_timestamp_labels_id, np.array([1], dtype=np.int32))


def test_estimate_camera_time_from_mcu_time():
    position_timestamps = pd.DataFrame([10, 20, 30], index=[1, 2, 3])
    mcu_timestamps = pd.DataFrame([15, 25], index=[1, 3])
    camera_systime, is_valid = estimate_camera_time_from_mcu_time(
        position_timestamps, mcu_timestamps
    )
    assert np.array_equal(camera_systime.squeeze(), [15, 25])
    assert np.array_equal(is_valid, [True, False, True])


def test_estimate_camera_to_mcu_lag():
    convert.setup_logger("convert", "testing.log")
    camera_systime = np.array([1000, 2000, 3000])
    dio_systime = np.array([900, 1800, 2700])
    lag = estimate_camera_to_mcu_lag(camera_systime, dio_systime)
    assert np.isclose(lag, 200.0)
    lag = estimate_camera_to_mcu_lag(camera_systime, dio_systime, n_breaks=1)
    assert np.isclose(lag, 100.0)


def test_remove_acquisition_timing_pause_non_ptp():
    dio_systime = np.array([100, 200, 300])
    frame_count = np.array([5, 10, 15])
    camera_systime = np.array([50, 150, 250])
    is_valid_camera_time = np.array([True, True, True])
    pause_mid_time = 150
    results = remove_acquisition_timing_pause_non_ptp(
        dio_systime, frame_count, camera_systime, is_valid_camera_time, pause_mid_time
    )
    assert np.array_equal(results[0], [200, 300])
    assert np.array_equal(results[1], [15])
    assert np.array_equal(results[2], [False, False, True])
    assert np.array_equal(results[3], [250])


def test_get_framerate():
    timestamps = np.array([0, 1000000000, 2000000000, 3000000000])
    framerate = get_framerate(timestamps)
    assert framerate == 1.0


def test_find_acquisition_timing_pause():
    timestamps = np.array(
        [0, 1000000000, 1500000000, 2500000000, 3500000000, 4500000000]
    )
    pause_mid_time = find_acquisition_timing_pause(
        timestamps, min_duration=0.4, max_duration=1.0, n_search=100
    )
    assert pause_mid_time == 1250000000

    pause_mid_time = find_acquisition_timing_pause(
        timestamps, min_duration=0.4, max_duration=1.1, n_search=100
    )
    assert pause_mid_time == 500000000


def test_correct_timestamps_for_camera_to_mcu_lag():
    NANOSECONDS_PER_SECOND = 1e9
    frame_count = np.arange(5)
    camera_systime = np.array([10, 20, 30, 40, 50]) * NANOSECONDS_PER_SECOND
    camera_to_mcu_lag = np.ones((5,)) * 10 * NANOSECONDS_PER_SECOND

    corrected_camera_systime = correct_timestamps_for_camera_to_mcu_lag(
        frame_count, camera_systime, camera_to_mcu_lag
    )

    expected_corrected_camera_systime = np.arange(0, 50, 10)

    # Assert that the corrected timestamps are as expected
    np.allclose(corrected_camera_systime, expected_corrected_camera_systime)


def test_add_position():
    probe_metadata = [data_path / "tetrode_12.5.yml"]

    # make session_df
    path_df = get_file_info(data_path)
    session_df = path_df[(path_df.animal == "sample")]

    # get metadata
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    rec_file = session_df[
        (session_df.epoch == 1) & (session_df.file_extension == ".rec")
    ].full_path.to_list()[0]
    rec_header = convert_rec_header.read_header(rec_file)

    # make nwb file
    nwbfile = convert_yaml.initialize_nwb(metadata, rec_header)

    # run add_position and prerequisite functions
    convert_yaml.add_cameras(nwbfile, metadata)
    add_position(nwbfile, metadata, session_df, rec_header)

    # Check that the objects were properly added
    assert "position" in nwbfile.processing["behavior"].data_interfaces
    assert "non_repeat_timestamp_labels" in nwbfile.processing
    assert "position_frame_index" in nwbfile.processing

    # save the created file
    filename = data_path / "test_add_position.nwb"
    with NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    # Read the created file and its original counterpart
    with NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()

        rec_to_nwb_file = data_path / "minirec20230622_.nwb"
        with NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()

            # check the data series was added
            for series in [
                "led_0_series_1",
                "led_0_series_2",
                "led_1_series_1",
                "led_1_series_2",
            ]:
                # check series in new nwbfile
                assert (
                    series
                    in nwbfile.processing["behavior"]["position"].spatial_series.keys()
                )
                # find the corresponding data in the old file
                validated = False
                for old_series in old_nwbfile.processing["behavior"][
                    "position"
                ].spatial_series.keys():
                    # check that led number matches
                    if not series.split("_")[1] == old_series.split("_")[1]:
                        continue
                    # check if timestamps end the same
                    timestamps = nwbfile.processing["behavior"]["position"][
                        series
                    ].timestamps[:]
                    old_timestamps = old_nwbfile.processing["behavior"]["position"][
                        old_series
                    ].timestamps[:]
                    if np.allclose(
                        timestamps[-30:],
                        old_timestamps[-30:],
                        rtol=0,
                        atol=np.mean(np.diff(old_timestamps[-30:])),
                    ):
                        pos = nwbfile.processing["behavior"]["position"][series].data[:]
                        old_pos = old_nwbfile.processing["behavior"]["position"][
                            old_series
                        ].data[:]
                        # check that the data is the same
                        assert np.allclose(pos[-30:], old_pos[-30:], rtol=0, atol=1e-6)
                        validated = True
                        break
                assert validated, f"Could not find matching series for {series}"
    # cleanup
    os.remove(filename)


from trodes_to_nwb.convert_position import read_trodes_datafile


def test_add_position_non_ptp():
    # make session_df
    path_df = get_file_info(data_path)
    session_df = path_df[(path_df.animal == "ginny")]
    # get metadata
    metadata_path = data_path / "nonptp_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    # load the prepped non-ptp nwbfile
    with NWBHDF5IO(data_path / "non_ptp_prep.nwb", "a", load_namespaces=True) as io:
        nwbfile = io.read()
        add_position(
            nwbfile,
            metadata,
            session_df,
            ptp_enabled=False,
            rec_dci_timestamps=nwbfile.processing["sample_count"][
                "sample_count"
            ].timestamps[:]
            / 1e9,
            sample_count=nwbfile.processing["sample_count"]
            .data_interfaces["sample_count"]
            .data[:],
        )
        # read in position data to compare to
        ref_pos = pd.read_pickle(data_path / "position_df.pkl")
        # check that the data is the same
        for series in [
            "led_0_series_1",
            "led_0_series_2",
            "led_1_series_1",
            "led_1_series_2",
        ]:
            # check series in new nwbfile
            assert (
                series
                in nwbfile.processing["behavior"]["position"].spatial_series.keys()
            )
            # get the data for this series
            t_new = nwbfile.processing["behavior"]["position"][series].timestamps[:]
            pos_new = nwbfile.processing["behavior"]["position"][series].data[:]
            assert t_new.size == pos_new.shape[0]
            assert pos_new.shape[1] == 2
            validated = False
            for name in ref_pos.name.to_list():
                t_ref = ref_pos[ref_pos.name == name].timestamps.values[0]

                if t_new.size == t_ref.size and np.allclose(
                    t_ref, t_new, atol=1, rtol=0
                ):
                    pos_ref = ref_pos[ref_pos.name == name].data.to_list()[0]
                    if np.allclose(
                        pos_ref[:100, :2], pos_new[:100], atol=1e-6, rtol=0
                    ) or np.allclose(
                        pos_ref[:100, 2:], pos_new[:100], atol=1e-6, rtol=0
                    ):
                        validated = True
                        break
            assert validated, f"Could not find matching series for {series}"
