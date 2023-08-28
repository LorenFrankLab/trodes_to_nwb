import numpy as np
import pandas as pd
import pytest

from spikegadgets_to_nwb.convert_position import (
    detect_repeat_timestamps,
    detect_trodes_time_repeats_or_frame_jumps,
    estimate_camera_time_from_mcu_time,
    estimate_camera_to_mcu_lag,
    find_acquisition_timing_pause,
    find_large_frame_jumps,
    get_framerate,
    parse_dtype,
    read_trodes_datafile,
    remove_acquisition_timing_pause_non_ptp,
    correct_timestamps_for_camera_to_mcu_lag,
)


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
    frame_count = np.array([5, 10, 30, 40, 70])
    jumps = find_large_frame_jumps(frame_count, min_frame_jump=15)
    assert np.array_equal(jumps, [False, False, True, False, True])


def test_detect_repeat_timestamps():
    timestamps = np.array([1, 2, 2, 3, 3, 3, 4])
    repeats = detect_repeat_timestamps(timestamps)
    assert np.array_equal(repeats, [False, False, True, False, True, True, False])


def test_detect_trodes_time_repeats_or_frame_jumps():
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
