import os
import subprocess
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents, Position
from pynwb.device import Device
from pynwb.image import ImageSeries

from trodes_to_nwb import convert, convert_rec_header, convert_yaml
from trodes_to_nwb.convert_position import (
    NANOSECONDS_PER_SECOND,
    add_associated_video_files,
    add_position,
    convert_datafile_to_pandas,
    convert_h264_to_mp4,
    copy_video_to_directory,
    correct_timestamps_for_camera_to_mcu_lag,
    detect_repeat_timestamps,
    detect_trodes_time_repeats_or_frame_jumps,
    estimate_camera_time_from_mcu_time,
    estimate_camera_to_mcu_lag,
    find_acquisition_timing_pause,
    find_camera_dio_channel,
    find_camera_dio_channel_per_epoch,
    find_large_frame_jumps,
    find_wrap_point,
    get_framerate,
    get_position_timestamps,
    get_video_timestamps,
    parse_dtype,
    read_trodes_datafile,
    remove_acquisition_timing_pause_non_ptp,
    wrapped_digitize,
)
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.tests.utils import data_path


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
    # Data should match the dtype defined in 'fields'
    dtype = np.dtype([("field1", np.uint32), ("field2", np.int32)])
    data_array = np.array([(1, 2), (3, 4)], dtype=dtype)

    with open(filename, "wb") as file:
        file.write(content.encode())
        file.write(data_array.tobytes())

    result = read_trodes_datafile(filename)
    assert result["clock rate"] == "30000"
    assert "data" in result
    # Check the dtype of the returned data
    assert result["data"].dtype == dtype
    # Check the content of the returned data
    assert np.array_equal(result["data"]["field1"], np.array([1, 3], dtype=np.uint32))
    assert np.array_equal(result["data"]["field2"], np.array([2, 4], dtype=np.int32))

    # Optional: Check conversion to pandas
    expected_data = convert_datafile_to_pandas(result)
    assert "field1" in expected_data.columns and "field2" in expected_data.columns
    assert expected_data["field1"].dtype == np.uint32
    assert expected_data["field2"].dtype == np.int32
    assert np.array_equal(expected_data.field1.values, np.array([1, 3]))
    assert np.array_equal(expected_data.field2.values, np.array([2, 4]))


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


def test_read_trodes_datafile_not_found(tmp_path):
    filename = tmp_path / "non_existent_file.bin"
    result = read_trodes_datafile(filename)
    assert result is None  # Expect None for file not found


def test_find_large_frame_jumps():
    convert.setup_logger("convert", "testing.log")
    frame_count = np.array([5, 10, 30, 40, 70])
    jumps = find_large_frame_jumps(frame_count, min_frame_jump=15)
    assert np.array_equal(jumps, [False, False, True, False, True])
    # Test no jumps
    frame_count_no_jump = np.array([5, 10, 15, 20, 25])
    jumps_no_jump = find_large_frame_jumps(frame_count_no_jump, min_frame_jump=15)
    assert np.array_equal(jumps_no_jump, [False] * 5)


def test_detect_repeat_timestamps():
    timestamps = np.array([1, 2, 2, 3, 3, 3, 4])
    repeats = detect_repeat_timestamps(timestamps)
    assert np.array_equal(repeats, [False, False, True, False, True, True, False])
    # Test no repeats
    timestamps_no_repeats = np.array([1, 2, 3, 4, 5])
    repeats_no_repeats = detect_repeat_timestamps(timestamps_no_repeats)
    assert np.array_equal(repeats_no_repeats, [False] * 5)


def test_detect_trodes_time_repeats_or_frame_jumps():
    convert.setup_logger("convert", "testing.log")
    # Case 1: Repeat timestamp
    trodes_time_repeat = np.array([1, 2, 2, 3, 4, 5])
    frame_count_normal = np.array([0, 10, 20, 30, 40, 50])
    labels_repeat, ids_repeat = detect_trodes_time_repeats_or_frame_jumps(
        trodes_time_repeat, frame_count_normal
    )
    # Expecting label 0 for the repeat, and a positive label for non-repeats
    assert 0 in labels_repeat  # The repeat section should be labeled 0 after filtering
    assert len(ids_repeat) == 1  # One contiguous block of valid timestamps

    # Case 2: Frame jump
    trodes_time_normal = np.array([1, 2, 3, 4, 5, 6])
    frame_count_jump = np.array([0, 10, 20, 30, 40, 1000])
    labels_jump, ids_jump = detect_trodes_time_repeats_or_frame_jumps(
        trodes_time_normal, frame_count_jump
    )
    assert 0 in labels_jump  # The jump section should be labeled 0
    assert len(ids_jump) == 1  # One block before the jump

    # Case 3: Both repeat and jump
    trodes_time_both = np.array([1, 2, 2, 3, 4, 5])
    frame_count_both = np.array([0, 10, 20, 30, 40, 1000])
    labels_both, ids_both = detect_trodes_time_repeats_or_frame_jumps(
        trodes_time_both, frame_count_both
    )
    assert 0 in labels_both
    assert len(ids_both) == 1  # Should still be one valid block initially


def test_estimate_camera_time_from_mcu_time():
    # Using index for position_timestamps for simplicity
    position_timestamps = pd.DataFrame({"val": [10, 20, 30, 40]}, index=[1, 2, 3, 5])
    # Using index for mcu_timestamps, values are the systimes
    mcu_timestamps = pd.DataFrame({"systime": [15, 25, 45]}, index=[1, 3, 5])
    camera_systime, is_valid = estimate_camera_time_from_mcu_time(
        position_timestamps, mcu_timestamps["systime"]
    )  # Pass the Series
    assert np.array_equal(camera_systime.squeeze(), [15, 25, 45])
    assert np.array_equal(is_valid, [True, False, True, True])


def test_estimate_camera_to_mcu_lag():
    convert.setup_logger("convert", "testing.log")
    camera_systime = np.array([1000, 2000, 3000]) * NANOSECONDS_PER_SECOND
    dio_systime = np.array([900, 1800, 2700]) * NANOSECONDS_PER_SECOND
    lag = estimate_camera_to_mcu_lag(camera_systime, dio_systime)
    # Median difference is 200ms
    assert np.isclose(lag / NANOSECONDS_PER_SECOND, 0.2)

    # Test with n_breaks > 0 (uses first elements)
    lag_breaks = estimate_camera_to_mcu_lag(camera_systime, dio_systime, n_breaks=1)
    # Difference of first elements is 100ms
    assert np.isclose(lag_breaks / NANOSECONDS_PER_SECOND, 0.1)


def test_remove_acquisition_timing_pause_non_ptp():
    dio_systime = np.array([100, 200, 300, 400])
    frame_count_full = np.array([5, 10, 15, 20])  # Corresponds to valid camera times
    camera_systime = np.array([50, 150, 250, 350])
    is_valid_camera_time_full = np.array(
        [True, False, True, True, True]
    )  # Example with some invalid times initially
    is_valid_camera_time_input = is_valid_camera_time_full.copy()
    pause_mid_time = 150
    # We need frame_count only for the valid camera_systime entries
    frame_count_valid = frame_count_full[[0, 2, 3]]  # Indices where valid initially

    (
        dio_systime_res,
        frame_count_res,
        is_valid_res,
        camera_systime_res,
    ) = remove_acquisition_timing_pause_non_ptp(
        dio_systime,
        frame_count_valid,
        camera_systime[[0, 2, 3]],
        is_valid_camera_time_input,
        pause_mid_time,
    )

    # dio_systime should be filtered based on pause_mid_time
    assert np.array_equal(dio_systime_res, [200, 300, 400])
    # camera_systime and frame_count should be filtered based on pause_mid_time applied to camera_systime
    assert np.array_equal(camera_systime_res, [250, 350])
    assert np.array_equal(frame_count_res, [15, 20])
    # is_valid_camera_time should reflect the final state after filtering
    # Initial valid: [T, F, T, T, T] -> Filter by camera_systime > 150: [F, F, T, T, T]
    assert np.array_equal(is_valid_res, [False, False, True, True, True])


def test_get_framerate():
    timestamps = np.array([0, 1, 2, 3]) * NANOSECONDS_PER_SECOND  # 1 second intervals
    framerate = get_framerate(timestamps)
    assert framerate == 1.0

    timestamps_30fps = np.arange(0, 1, 1 / 30) * NANOSECONDS_PER_SECOND
    framerate_30fps = get_framerate(timestamps_30fps)
    assert np.isclose(framerate_30fps, 30.0)


def test_find_acquisition_timing_pause():
    # Pause between 1s and 1.5s (0.5s duration)
    timestamps = np.array([0, 0.1, 0.2, 1.0, 1.5, 1.6, 1.7]) * NANOSECONDS_PER_SECOND
    pause_mid_time = find_acquisition_timing_pause(
        timestamps, min_duration=0.4, max_duration=0.6, n_search=10
    )
    # Midpoint between 1.0s and 1.5s is 1.25s
    assert pause_mid_time == 1.25 * NANOSECONDS_PER_SECOND

    # Pause between 0.2s and 1.0s (0.8s duration)
    pause_mid_time_long = find_acquisition_timing_pause(
        timestamps, min_duration=0.7, max_duration=0.9, n_search=10
    )
    # Midpoint between 0.2s and 1.0s is 0.6s
    assert pause_mid_time_long == 0.6 * NANOSECONDS_PER_SECOND

    # No pause within range
    with pytest.raises(IndexError):
        find_acquisition_timing_pause(
            timestamps, min_duration=1.0, max_duration=2.0, n_search=10
        )


def test_correct_timestamps_for_camera_to_mcu_lag():
    frame_count = np.arange(5)
    # Perfect linear relationship: time = 10 * frame + 5 (in seconds)
    camera_systime = (frame_count * 10 + 5) * NANOSECONDS_PER_SECOND
    camera_to_mcu_lag = (
        2 * NANOSECONDS_PER_SECOND
    )  # Constant lag of 2 seconds (or 2e9 ns)

    # Expected corrected time = camera_systime - lag
    # = (frame * 10 + 5) - 2 = frame * 10 + 3 (in seconds)
    expected_corrected_camera_systime = (frame_count * 10 + 3) * NANOSECONDS_PER_SECOND

    corrected_camera_systime = correct_timestamps_for_camera_to_mcu_lag(
        frame_count, camera_systime, camera_to_mcu_lag
    )

    # Assert that the corrected timestamps are as expected (allow for float precision)
    assert np.allclose(
        corrected_camera_systime, expected_corrected_camera_systime, rtol=1e-9
    )


def test_find_wrap_point():
    # Case 1: Wrap point exists
    t_wrap = np.array([5, 6, 7, 0, 1, 2])
    assert find_wrap_point(t_wrap) == 3

    # Case 2: No wrap point (strictly increasing)
    t_no_wrap = np.array([0, 1, 2, 3, 4, 5])
    assert find_wrap_point(t_no_wrap) is None

    # Case 3: Edge case - wrap at the beginning
    t_wrap_start = np.array([6, 0, 1, 2, 3, 4])
    assert find_wrap_point(t_wrap_start) == 1

    # Case 4: Edge case - wrap at the end
    t_wrap_end = np.array([0, 1, 2, 3, 4, -1])  # Using negative for illustration
    assert find_wrap_point(t_wrap_end) == 5

    # Case 5: Array with two elements
    t_two_wrap = np.array([5, 0])
    assert find_wrap_point(t_two_wrap) == 1
    t_two_no_wrap = np.array([0, 5])
    assert find_wrap_point(t_two_no_wrap) is None


@patch("trodes_to_nwb.convert_position.read_trodes_datafile")
def test_get_video_timestamps(mock_read_datafile, tmp_path):
    # Mock the output of read_trodes_datafile
    mock_data = {
        "data": np.array(
            [(10, 1e9), (20, 2e9), (30, 3e9)],  # (frameCount, HWTimestamp)
            dtype=[("frameCount", "i4"), ("HWTimestamp", "u8")],
        )
    }
    mock_read_datafile.return_value = mock_data

    filepath = tmp_path / "video_timestamps.cameraHWSync"
    timestamps = get_video_timestamps(filepath)

    # Expected timestamps in seconds
    expected_timestamps = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(timestamps, expected_timestamps)
    mock_read_datafile.assert_called_once_with(filepath)


def test_find_camera_dio_channel():
    # Mock NWBFile structure
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=pd.Timestamp.now(tz="UTC"),
    )
    behavior_module = nwbfile.create_processing_module("behavior", "behavior data")
    behavioral_events = BehavioralEvents(name="behavioral_events")
    behavior_module.add(behavioral_events)

    # Case 1: One matching channel
    ts1 = TimeSeries(
        name="channel_1_camera_ticks", data=[1], timestamps=[0.1], unit="V"
    )
    behavioral_events.add_timeseries(ts1)
    assert find_camera_dio_channel(nwbfile) is ts1.timestamps

    # Case 2: No matching channel
    nwbfile.processing["behavior"].data_interfaces[
        "behavioral_events"
    ].time_series.clear()
    ts2 = TimeSeries(
        name="channel_2_other_signal", data=[1], timestamps=[0.1], unit="V"
    )
    behavioral_events.add_timeseries(ts2)
    with pytest.raises(ValueError, match="No camera dio channel found by name"):
        find_camera_dio_channel(nwbfile)

    # Case 3: Multiple matching channels
    nwbfile.processing["behavior"].data_interfaces[
        "behavioral_events"
    ].time_series.clear()
    ts3a = TimeSeries(name="cam1_camera_ticks", data=[1], timestamps=[0.1], unit="V")
    ts3b = TimeSeries(name="cam2_camera_ticks", data=[1], timestamps=[0.2], unit="V")
    behavioral_events.add_timeseries(ts3a)
    behavioral_events.add_timeseries(ts3b)
    with pytest.raises(ValueError, match="Multiple camera dio channels found"):
        find_camera_dio_channel(nwbfile)

    # Case 4: Behavioral events interface missing
    nwbfile.processing["behavior"].data_interfaces.pop("behavioral_events")
    with pytest.raises(
        KeyError
    ):  # Or appropriate error depending on implementation access
        find_camera_dio_channel(nwbfile)


def test_find_camera_dio_channel_per_epoch():
    # Mock NWBFile structure
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=pd.Timestamp.now(tz="UTC"),
    )
    behavior_module = nwbfile.create_processing_module("behavior", "behavior data")
    behavioral_events = BehavioralEvents(name="behavioral_events")
    behavior_module.add(behavioral_events)

    # Mock timestamps for two potential camera channels
    timestamps_cam1 = np.linspace(0, 10, 200)  # 200 ticks between 0 and 10s
    timestamps_cam2 = np.linspace(5, 15, 50)  # 50 ticks between 5 and 15s

    ts_cam1 = TimeSeries(
        name="channel_A_camera_ticks",
        data=np.ones_like(timestamps_cam1),
        timestamps=timestamps_cam1,
        unit="V",
    )
    ts_cam2 = TimeSeries(
        name="channel_B_camera_ticks",
        data=np.ones_like(timestamps_cam2),
        timestamps=timestamps_cam2,
        unit="V",
    )
    ts_other = TimeSeries(name="other_signal", data=[1], timestamps=[1], unit="V")

    behavioral_events.add_timeseries(ts_cam1)
    behavioral_events.add_timeseries(ts_cam2)
    behavioral_events.add_timeseries(ts_other)

    # Case 1: Epoch [1, 4] - Should find cam1 with > 100 ticks
    epoch_start_1, epoch_end_1 = 1.0, 4.0
    result_1 = find_camera_dio_channel_per_epoch(nwbfile, epoch_start_1, epoch_end_1)
    expected_1 = timestamps_cam1[
        (timestamps_cam1 >= epoch_start_1) & (timestamps_cam1 <= epoch_end_1)
    ]
    assert np.array_equal(result_1, expected_1)
    assert len(result_1) > 100

    # Case 2: Epoch [6, 8] - Should find cam1 first (more ticks), even though cam2 overlaps
    epoch_start_2, epoch_end_2 = 6.0, 8.0
    result_2 = find_camera_dio_channel_per_epoch(nwbfile, epoch_start_2, epoch_end_2)
    expected_2 = timestamps_cam1[
        (timestamps_cam1 >= epoch_start_2) & (timestamps_cam1 <= epoch_end_2)
    ]
    assert np.array_equal(result_2, expected_2)
    assert len(result_2) > 100

    # Case 3: Epoch [12, 14] - Should find cam2 (only one with ticks here > 0, but < 100) -> Raises Error
    epoch_start_3, epoch_end_3 = 12.0, 14.0
    with pytest.raises(ValueError, match="No camera dio has sufficient ticks"):
        find_camera_dio_channel_per_epoch(nwbfile, epoch_start_3, epoch_end_3)

    # Case 4: Epoch [16, 20] - No camera ticks
    epoch_start_4, epoch_end_4 = 16.0, 20.0
    with pytest.raises(ValueError, match="No camera dio has sufficient ticks"):
        find_camera_dio_channel_per_epoch(nwbfile, epoch_start_4, epoch_end_4)

    # Case 5: No "camera ticks" channels at all
    nwbfile.processing["behavior"].data_interfaces[
        "behavioral_events"
    ].time_series.clear()
    nwbfile.processing["behavior"].data_interfaces["behavioral_events"].add_timeseries(
        ts_other
    )
    with pytest.raises(ValueError, match="No camera dio channel found by name"):
        find_camera_dio_channel_per_epoch(nwbfile, 0, 1)


@patch("trodes_to_nwb.convert_position.subprocess.run")
@patch("trodes_to_nwb.convert_position.Path.exists")
def test_convert_h264_to_mp4(mock_exists, mock_run, tmp_path):
    convert.setup_logger("convert", "testing.log")
    input_file = "/path/to/video.h264"
    video_dir = tmp_path

    # Case 1: Output file does not exist, conversion runs
    mock_exists.return_value = False
    expected_output_file = video_dir / "video.mp4"
    result = convert_h264_to_mp4(input_file, str(video_dir))
    assert result == str(expected_output_file)
    mock_exists.assert_called_once_with()  # Called on the Path object instance
    mock_run.assert_called_once_with(
        f"ffmpeg -i {input_file} {expected_output_file}", shell=True
    )

    # Reset mocks for next case
    mock_exists.reset_mock()
    mock_run.reset_mock()

    # Case 2: Output file already exists, skips conversion
    mock_exists.return_value = True
    result = convert_h264_to_mp4(input_file, str(video_dir))
    assert result == str(expected_output_file)
    mock_exists.assert_called_once_with()
    mock_run.assert_not_called()

    # Case 3: ffmpeg command fails (raises CalledProcessError)
    mock_exists.return_value = False
    mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
    with pytest.raises(subprocess.CalledProcessError):
        convert_h264_to_mp4(input_file, str(video_dir))


@patch("trodes_to_nwb.convert_position.subprocess.run")
@patch("trodes_to_nwb.convert_position.Path.exists")
def test_copy_video_to_directory(mock_exists, mock_run, tmp_path):
    convert.setup_logger("convert", "testing.log")
    input_file = "/path/to/video.avi"
    video_dir = tmp_path

    # Case 1: Output file does not exist, copy runs
    mock_exists.return_value = False
    expected_output_file = video_dir / "video.avi"
    result = copy_video_to_directory(input_file, str(video_dir))
    assert result == str(expected_output_file)
    mock_exists.assert_called_once_with()
    mock_run.assert_called_once_with(
        f"cp {input_file} {expected_output_file}", shell=True
    )

    # Reset mocks
    mock_exists.reset_mock()
    mock_run.reset_mock()

    # Case 2: Output file already exists, skips copy
    mock_exists.return_value = True
    result = copy_video_to_directory(input_file, str(video_dir))
    assert result == str(expected_output_file)
    mock_exists.assert_called_once_with()
    mock_run.assert_not_called()

    # Case 3: cp command fails
    mock_exists.return_value = False
    mock_run.side_effect = subprocess.CalledProcessError(1, "cp")
    with pytest.raises(subprocess.CalledProcessError):
        copy_video_to_directory(input_file, str(video_dir))


@patch("trodes_to_nwb.convert_position.get_video_timestamps")
@patch("trodes_to_nwb.convert_position.convert_h264_to_mp4")
@patch("trodes_to_nwb.convert_position.copy_video_to_directory")
def test_add_associated_video_files(
    mock_copy, mock_convert, mock_get_timestamps, tmp_path
):
    # Setup Mock NWB File
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=pd.Timestamp.now(tz="UTC"),
    )
    # Add required device
    camera_device = Device(name="camera_device 0")
    nwbfile.add_device(camera_device)

    # Mock Metadata
    metadata = {
        "associated_video_files": [
            {
                "name": "video_epoch1.h264",
                "camera_id": 0,
                "task_epochs": [1],
            }
        ]
    }

    # Mock session_df
    session_df = pd.DataFrame(
        {
            "full_path": [
                str(tmp_path / "path/to/video_epoch1.rec.1.h264"),  # Video file
                str(tmp_path / "path/to/hw_sync.rec.1.cameraHWSync"),
            ],  # HWSync file
            "epoch": [1, 1],
            "file_extension": [".h264", ".cameraHWSync"],
        }
    )

    # Mock video directory
    video_directory = str(tmp_path / "output_videos")
    os.makedirs(video_directory, exist_ok=True)

    # Mock return values
    mock_timestamps = np.array([1.0, 2.0, 3.0])
    mock_get_timestamps.return_value = mock_timestamps
    mock_converted_path = str(Path(video_directory) / "video_epoch1.mp4")
    mock_copied_path = str(Path(video_directory) / "video_epoch1.h264")
    mock_convert.return_value = mock_converted_path
    mock_copy.return_value = mock_copied_path

    # --- Test Case 1: convert_video = True ---
    add_associated_video_files(
        nwbfile, metadata, session_df, video_directory, convert_video=True
    )

    # Check if processing module and BehavioralEvents were created
    assert "video_files" in nwbfile.processing
    assert "video" in nwbfile.processing["video_files"].data_interfaces
    video_events = nwbfile.processing["video_files"]["video"]
    assert isinstance(video_events, BehavioralEvents)

    # Check if ImageSeries was added correctly
    assert "video_epoch1.h264" in video_events.time_series
    img_series = video_events.time_series["video_epoch1.h264"]
    assert isinstance(img_series, ImageSeries)
    assert np.array_equal(img_series.timestamps[:], mock_timestamps)
    assert img_series.device == camera_device
    assert img_series.external_file[0] == "video_epoch1.mp4"  # Check converted name
    assert img_series.format == "external"

    # Check if mocks were called correctly
    mock_get_timestamps.assert_called_once_with(
        session_df.iloc[1]["full_path"]
    )  # Called with HWSync path
    mock_convert.assert_called_once_with(
        session_df.iloc[0]["full_path"], video_directory
    )  # Called with video path
    mock_copy.assert_not_called()

    # --- Reset and Test Case 2: convert_video = False ---
    # Re-create NWB file parts as they were modified
    nwbfile.processing.clear()
    mock_get_timestamps.reset_mock()
    mock_convert.reset_mock()
    mock_copy.reset_mock()

    add_associated_video_files(
        nwbfile, metadata, session_df, video_directory, convert_video=False
    )

    # Check ImageSeries again
    assert "video_files" in nwbfile.processing
    assert "video" in nwbfile.processing["video_files"].data_interfaces
    img_series_copy = nwbfile.processing["video_files"]["video"].time_series[
        "video_epoch1.h264"
    ]
    assert (
        img_series_copy.external_file[0] == "video_epoch1.h264"
    )  # Check original name

    # Check mocks
    mock_get_timestamps.assert_called_once_with(session_df.iloc[1]["full_path"])
    mock_convert.assert_not_called()
    mock_copy.assert_called_once_with(session_df.iloc[0]["full_path"], video_directory)

    # --- Test Case 3: Video file not found in session_df ---
    nwbfile.processing.clear()
    bad_metadata = {
        "associated_video_files": [
            {
                "name": "nonexistent_video.h264",
                "camera_id": 0,
                "task_epochs": [1],
            }
        ]
    }
    with pytest.raises(FileNotFoundError):
        add_associated_video_files(nwbfile, bad_metadata, session_df, video_directory)

    # --- Test Case 4: cameraHWSync file not found ---
    nwbfile.processing.clear()
    bad_session_df = pd.DataFrame(
        {
            "full_path": [str(tmp_path / "path/to/video_epoch1.rec.1.h264")],
            "epoch": [1],
            "file_extension": [".h264"],  # Missing .cameraHWSync
        }
    )
    with pytest.raises(ValueError, match="No cameraHWSync found"):
        add_associated_video_files(nwbfile, metadata, bad_session_df, video_directory)


# --- Keep Existing Integration Tests ---
def test_add_position(prior_position=False):
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

    # Optional test: add position data to nwbfile before running add_position
    if prior_position:
        nwbfile.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )
        position = Position(name="position")
        nwbfile.processing["behavior"].add(position)

    # run add_position and prerequisite functions
    convert_yaml.add_cameras(nwbfile, metadata)
    # Mock get_position_timestamps to avoid complex setup/real file dependency
    with patch(
        "trodes_to_nwb.convert_position.get_position_timestamps"
    ) as mock_get_pos:
        # Create realistic mock dataframes for each epoch
        mock_pos_df_epoch1 = pd.DataFrame(
            {
                "xloc": np.random.rand(100),
                "yloc": np.random.rand(100),
                "xloc2": np.random.rand(100),
                "yloc2": np.random.rand(100),
                "video_frame_ind": np.arange(100),
                "non_repeat_timestamp_labels": np.ones(100),
            },
            index=pd.Index(np.linspace(0, 10, 100), name="time"),
        )
        mock_pos_df_epoch2 = pd.DataFrame(
            {
                "xloc": np.random.rand(50),
                "yloc": np.random.rand(50),
                "xloc2": np.random.rand(50),
                "yloc2": np.random.rand(50),
                "video_frame_ind": np.arange(50),
                "non_repeat_timestamp_labels": np.ones(50)
                * 2,  # Different label for epoch 2
            },
            index=pd.Index(np.linspace(11, 15, 50), name="time"),
        )
        mock_get_pos.side_effect = [
            mock_pos_df_epoch1,
            mock_pos_df_epoch2,
        ]  # Return df for each epoch call

        add_position(
            nwbfile, metadata, session_df, ptp_enabled=True
        )  # Assume PTP for simplicity here

        # Check that the objects were properly added
        assert "behavior" in nwbfile.processing
        assert "position" in nwbfile.processing["behavior"].data_interfaces
        assert "non_repeat_timestamp_labels" in nwbfile.processing
        assert "position_frame_index" in nwbfile.processing

        # Check spatial series were created
        assert (
            "led_0_series_1"
            in nwbfile.processing["behavior"]["position"].spatial_series
        )
        assert (
            "led_1_series_1"
            in nwbfile.processing["behavior"]["position"].spatial_series
        )
        assert (
            "led_0_series_2"
            in nwbfile.processing["behavior"]["position"].spatial_series
        )
        assert (
            "led_1_series_2"
            in nwbfile.processing["behavior"]["position"].spatial_series
        )

        # Check data shape and timestamps match mock data for one series per epoch
        series1 = nwbfile.processing["behavior"]["position"]["led_0_series_1"]
        assert np.array_equal(series1.timestamps[:], mock_pos_df_epoch1.index.values)
        assert np.allclose(series1.data[:, 0], mock_pos_df_epoch1["xloc"].values)
        assert (
            series1.conversion == metadata["cameras"][0]["meters_per_pixel"]
        )  # Check conversion factor

        series2 = nwbfile.processing["behavior"]["position"]["led_1_series_2"]
        assert np.array_equal(series2.timestamps[:], mock_pos_df_epoch2.index.values)
        assert np.allclose(series2.data[:, 1], mock_pos_df_epoch2["yloc2"].values)

        # Check frame index and labels TimeSeries
        frame_ts1 = nwbfile.processing["position_frame_index"]["series_1"]
        assert np.array_equal(frame_ts1.timestamps[:], mock_pos_df_epoch1.index.values)
        assert np.array_equal(
            frame_ts1.data[:], mock_pos_df_epoch1["video_frame_ind"].values
        )

        label_ts2 = nwbfile.processing["non_repeat_timestamp_labels"]["series_2"]
        assert np.array_equal(label_ts2.timestamps[:], mock_pos_df_epoch2.index.values)
        assert np.array_equal(
            label_ts2.data[:], mock_pos_df_epoch2["non_repeat_timestamp_labels"].values
        )

        # Check that get_position_timestamps was called twice (once per epoch)
        assert mock_get_pos.call_count == 2


def test_add_position_preexisting():
    test_add_position(prior_position=True)


@patch("trodes_to_nwb.convert_position.get_position_timestamps")
@patch("trodes_to_nwb.convert_position.find_camera_dio_channel_per_epoch")
def test_add_position_non_ptp(mock_find_dio, mock_get_pos, tmp_path):
    session_df = pd.DataFrame(
        {
            "full_path": [
                str(tmp_path / "epoch1.videoPositionTracking"),
                str(tmp_path / "epoch1.cameraHWSync"),
                str(tmp_path / "epoch2.videoPositionTracking"),
                str(tmp_path / "epoch2.cameraHWSync"),
            ],
            "epoch": [1, 1, 2, 2],
            "file_extension": [
                ".videoPositionTracking",
                ".cameraHWSync",
                ".videoPositionTracking",
                ".cameraHWSync",
            ],
            "animal": ["ginny"] * 4,  # Add animal if needed for filtering later
        }
    )
    # Create dummy files so Paths resolve
    for p in session_df.full_path:
        Path(p).touch()

    # Get metadata
    metadata = {
        "session_id": "test_nonptp",
        "session_start_time": pd.Timestamp.now(tz="UTC"),
        "experiment_description": "desc",
        "identifier": "id",
        "institution": "inst",
        "lab": "lab",
        "cameras": [{"id": 0, "name": "cam0", "meters_per_pixel": 0.001}],
        "tasks": [{"camera_id": [0], "task_epochs": [1, 2]}],  # Assign camera to epochs
    }

    # Mock NWB file with epochs and sample count
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_nonptp",
        session_start_time=metadata["session_start_time"],
    )
    convert_yaml.add_cameras(nwbfile, metadata)  # Add camera device
    # Add mock epochs
    epoch_module = nwbfile.add_epoch(start_time=0.0, stop_time=10.0, tags="epoch1")
    nwbfile.add_epoch(start_time=10.0, stop_time=20.0, tags="epoch2")
    # Add mock sample count data needed for non-PTP
    sample_count_data = np.arange(
        0, 30000 * 20, 30
    )  # Trodes timestamps (e.g., 30kHz rate for 20s)
    rec_dci_timestamps = np.linspace(
        0, 20, len(sample_count_data)
    )  # Corresponding system clock times
    pm = nwbfile.create_processing_module("sample_count", "description")
    ts = TimeSeries(
        name="sample_count",
        data=sample_count_data,
        timestamps=rec_dci_timestamps * NANOSECONDS_PER_SECOND,
        unit="samples",
    )
    pm.add(ts)

    # Mock the return values of the patched functions
    mock_dio_timestamps = np.linspace(0.1, 19.9, 20 * 30)  # Mock DIO ticks (e.g., 30Hz)
    mock_find_dio.return_value = (
        mock_dio_timestamps  # Assume same channel for both epochs for simplicity
    )

    # Mock position dataframes returned by get_position_timestamps
    mock_pos_df_epoch1 = pd.DataFrame(
        {
            "xloc": np.random.rand(100),
            "yloc": np.random.rand(100),
            "video_frame_ind": np.arange(100),
            "non_repeat_timestamp_labels": np.ones(100),
        },
        index=pd.Index(np.linspace(0.5, 9.5, 100), name="time"),
    )
    mock_pos_df_epoch2 = pd.DataFrame(
        {
            "xloc2": np.random.rand(50),
            "yloc2": np.random.rand(50),
            "video_frame_ind": np.arange(50),
            "non_repeat_timestamp_labels": np.ones(50) * 2,
        },
        index=pd.Index(np.linspace(10.5, 19.5, 50), name="time"),
    )
    mock_get_pos.side_effect = [mock_pos_df_epoch1, mock_pos_df_epoch2]

    # Run the function
    add_position(
        nwbfile,
        metadata,
        session_df,
        ptp_enabled=False,
        rec_dci_timestamps=rec_dci_timestamps,  # Pass system clock times
        sample_count=sample_count_data,  # Pass trodes timestamps
    )

    # Assertions
    assert "behavior" in nwbfile.processing
    assert "position" in nwbfile.processing["behavior"].data_interfaces
    assert "led_0_series_1" in nwbfile.processing["behavior"]["position"].spatial_series
    assert (
        "led_1_series_2" in nwbfile.processing["behavior"]["position"].spatial_series
    )  # LED1 from epoch 2

    # Check calls to patched functions
    assert mock_find_dio.call_count == 2
    mock_find_dio.assert_has_calls(
        [
            call(nwb_file=nwbfile, epoch_start=0.0, epoch_end=10.0),
            call(nwb_file=nwbfile, epoch_start=10.0, epoch_end=20.0),
        ]
    )

    assert mock_get_pos.call_count == 2
    # Check some args passed to get_position_timestamps for the first call (epoch 1)
    call_args_epoch1 = mock_get_pos.call_args_list[0][1]
    assert call_args_epoch1["ptp_enabled"] is False
    assert (
        call_args_epoch1["position_tracking_filepath"]
        == session_df.iloc[0]["full_path"]
    )
    assert (
        call_args_epoch1["position_timestamps_filepath"]
        == session_df.iloc[1]["full_path"]
    )
    assert np.array_equal(call_args_epoch1["rec_dci_timestamps"], rec_dci_timestamps)
    assert np.array_equal(call_args_epoch1["sample_count"], sample_count_data)
    assert np.array_equal(
        call_args_epoch1["dio_camera_timestamps"], mock_dio_timestamps
    )  # Should be the full array passed to get_position_timestamps
    assert call_args_epoch1["epoch_interval"] == [0.0, 10.0]
