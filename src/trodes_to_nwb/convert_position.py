"""Handles the conversion of position tracking data (from .videoPositionTracking,
.cameraHWSync files) and associated video files (.h264) into NWB Position spatial series
and ImageSeries. Includes logic for PTP and non-PTP timestamp alignment.
"""

import datetime
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents, Position
from pynwb.image import ImageSeries
from scipy.ndimage import label
from scipy.stats import linregress

NANOSECONDS_PER_SECOND = 1e9
DEFAULT_MIN_PTP_PAUSE_S = (
    0.4  # Minimum duration for detecting acquisition timing pause (PTP)
)
DEFAULT_MAX_PTP_PAUSE_S = (
    2.0  # Maximum duration for detecting acquisition timing pause (PTP)
)


def find_wrap_point(t: np.ndarray) -> Optional[int]:
    """
    Finds the point at which the timestamps wrap around due to overflow.
    Returns None if no wrap point is found
    Parameters
    ----------
    t : np.ndarray
        Array of timestamps
    Returns
    -------
    wrap_point : int or None
        Index of the wrap point or None if no wrap point is found
    """
    wrap_point = None
    rng = [0, len(t) - 1]
    while t[rng[1]] <= t[rng[0]]:
        mid = np.mean(rng, dtype=int)
        if t[mid] <= t[rng[0]]:
            rng[1] = mid
        else:
            rng[0] = mid
        if rng[0] == rng[1]:
            wrap_point = rng[0] + 1
            break
    return wrap_point


def wrapped_digitize(
    x: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Digitize a location with timestamps that wrap around due to overflow.

    Parameters
    ----------
    x : np.ndarray
        indeces to digitize
    bins : np.ndarray
        bins to digitize into

    Returns
    -------
    np.ndarray
        digitized indices
    """
    wrap_point = find_wrap_point(bins)
    if wrap_point is None:
        return np.digitize(x, bins, right=True)
    ind_first = np.digitize(x, bins[:wrap_point], right=True)
    ind_second = np.digitize(x, bins[wrap_point:], right=True) + wrap_point
    section = (x < bins[0]).astype(int)  # True if in the second section (post-wrap)
    return ind_first * (1 - section) + ind_second * section


def parse_dtype(fieldstr: str) -> np.dtype:
    """
    Parses the last fields parameter (<time uint32><...>) as a single string.
    Assumes it is formatted as <name number * type> or <name type>. Returns a numpy dtype object.

    Parameters
    ----------
    fieldstr : str
        The string to parse.

    Returns
    -------
    np.dtype
        The numpy dtype object.

    Raises
    ------
    AttributeError
        If the field type is not valid.

    Examples
    --------
    >>> fieldstr = '<time uint32><x float32><y float32><z float32>'
    >>> parse_dtype(fieldstr)
    dtype([('time', '<u4', (1,)), ('x', '<f4', (1,)), ('y', '<f4', (1,)), ('z', '<f4', (1,))])

    """
    # Returns np.dtype from field string
    sep = " ".join(
        fieldstr.replace("><", " ").replace(">", " ").replace("<", " ").split()
    ).split()

    typearr = []

    # Every two elemets is fieldname followed by datatype
    for i in range(0, len(sep), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = "uint32"
        # Finds if a <num>* is included in datatype
        if "*" in sep[i + 1]:
            temptypes = re.split("\*", sep[i + 1])
            # Results in the correct assignment, whether str is num*dtype or dtype*num
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i + 1]
        try:
            fieldtype = getattr(np, ftype)
        except AttributeError:
            raise AttributeError(ftype + " is not a valid field type.\n")
        else:
            typearr.append((str(fieldname), fieldtype, repeats))

    return np.dtype(typearr)


def read_trodes_datafile(filename: Path) -> Optional[Dict[str, Any]]:
    """
    Read trodes binary.

    Parameters
    ----------
    filename : Path
        Path to the trodes binary file.

    Returns
    -------
    dict or None
        A dictionary containing the settings and data from the trodes binary file,
        or None if an error occurs (e.g., file not found).

    Raises
    ------
    Exception
        If the settings format is not supported.

    """
    logger = logging.getLogger("convert")
    try:
        with open(filename, "rb") as file:
            # Check if first line is start of settings block
            if file.readline().decode().strip() != "<Start settings>":
                raise Exception("Settings format not supported")
            fields_text = dict()
            for line in file:
                # Read through block of settings
                line = line.decode().strip()
                # filling in fields dict
                if line != "<End settings>":
                    settings_name, setting = line.split(": ")
                    fields_text[settings_name.lower()] = setting
                # End of settings block, signal end of fields
                else:
                    break
            # Reads rest of file at once, using dtype format generated by parse_dtype()
            try:
                fields_text["data"] = np.fromfile(
                    file, dtype=parse_dtype(fields_text["fields"])
                )
            except KeyError:
                fields_text["data"] = np.fromfile(file)
            return fields_text
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return None
    except PermissionError:
        logger.error(f"Permission denied when trying to read: {filename}")
        return None
    except IOError as e:
        logger.error(f"An I/O error occurred while reading {filename}: {e}")
        return None
    except Exception as e:  # Catch-all for unexpected errors during file processing
        logger.error(f"An unexpected error occurred processing {filename}: {e}")
        raise


def convert_datafile_to_pandas(datafile: Dict[str, Any]) -> pd.DataFrame:
    """Takes the output of read_trodes_datafile and converts it to a pandas dataframe.
    Added for changes identified in numpy 2.2.2

    Parameters
    ----------
    datafile : Dict[str, Any]
        The data file dictionary containing the data to convert.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the converted data.
    """
    return pd.DataFrame(
        {key: np.squeeze(datafile["data"][key]) for key in datafile["data"].dtype.names}
    )


def get_framerate(timestamps: np.ndarray) -> float:
    """
    Calculates the framerate of a video based on the timestamps of each frame.

    Parameters
    ----------
    timestamps : np.ndarray
        An array of timestamps for each frame in the video, units = nanoseconds.

    Returns
    -------
    frame_rate: float
        The framerate of the video in frames per second.
    """
    timestamps = np.asarray(timestamps)
    return NANOSECONDS_PER_SECOND / np.median(np.diff(timestamps))


def find_acquisition_timing_pause(
    timestamps: np.ndarray,
    min_duration: float = DEFAULT_MIN_PTP_PAUSE_S,
    max_duration: float = DEFAULT_MAX_PTP_PAUSE_S,
    n_search: int = 100,
) -> float:
    """
    Find the midpoint time of a timing pause in the video stream.

    Parameters
    ----------
    timestamps : np.ndarray
        An array of timestamps for each frame in the video. Expects units=nanoseconds.
    min_duration : float, optional
        The minimum duration of the pause in seconds, by default 0.4.
    max_duration : float, optional
        The maximum duration of the pause in seconds, by default 1.0.
    n_search : int, optional
        The number of frames to search for the pause, by default 100.

    Returns
    -------
    pause_mid_time : float
        The midpoint time of the timing pause in nanoseconds.

    Raises
    ------
    IndexError
        If no valid timing pause is found within the search window.

    """
    timestamps = np.asarray(timestamps)
    timestamp_difference = np.diff(timestamps[:n_search] / NANOSECONDS_PER_SECOND)
    is_valid_gap = (timestamp_difference > min_duration) & (
        timestamp_difference < max_duration
    )
    pause_start_ind = np.nonzero(is_valid_gap)[0][0]
    pause_end_ind = pause_start_ind + 1
    pause_mid_time = (
        timestamps[pause_start_ind]
        + (timestamps[pause_end_ind] - timestamps[pause_start_ind]) // 2
    )

    return pause_mid_time


def find_large_frame_jumps(
    frame_count: np.ndarray, min_frame_jump: int = 15
) -> np.ndarray:
    """
    Find large frame jumps in the video.

    Parameters
    ----------
    frame_count : np.ndarray
        An array of frame counts for each frame in the video.
    min_frame_jump : int, optional
        The minimum number of frames to consider a jump as large, by default 15.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each frame has a large jump.

    """
    logger = logging.getLogger("convert")
    frame_count = np.asarray(frame_count)

    is_large_frame_jump = np.insert(np.diff(frame_count) > min_frame_jump, 0, False)

    logger.info(f"big frame jumps: {np.nonzero(is_large_frame_jump)[0]}")

    return is_large_frame_jump


def detect_repeat_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Detects repeated timestamps in an array of timestamps.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps.

    Returns
    -------
    np.ndarray
        Boolean array indicating whether each timestamp is repeated.
    """
    if len(timestamps) < 2:
        return np.array([False] * len(timestamps), dtype=bool)
    # Detect where current timestamp is <= previous timestamp
    return np.insert(timestamps[:-1] >= timestamps[1:], 0, False)


def detect_trodes_time_repeats_or_frame_jumps(
    trodes_time: np.ndarray, frame_count: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects if a Trodes time index repeats, indicating that the Trodes clock has frozen
    due to headstage disconnects. Also detects large frame jumps.

    Parameters
    ----------
    trodes_time : np.ndarray
        Array of Trodes time indices.
    frame_count : np.ndarray
        Array of frame counts.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - non_repeat_timestamp_labels : np.ndarray
            Array of labels for non-repeating timestamps.
        - non_repeat_timestamp_labels_id : np.ndarray
            Array of unique IDs for non-repeating timestamps.
    """
    logger = logging.getLogger("convert")

    trodes_time = np.asarray(trodes_time)
    is_repeat_timestamp = detect_repeat_timestamps(trodes_time)
    logger.info(f"repeat timestamps ind: {np.nonzero(is_repeat_timestamp)[0]}")

    is_large_frame_jump = find_large_frame_jumps(frame_count)
    is_repeat_timestamp = np.logical_or(is_repeat_timestamp, is_large_frame_jump)

    repeat_timestamp_labels = label(is_repeat_timestamp)[0]
    repeat_timestamp_labels_id, repeat_timestamp_label_counts = np.unique(
        repeat_timestamp_labels, return_counts=True
    )
    is_repeat = np.logical_and(
        repeat_timestamp_labels_id != 0, repeat_timestamp_label_counts > 2
    )
    repeat_timestamp_labels_id = repeat_timestamp_labels_id[is_repeat]
    repeat_timestamp_label_counts = repeat_timestamp_label_counts[is_repeat]
    is_repeat_timestamp[
        ~np.isin(repeat_timestamp_labels, repeat_timestamp_labels_id)
    ] = False

    non_repeat_timestamp_labels = label(~is_repeat_timestamp)[0]
    non_repeat_timestamp_labels_id = np.unique(non_repeat_timestamp_labels)
    non_repeat_timestamp_labels_id = non_repeat_timestamp_labels_id[
        non_repeat_timestamp_labels_id != 0
    ]

    return (non_repeat_timestamp_labels, non_repeat_timestamp_labels_id)


def estimate_camera_time_from_mcu_time(
    position_timestamps: np.ndarray, mcu_timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    position_timestamps : pd.DataFrame
    mcu_timestamps : pd.DataFrame

    Returns
    -------
    camera_systime : np.ndarray, shape (n_frames_within_neural_time,)
    is_valid_camera_time : np.ndarray, shape (n_frames,)

    """
    is_valid_camera_time = np.isin(position_timestamps.index, mcu_timestamps.index)
    camera_systime = np.asarray(
        mcu_timestamps.loc[position_timestamps.index[is_valid_camera_time]]
    )

    return camera_systime, is_valid_camera_time


def estimate_camera_to_mcu_lag(
    camera_systime: np.ndarray, dio_systime: np.ndarray, n_breaks: int = 0
) -> float:
    """
    Estimate lag between camera frame system time and DIO trigger system time.

    Parameters
    ----------
    camera_systime : np.ndarray
        System timestamps (nanoseconds) of camera frames.
    dio_systime : np.ndarray
        System timestamps (nanoseconds) of corresponding DIO events.
    n_breaks : int, optional
        Number of detected breaks/discontinuities in the data. If 0, uses median lag.
        If > 0, uses lag from the first pair of points (less robust). Default is 0.

    Returns
    -------
    float
        Estimated lag in nanoseconds. Positive means camera time is later than DIO time.
    """
    logger = logging.getLogger("convert")
    if n_breaks == 0:
        dio_systime = dio_systime[: len(camera_systime)]
        camera_to_mcu_lag = np.median(camera_systime - dio_systime)
    else:
        camera_to_mcu_lag = camera_systime[0] - dio_systime[0]

    logger.info(
        "estimated trodes to camera lag: "
        f"{camera_to_mcu_lag / NANOSECONDS_PER_SECOND:0.3f} s"
    )
    return camera_to_mcu_lag


def remove_acquisition_timing_pause_non_ptp(
    dio_systime: np.ndarray,
    frame_count: np.ndarray,
    camera_systime: np.ndarray,
    is_valid_camera_time: np.ndarray,
    pause_mid_time: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes data points occurring before a detected acquisition timing pause.
    Used in non-PTP alignment. Operates on timestamps in seconds.

    Parameters
    ----------
    dio_systime : np.ndarray
        Digital I/O system time (seconds).
    frame_count : np.ndarray
        Frame count corresponding to `camera_systime` entries.
    camera_systime : np.ndarray
        Camera system time (seconds) corresponding to valid position frames.
    is_valid_camera_time : np.ndarray
        Boolean array indicating which *original* position frames correspond to `camera_systime`.
    pause_mid_time : float
        Midpoint time of the pause (seconds). Data before this time is removed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Filtered versions of the input arrays:
        - dio_systime : np.ndarray (seconds)
        - frame_count : np.ndarray (frames)
        - is_valid_camera_time : np.ndarray (boolean mask relative to original position data)
        - camera_systime : np.ndarray (seconds)
    """
    dio_systime = dio_systime[dio_systime > pause_mid_time]
    frame_count = frame_count[is_valid_camera_time][camera_systime > pause_mid_time]
    is_valid_camera_time[is_valid_camera_time] = camera_systime > pause_mid_time
    camera_systime = camera_systime[camera_systime > pause_mid_time]

    return dio_systime, frame_count, is_valid_camera_time, camera_systime


def correct_timestamps_for_camera_to_mcu_lag(
    frame_count: np.ndarray,  # units = frames
    camera_systime: np.ndarray,  # units = nanoseconds
    camera_to_mcu_lag: float,  # units = nanoseconds
) -> np.ndarray:  # returns units = nanoseconds
    """
    Corrects camera system timestamps using linear regression against frame counts,
    accounting for estimated lag relative to DIO triggers.

    Parameters
    ----------
    frame_count : np.ndarray
        Hardware frame counts corresponding to `camera_systime`.
    camera_systime : np.ndarray
        Estimated camera system timestamps (nanoseconds) before final correction.
    camera_to_mcu_lag : float
        Estimated lag (nanoseconds) between camera frame time and DIO trigger time.

    Returns
    -------
    np.ndarray
        Corrected camera system timestamps (nanoseconds) based on linear fit.
    """
    regression_result = linregress(frame_count, camera_systime - camera_to_mcu_lag)
    corrected_camera_systime = (
        regression_result.intercept + frame_count * regression_result.slope
    )

    return corrected_camera_systime


def find_camera_dio_channel(nwb_file: NWBFile) -> np.ndarray:
    """
    Finds the timestamp data for the camera DIO channel within an NWB file's
    behavioral events. Assumes a single channel name contains "camera ticks".

    Parameters
    ----------
    nwb_file : NWBFile
        The NWBFile object to search within.

    Returns
    -------
    np.ndarray
        The timestamps (in seconds) of the camera DIO channel.

    Raises
    ------
    ValueError
        If zero or multiple DIO channels containing "camera ticks" are found,
        or if the necessary processing modules/interfaces are missing.
    KeyError
        If 'behavior' processing module or 'behavioral_events' interface is missing.
    """
    try:
        behavior_module = nwb_file.processing["behavior"]
        behavioral_events = behavior_module.data_interfaces["behavioral_events"]
    except KeyError as e:
        raise KeyError(
            f"Missing required NWB structure: {e}. Ensure 'behavior' module and 'behavioral_events' interface exist."
        ) from e

    dio_camera_name = [
        key
        for key in nwb_file.processing["behavior"]
        .data_interfaces["behavioral_events"]
        .time_series
        if "camera ticks" in key
    ]
    if len(dio_camera_name) > 1:
        raise ValueError(
            f"Multiple camera DIO channels found by name ('camera ticks'): {dio_camera_name}. "
            "Processing supports only one such channel for non-PTP alignment."
        )

    if len(dio_camera_name) == 0:
        raise ValueError(
            "No camera DIO channel found by name containing 'camera ticks'. "
            "Check channel names in NWB file or metadata YAML. Required for non-PTP alignment."
        )

    return (
        nwb_file.processing["behavior"]
        .data_interfaces["behavioral_events"]
        .time_series[dio_camera_name[0]]
        .timestamps
    )


def get_video_timestamps(video_timestamps_filepath: Path) -> np.ndarray:
    """
    Reads hardware timestamps from a .cameraHWSync file and returns them in seconds.

    Parameters
    ----------
    video_timestamps_filepath : Path
        Path to the .cameraHWSync file.

    Returns
    -------
    np.ndarray
        An array of video timestamps in seconds. Returns empty array if file reading fails.

    Raises
    ------
    KeyError
        If the expected 'HWTimestamp' field is missing in the data file.
    """
    video_timestamps = read_trodes_datafile(video_timestamps_filepath)["data"]
    try:
        return (
            np.squeeze(video_timestamps["HWTimestamp"]).astype(np.float64)
            / NANOSECONDS_PER_SECOND
        )
    except KeyError:
        raise KeyError(
            "'HWTimestamp' field missing in the data file. Ensure the file is formatted correctly."
        )


def _get_position_timestamps_ptp(
    video_timestamps: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """
    Processes video timestamps using PTP logic. Sets index to PTP time (seconds)
    and removes initial frames before the timing pause.

    Parameters
    ----------
    video_timestamps_df : pd.DataFrame
        DataFrame containing 'HWTimestamp' (nanoseconds) and other columns like 'HWframeCount'.
        Index is typically Trodes time (sample count).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame indexed by PTP time in seconds, with initial pause frames removed.

    Raises
    ------
    KeyError
        If 'HWTimestamp' column is missing from the input DataFrame.
    """
    if "HWTimestamp" not in video_timestamps.columns:
        raise KeyError(
            "'HWTimestamp' column missing from input DataFrame for PTP processing."
        )

    ptp_systime = np.asarray(video_timestamps.HWTimestamp)
    # Convert from integer nanoseconds to float seconds
    ptp_timestamps = pd.Index(ptp_systime / NANOSECONDS_PER_SECOND, name="time")
    # Check that the PTP timestamps correspond to a time later than 2000, log a warning if not
    if datetime.datetime.fromtimestamp(ptp_timestamps[0]).year < 2000:
        logger.warning(
            "PTP timestamps correspond to a time earlier than 2000. This may be due to a PTP clock reset."
        )

    video_timestamps = video_timestamps.drop(
        columns=["HWframeCount", "HWTimestamp"]
    ).set_index(ptp_timestamps)

    # Ignore positions before the timing pause.
    pause_mid_ind = (
        np.nonzero(
            np.logical_and(
                np.diff(video_timestamps.index[:100]) > DEFAULT_MIN_PTP_PAUSE_S,
                np.diff(video_timestamps.index[:100]) < DEFAULT_MAX_PTP_PAUSE_S,
            )
        )[0][0]
        + 1
    )
    video_timestamps = video_timestamps.iloc[pause_mid_ind:]
    if len(video_timestamps.index) > 1:
        frame_rate = 1 / np.median(np.diff(video_timestamps.index))
        logger.info(
            f"Camera frame rate estimated from PTP timestamps (after pause removal):"
            f" {frame_rate:.1f} frames/s"
        )
    else:
        logger.warning(
            "Less than 2 timestamps remain after PTP pause removal; cannot estimate frame rate."
        )

    return video_timestamps


def _get_position_timestamps_no_ptp(
    rec_dci_timestamps: np.ndarray,
    video_timestamps: pd.DataFrame,
    logger: logging.Logger,
    dio_camera_timestamps: np.ndarray,
    sample_count: np.ndarray,
    epoch_interval: list[float],
    non_repeat_timestamp_labels_id: np.ndarray,
) -> pd.DataFrame:
    """Processes video timestamps using non-PTP logic (alignment via DIO triggers).

    Parameters
    ----------
    rec_dci_timestamps : np.ndarray
        System clock times from the rec file used for non-PTP data.
    video_timestamps : pd.DataFrame
        DataFrame containing 'HWTimestamp' (nanoseconds) and other columns like 'HWframeCount'.
    logger : logging.Logger
        Logger instance.
    dio_camera_timestamps : np.ndarray
        Timestamps of the dio camera ticks used for non-PTP data.
    sample_count : np.ndarray
        Trodes timestamps from the rec file used for non-PTP data.
    epoch_interval : list[float]
        The time interval for the epoch used for non-PTP data.
    non_repeat_timestamp_labels_id : np.ndarray
        Array of unique IDs for non-repeating timestamps.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame indexed by camera system time in seconds, with initial pause frames removed.

    Raises
    ------
    KeyError
        If 'HWTimestamp' column is missing from the input DataFrame.
    IndexError
        If no valid timing pause is found within the search window.
    ValueError
        If the length of dio_camera_timestamps does not match the length of video_timestamps.
    """
    try:
        # dio_camera_timestamps are in units of seconds
        dio_camera_timestamps_ns = dio_camera_timestamps * NANOSECONDS_PER_SECOND
        pause_mid_time = (
            find_acquisition_timing_pause(dio_camera_timestamps_ns)
            / NANOSECONDS_PER_SECOND  # convert to seconds
        )
        frame_rate_from_dio = get_framerate(
            dio_camera_timestamps_ns[dio_camera_timestamps > pause_mid_time]
        )
        logger.info(
            "Camera frame rate estimated from DIO camera ticks:"
            f" {frame_rate_from_dio:0.1f} frames/s"
        )
    except IndexError:
        pause_mid_time = -1

    frame_count = np.asarray(video_timestamps.HWframeCount)

    epoch_start_ind = np.digitize(epoch_interval[0], rec_dci_timestamps)
    epoch_end_ind = np.digitize(epoch_interval[1], rec_dci_timestamps)
    is_valid_camera_time = np.isin(
        video_timestamps.index, sample_count[epoch_start_ind:epoch_end_ind]
    )

    camera_systime = rec_dci_timestamps[
        wrapped_digitize(
            video_timestamps.index[is_valid_camera_time],
            sample_count[epoch_start_ind:epoch_end_ind],
        )
        + epoch_start_ind
    ]
    (
        dio_camera_timestamps,
        frame_count,
        is_valid_camera_time,
        camera_systime,
    ) = remove_acquisition_timing_pause_non_ptp(
        dio_camera_timestamps,
        frame_count,
        camera_systime,
        is_valid_camera_time,
        pause_mid_time,
    )
    video_timestamps = video_timestamps.iloc[is_valid_camera_time]
    frame_rate_from_camera_systime = get_framerate(camera_systime)
    logger.info(
        "Camera frame rate estimated from camera sys time:"
        f" {frame_rate_from_camera_systime:0.1f} frames/s"
    )
    camera_to_mcu_lag = estimate_camera_to_mcu_lag(
        camera_systime, dio_camera_timestamps, len(non_repeat_timestamp_labels_id)
    )
    corrected_camera_systime = []
    for id in non_repeat_timestamp_labels_id:
        is_chunk = video_timestamps.non_repeat_timestamp_labels == id
        corrected_camera_systime.append(
            correct_timestamps_for_camera_to_mcu_lag(
                frame_count[is_chunk],
                camera_systime[is_chunk],
                camera_to_mcu_lag,
            )
        )
    corrected_camera_systime = np.concatenate(corrected_camera_systime)

    video_timestamps = video_timestamps.set_index(
        pd.Index(corrected_camera_systime, name="time")
    )
    return video_timestamps.groupby(
        video_timestamps.index
    ).first()  # TODO: Figure out why duplicate timesteps make it to this point and why this line is necessary


def get_position_timestamps(
    position_timestamps_filepath: Path,
    position_tracking_filepath: Optional[Path] = None,
    rec_dci_timestamps: Optional[np.ndarray] = None,  # units = seconds
    dio_camera_timestamps: Optional[np.ndarray] = None,  # units = seconds
    sample_count: Optional[np.ndarray] = None,  # units = samples
    ptp_enabled: bool = True,
    epoch_interval: Optional[List[float]] = None,  # units = seconds
):
    """Get the timestamps for a position data file. Includes protocols for both ptp and non-ptp data.

    Parameters
    ----------
    position_timestamps_filepath : Path
        path to the position timestamps file
    position_tracking_filepath : Path, optional
        path to the position tracking file, by default None
    rec_dci_timestamps : np.ndarray, optional
        system clock times from the rec file used for non-ptp data, by default None
    dio_camera_timestamps : np.ndarray, optional
        Timestamps of the dio camera ticks used for non-ptp data, by default None
    sample_count : np.ndarray, optional
        trodes timestamps from the rec file used for non-ptp data, by default None
    ptp_enabled : bool, optional
        whether ptp was enabled for position tracking, by default True
    epoch_interval : list[float] optional
        the timeinterval for the epoch used for non-ptp data, by default None

    Returns
    -------
    np.ndarray
        timestamps for the position data
    """
    logger = logging.getLogger("convert")

    # Get video timestamps
    datafile = read_trodes_datafile(position_timestamps_filepath)
    video_timestamps = (
        convert_datafile_to_pandas(datafile)
        .set_index("PosTimestamp")
        .rename(columns={"frameCount": "HWframeCount"})
    )

    # On AVT cameras, HWFrame counts wraps to 0 above this value.
    video_timestamps["HWframeCount"] = np.unwrap(
        video_timestamps["HWframeCount"].astype(np.int32),
        period=np.iinfo(np.uint16).max,
    )
    # Keep track of video frames
    video_timestamps["video_frame_ind"] = np.arange(len(video_timestamps))

    # Disconnects manifest as repeats in the trodes time index
    (
        non_repeat_timestamp_labels,
        non_repeat_timestamp_labels_id,
    ) = detect_trodes_time_repeats_or_frame_jumps(
        video_timestamps.index, video_timestamps.HWframeCount
    )
    logger.info(f"\tnon_repeat_timestamp_labels = {non_repeat_timestamp_labels_id}")
    video_timestamps["non_repeat_timestamp_labels"] = non_repeat_timestamp_labels
    video_timestamps = video_timestamps.loc[
        video_timestamps.non_repeat_timestamp_labels > 0
    ]

    # Get position tracking information
    try:
        position_tracking = pd.DataFrame(
            convert_datafile_to_pandas(read_trodes_datafile(position_tracking_filepath))
        ).set_index("time")
        is_repeat_timestamp = detect_repeat_timestamps(position_tracking.index)
        position_tracking = position_tracking.iloc[~is_repeat_timestamp]

        # Match the camera frames to the position tracking
        # Number of video frames can be different from online tracking because
        # online tracking can be started or stopped before video is stopped.
        # Additionally, for offline tracking, frames can be skipped if the
        # frame is labeled as bad.
        video_timestamps = pd.merge(
            video_timestamps,
            position_tracking,
            right_index=True,
            left_index=True,
            how="left",
        )
    except (FileNotFoundError, TypeError):
        pass

    if ptp_enabled:
        return _get_position_timestamps_ptp(video_timestamps, logger)
    else:
        return _get_position_timestamps_no_ptp(
            rec_dci_timestamps,
            video_timestamps,
            logger,
            dio_camera_timestamps,
            sample_count,
            epoch_interval,
            non_repeat_timestamp_labels_id,
        )


def find_camera_dio_channel_per_epoch(
    nwb_file: NWBFile, epoch_start: float, epoch_end: float
):
    """
    Find the camera DIO channel timestamps relevant for a given epoch.
    Searches through DIO channels with "camera ticks" in the name.
    Selects first one with at least 100 ticks within the epoch interval.

    Parameters
    ----------
    nwb_file : NWBFile
        The NWBFile to find the DIO channel in.
    epoch_start : float
        Timestamp (seconds) of the start of the epoch.
    epoch_end : float
        Timestamp (seconds) of the end of the epoch.

    Returns
    -------
    dio_camera_timestamps : np.ndarray
        The DIO timestamps (seconds) for the selected camera channel, restricted to the epoch.

    Raises
    ------
    ValueError
        If no suitable camera DIO channel is found (either missing, multiple with insufficient ticks, or none with "camera ticks").
    KeyError
        If the 'behavior' processing module or 'behavioral_events' interface is missing.
    """
    dio_camera_list = [
        key
        for key in nwb_file.processing["behavior"]["behavioral_events"].time_series
        if "camera ticks" in key
    ]
    if not dio_camera_list:
        raise ValueError(
            "No camera dio channel found by name. Check metadata YAML. Name must contain 'camera ticks'"
        )
    for camera in dio_camera_list:
        dio_camera_timestamps = (
            nwb_file.processing["behavior"]["behavioral_events"]
            .time_series[camera]
            .timestamps
        )
        epoch_ind = np.logical_and(
            dio_camera_timestamps >= epoch_start, dio_camera_timestamps <= epoch_end
        )
        if np.sum(epoch_ind) > 100:
            return dio_camera_timestamps[epoch_ind]
    raise ValueError("No camera dio has sufficient ticks for this epoch")


def add_position(
    nwb_file: NWBFile,
    metadata: dict,
    session_df: pd.DataFrame,
    ptp_enabled: bool = True,
    rec_dci_timestamps: Optional[np.ndarray] = None,
    sample_count: Optional[np.ndarray] = None,
) -> None:
    """Add position data to an NWBFile.

    Parameters
    ----------
    nwb_file : NWBFile
        The NWBFile to add the position data to.
    metadata : dict
        Metadata about the experiment.
    session_df : pd.DataFrame
        A DataFrame containing information about the session.
    ptp_enabled : bool, optional
        Whether PTP was enabled, by default True.
    rec_dci_timestamps : np.ndarray, optional
        The recording timestamps, by default None. Only used if ptp not enabled.
    sample_count : np.ndarray, optional
        The trodes sample count, by default None. Only used if ptp not enabled.
    """
    logger = logging.getLogger("convert")

    LED_POS_NAMES = [
        [
            "xloc",
            "yloc",
        ],  # led 0
        [
            "xloc2",
            "yloc2",
        ],
    ]  # led 1

    camera_id_to_meters_per_pixel = {
        camera["id"]: camera["meters_per_pixel"] for camera in metadata["cameras"]
    }

    df = []
    for task in metadata["tasks"]:
        df.append(
            pd.DataFrame(
                [(task["camera_id"], epoch) for epoch in task["task_epochs"]],
                columns=["camera_id", "epoch"],
            )
        )

    epoch_to_camera_ids = pd.concat(df).set_index("epoch").sort_index()

    # Make a processing module for behavior and add to the nwbfile
    if not "behavior" in nwb_file.processing:
        nwb_file.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )

    if "position" not in nwb_file.processing["behavior"].data_interfaces:
        position = Position(name="position")
        nwb_file.processing["behavior"].add(position)
    else:
        position = nwb_file.processing["behavior"]["position"]

    # get epoch data to seperate dio timestamps into epochs
    if (not ptp_enabled) and (not len(nwb_file.epochs)):
        raise ValueError(
            "add_epochs() must be run before add_position() for non-ptp data"
        )
    if not ptp_enabled:
        epoch_df = nwb_file.epochs.to_dataframe()

    for epoch in session_df.epoch.unique():
        try:
            position_tracking_filepath = session_df.loc[
                np.logical_and(
                    session_df.epoch == epoch,
                    session_df.file_extension == ".videoPositionTracking",
                )
            ].full_path.to_list()[0]
            # find the matching hw timestamps filepath
            video_index = position_tracking_filepath.split(".")[-2]
            video_hw_df = session_df.loc[
                np.logical_and(
                    session_df.epoch == epoch,
                    session_df.file_extension == ".cameraHWSync",
                )
            ]
            position_timestamps_filepath = video_hw_df[
                [
                    full_path.split(".")[-3] == video_index
                    for full_path in video_hw_df.full_path
                ]
            ].full_path.to_list()[0]

        except IndexError:
            logging.warning(f"No position tracking data found for epoch {epoch}")
            continue

        logger.info(epoch)
        logger.info(f"\tposition_timestamps_filepath: {position_timestamps_filepath}")
        logger.info(f"\tposition_tracking_filepath: {position_tracking_filepath}")

        # restrict dio camera timestamps to the current epoch
        if not ptp_enabled:
            epoch_start = epoch_df[epoch_df.index == epoch - 1]["start_time"].iloc[0]
            epoch_end = epoch_df[epoch_df.index == epoch - 1]["stop_time"].iloc[0]
            dio_camera_timestamps_epoch = find_camera_dio_channel_per_epoch(
                nwb_file=nwb_file, epoch_start=epoch_start, epoch_end=epoch_end
            )
        else:
            dio_camera_timestamps_epoch = None

        position_df = get_position_timestamps(
            position_timestamps_filepath,
            position_tracking_filepath,
            ptp_enabled=ptp_enabled,
            rec_dci_timestamps=rec_dci_timestamps,
            dio_camera_timestamps=dio_camera_timestamps_epoch,
            sample_count=sample_count,
            epoch_interval=[epoch_start, epoch_end] if not ptp_enabled else None,
        )

        # TODO: Doesn't handle multiple cameras currently
        camera_id = epoch_to_camera_ids.loc[epoch].camera_id[0]
        meters_per_pixel = camera_id_to_meters_per_pixel[camera_id]

        if position_tracking_filepath is not None:
            for led_number, valid_keys in enumerate(LED_POS_NAMES):
                key_set = [
                    key for key in position_df.columns.tolist() if key in valid_keys
                ]
                if len(key_set) > 0:
                    position.create_spatial_series(
                        name=f"led_{led_number}_series_{epoch}",
                        description=", ".join(["xloc", "yloc"]),
                        data=np.asarray(position_df[key_set]),
                        conversion=meters_per_pixel,
                        reference_frame="Upper left corner of video frame",
                        timestamps=np.asarray(position_df.index),
                    )
        else:
            logging.warning(f"No position tracking data found for epoch {epoch}")

        # add the video frame index as a new processing module
        if "position_frame_index" not in nwb_file.processing:
            nwb_file.create_processing_module(
                name="position_frame_index",
                description="stores video frame index for each position timestep",
            )
        # add timeseries for each frame index set (once per series because led's share timestamps)
        nwb_file.processing["position_frame_index"].add(
            TimeSeries(
                name=f"series_{epoch}",
                data=np.asarray(position_df["video_frame_ind"]),
                unit="N/A",
                timestamps=np.asarray(position_df.index),
            )
        )
        # add the video non-repeat timestamp labels as a new processing module
        if "non_repeat_timestamp_labels" not in nwb_file.processing:
            nwb_file.create_processing_module(
                name="non_repeat_timestamp_labels",
                description="stores non_repeat_labels for each position timestep",
            )
        # add timeseries for each non-repeat timestamp labels set (once per series because led's share timestamps)
        nwb_file.processing["non_repeat_timestamp_labels"].add(
            TimeSeries(
                name=f"series_{epoch}",
                data=np.asarray(position_df["non_repeat_timestamp_labels"]),
                unit="N/A",
                timestamps=np.asarray(position_df.index),
            )
        )


def convert_h264_to_mp4(file: str, video_directory: str) -> str:
    """
    Converts an H.264 file to MP4 format using ffmpeg.
    Assumes ffmpeg is installed and accessible in the system path.

    Parameters
    ----------
    input_file : str
        The path to the input H.264 file.
    video_directory : str
        The directory where the output MP4 file should be saved.

    Returns
    -------
    str
        The path to the output MP4 file.

    Raises
    ------
    subprocess.CalledProcessError
        If the ffmpeg command fails.
    OSError
        If the video directory cannot be created.
    """
    new_file_name = Path(video_directory) / Path(file.replace(".h264", ".mp4")).name

    logger = logging.getLogger("convert")
    if new_file_name.exists():
        logger.info(f"Video file {new_file_name} already exists. Skipping conversion.")
        return str(new_file_name)
    else:
        new_file_name = str(new_file_name)

    try:
        # Construct the ffmpeg command
        subprocess.run(f"ffmpeg -i {file} {new_file_name}", shell=True)
        logger.info(
            f"Video conversion completed. {file} has been converted to {new_file_name}"
        )
        return new_file_name
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Video conversion FAILED. {file} has NOT been converted to {new_file_name}"
        )
        raise e


def copy_video_to_directory(file: str, video_directory: str) -> str:
    """
    Copies a video file to the specified directory using system's copy command.

    Parameters
    ----------
    input_file : str
        The path to the input video file.
    video_directory : str
        The directory where the video file should be copied.

    Returns
    -------
    str
        The path to the copied video file in the target directory.

    Raises
    ------
    subprocess.CalledProcessError
        If the copy command fails.
    OSError
        If the video directory cannot be created.
    """
    new_file_name = Path(video_directory) / Path(file).name
    logger = logging.getLogger("convert")
    if new_file_name.exists():
        return str(new_file_name)
    else:
        new_file_name = str(new_file_name)

    try:
        # Construct the ffmpeg command
        subprocess.run(f"cp {file} {new_file_name}", shell=True)
        logger.info(f"Video copy completed. {file} has been copied to {new_file_name}")
        return new_file_name
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Video copy FAILED. {file} has NOT been copied to {new_file_name}"
        )
        raise e


def add_associated_video_files(
    nwb_file: NWBFile,
    metadata: Dict[str, Any],
    session_df: pd.DataFrame,
    video_directory: str,
    convert_video: bool = False,
) -> None:
    """
    Adds associated video files mentioned in metadata as ImageSeries to the NWB file.
    Copies or converts video files to the specified video directory.

    Parameters
    ----------
    nwb_file : NWBFile
        The NWBFile object to add video data to.
    metadata : dict
        Metadata dictionary containing 'associated_video_files' list.
        Each item in the list should be a dict with 'name', 'camera_id', 'task_epochs'.
    session_df : pd.DataFrame
        DataFrame linking files to epochs, used to find video file paths and HWSync paths.
    video_directory : str
        Directory where video files will be copied or converted to.
    convert_video : bool, optional
        If True, convert H.264 videos to MP4 using ffmpeg.
        If False (default), copy original video files.

    Raises
    ------
    FileNotFoundError
        If a video file cannot be found in session_df.
    ValueError
        If no cameraHWSync file is found for a given epoch.
    """
    # make processing module for video files
    nwb_file.create_processing_module(
        name="video_files", description="Contains all associated video files data"
    )
    # make a behavioral Event object to hold videos
    video = BehavioralEvents(name="video")
    # add the video file data
    for video_metadata in metadata["associated_video_files"]:
        epoch = video_metadata["task_epochs"][0]
        # get the video file path
        video_path = None
        for file in session_df[
            np.logical_or(
                session_df.file_extension == ".h264",
                session_df.file_extension == ".mp4",
            )
        ].full_path:
            if video_metadata["name"].rsplit(".", 1)[0] in file:
                video_path = file
                break
        if video_path is None:
            raise FileNotFoundError(
                f"Could not find video file {video_metadata['name']} in session_df"
            )

        # get timestamps for this video
        # find the matching hw timestamps filepath
        video_index = video_path.split(".")[-2]
        video_hw_df = session_df.loc[
            np.logical_and(
                session_df.epoch == epoch,
                session_df.file_extension == ".cameraHWSync",
            )
        ]
        if not len(video_hw_df):
            raise ValueError(
                f"No cameraHWSync found for epoch {epoch}, video {video_index} in session_df"
            )
        video_timestamps_filepath = video_hw_df[
            [
                full_path.split(".")[-3] == video_index
                for full_path in video_hw_df.full_path
            ]
        ].full_path.to_list()[0]
        # get the timestamps
        video_timestamps = np.squeeze(get_video_timestamps(video_timestamps_filepath))

        if convert_video:
            video_file_name = convert_h264_to_mp4(video_path, video_directory)
        else:
            video_file_name = copy_video_to_directory(video_path, video_directory)

        video.add_timeseries(
            ImageSeries(
                device=nwb_file.devices[
                    "camera_device " + str(video_metadata["camera_id"])
                ],
                name=video_metadata["name"],
                timestamps=video_timestamps,
                external_file=[video_file_name.split("/")[-1]],
                format="external",
                starting_frame=[0],
                description="video of animal behavior from epoch",
            )
        )
    if video_metadata is None:
        raise KeyError(f"Missing video metadata for epoch {epoch}")

    nwb_file.processing["video_files"].add(video)
