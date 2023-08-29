import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import Position
from scipy.ndimage import label
from scipy.stats import linregress

from spikegadgets_to_nwb.convert_rec_header import detect_ptp_from_header

NANOSECONDS_PER_SECOND = 1e9


def parse_dtype(fieldstr: str) -> np.dtype:
    """Parses last fields parameter (<time uint32><...>) as a single string
    Assumes it is formatted as <name number * type> or <name type>
    Returns: np.dtype
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


def read_trodes_datafile(filename: Path) -> dict:
    """Read trodes binary.

    Parameters
    ----------
    filename : str

    Returns
    -------
    data_file : dict

    """
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


def get_framerate(timestamps: np.ndarray) -> float:
    """Frames per second"""
    timestamps = np.asarray(timestamps)
    return NANOSECONDS_PER_SECOND / np.median(np.diff(timestamps))


def find_acquisition_timing_pause(
    timestamps: np.ndarray,
    min_duration: float = 0.4,
    max_duration: float = 1.0,
    n_search: int = 100,
) -> float:
    """Landmark timing 'gap' (0.5 s pause in video stream) parameters

    Parameters
    ----------
    timestamps : int64
    min_duration : minimum duratino of gap (in seconds)
    max_duration : maximum duratino of gap (in seconds)
    n_search : search only the first `n_search` entries

    Returns
    -------
    pause_mid_time
        Midpoint time of timing pause

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
    """Want to avoid regressing over large frame count skips"""
    frame_count = np.asarray(frame_count)

    is_large_frame_jump = np.insert(np.diff(frame_count) > min_frame_jump, 0, False)

    print(f"big frame jumps: {np.nonzero(is_large_frame_jump)[0]}")

    return is_large_frame_jump


def detect_repeat_timestamps(timestamps: np.ndarray) -> np.ndarray:
    return np.insert(timestamps[:-1] >= timestamps[1:], 0, False)


def detect_trodes_time_repeats_or_frame_jumps(
    trodes_time: np.ndarray, frame_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """If a trodes time index repeats, then the Trodes clock has frozen
    due to headstage disconnects."""
    trodes_time = np.asarray(trodes_time)
    is_repeat_timestamp = detect_repeat_timestamps(trodes_time)
    print(f"repeat timestamps ind: {np.nonzero(is_repeat_timestamp)[0]}")

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
) -> tuple[np.ndarray, np.ndarray]:
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
    if n_breaks == 0:
        dio_systime = dio_systime[: len(camera_systime)]
        camera_to_mcu_lag = np.median(camera_systime - dio_systime)
    else:
        camera_to_mcu_lag = camera_systime[0] - dio_systime[0]

    print(
        "estimated trodes to camera lag: "
        f"{camera_to_mcu_lag / NANOSECONDS_PER_SECOND:0.3f} s"
    )
    return camera_to_mcu_lag


def remove_acquisition_timing_pause_non_ptp(
    dio_systime: np.ndarray,
    frame_count: np.ndarray,
    camera_systime: np.ndarray,
    is_valid_camera_time: np.ndarray,
    pause_mid_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dio_systime = dio_systime[dio_systime > pause_mid_time]
    frame_count = frame_count[is_valid_camera_time][camera_systime > pause_mid_time]
    is_valid_camera_time[is_valid_camera_time] = camera_systime > pause_mid_time
    camera_systime = camera_systime[camera_systime > pause_mid_time]

    return dio_systime, frame_count, is_valid_camera_time, camera_systime


def correct_timestamps_for_camera_to_mcu_lag(
    frame_count: np.ndarray, camera_systime: np.ndarray, camera_to_mcu_lag: np.ndarray
) -> np.ndarray:
    regression_result = linregress(frame_count, camera_systime - camera_to_mcu_lag)
    corrected_camera_systime = (
        regression_result.intercept + frame_count * regression_result.slope
    )
    corrected_camera_systime /= NANOSECONDS_PER_SECOND

    return corrected_camera_systime


def find_camera_dio_channel(dios):
    raise NotImplementedError


def get_position_timestamps(
    position_timestamps_filepath: Path,
    position_tracking_filepath=None | Path,
    mcu_neural_timestamps=None | np.ndarray,
    dios=None,
    ptp_enabled: bool = True,
):
    # Get video timestamps
    video_timestamps = (
        pd.DataFrame(read_trodes_datafile(position_timestamps_filepath)["data"])
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
    print(f"\tnon_repeat_timestamp_labels = {non_repeat_timestamp_labels_id}")
    video_timestamps["non_repeat_timestamp_labels"] = non_repeat_timestamp_labels
    video_timestamps = video_timestamps.loc[
        video_timestamps.non_repeat_timestamp_labels > 0
    ]

    # Get position tracking information
    try:
        position_tracking = pd.DataFrame(
            read_trodes_datafile(position_tracking_filepath)["data"]
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
        ptp_systime = np.asarray(video_timestamps.HWTimestamp)
        # Convert from integer nanoseconds to float seconds
        ptp_timestamps = pd.Index(ptp_systime / NANOSECONDS_PER_SECOND, name="time")
        video_timestamps = video_timestamps.drop(
            columns=["HWframeCount", "HWTimestamp"]
        ).set_index(ptp_timestamps)

        # Ignore positions before the timing pause.
        pause_mid_ind = (
            np.nonzero(
                np.logical_and(
                    np.diff(video_timestamps.index[:100]) > 0.4,
                    np.diff(video_timestamps.index[:100]) < 2.0,
                )
            )[0][0]
            + 1
        )
        original_video_timestamps = video_timestamps.copy()
        video_timestamps = video_timestamps.iloc[pause_mid_ind:]
        print(
            "Camera frame rate estimated from MCU timestamps:"
            f" {1 / np.median(np.diff(video_timestamps.index)):0.1f} frames/s"
        )
        return video_timestamps, original_video_timestamps
    else:
        dio_camera_ticks = find_camera_dio_channel(dios)
        is_valid_tick = np.isin(dio_camera_ticks, mcu_neural_timestamps.index)
        dio_systime = np.asarray(
            mcu_neural_timestamps.loc[dio_camera_ticks[is_valid_tick]]
        )
        # The DIOs and camera frames are initially unaligned. There is a
        # half second pause at the start to allow for alignment.
        pause_mid_time = find_acquisition_timing_pause(dio_systime)

        # Estimate the frame rate from the DIO camera ticks as a sanity check.
        frame_rate_from_dio = get_framerate(dio_systime[dio_systime > pause_mid_time])
        print(
            "Camera frame rate estimated from DIO camera ticks:"
            f" {frame_rate_from_dio:0.1f} frames/s"
        )
        frame_count = np.asarray(video_timestamps.HWframeCount)

        camera_systime, is_valid_camera_time = estimate_camera_time_from_mcu_time(
            video_timestamps, mcu_neural_timestamps
        )
        (
            dio_systime,
            frame_count,
            is_valid_camera_time,
            camera_systime,
        ) = remove_acquisition_timing_pause_non_ptp(
            dio_systime,
            frame_count,
            camera_systime,
            is_valid_camera_time,
            pause_mid_time,
        )
        original_video_timestamps = video_timestamps.copy()
        video_timestamps = video_timestamps.iloc[is_valid_camera_time]

        frame_rate_from_camera_systime = get_framerate(camera_systime)
        print(
            "Camera frame rate estimated from MCU timestamps:"
            f" {frame_rate_from_camera_systime:0.1f} frames/s"
        )

        camera_to_mcu_lag = estimate_camera_to_mcu_lag(
            camera_systime, dio_systime, len(non_repeat_timestamp_labels_id)
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
        video_timestamps.iloc[
            is_valid_camera_time
        ].index = corrected_camera_systime.index
        return (
            video_timestamps.set_index(pd.Index(corrected_camera_systime, name="time")),
            original_video_timestamps,
        )


def add_position(
    nwb_file: NWBFile,
    metadata: dict,
    session_df: pd.DataFrame,
    rec_header: ElementTree.ElementTree,
):
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

    position = Position(name="position")
    ptp_enabled = detect_ptp_from_header(rec_header)

    for epoch in session_df.epoch.unique():
        position_timestamps_filepath = session_df.loc[
            np.logical_and(
                session_df.epoch == epoch,
                session_df.file_extension == ".cameraHWSync",
            )
        ].full_path.to_list()[0]

        try:
            position_tracking_filepath = session_df.loc[
                np.logical_and(
                    session_df.epoch == epoch,
                    session_df.file_extension == ".videoPositionTracking",
                )
            ].full_path.to_list()[0]
        except IndexError:
            position_tracking_filepath = None

        print(epoch)
        print(f"\tposition_timestamps_filepath: {position_timestamps_filepath}")
        print(f"\tposition_tracking_filepath: {position_tracking_filepath}")

        position_df, original_video_timestamps = get_position_timestamps(
            position_timestamps_filepath,
            position_tracking_filepath,
            ptp_enabled=ptp_enabled,
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
            position.create_spatial_series(
                name=f"led_{led_number}_series_{epoch}",
                description=", ".join(["xloc", "yloc"]),
                data=np.asarray([]),
                conversion=meters_per_pixel,
                reference_frame="Upper left corner of video frame",
                timestamps=np.asarray(position_df.index),
            )

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
    nwb_file.processing["behavior"].add(position)
