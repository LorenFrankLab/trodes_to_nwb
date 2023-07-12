from pathlib import Path

import pandas as pd
import numpy as np

VALID_FILE_EXTENSIONS = [
    "rec",  # binary file containing the ephys recording, accelerometer, gyroscope, magnetometer, DIO data, header
    "videoPositionTracking",  # trodes tracked position
    "h264",  # video file
    "cameraHWSync",  # position timestamps
    "stateScriptLog",  # state script controls the experimenter parameters
    "yaml",  # metadata file
    "videoTimeStamps",  # not used
    "track_geometry",  # used if using Trodes linearization
]


def _process_path(path: Path) -> tuple[str, str, str, str, str, str, str]:
    date, animal_name, epoch, tag = path.stem.split("_")
    tag = tag.split(".")
    tag_index = tag[1] if len(tag) > 1 else 1
    tag = tag[0]
    full_path = str(path.absolute())
    extension = path.suffix
    try:
        # check if date, epoch, and tag_index are integers
        int(date), int(epoch), int(tag_index)
    except ValueError:
        print(f"Invalid file name: {path.stem}")
    return date, animal_name, epoch, tag, tag_index, extension, full_path


def get_file_info(path: Path) -> pd.DataFrame:
    COLUMN_NAMES = [
        "date",
        "animal",
        "epoch",
        "tag",
        "tag_index",
        "file_extension",
        "full_path",
    ]

    return pd.concat(
        [
            pd.DataFrame(
                [_process_path(files) for files in path.glob(f"**/*.{ext}")],
                columns=COLUMN_NAMES,
            )
            for ext in VALID_FILE_EXTENSIONS
        ]
    ).sort_values(by=["date", "animal", "epoch", "tag_index"])
