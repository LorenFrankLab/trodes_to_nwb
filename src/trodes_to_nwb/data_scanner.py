import logging
from pathlib import Path

import pandas as pd

VALID_FILE_EXTENSIONS = [
    "rec",  # binary file containing the ephys recording, accelerometer, gyroscope, magnetometer, DIO data, header
    "videoPositionTracking",  # trodes tracked position
    "h264",  # video file
    "cameraHWSync",  # position timestamps
    "stateScriptLog",  # state script controls the experimenter parameters
    "yml",  # metadata file
    "videoTimeStamps",  # not used
    "trackgeometry",  # used if using Trodes linearization
]


def _process_path(path: Path) -> tuple[str, str, str, str, str, str, str]:
    """Process a file path into its components

    Parameters
    ----------
    path : Path
        Filename to process

    Returns
    -------
    date : str
    animal_name : str
    epoch : str
    tag : str
    tag_index : str
    extension : str
    full_path : str

    """
    logger = logging.getLogger("convert")
    try:
        if path.suffix == ".yml":
            date, animal_name, _ = path.stem.split("_")
            epoch = 1
            tag = "NA"
            tag_index = 1

            try:
                # check if date is an integer
                date = int(date)
            except ValueError:
                logger.info(f"Invalid file name: {path.stem}. Skipping...")
                return None, None, None, None, None, None, None
        else:
            date, animal_name, epoch, tag = path.stem.split("_")
            tag = tag.split(".")
            tag_index = tag[1] if len(tag) > 1 else 1
            tag = tag[0]
            try:
                # check if date, epoch, and tag_index are integers
                date = int(date)
                epoch = int(epoch)
                tag_index = int(tag_index)
            except ValueError:
                logger.info(f"Invalid file name: {path.stem}. Skipping...")

        full_path = str(path.absolute())
        extension = path.suffix

        return date, animal_name, epoch, tag, tag_index, extension, full_path
    except ValueError:
        logger.info(f"Invalid file name: {path.stem}. Skipping...")
        return None, None, None, None, None, None, None


def get_file_info(path: Path) -> pd.DataFrame:
    """Get information about the files in a directory for grouping

    Parameters
    ----------
    path : Path
        Path to folder containing files

    Returns
    -------
    file_info : pd.DataFrame
        DataFrame containing information about the files in the folder

    """
    COLUMN_NAMES = [
        "date",
        "animal",
        "epoch",
        "tag",
        "tag_index",
        "file_extension",
        "full_path",
    ]

    return (
        pd.concat(
            [
                pd.DataFrame(
                    [_process_path(files) for files in path.glob(f"**/*.{ext}")],
                    columns=COLUMN_NAMES,
                )
                for ext in VALID_FILE_EXTENSIONS
            ]
        )
        .sort_values(by=["date", "animal", "epoch", "tag_index"])
        .dropna(how="all")
        .astype({"date": int, "epoch": int, "tag_index": int})
    )
