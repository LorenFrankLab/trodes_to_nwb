"""Scans a directory for Trodes-related data files based on naming conventions
and valid extensions. Organizes file paths into a pandas DataFrame grouped by
session information (date, animal, epoch).
"""

import logging
from pathlib import Path

import pandas as pd

VALID_FILE_EXTENSIONS = [
    "rec",  # binary file containing the ephys recording, accelerometer, gyroscope, magnetometer, DIO data, header
    "videoPositionTracking",  # trodes tracked position
    "h264",  # video file
    "mp4",  # video file
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
    none_result = (None, None, None, None, None, None, None)
    parts = path.stem.split("_")
    try:
        if path.suffix == ".yml":
            # {date}_{animal}_metadata.yml -- the animal name may itself contain
            # underscores, so take the first token as the date and everything
            # between it and the trailing "metadata" token as the animal.
            if len(parts) < 3:
                logger.info(f"Invalid file name: {path.stem}. Skipping...")
                return none_result
            date = int(parts[0])
            animal_name = "_".join(parts[1:-1])
            epoch = 1
            tag = "NA"
            tag_index = 1
        else:
            # {date}_{animal}_{epoch}_{tag}.{ext} -- the animal name may contain
            # underscores (so use the last two tokens for epoch/tag), and the tag
            # may carry a trailing ".{cameraN}" suffix.
            if len(parts) < 4:
                logger.info(f"Invalid file name: {path.stem}. Skipping...")
                return none_result
            date = int(parts[0])
            animal_name = "_".join(parts[1:-2])
            epoch = int(parts[-2])
            tag = parts[-1].split(".")
            tag_index = int(tag[1]) if len(tag) > 1 else 1
            tag = tag[0]
    except (ValueError, IndexError):
        # A non-integer date/epoch/tag_index (or otherwise unparseable name).
        # Return all-None so the row is dropped, rather than letting a string
        # date flow into the .astype(int) in get_file_info, which would raise
        # and abort the scan of the entire directory (see #170).
        logger.info(f"Invalid file name: {path.stem}. Skipping...")
        return none_result

    return (
        date,
        animal_name,
        epoch,
        tag,
        tag_index,
        path.suffix,
        str(path.absolute()),
    )


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
    logger = logging.getLogger("convert")
    COLUMN_NAMES = [
        "date",
        "animal",
        "epoch",
        "tag",
        "tag_index",
        "file_extension",
        "full_path",
    ]

    paths = [p for ext in VALID_FILE_EXTENSIONS for p in path.glob(f"**/*.{ext}")]
    file_info = pd.DataFrame(
        [_process_path(p) for p in paths], columns=COLUMN_NAMES
    )

    n_skipped = int(file_info["full_path"].isna().sum())
    if n_skipped:
        logger.warning(
            f"{n_skipped} file(s) did not match the expected naming convention "
            "'{date}_{animal}_{epoch}_{tag}.{ext}' and were skipped "
            "(see INFO logs for the specific filenames)."
        )

    return (
        file_info.sort_values(by=["date", "animal", "epoch", "tag_index"])
        .dropna(how="all")
        .astype({"date": int, "epoch": int, "tag_index": int})
    )
