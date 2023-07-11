from pathlib import Path

import pandas as pd
import numpy as np

VALID_FILE_EXTENSIONS = [
    "rec",
    "videoPositionTracking",
    "h264",
    "cameraHWSync",
    "stateScriptLog",
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


def _all_files_in_ext(file_extensions: list[str]) -> bool:
    return all(
        [ext.replace(".", "") in VALID_FILE_EXTENSIONS for ext in file_extensions]
    )


def check_has_all_file_extensions(file_info: pd.DataFrame) -> bool:
    has_all_file_extensions = file_info.groupby(
        ["date", "animal", "epoch"]
    ).file_extension.apply(_all_files_in_ext)
    if not np.all(has_all_file_extensions):
        print("missing files")
        print(file_info[~has_all_file_extensions])
    else:
        print("all files present")

    return np.all(has_all_file_extensions)
