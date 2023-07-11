from pathlib import Path

import pandas as pd


def _process_path(path: Path) -> tuple[str, str, str, str, str, str, str]:
    date, animal_name, epoch, tag = path.stem.split("_")
    tag = tag.split(".")
    tag_index = tag[1] if len(tag) > 1 else 1
    tag = tag[0]
    full_path = str(path.absolute())
    extension = path.suffix
    return date, animal_name, epoch, tag, tag_index, extension, full_path


def get_file_info(path: Path) -> pd.DataFrame:
    VALID_FILE_EXTENSIONS = [
        "rec",
        "videoPositionTracking",
        "h264",
        "cameraHWSync",
        "stateScriptLog",
    ]
    COLUMN_NAMES = [
        "date",
        "animal_name",
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
    )
