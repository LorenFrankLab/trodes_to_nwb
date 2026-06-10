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

# Extensions whose files are produced only as per-session/per-epoch recordings
# and therefore must follow the naming convention. A file with one of these
# extensions whose name *looks like* a session file (leading YYYYMMDD date) but
# does not parse is a botched session file that would silently drop an epoch's
# data, so the scan aborts loudly (see #170 and @samuelbray32's review of #179).
# The remaining valid extensions (yml, trackgeometry) are shared with
# auxiliary/config files -- probe & device metadata yamls, fsgui track geometry
# -- that legitimately do not follow the convention, so unparseable files there
# are skipped with a warning instead of aborting.
SESSION_DATA_EXTENSIONS = {
    "rec",
    "videoPositionTracking",
    "h264",
    "mp4",
    "cameraHWSync",
    "stateScriptLog",
    "videoTimeStamps",
}

DATE_PREFIX_LENGTH = 8  # session names start with a YYYYMMDD date


def _looks_like_session_filename(stem: str) -> bool:
    """Whether a filename stem looks like an attempted session file.

    Session files are named ``{date}_{animal}_{epoch}_{tag}`` with ``date`` an
    8-digit ``YYYYMMDD``. We only require the leading 8 digits (not the trailing
    underscore) so a missing separator -- e.g. ``20260610sample_03_r2`` -- still
    counts as an attempted session file and is flagged rather than silently
    dropped.

    Parameters
    ----------
    stem : str
        Filename stem (name without the final extension).

    Returns
    -------
    bool
        True if the stem starts with an 8-digit date.
    """
    return len(stem) >= DATE_PREFIX_LENGTH and stem[:DATE_PREFIX_LENGTH].isdigit()


def _process_path(
    path: Path,
) -> tuple[
    int | None, str | None, int | None, str | None, int | None, str | None, str | None
]:
    """Process a file path into its components.

    Parameters
    ----------
    path : Path
        Filename to process

    Returns
    -------
    tuple
        ``(date, animal_name, epoch, tag, tag_index, extension, full_path)`` --
        ``date``/``epoch``/``tag_index`` are ``int``, ``tag``/``extension``/
        ``full_path`` are ``str``. All seven are ``None`` if the name does not
        match the convention.

    """
    none_result = (None, None, None, None, None, None, None)
    parts = path.stem.split("_")
    try:
        if path.suffix == ".yml":
            # {date}_{animal}_metadata.yml -- the animal name may itself contain
            # underscores, so take the first token as the date and everything
            # between it and the trailing "metadata" token as the animal.
            if len(parts) < 3:
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
                return none_result
            date = int(parts[0])
            animal_name = "_".join(parts[1:-2])
            epoch = int(parts[-2])
            tag = parts[-1].split(".")
            tag_index = int(tag[1]) if len(tag) > 1 else 1
            tag = tag[0]
    except (ValueError, IndexError):
        # A non-integer date/epoch/tag_index (or otherwise unparseable name).
        # Return all-None; get_file_info decides whether that is a botched
        # session file (raise) or an auxiliary file to skip (see #170 and the
        # convention check in get_file_info).
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

    Raises
    ------
    ValueError
        If a file looks like a session recording (a session-data extension and a
        leading YYYYMMDD date) but does not match the naming convention.
        Converting would silently drop that epoch's data, so the scan aborts and
        lists every offending file.

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

    parsed = []
    misnamed = []  # botched session files -> abort (would silently drop data)
    skipped = []  # auxiliary/non-session files -> warn and ignore
    for p in paths:
        row = _process_path(p)
        if row[0] is not None:
            parsed.append(row)
        elif p.suffix[1:] in SESSION_DATA_EXTENSIONS and _looks_like_session_filename(
            p.stem
        ):
            misnamed.append(p)
        else:
            skipped.append(p)

    if misnamed:
        listing = "\n".join(
            f"  - {p.name}" for p in sorted(misnamed, key=lambda x: x.name)
        )
        raise ValueError(
            f"{len(misnamed)} file(s) look like session recordings (a session-data "
            "extension and a leading YYYYMMDD date) but do not match the required "
            "naming convention '{date}_{animal}_{epoch}_{tag}.{ext}' (epoch a "
            "zero-padded integer). Converting would silently skip them and drop "
            "that recording/video/position data. Rename them to the convention, "
            f"or move them out of the data directory:\n{listing}"
        )

    if skipped:
        listing = ", ".join(sorted(p.name for p in skipped))
        logger.warning(
            f"{len(skipped)} file(s) did not match the session naming convention "
            f"and were ignored (not treated as session data): {listing}"
        )

    return (
        pd.DataFrame(parsed, columns=COLUMN_NAMES)
        .sort_values(by=["date", "animal", "epoch", "tag_index"])
        .dropna(how="all")
        .astype({"date": int, "epoch": int, "tag_index": int})
    )
