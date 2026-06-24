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

# The convention every session file follows, in one place so the parser and the
# user-facing error message describe the same thing.
SESSION_FILENAME_CONVENTION = "{date}_{animal}_{epoch}_{tag}.{ext}"

# Extensions produced only as per-session/per-epoch recordings: an unparseable
# file with one of these and a leading YYYYMMDD date is a botched session file
# that would silently drop an epoch's data, so the scan aborts loudly.
SESSION_DATA_EXTENSIONS = {
    "rec",
    "videoPositionTracking",
    "h264",
    "mp4",
    "cameraHWSync",
    "stateScriptLog",
    "videoTimeStamps",
}

# The remaining valid extensions are shared with auxiliary/config files (probe &
# device metadata yamls, fsgui track geometry) that legitimately do not follow
# the convention, so unparseable files there are skipped with a warning. The
# assert keeps the split exhaustive: adding an extension to VALID_FILE_EXTENSIONS
# without classifying it here fails at import rather than silently downgrading a
# botched session file from "abort" to "skip".
AUXILIARY_FILE_EXTENSIONS = {"yml", "trackgeometry"}
assert set(VALID_FILE_EXTENSIONS) == SESSION_DATA_EXTENSIONS | AUXILIARY_FILE_EXTENSIONS

DATE_PREFIX_LENGTH = 8  # session names start with a YYYYMMDD date


def _looks_like_session_filename(stem: str) -> bool:
    """Return whether a filename stem looks like an attempted session file.

    Session files are named ``{date}_{animal}_{epoch}_{tag}`` with ``date`` an
    8-digit ``YYYYMMDD``. Only the leading 8 digits are required (not the
    trailing underscore), so a missing separator -- e.g.
    ``20260610sample_03_r2`` -- still counts as an attempted session file and is
    flagged rather than silently dropped.

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
        File path to process.

    Returns
    -------
    date : int or None
    animal_name : str or None
    epoch : int or None
    tag : str or None
    tag_index : int or None
    extension : str or None
    full_path : str or None

    Notes
    -----
    The seven values are returned together: either all are populated, or all are
    ``None`` when the filename does not match the naming convention.

    """
    parts = path.stem.split("_")
    try:
        if path.suffix == ".yml":
            # {date}_{animal}_metadata.yml -- exactly three underscore-separated
            # tokens (animal names may not contain "_").
            date, animal_name, _ = parts
            date = int(date)
            epoch = 1
            tag = "NA"
            tag_index = 1
        else:
            # {date}_{animal}_{epoch}_{tag}.{ext} -- exactly four tokens (animal
            # names may not contain "_"); the tag may carry a ".{cameraN}" suffix.
            date, animal_name, epoch, tag = parts
            date = int(date)
            epoch = int(epoch)
            # The tag itself (e.g. "s1", "r3") is a string; only an optional
            # trailing ".{cameraN}" suffix becomes the integer tag_index.
            tag, *camera = tag.split(".")
            tag_index = int(camera[0]) if camera else 1
        if not animal_name.strip():
            # An empty animal token (e.g. "20230622__01_a1") has the right arity
            # and integer date/epoch, so the unpack succeeds -- but it is not a
            # real session. Treat it as unparseable so get_file_info raises it
            # (a session file) or skips it (auxiliary) rather than emitting a
            # phantom (date, "") session downstream.
            raise ValueError("empty animal name")
    except ValueError:
        # Wrong token count (the strict unpack), an empty animal name, or a
        # non-integer date/epoch/tag_index. Return all-None; get_file_info
        # decides whether that is a botched session file (raise) or an auxiliary
        # file to skip.
        return (None, None, None, None, None, None, None)

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
    """Get information about the files in a directory for grouping.

    Parameters
    ----------
    path : Path
        Path to the folder containing files.

    Returns
    -------
    file_info : pd.DataFrame
        DataFrame containing information about the files in the folder.

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

    # Walk the tree once and keep files whose final extension is recognised,
    # rather than globbing the whole tree once per extension.
    valid_extensions = set(VALID_FILE_EXTENSIONS)
    paths = [p for p in path.rglob("*") if p.suffix[1:] in valid_extensions]

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
            # Log each skipped file individually (in addition to the summary
            # warning below) so a dropped file can be traced after conversion.
            logger.info(
                f"Skipping '{p.name}': does not match the session naming convention."
            )
            skipped.append(p)

    if misnamed:
        listing = "\n".join(
            f"  - {p.name}" for p in sorted(misnamed, key=lambda x: x.name)
        )
        raise ValueError(
            f"{len(misnamed)} file(s) look like session recordings (a session-data "
            "extension and a leading YYYYMMDD date) but do not match the required "
            f"naming convention '{SESSION_FILENAME_CONVENTION}' (epoch an integer, "
            "conventionally zero-padded). Converting would silently skip them and "
            "drop that recording/video/position data. Rename them to the "
            f"convention, or move them out of the data directory:\n{listing}"
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
        .astype({"date": int, "epoch": int, "tag_index": int})
    )
