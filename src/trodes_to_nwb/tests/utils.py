"""Set the path to the bulk test data dir and copies the yaml/config files there"""

import os
from pathlib import Path
import shutil

import numpy as np
import numpy.typing as npt

yaml_path = Path(__file__).resolve().parent / "test_data"

data_path = os.environ.get("DOWNLOAD_DIR", None)
if data_path is not None:
    # running from the GitHub Action workflow
    data_path = Path(data_path)
    shutil.copytree(
        yaml_path,
        data_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*os.listdir(data_path)),  # ignore existing
    )
else:
    # running locally -- bulk test data is the same directory as the test yaml files
    data_path = yaml_path

del yaml_path

DEFAULT_SAMPLING_RATE = 30_000


def assert_ephys_match_with_epoch_boundary_masking(
    new_data: npt.NDArray[np.int16],
    old_data: npt.NDArray[np.int16],
    timestamps: npt.NDArray[np.float64],
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
) -> None:
    """Assert ephys data matches, tolerating rec_to_nwb zero-fill at epoch boundaries.

    The rec_to_nwb reference files write zero-valued elements near epoch
    boundaries when stitching multiple .rec files.  Rather than masking with
    an arbitrary percentage threshold, this function verifies three structural
    properties of every discrepancy:

    1. Every mismatched element has old_data == 0 (rec_to_nwb zero-fill, not a
       real data difference).
    2. All mismatch rows are clustered near a timestamp discontinuity (epoch
       boundary), not scattered throughout the recording.
    3. Mismatches affect < 1% of all elements (a sanity bound, though the
       structural checks above are the real guards).
    """
    # Find elements that differ between new and old
    mismatch_mask = new_data != old_data
    if mismatch_mask.size == 0 or not np.any(mismatch_mask):
        return  # empty or perfect match, nothing to check

    # --- Check 1: every mismatch must be where old_data is zero ---
    mismatch_but_nonzero = mismatch_mask & (old_data != 0)
    n_bad = np.count_nonzero(mismatch_but_nonzero)
    assert n_bad == 0, (
        f"Found {n_bad} mismatched elements where the reference value is non-zero "
        f"(not a rec_to_nwb zero-fill artifact)"
    )

    # --- Check 2: mismatch rows must cluster near epoch boundaries ---
    # Detect epoch boundaries from timestamp gaps > 10x the expected interval
    dt = np.diff(timestamps)
    expected_dt = 1.0 / sampling_rate
    boundary_indices = np.where(dt > 10 * expected_dt)[0]

    mismatch_rows = np.where(np.any(mismatch_mask, axis=1))[0]
    if len(mismatch_rows) > 0:
        assert len(boundary_indices) > 0, (
            f"Found {len(mismatch_rows)} rows with zero-fill mismatches but no epoch "
            f"boundaries detected in timestamps"
        )
        # For each mismatch row, find distance to nearest epoch boundary
        distances = np.min(
            np.abs(mismatch_rows[:, np.newaxis] - boundary_indices[np.newaxis, :]),
            axis=1,
        )
        max_distance = np.max(distances)
        # Artifact should be within 0.2 seconds of a boundary
        max_allowed = int(0.2 * sampling_rate)
        assert max_distance <= max_allowed, (
            f"Mismatch row is {max_distance} samples from nearest epoch boundary "
            f"(max allowed: {max_allowed}). Farthest mismatch rows: "
            f"{mismatch_rows[distances > max_allowed][:10]}"
        )

    # --- Check 3: sanity bound on total mismatch fraction ---
    mismatch_frac = np.count_nonzero(mismatch_mask) / mismatch_mask.size
    assert mismatch_frac < 0.01, (
        f"Mismatches affect {mismatch_frac:.4%} of elements (expected < 1%)"
    )
