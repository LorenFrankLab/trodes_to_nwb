from spikegadgets_to_nwb.data_scanner import get_file_info

import os
from pathlib import Path

path = os.path.dirname(os.path.abspath(__file__))


def test_get_file_info():
    try:
        # running on github
        data_path = Path(os.environ.get("DOWNLOAD_DIR"))
    except (TypeError, FileNotFoundError):
        # running locally
        data_path = Path(path)
    path_df = get_file_info(data_path)

    for file_type in [
        ".h264",
        ".stateScriptLog",
        ".cameraHWSync",
        ".videoTimeStamps",
        ".videoPositionTracking",
        ".rec",
    ]:
        assert len(path_df[path_df.file_extension == file_type]) == 2

    assert set(path_df.animal) == {"sample"}
    assert set(path_df.date) == {20230622}
    assert set(path_df.epoch) == {1, 2}
    assert (set(path_df.tag) == {"a1"}) or (
        set(path_df.tag) == {"a1", "NA"}
    )  # yamlfiles only added in local testing
    for file in path_df.full_path:
        assert Path(file).exists()
