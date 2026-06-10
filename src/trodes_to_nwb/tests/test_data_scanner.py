"""Filename-parsing robustness tests for ``data_scanner`` (#170).

A stray/misnamed file used to crash the scan of the whole directory, and animal
names containing ``_`` were silently dropped. These tests pin both behaviours.
The scanner only reads file *names*, so empty placeholder files are sufficient.
"""

from pathlib import Path

import pytest

from trodes_to_nwb.data_scanner import _process_path, get_file_info

NONE_RESULT = (None, None, None, None, None, None, None)


def _touch(directory: Path, name: str) -> None:
    (directory / name).write_text("")


def test_one_misnamed_file_does_not_crash_the_scan(tmp_path):
    # a valid session ...
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    # ... plus a stray file that previously aborted get_file_info entirely
    _touch(tmp_path, "notadate_sample_01_a1.rec")

    df = get_file_info(tmp_path)  # must not raise

    names = {Path(p).name for p in df.full_path}
    assert names == {
        "20230622_sample_01_a1.rec",
        "20230622_sample_metadata.yml",
    }


def test_animal_name_with_underscore_is_not_dropped(tmp_path):
    _touch(tmp_path, "20230622_my_rat_01_r1.rec")
    _touch(tmp_path, "20230622_my_rat_metadata.yml")

    df = get_file_info(tmp_path)

    assert (df.animal == "my_rat").all()
    assert set(df.file_extension) == {".rec", ".yml"}
    rec = df[df.file_extension == ".rec"].iloc[0]
    assert rec.epoch == 1
    assert rec.tag == "r1"


def test_video_camera_suffix_parses(tmp_path):
    _touch(tmp_path, "20230622_sample_01_s1.1.h264")

    df = get_file_info(tmp_path)

    row = df.iloc[0]
    assert row.epoch == 1
    assert row.tag == "s1"
    assert row.tag_index == 1


@pytest.mark.parametrize(
    "bad_name",
    ["randomfile.rec", "foo_bar.rec", "notadate_sample_01_a1.rec", "x_y_z.yml"],
)
def test_unparseable_names_return_none(bad_name, tmp_path):
    p = tmp_path / bad_name
    p.write_text("")
    assert _process_path(p) == NONE_RESULT


@pytest.mark.parametrize(
    "bad_name",
    [
        "20230622_metadata.yml",  # .yml with too few tokens (no animal)
        "20230622_01_a1.rec",  # data file with too few tokens (no animal/epoch/tag)
    ],
)
def test_too_few_tokens_are_skipped(bad_name, tmp_path):
    # Without the length guards these would parse to an *empty* animal name with
    # no exception raised (all remaining tokens are integers), silently producing
    # a bogus row. The guards must reject them.
    p = tmp_path / bad_name
    p.write_text("")
    assert _process_path(p) == NONE_RESULT
