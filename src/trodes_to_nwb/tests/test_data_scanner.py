"""Filename-parsing robustness tests for ``data_scanner`` (#170, #179).

A stray/misnamed file used to crash the scan of the whole directory. Per
@samuelbray32's review, parsing is now a strict token split: animal names may
not contain ``_`` (verified against the lab's real metadata filenames), and a
file that *looks like* a botched session recording (a session-data extension and
a leading YYYYMMDD date) aborts the scan loudly rather than being silently
skipped, while genuine non-session files (auxiliary configs, un-dated fixtures)
are ignored with a warning. These tests pin those behaviours. The scanner only
reads file *names*, so empty placeholder files are sufficient.
"""

import logging
from pathlib import Path

import pytest

from trodes_to_nwb.data_scanner import _process_path, get_file_info

NONE_RESULT = (None, None, None, None, None, None, None)


def _touch(directory: Path, name: str) -> None:
    (directory / name).touch()


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


def test_underscore_animal_name_in_rec_raises(tmp_path):
    # Animal names may not contain "_" (strict four-token parse). A .rec whose
    # name has an extra underscore looks like a session recording but does not
    # parse, so it aborts loudly rather than being silently dropped (#179 review).
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "20230622_my_rat_01_r1.rec")  # underscore in animal name
    with pytest.raises(ValueError, match="naming convention"):
        get_file_info(tmp_path)


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
    p.touch()
    assert _process_path(p) == NONE_RESULT


@pytest.mark.parametrize(
    "bad_name",
    [
        "20230622_metadata.yml",  # .yml with too few tokens (no animal)
        "20230622_01_a1.rec",  # data file with too few tokens (no animal/epoch/tag)
    ],
)
def test_too_few_tokens_are_skipped(bad_name, tmp_path):
    # The strict unpack rejects the wrong token count: "20230622_metadata.yml"
    # has two tokens (the .yml form needs three) and "20230622_01_a1.rec" has
    # three (a data file needs four), so both return all-None instead of
    # producing a bogus row.
    p = tmp_path / bad_name
    p.touch()
    assert _process_path(p) == NONE_RESULT


def test_date_prefixed_botched_session_file_raises(tmp_path):
    # Looks like a session recording (data extension + leading YYYYMMDD date) but
    # is malformed -- must abort rather than silently dropping the epoch (#179).
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "20260610sample_03_r2.rec")  # missing underscore after date
    with pytest.raises(ValueError, match="naming convention"):
        get_file_info(tmp_path)


def test_non_dated_strict_file_is_skipped_not_raised(tmp_path, caplog):
    # A session-data extension with no leading date is clearly not an attempted
    # session file (e.g. the behavior_only.rec fixture) -- skip with a warning,
    # do not abort.
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "behavior_only.rec")
    with caplog.at_level(logging.WARNING, logger="convert"):
        df = get_file_info(tmp_path)  # must not raise

    names = {Path(p).name for p in df.full_path}
    assert "behavior_only.rec" not in names
    assert any("ignored" in r.message.lower() for r in caplog.records)


def test_auxiliary_and_botched_yaml_are_skipped_not_raised(tmp_path):
    # yaml is a lenient extension (shared with probe/device configs), so a
    # non-conforming yml -- even a date-prefixed, botched *metadata* file -- is
    # skipped, not raised. A missing metadata file is caught later by the
    # "exactly one metadata file per session" check, not here.
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "tetrode_12.5.yml")  # probe config, not a session file
    _touch(tmp_path, "20230622_metadata.yml")  # date-prefixed but lenient ext

    df = get_file_info(tmp_path)  # must not raise

    names = {Path(p).name for p in df.full_path}
    assert names == {"20230622_sample_01_a1.rec", "20230622_sample_metadata.yml"}
