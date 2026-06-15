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
    # Use camera 2 (not 1): tag_index for camera 1 is indistinguishable from the
    # default, so a ".2" suffix is what proves the index is actually extracted.
    _touch(tmp_path, "20230622_sample_01_s1.2.h264")

    df = get_file_info(tmp_path)

    row = df.iloc[0]
    assert row.epoch == 1
    assert row.tag == "s1"
    assert row.tag_index == 2


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
        "20230622_metadata.yml",  # .yml: 2 tokens, needs 3 (too few)
        "20230622_01_a1.rec",  # data: 3 tokens, needs 4 (too few)
        "20230622_my_rat_metadata.yml",  # .yml: 4 tokens, needs 3 (too many)
        "20230622_my_rat_01_r1.rec",  # data: 5 tokens, needs 4 (too many)
    ],
)
def test_wrong_token_count_returns_none(bad_name, tmp_path):
    # The strict unpack rejects any wrong token count -- too few or too many --
    # so the file returns all-None instead of producing a bogus row.
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


def test_auxiliary_and_botched_yaml_are_skipped_not_raised(tmp_path, caplog):
    # yaml is a lenient extension (shared with probe/device configs), so a
    # non-conforming yml -- even a date-prefixed, botched *metadata* file -- is
    # skipped, not raised. A missing metadata file is caught later by the
    # "exactly one metadata file per session" check, not here.
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "tetrode_12.5.yml")  # probe config, not a session file
    _touch(tmp_path, "20230622_metadata.yml")  # date-prefixed but lenient ext

    with caplog.at_level(logging.WARNING, logger="convert"):
        df = get_file_info(tmp_path)  # must not raise

    names = {Path(p).name for p in df.full_path}
    assert names == {"20230622_sample_01_a1.rec", "20230622_sample_metadata.yml"}
    # The skipped yamls are reported in the warning, not dropped silently.
    warnings = " ".join(r.message for r in caplog.records)
    assert "tetrode_12.5.yml" in warnings
    assert "20230622_metadata.yml" in warnings


def test_empty_animal_name_is_rejected(tmp_path):
    # A double underscore ("20230622__01_a1") yields the right token count with
    # an empty animal, so the strict unpack succeeds -- but it is not a real
    # session and must not become a phantom (date, "") row downstream.
    assert _process_path(Path("/x/20230622__01_a1.rec")) == NONE_RESULT
    assert _process_path(Path("/x/20230622__metadata.yml")) == NONE_RESULT

    # The lenient .yml form is skipped (no empty-animal row), not raised.
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "20230622__metadata.yml")  # empty animal, lenient ext
    df = get_file_info(tmp_path)
    assert "" not in set(df.animal)

    # The session-extension form aborts loudly instead of being silently dropped.
    _touch(tmp_path, "20230622__01_a1.rec")
    with pytest.raises(ValueError, match="naming convention"):
        get_file_info(tmp_path)


def test_all_botched_session_files_are_listed_in_the_error(tmp_path):
    # The abort collects *every* offending session file (count + sorted listing),
    # not just the first one it hits.
    _touch(tmp_path, "20230622_sample_01_a1.rec")  # valid
    _touch(tmp_path, "20260610sample_03_r2.rec")  # missing separator after date
    _touch(tmp_path, "20230622_my_rat_01_r1.rec")  # underscore in animal name
    with pytest.raises(ValueError) as exc:
        get_file_info(tmp_path)
    message = str(exc.value)
    assert "2 file(s)" in message
    assert "20260610sample_03_r2.rec" in message
    assert "20230622_my_rat_01_r1.rec" in message


def test_seven_digit_prefix_botched_file_is_skipped_not_raised(tmp_path):
    # The raise path requires an 8-digit date prefix; the same botched file with
    # only 7 leading digits is not session-like, so it is skipped, not raised --
    # bracketing the _looks_like_session_filename boundary against
    # test_date_prefixed_botched_session_file_raises (which uses 8 digits).
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    _touch(tmp_path, "2026061sample_03_r2.rec")  # 7-digit prefix
    df = get_file_info(tmp_path)  # must not raise
    names = {Path(p).name for p in df.full_path}
    assert "2026061sample_03_r2.rec" not in names


def test_output_columns_are_integer_typed(tmp_path):
    # Regression guard for #170: an unparseable token must never reach the final
    # .astype({"date": int, ...}); a clean scan yields integer-typed columns.
    _touch(tmp_path, "20230622_sample_01_a1.rec")
    _touch(tmp_path, "20230622_sample_metadata.yml")
    df = get_file_info(tmp_path)
    assert df.date.dtype == int
    assert df.epoch.dtype == int
    assert df.tag_index.dtype == int


def test_directory_with_only_skipped_files_returns_empty_frame(tmp_path):
    # A directory with no parseable session files must not crash on the empty
    # .astype; it returns an empty, correctly-columned frame.
    _touch(tmp_path, "tetrode_12.5.yml")  # probe config only
    df = get_file_info(tmp_path)
    assert len(df) == 0
    assert list(df.columns) == [
        "date",
        "animal",
        "epoch",
        "tag",
        "tag_index",
        "file_extension",
        "full_path",
    ]
