"""Output-safety tests for create_nwbs / _create_nwb (#174)."""

import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from trodes_to_nwb import convert
from trodes_to_nwb.convert import _create_nwb, create_nwbs


class _SyncFuture:
    """Stand-in for a dask Future that resolves to a value synchronously."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _SyncClient:
    """Synchronous stand-in for ``dask.distributed.Client``.

    Runs each mapped task in-process so the parallel branch of ``create_nwbs``
    can be exercised without spinning up a real cluster (matching how the
    parallel-mode swallow was originally reproduced with a mock).
    """

    def __init__(self, *args, **kwargs):
        pass

    def map(self, func, iterable):
        return [_SyncFuture(func(item)) for item in iterable]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# Permission bits are advisory for the superuser, so a read-only directory is
# still writable when the suite runs as root (common in CI containers).
_is_root = getattr(os, "geteuid", lambda: 1)() == 0


def test_create_nwb_refuses_to_overwrite_existing_output(tmp_path):
    # session is (date, animal); output file is f"{animal}{date}.nwb"
    session = ("20230101", "rat")
    (tmp_path / "rat20230101.nwb").write_text("")
    try:
        with pytest.raises(FileExistsError, match="already exists"):
            _create_nwb(
                session,
                pd.DataFrame(),
                output_dir=str(tmp_path),
                overwrite=False,
            )
    finally:
        # _create_nwb opens a per-session logfile in the CWD before the check
        logfile = Path("rat20230101_convert.log")
        if logfile.exists():
            logfile.unlink()


def test_create_nwbs_creates_missing_output_dir(tmp_path):
    out = tmp_path / "new" / "nested"
    assert not out.exists()
    # empty source dir -> no sessions to convert, but the output dir is still
    # created up front so a bad path fails before any conversion work.
    create_nwbs(path=tmp_path, output_dir=str(out))
    assert out.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
@pytest.mark.skipif(_is_root, reason="root bypasses directory permissions")
def test_create_nwbs_refuses_non_writable_output_dir(tmp_path):
    # An existing read-only output_dir passes mkdir(exist_ok=True) but cannot be
    # written to; the create/delete probe must catch this up front rather than
    # after a full conversion.
    out = tmp_path / "readonly"
    out.mkdir()
    os.chmod(out, stat.S_IRUSR | stat.S_IXUSR)  # read + execute, no write
    try:
        with pytest.raises(PermissionError, match="not writable"):
            create_nwbs(path=tmp_path, output_dir=str(out))
    finally:
        os.chmod(out, stat.S_IRWXU)


def test_create_nwbs_parallel_propagates_overwrite_refusal(tmp_path, monkeypatch):
    # n_workers > 1 must not report success when a session refuses to overwrite:
    # the worker-side FileExistsError has to propagate to the caller, not just be
    # printed (the failure mode flagged on #183).
    date, animal = "20230101", "rat"
    (tmp_path / f"{animal}{date}.nwb").write_text("")
    file_info = pd.DataFrame({"date": [date], "animal": [animal]})
    monkeypatch.setattr(convert, "get_file_info", lambda path: file_info)
    monkeypatch.setattr(convert, "Client", _SyncClient)
    try:
        with pytest.raises(FileExistsError, match="already exists"):
            create_nwbs(
                path=tmp_path,
                output_dir=str(tmp_path),
                n_workers=2,
                overwrite=False,
            )
    finally:
        logfile = Path(f"{animal}{date}_convert.log")
        if logfile.exists():
            logfile.unlink()
