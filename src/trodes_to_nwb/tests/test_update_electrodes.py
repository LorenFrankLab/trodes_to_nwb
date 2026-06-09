"""Tests for the update_electrodes module."""

import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from dateutil.tz import tzutc
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries

from trodes_to_nwb import convert_rec_header, convert_yaml
from trodes_to_nwb.tests.utils import data_path
from trodes_to_nwb.update_electrodes import (
    UPDATABLE_COLUMNS,
    _canonical_hwchan,
    build_electrodes_from_config,
    update_electrodes_from_config,
)

METADATA_PATH = data_path / "20230622_sample_metadataProbeReconfig.yml"
PROBE_METADATA_PATHS = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
TRODESCONF_FILE = data_path / "reconfig_probeDevice.trodesconf"
ELECTRODES_PATH = "/general/extracellular_ephys/electrodes"


def _swap_hwchans(nwb_path, i=0, j=1):
    """Swap two rows' hwChan values in-place to simulate an incorrect config."""
    with h5py.File(str(nwb_path), "a") as f:
        hw_chans = f[ELECTRODES_PATH]["hwChan"][:]
        decoded = [v.decode("utf-8") if isinstance(v, bytes) else v for v in hw_chans]
        decoded[i], decoded[j] = decoded[j], decoded[i]
        f[ELECTRODES_PATH]["hwChan"][...] = [v.encode("utf-8") for v in decoded]


def _create_test_nwb(nwb_path, with_eseries=False):
    """Helper to create a test NWB file with electrodes table using reconfig data.

    If ``with_eseries`` is True, an ElectricalSeries is added whose data column
    ``i`` is the constant ``i`` (its electrode-table row index), so tests can
    verify the electrode-row <-> data-column binding is preserved by an update.
    """
    metadata, probe_metadata = convert_yaml.load_metadata(
        METADATA_PATH, PROBE_METADATA_PATHS
    )

    # Create NWB file directly (without needing a .rec file for GlobalConfiguration)
    nwbfile = NWBFile(
        session_description=metadata["session_description"],
        experimenter=metadata["experimenter_name"],
        lab=metadata["lab"],
        institution=metadata["institution"],
        session_start_time=datetime(2023, 6, 22, tzinfo=tzutc()),
        timestamps_reference_time=datetime.fromtimestamp(0, tz=tzutc()),
        identifier=str(uuid.uuid1()),
        session_id=metadata["session_id"],
        experiment_description=metadata["experiment_description"],
    )

    rec_header = convert_rec_header.read_header(TRODESCONF_FILE)
    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    if with_eseries:
        n_electrodes = len(nwbfile.electrodes)
        region = nwbfile.create_electrode_table_region(
            region=list(range(n_electrodes)), description="all electrodes"
        )
        # data[:, i] == i marks which electrode-table row each data column maps
        # to, so we can detect any disturbance of the data <-> row binding.
        n_samples = 10
        data = np.tile(np.arange(n_electrodes, dtype="int16"), (n_samples, 1))
        nwbfile.add_acquisition(
            ElectricalSeries(
                name="e-series",
                data=data,
                electrodes=region,
                timestamps=np.arange(n_samples, dtype="float64"),
            )
        )

    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    return nwbfile


def test_build_electrodes_from_config():
    """Test that build_electrodes_from_config returns correct hw_chan mapping."""

    hw_chan_map = build_electrodes_from_config(
        METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
    )

    # Check that we got a non-empty dictionary
    assert len(hw_chan_map) > 0

    # Check that first few channels have expected fields
    assert "29" in hw_chan_map
    entry = hw_chan_map["29"]
    assert "group_name" in entry
    assert "location" in entry
    assert "rel_x" in entry
    assert "rel_y" in entry
    assert "rel_z" in entry
    assert "probe_shank" in entry
    assert "probe_electrode" in entry
    assert "ref_elect_id" in entry
    assert "ntrode_id" in entry
    assert "channel_id" in entry
    assert "bad_channel" in entry

    # Verify known values for channel 29 (first electrode of first group)
    assert entry["group_name"] == "0"
    assert entry["probe_electrode"] == 0


def test_update_electrodes_from_config_identity():
    """Test that updating with the same config leaves data unchanged."""

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        nwbfile = _create_test_nwb(nwb_path)

        # Get the original electrode table values
        original_df = nwbfile.electrodes.to_dataframe()

        # Update with the same config (should be identity operation)
        update_electrodes_from_config(
            nwb_path, METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
        )

        # Read back the updated file and check values are unchanged
        with NWBHDF5IO(str(nwb_path), "r") as io:
            updated_nwb = io.read()
            updated_df = updated_nwb.electrodes.to_dataframe()

        # All updatable columns should be unchanged
        for col in UPDATABLE_COLUMNS:
            assert list(updated_df[col]) == list(
                original_df[col]
            ), f"Column {col} changed after identity update"


def test_update_electrodes_from_config_swapped():
    """Test that updating with a config that has swapped channels
    correctly remaps the metadata."""

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        nwbfile = _create_test_nwb(nwb_path)

        # Get the original electrode table values
        original_df = nwbfile.electrodes.to_dataframe()

        # Swap two hwChan values in the NWB file to simulate an incorrect config
        _swap_hwchans(nwb_path)

        # Now update with correct config - should fix the metadata
        update_electrodes_from_config(
            nwb_path, METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
        )

        # Read back the updated file
        with NWBHDF5IO(str(nwb_path), "r") as io:
            updated_nwb = io.read()
            updated_df = updated_nwb.electrodes.to_dataframe()

        # After the update, row 0 should carry the full metadata that originally
        # belonged to row 1, and vice versa (since we swapped their hwChans).
        # Every updatable column must be remapped, not just probe_electrode.
        for col in UPDATABLE_COLUMNS:
            assert (
                updated_df.iloc[0][col] == original_df.iloc[1][col]
            ), f"Column {col} not remapped to swapped hwChan at row 0"
            assert (
                updated_df.iloc[1][col] == original_df.iloc[0][col]
            ), f"Column {col} not remapped to swapped hwChan at row 1"

        # The swap must actually change observable metadata, otherwise the
        # assertions above would pass even if nothing were remapped.
        changed = [
            col
            for col in UPDATABLE_COLUMNS
            if original_df.iloc[0][col] != original_df.iloc[1][col]
        ]
        assert changed, "test fixture rows 0/1 are identical; swap is not observable"


def test_update_electrodes_file_not_found():
    """Test that FileNotFoundError is raised for nonexistent NWB file."""
    with pytest.raises(FileNotFoundError):
        update_electrodes_from_config(
            "/nonexistent/path.nwb",
            METADATA_PATH,
            PROBE_METADATA_PATHS,
            TRODESCONF_FILE,
        )


def test_update_electrodes_missing_hw_chan():
    """Test that KeyError is raised when hwChan in file is not in new config."""

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        _create_test_nwb(nwb_path)

        # Overwrite a hwChan value with something that won't be in the config
        with h5py.File(str(nwb_path), "a") as f:
            hw_chans = f[ELECTRODES_PATH]["hwChan"][:]
            hw_chans[0] = b"99999"
            f[ELECTRODES_PATH]["hwChan"][...] = hw_chans

        with pytest.raises(KeyError):
            update_electrodes_from_config(
                nwb_path, METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
            )


def test_update_electrodes_device_change_raises():
    """Test that ValueError is raised when new config would change probe device."""

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        _create_test_nwb(nwb_path)

        # Manually change a group_name in the existing file so it no longer
        # matches what the config would produce for that hwChan
        with h5py.File(str(nwb_path), "a") as f:
            group_names = f[ELECTRODES_PATH]["group_name"][:]
            # Change the first electrode's group_name to a different value
            group_names[0] = b"999"
            f[ELECTRODES_PATH]["group_name"][...] = group_names

        with pytest.raises(ValueError, match="different probe devices"):
            update_electrodes_from_config(
                nwb_path, METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
            )


def test_update_preserves_electrical_series_alignment():
    """Updating metadata must not disturb the electrode-row <-> data-column
    binding, and each row must end up describing the electrode at its hwChan."""

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        _create_test_nwb(nwb_path, with_eseries=True)

        # Capture the ElectricalSeries data and the electrode region before the
        # update.
        with h5py.File(str(nwb_path), "a") as f:
            original_es_data = f["/acquisition/e-series/data"][:].copy()
            original_region = f["/acquisition/e-series/electrodes"][:].copy()

        # Simulate the bug by swapping two rows' hwChans.
        _swap_hwchans(nwb_path)

        update_electrodes_from_config(
            nwb_path, METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
        )

        expected = build_electrodes_from_config(
            METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
        )

        with NWBHDF5IO(str(nwb_path), "r") as io:
            updated_nwb = io.read()
            es = updated_nwb.acquisition["e-series"]
            updated_es_data = es.data[:]
            updated_region = np.asarray(es.electrodes.data[:])
            updated_df = updated_nwb.electrodes.to_dataframe()
            row_hw_chans = list(updated_df["hwChan"])

        # The module must never touch the data or the region it indexes.
        np.testing.assert_array_equal(updated_es_data, original_es_data)
        np.testing.assert_array_equal(updated_region, original_region)

        # Each electrode row must now describe the electrode at its hwChan.
        for i, hw_chan in enumerate(row_hw_chans):
            meta = expected[_canonical_hwchan(hw_chan)]
            for col in UPDATABLE_COLUMNS:
                assert updated_df.iloc[i][col] == meta[col], (
                    f"Row {i} (hwChan {hw_chan}) column {col} does not match "
                    "the corrected config"
                )


def test_update_electrodes_missing_required_column_raises():
    """A file missing a required updatable column must raise rather than
    silently perform a partial update."""

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        _create_test_nwb(nwb_path)

        with h5py.File(str(nwb_path), "a") as f:
            del f[ELECTRODES_PATH]["ref_elect_id"]

        with pytest.raises(KeyError, match="ref_elect_id"):
            update_electrodes_from_config(
                nwb_path, METADATA_PATH, PROBE_METADATA_PATHS, TRODESCONF_FILE
            )


@pytest.mark.parametrize(
    "value,expected",
    [
        (b"29", "29"),
        ("29", "29"),
        (29, "29"),
        (29.0, "29"),
        (np.int64(29), "29"),
        (np.float64(29.0), "29"),
        (" 29 ", "29"),
        ("ref", "ref"),
    ],
)
def test_canonical_hwchan(value, expected):
    """hwChan keys are canonical across bytes/str/int/float representations."""
    assert _canonical_hwchan(value) == expected
