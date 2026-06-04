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

from trodes_to_nwb import convert_rec_header, convert_yaml
from trodes_to_nwb.tests.utils import data_path
from trodes_to_nwb.update_electrodes import (
    UPDATABLE_COLUMNS,
    build_electrodes_from_config,
    update_electrodes_from_config,
)


def _create_test_nwb(nwb_path):
    """Helper to create a test NWB file with electrodes table using reconfig data."""
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata_paths = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(
        metadata_path, probe_metadata_paths
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

    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)
    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    return nwbfile


def test_build_electrodes_from_config():
    """Test that build_electrodes_from_config returns correct hw_chan mapping."""
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata_paths = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"

    hw_chan_map = build_electrodes_from_config(
        metadata_path, probe_metadata_paths, trodesconf_file
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
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata_paths = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        nwbfile = _create_test_nwb(nwb_path)

        # Get the original electrode table values
        original_df = nwbfile.electrodes.to_dataframe()

        # Update with the same config (should be identity operation)
        update_electrodes_from_config(
            nwb_path, metadata_path, probe_metadata_paths, trodesconf_file
        )

        # Read back the updated file and check values are unchanged
        with NWBHDF5IO(str(nwb_path), "r") as io:
            updated_nwb = io.read()
            updated_df = updated_nwb.electrodes.to_dataframe()

        # All updatable columns should be unchanged
        for col in UPDATABLE_COLUMNS:
            assert list(updated_df[col]) == list(original_df[col]), (
                f"Column {col} changed after identity update"
            )


def test_update_electrodes_from_config_swapped():
    """Test that updating with a config that has swapped channels
    correctly remaps the metadata."""
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata_paths = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        nwbfile = _create_test_nwb(nwb_path)

        # Get the original electrode table values
        original_df = nwbfile.electrodes.to_dataframe()

        # Now manually swap two hwChan values in the NWB file to simulate
        # an incorrect config
        electrodes_path = "/general/extracellular_ephys/electrodes"
        with h5py.File(str(nwb_path), "a") as f:
            hw_chans = f[electrodes_path]["hwChan"][:]
            # Decode
            hw_chans_decoded = [
                v.decode("utf-8") if isinstance(v, bytes) else v
                for v in hw_chans
            ]
            # Swap first two hw channels
            hw_chans_decoded[0], hw_chans_decoded[1] = (
                hw_chans_decoded[1],
                hw_chans_decoded[0],
            )
            # Write back
            encoded = [v.encode("utf-8") for v in hw_chans_decoded]
            f[electrodes_path]["hwChan"][...] = encoded

        # Now update with correct config - should fix the metadata
        update_electrodes_from_config(
            nwb_path, metadata_path, probe_metadata_paths, trodesconf_file
        )

        # Read back the updated file
        with NWBHDF5IO(str(nwb_path), "r") as io:
            updated_nwb = io.read()
            updated_df = updated_nwb.electrodes.to_dataframe()

        # After the update, row 0 should have metadata for hwChan that was
        # originally at row 1, and vice versa (since we swapped hwChans)
        # Row 0 now has hw_chan that originally was at row 1
        assert updated_df.iloc[0]["probe_electrode"] == original_df.iloc[1]["probe_electrode"]
        assert updated_df.iloc[1]["probe_electrode"] == original_df.iloc[0]["probe_electrode"]


def test_update_electrodes_file_not_found():
    """Test that FileNotFoundError is raised for nonexistent NWB file."""
    with pytest.raises(FileNotFoundError):
        update_electrodes_from_config(
            "/nonexistent/path.nwb",
            data_path / "20230622_sample_metadataProbeReconfig.yml",
            [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"],
            data_path / "reconfig_probeDevice.trodesconf",
        )


def test_update_electrodes_missing_hw_chan():
    """Test that KeyError is raised when hwChan in file is not in new config."""
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata_paths = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        _create_test_nwb(nwb_path)

        # Overwrite a hwChan value with something that won't be in the config
        electrodes_path = "/general/extracellular_ephys/electrodes"
        with h5py.File(str(nwb_path), "a") as f:
            hw_chans = f[electrodes_path]["hwChan"][:]
            hw_chans[0] = b"99999"
            f[electrodes_path]["hwChan"][...] = hw_chans

        with pytest.raises(KeyError):
            update_electrodes_from_config(
                nwb_path, metadata_path, probe_metadata_paths, trodesconf_file
            )


def test_update_electrodes_device_change_raises():
    """Test that ValueError is raised when new config would change probe device."""
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata_paths = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"

    with tempfile.TemporaryDirectory() as tmpdir:
        nwb_path = Path(tmpdir) / "test.nwb"
        _create_test_nwb(nwb_path)

        # Manually change a group_name in the existing file so it no longer
        # matches what the config would produce for that hwChan
        electrodes_path = "/general/extracellular_ephys/electrodes"
        with h5py.File(str(nwb_path), "a") as f:
            group_names = f[electrodes_path]["group_name"][:]
            # Change the first electrode's group_name to a different value
            group_names[0] = b"999"
            f[electrodes_path]["group_name"][...] = group_names

        with pytest.raises(ValueError, match="different probe devices"):
            update_electrodes_from_config(
                nwb_path, metadata_path, probe_metadata_paths, trodesconf_file
            )
