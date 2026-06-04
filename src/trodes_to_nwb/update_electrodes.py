"""Provides a function to update the electrodes table in an existing NWB file
using a corrected Trodes configuration file. This allows fixing electrode
metadata (e.g., shank/probe position info) without rewriting the entire file.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from dateutil.tz import tzutc
from pynwb import NWBFile

from trodes_to_nwb import convert_rec_header, convert_yaml

logger = logging.getLogger("convert")

# Columns in the electrodes table that define electrode identity/position
# These are the columns that get remapped when hw channel assignments change
UPDATABLE_COLUMNS = [
    "group_name",
    "location",
    "rel_x",
    "rel_y",
    "rel_z",
    "ntrode_id",
    "channel_id",
    "bad_channel",
    "probe_shank",
    "probe_electrode",
    "ref_elect_id",
]


def build_electrodes_from_config(
    metadata_path: str | Path,
    probe_metadata_paths: list[str | Path],
    trodesconf_path: str | Path,
) -> dict:
    """Build a correct electrodes table from a trodes config and metadata yaml.

    Uses the same add_electrode_groups code path as normal conversion to ensure
    consistency, by constructing a temporary NWBFile and reading the resulting
    electrode table.

    Parameters
    ----------
    metadata_path : str or Path
        Path to the metadata yaml file.
    probe_metadata_paths : list of str or Path
        List of paths to probe metadata yaml files.
    trodesconf_path : str or Path
        Path to the corrected trodes configuration file (.trodesconf or .rec).

    Returns
    -------
    dict
        A dictionary mapping hwChan (str) to a dict of electrode metadata fields.
    """
    metadata, probe_metadata = convert_yaml.load_metadata(
        metadata_path, probe_metadata_paths
    )

    rec_header = convert_rec_header.read_header(trodesconf_path)
    spike_config = rec_header.find("SpikeConfiguration")

    hw_channel_map = convert_rec_header.make_hw_channel_map(metadata, spike_config)
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, spike_config
    )

    # Build a temporary NWBFile and populate electrodes using the same code
    # path as normal conversion (add_electrode_groups) to ensure consistency.
    tmp_nwbfile = NWBFile(
        session_description="temp",
        identifier=str(uuid.uuid4()),
        session_start_time=datetime(2000, 1, 1, tzinfo=tzutc()),
        timestamps_reference_time=datetime.fromtimestamp(0, tz=tzutc()),
    )

    convert_yaml.add_electrode_groups(
        tmp_nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    # Read the electrode table from the temporary NWBFile
    electrode_df = tmp_nwbfile.electrodes.to_dataframe()

    # Build the hwChan-keyed mapping from the electrode table
    hw_chan_to_metadata = {}
    for idx, row in electrode_df.iterrows():
        hw_chan = str(row["hwChan"])
        hw_chan_to_metadata[hw_chan] = {
            "hwChan": hw_chan,
            "group_name": str(row["group_name"]),
            "location": row["location"],
            "rel_x": float(row["rel_x"]),
            "rel_y": float(row["rel_y"]),
            "rel_z": float(row["rel_z"]),
            "ntrode_id": row["ntrode_id"],
            "channel_id": row["channel_id"],
            "bad_channel": bool(row["bad_channel"]),
            "probe_shank": row["probe_shank"],
            "probe_electrode": row["probe_electrode"],
            "ref_elect_id": row["ref_elect_id"],
        }

    return hw_chan_to_metadata


def update_electrodes_from_config(
    nwb_file_path: str | Path,
    metadata_path: str | Path,
    probe_metadata_paths: list[str | Path],
    trodesconf_path: str | Path,
) -> None:
    """Update the electrodes table in an existing NWB file using a corrected
    Trodes configuration.

    For each row in the existing electrodes table, the hardware channel (hwChan)
    is used to identify the correct electrode metadata from the new configuration.
    The row's metadata is then updated in-place using h5py.

    Parameters
    ----------
    nwb_file_path : str or Path
        Path to the existing NWB file to update.
    metadata_path : str or Path
        Path to the metadata yaml file.
    probe_metadata_paths : list of str or Path
        List of paths to probe metadata yaml files.
    trodesconf_path : str or Path
        Path to the corrected trodes configuration file (.trodesconf or .rec).

    Raises
    ------
    KeyError
        If a hardware channel in the existing file cannot be found in the new config.
    ValueError
        If the new configuration would reassign an electrode to a different
        probe device (group_name). This operation only updates metadata within
        the same device; changing device assignment requires full reconversion.
    """
    nwb_file_path = Path(nwb_file_path)
    if not nwb_file_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_file_path}")

    # Build the correct electrode metadata from the new config
    hw_chan_to_metadata = build_electrodes_from_config(
        metadata_path, probe_metadata_paths, trodesconf_path
    )

    # Open existing NWB file and update electrodes table in-place
    electrodes_path = "/general/extracellular_ephys/electrodes"

    with h5py.File(nwb_file_path, "a") as f:
        electrodes_group = f[electrodes_path]

        # Read existing hwChan values to use as the key for matching
        existing_hw_chans = electrodes_group["hwChan"][:]
        # Decode bytes to str if necessary
        if existing_hw_chans.dtype.kind == "O" or existing_hw_chans.dtype.kind == "S":
            existing_hw_chans = [
                v.decode("utf-8") if isinstance(v, bytes) else v
                for v in existing_hw_chans
            ]
        else:
            existing_hw_chans = [str(v) for v in existing_hw_chans]

        n_electrodes = len(existing_hw_chans)

        # Build new column data arrays
        new_data = {col: [] for col in UPDATABLE_COLUMNS}

        for hw_chan in existing_hw_chans:
            if hw_chan not in hw_chan_to_metadata:
                raise KeyError(
                    f"Hardware channel {hw_chan} in existing NWB file not found "
                    f"in the new trodes configuration."
                )
            meta = hw_chan_to_metadata[hw_chan]
            for col in UPDATABLE_COLUMNS:
                new_data[col].append(meta[col])

        # Check if the new config would change any electrode's probe device.
        # This function only remaps metadata within the same device; changing
        # the device assignment is not supported and requires full reconversion.
        if "group_name" in electrodes_group:
            existing_group_names = electrodes_group["group_name"][:]
            existing_group_names = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in existing_group_names
            ]
            mismatched = []
            for i, (existing, new) in enumerate(
                zip(existing_group_names, new_data["group_name"])
            ):
                if existing != new:
                    mismatched.append(
                        f"  hwChan {existing_hw_chans[i]}: "
                        f"'{existing}' -> '{new}'"
                    )
            if mismatched:
                details = "\n".join(mismatched[:10])
                raise ValueError(
                    "The new configuration would reassign electrodes to "
                    "different probe devices (group_name changes detected). "
                    "This operation only supports updating metadata within "
                    "the same device. Mismatched electrodes:\n" + details
                )

        # Write updated data back to the HDF5 file
        for col in UPDATABLE_COLUMNS:
            if col not in electrodes_group:
                logger.warning(
                    f"Column '{col}' not found in electrodes table, skipping."
                )
                continue

            dataset = electrodes_group[col]
            values = new_data[col]

            # Handle string vs numeric data
            if dataset.dtype.kind in ("O", "S") or (
                len(values) > 0 and isinstance(values[0], str)
            ):
                # String data - need to handle variable-length strings
                encoded = [
                    v.encode("utf-8") if isinstance(v, str) else v for v in values
                ]
                dataset[...] = encoded
            elif len(values) > 0 and isinstance(values[0], bool):
                dataset[...] = np.array(values, dtype=dataset.dtype)
            else:
                dataset[...] = np.array(values, dtype=dataset.dtype)

    logger.info(
        f"Successfully updated electrodes table in {nwb_file_path} "
        f"with {n_electrodes} electrodes."
    )
