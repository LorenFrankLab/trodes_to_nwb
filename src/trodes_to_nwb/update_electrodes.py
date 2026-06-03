"""Provides a function to update the electrodes table in an existing NWB file
using a corrected Trodes configuration file. This allows fixing electrode
metadata (e.g., shank/probe position info) without rewriting the entire file.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

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

    # Build the electrodes table data by iterating through metadata
    # (mirrors the logic in convert_yaml.add_electrode_groups)
    electrodes = []
    for egroup_metadata in metadata["electrode_groups"]:
        # find correct channel map info
        channel_map = None
        for test_meta in metadata["ntrode_electrode_group_channel_map"]:
            if test_meta["electrode_group_id"] == egroup_metadata["id"]:
                channel_map = test_meta
                break

        # find the probe corresponding to the device type
        probe_meta = None
        for test_meta in probe_metadata:
            if test_meta.get("probe_type", None) == egroup_metadata["device_type"]:
                probe_meta = test_meta
                break
        if probe_meta is None:
            raise FileNotFoundError(
                f"No probe metadata found for {egroup_metadata['device_type']}"
            )

        electrode_counter_probe = 0
        for shank_counter, shank_meta in enumerate(probe_meta["shanks"]):
            for electrode_meta in shank_meta["electrodes"]:
                hw_chan = hw_channel_map[egroup_metadata["id"]][
                    str(electrode_meta["id"])
                ]
                electrodes.append(
                    {
                        "hwChan": hw_chan,
                        "group_name": str(egroup_metadata["id"]),
                        "location": egroup_metadata["targeted_location"],
                        "rel_x": float(electrode_meta["rel_x"]),
                        "rel_y": float(electrode_meta["rel_y"]),
                        "rel_z": float(electrode_meta["rel_z"]),
                        "ntrode_id": channel_map["ntrode_id"],
                        "channel_id": electrode_counter_probe,
                        "bad_channel": bool(
                            electrode_counter_probe in channel_map["bad_channels"]
                        ),
                        "probe_shank": shank_counter,
                        "probe_electrode": electrode_counter_probe,
                    }
                )
                electrode_counter_probe += 1

    # Compute ref_elect_id for each electrode
    # Build index: (group_name, probe_electrode) -> row index in electrodes list
    index_map = {}
    for idx, elec in enumerate(electrodes):
        key = (elec["group_name"], elec["probe_electrode"])
        index_map[key] = idx

    for elec in electrodes:
        group_name = elec["group_name"]
        ref_group, ref_electrode = ref_electrode_map[group_name]
        if ref_group == -1:
            elec["ref_elect_id"] = -1
        else:
            ref_key = (str(ref_group), ref_electrode)
            elec["ref_elect_id"] = index_map[ref_key]

    # Return as a dict keyed by hwChan for easy lookup
    hw_chan_to_metadata = {}
    for elec in electrodes:
        hw_chan_to_metadata[elec["hwChan"]] = elec

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
