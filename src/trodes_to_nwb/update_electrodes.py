"""In-place correction of the electrodes table in an existing NWB file.

Provides functions to fix electrode metadata (e.g. shank/probe position info)
using a corrected Trodes configuration, without rewriting the entire file.
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

# Electrode-table columns that this module overwrites from the trodes config.
# These describe electrode identity, position, referencing, and indexing for a
# given hardware channel. ``hwChan`` is intentionally excluded: it is the key
# used to match each existing row to the corrected metadata and must never be
# modified (it is also the column the underlying ElectricalSeries data is
# ordered by, so changing it would break the data <-> row binding).
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

# Coercions applied to specific columns when reading the corrected electrode
# table, so the returned mapping holds clean Python scalars. Columns in
# UPDATABLE_COLUMNS that are not listed here are passed through unchanged.
_COLUMN_COERCIONS = {
    "group_name": str,
    "rel_x": float,
    "rel_y": float,
    "rel_z": float,
    "bad_channel": bool,
}


def _canonical_hwchan(value) -> str:
    """Normalize a hardware-channel identifier to a canonical string key.

    Both the corrected-config side and the on-disk side build their hwChan
    lookup keys through this function so the keys match regardless of the
    underlying representation (bytes, numpy scalar, int, or float). Integer
    valued inputs canonicalize to their integer string (e.g. ``29``, ``29.0``,
    and ``b"29"`` all map to ``"29"``), so a float vs int dtype difference
    between the file and the config cannot cause a spurious lookup miss.

    Parameters
    ----------
    value : bytes, str, int, float, or numpy scalar
        The hardware-channel identifier to normalize.

    Returns
    -------
    str
        The canonical string key. Integer-valued inputs return their integer
        string; non-numeric inputs return ``str(value)`` unchanged.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_float.is_integer():
        return str(int(as_float))
    return str(as_float)


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
        A dictionary mapping a canonical hwChan key (str) to a dict of electrode
        metadata fields. Each value contains ``hwChan`` plus every column in
        ``UPDATABLE_COLUMNS`` (``group_name``, ``location``, ``rel_x``,
        ``rel_y``, ``rel_z``, ``ntrode_id``, ``channel_id``, ``bad_channel``,
        ``probe_shank``, ``probe_electrode``, ``ref_elect_id``), with string,
        float, and bool coercions applied as appropriate.

    Raises
    ------
    ValueError
        If the corrected configuration does not assign a unique hwChan to every
        electrode (hwChan must be a unique key for the remap to be well-defined).

    Notes
    -----
    Assumes each ``hwChan`` value is unique across the electrodes table; this is
    the invariant the whole remap relies on and it is enforced here.
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
    # UPDATABLE_COLUMNS is the single source of truth for which fields are
    # carried; _COLUMN_COERCIONS supplies the per-field scalar coercion.
    hw_chan_to_metadata = {}
    for _, row in electrode_df.iterrows():
        hw_chan = _canonical_hwchan(row["hwChan"])
        entry = {"hwChan": hw_chan}
        for col in UPDATABLE_COLUMNS:
            coerce = _COLUMN_COERCIONS.get(col)
            entry[col] = coerce(row[col]) if coerce else row[col]
        entry["probe_type"] = row["group"].device.probe_type
        hw_chan_to_metadata[hw_chan] = entry

    # hwChan must be a unique key: a collision would silently overwrite an
    # electrode's metadata with another's and corrupt the remap.
    if len(hw_chan_to_metadata) != len(electrode_df):
        raise ValueError(
            "The corrected configuration does not assign a unique hwChan to "
            f"every electrode ({len(electrode_df)} electrodes but only "
            f"{len(hw_chan_to_metadata)} unique hwChan values). hwChan must be "
            "unique to remap electrode metadata."
        )

    return hw_chan_to_metadata


def _prepare_column_data(col: str, dataset: "h5py.Dataset", values: list):
    """Validate and encode one column's values for an in-place HDF5 write.

    Validation happens before any data is written, so the caller can check
    every column up front and guarantee the actual write loop never fails
    partway through.

    Parameters
    ----------
    col : str
        Name of the electrodes-table column being written (used in error
        messages).
    dataset : h5py.Dataset
        The existing on-disk dataset whose dtype the returned values must match.
    values : list
        The new per-row values to write, in electrode-table row order.

    Returns
    -------
    list of bytes or numpy.ndarray
        The values ready to assign to ``dataset[...]``: a list of UTF-8 bytes
        for string columns, or an array cast to the dataset dtype for numeric
        columns.

    Raises
    ------
    ValueError
        If a string value would be truncated by a fixed-width string dataset,
        or a numeric cast to the dataset dtype would lose precision or overflow.
    """
    kind = dataset.dtype.kind

    if kind in ("O", "S"):
        encoded = [v.encode("utf-8") if isinstance(v, str) else v for v in values]
        # Fixed-width string datasets silently truncate over-long values; refuse.
        if kind == "S":
            itemsize = dataset.dtype.itemsize
            too_long = [
                e for e in encoded if isinstance(e, bytes) and len(e) > itemsize
            ]
            if too_long:
                raise ValueError(
                    f"Column '{col}': new value(s) exceed the fixed string width "
                    f"of {itemsize} bytes and would be truncated: "
                    f"{[e.decode('utf-8', 'replace') for e in too_long]}"
                )
        return encoded

    # Numeric (and bool) columns: refuse casts that would silently round or
    # overflow (e.g. float coordinates into an int dataset, or a -1 reference
    # sentinel into an unsigned dataset).
    arr = np.asarray(values)
    if np.can_cast(arr.dtype, dataset.dtype, casting="safe"):
        return arr.astype(dataset.dtype)
    cast = arr.astype(dataset.dtype)
    if not np.array_equal(cast.astype(arr.dtype), arr):
        raise ValueError(
            f"Column '{col}': new values cannot be cast from {arr.dtype} to "
            f"{dataset.dtype} without loss or overflow."
        )
    return cast


def update_electrodes_from_config(
    nwb_file_path: str | Path,
    metadata_path: str | Path,
    probe_metadata_paths: list[str | Path],
    trodesconf_path: str | Path,
) -> None:
    """Correct an existing NWB file's electrodes table from a Trodes config.

    For each row in the existing electrodes table, the hardware channel (hwChan)
    is used to identify the correct electrode metadata from the new
    configuration. The row's metadata is then updated in-place using h5py.

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
    FileNotFoundError
        If ``nwb_file_path`` does not exist, or (indirectly) if no probe
        metadata matches a device type in the configuration.
    KeyError
        If a hardware channel in the existing file cannot be found in the new
        config, or a required column is missing from the existing electrodes
        table.
    ValueError
        If the new configuration would reassign an electrode to a different
        probe device (group_name); if the existing table is empty or
        internally inconsistent; or if a corrected value cannot be written to
        an existing column without silent truncation/overflow. This operation
        only updates metadata within the same device; changing device
        assignment requires full reconversion.

    Notes
    -----
    The file is modified in place with no backup. All validation (device
    change, missing columns, hwChan matching, value encodability) runs *before*
    any data is written, so a validation failure leaves the file untouched.
    The per-column writes themselves are not transactional, however: a hard
    interruption (e.g. power loss) during the write phase could still leave the
    electrodes table partially updated. Back up the file first if that matters.
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

        # Existing hwChan values, normalized to canonical keys for matching.
        existing_hw_chans = [
            _canonical_hwchan(v) for v in electrodes_group["hwChan"][:]
        ]
        n_electrodes = len(existing_hw_chans)
        if n_electrodes == 0:
            raise ValueError(
                f"Electrodes table in {nwb_file_path} is empty; nothing to update."
            )

        # Build new column data arrays, keyed by hwChan, in existing row order.
        new_data = {col: [] for col in UPDATABLE_COLUMNS}
        new_data["probe_type"] = (
            []
        )  # not updatable, but we still need it for device-change checks
        for hw_chan in existing_hw_chans:
            if hw_chan not in hw_chan_to_metadata:
                raise KeyError(
                    f"Hardware channel {hw_chan} in existing NWB file not found "
                    f"in the new trodes configuration."
                )
            meta = hw_chan_to_metadata[hw_chan]
            for col in UPDATABLE_COLUMNS:
                new_data[col].append(meta[col])
            new_data["probe_type"].append(meta["probe_type"])

        # Refuse to silently move an electrode to a different probe device.
        # group_name is compared positionally, in electrode-table row order:
        # both arrays are indexed by the same existing rows.
        if "group_name" in electrodes_group:
            existing_group_names = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in electrodes_group["group_name"][:]
            ]
            if len(existing_group_names) != n_electrodes:
                raise ValueError(
                    f"Column 'group_name' has length {len(existing_group_names)} "
                    f"but the table has {n_electrodes} electrodes; file is "
                    "inconsistent."
                )
            mismatched = [
                f"  hwChan {existing_hw_chans[i]}: '{existing}' -> '{new}'"
                for i, (existing, new) in enumerate(
                    zip(existing_group_names, new_data["group_name"])
                )
                if existing != new
            ]
            if mismatched:
                details = "\n".join(mismatched[:10])
                raise ValueError(
                    "The new configuration would reassign electrodes to "
                    "different probe devices (group_name changes detected). "
                    "This operation only supports updating metadata within "
                    "the same device. Mismatched electrodes:\n" + details
                )

        # Check that the new metadata does not map any existing hwChan to a different
        # probe_type, which would also indicate a device change. This is NOT
        # supported by this update operation and would require a full reconversion.
        if "group" in electrodes_group:
            existing_probe_types = [
                f[ref]["device"].attrs["probe_type"]
                for ref in electrodes_group["group"][:]
            ]
            mismatched = [
                f"  hwChan {existing_hw_chans[i]}: '{existing}' -> '{new}'"
                for i, (existing, new) in enumerate(
                    zip(existing_probe_types, new_data["probe_type"])
                )
            ]
            if mismatched:
                details = "\n".join(mismatched[:10])
                raise ValueError(
                    "The new configuration would reassign electrodes to "
                    "different probe devices (probe_type changes detected). "
                    "This operation only supports updating metadata within "
                    "the same device. Mismatched electrodes:\n" + details
                )

        # Pre-flight: validate and encode EVERY column before mutating anything.
        # Every column is required (the set is a coherent description of an
        # electrode; a partial update would leave the table inconsistent), each
        # must match the table length, and each value must be writable without
        # silent loss. Any failure here aborts before a single byte is written.
        prepared = {}
        for col in UPDATABLE_COLUMNS:
            if col not in electrodes_group:
                raise KeyError(
                    f"Required column '{col}' is missing from the electrodes "
                    f"table in {nwb_file_path}; refusing to perform a partial "
                    "update."
                )
            dataset = electrodes_group[col]
            if dataset.shape[0] != n_electrodes:
                raise ValueError(
                    f"Column '{col}' has length {dataset.shape[0]} but the table "
                    f"has {n_electrodes} electrodes; file is inconsistent."
                )
            prepared[col] = _prepare_column_data(col, dataset, new_data[col])

        # All columns validated; perform the writes. This loop cannot raise.
        for col, arr in prepared.items():
            electrodes_group[col][...] = arr

    logger.info(
        f"Successfully updated electrodes table in {nwb_file_path} "
        f"with {n_electrodes} electrodes."
    )
