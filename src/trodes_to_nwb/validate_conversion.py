"""Validation utilities for comparing SpikeGadgets rec files against an NWB output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO
from trodes_to_nwb.convert_analog import get_analog_channel_names
from trodes_to_nwb.convert_dios import _get_channel_name_map
from trodes_to_nwb.convert_ephys import (
    DEFAULT_CHUNK_TIME_DIM,
    MICROVOLTS_PER_VOLT,
    RecFileDataChunkIterator,
)
from trodes_to_nwb.convert_rec_header import (
    make_hw_channel_map,
    make_ref_electrode_map,
    read_header,
    validate_yaml_header_electrode_map,
)
from trodes_to_nwb.convert_yaml import load_metadata
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO

HEADER_DEVICE_FIELDS = {
    "headstage_serial": "headstageSerial",
    "headstage_smart_ref_on": "headstageSmartRefOn",
    "realtime_mode": "realtimeMode",
    "headstage_auto_settle_on": "headstageAutoSettleOn",
    "timestamp_at_creation": "timestampAtCreation",
    "controller_firmware_version": "controllerFirmwareVersion",
    "controller_serial": "controllerSerial",
    "save_displayed_chan_only": "saveDisplayedChanOnly",
    "headstage_firmware_version": "headstageFirmwareVersion",
    "qt_version": "qtVersion",
    "compile_date": "compileDate",
    "compile_time": "compileTime",
    "file_prefix": "filePrefix",
    "headstage_gyro_sensor_on": "headstageGyroSensorOn",
    "headstage_mag_sensor_on": "headstageMagSensorOn",
    "trodes_version": "trodesVersion",
    "headstage_accel_sensor_on": "headstageAccelSensorOn",
    "commit_head": "commitHead",
    "system_time_at_creation": "systemTimeAtCreation",
    "file_path": "filePath",
}


@dataclass
class ValidationContext:
    rec_filepaths: list[Path]
    nwb_filepath: Path
    metadata_filepath: Path
    metadata: dict[str, Any]
    device_metadata: list[dict[str, Any]]
    rec_header: Any
    effective_header: Any
    neo_io: list[SpikeGadgetsRawIO]
    nwb_io: NWBHDF5IO
    nwbfile: Any
    behavior_only: bool
    timestamp_tolerance_s: float
    ephys_tolerance_uv: float
    max_ephys_frames_per_chunk: int


def validate_conversion(
    *,
    rec_filepaths: list[Path] | list[str] | None = None,
    nwb_filepath: Path | str,
    metadata_filepath: Path | str,
    data_path: Path | str | None = None,
    header_reconfig_path: Path | str | None = None,
    device_metadata_paths: list[Path] | list[str] | None = None,
    behavior_only: bool = False,
    max_ephys_frames_per_chunk: int = DEFAULT_CHUNK_TIME_DIM,
    ephys_tolerance_uv: float = 1.0,
    timestamp_tolerance_s: float | None = None,
) -> dict[str, Any]:
    """Validate that an NWB file matches its rec-file inputs.

    Parameters
    ----------
    rec_filepaths : list[Path] | list[str] | None
        Ordered list of rec files used for the conversion. If omitted, rec files
        are discovered from `data_path` using the same scanning logic as conversion.
    nwb_filepath : Path | str
        Path to the NWB file to validate.
    metadata_filepath : Path | str
        Path to the YAML metadata used during conversion.
    data_path : Path | str | None, optional
        Root directory to scan for rec files when `rec_filepaths` is not provided.
    header_reconfig_path : Path | str | None, optional
        Optional header override used during conversion.
    device_metadata_paths : list[Path] | list[str] | None, optional
        Optional list of device metadata YAML files. Uses included metadata by default.
    behavior_only : bool, optional
        Whether the recording was converted in behavior-only mode.
    max_ephys_frames_per_chunk : int, optional
        Chunk size for chunked array comparisons.
    ephys_tolerance_uv : float, optional
        Allowed absolute error for ephys samples in stored microvolt units.
    timestamp_tolerance_s : float | None, optional
        Allowed absolute timestamp error in seconds. Defaults to one sample period.

    Returns
    -------
    dict[str, Any]
        JSON-serializable validation report.
    """
    context = _load_validation_context(
        rec_filepaths=rec_filepaths,
        nwb_filepath=nwb_filepath,
        metadata_filepath=metadata_filepath,
        data_path=data_path,
        header_reconfig_path=header_reconfig_path,
        device_metadata_paths=device_metadata_paths,
        behavior_only=behavior_only,
        max_ephys_frames_per_chunk=max_ephys_frames_per_chunk,
        ephys_tolerance_uv=ephys_tolerance_uv,
        timestamp_tolerance_s=timestamp_tolerance_s,
    )

    try:
        checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        errors: list[str] = []

        for validator in (
            _validate_header_and_metadata,
            _validate_electrodes,
            _validate_sample_count,
            _validate_dios,
            _validate_analog,
            _validate_ephys,
        ):
            try:
                checks.append(validator(context))
            except Exception as exc:  # pragma: no cover - defensive path
                errors.append(f"{validator.__name__}: {exc}")
                checks.append(
                    _make_check(
                        name=validator.__name__,
                        passed=False,
                        details=str(exc),
                    )
                )

        passed = all(check["passed"] for check in checks) and not errors
        return {
            "passed": passed,
            "summary": {
                "n_checks": len(checks),
                "n_passed": sum(check["passed"] for check in checks),
                "n_failed": sum(not check["passed"] for check in checks),
            },
            "inputs": {
                "rec_filepaths": [str(path) for path in context.rec_filepaths],
                "nwb_filepath": str(context.nwb_filepath),
                "metadata_filepath": str(context.metadata_filepath),
                "behavior_only": context.behavior_only,
            },
            "checks": checks,
            "warnings": warnings,
            "errors": errors,
        }
    finally:
        context.nwb_io.close()


def _load_validation_context(
    *,
    rec_filepaths: list[Path] | list[str] | None,
    nwb_filepath: Path | str,
    metadata_filepath: Path | str,
    data_path: Path | str | None,
    header_reconfig_path: Path | str | None,
    device_metadata_paths: list[Path] | list[str] | None,
    behavior_only: bool,
    max_ephys_frames_per_chunk: int,
    ephys_tolerance_uv: float,
    timestamp_tolerance_s: float | None,
) -> ValidationContext:
    metadata_path = Path(metadata_filepath)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    rec_paths = _resolve_rec_filepaths(
        rec_filepaths=rec_filepaths,
        data_path=data_path,
        metadata_filepath=metadata_path,
    )

    nwb_path = Path(nwb_filepath)
    if not nwb_path.exists():
        raise FileNotFoundError(nwb_path)

    if device_metadata_paths is None:
        device_metadata_paths = list(_get_included_device_metadata_paths())
    else:
        device_metadata_paths = [Path(path) for path in device_metadata_paths]

    metadata, device_metadata = load_metadata(
        str(metadata_path),
        [str(path) for path in device_metadata_paths],
    )
    rec_header = read_header(rec_paths[0])
    effective_header = (
        read_header(header_reconfig_path)
        if header_reconfig_path is not None
        else rec_header
    )
    validate_yaml_header_electrode_map(
        metadata,
        effective_header.find("SpikeConfiguration"),
    )

    neo_io = [SpikeGadgetsRawIO(filename=str(path)) for path in rec_paths]
    [io.parse_header() for io in neo_io]

    if timestamp_tolerance_s is None:
        sampling_rate = float(
            rec_header.find("HardwareConfiguration").attrib["samplingRate"]
        )
        timestamp_tolerance_s = 1.0 / sampling_rate

    nwb_io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
    nwbfile = nwb_io.read()

    return ValidationContext(
        rec_filepaths=rec_paths,
        nwb_filepath=nwb_path,
        metadata_filepath=metadata_path,
        metadata=metadata,
        device_metadata=device_metadata,
        rec_header=rec_header,
        effective_header=effective_header,
        neo_io=neo_io,
        nwb_io=nwb_io,
        nwbfile=nwbfile,
        behavior_only=behavior_only,
        timestamp_tolerance_s=timestamp_tolerance_s,
        ephys_tolerance_uv=ephys_tolerance_uv,
        max_ephys_frames_per_chunk=max_ephys_frames_per_chunk,
    )


def _validate_header_and_metadata(context: ValidationContext) -> dict[str, Any]:
    nwbfile = context.nwbfile
    metadata = context.metadata
    rec_global = context.rec_header.find("GlobalConfiguration")

    if "header_device" not in nwbfile.devices:
        return _make_check(
            name="header_and_metadata",
            passed=False,
            details="NWB file is missing the header_device entry.",
        )

    header_device = nwbfile.devices["header_device"]
    mismatches = []
    for field_name, header_key in HEADER_DEVICE_FIELDS.items():
        actual = getattr(header_device, field_name)
        expected = rec_global.attrib[header_key]
        if str(actual) != str(expected):
            mismatches.append(f"{field_name}: expected {expected!r}, found {actual!r}")

    expected_start_time = datetime.fromtimestamp(
        int(rec_global.attrib["systemTimeAtCreation"].strip()) / 1000
    )
    if nwbfile.session_start_time.replace(tzinfo=None) != expected_start_time:
        mismatches.append(
            "session_start_time does not match rec systemTimeAtCreation"
        )

    metadata_pairs = {
        "session_description": metadata["session_description"],
        "session_id": metadata["session_id"],
        "experiment_description": metadata["experiment_description"],
        "lab": metadata["lab"],
        "institution": metadata["institution"],
    }
    for field_name, expected in metadata_pairs.items():
        actual = getattr(nwbfile, field_name)
        if actual != expected:
            mismatches.append(f"{field_name}: expected {expected!r}, found {actual!r}")

    if nwbfile.subject is None:
        mismatches.append("subject is missing")
    else:
        subject = nwbfile.subject
        expected_subject = metadata["subject"]
        for field_name in (
            "subject_id",
            "description",
            "genotype",
            "sex",
            "species",
        ):
            if getattr(subject, field_name) != expected_subject[field_name]:
                mismatches.append(
                    f"subject.{field_name}: expected {expected_subject[field_name]!r}, "
                    f"found {getattr(subject, field_name)!r}"
                )

    return _finalize_check("header_and_metadata", mismatches)


def _validate_electrodes(context: ValidationContext) -> dict[str, Any]:
    nwbfile = context.nwbfile
    if context.behavior_only:
        if "e-series" in nwbfile.acquisition:
            return _make_check(
                name="electrodes",
                passed=False,
                details="behavior_only=True but e-series is present in acquisition.",
            )
        return _make_check(
            name="electrodes",
            passed=True,
            details="behavior_only=True and no ephys acquisition is present.",
        )

    if nwbfile.electrodes is None:
        return _make_check(
            name="electrodes",
            passed=False,
            details="NWB file is missing the electrodes table.",
        )

    electrode_df = nwbfile.electrodes.to_dataframe()
    hw_channel_map = make_hw_channel_map(
        context.metadata,
        context.effective_header.find("SpikeConfiguration"),
    )
    ref_electrode_map = make_ref_electrode_map(
        context.metadata,
        context.effective_header.find("SpikeConfiguration"),
    )
    expected_rows = _build_expected_electrode_rows(
        context.metadata,
        context.device_metadata,
        hw_channel_map,
    )

    mismatches = []
    if len(electrode_df) != len(expected_rows):
        mismatches.append(
            f"electrode count mismatch: expected {len(expected_rows)}, "
            f"found {len(electrode_df)}"
        )
        return _finalize_check("electrodes", mismatches)

    for row_index, expected_row in enumerate(expected_rows):
        for field_name in ("group_name", "hwChan", "probe_electrode", "probe_shank"):
            actual = electrode_df.iloc[row_index][field_name]
            expected = expected_row[field_name]
            if str(actual) != str(expected):
                mismatches.append(
                    f"row {row_index} {field_name}: expected {expected!r}, found {actual!r}"
                )

    for row_index, row in electrode_df.iterrows():
        ref_group, ref_probe_electrode = ref_electrode_map[str(row["group_name"])]
        if ref_group == -1:
            continue
        ref_index = electrode_df.index[
            (electrode_df["group_name"] == ref_group)
            & (electrode_df["probe_electrode"] == ref_probe_electrode)
        ][0]
        actual_ref = int(electrode_df.iloc[row_index]["ref_elect_id"])
        if actual_ref != int(ref_index):
            mismatches.append(
                f"row {row_index} ref_elect_id: expected {ref_index}, found {actual_ref}"
            )

    return _finalize_check("electrodes", mismatches)


def _validate_sample_count(context: ValidationContext) -> dict[str, Any]:
    sample_count_series = (
        context.nwbfile.processing["sample_count"].data_interfaces["sample_count"]
    )
    expected_timestamps = _get_rec_timestamps(
        context, stream_id=_get_primary_stream_id(context)
    )
    expected_sample_count = np.concatenate(
        [io.get_analogsignal_timestamps(0, None) for io in context.neo_io]
    )

    mismatches: list[str] = []
    data_mismatch_count = 0
    if sample_count_series.data.shape[0] != expected_sample_count.shape[0]:
        mismatches.append(
            "sample_count data length mismatch: "
            f"expected {expected_sample_count.shape[0]}, "
            f"found {sample_count_series.data.shape[0]}"
        )
        data_mismatch_count = 1
    else:
        actual_data = np.asarray(sample_count_series.data[:])
        if not np.array_equal(actual_data, expected_sample_count):
            data_mismatch_count = int(
                np.count_nonzero(actual_data != expected_sample_count)
            )
            mismatches.append(
                f"sample_count data mismatch_count={data_mismatch_count}"
            )

    timestamp_result = _compare_1d_arrays(
        expected=expected_timestamps,
        actual=np.asarray(sample_count_series.timestamps[:]),
        atol=context.timestamp_tolerance_s,
    )
    mismatches.extend(timestamp_result["messages"])
    return _finalize_check(
        "sample_count",
        mismatches,
        mismatch_count=data_mismatch_count + timestamp_result["mismatch_count"],
        max_abs_error=timestamp_result["max_abs_error"],
    )


def _validate_dios(context: ValidationContext) -> dict[str, Any]:
    behavioral_events = context.nwbfile.processing["behavior"]["behavioral_events"]
    channel_name_map = _get_channel_name_map(context.metadata)
    stream_name = "ECU_digital"
    prefix = _get_dio_prefix(context.neo_io[0], stream_name)
    mismatches: list[str] = []
    mismatch_count = 0
    max_abs_error = 0.0

    for channel_description, channel_info in channel_name_map.items():
        expected_name = channel_info["name"]
        if expected_name not in behavioral_events.time_series:
            mismatches.append(f"Missing DIO timeseries {expected_name!r}")
            mismatch_count += 1
            continue

        actual_series = behavioral_events[expected_name]
        if actual_series.description != channel_description:
            mismatches.append(
                f"{expected_name} description mismatch: expected {channel_description!r}, "
                f"found {actual_series.description!r}"
            )
            mismatch_count += 1
        if actual_series.comments != channel_info["comments"]:
            mismatches.append(
                f"{expected_name} comments mismatch: expected {channel_info['comments']!r}, "
                f"found {actual_series.comments!r}"
            )
            mismatch_count += 1

        expected_timestamps = []
        expected_state_changes = []
        for io in context.neo_io:
            timestamps, state_changes = io.get_digitalsignal(
                stream_name, prefix + channel_description
            )
            expected_timestamps.append(timestamps)
            expected_state_changes.append(state_changes)
        expected_timestamps = np.concatenate(expected_timestamps)
        expected_state_changes = np.concatenate(expected_state_changes)
        actual_state_changes = np.asarray(actual_series.data[:])

        if actual_state_changes.shape != expected_state_changes.shape:
            mismatches.append(
                f"{expected_name} state shape mismatch: expected {expected_state_changes.shape}, "
                f"found {actual_state_changes.shape}"
            )
            mismatch_count += 1
        elif not np.array_equal(actual_state_changes, expected_state_changes):
            this_mismatch_count = int(
                np.count_nonzero(actual_state_changes != expected_state_changes)
            )
            mismatch_count += this_mismatch_count
            mismatches.append(
                f"{expected_name} state mismatch_count={this_mismatch_count}"
            )

        timestamp_result = _compare_1d_arrays(
            expected=expected_timestamps,
            actual=np.asarray(actual_series.timestamps[:]),
            atol=context.timestamp_tolerance_s,
        )
        mismatch_count += timestamp_result["mismatch_count"]
        max_abs_error = max(max_abs_error, timestamp_result["max_abs_error"])
        mismatches.extend(
            f"{expected_name} {message}" for message in timestamp_result["messages"]
        )

    return _finalize_check(
        "dios",
        mismatches,
        mismatch_count=mismatch_count,
        max_abs_error=max_abs_error,
    )


def _validate_analog(context: ValidationContext) -> dict[str, Any]:
    analog_series = context.nwbfile.processing["analog"]["analog"]["analog"]
    analog_channel_names = get_analog_channel_names(context.rec_header)
    multiplexed_channel_names = list(context.neo_io[0].multiplexed_channel_xml.keys())
    expected_channel_names = analog_channel_names + multiplexed_channel_names

    actual_channel_names = analog_series.description.split("   ")[:-1]
    mismatches: list[str] = []
    if actual_channel_names != expected_channel_names:
        mismatches.append(
            "analog channel order mismatch: "
            f"expected {expected_channel_names!r}, found {actual_channel_names!r}"
        )

    analog_iterator = RecFileDataChunkIterator(
        [str(path) for path in context.rec_filepaths],
        nwb_hw_channel_order=analog_channel_names,
        stream_id="ECU_analog",
        is_analog=True,
        timestamps=_get_rec_timestamps(context, stream_id=_get_primary_stream_id(context)),
        behavior_only=context.behavior_only,
    )

    data_result = _compare_chunked_time_series(
        expected_source=analog_iterator,
        actual_source=analog_series.data,
        n_rows=analog_iterator._get_maxshape()[0],
        n_columns=analog_iterator._get_maxshape()[1],
        chunk_size=context.max_ephys_frames_per_chunk,
        atol=0.0,
    )
    timestamp_result = _compare_1d_arrays(
        expected=analog_iterator.timestamps,
        actual=np.asarray(analog_series.timestamps[:]),
        atol=context.timestamp_tolerance_s,
    )
    mismatches.extend(data_result["messages"])
    mismatches.extend(timestamp_result["messages"])
    return _finalize_check(
        "analog",
        mismatches,
        mismatch_count=data_result["mismatch_count"] + timestamp_result["mismatch_count"],
        max_abs_error=max(data_result["max_abs_error"], timestamp_result["max_abs_error"]),
    )


def _validate_ephys(context: ValidationContext) -> dict[str, Any]:
    nwbfile = context.nwbfile
    if context.behavior_only:
        if "e-series" in nwbfile.acquisition:
            return _make_check(
                name="ephys",
                passed=False,
                details="behavior_only=True but e-series is present in acquisition.",
            )
        return _make_check(
            name="ephys",
            passed=True,
            details="behavior_only=True and ephys validation was skipped.",
        )

    e_series = nwbfile.acquisition["e-series"]
    electrode_df = nwbfile.electrodes.to_dataframe()
    nwb_hw_chan_order = [int(hw_chan) for hw_chan in electrode_df["hwChan"].tolist()]
    spike_config = context.rec_header.find("SpikeConfiguration")
    if "rawScalingToUv" in spike_config[0].attrib:
        conversion = float(spike_config[0].attrib["rawScalingToUv"])
    else:
        conversion = context.metadata["raw_data_to_volts"] * MICROVOLTS_PER_VOLT

    ephys_iterator = RecFileDataChunkIterator(
        [str(path) for path in context.rec_filepaths],
        nwb_hw_channel_order=nwb_hw_chan_order,
        conversion=conversion,
        interpolate_dropped_packets=True,
        stream_id="trodes",
    )

    data_result = _compare_chunked_time_series(
        expected_source=ephys_iterator,
        actual_source=e_series.data,
        n_rows=ephys_iterator._get_maxshape()[0],
        n_columns=ephys_iterator._get_maxshape()[1],
        chunk_size=context.max_ephys_frames_per_chunk,
        atol=context.ephys_tolerance_uv,
    )
    timestamp_result = _compare_1d_arrays(
        expected=ephys_iterator.timestamps,
        actual=np.asarray(e_series.timestamps[:]),
        atol=context.timestamp_tolerance_s,
    )

    mismatches = data_result["messages"] + timestamp_result["messages"]
    return _finalize_check(
        "ephys",
        mismatches,
        mismatch_count=data_result["mismatch_count"] + timestamp_result["mismatch_count"],
        max_abs_error=max(data_result["max_abs_error"], timestamp_result["max_abs_error"]),
    )


def _build_expected_electrode_rows(
    metadata: dict[str, Any],
    device_metadata: list[dict[str, Any]],
    hw_channel_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    expected_rows = []
    for electrode_group_metadata in metadata["electrode_groups"]:
        channel_map = next(
            test_meta
            for test_meta in metadata["ntrode_electrode_group_channel_map"]
            if test_meta["electrode_group_id"] == electrode_group_metadata["id"]
        )
        probe_metadata = next(
            test_meta
            for test_meta in device_metadata
            if test_meta.get("probe_type") == electrode_group_metadata["device_type"]
        )
        electrode_counter_probe = 0
        for shank_counter, shank_metadata in enumerate(probe_metadata["shanks"]):
            for electrode_metadata in shank_metadata["electrodes"]:
                electrode_id = str(electrode_metadata["id"])
                expected_rows.append(
                    {
                        "group_name": str(electrode_group_metadata["id"]),
                        "hwChan": str(
                            hw_channel_map[electrode_group_metadata["id"]][electrode_id]
                        ),
                        "probe_electrode": electrode_counter_probe,
                        "probe_shank": shank_counter,
                        "ntrode_id": channel_map["ntrode_id"],
                    }
                )
                electrode_counter_probe += 1
    return expected_rows


def _get_dio_prefix(io: SpikeGadgetsRawIO, stream_name: str) -> str:
    for channel_name in io._mask_channels_ids[stream_name]:
        if "ECU_" in channel_name:
            return "ECU_"
    return ""


def _get_primary_stream_id(context: ValidationContext) -> str:
    return "ECU_analog" if context.behavior_only else "trodes"


def _get_included_device_metadata_paths() -> list[Path]:
    package_dir = Path(__file__).parent.resolve()
    device_folder = package_dir / "device_metadata"
    return list(device_folder.rglob("*.yml"))


def _resolve_rec_filepaths(
    *,
    rec_filepaths: list[Path] | list[str] | None,
    data_path: Path | str | None,
    metadata_filepath: Path,
) -> list[Path]:
    if rec_filepaths is not None:
        rec_paths = [Path(path) for path in rec_filepaths]
        if len(rec_paths) == 0:
            raise ValueError("At least one rec file path is required.")
        for path in rec_paths:
            if not path.exists():
                raise FileNotFoundError(path)
        return rec_paths

    if data_path is None:
        raise ValueError(
            "Either rec_filepaths or data_path must be provided for validation."
        )

    data_root = Path(data_path)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    session_date, session_animal = _parse_session_from_metadata_filepath(metadata_filepath)
    file_info = get_file_info(data_root)
    session_df = file_info[
        (file_info["date"] == session_date)
        & (file_info["animal"] == session_animal)
        & (file_info["file_extension"] == ".rec")
    ]
    rec_paths = [Path(path) for path in session_df["full_path"].tolist()]
    if len(rec_paths) == 0:
        raise FileNotFoundError(
            "No rec files found for session "
            f"{session_date}_{session_animal} under {data_root}"
        )
    return rec_paths


def _parse_session_from_metadata_filepath(metadata_filepath: Path) -> tuple[int, str]:
    parts = metadata_filepath.stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Metadata file name {metadata_filepath.name!r} does not match expected pattern."
        )
    try:
        session_date = int(parts[0])
    except ValueError as exc:
        raise ValueError(
            f"Metadata file name {metadata_filepath.name!r} does not begin with YYYYMMDD."
        ) from exc
    session_animal = parts[1]
    return session_date, session_animal


def _get_rec_timestamps(context: ValidationContext, stream_id: str) -> np.ndarray:
    iterator = RecFileDataChunkIterator(
        [str(path) for path in context.rec_filepaths],
        stream_id=stream_id,
        behavior_only=context.behavior_only,
    )
    return np.asarray(iterator.timestamps)


def _compare_1d_arrays(
    *,
    expected: np.ndarray,
    actual: np.ndarray,
    atol: float,
) -> dict[str, Any]:
    messages: list[str] = []
    if expected.shape != actual.shape:
        return {
            "messages": [
                f"shape mismatch: expected {expected.shape}, found {actual.shape}"
            ],
            "mismatch_count": 1,
            "max_abs_error": 0.0,
        }

    abs_error = np.abs(actual - expected)
    mismatch_mask = abs_error > atol
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    if mismatch_count > 0:
        first_index = int(np.flatnonzero(mismatch_mask)[0])
        messages.append(
            f"mismatch_count={mismatch_count}, first_mismatch_index={first_index}"
        )
    return {
        "messages": messages,
        "mismatch_count": mismatch_count,
        "max_abs_error": float(np.max(abs_error)) if abs_error.size > 0 else 0.0,
    }


def _compare_chunked_time_series(
    *,
    expected_source: RecFileDataChunkIterator,
    actual_source: Any,
    n_rows: int,
    n_columns: int,
    chunk_size: int,
    atol: float,
) -> dict[str, Any]:
    messages: list[str] = []
    actual_shape = actual_source.shape
    if len(actual_shape) != 2 or actual_shape != (n_rows, n_columns):
        return {
            "messages": [
                f"shape mismatch: expected {(n_rows, n_columns)}, found {actual_shape}"
            ],
            "mismatch_count": 1,
            "max_abs_error": 0.0,
        }

    mismatch_count = 0
    max_abs_error = 0.0
    first_mismatch: tuple[int, int] | None = None
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        expected_chunk = np.asarray(
            expected_source._get_data((slice(start, stop), slice(0, n_columns)))
        )
        actual_chunk = np.asarray(actual_source[start:stop, :])
        abs_error = np.abs(actual_chunk.astype(np.float64) - expected_chunk.astype(np.float64))
        mismatch_mask = abs_error > atol
        this_mismatch_count = int(np.count_nonzero(mismatch_mask))
        if this_mismatch_count > 0 and first_mismatch is None:
            first = np.argwhere(mismatch_mask)[0]
            first_mismatch = (int(start + first[0]), int(first[1]))
        mismatch_count += this_mismatch_count
        if abs_error.size > 0:
            max_abs_error = max(max_abs_error, float(np.max(abs_error)))

    if mismatch_count > 0:
        messages.append(
            f"mismatch_count={mismatch_count}, first_mismatch={first_mismatch}"
        )
    return {
        "messages": messages,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
    }


def _make_check(
    *,
    name: str,
    passed: bool,
    details: str,
    mismatch_count: int = 0,
    max_abs_error: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "details": details,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
    }


def _finalize_check(
    name: str,
    mismatches: list[str],
    *,
    mismatch_count: int | None = None,
    max_abs_error: float = 0.0,
) -> dict[str, Any]:
    if mismatch_count is None:
        mismatch_count = len(mismatches)
    return _make_check(
        name=name,
        passed=len(mismatches) == 0,
        details="; ".join(mismatches) if mismatches else "OK",
        mismatch_count=mismatch_count,
        max_abs_error=max_abs_error,
    )
