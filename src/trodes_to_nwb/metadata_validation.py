"""Provides functions for validating user-provided metadata dictionaries against
a predefined JSON schema to ensure completeness and correctness before NWB conversion.
"""

import copy
import datetime
from pathlib import Path

import jsonschema
import yaml


def _get_nwb_json_schema_path() -> str:
    """Get the NWB JSON Schema file path.

    Returns
    -------
    str
        NWB Schema file path.
    """
    return str((Path(__file__).parent / "nwb_schema.json").resolve())


def _get_json_schema() -> str:
    """Get JSON Schema

    Returns
    -------
    str
        JSON Schema content
    """
    json_schema = None
    json_schema_path = _get_nwb_json_schema_path()
    with open(json_schema_path) as stream:
        json_schema = yaml.safe_load(stream)
    return json_schema


def validate(metadata: dict) -> tuple:
    """Validates metadata

    Parameters
    ----------
    metadata : dict
        metadata documenting the particulars of a session

    Returns
    -------
    tuple
        information of the validity of the metadata data and any errors
    """
    assert metadata is not None  # metadata cannot be null
    assert isinstance(metadata, dict)  # cannot proceed if metadata is not a dictionary

    # date_of_birth is set to a datetime by the YAML-to-dict converter.
    # This code converts date_of_birth  to string
    metadata_content = copy.deepcopy(metadata) or {}
    if (
        metadata_content["subject"]
        and metadata_content["subject"]["date_of_birth"]
        and type(metadata_content["subject"]["date_of_birth"]) is datetime.datetime
    ):
        metadata_content["subject"]["date_of_birth"] = (
            metadata_content["subject"]["date_of_birth"].utcnow().isoformat()
        )

    schema = _get_json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    metadata_validation_errors = validator.iter_errors(metadata_content)
    errors = []

    for metadata_validation_error in metadata_validation_errors:
        errors.append(metadata_validation_error.message)

    is_valid = len(errors) == 0

    return is_valid, errors


def _duplicates(values: list) -> list:
    """Return the values that appear more than once, in first-seen order."""
    seen: set = set()
    duplicated: list = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated


def _dict_items(value) -> list[dict]:
    """Return only the ``dict`` elements of a value expected to be a list of dicts.

    Tolerates ``None``, a non-list, or a list with non-dict entries so the
    reference checks degrade to a clean message rather than crashing on
    malformed metadata.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_list(value) -> list:
    """Coerce ``None``/a scalar to a list so a mistakenly-scalar field is iterated."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _sort_key(value):
    """Sort key that tolerates mixed/None reference values for error messages."""
    return (value is None, str(value))


def validate_metadata_references(metadata: dict) -> list[str]:
    """Check cross-references *within* the metadata that the JSON schema cannot
    express, so broken references surface as clear errors up front instead of as
    cryptic failures deep in conversion (issue #147).

    Checks performed:

    - electrode group ids are unique;
    - every electrode group has at least one
      ``ntrode_electrode_group_channel_map`` entry (an unmapped group otherwise
      crashes ``add_electrode_groups`` with a ``NoneType`` error), and every
      ntrode map entry references an electrode group that exists;
    - ``ntrode_id`` values are unique;
    - camera ids are unique, and every ``camera_id`` referenced by ``tasks`` or
      ``associated_video_files`` is defined in ``cameras``.

    Intended to run *after* JSON-schema validation (see ``load_metadata``),
    which already guarantees ``metadata`` is a dict with correctly-typed fields.
    The caller raises on any returned message, so a broken reference fails fast
    -- before the long conversion -- rather than crashing cryptically partway
    through. Broken references are returned as messages (not raised here) so the
    caller can report them all at once. Absent or partial sections (a missing
    key, a ``None`` section, a non-dict list element, or a scalar ``camera_id``
    where a list is expected) are treated as "nothing to cross-check" rather
    than crashing; field *types* themselves are the schema's responsibility, not
    re-validated here, so this is not a substitute for schema validation.
    ``device_type``-vs-probe coverage is intentionally not checked here --
    probes are only required at ``add_electrode_groups``, which validates that
    itself.

    Parameters
    ----------
    metadata : dict
        Parsed, schema-valid session metadata.

    Returns
    -------
    list[str]
        Human-readable error messages; empty if every reference is consistent.
    """
    errors: list[str] = []

    electrode_groups = _dict_items(metadata.get("electrode_groups"))
    electrode_group_ids = [group.get("id") for group in electrode_groups]
    duplicate_group_ids = _duplicates(electrode_group_ids)
    if duplicate_group_ids:
        errors.append(f"Duplicate electrode_groups id(s): {duplicate_group_ids}.")
    electrode_group_id_set = set(electrode_group_ids)

    ntrode_map = _dict_items(metadata.get("ntrode_electrode_group_channel_map"))
    mapped_group_ids = [entry.get("electrode_group_id") for entry in ntrode_map]
    dangling = sorted(
        {gid for gid in mapped_group_ids if gid not in electrode_group_id_set},
        key=_sort_key,
    )
    if dangling:
        errors.append(
            "ntrode_electrode_group_channel_map references electrode_group_id(s) "
            f"{dangling} not defined in electrode_groups."
        )
    unmapped = sorted(electrode_group_id_set - set(mapped_group_ids), key=_sort_key)
    if unmapped:
        errors.append(
            f"electrode_groups id(s) {unmapped} have no entry in "
            "ntrode_electrode_group_channel_map (each electrode group needs one)."
        )
    duplicate_ntrode_ids = _duplicates([entry.get("ntrode_id") for entry in ntrode_map])
    if duplicate_ntrode_ids:
        errors.append(
            "Duplicate ntrode_id(s) in ntrode_electrode_group_channel_map: "
            f"{duplicate_ntrode_ids}."
        )

    cameras = _dict_items(metadata.get("cameras"))
    camera_ids = [camera.get("id") for camera in cameras]
    duplicate_camera_ids = _duplicates(camera_ids)
    if duplicate_camera_ids:
        errors.append(f"Duplicate camera id(s): {duplicate_camera_ids}.")
    camera_id_set = set(camera_ids)
    for task in _dict_items(metadata.get("tasks")):
        # camera_id should be a list, but tolerate a scalar (a common mistake)
        # so a bad reference is still reported rather than silently skipped.
        for camera_id in _as_list(task.get("camera_id")):
            if camera_id not in camera_id_set:
                errors.append(
                    f"task '{task.get('task_name')}' references camera_id "
                    f"{camera_id} not defined in cameras."
                )
    for video in _dict_items(metadata.get("associated_video_files")):
        camera_id = video.get("camera_id")
        if camera_id is not None and camera_id not in camera_id_set:
            errors.append(
                f"associated_video_files entry '{video.get('name')}' references "
                f"camera_id {camera_id} not defined in cameras."
            )

    return errors
