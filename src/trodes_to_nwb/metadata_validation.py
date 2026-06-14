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
    # This code converts date_of_birth to an ISO-8601 string for schema checking.
    metadata_content = copy.deepcopy(metadata) or {}
    # Use .get() so metadata missing `subject` (or a subject missing
    # `date_of_birth`) is reported through the schema validator below rather than
    # raising a KeyError here -- otherwise strict=True conversion would surface a
    # bare KeyError instead of the schema-validation report.
    subject = metadata_content.get("subject")
    if (
        subject
        and subject.get("date_of_birth")
        and type(subject["date_of_birth"]) is datetime.datetime
    ):
        # NOTE: use the instance's own value. `datetime.utcnow()` is a classmethod
        # that ignores the instance and returns the current time, so the previous
        # `.utcnow().isoformat()` validated *today's* date instead of the subject's
        # date of birth.
        subject["date_of_birth"] = subject["date_of_birth"].isoformat()

    schema = _get_json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    metadata_validation_errors = validator.iter_errors(metadata_content)
    errors = []

    for metadata_validation_error in metadata_validation_errors:
        errors.append(metadata_validation_error.message)

    is_valid = len(errors) == 0

    return is_valid, errors
