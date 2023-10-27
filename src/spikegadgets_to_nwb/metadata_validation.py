import copy
import datetime
import os

import jsonschema
import yaml


def _get_nwb_json_schema_path() -> str:
    """Get the NWB JSON Schema file path

    Returns
    -------
    str
        NWB Schema file Path
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    json_schema_file = "/./nwb_schema.json"
    return f"{current_path}{os.path.normpath(json_schema_file)}"


def _get_json_schema() -> str:
    """Get JSON Schema

    Returns
    -------
    str
        JSON Schema content
    """
    json_schema = None
    json_schema_path = _get_nwb_json_schema_path()
    with open(json_schema_path, "r") as stream:
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
