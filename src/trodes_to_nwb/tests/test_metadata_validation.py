import copy
import datetime
from unittest.mock import patch

from trodes_to_nwb.metadata_validation import _get_nwb_json_schema_path, validate
from trodes_to_nwb.tests.test_data import test_metadata_dict_samples


def test_path_to_json_schema_is_correct():
    path = _get_nwb_json_schema_path()
    json_schema_file = "nwb_schema.json"

    assert json_schema_file in path


@patch("trodes_to_nwb.metadata_validation._get_json_schema")
@patch("jsonschema.Draft202012Validator")
def test_verify_validation_called(jsonValidator, getSchema):
    basic_test_data = copy.deepcopy(test_metadata_dict_samples.basic_data)
    basic_test_data["subject"]["date_of_birth"] = datetime.datetime.now().isoformat()
    validate(basic_test_data)
    assert getSchema.call_count == 1
    assert jsonValidator.call_count == 1


def test_validate_does_not_raise_when_subject_missing():
    # Metadata with no ``subject`` key must be reported through the normal
    # (is_valid, errors) channel, not crash validate() with a KeyError from the
    # date_of_birth pre-processing. With strict=True conversion now raising on
    # invalid metadata, a KeyError here would bypass the schema report entirely.
    data = copy.deepcopy(test_metadata_dict_samples.basic_data)
    del data["subject"]
    is_valid, errors = validate(data)
    # subject is not in the schema's required list, so the rest being valid
    # leaves the metadata valid -- the point is that validate() does not raise.
    assert is_valid is True
    assert errors == []


def test_validate_does_not_raise_when_date_of_birth_missing():
    # A subject dict without date_of_birth must not crash the date_of_birth
    # pre-processing; date_of_birth is required by the subject sub-schema, so it
    # is reported as a normal validation error instead.
    data = copy.deepcopy(test_metadata_dict_samples.basic_data)
    data["subject"].pop("date_of_birth")
    is_valid, errors = validate(data)
    assert is_valid is False
    assert any("date_of_birth" in error for error in errors)
