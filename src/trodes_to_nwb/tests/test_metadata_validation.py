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
@patch("jsonschema.Draft7Validator")
def test_verify_validation_called(jsonValidator, getSchema):
    basic_test_data = copy.deepcopy(test_metadata_dict_samples.basic_data)
    basic_test_data["subject"]["date_of_birth"] = datetime.datetime.now().isoformat()
    validate(basic_test_data)
    assert getSchema.call_count == 1
    assert jsonValidator.call_count == 1


def test_date_of_birth_serialized_correctly():
    """Test that a datetime date_of_birth is serialized to its own value,
    not replaced by the current UTC time."""
    basic_test_data = copy.deepcopy(test_metadata_dict_samples.basic_data)
    known_dob = datetime.datetime(2020, 1, 15, 10, 30, 0)
    basic_test_data["subject"]["date_of_birth"] = known_dob

    # Patch the validator to capture the metadata dict passed to it
    with patch(
        "trodes_to_nwb.metadata_validation.jsonschema.Draft7Validator"
    ) as mock_validator:
        mock_validator.return_value.iter_errors.return_value = iter([])
        is_valid, errors = validate(basic_test_data)

    assert is_valid
    assert not errors

    # Verify the validator received the known date_of_birth serialized correctly
    call_args = mock_validator.return_value.iter_errors.call_args[0][0]
    assert call_args["subject"]["date_of_birth"] == known_dob.isoformat()

    # The original datetime should not be mutated
    assert basic_test_data["subject"]["date_of_birth"] == known_dob
