import copy
import datetime
from unittest.mock import MagicMock, patch

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
