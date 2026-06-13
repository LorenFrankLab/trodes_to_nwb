import copy
import datetime
from unittest.mock import patch

import pytest
import yaml

from trodes_to_nwb.metadata_validation import (
    _get_nwb_json_schema_path,
    validate,
    validate_metadata_references,
)
from trodes_to_nwb.tests.test_data import test_metadata_dict_samples
from trodes_to_nwb.tests.utils import data_path


def _reference_metadata() -> dict:
    """Minimal, internally-consistent metadata for the reference checks."""
    return {
        "electrode_groups": [{"id": 0}, {"id": 1}],
        "ntrode_electrode_group_channel_map": [
            {"ntrode_id": 1, "electrode_group_id": 0},
            {"ntrode_id": 2, "electrode_group_id": 1},
        ],
        "cameras": [{"id": 0}, {"id": 1}],
        "tasks": [{"task_name": "run", "camera_id": [0, 1]}],
        "associated_video_files": [{"name": "v.h264", "camera_id": 0}],
        "fs_gui_yamls": [{"name": "opto.yaml", "camera_id": 1}],
    }


def test_validate_metadata_references_consistent_is_empty():
    assert validate_metadata_references(_reference_metadata()) == []


def test_validate_metadata_references_unmapped_and_dangling_electrode_groups():
    metadata = _reference_metadata()
    # point one ntrode entry at a nonexistent group -> dangling ref AND leaves
    # electrode group 1 with no map entry (the NoneType-crash case).
    metadata["ntrode_electrode_group_channel_map"][1]["electrode_group_id"] = 99
    errors = validate_metadata_references(metadata)
    assert any("not defined in electrode_groups" in e and "99" in e for e in errors)
    assert any("no entry in" in e and "[1]" in e for e in errors)


def test_validate_metadata_references_bad_camera_id():
    metadata = _reference_metadata()
    metadata["tasks"][0]["camera_id"] = [7]
    metadata["associated_video_files"][0]["camera_id"] = 8
    errors = validate_metadata_references(metadata)
    assert any("camera_id 7" in e for e in errors)
    assert any("camera_id 8" in e for e in errors)


def test_validate_metadata_references_bad_fs_gui_camera_id():
    # fs_gui_yamls camera_id is resolved as a camera device in
    # add_optogenetic_epochs; a dangling id otherwise fails only late in
    # conversion (#147). The schema requires it but cannot check the reference.
    metadata = _reference_metadata()
    metadata["fs_gui_yamls"][0]["camera_id"] = 9  # no such camera
    errors = validate_metadata_references(metadata)
    assert any("fs_gui_yamls" in e and "camera_id 9" in e for e in errors)


def test_validate_metadata_references_duplicate_ids():
    metadata = _reference_metadata()
    metadata["electrode_groups"].append({"id": 0})
    metadata["cameras"].append({"id": 1})
    metadata["ntrode_electrode_group_channel_map"].append(
        {"ntrode_id": 1, "electrode_group_id": 0}
    )
    errors = validate_metadata_references(metadata)
    assert any("Duplicate electrode_groups id" in e for e in errors)
    assert any("Duplicate camera id" in e for e in errors)
    assert any("Duplicate ntrode_id" in e for e in errors)


def test_validate_metadata_references_scalar_camera_id_is_checked():
    # A scalar camera_id (a common mistake; the schema expects a list) must still
    # be validated, not silently skipped by a falsy `0 or []`.
    metadata = _reference_metadata()
    metadata["tasks"][0]["camera_id"] = 0  # scalar, references existing camera 0
    assert validate_metadata_references(metadata) == []
    metadata["tasks"][0]["camera_id"] = 7  # scalar, nonexistent
    assert any("camera_id 7" in e for e in validate_metadata_references(metadata))


def test_validate_metadata_references_tolerates_partial_sections():
    # Absent/partial sections (a None section, a non-dict list element, a
    # string-where-list, a scalar camera_id) must degrade to "nothing to
    # cross-check" rather than crashing -- the checker reports broken references
    # as messages, it does not re-validate field *types* (the schema does that,
    # and runs first). So this returns a list, never a stack trace (#147).
    partial = {
        "electrode_groups": ["not-a-dict"],
        "ntrode_electrode_group_channel_map": "oops",
        "cameras": [5],
        "tasks": [{"task_name": "t", "camera_id": 0}],
        "associated_video_files": None,
    }
    result = validate_metadata_references(partial)
    assert isinstance(result, list)  # did not raise on partial / non-dict shapes


@pytest.mark.parametrize(
    "filename",
    ["20230622_sample_metadata.yml", "20230622_sample_metadataProbeReconfig.yml"],
)
def test_validate_metadata_references_no_false_positives_on_real_metadata(filename):
    # The highest-risk property: a single over-eager check would break every real
    # conversion. The reconfig file legitimately has multiple ntrode entries per
    # electrode group.
    with open(data_path / filename) as stream:
        metadata = yaml.safe_load(stream)
    assert validate_metadata_references(metadata) == []


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
