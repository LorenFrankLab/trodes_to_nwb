import os

import pytest
from trodes_to_nwb.metadata_validation import validate
from trodes_to_nwb.tests.test_data.test_metadata_dict_samples import (
    basic_data,
    basic_data_with_optional_arrays,
    basic_ntrode_electrode_group_channel_map,
    data_acq_device_with_no_values,
    empty_data_acq_device,
    empty_device_name,
    empty_experiment_description,
    empty_experimenter_name,
    empty_institution,
    empty_keywords,
    empty_lab,
    empty_session_description,
    empty_session_id,
    empty_subject,
    empty_units,
    invalid_default_header_file_path,
    invalid_raw_data_to_volts,
    invalid_times_period_multiplier,
    keywords_array_with_empty_item,
    not_array_keywords,
    string_as_experimenter_name,
    subject_with_empty_values,
    subject_with_invalid_date,
    subject_with_invalid_sex,
)


@pytest.mark.parametrize("metadata", [(None), ("")])
def test_metadata_validation_only_accepts_right_data_type(metadata):
    with pytest.raises(AssertionError) as e:
        validate(metadata)


@pytest.mark.parametrize("metadata", [(basic_data), (basic_data_with_optional_arrays)])
def test_metadata_validation_verification_on_valid_data(metadata):
    is_valid, errors = validate(metadata)

    assert is_valid, "".join(errors)


@pytest.mark.parametrize(
    "metadata, key, expected",
    [
        (empty_experimenter_name, "experimenter_name", ["[] is too short"]),
        (
            string_as_experimenter_name,
            "experimenter_name",
            [
                "'' is not of type 'array'",
            ],
        ),
        (empty_lab, "lab", ["does not match"]),
        (empty_institution, "institution", ["does not match"]),
        (
            empty_experiment_description,
            "experiment_description",
            ["does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'"],
        ),
        (
            empty_session_description,
            "session_description",
            ["does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'"],
        ),
        (
            empty_session_id,
            "session_id",
            ["does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'"],
        ),
        (
            empty_keywords,
            "keywords",
            [
                "[] is too short",
            ],
        ),
        (
            keywords_array_with_empty_item,
            "keywords",
            [
                "",
            ],
        ),
        (not_array_keywords, "keywords", ["is not of type 'array'"]),
        (
            subject_with_empty_values,
            "subject1",
            [
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "is not one of ['M', 'F', 'U', 'O']",
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "-1 is less than the minimum of 0",
                "does not match '(\\\\d{4}-[01]\\\\d-[0-3]\\\\dT[0-2]\\\\d:[0-5]\\\\d:[0-5]\\\\d\\\\.\\\\d+)|(\\\\d{4}-[01]\\\\d-[0-3]\\\\dT[0-2]\\\\d:[0-5]\\\\d:[0-5]\\\\d)|(\\\\d{4}-[01]\\\\d-[0-3]\\\\dT[0-2]\\\\d:[0-5]\\\\d)'",
            ],
        ),
        (
            empty_subject,
            "subject",
            [
                "'description' is a required property",
                "'genotype' is a required property",
                "'sex' is a required property",
                "'species' is a required property",
                "'subject_id' is a required property",
                "'weight' is a required property",
                "'date_of_birth' is a required property",
            ],
        ),
        (subject_with_invalid_sex, "subject", ["is not one of ['M', 'F', 'U', 'O']"]),
        (
            subject_with_invalid_date,
            "subject",
            [
                "'2023-01-04' does not match '(\\\\d{4}-[01]\\\\d-[0-3]\\\\dT[0-2]\\\\d:[0-5]\\\\d:[0-5]\\\\d\\\\.\\\\d+)|(\\\\d{4}-[01]\\\\d-[0-3]\\\\dT[0-2]\\\\d:[0-5]\\\\d:[0-5]\\\\d)|(\\\\d{4}-[01]\\\\d-[0-3]\\\\dT[0-2]\\\\d:[0-5]\\\\d)'"
            ],
        ),
        (
            data_acq_device_with_no_values,
            "data_acq_device1",
            [
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
                "does not match '^(.|\\\\s)*\\\\S(.|\\\\s)*$'",
            ],
        ),
        (empty_data_acq_device, "data_acq_device", ["[] is too short"]),
        (
            empty_units,
            "units",
            [
                "'analog' is a required property",
                "'behavioral_events' is a required property",
            ],
        ),
        (
            invalid_times_period_multiplier,
            "times_period_multiplier",
            [
                "'a' is not of type 'number'",
            ],
        ),
        (
            invalid_raw_data_to_volts,
            "raw_data_to_volts",
            ["'a' is not of type 'number'"],
        ),
        (
            invalid_default_header_file_path,
            "default_header_file_path",
            ["None is not of type 'string'"],
        ),
        (empty_device_name, "device_name", ["'name' is a required property"]),
        (
            basic_ntrode_electrode_group_channel_map,
            "ntrode_electrode_group_channel_map",
            (
                [
                    "'a' is not of type 'integer'",
                    "'z' is not of type 'integer'",
                    "'z' is not of type 'integer'",
                    "'0' is not of type 'integer'",
                    "'0' is not one of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]",
                    "'t' is not of type 'integer'",
                    "'t' is not one of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]",
                    "'a' is not of type 'integer'",
                    "'a' is not one of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]",
                    "-3 is not one of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]",
                    "-3 is less than the minimum of 0",
                ]
            ),
        ),
    ],
)
def test_metadata_validation_verification_on_invalid_data(metadata, key, expected):
    is_valid, errors = validate(metadata)

    assert len(errors) == len(expected), f"Not all the errors are occurring in - {key}"
    assert not is_valid, f"{key} should be invalid"
    for index, error in enumerate(errors):
        assert (
            expected[index] in error
        ), f"Expected error not found - ${expected[index]}"
