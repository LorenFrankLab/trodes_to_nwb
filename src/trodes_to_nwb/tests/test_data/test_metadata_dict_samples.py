import pytest
import copy

basic_data = {
    "experimenter_name": ["michael jackson"],
    "lab": "Loren Frank Lab",
    "institution": "University of California, San Francisco",
    "experiment_description": "making of thriller",
    "session_description": "make an album like the nutcracker",
    "session_id": "6",
    "keywords": ["best"],
    "subject": {
        "description": "Long-Evans Rat",
        "genotype": "Wild Type",
        "sex": "M",
        "species": "Rattus norvegicus",
        "subject_id": "1",
        "date_of_birth": "2023-07-24T00:00:00.000Z",
        "weight": 100,
    },
    "data_acq_device": [
        {
            "name": "SpikeGadgets",
            "system": "SpikeGadgets",
            "amplifier": "Intan",
            "adc_circuit": "Intan",
        }
    ],
    "cameras": [],
    "tasks": [],
    "associated_files": [],
    "associated_video_files": [],
    "units": {"analog": "1", "behavioral_events": "1"},
    "times_period_multiplier": 1,
    "raw_data_to_volts": 1,
    "default_header_file_path": "epic/michaeljackson/thriller",
    "behavioral_events": [],
    "device": {"name": ["Trodes"]},
    "electrode_groups": [],
    "ntrode_electrode_group_channel_map": [],
}

basic_data_with_optional_arrays = {
    "experimenter_name": ["michael jackson"],
    "lab": "Loren Frank Lab",
    "institution": "University of California, San Francisco",
    "experiment_description": "making of thriller",
    "session_description": "make an album like the nutcracker",
    "session_id": "6",
    "keywords": ["best"],
    "subject": {
        "description": "Long-Evans Rat",
        "genotype": "Wild Type",
        "sex": "M",
        "species": "Rattus norvegicus",
        "subject_id": "1",
        "date_of_birth": "2023-07-24T00:00:00.000Z",
        "weight": 100,
    },
    "data_acq_device": [
        {
            "name": "SpikeGadgets",
            "system": "SpikeGadgets",
            "amplifier": "Intan",
            "adc_circuit": "Intan",
        }
    ],
    "cameras": [
        {
            "id": 10,
            "meters_per_pixel": 1,
            "manufacturer": "Epic Record",
            "model": "555",
            "lens": "MJ lens",
            "camera_name": "MJ cam",
        },
    ],
    "tasks": [],
    "associated_files": [
        {
            "name": "Michael Jackson",
            "description": "Thriller25",
            "path": "Hard work",
            "task_epochs": 0,
        },
        {
            "name": "Michael Jackson2",
            "description": "HIStory",
            "path": "Making/a/statement",
            "task_epochs": 1,
        },
    ],
    "associated_video_files": [
        {
            "name": "Michael Jackson",
            "camera_id": 1,
            "task_epochs": 1,
        }
    ],
    "units": {"analog": "1", "behavioral_events": "1"},
    "times_period_multiplier": 1,
    "raw_data_to_volts": 1,
    "default_header_file_path": "epic/michaeljackson/thriller",
    "behavioral_events": [
        {
            "description": "Din555",
            "name": "M. Joe Jackson",
        }
    ],
    "device": {"name": ["Trodes"]},
    "electrode_groups": [],
    "ntrode_electrode_group_channel_map": [],
}

empty_experimenter_name = copy.deepcopy(basic_data)
empty_experimenter_name["experimenter_name"] = []

string_as_experimenter_name = copy.deepcopy(basic_data)
string_as_experimenter_name["experimenter_name"] = ""

empty_lab = copy.deepcopy(basic_data)
empty_lab["lab"] = ""

empty_institution = copy.deepcopy(basic_data)
empty_institution["institution"] = ""

empty_experiment_description = copy.deepcopy(basic_data)
empty_experiment_description["experiment_description"] = ""

empty_session_description = copy.deepcopy(basic_data)
empty_session_description["session_description"] = ""

empty_session_id = copy.deepcopy(basic_data)
empty_session_id["session_id"] = ""

empty_keywords = copy.deepcopy(basic_data)
empty_keywords["keywords"] = []

keywords_array_with_empty_item = copy.deepcopy(basic_data)
keywords_array_with_empty_item["keywords"] = [""]

not_array_keywords = copy.deepcopy(basic_data)
not_array_keywords["keywords"] = "test"

subject_with_empty_values = copy.deepcopy(basic_data)
subject_with_empty_values["subject"] = {
    "description": "",
    "genotype": "",
    "sex": "",
    "species": "",
    "subject_id": "",
    "date_of_birth": "",
    "weight": -1,
}

empty_subject = copy.deepcopy(basic_data)
empty_subject["subject"] = {}

subject_with_invalid_sex = copy.deepcopy(basic_data)
subject_with_invalid_sex["subject"] = {
    "description": "Long-Evans Rat",
    "genotype": "Wild Type",
    "sex": "m",
    "species": "Rattus norvegicus",
    "subject_id": "1",
    "date_of_birth": "2023-07-24T00:00:00.000Z",
    "weight": 100,
}

subject_with_invalid_date = copy.deepcopy(basic_data)
subject_with_invalid_date["subject"] = {
    "description": "Long-Evans Rat",
    "genotype": "Wild Type",
    "sex": "M",
    "species": "Rattus norvegicus",
    "subject_id": "1",
    "date_of_birth": "2023-01-04",
    "weight": 100,
}

data_acq_device_with_no_values = copy.deepcopy(basic_data)
data_acq_device_with_no_values["data_acq_device"] = [
    {"name": "", "system": "", "amplifier": "", "adc_circuit": ""}
]

empty_data_acq_device = copy.deepcopy(basic_data)
empty_data_acq_device["data_acq_device"] = []

empty_units = copy.deepcopy(basic_data)
empty_units["units"] = {}

invalid_times_period_multiplier = copy.deepcopy(basic_data)
invalid_times_period_multiplier["times_period_multiplier"] = "a"

invalid_raw_data_to_volts = copy.deepcopy(basic_data)
invalid_raw_data_to_volts["raw_data_to_volts"] = "a"

invalid_default_header_file_path = copy.deepcopy(basic_data)
invalid_default_header_file_path["default_header_file_path"] = None

empty_device_name = copy.deepcopy(basic_data)
empty_device_name["device"] = {}

basic_ntrode_electrode_group_channel_map = copy.deepcopy(basic_data)
basic_ntrode_electrode_group_channel_map["ntrode_electrode_group_channel_map"] = [
    {
        "ntrode_id": "a",
        "electrode_group_id": "z",
        "bad_channels": ["z"],
        "map": {"0": "0", "1": "t", "2": "a", "3": -3},
    }
]
