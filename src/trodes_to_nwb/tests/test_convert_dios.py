import os

import numpy as np
import pynwb
from trodes_to_nwb.convert_dios import add_dios
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path

from trodes_to_nwb import convert_yaml


def test_add_dios_single_rec():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, _ = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    recfile = [data_path / "20230622_sample_01_a1.rec"]
    rec_to_nwb_file = data_path / "20230622_155936.nwb"  # comparison file

    add_dios(nwbfile, recfile, metadata)

    filename = "test_add_dios_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "behavior" in read_nwbfile.processing
        assert (
            "behavioral_events" in read_nwbfile.processing["behavior"].data_interfaces
        )
        expected_dios = ["Poke_1", "Light_1", "Light_2"]
        for name in expected_dios:
            assert (
                name
                in read_nwbfile.processing["behavior"]["behavioral_events"].time_series
            )

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            for old_dio in old_nwbfile.processing["behavior"][
                "behavioral_events"
            ].time_series.values():
                current_dio = read_nwbfile.processing["behavior"]["behavioral_events"][
                    old_dio.name
                ]
                # check that timeseries match
                np.testing.assert_array_equal(current_dio.data[:], old_dio.data[:])
                assert np.allclose(
                    current_dio.timestamps[:],
                    old_dio.timestamps[:],
                    rtol=0,
                    atol=1.0 / 30000,
                )
                assert current_dio.unit == old_dio.unit
                assert current_dio.description == old_dio.description

    os.remove(filename)


def test_add_dios_two_epoch():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, _ = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    recfile = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    rec_to_nwb_file = data_path / "minirec20230622_.nwb"  # comparison file

    add_dios(nwbfile, recfile, metadata)

    filename = "test_add_dios_two_epoch.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "behavior" in read_nwbfile.processing
        assert (
            "behavioral_events" in read_nwbfile.processing["behavior"].data_interfaces
        )
        expected_dios = ["Poke_1", "Light_1", "Light_2"]
        for name in expected_dios:
            assert (
                name
                in read_nwbfile.processing["behavior"]["behavioral_events"].time_series
            )

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            for old_dio in old_nwbfile.processing["behavior"][
                "behavioral_events"
            ].time_series.values():
                current_dio = read_nwbfile.processing["behavior"]["behavioral_events"][
                    old_dio.name
                ]
                # check that timeseries match
                np.testing.assert_array_equal(current_dio.data, old_dio.data)
                assert np.allclose(
                    current_dio.timestamps[:],
                    old_dio.timestamps[:],
                    rtol=0,
                    atol=1.0 / 30000,
                )
                assert current_dio.unit == old_dio.unit
                assert current_dio.description == old_dio.description

    os.remove(filename)
