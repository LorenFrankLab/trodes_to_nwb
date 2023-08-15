import numpy as np
import os
import pynwb
from spikegadgets_to_nwb.convert_dios import add_dios
from spikegadgets_to_nwb import convert_yaml, convert_rec_header
from spikegadgets_to_nwb.tests.test_convert_rec_header import default_test_xml_tree

MICROVOLTS_PER_VOLT = 1e6
path = os.path.dirname(os.path.abspath(__file__))


def test_add_dios_single_rec():
    # load metadata yml and make nwb file
    metadata_path = path + "/test_data/20230622_sample_metadata.yml"
    probe_metadata = [
        path + "/test_data/tetrode_12.5.yml",
    ]
    metadata, _ = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    try:
        # running on github
        recfile = os.environ.get("DOWNLOAD_DIR") + "/20230622_sample_01_a1.rec"
        rec_to_nwb_file = os.environ.get("DOWNLOAD_DIR") + "/20230622_155936.nwb"
    except (TypeError, FileNotFoundError):
        # running locally
        recfile = path + "/test_data/20230622_sample_01_a1.rec"
        rec_to_nwb_file = path + "/test_data/20230622_155936.nwb"

    add_dios(
        nwbfile,
        [
            recfile,
        ],
        metadata,
    )

    filename = "test_add_dios_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "behavior" in read_nwbfile.processing
        assert "behavioral_events" in read_nwbfile.processing["behavior"].data_interfaces
        expected_dios = ["Poke_1", "Light_1", "Light_2"]
        for name in expected_dios:
            assert name in read_nwbfile.processing["behavior"]["behavioral_events"].time_series

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            for old_dio in read_nwbfile.processing["behavior"]["behavioral_events"].time_series.values():
                current_dio = read_nwbfile.processing["behavior"]["behavioral_events"][old_dio.name]
                # check that timeseries match
                assert np.all(current_dio.data == old_dio.data)
                assert np.all(current_dio.timestamps == old_dio.timestamps)
                assert current_dio.unit == old_dio.unit
                assert current_dio.description == old_dio.description

    os.remove(filename)