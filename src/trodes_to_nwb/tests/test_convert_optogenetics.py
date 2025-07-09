import numpy as np
from ndx_franklab_novela import CameraDevice
from ndx_optogenetics import (
    ExcitationSource,
    ExcitationSourceModel,
    OpticalFiber,
    OpticalFiberModel,
    OptogeneticEpochsTable,
    OptogeneticVirus,
    OptogeneticViruses,
    OptogeneticVirusInjection,
    OptogeneticVirusInjections,
)
from pynwb import TimeSeries

from trodes_to_nwb import convert, convert_optogenetics, convert_yaml
from trodes_to_nwb.convert_dios import add_dios
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.convert_intervals import add_epochs
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path


def test_add_optogenetic_devices():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    device_paths = convert.get_included_device_metadata_paths()
    metadata, device_metadata = convert_yaml.load_metadata(metadata_path, device_paths)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    convert_optogenetics.add_optogenetics(nwbfile, metadata, device_metadata)
    assert "optogenetic_experiment_metadata" in nwbfile.lab_meta_data
    opto_data = nwbfile.lab_meta_data["optogenetic_experiment_metadata"]
    assert isinstance(opto_data.optogenetic_viruses, OptogeneticViruses)
    assert isinstance(opto_data.optogenetic_viruses["demo_virus_1"], OptogeneticVirus)
    assert isinstance(
        opto_data.optogenetic_virus_injections,
        OptogeneticVirusInjections,
    )
    assert isinstance(
        opto_data.optogenetic_virus_injections["Injection 1"], OptogeneticVirusInjection
    )
    assert isinstance(nwbfile.devices["Omicron LuxX+ 488-100"], ExcitationSourceModel)
    assert isinstance(nwbfile.devices["Omicron LuxX+ Blue"], ExcitationSource)
    assert isinstance(nwbfile.devices["demo fiber device"], OpticalFiberModel)
    assert isinstance(nwbfile.devices["Fiber 1"], OpticalFiber)


def test_add_optogenetic_epochs():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    device_paths = convert.get_included_device_metadata_paths()
    metadata, device_metadata = convert_yaml.load_metadata(metadata_path, device_paths)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    file_info = get_file_info(data_path)
    file_info = file_info[file_info.animal == "sample"]
    file_info = file_info[file_info.date == 20230622]
    rec_dci = RecFileDataChunkIterator(
        file_info[file_info.file_extension == ".rec"].full_path.to_list(),
        stream_id="trodes",
    )
    add_epochs(nwbfile, file_info, rec_dci.neo_io)
    recfile = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    add_dios(nwbfile, recfile, metadata)
    convert_yaml.add_subject(nwbfile, metadata)
    convert_yaml.add_tasks(nwbfile, metadata)
    convert_yaml.add_cameras(nwbfile, metadata)
    convert_optogenetics.add_optogenetics(nwbfile, metadata, device_metadata)
    convert_optogenetics.add_optogenetic_epochs(nwbfile, metadata, data_path)
    # test added info
    assert isinstance(nwbfile.intervals["optogenetic_epochs"], OptogeneticEpochsTable)
    opto_df = nwbfile.intervals["optogenetic_epochs"].to_dataframe()
    np.allclose(opto_df.start_time.values, np.array([1.68747480e09, 1.68747482e09]))
    np.allclose(opto_df.stop_time.values, np.array([1.68747481e09, 1.68747484e09]))
    assert opto_df.stimulation_on.values[0]
    assert opto_df.spatial_filter_region_node_coordinates_in_pixels.values[0].shape == (
        1,
        7,
        2,
    )
    assert isinstance(opto_df.spatial_filter_cameras.values[0][0], CameraDevice)
    assert opto_df.spatial_filter_cameras.values[0][0].name == "camera_device 0"
    assert opto_df.spatial_filter_cameras_cm_per_pixel.values[0][0] == 0.1
    stim_obj = opto_df.stimulus_signal.values[0]
    assert isinstance(stim_obj, TimeSeries)
    assert stim_obj.name == "Light_1"
    assert np.allclose(stim_obj.timestamps, np.array([1.68747480e09, 1.68747483e09]))
