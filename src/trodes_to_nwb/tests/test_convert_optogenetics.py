from ndx_optogenetics import (
    ExcitationSource,
    ExcitationSourceModel,
    OpticalFiber,
    OpticalFiberModel,
    OptogeneticVirus,
    OptogeneticViruses,
    OptogeneticVirusInjection,
    OptogeneticVirusInjections,
)

from trodes_to_nwb import convert, convert_optogenetics, convert_yaml
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
