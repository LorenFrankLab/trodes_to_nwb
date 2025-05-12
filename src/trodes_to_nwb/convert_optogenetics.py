from typing import List, Tuple

from ndx_optogenetics import (
    ExcitationSource,
    ExcitationSourceModel,
    OpticalFiber,
    OpticalFiberLocationsTable,
    OpticalFiberModel,
    OptogeneticEpochsTable,
    OptogeneticExperimentMetadata,
    OptogeneticVirus,
    OptogeneticViruses,
    OptogeneticVirusInjection,
    OptogeneticVirusInjections,
)
from pynwb import NWBFile


def add_optogenetics(nwbfile: NWBFile, metadata: dict):
    """
    Add optogenetics data to the NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to which the optogenetics data will be added.
    metadata : dict
        Metadata containing information about the optogenetics data.
    """
    if not all(
        [
            x in metadata
            for x in [
                "optogenetic_experiment",
                "virus_injection",
                "optogenetic_source",
                "optical_fiber",
                "optogenetic_stimulation_software",
            ]
        ]
    ):
        # TODO Log lack of metadata
        return

    # Add optogenetic experiment metadata

    virus, virus_injection = make_virus_injecton(
        nwbfile, metadata.get("virus_injection")
    )
    excitation_source = make_optogenetic_source(
        nwbfile, metadata.get("optogenetic_source")
    )
    fiber_table = make_optical_fiber(
        nwbfile, metadata.get("optical_fiber"), excitation_source
    )

    # add them combined metadata to the nwb file
    # Create experiment metadata container
    optogenetic_experiment_metadata = OptogeneticExperimentMetadata(
        optical_fiber_locations_table=fiber_table,
        optogenetic_viruses=virus,
        optogenetic_virus_injections=virus_injection,
        stimulation_software=metadata["optogenetic_stimulation_software"],
    )
    nwbfile.add_lab_meta_data(optogenetic_experiment_metadata)


def make_optogenetic_source(
    nwbfile: NWBFile, source_metadata: dict
) -> ExcitationSource:
    model_metadata = get_optogenetic_source_device(source_metadata["model_name"])
    excitation_source_model = ExcitationSourceModel(
        name=source_metadata["model_name"],
        description=model_metadata["description"],
        manufacturer=model_metadata["manufacturer"],
        illumination_type=model_metadata["illumination_type"],
        wavelength_range_in_nm=model_metadata["wavelength_range_in_nm"],
    )
    excitation_source = ExcitationSource(
        name=source_metadata["name"],
        model=excitation_source_model,
        wavelength_in_nm=source_metadata["wavelength_in_nm"],
        power_in_W=source_metadata["power_in_W"],
        intensity_in_W_per_m2=source_metadata["intensity_in_W_per_m2"],
    )
    nwbfile.add_device(excitation_source_model)
    nwbfile.add_device(excitation_source)
    return excitation_source


def make_optical_fiber(
    nwbfile: NWBFile,
    fiber_metadata: dict,
    excitation_source: ExcitationSource,
) -> OpticalFiber:
    # get device metadata
    fiber_model_metadata = get_fiber_device(fiber_metadata["name"])
    # make the fiber objects
    optical_fiber_model = OpticalFiberModel(
        name=fiber_metadata[
            "name"
        ],  # TODO: decide if should differentiate name and model name
        description=fiber_model_metadata["description"],
        fiber_name=fiber_model_metadata["fiber_name"],
        fiber_model=fiber_model_metadata["fiber_model"],
        manufacturer=fiber_model_metadata["manufacturer"],
        numerical_aperture=fiber_model_metadata["numerical_aperture"],
        core_diameter_in_um=fiber_model_metadata["core_diameter_in_um"],
        active_length_in_mm=fiber_model_metadata["active_length_in_mm"],
        ferrule_name=fiber_model_metadata["ferrule_name"],
        ferrule_diameter_in_mm=fiber_model_metadata["ferrule_diameter_in_mm"],
    )

    optical_fiber = OpticalFiber(
        name=fiber_metadata["name"],
        model=optical_fiber_model,
    )
    # add the fiber devices to the NWB file
    nwbfile.add_device(optical_fiber_model)
    nwbfile.add_device(optical_fiber)

    # make the locations table
    optical_fiber_locations_table = OpticalFiberLocationsTable(
        description="Information about implanted optical fiber locations",
        reference=fiber_metadata["reference"],
    )
    optical_fiber_locations_table.add_row(
        implanted_fiber_description=fiber_metadata["description"],
        location=fiber_metadata["location"],
        hemisphere=fiber_metadata["hemisphere"],
        ap_in_mm=fiber_metadata["ap_in_mm"],
        ml_in_mm=fiber_metadata["ml_in_mm"],
        dv_in_mm=fiber_metadata["dv_in_mm"],
        roll_in_deg=fiber_metadata["roll_in_deg"],
        pitch_in_deg=fiber_metadata["pitch_in_deg"],
        yaw_in_deg=fiber_metadata["yaw_in_deg"],
        excitation_source=excitation_source,
        optical_fiber=optical_fiber,
    )

    return optical_fiber_locations_table


def make_virus_injecton(
    virus_injection_metadata_list: List[dict], device_metadata: List[dict]
) -> Tuple[OptogeneticViruses, OptogeneticVirusInjections]:
    """
    Add virus injection data to the NWB file.

    Parameters
    ----------
    virus_injection_metadata : dict
        Metadata containing information about the virus injection.
    """
    included_viruses = {}
    injections_list = []
    for virus_injection_metadata in virus_injection_metadata_list:
        # get virus "device"
        virus_name = virus_injection_metadata["virus_name"]
        if virus_name in included_viruses:
            virus = included_viruses[virus_name]
        else:
            virus_metadata = get_virus_device(
                virus_name,
                device_metadata=device_metadata,
            )
            # make the virus object
            virus = OptogeneticVirus(
                name=virus_name,
                construct_name=virus_metadata["construct_name"],
                description=virus_metadata["description"],
                manufacturer=virus_metadata["manufacturer"],
                titer_in_vg_per_ml=virus_injection_metadata["titer_in_vg_per_ml"],
            )
            included_viruses[virus_name] = virus

        # validation
        hemisphere = virus_injection_metadata["hemisphere"].lower()
        if hemisphere not in ["left", "right"]:
            raise ValueError(
                f"Invalid hemisphere '{hemisphere}' in virus injection metadata. "
                "Expected 'left' or 'right'."
            )

        # make the injection object referencing the virus
        virus_injection = OptogeneticVirusInjection(
            name=virus_injection_metadata["name"],
            description=virus_injection_metadata["description"],
            hemisphere=hemisphere,
            location=virus_injection_metadata["location"],
            ap_in_mm=virus_injection_metadata["ap_in_mm"],
            ml_in_mm=virus_injection_metadata["ml_in_mm"],
            dv_in_mm=virus_injection_metadata["dv_in_mm"],
            roll_in_deg=virus_injection_metadata["roll_in_deg"],
            pitch_in_deg=virus_injection_metadata["pitch_in_deg"],
            yaw_in_deg=virus_injection_metadata["yaw_in_deg"],
            reference=virus_injection_metadata["reference"],
            virus=virus,
            volume_in_uL=virus_injection_metadata["volume_in_uL"],
        )
        injections_list.append(virus_injection)
    # make the compiled objects
    optogenetic_viruses = OptogeneticViruses(
        optogenetic_virus=list(included_viruses.values())
    )
    optogenetic_virus_injections = OptogeneticVirusInjections(
        optogenetic_virus_injections=injections_list
    )

    return optogenetic_viruses, optogenetic_virus_injections


def get_virus_device(virus_name, device_metadata) -> dict:
    for device in device_metadata:
        if device.get("virus_name", None) == virus_name:
            return device
    raise ValueError(f"Virus with name '{virus_name}' not found in device metadata.")


def get_fiber_device(fiber_name) -> dict:
    # TODO: Implement this function to retrieve the fiber device information
    return {}


def get_optogenetic_source_device(source_name) -> dict:
    # TODO: Implement this function to retrieve the optogenetic source device information
    return {}
