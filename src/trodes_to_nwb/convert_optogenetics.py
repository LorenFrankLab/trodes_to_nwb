import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ndx_optogenetics import (
    ExcitationSource,
    ExcitationSourceModel,
    OpticalFiber,
    OpticalFiberLocationsTable,
    OpticalFiberModel,
    OptogeneticExperimentMetadata,
    OptogeneticVirus,
    OptogeneticViruses,
    OptogeneticVirusInjection,
    OptogeneticVirusInjections,
)
from pynwb import NWBFile


def add_optogenetics(nwbfile: NWBFile, metadata: dict, device_metadata: List[dict]):
    """
    Add optogenetics data to the NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to which the optogenetics data will be added.
    metadata : dict
        Metadata containing information about the optogenetics data.
    device_metadata : list
        List of dictionaries containing metadata for devices used in the experiment.
    """
    logger = logging.getLogger("convert")
    necessary_metadata = [
        "virus_injection",
        "opto_excitation_source",
        "optical_fiber",
        "optogenetic_stimulation_software",
    ]

    if not (
        all([((x in metadata) and len(metadata[x]) > 0) for x in necessary_metadata])
    ):
        logger.info("No available optogenetic metadata")
        return

    # Add optogenetic experiment metadata

    virus, virus_injection = make_virus_injecton(
        metadata.get("virus_injection"), device_metadata
    )
    excitation_metadata = metadata.get("opto_excitation_source")
    if len(excitation_metadata) > 1:
        raise ValueError(
            "Multiple optogenetic sources are not supported. "
            "Please provide a single optogenetic source "
            "or submit request for new feature."
        )
    excitation_metadata = excitation_metadata[0]
    excitation_source = make_optogenetic_source(
        nwbfile, excitation_metadata, device_metadata
    )
    fiber_table = make_optical_fiber(
        nwbfile, metadata.get("optical_fiber"), excitation_source, device_metadata
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
    nwbfile: NWBFile, source_metadata: dict, device_metadata: List[dict]
) -> ExcitationSource:
    """Create an ExcitationSource object and add it to the NWB file.

    Parameters:
    ----------
        nwbfile (NWBFile): The NWB file to which the excitation source will be added.
        source_metadata (dict): Metadata for the excitation source.
        device_metadata (List[dict]): Metadata for the excitation device used in the experiment.

    Returns:
    -------
        ExcitationSource: The created ExcitationSource object.
    """
    model_metadata = get_optogenetic_source_device(
        source_metadata["model_name"], device_metadata
    )
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
        wavelength_in_nm=float(source_metadata["wavelength_in_nm"]),
        power_in_W=float(source_metadata["power_in_W"]),
        intensity_in_W_per_m2=float(source_metadata["intensity_in_W_per_m2"]),
    )
    nwbfile.add_device(excitation_source_model)
    nwbfile.add_device(excitation_source)
    return excitation_source


def make_optical_fiber(
    nwbfile: NWBFile,
    fiber_metadata_list: dict,
    excitation_source: ExcitationSource,
    device_metadata: List[dict],
) -> OpticalFiber:
    """Create an OpticalFiberLocationsTable and populate it with optical fiber data.

    Parameters:
    ----------
        nwbfile (NWBFile): The NWB file to which the optical fiber locations table will be added.
        fiber_metadata_list (dict): Metadata for the optical fibers.
        excitation_source (ExcitationSource): The excitation source associated with the optical fibers.

    Returns:
    -------
        OpticalFiber: The created OpticalFiber object.
    """
    # make the locations table
    optical_fiber_locations_table = OpticalFiberLocationsTable(
        description="Information about implanted optical fiber locations",
        reference=fiber_metadata_list[0]["reference"],
    )
    added_fiber_models = {}
    for fiber_metadata in fiber_metadata_list:
        model_name = fiber_metadata["hardware_name"]
        if model_name in added_fiber_models:
            optical_fiber_model = added_fiber_models[model_name]
        else:
            # get device metadata
            fiber_model_metadata = get_fiber_device(model_name, device_metadata)
            # make the fiber objects
            optical_fiber_model = OpticalFiberModel(
                name=fiber_metadata["hardware_name"],
                description=fiber_model_metadata["description"],
                fiber_name=fiber_model_metadata["hardware_name"],
                fiber_model=fiber_model_metadata["fiber_model"],
                manufacturer=fiber_model_metadata["manufacturer"],
                numerical_aperture=fiber_model_metadata["numerical_aperture"],
                core_diameter_in_um=fiber_model_metadata["core_diameter_in_um"],
                active_length_in_mm=fiber_model_metadata["active_length_in_mm"],
                ferrule_name=fiber_model_metadata["ferrule_name"],
                ferrule_diameter_in_mm=fiber_model_metadata["ferrule_diameter_in_mm"],
            )
            added_fiber_models[model_name] = optical_fiber_model
            nwbfile.add_device(optical_fiber_model)

        # make the fiber object
        optical_fiber = OpticalFiber(
            name=fiber_metadata["name"],
            model=optical_fiber_model,
        )
        # add the fiber to the NWB file
        nwbfile.add_device(optical_fiber)
        # add the fiber to the locations table
        optical_fiber_locations_table.add_row(
            implanted_fiber_description=fiber_metadata["implanted_fiber_description"],
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
    device_metadata : list
        List of dictionaries containing metadata for virus "devices" used in the experiment.

    Returns
    -------
    Tuple[OptogeneticViruses, OptogeneticVirusInjections]
        A tuple containing the OptogeneticViruses and OptogeneticVirusInjections objects.
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
                titer_in_vg_per_ml=float(
                    virus_injection_metadata["titer_in_vg_per_ml"]
                ),
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
            ap_in_mm=float(virus_injection_metadata["ap_in_mm"]),
            ml_in_mm=float(virus_injection_metadata["ml_in_mm"]),
            dv_in_mm=float(virus_injection_metadata["dv_in_mm"]),
            roll_in_deg=float(virus_injection_metadata["roll_in_deg"]),
            pitch_in_deg=float(virus_injection_metadata["pitch_in_deg"]),
            yaw_in_deg=float(virus_injection_metadata["yaw_in_deg"]),
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


def get_fiber_device(fiber_name, device_metadata) -> dict:
    for device in device_metadata:
        if device.get("hardware_name", None) == fiber_name:
            return device
    raise ValueError(
        f"Optical fiber model with name '{fiber_name}' not found in device metadata."
    )


def get_optogenetic_source_device(source_name, device_metadata) -> dict:
    for device in device_metadata:
        if device.get("model_name", None) == source_name:
            return device
    raise ValueError(
        f"Optogenetic source with name '{source_name}' not found in device metadata."
    )


def add_optogenetic_epochs(
    nwbfile: NWBFile,
    metadata: dict,
    file_dir: str = "",
):
    """
    Add optogenetic epochs to the NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to which the optogenetic epochs will be added.
    metadata : dict
        Metadata containing information about the optogenetic epochs.
    file_dir : str, optional
        Directory appended to the file path given in the metadata. Default is empty string.
    """

    opto_epochs_metadata = metadata.get("fs_gui_yamls", [])
    if len(opto_epochs_metadata) == 0:
        print("No optogenetic epochs found in metadata.")
        return

    from ndx_franklab_novela import FrankLabOptogeneticEpochsTable

    opto_epochs_table = FrankLabOptogeneticEpochsTable(
        name="optogenetic_epochs",
        description="Metadata about optogenetic stimulation parameters per epoch",
    )

    # loop through each fsgui script, which can apply to multiple epochs
    for fs_gui_metadata in opto_epochs_metadata:
        new_rows = compile_opto_entries(
            fs_gui_metadata=fs_gui_metadata,
            nwbfile=nwbfile,
            file_dir=file_dir,
        )
        for row in new_rows:
            opto_epochs_table.add_row(**row)

    nwbfile.add_time_intervals(opto_epochs_table)


def compile_opto_entries(
    fs_gui_metadata: dict,
    nwbfile: NWBFile,
    file_dir: str = "",
) -> List[dict]:
    """
    Compile an entry for the optogenetic epochs table.

    Parameters
    ----------
    opto_epoch_metadata : dict
        Metadata containing information about the optogenetic epochs.
    nwbfile : NWBFile
        The NWB file to which the optogenetic epochs will be added.
    file_dir : str, optional
        Directory appended to the file path given in the metadata. Default is empty string.

    Returns
    -------
    List[dict]
        A list of dictionaries containing the compiled entries for the optogenetic epochs table.
    """
    import yaml

    # load the fs_gui yaml
    fs_gui_path = Path(file_dir) / fs_gui_metadata["name"]
    protocol_metadata = None
    with open(fs_gui_path, "r") as stream:
        protocol_metadata = yaml.safe_load(stream)
    protocol_metadata = {
        x["instance_id"]: x for x in protocol_metadata["nodes"]
    }  # dictionary of instance_ids to matadata

    # Find the opto metadata item
    opto_metadata = [
        x for x in protocol_metadata.values() if "action-type" in x["type_id"]
    ]
    if len(opto_metadata) == 0:
        raise ValueError(f"No opto metadata found in {fs_gui_path}")
    if len(opto_metadata) > 1:
        raise ValueError(f"More than one opto metadata found in {fs_gui_path}")
    opto_metadata = opto_metadata[0]

    epoch_df = nwbfile.epochs.to_dataframe()

    new_rows = []
    # make a new row entry for each epoch this protocol was run
    for epoch in fs_gui_metadata["epochs"]:
        # info about the stimulus and epoch
        epoch_dict = dict(
            pulse_length_in_ms=get_epoch_info_entry(
                opto_metadata, fs_gui_metadata, "pulseLength"
            ),
            number_pulses_per_pulse_train=get_epoch_info_entry(
                opto_metadata, fs_gui_metadata, "nPulses"
            ),
            period_in_ms=get_epoch_info_entry(
                opto_metadata, fs_gui_metadata, "sequencePeriod"
            ),
            number_trains=get_epoch_info_entry(
                opto_metadata, fs_gui_metadata, "nOutputTrains"
            ),
            intertrain_interval_in_ms=get_epoch_info_entry(
                opto_metadata, fs_gui_metadata, "trainInterval"
            ),
            power_in_mW=fs_gui_metadata["power_in_mW"],
            stimulation_on=True,
            start_time=epoch_df.start_time.values[
                epoch - 1
            ],  # get from nwbfile for epoch
            stop_time=epoch_df.stop_time.values[epoch - 1],
            epoch_name=epoch_df.tags.values[epoch - 1][0],
            epoch_number=epoch,
            convenience_code=opto_metadata["nickname"],
            epoch_type="optogenetic",
            stimulus_signal=nwbfile.processing["behavior"]["behavioral_events"][
                fs_gui_metadata["dio_output_name"]
            ],
        )
        # info about the trigger condition
        trigger_dict = dict(
            ripple_filter_on=False,
            ripple_filter_num_above_threshold=-1,
            ripple_filter_threshold_sd=-1,
            ripple_filter_lockout_period_in_samples=-1,
            theta_filter_on=False,
            theta_filter_lockout_period_in_samples=-1,
            theta_filter_phase_in_deg=-1,
            theta_filter_reference_ntrode=-1,
        )
        if "trigger_id" in opto_metadata:
            trigger_id = opto_metadata["trigger_id"]["data"]["value"]
            trigger_metadata = protocol_metadata[trigger_id]
            # ripple trigger
            if "ripple" in trigger_metadata["type_id"]:
                trigger_dict["ripple_filter_on"] = True
                trigger_dict["ripple_filter_num_above_threshold"] = trigger_metadata[
                    "n_above_threshold"
                ]
                trigger_dict["ripple_filter_threshold_sd"] = trigger_metadata[
                    "sd_threshold"
                ]
                trigger_dict["ripple_filter_lockout_period_in_samples"] = opto_metadata[
                    "lockout_time"
                ]
            # theta trigger
            elif "theta" in trigger_metadata["type_id"]:
                trigger_dict["theta_filter_on"] = True
                trigger_dict["theta_filter_lockout_period_in_samples"] = opto_metadata[
                    "lockout_time"
                ]
                trigger_dict["theta_filter_phase_in_deg"] = trigger_metadata[
                    "theta_filter_degrees"
                ]
                trigger_dict["theta_filter_reference_ntrode"] = trigger_metadata[
                    "reference_ntrode"
                ]
        else:
            print("WARNING: Trigger info can not be pulled from statescript")

        # conditions for trigger activation (can be multiple)
        condition_dict = {}
        condition_ids = get_condition_ids(opto_metadata.get("condition_id", None))
        condition_ids.extend(get_condition_ids(opto_metadata.get("filter_id", None)))
        geometry_filter_metadata_list = []
        for condition_id in condition_ids:
            condition_metadata = protocol_metadata[condition_id]

            if condition_metadata["type_id"] == "geometry-filter-type":
                # all geometry filters must be compiled at once per epoch
                # log this one and add with rest at the end
                geometry_filter_metadata_list.append(condition_metadata)
                continue

            elif condition_metadata["type_id"] == "speed-filter-type":
                condition_dict["speed_filter_on"] = True
                condition_dict["speed_filter_threshold_in_cm_per_s"] = (
                    condition_metadata["threshold"]
                )
                condition_dict["speed_filter_on_above_threshold"] = condition_metadata[
                    "threshold_above"
                ]
        geometry_dict = compile_geometry_filters(geometry_filter_metadata_list)

        # add camera information if speed or spatial filter is on
        if "speed_filter_on" in condition_dict or "spatial_filter_on" in geometry_dict:
            print("ADDING CAMERA INFO")
            if (camera_id := fs_gui_metadata.get("camera_id", None)) is None:
                raise ValueError(
                    "Camera ID not found in metadata. "
                    "Please provide a camera_id for speed or spatial filters."
                )
            camera_name = f"camera_device {camera_id}"
            camera_device = nwbfile.devices.get(camera_name, None)
            if camera_device is None:
                raise ValueError(
                    f"Camera device '{camera_name}' not found in NWB file. "
                    "Please ensure the camera is defined in the metadata yaml."
                )
            condition_dict["spatial_filter_cameras"] = [camera_device]
            condition_dict["spatial_filter_cameras_cm_per_pixel"] = [
                camera_device.meters_per_pixel * 100
            ]
        # compile row
        row = {**epoch_dict, **trigger_dict, **condition_dict, **geometry_dict}
        new_rows.append(row)
    return new_rows


def get_epoch_info_entry(
    opto_metadata: dict,
    fs_gui_metadata: dict,
    key: str,
):
    """
    Get the value for a specific key from the opto_metadata or fs_gui_metadata.

    Parameters
    ----------
    opto_metadata : dict
        Metadata containing information about the optogenetic epochs.
    fs_gui_metadata : dict
        Metadata containing information about the fs_gui script.
    key : str
        The key for which to retrieve the value.

    Returns
    -------
    Any
        The value corresponding to the specified key, or None if not found.
    """
    value = opto_metadata.get(key, fs_gui_metadata.get(key, None))
    if value is None:
        raise ValueError(
            f"Key '{key}' not found in either the fsgui yaml script or the experiment "
            + "metadata yaml."
        )
    return value


def compile_geometry_filters(geometry_filter_metadata_list: List[str]) -> dict:
    if len(geometry_filter_metadata_list) == 0:
        return {}

    geometry_dict = {"spatial_filter_on": True}
    geometry_file_path = geometry_filter_metadata_list[0]["trackgeometry"]["filename"]
    target_zones = [
        x["trackgeometry"]["zone_id"] for x in geometry_filter_metadata_list
    ]
    geometry_zones_info = get_geometry_zones_info(geometry_file_path, target_zones)

    n_nodes = [len(data["nodes_x"]) for data in geometry_zones_info.values()]
    max_nodes = max(n_nodes)

    node_data = -1 * np.ones((len(geometry_zones_info), max_nodes, 2))
    n_pixels_x = geometry_filter_metadata_list[0]["cameraWidth"]
    n_pixels_y = geometry_filter_metadata_list[0]["cameraHeight"]
    for i, zone_id in enumerate(target_zones):
        nodes_x = np.array(geometry_zones_info[zone_id]["nodes_x"])
        nodes_y = np.array(geometry_zones_info[zone_id]["nodes_y"])
        node_data[i, : len(nodes_x), 0] = nodes_x * n_pixels_x
        node_data[i, : len(nodes_y), 1] = nodes_y * n_pixels_y

    geometry_dict["spatial_filter_region_node_coordinates_in_pixels"] = node_data

    return geometry_dict


def get_geometry_zones_info(geometry_file_path, target_zones):
    """
    Extracts zone information from a geometry file for specified zones.

    Parameters:
    - geometry_file_path: Path to the geometry file.
    - target_zones: List of zone IDs to extract information for.

    Returns:
    - A dictionary with zone IDs as keys and their respective node relative coordinates.
    """
    zones = {i: {} for i in target_zones}
    import os

    if not os.path.exists(geometry_file_path):
        try:
            from trodes_to_nwb.tests.utils import data_path

            geometry_file_path = Path(data_path) / Path(geometry_file_path).name
            os.path.exists(geometry_file_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Geometry file {geometry_file_path} not found. "
                "Please check the path and try again."
            ) from e

    with open(geometry_file_path, "r", encoding="utf-8") as f:
        zone_id = None
        for line in f:
            if line.startswith("Zone id:"):
                zone_id = int(line.split(":")[1].strip())
                continue
            if zone_id is None or zone_id not in target_zones:
                continue

            if line.startswith("nodes_x"):
                nodes_x = [float(x) for x in line.split(":")[1].strip().split(" ")]
                zones[zone_id]["nodes_x"] = nodes_x
                continue
            if line.startswith("nodes_y"):
                nodes_y = [float(x) for x in line.split(":")[1].strip().split(" ")]
                zones[zone_id]["nodes_y"] = nodes_y
                continue

    return zones


def get_condition_ids(metadata_dict: dict) -> List[str]:
    """
    Recursively extracts condition IDs from a metadata dictionary.

    Parameters:
    ----------
        metadata_dict (dict): A dictionary containing metadata, which may
        include nested children.
    Returns:
    -------
        List[str]: A list of condition IDs extracted from the metadata. Corresponds
        to the extracted keys in the protocol_metadata
    """
    if metadata_dict is None or "data" not in metadata_dict:
        return []
    condition_ids = []
    if len(metadata_dict["children"]):
        for child in metadata_dict["children"]:
            condition_ids.extend(get_condition_ids(child))
    else:
        condition_ids.append(metadata_dict["data"]["value"])
    return condition_ids
