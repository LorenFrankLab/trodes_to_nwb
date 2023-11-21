import logging
import uuid
from copy import deepcopy
from datetime import datetime
from xml.etree import ElementTree

import pandas as pd
import pytz
import trodes_to_nwb.metadata_validation
import yaml
from hdmf.common.table import DynamicTable, VectorData
from ndx_franklab_novela import (
    AssociatedFiles,
    CameraDevice,
    DataAcqDevice,
    Probe,
    Shank,
    ShanksElectrode,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectrodeGroup
from pynwb.file import ProcessingModule, Subject

from trodes_to_nwb import __version__


def load_metadata(
    metadata_path: str, probe_metadata_paths: list[str]
) -> tuple[dict, list[dict]]:
    """loads metadata files as dictionaries

    Parameters
    ----------
    metadata_path : str
        path to file made by yaml generator
    probe_metadata_paths : list[str]
        list of paths to yaml files with information on probe types

    Returns
    -------
    tuple[dict, list[dict]]
        the yaml generator metadata and list of probe metadatas
    """
    metadata = None
    with open(metadata_path, "r") as stream:
        metadata = yaml.safe_load(stream)
    (
        is_metadata_valid,
        metadata_errors,
    ) = trodes_to_nwb.metadata_validation.validate(metadata)
    if not is_metadata_valid:
        logger = logging.getLogger("convert")
        logger.exception("".join(metadata_errors))
    probe_metadata = []
    for path in probe_metadata_paths:
        with open(path, "r") as stream:
            probe_metadata.append(yaml.safe_load(stream))
    if not metadata["associated_files"] is None:
        for file in metadata["associated_files"]:
            file["task_epochs"] = [file["task_epochs"]]
    if not metadata["associated_video_files"] is None:
        for file in metadata["associated_video_files"]:
            file["task_epochs"] = [file["task_epochs"]]
    return metadata, probe_metadata


def initialize_nwb(metadata: dict, first_epoch_config: ElementTree) -> NWBFile:
    """constructs an NWBFile with basic session data

    Parameters
    ----------
    metadata : dict
        metadata from the yaml generator
    first_epoch_config : ElementTree
        xml tree of the first epoch's trodes rec file

    Returns
    -------
    NWBFile
        nwb file with basic session information
    """
    gconf = first_epoch_config.find("GlobalConfiguration")
    nwbfile = NWBFile(
        session_description=metadata["session_description"],
        experimenter=metadata["experimenter_name"],
        lab=metadata["lab"],
        institution=metadata["institution"],
        session_start_time=datetime.fromtimestamp(
            int(gconf.attrib["systemTimeAtCreation"].strip()) / 1000
        ),
        timestamps_reference_time=datetime.fromtimestamp(0, pytz.utc),
        identifier=str(uuid.uuid1()),
        session_id=metadata["session_id"],
        # notes=self.link_to_notes, TODO
        experiment_description=metadata["experiment_description"],
        source_script="trodes_to_nwb " + __version__,
        source_script_file_name="convert.py",
    )
    return nwbfile


def add_subject(nwbfile: NWBFile, metadata: dict) -> None:
    """Inserts Subject information into nwbfile

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    """
    subject_metadata = deepcopy(metadata["subject"])
    # Convert weight to string and add units
    subject_metadata.update({"weight": f"{subject_metadata['weight']} g"})
    # Add the subject information to the file
    nwbfile.subject = Subject(**subject_metadata)


def add_cameras(nwbfile: NWBFile, metadata: dict) -> None:
    """Add information about each camera device to the nwb file

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    """
    # add each camera device to the nwb
    for camera_metadata in metadata["cameras"]:
        nwbfile.add_device(
            CameraDevice(
                name="camera_device " + str(camera_metadata["id"]),
                meters_per_pixel=camera_metadata["meters_per_pixel"],
                manufacturer=camera_metadata["manufacturer"],
                model=camera_metadata["model"],
                lens=camera_metadata["lens"],
                camera_name=camera_metadata["camera_name"],
            )
        )


def add_acquisition_devices(nwbfile: NWBFile, metadata: dict) -> None:
    """Add information about each acquisition device to the nwb file

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    """
    # add each acquisition device to the nwb
    for i, acq_metadata in enumerate(metadata["data_acq_device"]):
        # NOTE: naming convention taken from the old rec_to_nwb"
        nwbfile.add_device(
            DataAcqDevice(
                name=f"dataacq_device{i}",
                system=acq_metadata["system"],
                amplifier=acq_metadata["amplifier"],
                adc_circuit=acq_metadata["adc_circuit"],
            )
        )


def add_electrode_groups(
    nwbfile: NWBFile,
    metadata: dict,
    probe_metadata: list[dict],
    hw_channel_map: dict,
    ref_electrode_map: dict,
) -> None:
    """Adds electrode groups, probes, shanks, and electrodes to nwb file

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    probe_metadata : list[dict]
        list of metadata about each probe type in the experiment
    hw_channel_map: dict
        A dictionary of dictionaries mapping {nwb_group_id->{nwb_electrode_id->hwChan}}
    """

    electrode_df_list = (
        []
    )  # dataframe to track non-default electrode data. add to electrodes table at end
    # loop through the electrode groups
    for probe_counter, egroup_metadata in enumerate(metadata["electrode_groups"]):
        # find correct channel map info
        channel_map = None
        for test_meta in metadata["ntrode_electrode_group_channel_map"]:
            if test_meta["electrode_group_id"] == egroup_metadata["id"]:
                channel_map = test_meta
                break
        # find the probe corresponding to the device type
        probe_meta = None
        for test_meta in probe_metadata:
            if test_meta["probe_type"] == egroup_metadata["device_type"]:
                probe_meta = test_meta
                break
        if probe_meta is None:
            raise FileNotFoundError(
                f"No probe metadata found for {egroup_metadata['device_type']}"
            )
        # Build the relevant Probe
        probe = Probe(
            id=egroup_metadata["id"],
            name=f"probe {egroup_metadata['id']}",
            probe_type=probe_meta["probe_type"],
            units=probe_meta["units"],
            probe_description=probe_meta["probe_description"],
            contact_side_numbering=probe_meta["contact_side_numbering"],
            contact_size=probe_meta["contact_size"],
        )
        # make the electrode group with the probe (Do it here so have electrodeGroup object to reference when making electrodes)
        e_group = ElectrodeGroup(
            name=str(egroup_metadata["id"]),
            description=egroup_metadata["description"],
            location=egroup_metadata["targeted_location"],
            device=probe,
        )
        nwbfile.add_electrode_group(e_group)
        # add Shanks to Probe
        electrode_counter_probe = 0
        for shank_counter, shank_meta in enumerate(probe_meta["shanks"]):
            # build the shank and add
            shank = Shank(
                name=str(shank_meta["shank_id"]),
            )
            # add the shank's electrodes based on metadata
            for electrode_meta in shank_meta["electrodes"]:
                shank.add_shanks_electrode(
                    ShanksElectrode(
                        name=str(electrode_meta["id"]),
                        rel_x=float(electrode_meta["rel_x"]),
                        rel_y=float(electrode_meta["rel_y"]),
                        rel_z=float(electrode_meta["rel_z"]),
                    )
                )
                # add the default electrode info to the nwb electrode table
                nwbfile.add_electrode(
                    location=egroup_metadata["targeted_location"],
                    group=e_group,
                    rel_x=float(electrode_meta["rel_x"]),
                    rel_y=float(electrode_meta["rel_y"]),
                    rel_z=float(electrode_meta["rel_z"]),
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    imp=0.0,
                    filtering="None",
                )
                # track additional electrode data
                electrode_df_list.append(
                    pd.DataFrame.from_dict(
                        (
                            {
                                "hwChan": hw_channel_map[egroup_metadata["id"]][
                                    str(electrode_meta["id"])
                                ],
                                "ntrode_id": channel_map["ntrode_id"],
                                "channel_id": electrode_counter_probe,
                                "bad_channel": bool(
                                    electrode_counter_probe
                                    in channel_map["bad_channels"]
                                ),
                                "probe_shank": shank_counter,
                                "probe_electrode": electrode_counter_probe,
                            },
                        )
                    )
                )
                electrode_counter_probe += 1
            # add the shank to the probe
            probe.add_shank(shank)
        # add the completed probe to the nwb as a device
        nwbfile.add_device(probe)
    # add the electrode table information
    extend_electrode_table(nwbfile, pd.concat(electrode_df_list))
    # define the ref electrode for each row in the table
    # (done seperately because depends on indexing group id and channel)
    ref_electrode_id = []
    electrode_table = nwbfile.electrodes.to_dataframe()
    for nwb_group in list(electrode_table["group_name"]):
        # use the refference electrode map and defined electrode table to find index of the reference electrode
        ref_group, ref_electrode = ref_electrode_map[str(nwb_group)]
        ref_electrode_id.append(
            electrode_table.index[
                (electrode_table["group_name"] == ref_group)
                & (electrode_table["probe_electrode"] == ref_electrode)
            ][0]
        )
    # add the ref electrode id list to the electrodes table
    nwbfile.electrodes.add_column(
        name="ref_elect_id",
        description="Experimenter selected reference electrode id",
        data=ref_electrode_id,
    ),


def extend_electrode_table(nwbfile, electrode_df):
    nwbfile.electrodes.add_column(
        name="hwChan",
        description="SpikeGadgets Hardware channel",
        data=list(electrode_df["hwChan"]),
    )
    nwbfile.electrodes.add_column(
        name="ntrode_id",
        description="Experimenter defined ID for this probe",
        data=list(electrode_df["ntrode_id"]),
    )
    nwbfile.electrodes.add_column(
        name="channel_id",
        description="Channel number of electrode within the probe",
        data=list(electrode_df["channel_id"]),
    )
    nwbfile.electrodes.add_column(
        name="bad_channel",
        description="True if noisy or disconnected",
        data=list(electrode_df["bad_channel"]),
    )
    nwbfile.electrodes.add_column(
        name="probe_shank",
        description="The shank of the probe this channel is located on",
        data=list(electrode_df["probe_shank"]),
    )
    nwbfile.electrodes.add_column(
        name="probe_electrode",
        description="the number of this electrode with respect to the probe",
        data=list(electrode_df["probe_electrode"]),
    )


def add_tasks(nwbfile: NWBFile, metadata: dict) -> None:
    """Creates processing module for tasks and adds their metadata info

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    """
    # make a processing module for task data
    nwbfile.add_processing_module(
        ProcessingModule("tasks", "Contains all tasks information")
    )
    # loop through tasks in the metadata and add them
    for i, task_metadata in enumerate(metadata["tasks"]):
        task_name = VectorData(
            name="task_name",
            description="the name of the task",
            data=[task_metadata["task_name"]],
        )
        task_description = VectorData(
            name="task_description",
            description="a description of the task",
            data=[task_metadata["task_description"]],
        )
        camera_id = VectorData(
            name="camera_id",
            description="the ID number of the camera used for video",
            data=[[int(camera_id) for camera_id in task_metadata["camera_id"]]],
        )
        task_epochs = VectorData(
            name="task_epochs",
            description="the temporal epochs where the animal was exposed to this task",
            data=[[int(epoch) for epoch in task_metadata["task_epochs"]]],
        )
        # NOTE: rec_to_nwb checked that this value existed and filed with none otherwise. Do we require this in yaml?
        task_environment = VectorData(
            name="task_environment",
            description="the environment in which the animal performed the task",
            data=[task_metadata["task_environment"]],
        )
        task = DynamicTable(
            name=f"task_{i}",  # NOTE: Do we want this name to match the descriptive name entered?
            description="",
            columns=[
                task_name,
                task_description,
                camera_id,
                task_epochs,
                task_environment,
            ],
        )
        nwbfile.processing["tasks"].add(task)


def add_associated_files(nwbfile: NWBFile, metadata: dict) -> None:
    """Adds associated files processing module. Reads in file referenced in metadata and stores in processing

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    """
    if "associated_files" not in metadata:
        return

    logger = logging.getLogger("convert")
    associated_files = []
    for file in metadata["associated_files"]:
        # read file content
        content = ""
        try:
            with open(file["path"], "r") as open_file:
                content = open_file.read()
        except FileNotFoundError as err:
            logger.info(f"ERROR: associated file {file['path']} does not exist")
            logger.info(str(err))
        except IOError as err:
            logger.info(f"ERROR: Cannot read file at {file['path']}")
            logger.info(str(err))
        # convert task epoch values into strings
        task_epochs = "".join([str(element) + ", " for element in file["task_epochs"]])
        associated_files.append(
            AssociatedFiles(
                name=file["name"],
                description=file["description"],
                content=content,
                task_epochs=task_epochs,
            )
        )
    if "associated_files" not in nwbfile.processing:
        nwbfile.create_processing_module(
            name="associated_files", description="Contains all associated files data"
        )
    nwbfile.processing["associated_files"].add(associated_files)
