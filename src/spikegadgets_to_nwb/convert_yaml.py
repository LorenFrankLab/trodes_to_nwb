import yaml
from xml.etree import ElementTree
from spikegadgets_to_nwb import (
    metadata_validation
)

from pynwb import NWBFile, TimeSeries
from pynwb.file import Subject, ProcessingModule
from pynwb.behavior import BehavioralEvents
from pynwb.image import ImageSeries
from hdmf.common.table import DynamicTable, VectorData
from copy import copy
from datetime import datetime
import pandas as pd
import numpy as np
import os
import pytz
import uuid

from ndx_franklab_novela import CameraDevice, DataAcqDevice
from ndx_franklab_novela import Probe, Shank, ShanksElectrode
from ndx_franklab_novela import AssociatedFiles
from pynwb.ecephys import ElectrodeGroup


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
    with open(metadata_path, "r") as stream:
        metadata = yaml.safe_load(stream)
    
    is_metadata_valid, metadata_errors = metadata_validation.validate(metadata)
    if not is_metadata_valid:
        raise ValueError(''.join(metadata_errors))
    
    probe_metadata = []
    for path in probe_metadata_paths:
        with open(path, "r") as stream:
            probe_metadata.append(yaml.safe_load(stream))
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
    subject_metadata = copy(metadata["subject"])
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
    for acq_metadata in metadata["data_acq_device"]:
        # NOTE: naming convention taken from the old rec_to_nwb"
        nwbfile.add_device(
            DataAcqDevice(
                name=f'dataacq_device{acq_metadata["name"]}',
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


def add_dios(nwbfile: NWBFile, metadata: dict) -> None:
    """Adds DIO event information and data to nwb file

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    metadata : dict
        metadata from the yaml generator
    """
    # TODO: pass the dio data and include in this
    # Make a processing module for behavior and add to the nwbfile
    if not "behavior" in nwbfile.processing:
        nwbfile.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )
    # Make Behavioral events object to hold DIO data
    events = BehavioralEvents(name="behavioral_events")
    # Loop through and add timeseries for each one
    dio_metadata = metadata["behavioral_events"]
    for dio_event in dio_metadata:
        events.add_timeseries(
            TimeSeries(
                name=dio_event["name"],
                description=dio_event["description"],
                data=np.array(
                    []
                ),  # TODO: from rec file // self.data[dio_event['description']],
                unit="N/A",
                timestamps=np.array([]),
                # TODO: data, timestamps,
            )
        )
    # add it to your file
    nwbfile.processing["behavior"].add(events)


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
    associated_files = []
    for file in metadata["associated_files"]:
        # read file content
        content = ""
        try:
            with open(file["path"] + file["name"], "r") as open_file:
                content = open_file.read()
        except FileNotFoundError as err:
            print(
                f"ERROR: associated file {file['path']+file['name']} does not exist",
                err,
            )
        except IOError as err:
            print(f"ERROR: Cannot read file at {file['path']+file['name']}", err)
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
    nwbfile.create_processing_module(
        name="associated_files", description="Contains all associated files data"
    )
    nwbfile.processing["associated_files"].add(associated_files)


def add_associated_video_files(
    nwbfile: NWBFile,
    metadata: dict,
    video_directory: str,
    raw_data_path: str,
    convert_timestamps: bool,
) -> None:
    """"""
    # make processing module for video files
    nwbfile.create_processing_module(
        name="video_files", description="Contains all associated video files data"
    )
    # make a behavioral Event object to hold videos
    video = BehavioralEvents(name="video")
    # loop and make an image series for each video file. Add it to the behavioral event object
    for video_metadata in metadata["associated_video_files"]:
        try:
            # PTP active
            # TODO: Read rec data at path below and get the timestamp data
            os.path.join(
                raw_data_path,
                os.path.splitext(video_metadata["name"])[0]
                + ".videoTimeStamps.cameraHWSync",
            )["data"]["HWTimestamp"]
            video_timestamps = None
            is_old_dataset = False
        except FileNotFoundError:
            # old dataset (PTP inactive)
            # TODO: Read rec data at path below and get the timestamp data
            os.path.join(
                raw_data_path,
                os.path.splitext(video_metadata["name"])[0]
                + ".videoTimeStamps.cameraHWFrameCount",
            )["data"]["frameCount"]

            video_timestamps = None
            is_old_dataset = True
        if (not is_old_dataset) or (convert_timestamps):
            # for now, FORCE turn off convert_timestamps for old dataset
            video_timestamps = video_timestamps / 1e9

        video.add_timeseries(
            ImageSeries(
                device=nwbfile.devices[
                    "camera_device " + str(video_metadata["camera_id"])
                ],
                name=video_metadata["name"],
                timestamps=video_timestamps,
                external_file=[os.path.join(video_directory, video_metadata["name"])],
                format="external",
                starting_frame=[0],
                description="video of animal behavior from epoch",
            )
        )
    # add the behavioralEvents object to the video_files processing module to the nwbfile
    nwbfile.processing["video_files"].add(video)
