import yaml

from pynwb import NWBFile
from pynwb.file import Subject
from copy import deepcopy
from datetime import datetime

import pytz
import uuid

from ndx_franklab_novela import CameraDevice, DataAcqDevice
from pynwb.device import Device


def initialNWB(metadata):
    nwb_file = NWBFile(
                session_description=metadata["session_description"],
                experimenter=metadata["experimenter_name"],
                lab=metadata["lab"],
                institution=metadata["institution"],
                session_start_time=datetime(2000,1,1),#session_start_time, TODO: requires .rec data
                timestamps_reference_time=datetime.fromtimestamp(0, pytz.utc),
                identifier=str(uuid.uuid1()),
                session_id=metadata["session_id"],
                # notes=self.link_to_notes, TODO
                experiment_description=metadata["experiment_description"],
            )
    return nwb_file

def add_subject(nwb_file,metadata):
    subject = Subject()
    subject_metadata = deepcopy(metadata["subject"])
    # Convert weight to string and add units
    subject_metadata.update({"weight": f"{subject_metadata['weight']} g"})
    #Add the subject information to the file
    nwb_file.subject=Subject(**subject_metadata)

def add_cameras(nwb_file,metadata):
    #add each camera device to the nwb
    for camera_metadata in metadata['cameras']:
        nwb_file.add_device(CameraDevice(name=camera_metadata['camera_name'],
                                meters_per_pixel=camera_metadata['meters_per_pixel'],
                                manufacturer=camera_metadata['manufacturer'],
                                model=camera_metadata['model'],
                                lens=camera_metadata['lens'],
                                camera_name=camera_metadata['camera_name']))
        
def add_acqDevices(nwbfile,metadata):
    #add each acquisition device to the nwb
    for acq_metadata in metadata["data_acq_device"]:
        #NOTE: naming convention taken from the old rec_to_nwb"
        nwbfile.add_device(DataAcqDevice(name=f'dataacq_device{acq_metadata["name"]}',
                            system=acq_metadata['system'],
                            amplifier=acq_metadata['amplifier'],
                            adc_circuit=acq_metadata['adc_circuit']))


