import yaml

from pynwb import NWBFile
from pynwb.file import Subject
from copy import deepcopy
from datetime import datetime

import pytz
import uuid

from ndx_franklab_novela import CameraDevice, DataAcqDevice
from ndx_franklab_novela import Probe, Shank, ShanksElectrode
from pynwb.ecephys import ElectrodeGroup
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


def add_electrodeGroups(nwbfile,metadata,probe_metadata):
    probe_counter = 0 #used to track global probe ID across groups for naming
    #loop through the electrode groups
    for egroup_metadata in metadata['electrode_groups']:
        #find the probe corresponding to the device type
        probe_meta = None
        for test_meta in probe_metadata:
            if test_meta['probe_type'] == egroup_metadata['device_type']:
                probe_meta = test_meta
        #Build the relevant Probe
        probe_list = []
        
        #make the Probe object
        probe = Probe(
            id=probe_counter,
            name=f'probe {probe_counter}',
            probe_type=probe_meta['probe_type'],
            units=probe_meta['units'],
            probe_description=probe_meta['probe_description'],
            contact_side_numbering=probe_meta['contact_side_numbering'],
            contact_size=probe_meta['contact_size']
        )
        probe_counter += 1
        #add Shanks to Probe
        for shank_meta in probe_meta['shanks']:
            # build the shank and add
            shank = Shank(name=str(shank_meta['shank_id']),)
            # add the shanks electrodes based on metadata
            for electrode_meta in shank_meta['electrodes']:
                shank.add_shanks_electrode(ShanksElectrode(name=str(electrode_meta['id']),
                                                            rel_x=float(electrode_meta['rel_x']),
                                                            rel_y=float(electrode_meta['rel_y']),
                                                            rel_z=float(electrode_meta['rel_z']),))
            #add the shank to the probe
            probe.add_shank(shank)          
        #make the electrode group
        nwbfile.add_electrode_group(ElectrodeGroup(name=str(egroup_metadata['id']),
                                                description=egroup_metadata['description'],
                                                location=egroup_metadata['targeted_location'],
                                                device=probe,))
