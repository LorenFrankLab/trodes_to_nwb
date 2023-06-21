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

import pandas as pd

def add_electrodeGroups(nwbfile,metadata,probe_metadata):
    probe_counter = 0 #used to track global probe ID across groups for naming
    electrode_counter = 0 #used to track global probe ID across groups for naming
    electrode_df = pd.DataFrame(columns=['hwChan', 'ntrode_id', 'channel_id', 'bad_channel', 'rel_x', 'rel_y',
       'rel_z', 'probe_shank', 'probe_electrode', 'ref_elect_id']) #dataframe to track non-default electrode data. add to electrodes table at end
    #loop through the electrode groups
    for egroup_metadata in metadata['electrode_groups']:
        #find correct channel map info
        channel_map = None
        for test_meta in metadata['ntrode_electrode_group_channel_map']:
            if test_meta['electrode_group_id'] == egroup_metadata['id']:
                channel_map = test_meta                
        #find the probe corresponding to the device type
        probe_meta = None
        for test_meta in probe_metadata:
            if test_meta['probe_type'] == egroup_metadata['device_type']:
                probe_meta = test_meta
        #Build the relevant Probe        
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
        #make the electrode group with the probe (Do it here so have electrodeGroup object to reference when making electrodes)
        e_group = ElectrodeGroup(name=str(egroup_metadata['id']),
                                description=egroup_metadata['description'],
                                location=egroup_metadata['targeted_location'],
                                device=probe,)
        nwbfile.add_electrode_group(e_group)
        #add Shanks to Probe
        electrode_counter_probe = 0
        for shank_counter,shank_meta in enumerate(probe_meta['shanks']):
            # build the shank and add
            shank = Shank(name=str(shank_meta['shank_id']),)
            # add the shank's electrodes based on metadata
            for electrode_meta in shank_meta['electrodes']:
                shank.add_shanks_electrode(ShanksElectrode(name=str(electrode_meta['id']),
                                                            rel_x=float(electrode_meta['rel_x']),
                                                            rel_y=float(electrode_meta['rel_y']),
                                                            rel_z=float(electrode_meta['rel_z']),))
                #add the default electrode info to the nwb electrode table
                nwbfile.add_electrode(x=0.0,y=0.0,z=0.0,imp=0.0,location='None',filtering='None',
                                    group=e_group,id=electrode_counter) #TODO: keep default values from rec_to_nwb?
                #track additional electrode data
                electrode_df = electrode_df.append({'hwChan':None, 'ntrode_id':channel_map['ntrode_id'],
                                                    'channel_id':electrode_counter_probe,
                                                    'bad_channel':bool(electrode_counter_probe in channel_map['bad_channels']),
                                                    'rel_x':electrode_meta['rel_x'], 'rel_y':electrode_meta['rel_y'],
                                                    'rel_z':electrode_meta['rel_z'], 'probe_shank':shank_counter, 
                                                    'probe_electrode':electrode_counter_probe,
                                                    'ref_elect_id':None}, ignore_index=True)
                #TODO: ref_elect_id, hwchan, channel_id==count within the probe?,
                
                electrode_counter += 1
                electrode_counter_probe += 1
            #add the shank to the probe
            probe.add_shank(shank)
        #add the completed probe to the nwb as a device
        nwbfile.add_device(probe)
    #add the electrode table information
    extend_electrode_table(nwbfile, electrode_df)

        
def extend_electrode_table(nwbfile, electrode_df):
    nwbfile.electrodes.add_column(
        name='hwChan',
        description='SpikeGadgets Hardware channel',
        data=list(electrode_df['hwChan'])
    )
    nwbfile.electrodes.add_column(
        name='ntrode_id',
        description='Experimenter defined ID for this probe',
        data=list(electrode_df['ntrode_id'])
    )
    nwbfile.electrodes.add_column(
        name='channel_id',
        description='None',
        data=list(electrode_df['channel_id'])
    )
    nwbfile.electrodes.add_column(
        name='bad_channel',
        description='True if noisy or disconnected',
        data=list(electrode_df['bad_channel'])
    )
    nwbfile.electrodes.add_column(
        name='rel_x',
        description='None',
        data=list(electrode_df['rel_x'])
    )
    nwbfile.electrodes.add_column(
        name='rel_y',
        description='None',
        data=list(electrode_df['rel_y'])
    )
    nwbfile.electrodes.add_column(
        name='rel_z',
        description='None',
        data=list(electrode_df['rel_z'])
    )
    nwbfile.electrodes.add_column(
        name='probe_shank',
        description='The shank of the probe this channel is located on',
        data=list(electrode_df['probe_shank'])
    )
    nwbfile.electrodes.add_column(
        name='probe_electrode',
        description='the number of this electrode with respect to the probe',
        data=list(electrode_df['probe_electrode'])
    )
    nwbfile.electrodes.add_column(
        name='ref_elect_id',
        description='“Experimenter selected reference electrode id”',
        data=list(electrode_df['ref_elect_id'])
    )