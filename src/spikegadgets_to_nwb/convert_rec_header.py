from xml.etree import ElementTree
from ndx_franklab_novela import HeaderDevice
from pynwb import NWBFile


def add_header_device(nwbfile: NWBFile, recfile: str) -> None:
    """Reads global configuration from rec file and inserts into a header device within the nwbfile

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    recfile : str
        path to rec file
    """
    # open the rec file and find the header
    header_size = None
    with open(recfile, mode="rb") as f:
        while True:
            line = f.readline()
            if b"</Configuration>" in line:
                header_size = f.tell()
                break

        if header_size is None:
            ValueError(
                "SpikeGadgets: the xml header does not contain '</Configuration>'"
            )

        f.seek(0)
        header_txt = f.read(header_size).decode("utf8")

    # explore xml header
    root = ElementTree.fromstring(header_txt)
    global_configuration = root.find("GlobalConfiguration")

    nwbfile.add_device(
        HeaderDevice(
            name="header_device",
            headstage_serial=global_configuration.attrib["headstageSerial"],
            headstage_smart_ref_on=global_configuration.attrib["headstageSmartRefOn"],
            realtime_mode=global_configuration.attrib["realtimeMode"],
            headstage_auto_settle_on=global_configuration.attrib[
                "headstageAutoSettleOn"
            ],
            timestamp_at_creation=global_configuration.attrib["timestampAtCreation"],
            controller_firmware_version=global_configuration.attrib[
                "controllerFirmwareVersion"
            ],
            controller_serial=global_configuration.attrib["controllerSerial"],
            save_displayed_chan_only=global_configuration.attrib[
                "saveDisplayedChanOnly"
            ],
            headstage_firmware_version=global_configuration.attrib[
                "headstageFirmwareVersion"
            ],
            qt_version=global_configuration.attrib["qtVersion"],
            compile_date=global_configuration.attrib["compileDate"],
            compile_time=global_configuration.attrib["compileTime"],
            file_prefix=global_configuration.attrib["filePrefix"],
            headstage_gyro_sensor_on=global_configuration.attrib[
                "headstageGyroSensorOn"
            ],
            headstage_mag_sensor_on=global_configuration.attrib["headstageMagSensorOn"],
            trodes_version=global_configuration.attrib["trodesVersion"],
            headstage_accel_sensor_on=global_configuration.attrib[
                "headstageAccelSensorOn"
            ],
            commit_head=global_configuration.attrib["commitHead"],
            system_time_at_creation=global_configuration.attrib["systemTimeAtCreation"],
            file_path=global_configuration.attrib["filePath"],
        )
    )


def validate_yaml_header_electrode_map(
    metadata: dict, spike_config: ElementTree.Element
) -> None:
    """checks that the channel and grouping defined by the yaml matches that found in the header file

    Parameters
    ----------
    metadata : dict
        metadata from the yaml generator
    spike_config : xml.etree.ElementTree.Element
        Information from the xml header on ntrode grouping of channels
    """
    # validate every ntrode in header corresponds with egroup in yaml
    validated_channel_maps = []
    for group in spike_config:
        ntrode_id = group.attrib["id"]
        # find appropriate channel map metadata
        channel_map = None
        map_number = None
        for map_number, test_meta in enumerate(
            metadata["ntrode_electrode_group_channel_map"]
        ):
            if str(test_meta["ntrode_id"]) == ntrode_id:
                channel_map = test_meta
                break
        if channel_map is None:
            print(f"ERROR: Missing yaml metadata for ntrodes {ntrode_id}")
        elif not len(group) == len(channel_map["map"]):
            print(
                f"ERROR: ntrode group {ntrode_id} does not contain the number of channels indicated by the metadata yaml"
            )
        else:
            # add this channel map to the validated list
            validated_channel_maps.append(map_number)

    if len(validated_channel_maps) < len(
        metadata["ntrode_electrode_group_channel_map"]
    ):
        print("ERROR: XML Header contains less ntrodes than the yaml indicates")
    print(validated_channel_maps)
    # print(metadata["ntrode_electrode_group_channel_map"])


def make_hw_channel_map(metadata: dict, spike_config: ElementTree.Element) -> dict:
    """Generates the mappings from an electrode id in a electrode group to it's hwChan in the header file

    Parameters
    ----------
    metadata : dict
        metadata from the yaml generator
    spike_config : xml.etree.ElementTree.Element
        Information from the xml header on ntrode grouping of channels and hwChan info for each

    Returns
    -------
    hw_channel_map: dict
        A dictionary of dictionaries mapping {nwb_group_id->{nwb_electrode_id->hwChan}}
    """
    hw_channel_map = {}  # {nwb_group_id->{nwb_electrode_id->hwChan}}
    for group in spike_config:
        ntrode_id = group.attrib["id"]
        # find appropriate channel map metadata
        channel_map = None
        for test_meta in metadata["ntrode_electrode_group_channel_map"]:
            if str(test_meta["ntrode_id"]) == ntrode_id:
                channel_map = test_meta
                break
        if (
            channel_map is None
        ):  # TODO: Expected behavior if channels in the config are not in the yaml metadata?
            continue
        nwb_group_id = channel_map["electrode_group_id"]
        # make a dictinary for the nwbgroup to map nwb_electrode_id -> hwchan, may not be necessary for probes with multiple ntrode groups per nwb group
        if not nwb_group_id in hw_channel_map:
            hw_channel_map[nwb_group_id] = {}
        # add each nwb_electrode_id to dictionary mapping to its hardware channel
        for config_electrode_id, channel in enumerate(group):
            # find nwb_electrode_id for this stream in the config file
            nwb_electrode_id = channel_map["map"][str(config_electrode_id)]
            hw_channel_map[nwb_group_id][str(nwb_electrode_id)] = channel.attrib[
                "hwChan"
            ]
        return hw_channel_map
