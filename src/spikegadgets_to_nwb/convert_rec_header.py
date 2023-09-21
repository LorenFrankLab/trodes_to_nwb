import logging
from pathlib import Path
from xml.etree import ElementTree

from ndx_franklab_novela import HeaderDevice
from pynwb import NWBFile


def read_header(recfile: Path | str) -> ElementTree.Element:
    """Read xml header from rec file

    Parameters
    ----------
    recfile : Path
        Path to rec file

    Returns
    -------
    ElementTree.Element
        xml header

    Raises
    ------
    ValueError
        If the xml header does not contain '</Configuration>'
    """
    header_size = None
    with open(recfile, mode="rb") as f:
        line = f.readline()
        while line:
            if b"</Configuration>" in line:
                header_size = f.tell()
                break
            line = f.readline()

        if header_size is None:
            raise ValueError(
                "SpikeGadgets: the xml header does not contain '</Configuration>'"
            )

        f.seek(0)
        header_txt = f.read(header_size).decode("utf8")

    return ElementTree.fromstring(header_txt)


def add_header_device(nwbfile: NWBFile, rec_header: ElementTree.Element) -> None:
    """Reads global configuration from rec file and inserts into a header device within the nwbfile

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    recfile : ElementTree.Element
        xml header from rec file
    """

    global_configuration = rec_header.find("GlobalConfiguration")

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
            raise (KeyError(f"Missing yaml metadata for ntrodes {ntrode_id}"))
        elif not len(group) == len(channel_map["map"]):
            raise ValueError(
                f"Ntrode group {ntrode_id} does not contain the number of channels indicated by the metadata yaml"
            )
        else:
            # add this channel map to the validated list
            validated_channel_maps.append(map_number)

    if len(validated_channel_maps) < len(
        metadata["ntrode_electrode_group_channel_map"]
    ):
        raise (IndexError("XML Header contains less ntrodes than the yaml indicates"))

    pass


def make_hw_channel_map(
    metadata: dict, spike_config: ElementTree.Element
) -> dict[dict]:
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
        nwb_group_id = channel_map["electrode_group_id"]
        # make a dictinary for the nwbgroup to map nwb_electrode_id -> hwchan, may not be necessary for probes with multiple ntrode groups per nwb group
        if nwb_group_id not in hw_channel_map:
            hw_channel_map[nwb_group_id] = {}
        # add each nwb_electrode_id to dictionary mapping to its hardware channel
        for config_electrode_id, channel in enumerate(group):
            # find nwb_electrode_id for this stream in the config file
            nwb_electrode_id = channel_map["map"][str(config_electrode_id)]
            hw_channel_map[nwb_group_id][str(nwb_electrode_id)] = channel.attrib[
                "hwChan"
            ]
    return hw_channel_map


def make_ref_electrode_map(
    metadata: dict, spike_config: ElementTree.Element
) -> dict[tuple]:
    """Generates a dictionary mapping an nwb electrode group to its reference electrode tuple(nwb_group_id,electrode_id).
    Values of -1 in the tuple indicate no reference electrode

    Parameters
    ----------
    metadata : dict
        metadata from the yaml generator
    spike_config : xml.etree.ElementTree.Element
        Information from the xml header on ntrode grouping of channels and hwChan info for each
    Returns
    -------
    ref_electrode_map: dict
        A dictionary mapping a nwb_group_id to its ref electrode e.g. {nwb_group_id->(nwb_group_id,nwb_electrode_id)}
    """
    ref_electrode_map = {}  # {nwb_group_id -> ref_id = (nwbb_group_id,electid)}
    # make dictionary for {ntrodeid:nwbid}
    ntrode_id_to_nwb = {}
    for test_meta in metadata["ntrode_electrode_group_channel_map"]:
        ntrode_id_to_nwb[str(test_meta["ntrode_id"])] = str(
            test_meta["electrode_group_id"]
        )

    for group in spike_config:
        if "refNTrodeID" in group.attrib:
            # define the current ntrode group's nwb id
            ntrode_id = group.attrib["id"]
            nwb_group_id = ntrode_id_to_nwb[ntrode_id]
            # find channel map for ref group
            ntrode_ref_group_id = group.attrib["refNTrodeID"]
            ref_channel_map = None
            for test_meta in metadata["ntrode_electrode_group_channel_map"]:
                if str(test_meta["ntrode_id"]) == ntrode_ref_group_id:
                    ref_channel_map = test_meta
                    break
            # get the nwb group and electrode for the reference channel
            ref_group_nwb = ntrode_id_to_nwb[ntrode_ref_group_id]
            ref_electrode_nwb = ref_channel_map["map"][
                str(int(group.attrib["refChan"]) - 1)
            ]  # adjusted because trodes is 1-indexed
            # add it to the map (only need one per group)
            ref_electrode_map[nwb_group_id] = (ref_group_nwb, ref_electrode_nwb)
        else:  # pragma: no cover
            # no reference defined
            ref_electrode_map[nwb_group_id] = (-1, -1)
    return ref_electrode_map


def detect_ptp_from_header(header: ElementTree.ElementTree) -> bool:
    VALID_CAMERA_MODULE_NAMES = ["cameraModule", "./cameraModule"]

    mconf = header.find("ModuleConfiguration")
    ptp_enabled = False
    for smconf in mconf.findall("SingleModuleConfiguration"):
        if smconf.get("moduleName") in VALID_CAMERA_MODULE_NAMES:
            for arg in smconf.findall("Argument"):
                ptp_enabled = "-ptpEnabled" in arg.attrib.values()
                if ptp_enabled:
                    break
            if ptp_enabled:
                break
    logging.getLogger("convert").info("PTP enabled: " + str(ptp_enabled))
    return ptp_enabled
