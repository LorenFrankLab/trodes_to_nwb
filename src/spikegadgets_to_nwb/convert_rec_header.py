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
