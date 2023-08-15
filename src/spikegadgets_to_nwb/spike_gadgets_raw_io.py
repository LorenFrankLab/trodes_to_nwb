# TODO use neo.rawio.SpikeGadgetsRawIO instead of this file when it is available in neo
# see https://github.com/NeuralEnsemble/python-neo/pull/1303

from neo.rawio.baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)  # TODO the import location was updated for this notebook

import numpy as np

from xml.etree import ElementTree


class SpikeGadgetsRawIO(BaseRawIO):
    extensions = ["rec"]
    rawmode = "one-file"

    def __init__(self, filename="", selected_streams=None):
        """
        Class for reading spikegadgets files.
        Only continuous signals are supported at the moment.

        Initialize a SpikeGadgetsRawIO for a single ".rec" file.

        Args:
            filename: str
                The filename
            selected_streams: None, list, str
                sublist of streams to load/expose to API
                useful for spikeextractor when one stream only is needed.
                For instance streams = ['ECU', 'trodes']
                'trodes' is name for ephy channel (ntrodes)
        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.selected_streams = selected_streams

    def _source_name(self):
        return self.filename

    def _produce_ephys_channel_ids(self, n_total_channels, n_channels_per_chip):
        """Compute the channel ID labels
        The ephys channels in the .rec file are stored in the following order:
        hwChan ID of channel 0 of first chip, hwChan ID of channel 0 of second chip, ..., hwChan ID of channel 0 of Nth chip,
        hwChan ID of channel 1 of first chip, hwChan ID of channel 1 of second chip, ..., hwChan ID of channel 1 of Nth chip,
        ...
        So if there are 32 channels per chip and 128 channels (4 chips), then the channel IDs are:
        0, 32, 64, 96, 1, 33, 65, 97, ..., 128
        See also: https://github.com/NeuralEnsemble/python-neo/issues/1215
        """
        x = []
        for k in range(n_channels_per_chip):
            x.append(
                [
                    k + i * n_channels_per_chip
                    for i in range(int(n_total_channels / n_channels_per_chip))
                ]
            )
        return [item for sublist in x for item in sublist]

    def _parse_header(self):
        # parse file until "</Configuration>"
        header_size = None
        with open(self.filename, mode="rb") as f:
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
        gconf = sr = root.find("GlobalConfiguration")
        hconf = root.find("HardwareConfiguration")
        sconf = root.find("SpikeConfiguration")

        # unix time in milliseconds at creation
        self.system_time_at_creation = gconf.attrib["systemTimeAtCreation"].strip()
        self.timestamp_at_creation = gconf.attrib["timestampAtCreation"].strip()
        # convert to python datetime object
        # dt = datetime.datetime.fromtimestamp(int(self.system_time_at_creation) / 1000.0)

        self._sampling_rate = float(hconf.attrib["samplingRate"])
        num_ephy_channels = int(hconf.attrib["numChannels"])
        num_chan_per_chip = int(sconf.attrib["chanPerChip"])

        # explore sub stream and count packet size
        # first bytes is 0x55
        packet_size = 1
        device_bytes = {}
        for device in hconf:
            device_name = device.attrib["name"]
            num_bytes = int(device.attrib["numBytes"])
            device_bytes[device_name] = packet_size
            packet_size += num_bytes

        # timestamps 4 uint32
        self._timestamp_byte = packet_size
        packet_size += 4
        assert (
            "sysTimeIncluded" not in hconf.attrib
        ), "sysTimeIncluded not supported yet"
        # if sysTimeIncluded, then 8-byte system clock is included after timestamp

        packet_size += 2 * num_ephy_channels

        # read the binary part lazily
        raw_memmap = np.memmap(self.filename, mode="r", offset=header_size, dtype="<u1")

        num_packet = raw_memmap.size // packet_size
        raw_memmap = raw_memmap[: num_packet * packet_size]
        self._raw_memmap = raw_memmap.reshape(-1, packet_size)

        # create signal channels - parallel lists
        stream_ids = []
        signal_streams = []
        signal_channels = []

        self._mask_channels_ids = {}
        self._mask_channels_bytes = {}
        self._mask_channels_bits = {}  # for digital data

        # walk through xml devices
        for device in hconf:
            device_name = device.attrib["name"]
            for channel in device:
                # one device can have streams with different data types,
                # so create a stream_id that differentiates them.
                # users need to be aware of this when using the API
                stream_id = device_name + "_" + channel.attrib["dataType"]

                if "interleavedDataIDByte" in channel.attrib:
                    # TODO LATER: deal with "headstageSensor" which have interleaved
                    continue

                if channel.attrib["dataType"] == "analog":
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._mask_channels_ids[stream_id] = []
                        self._mask_channels_bytes[stream_id] = []
                        self._mask_channels_bits[stream_id] = []

                    name = channel.attrib["id"]
                    chan_id = channel.attrib["id"]
                    dtype = "int16"
                    # TODO LATER : handle gain correctly according the file version
                    units = ""
                    gain = 1.0
                    offset = 0.0
                    signal_channels.append(
                        (
                            name,
                            chan_id,
                            self._sampling_rate,
                            dtype,
                            units,
                            gain,
                            offset,
                            stream_id,
                        )
                    )

                    self._mask_channels_ids[stream_id].append(channel.attrib["id"])

                    num_bytes = device_bytes[device_name] + int(
                        channel.attrib["startByte"]
                    )
                    chan_mask_bytes = np.zeros(packet_size, dtype="bool")
                    chan_mask_bytes[num_bytes] = True
                    chan_mask_bytes[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask_bytes)
                    chan_mask_bits = np.zeros(packet_size * 8, dtype="bool")  # TODO
                    self._mask_channels_bits[stream_id].append(chan_mask_bits)

                elif channel.attrib["dataType"] == "digital":  # handle DIO
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._mask_channels_ids[stream_id] = []
                        self._mask_channels_bytes[stream_id] = []
                        self._mask_channels_bits[stream_id] = []

                    # NOTE store data in signal_channels to make neo happy
                    name = channel.attrib["id"]
                    chan_id = channel.attrib["id"]
                    dtype = "int8"
                    units = ""
                    gain = 1.0
                    offset = 0.0

                    signal_channels.append(
                        (
                            name,
                            chan_id,
                            self._sampling_rate,
                            dtype,
                            units,
                            gain,
                            offset,
                            stream_id,
                        )
                    )

                    self._mask_channels_ids[stream_id].append(channel.attrib["id"])

                    # to handle digital data, need to split the data by bits
                    num_bytes = device_bytes[device_name] + int(
                        channel.attrib["startByte"]
                    )
                    chan_byte_mask = np.zeros(packet_size, dtype="bool")
                    chan_byte_mask[num_bytes] = True
                    self._mask_channels_bytes[stream_id].append(chan_byte_mask)

                    # within the concatenated, masked bytes, mask the bit (flipped order)
                    chan_bit_mask = np.zeros(8 * 1, dtype="bool")
                    chan_bit_mask[int(channel.attrib["bit"])] = True
                    chan_bit_mask = np.flip(chan_bit_mask)
                    self._mask_channels_bits[stream_id].append(chan_bit_mask)

                    # NOTE: _mask_channels_ids, _mask_channels_bytes, and
                    # _mask_channels_bits are parallel lists

        if num_ephy_channels > 0:
            stream_id = "trodes"
            stream_name = stream_id
            signal_streams.append((stream_name, stream_id))
            self._mask_channels_bytes[stream_id] = []

            channel_ids = self._produce_ephys_channel_ids(
                num_ephy_channels, num_chan_per_chip
            )

            chan_ind = 0
            for trode in sconf:
                for schan in trode:
                    chan_id = str(channel_ids[chan_ind])
                    name = "chan" + chan_id

                    # TODO LATER : handle gain correctly according the file version
                    units = ""
                    gain = 1.0
                    offset = 0.0
                    signal_channels.append(
                        (
                            name,
                            chan_id,
                            self._sampling_rate,
                            "int16",
                            units,
                            gain,
                            offset,
                            stream_id,
                        )
                    )

                    chan_mask = np.zeros(packet_size, dtype="bool")
                    num_bytes = packet_size - 2 * num_ephy_channels + 2 * chan_ind
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)

                    chan_ind += 1

        # make mask as array (used in _get_analogsignal_chunk(...))
        self._mask_streams = {}
        for stream_id, l in self._mask_channels_bytes.items():
            mask = np.array(l)
            self._mask_channels_bytes[stream_id] = mask
            self._mask_streams[stream_id] = np.any(mask, axis=0)

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # remove some stream if no wanted
        if self.selected_streams is not None:
            if isinstance(self.selected_streams, str):
                self.selected_streams = [self.selected_streams]
            assert isinstance(self.selected_streams, list)

            keep = np.in1d(signal_streams["id"], self.selected_streams)
            signal_streams = signal_streams[keep]

            keep = np.in1d(signal_channels["stream_id"], self.selected_streams)
            signal_channels = signal_channels[keep]

        # No events channels
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes  channels
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()
        # info from GlobalConfiguration in xml are copied to block and seg annotations
        bl_ann = self.raw_annotations["blocks"][0]
        seg_ann = self.raw_annotations["blocks"][0]["segments"][0]
        for ann in (bl_ann, seg_ann):
            ann.update(gconf.attrib)

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        size = self._raw_memmap.shape[0]
        t_stop = size / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        size = self._raw_memmap.shape[0]
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(
        self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes
    ):
        stream_id = self.header["signal_streams"][stream_index]["id"]

        raw_unit8 = self._raw_memmap[i_start:i_stop]

        num_chan = len(self._mask_channels_bytes[stream_id])
        re_order = None
        if channel_indexes is None:
            # no loop : entire stream mask
            stream_mask = self._mask_streams[stream_id]
        else:
            # accumulate mask
            if isinstance(channel_indexes, slice):
                chan_inds = np.arange(num_chan)[channel_indexes]
            else:
                chan_inds = channel_indexes

                if np.any(np.diff(channel_indexes) < 0):
                    # handle channel are not ordered
                    sorted_channel_indexes = np.sort(channel_indexes)
                    re_order = np.array(
                        [
                            list(sorted_channel_indexes).index(ch)
                            for ch in channel_indexes
                        ]
                    )

            stream_mask = np.zeros(raw_unit8.shape[1], dtype="bool")
            for chan_ind in chan_inds:
                chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
                stream_mask |= chan_mask

        # this copies the data from the memmap into memory
        raw_unit8_mask = raw_unit8[:, stream_mask]
        shape = raw_unit8_mask.shape
        shape = (shape[0], shape[1] // 2)
        # reshape the and retype by view
        raw_unit16 = raw_unit8_mask.flatten().view("int16").reshape(shape)

        if re_order is not None:
            raw_unit16 = raw_unit16[:, re_order]

        return raw_unit16

    def get_analogsignal_timestamps(self, i_start, i_stop):
        raw_uint8 = self._raw_memmap[
            i_start:i_stop, self._timestamp_byte : self._timestamp_byte + 4
        ]
        raw_uint32 = raw_uint8.flatten().view("uint32")
        return raw_uint32

    def get_digitalsignal(self, stream_id, channel_id):
        # stream_id = self.header["signal_streams"][stream_index]["id"]

        # for now, allow only reading the entire dataset
        i_start = 0
        i_stop = self._raw_memmap.shape[0]
        raw_packets = self._raw_memmap[i_start:i_stop]

        channel_index = -1
        for i, chan_id in enumerate(self._mask_channels_ids[stream_id]):
            if chan_id == channel_id:
                channel_index = i
                break
        assert (
            channel_index >= 0
        ), f"channel_id {channel_id} not found in stream {stream_id}"

        # num_chan = len(self._mask_channels_bytes[stream_id])
        # re_order = None
        # if channel_indexes is None:
        #     # no loop : entire stream mask
        #     stream_mask = self._mask_streams[stream_id]
        # else:
        #     # accumulate mask
        #     if isinstance(channel_indexes, slice):
        #         chan_inds = np.arange(num_chan)[channel_indexes]
        #     else:
        #         chan_inds = channel_indexes

        #         if np.any(np.diff(channel_indexes) < 0):
        #             # handle channel are not ordered
        #             sorted_channel_indexes = np.sort(channel_indexes)
        #             re_order = np.array(
        #                 [
        #                     list(sorted_channel_indexes).index(ch)
        #                     for ch in channel_indexes
        #                 ]
        #             )

        #     stream_mask = np.zeros(raw_packets.shape[1], dtype="bool")
        #     for chan_ind in chan_inds:
        #         chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
        #         stream_mask |= chan_mask

        # this copies the data from the memmap into memory
        byte_mask = self._mask_channels_bytes[stream_id][channel_index]
        raw_packets_masked = raw_packets[:, byte_mask]

        bit_mask = self._mask_channels_bits[stream_id][channel_index]
        continuous_dio = np.unpackbits(raw_packets_masked, axis=1)[
            :, bit_mask
        ].flatten()
        change_dir = np.diff(continuous_dio).astype(
            "int8"
        )  # possible values: [-1, 0, 1]
        change_dir_trim = change_dir[change_dir != 0]  # keeps -1 and 1
        change_dir_trim[change_dir_trim == -1] = 0  # change -1 to 0
        # resulting array has 1 when there is a change from 0 to 1,
        # 0 when there is change from 1 to 0

        # track the timestamps when there is a change from 0 to 1 or 1 to 0
        timestamps = self.get_analogsignal_timestamps(i_start, i_stop)
        dio_change_times = timestamps[np.where(change_dir)[0] + 1]

        # insert the first timestamp with the first value
        dio_change_times = np.insert(dio_change_times, 0, timestamps[0])
        change_dir_trim = np.insert(change_dir_trim, 0, continuous_dio[0])

        # if re_order is not None:
        #     raw_unit16 = raw_unit16[:, re_order]

        return dio_change_times, change_dir_trim
