"""Custom implementation of a Neo RawIO class for reading SpikeGadgets .rec files.

Handles parsing the header, reading continuous signal chunks (ephys, analog, DIO),
timestamp extraction (Trodes, system clock), and optional interpolation of dropped packets.
Intended as a temporary solution until official support is available in Neo.
"""

# TODO use neo.rawio.SpikeGadgetsRawIO instead of this file when it is available in neo
# see https://github.com/NeuralEnsemble/python-neo/pull/1303

import functools
from typing import List, Optional
from xml.etree import ElementTree

import numpy as np
from neo.rawio.baserawio import (  # TODO the import location was updated for this notebook
    BaseRawIO,
    _event_channel_dtype,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
)
from scipy.stats import linregress

INT_16_CONVERSION = 256
BITS_PER_BYTE = 8
TIMESTAMP_SIZE_BYTES = 4  # uint32
SYSCLOCK_SIZE_BYTES = 8  # int64
EPHYS_SAMPLE_SIZE_BYTES = 2  # int16
EXPECTED_TIMESTAMP_DIFF_DROP = 2  # Indicates a single dropped packet


class SpikeGadgetsRawIO(BaseRawIO):
    extensions = ["rec"]
    rawmode = "one-file"

    def __init__(
        self,
        filename: str = "",
        selected_streams: Optional[list[str] | str] = None,
        interpolate_dropped_packets: bool = False,
    ):
        """
        Initializes the SpikeGadgetsRawIO class for reading SpikeGadgets .rec files.

        Parameters
        ----------
        filename : str, optional
            The path to the .rec file. Default is an empty string.
        selected_streams : list of str or str, optional
            Stream(s) to be loaded. If `None`, loads all available streams. If a string, loads only the specified stream.
            If a list, loads multiple specified streams.
        interpolate_dropped_packets : bool, optional
            If True, enables interpolation for dropped packets in analog data. Default is False.
        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.selected_streams = selected_streams
        self.interpolate_dropped_packets = interpolate_dropped_packets

    def _source_name(self) -> str:
        """
        Returns the filename of the current SpikeGadgetsRawIO instance.

        Returns
        -------
        str
            The filename of the associated .rec file.
        """
        return self.filename

    @staticmethod
    def _produce_ephys_channel_ids(
        n_total_channels: int,
        n_channels_recorded: int,
        n_channels_per_chip: int,
        hw_channels_recorded: List[str] = None,
    ) -> list[int]:
        """Computes the hardware channel IDs for ephys data.

        The ephys channels in the .rec file are stored in the following order:
        hwChan ID of channel 0 of first chip, hwChan ID of channel 0 of second chip, ..., hwChan ID of channel 0 of Nth chip,
        hwChan ID of channel 1 of first chip, hwChan ID of channel 1 of second chip, ..., hwChan ID of channel 1 of Nth chip,
        ...
        So if there are 32 channels per chip and 128 channels (4 chips), then the channel IDs are:
        0, 32, 64, 96, 1, 33, 65, 97, ..., 128

        Parameters
        ----------
        n_total_channels : int
            Total number of ephys channels in the hardware configuration.
        n_channels_recorded : int
            Total number of ephys channels recorded.
        n_channels_per_chip : int
            Number of channels per headstage chip/amplifier.
        hw_channels_recorded : list of str, optional
            List of hardware channel IDs that were actually recorded. If `None`, all channels are assumed
            to be recorded. This is used to filter the returned list if `n_total_channels`
            is not equal to `n_channels_recorded`.

        Returns
        -------
        list[int]
            List of hardware channel IDs in the interleaved order as they
            appear in the data packets. Returns an empty list if
            `n_total_channels` or `n_channels_per_chip` is 0.

        Raises
        ------
        ValueError
            If `n_total_channels` is not a multiple of `n_channels_per_chip`.

        See Also
        --------
        https://github.com/NeuralEnsemble/python-neo/issues/1215 : Discussion
            on SpikeGadgets channel ordering in Neo.
        """
        if n_total_channels == 0 or n_channels_per_chip == 0:
            return []  # Handle edge case of zero channels

        if n_total_channels % n_channels_per_chip != 0:
            raise ValueError(
                "Total number of channels must be a multiple of channels per chip "
                f"({n_total_channels} % {n_channels_per_chip} != 0)"
            )

        x = []
        for k in range(n_channels_per_chip):
            x.append(
                [
                    k + i * n_channels_per_chip
                    for i in range(int(n_total_channels / n_channels_per_chip))
                ]
            )

        channel_names = [item for sublist in x for item in sublist]

        if n_total_channels == n_channels_recorded:
            # case where all channels are recorded, no censoring required
            return channel_names

        if not hw_channels_recorded or len(hw_channels_recorded) != n_channels_recorded:
            raise ValueError(
                "If n_total_channels != n_channels_recorded, "
                "hw_channels_recorded must be provided to censor the returned list."
            )
        return [x for x in channel_names if str(x) in hw_channels_recorded]

    def _parse_header(self):
        """
        Parses the XML header of the SpikeGadgets .rec file to extract important configuration data.

        This includes system time at creation, sampling rate, number of channels, and other configuration
        details from the XML metadata.

        Raises
        ------
        ValueError
            If the header XML is missing or invalid.
        """
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
        gconf = root.find("GlobalConfiguration")
        hconf = root.find("HardwareConfiguration")
        sconf = root.find("SpikeConfiguration")

        # unix time in milliseconds at creation
        self.system_time_at_creation = gconf.attrib["systemTimeAtCreation"].strip()
        self.timestamp_at_creation = gconf.attrib["timestampAtCreation"].strip()
        # convert to python datetime object
        # dt = datetime.datetime.fromtimestamp(int(self.system_time_at_creation) / 1000.0)

        self._sampling_rate = float(hconf.attrib["samplingRate"])
        num_chip_channels = int(
            hconf.attrib["numChannels"]
        )  # number of channels the hardware supports
        num_ephy_channels = num_chip_channels  # number of channels recorded
        # check for agreement with number of channels in xml
        sconf_channels = np.sum([len(x) for x in sconf])
        if sconf_channels < num_ephy_channels:
            # Case: not every channel was saved to recording
            num_ephy_channels = sconf_channels
        if sconf_channels > num_ephy_channels:
            raise ValueError(
                "SpikeGadgets: the number of channels in the spike configuration is larger than the number of channels in the hardware configuration"
            )

        try:
            num_chan_per_chip = int(sconf.attrib["chanPerChip"])
        except KeyError:
            num_chan_per_chip = 32  # default value

        # explore sub stream and count packet size
        # first bytes is 0x55
        packet_size = 1
        device_bytes = {}
        for device in hconf:
            device_name = device.attrib["name"]
            num_bytes = int(device.attrib["numBytes"])
            device_bytes[device_name] = packet_size
            packet_size += num_bytes
        self.sysClock_byte = (
            device_bytes["SysClock"] if "SysClock" in device_bytes else False
        )

        # timestamps 4 uint32
        self._timestamp_byte = packet_size
        packet_size += TIMESTAMP_SIZE_BYTES
        assert (
            "sysTimeIncluded" not in hconf.attrib
        ), "sysTimeIncluded not supported yet"
        # if sysTimeIncluded, then 8-byte system clock is included after timestamp

        packet_size += EPHYS_SAMPLE_SIZE_BYTES * num_ephy_channels

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

        self.multiplexed_channel_xml = {}  # dictionary from id to channel xml
        if "Multiplexed" in device_bytes:
            self._multiplexed_byte_start = device_bytes["Multiplexed"]
        elif "headstageSensor" in device_bytes:
            self._multiplexed_byte_start = device_bytes["headstageSensor"]

        # walk through xml devices
        for device in hconf:
            device_name = device.attrib["name"]
            for channel in device:
                if (
                    device.attrib["name"] in ["Multiplexed", "headstageSensor"]
                    and channel.attrib["dataType"] == "analog"
                ):
                    # the multiplexed analog device has interleaved data from multiple sources
                    # that are sampled at a lower rate.
                    # for each packet,
                    # the interleavedDataIDByte and the interleavedDataIDBit indicate which
                    # channel has an updated value.
                    # the startByte contains the int16 updated value.
                    # if there was no update, use the last value received.
                    # thus, there is a value at every timestamp, but usually it will be the same
                    # as the previous value.
                    # it is assumed that for a given startByte, only one of the
                    # interleavedDataIDByte and interleavedDataIDBit combinations that
                    # use that startByte is active at any given timestamp,
                    # i.e. there should be at most one 1 in the interleavedDataIDByte value
                    # at each timestamp.

                    # the typical mask approach will not work, so store the channel specs
                    # and use them to read the analog data on demand.
                    self.multiplexed_channel_xml[channel.attrib["id"]] = channel
                    continue

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
                        signal_streams.append(
                            (
                                stream_name,
                                stream_id,
                                "",
                            )
                        )
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
                            "",
                        )
                    )

                    self._mask_channels_ids[stream_id].append(channel.attrib["id"])

                    num_bytes_offset = device_bytes[device_name] + int(
                        channel.attrib["startByte"]
                    )
                    chan_mask_bytes = np.zeros(packet_size, dtype="bool")
                    chan_mask_bytes[
                        num_bytes_offset : num_bytes_offset + EPHYS_SAMPLE_SIZE_BYTES
                    ] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask_bytes)
                    chan_mask_bits = np.zeros(
                        packet_size * BITS_PER_BYTE, dtype="bool"
                    )  # TODO
                    self._mask_channels_bits[stream_id].append(chan_mask_bits)

                elif channel.attrib["dataType"] == "digital":  # handle DIO
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append(
                            (
                                stream_name,
                                stream_id,
                                "",
                            )
                        )
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
                            "",
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
                    chan_bit_mask = np.zeros(BITS_PER_BYTE * 1, dtype="bool")
                    chan_bit_mask[int(channel.attrib["bit"])] = True
                    chan_bit_mask = np.flip(chan_bit_mask)
                    self._mask_channels_bits[stream_id].append(chan_bit_mask)

                    # NOTE: _mask_channels_ids, _mask_channels_bytes, and
                    # _mask_channels_bits are parallel lists

        if num_ephy_channels > 0:
            stream_id = "trodes"
            stream_name = stream_id
            signal_streams.append(
                (
                    stream_name,
                    stream_id,
                    "",
                )
            )
            self._mask_channels_bytes[stream_id] = []

            # get list of all hardware channels recorded
            hw_channels_recorded = []
            for trode in sconf:
                for schan in trode:
                    hw_channels_recorded.append(schan.attrib["hwChan"])

            channel_ids = self._produce_ephys_channel_ids(
                num_chip_channels,
                num_ephy_channels,
                num_chan_per_chip,
                hw_channels_recorded,
            )

            chan_ind = 0
            for chan_ind in range(len(channel_ids)):
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
                        "",
                    )
                )

                chan_mask = np.zeros(packet_size, dtype="bool")
                num_bytes_offset = (
                    packet_size
                    - (EPHYS_SAMPLE_SIZE_BYTES * num_ephy_channels)
                    + (EPHYS_SAMPLE_SIZE_BYTES * chan_ind)
                )
                chan_mask[
                    num_bytes_offset : num_bytes_offset + EPHYS_SAMPLE_SIZE_BYTES
                ] = True
                self._mask_channels_bytes[stream_id].append(chan_mask)

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

        # initialize interpolate index as none so can check if it has been set in a trodes timestamps call
        self.interpolate_index = None

        # initialize systime parameters as empty dict so can check if they have been set in a get_regressed_systime call
        self.regressed_systime_parameters = {}

        self._generate_minimal_annotations()
        # info from GlobalConfiguration in xml are copied to block and seg annotations
        bl_ann = self.raw_annotations["blocks"][0]
        seg_ann = self.raw_annotations["blocks"][0]["segments"][0]
        for ann in (bl_ann, seg_ann):
            ann.update(gconf.attrib)

    def _segment_t_start(self, block_index: int, seg_index: int) -> float:
        return 0.0

    def _segment_t_stop(self, block_index: int, seg_index: int) -> float:
        size = self._raw_memmap.shape[0]
        t_stop = size / self._sampling_rate
        return t_stop

    def _get_signal_size(
        self, block_index: int, seg_index: int, stream_index: int
    ) -> int:
        """
        Returns the size of the signal for a given stream.

        Parameters
        ----------
        block_index : int
            The block index.
        seg_index : int
            The segment index.
        stream_index : int
            The stream index.

        Returns
        -------
        int
            The size of the signal in the specified block and segment.

        Raises
        ------
        ValueError
            If interpolation index is not set but interpolation is enabled.
        """
        if self.interpolate_dropped_packets and self.interpolate_index is None:
            raise ValueError("interpolate_index must be set before calling this")
        size = self._raw_memmap.shape[0]
        return size

    def _get_signal_t_start(
        self, block_index: int, seg_index: int, stream_index: int
    ) -> float:
        return 0.0

    def _get_analogsignal_chunk(
        self,
        block_index: int,
        seg_index: int,
        i_start: int,
        i_stop: int,
        stream_index: int,
        channel_indexes: Optional[int | np.ndarray | slice] = None,
    ) -> np.ndarray:
        """
        Returns a chunk of the analog signal data from the .rec file.

        Parameters
        ----------
        block_index : int
            The block index.
        seg_index : int
            The segment index.
        i_start : int
            The start index for the chunk.
        i_stop : int
            The stop index for the chunk.
        stream_index : int
            The index of the signal stream to be retrieved.
        channel_indexes : int, np.ndarray, slice or None, optional
            The specific channel(s) to retrieve from the stream. If `None`, retrieves all channels.

        Returns
        -------
        np.ndarray
            A NumPy array containing the requested chunk of the analog signal data.
        """
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
        raw_unit16 = raw_unit8_mask.reshape(-1).view("int16").reshape(shape)

        if re_order is not None:
            raw_unit16 = raw_unit16[:, re_order]

        if stream_id == "ECU_analog":
            # automatically include the interleaved analog signals:
            analog_multiplexed_data = self.get_analogsignal_multiplexed()[
                i_start:i_stop, :
            ]
            raw_unit16 = np.concatenate((raw_unit16, analog_multiplexed_data), axis=1)

        return raw_unit16

    def get_analogsignal_timestamps(self, i_start: int, i_stop: int) -> np.ndarray:
        if not self.interpolate_dropped_packets:
            # no interpolation
            raw_uint8 = self._raw_memmap[
                i_start:i_stop,
                self._timestamp_byte : self._timestamp_byte + TIMESTAMP_SIZE_BYTES,
            ]
            raw_uint32 = (
                raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
            )
            return raw_uint32

        if self.interpolate_dropped_packets and self.interpolate_index is None:
            # first call in a interpolation iterator, needs to find the dropped packets
            # has to run through the entire file to find missing packets
            raw_uint8 = self._raw_memmap[
                :, self._timestamp_byte : self._timestamp_byte + TIMESTAMP_SIZE_BYTES
            ]
            raw_uint32 = (
                raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
            )
            self.interpolate_index = np.where(
                np.diff(raw_uint32) == EXPECTED_TIMESTAMP_DIFF_DROP
            )[
                0
            ]  # find locations of single dropped packets
            self._interpolate_raw_memmap()  # interpolates in the memmap

        # subsequent calls in a interpolation iterator don't remake the interpolated memmap, start here
        if i_stop is None:
            i_stop = self._raw_memmap.shape[0]
        raw_uint8 = self._raw_memmap[
            i_start:i_stop,
            self._timestamp_byte : self._timestamp_byte + TIMESTAMP_SIZE_BYTES,
        ]
        raw_uint32 = raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
        # add +1 to the inserted locations
        inserted_locations = np.array(self._raw_memmap.inserted_locations) - i_start + 1
        inserted_locations = inserted_locations[
            (inserted_locations >= 0) & (inserted_locations < i_stop - i_start)
        ]
        if not len(inserted_locations) == 0:
            raw_uint32[inserted_locations] += 1
        return raw_uint32

    def get_sys_clock(self, i_start: int, i_stop: int) -> np.ndarray:
        if not self.sysClock_byte:
            raise ValueError("sysClock not available")
        if i_stop is None:
            i_stop = self._raw_memmap.shape[0]
        raw_uint8 = self._raw_memmap[
            i_start:i_stop,
            self.sysClock_byte : self.sysClock_byte + SYSCLOCK_SIZE_BYTES,
        ]
        raw_uint64 = raw_uint8.view(dtype=np.int64).reshape(-1)
        return raw_uint64

    @functools.lru_cache(maxsize=2)
    def get_analogsignal_multiplexed(
        self, channel_names: Optional[list[str]] = None
    ) -> np.ndarray:
        """
        Retrieves multiplexed analog signal data.

        If `channel_names` is provided, it retrieves data for the specified channels.
        Otherwise, it fetches all multiplexed channels.

        Parameters
        ----------
        channel_names : list of str, optional
            A list of channel names to retrieve. If `None`, fetches all channels.

        Returns
        -------
        np.ndarray
            A NumPy array containing the multiplexed analog signal data.

        Raises
        ------
        ValueError
            If any specified `channel_names` are not found in the file.
        """
        print("compute multiplex cache", self.filename)
        if channel_names is None:
            # read all multiplexed channels
            channel_names = list(self.multiplexed_channel_xml.keys())
        else:
            for ch_name in channel_names:
                if ch_name not in self.multiplexed_channel_xml:
                    raise ValueError(f"Channel name '{ch_name}' not found in file.")

        # because of the encoding scheme, it is easiest to read all the data in sequence
        # one packet at a time
        num_packet = self._raw_memmap.shape[0]
        analog_multiplexed_data = np.empty(
            (num_packet, len(channel_names)), dtype=np.int16
        )

        # precompute the static data offsets
        data_offsets = np.empty((len(channel_names), 3), dtype=int)
        for j, ch_name in enumerate(channel_names):
            ch_xml = self.multiplexed_channel_xml[ch_name]
            data_offsets[j, 0] = int(
                self._multiplexed_byte_start + int(ch_xml.attrib["startByte"])
            )
            data_offsets[j, 1] = int(ch_xml.attrib["interleavedDataIDByte"])
            data_offsets[j, 2] = int(ch_xml.attrib["interleavedDataIDBit"])
        interleaved_data_id_byte_values = self._raw_memmap[:, data_offsets[:, 1]]
        interleaved_data_id_bit_values = (
            interleaved_data_id_byte_values >> data_offsets[:, 2]
        ) & 1
        # calculate which packets encode for which channel
        initialize_stream_mask = np.logical_or(
            (np.arange(num_packet) == 0)[:, None], interleaved_data_id_bit_values == 1
        )
        # read the data into int16
        data = (
            self._raw_memmap[:, data_offsets[:, 0]].astype(np.int16)
            + self._raw_memmap[:, data_offsets[:, 0] + 1].astype(np.int16)
            * INT_16_CONVERSION
        )
        # initialize the first row
        analog_multiplexed_data[0] = data[0]
        # for packets that do not have an update for a channel, use the previous value
        for i in range(1, num_packet):
            analog_multiplexed_data[i] = np.where(
                initialize_stream_mask[i], data[i], analog_multiplexed_data[i - 1]
            )
        return analog_multiplexed_data

    def get_analogsignal_multiplexed_partial(
        self,
        i_start: int,
        i_stop: int,
        channel_names: list = None,
        padding: int = 30000,
    ) -> np.ndarray:
        """Alternative method to access part of the multiplexed data.
        Not memory efficient for many calls because it reads a buffer chunk before the requested data.
        Better than get_analogsignal_multiplexed when need one call to specific time region

        Parameters
        ----------
        i_start : int
            index start
        i_stop : int
            index stop
        channel_names : list[str], optional
            channels to get, by default None will get all multiplex channels
        padding : int, optional
            how many packets before the desired series to load to ensure every channel receives update before requested,
            by default 30000

        Returns
        -------
        np.ndarray
            multiplex data

        Raises
        ------
        ValueError
            _description_
        """
        print("compute multiplex cache", self.filename)
        if channel_names is None:
            # read all multiplexed channels
            channel_names = list(self.multiplexed_channel_xml.keys())
        else:
            for ch_name in channel_names:
                if ch_name not in self.multiplexed_channel_xml:
                    raise ValueError(f"Channel name '{ch_name}' not found in file.")
        # determine which packets to get from data
        padding = min(padding, i_start)
        i_start = i_start - padding
        if i_stop is None:
            i_stop = self._raw_memmap.shape[0]

        # Make object to hold data
        num_packet = i_stop - i_start
        analog_multiplexed_data = np.empty(
            (num_packet, len(channel_names)), dtype=np.int16
        )

        # precompute the static data offsets
        data_offsets = np.empty((len(channel_names), 3), dtype=int)
        for j, ch_name in enumerate(channel_names):
            ch_xml = self.multiplexed_channel_xml[ch_name]
            data_offsets[j, 0] = int(
                self._multiplexed_byte_start + int(ch_xml.attrib["startByte"])
            )
            data_offsets[j, 1] = int(ch_xml.attrib["interleavedDataIDByte"])
            data_offsets[j, 2] = int(ch_xml.attrib["interleavedDataIDBit"])
        interleaved_data_id_byte_values = self._raw_memmap[
            i_start:i_stop, data_offsets[:, 1]
        ]
        interleaved_data_id_bit_values = (
            interleaved_data_id_byte_values >> data_offsets[:, 2]
        ) & 1
        # calculate which packets encode for which channel
        initialize_stream_mask = np.logical_or(
            (np.arange(num_packet) == 0)[:, None], interleaved_data_id_bit_values == 1
        )
        # read the data into int16
        data = (
            self._raw_memmap[i_start:i_stop, data_offsets[:, 0]].astype(np.int16)
            + self._raw_memmap[i_start:i_stop, data_offsets[:, 0] + 1].astype(np.int16)
            * INT_16_CONVERSION
        )
        # initialize the first row
        analog_multiplexed_data[0] = data[0]
        # for packets that do not have an update for a channel, use the previous value
        # this method assumes that every channel has an update within the buffer
        for i in range(1, num_packet):
            analog_multiplexed_data[i] = np.where(
                initialize_stream_mask[i], data[i], analog_multiplexed_data[i - 1]
            )
        return analog_multiplexed_data[padding:]

    def get_digitalsignal(
        self, stream_id: int, channel_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the digital signal changes for a given stream and channel.

        This method returns the timestamps and the change direction (0 to 1 or 1 to 0) for the specified channel in the stream.

        Parameters
        ----------
        stream_id : int
            The ID of the stream to retrieve the signal from.
        channel_id : int
            The ID of the channel within the stream.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing:
            - An array of timestamps when the signal changed.
            - An array of the change direction (0 or 1) for each timestamp.
        """
        # stream_id = self.header["signal_streams"][stream_index]["id"]

        # for now, allow only reading the entire dataset
        i_start = 0
        i_stop = None

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
        raw_packets_masked = self._raw_memmap[i_start:i_stop, byte_mask]

        bit_mask = self._mask_channels_bits[stream_id][channel_index]
        continuous_dio = np.unpackbits(raw_packets_masked, axis=1)[:, bit_mask].reshape(
            -1
        )
        change_dir = np.diff(continuous_dio).astype(
            np.int8
        )  # possible values: [-1, 0, 1]
        change_dir_trim = change_dir[change_dir != 0]  # keeps -1 and 1
        change_dir_trim[change_dir_trim == -1] = 0  # change -1 to 0
        # resulting array has 1 when there is a change from 0 to 1,
        # 0 when there is change from 1 to 0

        # track the timestamps when there is a change from 0 to 1 or 1 to 0
        if self.sysClock_byte:
            timestamps = self.get_regressed_systime(i_start, i_stop)
        else:
            timestamps = self.get_systime_from_trodes_timestamps(i_start, i_stop)
        dio_change_times = timestamps[np.where(change_dir)[0] + 1]

        # insert the first timestamp with the first value
        dio_change_times = np.insert(dio_change_times, 0, timestamps[0])
        change_dir_trim = np.insert(change_dir_trim, 0, continuous_dio[0])

        change_dir_trim = change_dir_trim.astype(np.uint8)

        # if re_order is not None:
        #     raw_unit16 = raw_unit16[:, re_order]

        return dio_change_times, change_dir_trim

    @functools.lru_cache(maxsize=1)
    def get_regressed_systime(
        self, i_start: int, i_stop: Optional[int] = None
    ) -> np.ndarray:
        """
        Retrieves the regressed system time based on the Trodes timestamp and the system clock.

        This method performs a linear regression between the Trodes timestamps and the system clock
        to adjust the timestamps to the system time.

        Parameters
        ----------
        i_start : int
            The start index for the time range.
        i_stop : int, optional
            The stop index for the time range. If `None`, uses the full available range.

        Returns
        -------
        np.ndarray
            A NumPy array containing the adjusted system timestamps.
        """
        NANOSECONDS_PER_SECOND = 1e9
        # get trodes timestamp values
        trodestime = self.get_analogsignal_timestamps(i_start, i_stop)
        # Convert
        trodestime_index = np.asarray(trodestime, dtype=np.float64)
        if not self.regressed_systime_parameters:
            # get raw systime values
            systime_seconds = self.get_sys_clock(i_start, i_stop)
            # regress
            slope, intercept, _, _, _ = linregress(trodestime_index, systime_seconds)
            self.regressed_systime_parameters = {
                "slope": slope,
                "intercept": intercept,
            }
        else:
            slope = self.regressed_systime_parameters["slope"]
            intercept = self.regressed_systime_parameters["intercept"]
        adjusted_timestamps = intercept + slope * trodestime_index
        return (adjusted_timestamps) / NANOSECONDS_PER_SECOND

    @functools.lru_cache(maxsize=1)
    def get_systime_from_trodes_timestamps(
        self, i_start: int, i_stop: Optional[int] = None
    ) -> np.ndarray:
        """
        Retrieves system time based on Trodes timestamps.

        This method computes the system time by using the Trodes timestamps
        and the system time at creation.

        Parameters
        ----------
        i_start : int
            The start index for the time range.
        i_stop : int, optional
            The stop index for the time range. If `None`, uses the full available range.

        Returns
        -------
        np.ndarray
            A NumPy array containing the computed system time values.
        """
        MILLISECONDS_PER_SECOND = 1e3
        # get values
        trodestime = self.get_analogsignal_timestamps(i_start, i_stop)
        initial_time = self.get_analogsignal_timestamps(0, 1)[0]
        return (trodestime - initial_time) * (1.0 / self._sampling_rate) + int(
            self.system_time_at_creation
        ) / MILLISECONDS_PER_SECOND

    def _interpolate_raw_memmap(
        self,
    ):
        # """Interpolates single dropped packets in the analog data."""
        print("Interpolate memmap: ", self.filename)
        self._raw_memmap = InsertedMemmap(self._raw_memmap, self.interpolate_index)

    def get_stream_index_from_id(self, stream_id: int) -> int:
        return np.where(self.header["signal_streams"]["id"] == stream_id)[0][0]

    def get_stream_id_from_index(self, stream_index: int) -> int:
        return self.header["signal_streams"]["id"][stream_index]


class InsertedMemmap:
    """
    class to return slices into an interpolated memmap
    Avoids loading data into memory during np.insert
    """

    def __init__(
        self, _raw_memmap: np.ndarray, inserted_index: Optional[np.ndarray] = None
    ) -> None:
        """
        Initializes an InsertedMemmap object to handle slices into an interpolated memmap.

        Parameters
        ----------
        _raw_memmap : np.ndarray
            The raw memory-mapped data to be accessed.
        inserted_index : np.ndarray, optional
            The indices where interpolation has occurred. Defaults to an empty array if not provided.
        """
        if inserted_index is None:
            inserted_index = np.array([])
        self._raw_memmap = _raw_memmap
        self.mapped_index = np.arange(self._raw_memmap.shape[0])
        self.mapped_index = np.insert(
            self.mapped_index, inserted_index, self.mapped_index[inserted_index]
        )
        self.inserted_locations = inserted_index + np.arange(len(inserted_index))
        self.shape = (self.mapped_index.size, self._raw_memmap.shape[1])

    def __getitem__(self, index: int | slice | tuple) -> np.ndarray:
        """
        Retrieves data from the memory-mapped array based on the given index or slice.

        Parameters
        ----------
        index : int, slice or tuple
            The index or slice for time and/or channel selection.

        Returns
        -------
        np.ndarray
            A NumPy array containing the selected data.
        """
        # request a slice in both time and channel
        if isinstance(index, tuple):
            index_chan = index[1]
            return self._raw_memmap[self.access_coordinates(index[0]), index_chan]
        # request a slice in time
        return self._raw_memmap[self.access_coordinates(index)]

    def access_coordinates(self, index: int | slice) -> np.ndarray:
        """
        Returns the coordinates of the memory-mapped data based on the provided index or slice.

        Parameters
        ----------
        index : int or slice
            The index or slice to select coordinates.

        Returns
        -------
        np.ndarray
            A NumPy array of coordinates for the selected index or slice.
        """
        if isinstance(index, int):
            return self.mapped_index[index]
        # if slice object
        elif isinstance(index, slice):
            # see if slice contains inserted values
            if (
                (
                    (not index.start is None)
                    and (not index.stop is None)
                    and np.any(
                        (self.inserted_locations >= index.start)
                        & (self.inserted_locations < index.stop)
                    )
                )
                | (
                    (index.start is None)
                    and (not index.stop is None)
                    and np.any(self.inserted_locations < index.stop)
                )
                | (
                    index.stop is None
                    and (not index.start is None)
                    and np.any(self.inserted_locations > index.start)
                )
                | (
                    index.start is None
                    and index.stop is None
                    and len(self.inserted_locations) > 0
                )
            ):
                # if so, need to use advanced indexing. return list of indeces
                return self.mapped_index[index]
            # if not, return slice object with coordinates adjusted
            else:
                return slice(
                    index.start
                    - np.searchsorted(self.inserted_locations, index.start, "right"),
                    index.stop
                    - np.searchsorted(self.inserted_locations, index.stop, "right"),
                    index.step,
                )
        # if list of indeces
        else:
            return self.mapped_index[index]


class SpikeGadgetsRawIOPartial(SpikeGadgetsRawIO):
    extensions = ["rec"]
    rawmode = "one-file"

    def __init__(
        self,
        full_io: SpikeGadgetsRawIO,
        start_index: int,
        stop_index: int,
        previous_multiplex_state: Optional[np.ndarray] = None,
    ):
        """Initialize a partial SpikeGadgetsRawIO object.

        Parameters
        ----------
        full_io : SpikeGadgetsRawIO
            The SpikeGadgetsRawIO for the complete rec file
        start_index : int
            Where this partial file starts in the complete file
        stop_index : int
            Where this partial file stops in the complete file
        previous_multiplex_state : np.ndarray, optional
            The last multiplex state in the previous partial file.
            If None, will default to behavior of SpikeGadgetsRawIO.
            Use None if first partial iterator for the rec file or if not accessing multiplex data.
            By default None
        """
        # initialization from the base class
        BaseRawIO.__init__(self)
        self.filename = full_io.filename
        self.selected_streams = full_io.selected_streams
        self.interpolate_dropped_packets = full_io.interpolate_dropped_packets

        # define some key information
        self.interpolate_index = None
        self.previous_multiplex_state = previous_multiplex_state

        # copy conserved information from parsed_header from full_io
        self.header = full_io.header
        self.system_time_at_creation = full_io.system_time_at_creation
        self.timestamp_at_creation = full_io.timestamp_at_creation
        self._sampling_rate = full_io._sampling_rate
        self.sysClock_byte = full_io.sysClock_byte
        self._timestamp_byte = full_io._timestamp_byte
        self._mask_channels_ids = full_io._mask_channels_ids
        self._mask_channels_bytes = full_io._mask_channels_bytes
        self._mask_channels_bits = full_io._mask_channels_bits
        self.multiplexed_channel_xml = full_io.multiplexed_channel_xml
        self._multiplexed_byte_start = full_io._multiplexed_byte_start
        self._mask_streams = full_io._mask_streams
        self.selected_streams = full_io.selected_streams
        self._generate_minimal_annotations()
        self.regressed_systime_parameters = full_io.regressed_systime_parameters

        # crop key information to range of interest
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
        # Inherit the original memmap object from the full_io object to conserve virtual memory
        if isinstance(full_io._raw_memmap, InsertedMemmap):
            self._raw_memmap = full_io._raw_memmap._raw_memmap
        else:
            self._raw_memmap = full_io._raw_memmap
        self._raw_memmap = self._raw_memmap[start_index:stop_index]
        # ensure interpolation
        if self.interpolate_dropped_packets and self.interpolate_index is None:
            raw_uint8 = self._raw_memmap[
                :, self._timestamp_byte : self._timestamp_byte + TIMESTAMP_SIZE_BYTES
            ]
            raw_uint32 = (
                raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
            )
            self.interpolate_index = np.where(
                np.diff(raw_uint32) == EXPECTED_TIMESTAMP_DIFF_DROP
            )[
                0
            ]  # find locations of single dropped packets
            self._interpolate_raw_memmap()

    @functools.lru_cache(maxsize=2)
    def get_analogsignal_multiplexed(
        self, channel_names: Optional[list[str]] = None
    ) -> np.ndarray:
        """
        Overide of the superclass to use the last state of the previous file segment
        to define the first state of the current file segment.
        """
        print("compute multiplex cache", self.filename)
        if channel_names is None:
            # read all multiplexed channels
            channel_names = list(self.multiplexed_channel_xml.keys())
        else:
            for ch_name in channel_names:
                if ch_name not in self.multiplexed_channel_xml:
                    raise ValueError(f"Channel name '{ch_name}' not found in file.")

        # because of the encoding scheme, it is easiest to read all the data in sequence
        # one packet at a time
        num_packet = self._raw_memmap.shape[0]
        analog_multiplexed_data = np.empty(
            (num_packet, len(channel_names)), dtype=np.int16
        )

        # precompute the static data offsets
        data_offsets = np.empty((len(channel_names), 3), dtype=int)
        for j, ch_name in enumerate(channel_names):
            ch_xml = self.multiplexed_channel_xml[ch_name]
            data_offsets[j, 0] = int(
                self._multiplexed_byte_start + int(ch_xml.attrib["startByte"])
            )
            data_offsets[j, 1] = int(ch_xml.attrib["interleavedDataIDByte"])
            data_offsets[j, 2] = int(ch_xml.attrib["interleavedDataIDBit"])
        interleaved_data_id_byte_values = self._raw_memmap[:, data_offsets[:, 1]]
        interleaved_data_id_bit_values = (
            interleaved_data_id_byte_values >> data_offsets[:, 2]
        ) & 1
        # calculate which packets encode for which channel
        initialize_stream_mask = np.logical_or(
            (np.arange(num_packet) == 0)[:, None], interleaved_data_id_bit_values == 1
        )
        # read the data into int16
        data = (
            self._raw_memmap[:, data_offsets[:, 0]].astype(np.int16)
            + self._raw_memmap[:, data_offsets[:, 0] + 1].astype(np.int16)
            * INT_16_CONVERSION
        )
        # initialize the first row
        # if no previous state, assume first segment. Default to superclass behavior
        analog_multiplexed_data[0] = data[0]
        if self.previous_multiplex_state is not None:
            # if previous state, use it to initialize elements of first row not updated in that packet
            ind = np.where(initialize_stream_mask[0])[0]
            analog_multiplexed_data[0][ind] = self.previous_multiplex_state[ind]
        # for packets that do not have an update for a channel, use the previous value
        for i in range(1, num_packet):
            analog_multiplexed_data[i] = np.where(
                initialize_stream_mask[i], data[i], analog_multiplexed_data[i - 1]
            )
        return analog_multiplexed_data

    def get_digitalsignal(
        self, stream_id: int, channel_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the digital signal changes for a given stream and channel.

        This method returns the timestamps and the change direction (0 to 1 or 1 to 0) for the specified channel in the stream.

        Parameters
        ----------
        stream_id : int
            The ID of the stream to retrieve the signal from.
        channel_id : int
            The ID of the channel within the stream.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing:
            - An array of timestamps when the signal changed.
            - An array of the change direction (0 or 1) for each timestamp.
        """
        dio_change_times, change_dir_trim = super().get_digitalsignal(
            stream_id, channel_id
        )
        # clip the setting of the first state
        return dio_change_times[1:], change_dir_trim[1:]
