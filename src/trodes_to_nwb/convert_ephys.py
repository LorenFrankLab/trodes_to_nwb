import logging
from typing import Tuple
from warnings import warn

import numpy as np
from hdmf.backends.hdf5 import H5DataIO
from hdmf.data_utils import GenericDataChunkIterator
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries

from trodes_to_nwb import convert_rec_header

from .spike_gadgets_raw_io import SpikeGadgetsRawIO, SpikeGadgetsRawIOPartial

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1e-6
MILLISECONDS_PER_SECOND = 1e3
NANOSECONDS_PER_SECOND = 1e9
MAXIMUM_ITERATOR_SIZE = int(30000 * 60 * 30)  # 30 min of data at 30 kHz


class RecFileDataChunkIterator(GenericDataChunkIterator):
    """Data chunk iterator for SpikeGadgets rec files."""

    def __init__(
        self,
        rec_file_path: list[str],
        nwb_hw_channel_order=[],
        conversion: float = 1.0,
        stream_index: int = 3,  # TODO use the stream name instead of the index
        is_analog: bool = False,
        interpolate_dropped_packets: bool = False,
        timestamps=None,  # Use this if you already have timestamps from intializing another rec iterator on the same files
        **kwargs,
    ):
        if not rec_file_path:
            raise FileNotFoundError("Must provide at least one rec file path")
        logger = logging.getLogger("convert")
        self.conversion = conversion
        self.is_analog = is_analog
        self.neo_io = [
            SpikeGadgetsRawIO(
                filename=file, interpolate_dropped_packets=interpolate_dropped_packets
            )
            for file in rec_file_path
        ]  # get all streams for all files
        logger.info("Parsing headers")
        [neo_io.parse_header() for neo_io in self.neo_io]
        # TODO see what else spikeinterface does and whether it is necessary
        logger.info("Parsing header COMPLETE")
        # for now, make sure that there is only one block, one segment, and four streams:
        # Controller_DIO_digital
        # ECU_digital
        # ECU_analog
        # trodes
        assert all([neo_io.block_count() == 1 for neo_io in self.neo_io])
        assert all([neo_io.segment_count(0) == 1 for neo_io in self.neo_io])
        assert all([neo_io.signal_streams_count() == 4 for neo_io in self.neo_io])

        self.block_index = 0
        self.seg_index = 0
        self.stream_index = stream_index  # TODO confirm that the stream index is trodes

        # check that all files have the same number of channels.
        if (
            len(
                set(
                    [
                        neo_io.signal_channels_count(stream_index=self.stream_index)
                        for neo_io in self.neo_io
                    ]
                )
            )
            > 1
        ):
            raise ValueError("All files must have the same number of signal channels.")
        self.n_channel = self.neo_io[0].signal_channels_count(
            stream_index=self.stream_index
        )
        self.n_multiplexed_channel = 0
        if self.is_analog:
            self.n_multiplexed_channel += len(self.neo_io[0].multiplexed_channel_xml)

        # order that the hw channels are in within the nwb table
        if len(nwb_hw_channel_order) == 0:  # TODO: raise error instead?
            self.nwb_hw_channel_order = np.arange(self.n_channel)
        else:
            self.nwb_hw_channel_order = nwb_hw_channel_order

        """split excessively large iterators into smaller ones
        """
        iterator_size = [neo_io._raw_memmap.shape[0] for neo_io in self.neo_io]
        iterator_size.reverse()
        for i, size in enumerate(
            iterator_size
        ):  # iterate backwards so can insert new iterators
            if size > MAXIMUM_ITERATOR_SIZE:
                # split into smaller iterators
                sub_iterators = []
                j = 0
                previous_multiplex_state = None
                iterator_loc = len(iterator_size) - i - 1
                while j < size:
                    sub_iterators.append(
                        SpikeGadgetsRawIOPartial(
                            self.neo_io[iterator_loc],
                            start_index=j,
                            stop_index=j + MAXIMUM_ITERATOR_SIZE,
                            previous_multiplex_state=previous_multiplex_state,
                        )
                    )
                    if self.n_multiplexed_channel > 0:
                        partial_size = sub_iterators[-1]._raw_memmap.shape[0]
                        previous_multiplex_state = sub_iterators[
                            -1
                        ].get_analogsignal_multiplexed_partial(
                            i_start=partial_size - 10,
                            i_stop=partial_size,
                            padding=30000,
                        )[
                            -1
                        ]
                    j += MAXIMUM_ITERATOR_SIZE
                self.neo_io.pop(iterator_loc)
                self.neo_io[iterator_loc:iterator_loc] = sub_iterators
        logger.info(f"# iterators: {len(self.neo_io)}")
        # NOTE: this will read all the timestamps from the rec file, which can be slow
        if timestamps is not None:
            self.timestamps = timestamps

        elif self.neo_io[0].sysClock_byte:  # use this if have sysClock
            self.timestamps = np.concatenate(
                [neo_io.get_regressed_systime(0, None) for neo_io in self.neo_io]
            )

        else:  # use this to convert Trodes timestamps into systime based on sampling rate
            self.timestamps = np.concatenate(
                [
                    neo_io.get_systime_from_trodes_timestamps(0, None)
                    for neo_io in self.neo_io
                ]
            )

        logger.info("Reading timestamps COMPLETE")
        is_timestamps_sequential = np.all(np.diff(self.timestamps))
        if not is_timestamps_sequential:
            warn(
                "Timestamps are not sequential. This may cause problems with some software or data analysis."
            )

        self.n_time = [
            neo_io.get_signal_size(
                block_index=self.block_index,
                seg_index=self.seg_index,
                stream_index=self.stream_index,
            )
            for neo_io in self.neo_io
        ]

        super().__init__(**kwargs)

    def _get_data(self, selection: Tuple[slice]) -> np.ndarray:
        # selection is (time, channel)
        assert selection[0].step is None

        # slice to indices
        # DCI will want channels 0 to X first to put into the array in that order
        # those are stored in the file as channel IDs
        # make into list form passed to neo_io
        selection_list = list(selection)
        if self.is_analog:
            selection_list[1] = slice(
                selection[1].start,
                min(selection[1].stop, self.n_channel),
                selection[1].step,
            )
        channel_ids = [str(x) for x in self.nwb_hw_channel_order[selection_list[1]]]
        # what global index each file starts at
        file_start_ind = np.append(np.zeros(1), np.cumsum(self.n_time))
        # the time indexes we want
        time_index = np.arange(selection_list[0].start, selection_list[0].stop)[
            :: selection_list[0].step
        ]
        data = []
        i = time_index[0]
        while i < min(time_index[-1], self._get_maxshape()[0]):
            # find the stream where this piece of slice begins
            io_stream = np.argmin(i >= file_start_ind) - 1
            # get the data from that stream
            data.append(
                (
                    self.neo_io[io_stream].get_analogsignal_chunk(
                        block_index=self.block_index,
                        seg_index=self.seg_index,
                        i_start=int(i - file_start_ind[io_stream]),
                        i_stop=int(
                            min(
                                time_index[-1] - file_start_ind[io_stream],
                                self.n_time[io_stream],
                            )
                        )
                        + 1,
                        stream_index=self.stream_index,
                        channel_ids=channel_ids,
                    )
                )
            )
            i += min(
                self.n_time[io_stream]
                - (i - file_start_ind[io_stream]),  # if added up to the end of stream
                time_index[-1] - i,  # if finished in this stream
            )

        data = (np.concatenate(data) * self.conversion).astype("int16")
        # Handle the appended multiplex data
        if (
            self.neo_io[0].header["signal_streams"][self.stream_index]["id"]
            == "ECU_analog"
        ) and self.is_analog:
            multiplex_keys = self.neo_io[0].multiplexed_channel_xml.keys()
            n_multiplex = len(multiplex_keys)
            n_analog = (
                self.n_channel
            )  # number of non-multiplexed channels in the dataset
            n_analog_selected = data.shape[1] - n_multiplex
            return_indices = np.arange(
                n_analog_selected
            )  # include all non-multiplexed channels pulled
            # determine which multiplex channels are being requested
            if (
                selection[1].stop > n_analog
            ):  # if multiplexed channels are being requested
                requested_multiplex = np.arange(n_multiplex) + n_analog_selected
                multiplex_slice = slice(
                    max(selection[1].start - n_analog, 0),
                    max(selection[1].stop - n_analog, 0),
                    selection[1].step,
                )
                requested_multiplex = requested_multiplex[multiplex_slice]
                return_indices = np.append(return_indices, requested_multiplex)
            data = data[:, return_indices]

        return data

    def _get_maxshape(self) -> Tuple[int, int]:
        return (
            np.sum(self.n_time),
            self.n_channel + self.n_multiplexed_channel,
        )  # TODO: Is this right for maxshape @rly

    def _get_dtype(self) -> np.dtype:
        return np.dtype("int16")


def add_raw_ephys(
    nwbfile: NWBFile,
    recfile: list[str],
    electrode_row_indices: list[int],
    metadata: dict = None,
) -> None:
    """Adds the raw ephys data to a NWB file. Must be called after add_electrode_groups

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    recfile : list[str]
        ordered list of file paths to all recfiles with session's data
    electrode_row_indices : list
        which electrodes to add to table
    metadata : dict, optional
        metadata dictionary, useed only for conversion if not in rec, by default None
    """

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=electrode_row_indices,
        description="electrodes used in raw e-series recording",
    )
    # get hw channel order
    nwb_hw_chan_order = [
        int(x) for x in list(nwbfile.electrodes.to_dataframe()["hwChan"])
    ]
    # get conversion factor from rec file
    rec_header = convert_rec_header.read_header(recfile[0])
    spike_config = rec_header.find("SpikeConfiguration")
    if "rawScalingToUv" in spike_config[0].attrib:
        conversion = float(spike_config[0].attrib["rawScalingToUv"])
    else:
        conversion = (
            metadata["raw_data_to_volts"] * MICROVOLTS_PER_VOLT
        )  # Use metadata-provided conversion if not available in rec file

    # make the data iterator
    rec_dci = RecFileDataChunkIterator(
        recfile,
        nwb_hw_channel_order=nwb_hw_chan_order,
        conversion=conversion,
        interpolate_dropped_packets=True,
    )  # can set buffer_gb if needed

    # (16384, 32) chunks of dtype int16 (2 bytes) is 1 MB, which is recommended
    # by studies by the NWB team.
    # could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    data_data_io = H5DataIO(rec_dci, chunks=(16384, min(rec_dci.n_channel, 32)))

    # do we want to pull the timestamps from the rec file? or is there another source?
    eseries = ElectricalSeries(
        name="e-series",
        data=data_data_io,
        timestamps=rec_dci.timestamps,
        electrodes=electrode_table_region,  # TODO
        conversion=VOLTS_PER_MICROVOLT,
        offset=0.0,  # TODO
    )

    nwbfile.add_acquisition(eseries)
