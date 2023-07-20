from typing import Tuple
from hdmf.data_utils import GenericDataChunkIterator
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from warnings import warn
from spikegadgets_to_nwb import convert_rec_header

from .spike_gadgets_raw_io import SpikeGadgetsRawIO

MICROVOLTS_PER_VOLT = 1e6


class RecFileDataChunkIterator(GenericDataChunkIterator):
    """Data chunk iterator for SpikeGadgets rec files."""

    def __init__(self, rec_file_path: list[str], nwb_hw_channel_order=[], **kwargs):
        self.neo_io = [
            SpikeGadgetsRawIO(filename=file) for file in rec_file_path
        ]  # get all streams for all files
        [neo_io.parse_header() for neo_io in self.neo_io]
        # TODO see what else spikeinterface does and whether it is necessary

        # for now, make sure that there is only one block, one segment, and two streams
        assert all([neo_io.block_count() == 1 for neo_io in self.neo_io])
        assert all([neo_io.segment_count(0) == 1 for neo_io in self.neo_io])
        assert all([neo_io.signal_streams_count() == 2 for neo_io in self.neo_io])

        self.block_index = 0
        self.seg_index = 0
        self.stream_index = 1  # TODO confirm that the stream index is trodes

        self.n_time = [
            neo_io.get_signal_size(
                block_index=self.block_index,
                seg_index=self.seg_index,
                stream_index=self.stream_index,
            )
            for neo_io in self.neo_io
        ]

        # check that all files have the same number of channels.
        assert (
            len(
                set(
                    [
                        neo_io.signal_channels_count(stream_index=self.stream_index)
                        for neo_io in self.neo_io
                    ]
                )
            )
            == 1
        )
        self.n_channel = self.neo_io[0].signal_channels_count(
            stream_index=self.stream_index
        )

        # order that the hw channels are in within the nwb table
        if len(nwb_hw_channel_order) == 0:  # TODO: raise error instead?
            self.nwb_hw_channel_order = np.arange(self.n_channel)
        else:
            self.nwb_hw_channel_order = nwb_hw_channel_order

        # NOTE: this will read all the timestamps from the rec file, which can be slow
        self.timestamps = []
        [
            self.timestamps.extend(neo_io.get_analogsignal_timestamps(0, n_time))
            for neo_io, n_time in zip(self.neo_io, self.n_time)
        ]
        is_timestamps_sequential = np.all(np.diff(self.timestamps))
        if not is_timestamps_sequential:
            warn(
                "Timestamps are not sequential. This may cause problems with some software or data analysis."
            )

        super().__init__(**kwargs)

    def _get_data(self, selection: Tuple[slice]) -> np.ndarray:
        # selection is (time, channel)
        assert selection[0].step is None

        # slice to indices
        # DCI will want channels 0 to X first to put into the array in that order
        # those are stored in the file as channel IDs
        # make into list form passed to neo_io
        channel_ids = [str(x) for x in self.nwb_hw_channel_order[selection[1]]]
        # what global index each file starts at
        file_start_ind = np.append(np.zeros(1), np.cumsum(self.n_time))
        # the time indexes we want
        time_index = np.arange(self._get_maxshape()[0])[selection[0]]
        data = []
        i = time_index[0]
        while i < min(time_index[-1], self._get_maxshape()[0]):
            # find the stream where this piece of slice begins
            io_stream = np.argmin(i >= file_start_ind) - 1
            # get the data from that stream
            data.extend(
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
            print(
                "added",
                self.n_time[io_stream]
                - (i - file_start_ind[io_stream]),  # if added up to the end of stream
                time_index[-1] - i,  # if finished in this stream
            )
            i += min(
                self.n_time[io_stream]
                - (i - file_start_ind[io_stream]),  # if added up to the end of stream
                time_index[-1] - i,  # if finished in this stream
            )
        data = np.array(data).astype("int16")
        return data

    def _get_maxshape(self) -> Tuple[int, int]:
        return (
            np.sum(self.n_time),
            self.n_channel,
        )  # TODO: Is this right for maxshape @rly

    def _get_dtype(self) -> np.dtype:
        return np.dtype("int16")


def add_raw_ephys(
    nwbfile: NWBFile,
    recfile: list[str],
    electrode_row_indices: list[int],
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
    conversion : float
        The conversion factor from nwb data to volts
    """

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=electrode_row_indices,
        description="electrodes used in raw e-series recording",
    )
    # get hw channel order
    nwb_hw_chan_order = [
        int(x) for x in list(nwbfile.electrodes.to_dataframe()["hwChan"])
    ]
    # make the data iterator
    rec_dci = RecFileDataChunkIterator(
        recfile,
        nwb_hw_channel_order=nwb_hw_chan_order,
    )  # can set buffer_gb if needed

    # (16384, 32) chunks of dtype int16 (2 bytes) is 1 MB, which is recommended
    # by studies by the NWB team.
    # could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    data_data_io = H5DataIO(rec_dci, chunks=(16384, min(rec_dci.n_channel, 32)))

    # get conversion factor from rec file
    rec_header = convert_rec_header.read_header(recfile[0])
    spike_config = rec_header.find("SpikeConfiguration")
    conversion = float(spike_config[0].attrib["rawScalingToUv"]) / MICROVOLTS_PER_VOLT
    # do we want to pull the timestamps from the rec file? or is there another source?
    eseries = ElectricalSeries(
        name="e-series",
        data=data_data_io,
        timestamps=rec_dci.timestamps,
        electrodes=electrode_table_region,  # TODO
        conversion=conversion,
        offset=0.0,  # TODO
    )

    nwbfile.add_acquisition(eseries)
