"""Provides functions for reading, parsing, and interpreting the XML header
information embedded within Trodes .rec files. Extracts hardware configuration,
electrode mappings, and other essential metadata.
"""

import copy
import logging
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from ndx_franklab_novela import HeaderDevice
from pynwb import NWBFile


def read_header(recfile: Path | str) -> ElementTree.Element:
    """Read XML header from rec file.

    Parameters
    ----------
    recfile : Path or str
        Path to rec file.

    Returns
    -------
    ElementTree.Element
        XML header element.

    Raises
    ------
    ValueError
        If the XML header does not contain '</Configuration>'.
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


def group_ntrodes_uniformly(n_source: int, per_group: int) -> list[list[int]]:
    """Build uniform reconfig merge-groups of consecutive source ntrode ids.

    Convenience for the common case of a probe whose shanks each have the same
    number of source ntrodes -- e.g. a 128-channel, 4-shank probe recorded as 32
    tetrodes is ``group_ntrodes_uniformly(32, 8)``. Source ntrode ids are 1-based
    and taken in order, so this assumes the source ntrodes are ordered such that
    each consecutive block of ``per_group`` belongs to one shank. For non-uniform
    probes, or wiring where consecutive source ntrodes do not map to one shank,
    pass explicit groups to :func:`generate_reconfig_header` instead -- there is
    no way to infer the grouping from the metadata.

    Parameters
    ----------
    n_source : int
        Total number of source ntrodes in the header (a positive multiple of
        ``per_group``).
    per_group : int
        Number of consecutive source ntrodes merged into each new ntrode.

    Returns
    -------
    list[list[int]]
        ``[[1, ..., per_group], [per_group + 1, ...], ...]``, suitable for
        :func:`generate_reconfig_header`.

    Raises
    ------
    ValueError
        If ``n_source`` or ``per_group`` is not positive, or ``n_source`` is not
        a multiple of ``per_group`` (the groups would not be uniform).
    """
    if n_source < 1 or per_group < 1:
        raise ValueError("n_source and per_group must be positive integers")
    if n_source % per_group != 0:
        raise ValueError(
            f"n_source ({n_source}) is not a multiple of per_group ({per_group}); "
            "the groups would not be uniform. Pass explicit ntrode_groups to "
            "generate_reconfig_header for non-uniform probes."
        )
    return [
        list(range(start, start + per_group))
        for start in range(1, n_source + 1, per_group)
    ]


def generate_reconfig_header(
    rec_header: ElementTree.Element,
    ntrode_groups: list[list[int]],
    allow_partial: bool = False,
) -> ElementTree.Element:
    """Build a reconfigured header by merging SpikeNTrode groups.

    Trodes records one ``SpikeNTrode`` per acquisition group (e.g. one per
    tetrode). To represent a multi-contact probe as a single electrode group,
    those ntrodes must be merged into one ntrode per probe/shank -- the manual
    "delete groupings" step otherwise done by hand in the Trodes GUI. This
    function automates it by concatenating, in order, the channels of the source
    ntrodes in each group into a single new ntrode.

    Each merged ntrode inherits the reference of its first source ntrode. Because
    the merged ntrodes are renumbered, ntrode-level references are retargeted to
    the merged numbering. The reference may be stored as ``refNTrodeID`` (the
    referenced ntrode's id) or the legacy ``refNTrode`` (its 1-based position,
    used by default Trodes configs); both are repointed at the merged ntrode that
    now holds the referenced channel, and ``refChan`` is shifted by that channel's
    offset within the merged ntrode. ``refNTrodeID`` is always written on the
    output -- it is the canonical reference Trodes emits on save and the only form
    the converter reads, so a legacy ``refNTrode``-only source gains it rather
    than losing the reference downstream.

    Parameters
    ----------
    rec_header : xml.etree.ElementTree.Element
        Parsed header (root ``<Configuration>``) returned by :func:`read_header`.
    ntrode_groups : list[list[int]]
        Each inner list gives the source ``SpikeNTrode`` ids to merge, in order,
        into one new ntrode. Inner lists may have different lengths to support
        probes with differing contact counts. New ntrodes are numbered
        ``1..len(ntrode_groups)``. Every referenced id must exist in the header,
        and (unless ``allow_partial``) ``ntrode_groups`` must be a partition of
        the source ntrodes -- each used exactly once.
    allow_partial : bool, optional
        If False (default), every source ntrode must be assigned to exactly one
        group; leaving some unassigned raises (their channels would be silently
        dropped). Set True to intentionally drop the unassigned source ntrodes.

    Returns
    -------
    xml.etree.ElementTree.Element
        A deep copy of ``rec_header`` whose ``SpikeConfiguration`` holds the
        merged ntrodes. The input element is not modified.

    Raises
    ------
    ValueError
        If the header has no ``SpikeConfiguration``; if ``ntrode_groups`` is
        empty or contains an empty group; if a source ntrode is assigned to more
        than one group (which would place one channel in two electrode groups);
        or if source ntrodes are left unassigned and ``allow_partial`` is False.
    KeyError
        If a referenced ntrode id is not present in the header.
    """
    if not ntrode_groups:
        raise ValueError("ntrode_groups must contain at least one group")
    if any(not group for group in ntrode_groups):
        raise ValueError("each entry in ntrode_groups must be non-empty")

    new_header = copy.deepcopy(rec_header)
    spike_config = new_header.find("SpikeConfiguration")
    if spike_config is None:
        raise ValueError("rec_header has no SpikeConfiguration element")

    source_by_id = {ntrode.attrib["id"]: ntrode for ntrode in spike_config}
    # Document order of source ntrodes; the legacy refNTrode reference is a 1-based
    # index into this list (Trodes stores the reference both as refNTrodeID -- the
    # referenced ntrode's id -- and refNTrode -- its position), so resolving it
    # needs position->id.
    source_ids_in_order = [ntrode.attrib["id"] for ntrode in spike_config]

    # Validate that ntrode_groups is a clean partition of the source ntrodes.
    # Assigning a source ntrode to two groups (or twice within a group) would
    # place the same physical channels in two electrode groups, and leaving a
    # source ntrode unassigned silently drops its channels -- both are almost
    # always caller typos, so fail loudly rather than misbuild the probe map.
    flat = [str(src_id) for group in ntrode_groups for src_id in group]
    missing = [src_id for src_id in dict.fromkeys(flat) if src_id not in source_by_id]
    if missing:
        raise KeyError(
            f"ntrode id(s) {missing} not present in the header SpikeConfiguration"
        )
    counts = Counter(flat)
    duplicated = sorted((src_id for src_id, n in counts.items() if n > 1), key=int)
    if duplicated:
        raise ValueError(
            f"ntrode id(s) {duplicated} appear in more than one reconfig group "
            "(or twice in one group); each source ntrode's channels can only be "
            "assigned to a single merged ntrode."
        )
    unassigned = sorted(
        (src_id for src_id in source_by_id if src_id not in counts), key=int
    )
    if unassigned and not allow_partial:
        raise ValueError(
            f"source ntrode id(s) {unassigned} are not assigned to any reconfig "
            "group; their channels would be dropped from the reconfigured header. "
            "Assign them to a group, or pass allow_partial=True to drop them "
            "intentionally."
        )

    # Map each source ntrode id -> (merged ntrode id, channel offset within the
    # merged ntrode). An ntrode reference (refNTrodeID/refChan) points at a source
    # ntrode and a 1-based channel within it; after merging+renumbering it must be
    # retargeted to the merged ntrode/channel that now holds that channel.
    source_to_merged = {}
    merged_ntrodes = []
    for new_id, group in enumerate(ntrode_groups, start=1):
        # Base the merged ntrode on the first source ntrode so it keeps the
        # group-level attributes (scaling, reference settings, ...).
        merged = copy.deepcopy(source_by_id[str(group[0])])
        merged.attrib["id"] = str(new_id)
        for channel in list(merged):
            merged.remove(channel)
        offset = 0
        for src_id in group:
            source = source_by_id[str(src_id)]
            source_to_merged[str(src_id)] = (new_id, offset)
            for channel in source:
                merged.append(copy.deepcopy(channel))
            offset += len(source)
        merged_ntrodes.append(merged)

    # Retarget each merged ntrode's reference (inherited from its first source
    # ntrode) to the merged numbering. Leaving the source values would dangle --
    # make_ref_electrode_map raises KeyError on a refNTrodeID absent from the
    # reconfig metadata, or silently resolves to the wrong merged group when a
    # stale source id happens to collide with a merged id; a stale refNTrode index
    # likewise points outside the merged ntrode list when the header is reopened
    # in Trodes (default Trodes configs use refNTrode without refNTrodeID).
    for merged in merged_ntrodes:
        attrib = merged.attrib
        ref_id = attrib.get("refNTrodeID")
        ref_index = attrib.get("refNTrode")
        # Resolve the referenced source ntrode id. refNTrodeID (the id) is
        # authoritative when present and positive; otherwise fall back to the
        # legacy refNTrode (a 1-based position in the source ntrode list).
        if ref_id is not None and int(ref_id) > 0:
            src_ref_id = ref_id
        elif ref_index is not None and int(ref_index) > 0:
            idx = int(ref_index) - 1
            if not 0 <= idx < len(source_ids_in_order):
                raise ValueError(
                    f"ntrode reference refNTrode={ref_index} is out of range for "
                    f"the {len(source_ids_in_order)} source ntrodes."
                )
            src_ref_id = source_ids_in_order[idx]
        else:
            continue  # Trodes "no reference" sentinel; nothing to retarget.
        target = source_to_merged.get(src_ref_id)
        if target is None:
            raise ValueError(
                f"ntrode reference points to source ntrode {src_ref_id}, which is "
                "not part of any reconfig group, so the reference cannot be "
                "retargeted to the merged header. Include that source ntrode in a "
                "group, or clear the reference before reconfiguring."
            )
        new_ref_id, ref_offset = target
        # Merged ntrodes are numbered 1..N in document order, so a merged ntrode's
        # id equals its 1-based position; refNTrodeID and refNTrode both become it.
        # Always emit refNTrodeID -- it is the canonical reference Trodes writes on
        # save, and the only one make_ref_electrode_map reads, so a legacy
        # refNTrode-only source must gain it or the reference is dropped from NWB.
        attrib["refNTrodeID"] = str(new_ref_id)
        if ref_index is not None:
            attrib["refNTrode"] = str(new_ref_id)
        if "refChan" in attrib:
            # refChan is 1-based within the referenced source ntrode; shift it by
            # that ntrode's channel offset within its merged ntrode.
            attrib["refChan"] = str(int(attrib["refChan"]) + ref_offset)

    for ntrode in list(spike_config):
        spike_config.remove(ntrode)
    spike_config.extend(merged_ntrodes)
    return new_header


def write_reconfig_trodesconf(
    rec_header_path: Path | str,
    output_path: Path | str,
    ntrode_groups: list[list[int]],
    allow_partial: bool = False,
) -> Path:
    """Read a header, merge its ntrodes, and write a reconfigured ``.trodesconf``.

    Parameters
    ----------
    rec_header_path : Path or str
        Path to the source ``.rec`` or ``.trodesconf`` whose header is reconfigured.
    output_path : Path or str
        Where to write the generated ``.trodesconf`` file.
    ntrode_groups : list[list[int]]
        Merge-groups passed to :func:`generate_reconfig_header`.
    allow_partial : bool, optional
        Passed through to :func:`generate_reconfig_header`; if False (default),
        every source ntrode must be assigned to a group.

    Returns
    -------
    Path
        The ``output_path`` that was written.
    """
    new_header = generate_reconfig_header(
        read_header(rec_header_path), ntrode_groups, allow_partial=allow_partial
    )
    output_path = Path(output_path)
    # read_header re-parses this file by line-scanning for the line containing
    # </Configuration> and truncating there, so the writer must emit no XML
    # declaration and must keep </Configuration> on a single line (no
    # pretty-printing that would split it). encoding="unicode" + the default
    # serializer satisfy both and match the embedded .rec header format.
    ElementTree.ElementTree(new_header).write(output_path, encoding="unicode")
    return output_path


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
        for _, test_meta in enumerate(metadata["ntrode_electrode_group_channel_map"]):
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
        # define the current ntrode group's nwb id
        ntrode_id = group.attrib["id"]
        nwb_group_id = ntrode_id_to_nwb[ntrode_id]
        if "refNTrodeID" in group.attrib:
            ntrode_ref_group_id = group.attrib["refNTrodeID"]
            # if reference defined on null ntrode value, set to -1
            if int(ntrode_ref_group_id) <= 0:
                ref_electrode_map[nwb_group_id] = (-1, -1)
                continue
            # find channel map for ref group
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
