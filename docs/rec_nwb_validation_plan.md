# Rec-to-NWB Validation Feature Plan

## Goal

Add a feature that validates whether a converted NWB file faithfully represents the source SpikeGadgets `.rec` recording, using:

- one or more `.rec` files
- the output `.nwb` file
- the metadata YAML used for conversion

The validator should focus on rec-derived and YAML-configured content that this package is responsible for creating, and should produce a structured report that makes mismatches actionable.

## Why the YAML Must Be an Input

The `.rec` files alone are not enough to validate the full conversion. The YAML is required for:

- electrode group to ntrode mapping
- expected electrode ordering in the NWB electrodes table
- DIO channel naming and comments
- distinguishing which NWB structures are expected for the session
- validating metadata-derived objects that are intentionally inserted during conversion

Without the YAML, validation can only cover a subset of the conversion.

## Proposed Public API

Add a new production module:

- `src/trodes_to_nwb/validate_conversion.py`

Expose a public function:

```python
from pathlib import Path

def validate_conversion(
    rec_filepaths: list[Path] | list[str],
    nwb_filepath: Path | str,
    metadata_filepath: Path | str,
    *,
    header_reconfig_path: Path | str | None = None,
    device_metadata_paths: list[Path] | list[str] | None = None,
    behavior_only: bool = False,
    max_ephys_frames_per_chunk: int = 16384,
    ephys_tolerance_uv: float = 1.0,
    timestamp_tolerance_s: float | None = None,
) -> dict:
    ...
```

Add a convenience wrapper for a single session directory later if useful, but do not make that the primary API.

## Report Format

Return a structured dictionary with top-level sections:

- `passed`
- `summary`
- `inputs`
- `checks`
- `errors`
- `warnings`

Each item in `checks` should include:

- `name`
- `passed`
- `details`
- `mismatch_count`
- `max_abs_error`

This should be serializable to JSON without post-processing.

## Validation Scope

### Phase 1: Core Validation

Implement these checks first:

1. Header and session consistency
2. Electrode table and channel ordering
3. Raw ephys data and timestamps
4. Analog data and timestamps
5. DIO event values and timestamps
6. Sample count mapping

### Phase 2: Metadata/YAML-Driven Structural Validation

Add checks for:

1. subject/session fields inserted from YAML
2. camera and acquisition device entries
3. electrode group identities and expected counts
4. existence of associated files and video-file metadata entries

### Out of Scope for First Pass

Do not block the first implementation on:

- exact validation of video payloads
- full position-data reconstruction from external tracking files
- byte-for-byte comparison of HDF5 layout
- NWB Inspector output comparison

Those can be separate checks later.

## Comparison Rules by Data Type

### 1. Header and Session Consistency

Reuse existing logic from:

- `src/trodes_to_nwb/convert_rec_header.py`
- `src/trodes_to_nwb/convert_yaml.py`

Checks:

- rec header is readable
- NWB `session_start_time` matches rec `systemTimeAtCreation`
- `header_device` exists
- header-derived fields stored in `header_device` match the rec header
- YAML/header compatibility passes `validate_yaml_header_electrode_map(...)`

### 2. Electrode Table and Channel Ordering

Use the YAML plus rec header to reconstruct expected mappings:

- `make_hw_channel_map(...)`
- `make_ref_electrode_map(...)`

Checks:

- NWB electrode count matches expected count from YAML/probe metadata
- NWB `hwChan` column matches the expected order used during conversion
- NWB `ref_elect_id` column matches the reference mapping implied by the header/YAML
- behavior-only mode rejects presence of ephys stream validation

### 3. Raw Ephys Validation

Reuse:

- `RecFileDataChunkIterator` from `src/trodes_to_nwb/convert_ephys.py`

Implementation rules:

- read the NWB `hwChan` ordering from the electrodes table
- read `rawScalingToUv` from the rec header, falling back to YAML `raw_data_to_volts`
- reconstruct the expected ephys stream using the same iterator settings as conversion:
  - `stream_id="trodes"`
  - `interpolate_dropped_packets=True`
- compare chunk by chunk to avoid loading full recordings into memory

Acceptance rules:

- shapes must match
- timestamps must match within one sample period
- data comparison should allow small quantization noise caused by ADC-to-microvolt conversion and int16 storage

Recommended ephys comparison rule:

1. convert rec-derived data into the same stored representation used in the NWB ElectricalSeries
2. compare with an absolute tolerance in microvolts

Default tolerance:

- `ephys_tolerance_uv = 1.0`

The validator report should include:

- total compared samples
- mismatch count
- max absolute deviation
- first mismatch location

### 4. Analog Validation

Reuse:

- `add_analog_data(...)` logic from `src/trodes_to_nwb/convert_analog.py`

Checks:

- analog processing module exists
- analog channel IDs encoded in the NWB timeseries description match header-derived IDs
- reconstructed analog data from `ECU_analog` matches NWB data
- timestamps match the shared rec-derived timestamps

This comparison can be exact for stored values unless a specific analog conversion path introduces rounding.

### 5. DIO Validation

Reuse:

- `SpikeGadgetsRawIO.get_digitalsignal(...)`
- `_get_channel_name_map(...)` from `src/trodes_to_nwb/convert_dios.py`

Checks:

- every YAML behavioral event is present in NWB
- NWB timeseries `description` matches the raw Trodes channel ID from YAML
- NWB timeseries `name` and `comments` match the YAML mapping
- DIO state changes match raw rec state changes
- DIO timestamps match within one sample period

### 6. Sample Count Validation

Reuse:

- `add_sample_count(...)` logic from `src/trodes_to_nwb/convert_intervals.py`

Checks:

- `processing["sample_count"]["sample_count"]` exists
- timeseries `data` matches raw Trodes timestamps from the rec files
- timeseries `timestamps` match the reconstructed rec system times

## Production Code Changes

### 1. Add New Module

Create:

- `src/trodes_to_nwb/validate_conversion.py`

This module should contain:

- public API
- per-section validator functions
- chunked array comparison helpers
- report assembly helpers

### 2. Refactor Shared Logic Out of Tests

Current comparison logic in:

- `src/trodes_to_nwb/tests/test_convert.py`

should be treated as a source of acceptance criteria, not kept as the only implementation of comparison behavior.

Refactor reusable comparison helpers into production code where appropriate, then keep tests thin.

### 3. Optional Integration into Conversion Flow

After the standalone validator works, add an optional hook in:

- `src/trodes_to_nwb/convert.py`

Proposed flags for `create_nwbs(...)`:

- `validate_conversion: bool = False`
- `conversion_validation_output_dir: str | None = None`

Behavior:

- after writing the NWB, run `validate_conversion(...)`
- save a report next to the NWB as `<stem>_conversion_validation.json`
- do not make this mandatory for all conversions in the first pass

## Internal Helper Design

Recommended helper layout inside `validate_conversion.py`:

- `_load_validation_context(...)`
- `_validate_header_and_yaml(...)`
- `_validate_electrodes(...)`
- `_validate_ephys(...)`
- `_validate_analog(...)`
- `_validate_dios(...)`
- `_validate_sample_count(...)`
- `_compare_numeric_arrays_chunked(...)`
- `_record_check_result(...)`

The validation context should cache:

- parsed rec headers
- loaded YAML metadata
- loaded device metadata
- reconstructed channel maps
- open NWB file handle

## Testing Plan

Add tests in a new file:

- `src/trodes_to_nwb/tests/test_validate_conversion.py`

### Unit Tests

Add targeted tests for:

1. success case on known-good fixture data
2. ephys mismatch detection with small tolerated differences
3. electrode ordering mismatch detection
4. missing DIO series detection
5. sample count mismatch detection
6. behavior-only validation path

### Integration Test

Flow:

1. run `create_nwbs(...)` on fixture data
2. run `validate_conversion(...)` on the source rec files, generated NWB, and metadata YAML
3. assert `report["passed"] is True`

### Fixture Constraint

The repo already relies on externally downloaded large test data via `DOWNLOAD_DIR`. The new validator tests should use the same mechanism rather than assuming all binary fixtures live in the git tree.

## Implementation Sequence

1. Create `validate_conversion.py` with report scaffolding and input loading.
2. Implement header/YAML compatibility and electrode-table checks.
3. Implement sample-count validation.
4. Implement DIO validation.
5. Implement analog validation.
6. Implement chunked ephys validation with tolerance handling.
7. Add tests for each section.
8. Add optional `create_nwbs(...)` integration and report writing.
9. Update README usage documentation.

## Design Constraints and Risks

### Memory

Full recordings may be large. Ephys and analog validation must be chunked and must not materialize the entire recording in RAM.

### Timing Tolerance

Timestamps should use a tolerance based on sample rate. Default to:

- `1 / 30000` seconds

but derive from the rec sampling rate when possible.

### Ephys Quantization

Raw `.rec` values and NWB-stored values are not expected to match sample-for-sample in ADC units. The validator must compare after applying the same scaling assumptions as conversion, and must report bounded error rather than requiring raw integer identity.

### Metadata Drift

If the YAML used for validation is not the same YAML used during conversion, the validator should fail early with a clear message that the configuration appears inconsistent with the NWB content.

## Deliverables

Implementation of this feature should produce:

1. `src/trodes_to_nwb/validate_conversion.py`
2. tests in `src/trodes_to_nwb/tests/test_validate_conversion.py`
3. optional integration in `src/trodes_to_nwb/convert.py`
4. user-facing documentation update in `README.md`
5. machine-readable validation report output
