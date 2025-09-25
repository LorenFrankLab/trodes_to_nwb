# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package that converts SpikeGadgets .rec files (electrophysiology data) to NWB 2.0+ format. The conversion includes ephys data, position tracking, video files, DIO events, and behavioral metadata, with validation for DANDI archive compatibility.

## Development Setup Commands

**Environment Setup:**

```bash
mamba env create -f environment.yml
mamba activate trodes_to_nwb
pip install -e .
```

**Testing:**

```bash
pytest --cov=src --cov-report=xml --doctest-modules -v --pyargs trodes_to_nwb
```

**Linting:**

```bash
black .
```

**Build Package:**

```bash
python -m build
twine check dist/*
```

## Architecture

### Core Conversion Pipeline

The main conversion happens in `src/trodes_to_nwb/convert.py` with the `create_nwbs()` function which orchestrates:

1. **File Discovery** (`data_scanner.py`): Scans directories for .rec files and associated data files
2. **Metadata Loading** (`convert_yaml.py`): Loads and validates YAML metadata files
3. **Header Processing** (`convert_rec_header.py`): Extracts device configuration from .rec file headers
4. **Data Conversion**: Modular converters for different data types:
   - `convert_ephys.py`: Raw electrophysiology data
   - `convert_position.py`: Position tracking and video
   - `convert_dios.py`: Digital I/O events
   - `convert_analog.py`: Analog signals
   - `convert_intervals.py`: Epoch and behavioral intervals
   - `convert_optogenetics.py`: Optogenetic stimulation data

### File Structure Requirements

Input files must follow naming convention: `{YYYYMMDD}_{animal}_{epoch}_{tag}.{extension}`

Required files per session:

- `.rec`: Main recording file
- `{date}_{animal}.metadata.yml`: Session metadata
- Optional: `.h264`, `.videoPositionTracking`, `.cameraHWSync`, `.stateScriptLog`

### Metadata System

- Uses YAML metadata files validated against JSON schema (`nwb_schema.json`)
- Probe configurations stored in `device_metadata/probe_metadata/`
- Virus metadata in `device_metadata/virus_metadata/`
- See `docs/yaml_mapping.md` for complete metadata field mapping

### Key Data Processing

- Uses Neo library (`spike_gadgets_raw_io.py`) for .rec file I/O
- Implements chunked data loading (`RecFileDataChunkIterator`) for memory efficiency
- Parallel processing support via Dask for batch conversions
- NWB validation using nwbinspector after conversion

## Testing

- Unit tests in `src/trodes_to_nwb/tests/`
- Integration tests in `tests/integration-tests/`
- Test data downloaded from secure UCSF Box in CI
- Coverage reports uploaded to Codecov

## Release Process

1. Tag release commit (e.g. `v0.1.0`)
2. Push tag to GitHub (triggers PyPI upload)
3. Create GitHub release

## Important Notes

- Package supports Python >=3.8
- Requires `ffmpeg` for video conversion
- Uses hatch for build system with VCS-based versioning
- Main branch protected, requires PR reviews
