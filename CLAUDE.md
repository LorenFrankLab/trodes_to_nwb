# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package that converts SpikeGadgets .rec files (electrophysiology data) to NWB 2.0+ format. The conversion includes ephys data, position tracking, video files, DIO events, and behavioral metadata, with validation for DANDI archive compatibility.

## Development Setup Commands

**Environment Setup:**

```bash
# Use either conda or mamba
conda env create -f environment.yml
# OR
mamba env create -f environment.yml

# Activate environment
conda activate trodes_to_nwb
# OR
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

- Optional: `.rec`: Main recording file
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

## Development Best Practices

### Test-Driven Development (TDD)
- **Write tests first** before implementing new features or fixing bugs
- Follow the TDD cycle: Red (write failing test) → Green (make it pass) → Refactor
- All new functionality must have corresponding unit tests
- Integration tests required for complex workflows involving multiple components

### Quality Assurance Requirements
**All code changes must pass:**

```bash
# Run linting (must pass with no errors)
black .
ruff check .

# Run type checking
mypy src/

# Run full test suite with coverage
pytest --cov=src --cov-report=xml --doctest-modules -v --pyargs trodes_to_nwb
```

### Development Workflow
1. **Feature Branches**: Create feature branches for all changes (`git checkout -b feature/issue-XXX-description`)
2. **Incremental Development**: Make small, focused commits that can be easily reviewed and tested
3. **Continuous Testing**: Run tests frequently during development to catch issues early
4. **Pre-commit Validation**: Ensure linting and tests pass before committing

### Code Quality Standards
- **Test Coverage**: Maintain >90% code coverage for new code
- **Documentation**: All public functions must have docstrings with examples
- **Type Hints**: Use type annotations for all function parameters and return values
- **Error Handling**: Provide clear, actionable error messages with debugging context
- **Performance**: Consider memory usage and processing time for large datasets (17+ hour recordings)

### Pull Request Requirements
Before submitting PRs, ensure:
- [ ] All linting passes (`black .`, `ruff check .`)
- [ ] All tests pass with no failures or warnings
- [ ] New functionality includes unit tests
- [ ] Code coverage remains above current level
- [ ] Documentation updated for user-facing changes
- [ ] Performance impact assessed for large files

### Testing Strategy
**Unit Tests**: Focus on individual functions and classes
- Mock external dependencies and file I/O
- Test edge cases and error conditions
- Validate data transformations and calculations

**Integration Tests**: Test complete workflows
- Use real test data files where possible
- Validate end-to-end conversion pipelines
- Test memory usage on realistic file sizes

**Performance Tests**: Ensure scalability
- Benchmark conversion times for different file sizes
- Monitor memory usage during processing
- Validate parallel processing efficiency

## Important Notes

- Package supports Python >=3.8
- Requires `ffmpeg` for video conversion
- Uses hatch for build system with VCS-based versioning
- Main branch protected, requires PR reviews
