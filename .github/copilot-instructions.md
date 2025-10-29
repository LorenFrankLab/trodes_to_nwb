# trodes_to_nwb

trodes_to_nwb converts data from SpikeGadgets .rec files to the NWB 2.0+ Data Format. It validates NWB files using the NWB Inspector for compatibility with the DANDI archive.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- Bootstrap the development environment:
  - `conda env create -f environment.yml` -- takes 5 minutes to complete. NEVER CANCEL. Set timeout to 10+ minutes.
  - `conda activate trodes_to_nwb`
  - **CRITICAL**: pip install will fail due to network timeouts to PyPI. Use the PYTHONPATH workaround instead:
  - `export PYTHONPATH=/home/runner/work/trodes_to_nwb/trodes_to_nwb/src:$PYTHONPATH`
  
### Development Workflow
- **Working code setup**: The package works without pip install using PYTHONPATH:
  - `cd /home/runner/work/trodes_to_nwb/trodes_to_nwb`
  - `source /usr/share/miniconda/etc/profile.d/conda.sh && conda activate trodes_to_nwb`
  - `export PYTHONPATH=/home/runner/work/trodes_to_nwb/trodes_to_nwb/src:$PYTHONPATH`
  - `python -c "import trodes_to_nwb; print('Import successful')"`

### Building and Testing
- **Run tests**: `python -m pytest src/trodes_to_nwb/tests/ -v` -- takes 1-2 minutes. NEVER CANCEL. Set timeout to 5+ minutes.
  - **NOTE**: Many tests fail due to missing large .rec data files (normal for development)
  - Integration tests that pass validate core functionality without data files
  - Tests that require .rec files will show `FileNotFoundError` (expected)
- **Lint code**: `black --check --diff src/` -- takes < 1 second
- **Format code**: `black src/` to auto-format files

### Validation Steps
- **Always run the basic functionality test** after making changes:
  - Copy the validation script from `/tmp/test_basic_functionality.py` if available
  - Or run: `python -c "from trodes_to_nwb.convert import create_nwbs; import trodes_to_nwb; print(f'Package working, version: {trodes_to_nwb.__version__}')"` (takes ~2 seconds)
- **Always run black formatting** before committing: `black src/`
- **Test core imports** work: `from trodes_to_nwb.convert import create_nwbs`

## Key Information

### Package Structure
- **Main package**: `src/trodes_to_nwb/`
- **Core conversion**: `src/trodes_to_nwb/convert.py` - contains `create_nwbs()` function
- **Tests**: `src/trodes_to_nwb/tests/` - unit and integration tests
- **Documentation**: `docs/`, `notebooks/conversion_tutorial.ipynb`
- **Metadata**: Device configs in `src/trodes_to_nwb/device_metadata/`

### Critical Installation Notes
- **NEVER use pip install -e .** - it will timeout connecting to PyPI for ffmpeg package
- **FFmpeg is available via conda** - don't try to install via pip
- **Use PYTHONPATH approach** for development - fully functional
- **Environment creation partially fails** - conda packages install but pip dependencies timeout (this is expected)

### Main Function Usage
```python
from trodes_to_nwb.convert import create_nwbs

# Core parameters:
# path: pathlib.Path - directory containing .rec files and metadata YAML
# output_dir: str - where to save .nwb files (default: '/stelmo/nwb/raw')
# header_reconfig_path: optional path to header reconfiguration file  
# device_metadata_paths: optional list of device metadata YAML files
# convert_video: bool - whether to convert .h264 to .mp4 (requires ffmpeg)
# n_workers: int - parallel processing workers (default: 1)
# query_expression: str - filter which files to convert (e.g., "animal == 'sample'")
```

### File Naming Convention
Data files must follow: `{YYYYMMDD}_{animal}_{epoch}_{tag}.{extension}`
- `.rec`: binary ephys recording
- `.h264`: video file  
- `.videoPositionTracking`: position tracking data
- `.cameraHWSync`: position timestamps
- `.stateScriptLog`: experimental parameters

Metadata file: `{YYYYMMDD}_{animal}_metadata.yml`

### GitHub Workflows
- **Tests**: `.github/workflows/test_package_build.yml` - builds package, runs tests
- **Linting**: `.github/workflows/lint.yml` - runs black formatting check
- **CI downloads test data** from UCSF Box during workflow runs
- **Coverage reports** upload to Codecov

## Common Tasks

### Repository Overview
```
├── src/trodes_to_nwb/          # Main package
│   ├── convert.py              # Core conversion functions
│   ├── tests/                  # Test suite
│   └── device_metadata/        # Device configurations
├── docs/                       # Documentation
├── notebooks/                  # Jupyter tutorials
├── environment.yml             # Conda environment specification
└── pyproject.toml             # Package configuration
```

### Timing Expectations
- **Environment creation**: 5 minutes (conda packages only, pip fails)
- **Package import**: 2 seconds
- **Test suite**: 1-2 minutes (many tests skip due to missing data)
- **Black linting**: < 1 second
- **Basic functionality validation**: 2 seconds

### Known Issues and Workarounds
- **pip install timeouts**: Use PYTHONPATH instead of pip install
- **Missing test data**: Large .rec files not in repo, downloaded in CI
- **ffmpeg dependency**: Use conda version, not pip version
- **Network connectivity**: PyPI access may be limited in some environments

### Example Development Session
```bash
cd /home/runner/work/trodes_to_nwb/trodes_to_nwb
conda activate trodes_to_nwb
export PYTHONPATH=/home/runner/work/trodes_to_nwb/trodes_to_nwb/src:$PYTHONPATH

# Test your changes
python -c "from trodes_to_nwb.convert import create_nwbs; print('Working')"
black src/
python -m pytest src/trodes_to_nwb/tests/ -x  # Stop on first failure
```

### Before Committing
- Run `black src/` to format code
- Verify imports work: `python -c "import trodes_to_nwb"`
- Consider running a few tests: `python -m pytest src/trodes_to_nwb/tests/integration-tests/ -v`