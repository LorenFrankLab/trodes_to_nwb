# Change Log

## [0.1.10] (Unreleased)

### Release Notes

### Bug Fixes

- Fix analog demuxing: offset `interleavedDataIDByte` by device start byte to correctly read multiplexed channels #156
- Add `update_analog_data()` function to patch existing NWB files affected by the demuxing bug #156
- Fix timestamp sequential check: use `> 0` instead of truthy check so decreasing timestamps are detected
- Fix `date_of_birth` validation: was calling `.utcnow().isoformat()` which discarded the actual value and substituted the current time
- Fix JSON Schema version mismatch: code used `Draft202012Validator` but schema declares draft-07
- Halt conversion on invalid metadata instead of silently continuing with bad data
- Fix `convert_optogenetics.py` importing from test utilities in production code
- Remove debug `print()` statements from `spike_gadgets_raw_io.py`

### General

- Update conversions for use of `pynwb>=3.1` #153, #155
- Replace Black with ruff for linting and formatting
- Replace `print()` statements with `logging` calls in `convert.py`, `convert_optogenetics.py`, and `spike_gadgets_raw_io.py`
- Fix log levels: `logger.info("ERROR: ...")` changed to `logger.warning()` with proper messages
- Add error guards for missing header elements (`GlobalConfiguration`, `HardwareConfiguration`, `ECU`)
- Fix wheel packaging: test data exclusion path corrected
- Export `create_nwbs` from package `__init__.py`

### Testing

- Strengthen analog tests: compare all 22 channels across all timepoints (was 1 of 22)
- Strengthen ephys tests: compare all 32 channels across all timepoints (was 1 of 32)
- Strengthen end-to-end tests: full-array comparisons in `compare_nwbfiles`
- Fix 3 silent test failures: add missing `assert` on `np.allclose()` calls in position and optogenetics tests
- Add dtype assertions to analog and ephys tests
- Migrate analog and ephys tests to use `tmp_path` fixture for reliable cleanup
- Add tests for metadata validation error propagation, `date_of_birth` serialization, and missing channel map errors

### Optogenetics

- fix hfpy write error when different number of spatial node regions between epochs #135
- Run `add_optogenetic_epochs` in the create nwb function #135
