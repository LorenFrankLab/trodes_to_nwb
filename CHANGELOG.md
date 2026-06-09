# Change Log

## [0.1.10] (Unreleased)

### Release Notes

### General

- Update conversions for use of `pynwb>=3.1` #153, #155

### Analog / headstage sensors

- **Breaking:** analog data is now written as separate `TimeSeries` carrying
  physical units in `nwbfile.acquisition` (one per sensor type:
  `accelerometer`, `gyroscope`, `magnetometer`, `analog_input`, and `other`
  for unrecognized channels), replacing the single combined
  `processing["analog"]["analog"]["analog"]` stream. Headstage accelerometer
  and gyroscope data now carry correct physical units (`g`, `d/s`) via each
  `TimeSeries.conversion` factor. Data remains lazy/chunked; stored values are
  raw int16 and `ts.data[:]` returns raw counts — multiply by `ts.conversion`
  (or read through a conversion-aware API) to get physical units. Optional
  `sensor_units` metadata can override a sensor's unit *label*. #19

  Migration: code that read `nwbfile.processing["analog"]["analog"]["analog"]`
  must instead read the relevant `nwbfile.acquisition[...]` stream. Existing
  files in the old layout can still be repaired with `update_analog_data`.

### Optogenetics

- fix hfpy write error when different number of spatial node regions between epochs #135
- Run `add_optogenetic_epochs` in the create nwb function #135
