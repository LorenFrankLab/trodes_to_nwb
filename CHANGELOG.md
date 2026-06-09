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
  and gyroscope data now carry physical units in SI (`m/s^2`, `rad/s`, the NWB
  convention) via each `TimeSeries.conversion` factor; the native sensor scale
  (±2 g, ±2000 deg/s) is noted in each stream's description. Stored values are
  raw int16 and `ts.data[:]` returns raw counts — multiply by `ts.conversion`
  to get the SI value.
  Optional `sensor_units` metadata can override a sensor's unit *label*. #19
- **Headstage IMU stored at its true rate.** The multiplexed IMU sensors are
  transmitted at the sensor's native rate (~100 Hz) and expanded to the
  acquisition rate by sample-and-hold in the `.rec` stream. They are now
  decimated back to their true rate using the per-packet update flags and
  stored with explicit `timestamps` (accelerometer and gyroscope are sampled on
  interleaved schedules, so each carries its own timestamps). A sensor that
  never updates (disabled) is omitted with a warning. ECU analog inputs, which
  are genuinely continuous, remain at the full acquisition rate and lazy/chunked.

  Migration: code that read `nwbfile.processing["analog"]["analog"]["analog"]`
  must instead read the relevant `nwbfile.acquisition[...]` stream. Note IMU
  streams are now ~100 Hz with their own `timestamps`, not the acquisition rate.
  Existing files in the old layout can still be repaired with `update_analog_data`.

### Optogenetics

- fix hfpy write error when different number of spatial node regions between epochs #135
- Run `add_optogenetic_epochs` in the create nwb function #135
