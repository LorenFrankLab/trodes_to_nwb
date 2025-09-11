# Headstage Sensor Data Separation

This document describes the new functionality for separating headstage sensor data into individual TimeSeries objects with appropriate units and scaling.

## Overview

Previously, all analog data (including headstage sensors) was combined into a single TimeSeries object stored in the processing module with unit "-1". The new implementation creates separate TimeSeries objects for each sensor type and stores them in the acquisition module with proper units and scaling.

## Sensor Types and Scaling

The following sensor types are automatically detected and processed:

### Accelerometer Data
- **Channels**: `Headstage_AccelX`, `Headstage_AccelY`, `Headstage_AccelZ`
- **Scaling**: Raw values × 0.000061 (converts to g units)
- **Unit**: `g` (gravity units)
- **Description**: Headstage accelerometer data with ±2G full range

### Gyroscope Data  
- **Channels**: `Headstage_GyroX`, `Headstage_GyroY`, `Headstage_GyroZ`
- **Scaling**: Raw values × 0.061 (converts to degrees/second)
- **Unit**: `d/s` (degrees per second)
- **Description**: Headstage gyroscope data with ±2000 dps range

### Magnetometer Data
- **Channels**: `Headstage_MagX`, `Headstage_MagY`, `Headstage_MagZ`
- **Scaling**: No scaling applied (1.0)
- **Unit**: `unspecified`
- **Description**: Headstage magnetometer data

### Analog Input Channels
- **Channels**: `ECU_Ain1`, `ECU_Ain2`, etc., `Controller_Ain1`, etc.
- **Scaling**: No scaling applied (1.0)
- **Unit**: `unspecified` (can be customized in metadata)
- **Description**: Analog input channel data

## Metadata Configuration

### Basic Configuration
The existing `units` section in your YAML metadata still works:

```yaml
units:
  analog: "unspecified"
  behavioral_events: "unspecified"
```

### Advanced Configuration
You can now specify custom units for each sensor type using the new `sensor_units` section:

```yaml
sensor_units:
  accelerometer: "g"      # Custom unit for accelerometer
  gyroscope: "d/s"        # Custom unit for gyroscope  
  magnetometer: "T"       # Custom unit for magnetometer (Tesla)
  analog_input: "V"       # Custom unit for analog inputs (Volts)
```

### Behavioral Events with Units
Individual behavioral events can now specify their own units:

```yaml
behavioral_events:
  - description: Din1
    name: Light_1
    comments: Indicator for reward delivery
    unit: "unspecified"
  - description: ECU_Ain1
    name: Analog_Input_1
    comments: Voltage measurement
    unit: "V"
```

## Output Structure

### New Behavior (Default)
- Separate TimeSeries objects stored in `nwbfile.acquisition`
- Each sensor type gets its own TimeSeries with proper units
- Applied scaling factors for accelerometer and gyroscope data
- Descriptive names and channel information

### Legacy Behavior (Optional)
You can still use the old combined approach by setting `separate_sensor_data=False`:

```python
add_analog_data(nwbfile, rec_files, metadata=metadata, separate_sensor_data=False)
```

This will create the original single TimeSeries in the processing module.

## Example Usage

```python
from trodes_to_nwb.convert_analog import add_analog_data

# Load your metadata with optional sensor_units configuration
metadata = {
    "sensor_units": {
        "accelerometer": "g",
        "gyroscope": "d/s"
    }
}

# Add analog data with new separated sensor approach
add_analog_data(nwbfile, rec_files, metadata=metadata)

# Result: Individual TimeSeries in nwbfile.acquisition:
# - "accelerometer" (scaled data in g units)
# - "gyroscope" (scaled data in d/s units)  
# - "magnetometer" (raw data)
# - "ecu_analog_input" (ECU analog channels)
```

## Benefits

1. **Clear Data Organization**: Each sensor type has its own TimeSeries with descriptive names
2. **Proper Units**: Automatic application of correct physical units
3. **Accurate Scaling**: Raw integer values converted to meaningful physical measurements
4. **Better Documentation**: Channel names and descriptions preserved in TimeSeries
5. **NWB Compliance**: Data stored in appropriate acquisition module
6. **Backwards Compatibility**: Option to use legacy behavior if needed

## Migration Notes

- Existing code will continue to work with the new default behavior
- The new approach stores data in `acquisition` instead of `processing["analog"]`
- If you need the old behavior, use `separate_sensor_data=False`
- Update analysis code to read from `nwbfile.acquisition` instead of `nwbfile.processing["analog"]`