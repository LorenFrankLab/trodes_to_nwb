"""Test the new headstage sensor separation functionality"""

import numpy as np
from trodes_to_nwb.convert_analog import _categorize_sensor_channels, _create_sensor_timeseries, SENSOR_TYPE_CONFIG


def test_categorize_sensor_channels():
    """Test that sensor channels are correctly categorized by type"""
    # Test with typical headstage channel names
    test_channels = [
        "Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ",
        "Headstage_GyroX", "Headstage_GyroY", "Headstage_GyroZ", 
        "Headstage_MagX", "Headstage_MagY", "Headstage_MagZ",
        "ECU_Ain1", "ECU_Ain2", "Controller_Ain1",
        "Other_Channel"
    ]
    
    categorized = _categorize_sensor_channels(test_channels)
    
    # Check accelerometer channels
    assert "accelerometer" in categorized
    assert sorted(categorized["accelerometer"]) == ["Headstage_AccelX", "Headstage_AccelY", "Headstage_AccelZ"]
    
    # Check gyroscope channels
    assert "gyroscope" in categorized
    assert sorted(categorized["gyroscope"]) == ["Headstage_GyroX", "Headstage_GyroY", "Headstage_GyroZ"]
    
    # Check magnetometer channels  
    assert "magnetometer" in categorized
    assert sorted(categorized["magnetometer"]) == ["Headstage_MagX", "Headstage_MagY", "Headstage_MagZ"]
    
    # Check analog input channels
    assert "analog_input" in categorized
    assert sorted(categorized["analog_input"]) == ["Controller_Ain1", "ECU_Ain1", "ECU_Ain2"]
    
    # Check uncategorized channels
    assert "other" in categorized
    assert categorized["other"] == ["Other_Channel"]
    
    print("✓ Channel categorization test passed")


def test_create_sensor_timeseries():
    """Test TimeSeries creation with proper scaling and units"""
    # Test accelerometer scaling
    test_data = np.array([[1000, 2000], [1500, 2500]], dtype=np.int16)  # 2 timepoints, 2 channels
    test_timestamps = np.array([0.0, 0.1])
    
    # Test accelerometer
    accel_ts = _create_sensor_timeseries(
        sensor_type="accelerometer",
        channel_names=["Headstage_AccelX", "Headstage_AccelY"],
        data=test_data,
        timestamps=test_timestamps
    )
    
    # Check scaling is applied (should be original * 0.000061)
    expected_data = test_data * 0.000061
    assert np.allclose(accel_ts.data[:], expected_data)
    assert accel_ts.unit == "g"
    assert "accelerometer" in accel_ts.description.lower()
    assert "Headstage_AccelX" in accel_ts.description
    
    # Test gyroscope
    gyro_ts = _create_sensor_timeseries(
        sensor_type="gyroscope", 
        channel_names=["Headstage_GyroX"],
        data=test_data[:, [0]],  # Single channel
        timestamps=test_timestamps
    )
    
    # Check scaling is applied (should be original * 0.061)
    expected_gyro_data = test_data[:, [0]] * 0.061
    assert np.allclose(gyro_ts.data[:], expected_gyro_data)
    assert gyro_ts.unit == "d/s"
    
    # Test with custom metadata units
    metadata = {"sensor_units": {"accelerometer": "custom_g"}}
    custom_ts = _create_sensor_timeseries(
        sensor_type="accelerometer",
        channel_names=["Headstage_AccelX"],
        data=test_data[:, [0]],
        timestamps=test_timestamps,
        metadata=metadata
    )
    assert custom_ts.unit == "custom_g"
    
    print("✓ TimeSeries creation test passed")


def test_sensor_type_config():
    """Test that sensor type configuration is complete"""
    required_keys = ['pattern', 'scaling_factor', 'unit', 'description']
    
    for sensor_type, config in SENSOR_TYPE_CONFIG.items():
        for key in required_keys:
            assert key in config, f"Missing {key} in {sensor_type} config"
            
        # Test that patterns compile
        import re
        try:
            re.compile(config['pattern'])
        except re.error:
            assert False, f"Invalid regex pattern for {sensor_type}: {config['pattern']}"
    
    print("✓ Sensor type configuration test passed")


if __name__ == "__main__":
    test_categorize_sensor_channels()
    test_create_sensor_timeseries()
    test_sensor_type_config()
    print("\n✓ All tests passed!")