"""
Unit tests for test_sensor/modbus_client.py and test_sensor/sensor_reader.py.
All serial I/O is mocked — no real COM port or sensor needed.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from test_sensor.modbus_client import ModbusClient
from test_sensor.sensor_reader import (
    SensorReader,
    SensorReading,
    scale_registers,
    validate_reading,
    _twos_complement_16,
)


# ═══════════════════════════════════════════════════════════
# scale_registers — pure function tests
# ═══════════════════════════════════════════════════════════

class TestScaleRegisters:
    """Test the register-to-physical-value conversion logic."""

    def test_normal_values(self):
        """Example from sensor manual: positive temperature, normal ranges."""
        raw = [250, 356, 1234, 686, 135, 138, 142]
        result = scale_registers(raw)
        assert result["temperature"] == pytest.approx(25.0)
        assert result["moisture"] == pytest.approx(35.6)
        assert result["ec"] == pytest.approx(1234.0)
        assert result["ph"] == pytest.approx(6.86)
        assert result["nitrogen"] == pytest.approx(135.0)
        assert result["phosphorus"] == pytest.approx(138.0)
        assert result["potassium"] == pytest.approx(142.0)

    def test_negative_temperature_twos_complement(self):
        """0xFFDD = 65501 → two's complement → -35 → /10 → -3.5°C."""
        raw = [0xFFDD, 100, 0, 700, 0, 0, 0]
        result = scale_registers(raw)
        assert result["temperature"] == pytest.approx(-3.5)

    def test_zero_values(self):
        """All zeros should produce valid zero output."""
        raw = [0, 0, 0, 0, 0, 0, 0]
        result = scale_registers(raw)
        assert result["temperature"] == 0.0
        assert result["moisture"] == 0.0
        assert result["ec"] == 0.0
        assert result["ph"] == 0.0

    def test_max_boundary_values(self):
        """Test upper-range sensor values."""
        raw = [800, 1000, 20000, 900, 2999, 2999, 2999]
        result = scale_registers(raw)
        assert result["temperature"] == pytest.approx(80.0)
        assert result["moisture"] == pytest.approx(100.0)
        assert result["ec"] == pytest.approx(20000.0)
        assert result["ph"] == pytest.approx(9.0)
        assert result["nitrogen"] == pytest.approx(2999.0)

    def test_too_few_registers_raises(self):
        """Should raise ValueError if fewer than 7 registers provided."""
        with pytest.raises(ValueError, match="Expected 7"):
            scale_registers([100, 200, 300])


class TestTwosComplement:
    def test_positive(self):
        assert _twos_complement_16(250) == 250

    def test_negative(self):
        assert _twos_complement_16(0xFFDD) == -35

    def test_zero(self):
        assert _twos_complement_16(0) == 0

    def test_boundary_positive(self):
        assert _twos_complement_16(0x7FFF) == 32767

    def test_boundary_negative(self):
        assert _twos_complement_16(0x8000) == -32768


class TestValidateReading:
    def test_valid_values(self):
        values = {
            "temperature": 25.0, "moisture": 50.0, "ec": 1000.0,
            "ph": 6.5, "nitrogen": 100.0, "phosphorus": 80.0, "potassium": 150.0,
        }
        assert validate_reading(values) is True

    def test_temperature_out_of_range(self):
        values = {
            "temperature": 100.0, "moisture": 50.0, "ec": 1000.0,
            "ph": 6.5, "nitrogen": 100.0, "phosphorus": 80.0, "potassium": 150.0,
        }
        assert validate_reading(values) is False

    def test_ph_out_of_range_low(self):
        values = {
            "temperature": 25.0, "moisture": 50.0, "ec": 1000.0,
            "ph": 2.0, "nitrogen": 100.0, "phosphorus": 80.0, "potassium": 150.0,
        }
        assert validate_reading(values) is False

    def test_npk_out_of_range(self):
        values = {
            "temperature": 25.0, "moisture": 50.0, "ec": 1000.0,
            "ph": 6.5, "nitrogen": 5000.0, "phosphorus": 80.0, "potassium": 150.0,
        }
        assert validate_reading(values) is False


# ═══════════════════════════════════════════════════════════
# ModbusClient — mocked serial tests
# ═══════════════════════════════════════════════════════════

class TestModbusClient:
    """Tests for ModbusClient with mocked pymodbus serial client."""

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_connect_success(self, MockSerial):
        mock_instance = MockSerial.return_value
        mock_instance.connect.return_value = True
        client = ModbusClient(port="COM99")
        assert client.connect() is True
        MockSerial.assert_called_once()

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_connect_failure(self, MockSerial):
        mock_instance = MockSerial.return_value
        mock_instance.connect.return_value = False
        client = ModbusClient(port="COM99")
        assert client.connect() is False

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_disconnect(self, MockSerial):
        mock_instance = MockSerial.return_value
        mock_instance.connect.return_value = True
        client = ModbusClient(port="COM99")
        client.connect()
        client.disconnect()
        mock_instance.close.assert_called_once()
        assert client._client is None

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_read_registers_success(self, MockSerial):
        mock_instance = MockSerial.return_value
        mock_instance.connect.return_value = True
        mock_instance.is_socket_open.return_value = True

        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_result.registers = [250, 356, 1234, 686, 135, 138, 142]
        mock_instance.read_holding_registers.return_value = mock_result

        client = ModbusClient(port="COM99")
        client.connect()
        regs = client.read_registers()
        assert regs == [250, 356, 1234, 686, 135, 138, 142]

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_read_registers_error_response(self, MockSerial):
        mock_instance = MockSerial.return_value
        mock_instance.connect.return_value = True
        mock_instance.is_socket_open.return_value = True
        mock_result = MagicMock()
        mock_result.isError.return_value = True
        mock_instance.read_holding_registers.return_value = mock_result

        client = ModbusClient(port="COM99")
        client.connect()
        assert client.read_registers() is None

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_read_not_connected(self, MockSerial):
        client = ModbusClient(port="COM99")
        assert client.read_registers() is None

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_ping_success(self, MockSerial):
        mock_instance = MockSerial.return_value
        mock_instance.connect.return_value = True
        mock_instance.is_socket_open.return_value = True
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_instance.read_holding_registers.return_value = mock_result

        client = ModbusClient(port="COM99")
        client.connect()
        assert client.ping() is True

    @patch("test_sensor.modbus_client.ModbusSerialClient")
    def test_ping_not_connected(self, MockSerial):
        client = ModbusClient(port="COM99")
        assert client.ping() is False


# ═══════════════════════════════════════════════════════════
# SensorReader — integration with mocked client
# ═══════════════════════════════════════════════════════════

class TestSensorReader:
    """Tests for SensorReader using a mocked ModbusClient."""

    def _make_mock_client(self, registers=None):
        client = MagicMock(spec=ModbusClient)
        client.is_connected = True
        client.read_registers.return_value = registers
        client.connect.return_value = True
        return client

    def test_read_success(self):
        raw = [250, 356, 1234, 686, 135, 138, 142]
        mock_client = self._make_mock_client(registers=raw)
        reader = SensorReader(client=mock_client)

        reading = reader.read()
        assert reading is not None
        assert isinstance(reading, SensorReading)
        assert reading.temperature == pytest.approx(25.0)
        assert reading.moisture == pytest.approx(35.6)
        assert reading.ph == pytest.approx(6.86)
        assert reading.is_valid is True

    def test_read_returns_none_on_comm_failure(self):
        mock_client = self._make_mock_client(registers=None)
        reader = SensorReader(client=mock_client)
        assert reader.read() is None

    def test_read_out_of_range_flagged(self):
        """Values out of spec range should set is_valid=False."""
        raw = [5000, 356, 1234, 686, 135, 138, 142]  # 500°C - out of range
        mock_client = self._make_mock_client(registers=raw)
        reader = SensorReader(client=mock_client)

        reading = reader.read()
        assert reading is not None
        assert reading.is_valid is False

    def test_connect_disconnect(self):
        mock_client = self._make_mock_client()
        reader = SensorReader(client=mock_client)
        reader.connect()
        mock_client.connect.assert_called_once()
        reader.disconnect()
        mock_client.disconnect.assert_called_once()

