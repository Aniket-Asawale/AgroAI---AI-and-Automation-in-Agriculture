"""

Simulates a Modbus RTU slave on a virtual COM port for development/testing.

Usage:
    1. Open VSPE → Create a "Pair" of virtual COM ports (e.g., COM5 ↔ COM6)
    2. Run this script on one port:  python tools/sensor_simulator.py --port COM6
    3. Run the app on the other:     SENSOR_COM_PORT=COM5 python main.py

The simulator responds to Modbus function code 0x03 (Read Holding Registers)
with realistic soil sensor data that gradually varies over time.


"""

import argparse
import json
import math
import random
import struct
import time
from datetime import datetime
from pathlib import Path
import serial

# Weather cache path (shared with tools/weather.py and API)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WEATHER_CACHE = _PROJECT_ROOT / "weather_cache.json"
_WEATHER_CONFIG = _PROJECT_ROOT / "weather_config.json"

# Default soil profile (used when no soil type specified)
_DEFAULT_SOIL = {
    "water_retention": 0.65, "drainage_rate": 0.6,
    "ph_base": 6.8, "ph_range": (5.0, 8.0),
    "ec_base": 1200, "ec_range": (300, 3000),
    "n_base": 120, "p_base": 85, "k_base": 155,
}


def calculate_crc16(data: bytes) -> bytes:
    """Calculate Modbus RTU CRC-16."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return struct.pack("<H", crc)


# ─── Environment State (persists across reads) ───
class FieldEnvironment:
    """
    Simulates a real agricultural field with:
    - Real weather data from Open-Meteo API (via shared cache file)
    - Day/night temperature cycle based on actual clock time
    - Weather-driven rain events that raise moisture, lower EC (dilution), and shift pH
    - Slow nutrient depletion with occasional fertilizer application
    - Sensor noise and thermal drift

    Weather data is read from weather_cache.json (written by tools/weather.py).
    If no cache exists, falls back to clock-based simulation.
    """

    def __init__(self, soil_type: str = "", sensor_index: int = 0):
        """
        Args:
            soil_type: Key into SOIL_PROFILES (e.g. "Alluvial", "Black (Regur)").
                       Falls back to _DEFAULT_SOIL if empty/unknown.
            sensor_index: 0-based index for noise seed differentiation
                          (allows multiple sensors in the same environment).
        """
        # Load soil profile
        self._soil_type = soil_type
        self._sensor_index = sensor_index
        try:
            from tools.weather import SOIL_PROFILES
            self._soil = SOIL_PROFILES.get(soil_type, _DEFAULT_SOIL)
        except ImportError:
            self._soil = _DEFAULT_SOIL

        # Weather state
        self.is_raining = False
        self.rain_intensity = 0.0       # 0.0 – 1.0 (drizzle → heavy)
        self.rain_remaining_s = 0.0     # seconds left in current rain event
        self.hours_since_rain = 5.0     # affects soil drying rate

        # Soil state — initialized from soil profile baselines
        wr = self._soil["water_retention"]
        self.soil_moisture = 25.0 + wr * 30.0   # higher retention → higher base moisture
        self.soil_ec = float(self._soil["ec_base"])
        self.soil_ph = float(self._soil["ph_base"])
        self.soil_n = float(self._soil["n_base"])
        self.soil_p = float(self._soil["p_base"])
        self.soil_k = float(self._soil["k_base"])

        # Last update timestamp
        self._last_t = time.time()

        # Weather cache state
        self._weather_data = None       # Cached Open-Meteo data
        self._weather_read_at = 0       # When we last read the cache file
        self._weather_check_interval = 300  # Re-read cache file every 5 min

        # Per-sensor noise seed for repeatable variation
        self._rng = random.Random(hash((soil_type, sensor_index)))

    def _read_weather_cache(self) -> dict | None:
        """Read weather data from shared cache file (written by weather.py)."""
        now = time.time()
        if now - self._weather_read_at < self._weather_check_interval and self._weather_data:
            return self._weather_data
        self._weather_read_at = now
        try:
            if _WEATHER_CACHE.exists():
                raw = json.loads(_WEATHER_CACHE.read_text(encoding="utf-8"))
                # Accept data up to 6 hours old (generous TTL for simulator)
                if now - raw.get("_fetched_at", 0) < 21600:
                    self._weather_data = raw
                    return raw
        except (json.JSONDecodeError, OSError):
            pass
        return self._weather_data  # Return stale data if available

    def _update_weather(self, dt: float):
        """Use real weather data if available, otherwise simulate rain events."""
        weather = self._read_weather_cache()
        if weather:
            # Use real rain status from Open-Meteo
            self.is_raining = weather.get("is_raining", False)
            rain_mm = weather.get("rain", 0) + weather.get("precipitation", 0)
            # Map rainfall mm to intensity (0-1): 0.5mm=light, 5mm+=heavy
            self.rain_intensity = min(1.0, rain_mm / 5.0) if self.is_raining else 0.0
            if not self.is_raining:
                self.hours_since_rain += dt / 3600.0
            else:
                self.hours_since_rain = 0.0
        else:
            # Fallback: simulated rain events
            if self.is_raining:
                self.rain_remaining_s -= dt
                if self.rain_remaining_s <= 0:
                    self.is_raining = False
                    self.rain_intensity = 0.0
                    self.hours_since_rain = 0.0
            else:
                self.hours_since_rain += dt / 3600.0
                rain_chance = 0.0002 * dt
                if random.random() < rain_chance:
                    self.is_raining = True
                    self.rain_intensity = random.uniform(0.2, 1.0)
                    self.rain_remaining_s = random.uniform(300, 3600)

    def _get_air_temperature(self) -> float:
        """
        Temperature from Open-Meteo if available, otherwise clock-based simulation.
        Real weather temp gets small diurnal modulation and sensor noise.
        """
        weather = self._read_weather_cache()
        if weather and weather.get("temperature") is not None:
            # Use real temperature as base, add small diurnal variation
            real_temp = weather["temperature"]
            now = datetime.now()
            hour_frac = now.hour + now.minute / 60.0
            # Small ±1.5°C diurnal swing around real temp
            daily_mod = 1.5 * math.cos(2 * math.pi * (hour_frac - 14.0) / 24.0)
            temp = real_temp + daily_mod
        else:
            # Fallback: pure simulation
            now = datetime.now()
            hour_frac = now.hour + now.minute / 60.0
            daily_cycle = math.cos(2 * math.pi * (hour_frac - 14.0) / 24.0)
            base_temp = 28.0
            amplitude = 5.0
            temp = base_temp + amplitude * daily_cycle
            if self.is_raining:
                temp -= 2.0 * self.rain_intensity

        # Small random fluctuation (sensor noise + micro-gusts)
        temp += random.gauss(0, 0.2)
        return temp

    def _get_real_humidity(self) -> float | None:
        """Return real humidity from weather cache, or None."""
        weather = self._read_weather_cache()
        if weather and weather.get("humidity") is not None:
            return weather["humidity"]
        return None

    def update(self, dt: float):
        """Advance the field environment by dt seconds. Soil profile affects all dynamics."""
        self._update_weather(dt)

        air_temp = self._get_air_temperature()
        real_humidity = self._get_real_humidity()
        soil = self._soil

        # ── Moisture ──
        # Water retention affects how much moisture the soil holds;
        # drainage_rate affects how fast it loses moisture.
        wr = soil["water_retention"]       # 0-1
        dr = soil["drainage_rate"]         # 0-1

        if real_humidity is not None:
            # Target moisture scales with soil retention
            target_moisture = 10.0 + real_humidity * (0.4 + wr * 0.4)
            if self.is_raining:
                target_moisture += self.rain_intensity * (10.0 + wr * 10.0)
            drift_rate = 0.0005 * dt
            self.soil_moisture += (target_moisture - self.soil_moisture) * drift_rate
        else:
            if self.is_raining:
                self.soil_moisture += self.rain_intensity * 0.05 * wr * dt
            else:
                evap_rate = 0.002 * dr * max(0, (air_temp - 20) / 15)
                self.soil_moisture -= evap_rate * dt

        self.soil_moisture = max(8.0, min(95.0, self.soil_moisture))

        # ── EC (Electrical Conductivity) ──
        ec_lo, ec_hi = soil["ec_range"]
        if self.is_raining:
            self.soil_ec -= self.rain_intensity * 0.3 * dr * dt
        else:
            self.soil_ec += 0.01 * (1.0 - dr) * dt  # low-drainage soils concentrate salts faster
        if real_humidity is not None and real_humidity > 70:
            self.soil_ec -= 0.005 * dt
        self.soil_ec = max(ec_lo, min(ec_hi, self.soil_ec))

        # ── pH ──
        ph_lo, ph_hi = soil["ph_range"]
        ph_base = soil["ph_base"]
        if self.is_raining:
            self.soil_ph -= 0.0001 * self.rain_intensity * dt
        else:
            # Drift toward soil-specific natural pH
            self.soil_ph += 0.00002 * (ph_base - self.soil_ph) * dt
        self.soil_ph = max(ph_lo, min(ph_hi, self.soil_ph))

        # ── NPK (nutrients) ──
        # Drainage rate affects leaching — sandy/laterite soils lose nutrients faster
        leach_factor = 0.5 + dr * 0.5  # 0.5 – 1.0
        self.soil_n -= 0.001 * leach_factor * dt
        self.soil_p -= 0.0003 * leach_factor * dt
        self.soil_k -= 0.0005 * leach_factor * dt
        if self.is_raining:
            self.soil_n -= 0.002 * self.rain_intensity * leach_factor * dt

        # Occasional "fertilizer event" — very rare
        if random.random() < 0.00001 * dt:
            self.soil_n += random.uniform(20, 50)
            self.soil_p += random.uniform(10, 25)
            self.soil_k += random.uniform(15, 35)

        self.soil_n = max(10, min(300, self.soil_n))
        self.soil_p = max(5, min(250, self.soil_p))
        self.soil_k = max(10, min(300, self.soil_k))

    def read(self) -> list[int]:
        """
        Produce a single sensor reading as 7 raw uint16 register values.
        Includes realistic sensor noise on top of the environment state.
        """
        now = time.time()
        dt = now - self._last_t
        self._last_t = now

        self.update(dt)

        # Soil temperature lags air temperature and is more stable
        air_temp = self._get_air_temperature()
        # Soil at 10-15cm depth: dampened, delayed version of air temp
        soil_temp = air_temp * 0.7 + 28.0 * 0.3  # heavily smoothed toward mean

        # Add sensor-level noise (ADC jitter) — uses per-sensor RNG
        rng = self._rng
        temp_reading = soil_temp + rng.gauss(0, 0.15)
        moist_reading = self.soil_moisture + rng.gauss(0, 0.3)
        ec_reading = self.soil_ec + rng.gauss(0, 5)
        ph_reading = self.soil_ph + rng.gauss(0, 0.01)
        n_reading = self.soil_n + rng.gauss(0, 1.0)
        p_reading = self.soil_p + rng.gauss(0, 0.8)
        k_reading = self.soil_k + rng.gauss(0, 1.2)

        # Convert to uint16 register format
        return [
            int(temp_reading * 10) & 0xFFFF,   # temp ×10
            int(max(0, moist_reading) * 10),    # moisture ×10
            int(max(0, ec_reading)),             # EC raw
            int(max(0, ph_reading) * 100),       # pH ×100
            int(max(0, n_reading)),
            int(max(0, p_reading)),
            int(max(0, k_reading)),
        ]


# ─── Multi-Sensor Environment Manager ───
# Maintains a FieldEnvironment per (city, sensor_index) pair.
# The "active" city determines which environments produce readings.
_environments: dict[str, list[FieldEnvironment]] = {}
_active_city: str = ""


def _get_active_city() -> str:
    """Read the currently selected city from weather_config.json."""
    global _active_city
    try:
        if _WEATHER_CONFIG.exists():
            cfg = json.loads(_WEATHER_CONFIG.read_text(encoding="utf-8"))
            _active_city = cfg.get("city", "Kolhapur")
        elif not _active_city:
            _active_city = "Kolhapur"
    except (json.JSONDecodeError, OSError):
        if not _active_city:
            _active_city = "Kolhapur"
    return _active_city


def _get_city_environment(city_name: str) -> FieldEnvironment:
    """
    Get or create a single FieldEnvironment for a city.
    Checks DB-exported locations cache first, then falls back to hardcoded CITIES.
    """
    if city_name in _environments:
        return _environments[city_name][0]

    # Find city config to get soil type
    soil_type = ""

    # Try DB-sourced locations cache (written by the API on location changes)
    _locations_cache = _PROJECT_ROOT / "locations_cache.json"
    found = False
    try:
        if _locations_cache.exists():
            locs = json.loads(_locations_cache.read_text(encoding="utf-8"))
            for loc in locs:
                if loc.get("name", "").lower() == city_name.lower():
                    soil_type = loc.get("soil_type", "")
                    found = True
                    break
    except (json.JSONDecodeError, OSError):
        pass

    # Fallback: hardcoded CITIES list
    if not found:
        try:
            from tools.weather import CITIES
            for c in CITIES:
                if c["name"].lower() == city_name.lower():
                    soil_type = c.get("soil_type", "")
                    break
        except ImportError:
            pass

    env = FieldEnvironment(soil_type=soil_type, sensor_index=0)
    _environments[city_name] = [env]
    return env


def get_active_readings() -> list[tuple[str, int, list[int]]]:
    """
    Get a single reading from the active city's sensor.
    Returns list of (sensor_id_suffix, sensor_index, raw_registers).
    """
    city = _get_active_city()
    env = _get_city_environment(city)
    raw = env.read()
    suffix = f"{city.upper()}-001"
    return [(suffix, 0, raw)]


# Legacy: single-field fallback for Modbus slave (uses active city's sensor)
def _get_field() -> FieldEnvironment:
    city = _get_active_city()
    return _get_city_environment(city)


def handle_request(request: bytes, sensor_values: list[int], silent: bool = False) -> bytes | None:
    """
    Parse a Modbus RTU request and return a response.
    Only handles Function 0x03 (Read Holding Registers).
    """
    if len(request) < 8:
        return None

    slave_addr = request[0]
    func_code = request[1]

    # Verify CRC
    received_crc = request[-2:]
    expected_crc = calculate_crc16(request[:-2])
    if received_crc != expected_crc:
        _log(f"  CRC mismatch: got {received_crc.hex()}, expected {expected_crc.hex()}", silent)
        return None

    if func_code != 0x03:
        _log(f"  Unsupported function code: 0x{func_code:02X}", silent)
        # Return exception response
        resp = struct.pack("BBB", slave_addr, func_code | 0x80, 0x01)
        return resp + calculate_crc16(resp)

    start_reg = struct.unpack(">H", request[2:4])[0]
    num_regs = struct.unpack(">H", request[4:6])[0]

    # Build response
    byte_count = num_regs * 2
    resp = struct.pack("BBB", slave_addr, func_code, byte_count)

    for i in range(num_regs):
        reg_idx = start_reg + i
        if reg_idx < len(sensor_values):
            resp += struct.pack(">H", sensor_values[reg_idx] & 0xFFFF)
        else:
            resp += struct.pack(">H", 0)

    return resp + calculate_crc16(resp)


def _log(msg: str, silent: bool = False):
    """Print only if not in silent mode."""
    if not silent:
        print(msg)


def main():
    parser = argparse.ArgumentParser(description="AgroSensor Sensor Bridge")
    parser.add_argument("--port", default="COM10", help="Virtual COM port (default: COM10)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument("--address", type=int, default=1, help="Modbus slave address (default: 1)")
    parser.add_argument("--silent", action="store_true", help="Suppress all console output (for background/pythonw)")
    args = parser.parse_args()

    _log(f"AgroSensor starting on {args.port} @ {args.baud} baud (addr={args.address})", args.silent)

    # Try initial weather fetch so simulator has real data from the start
    try:
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT))
        from tools.weather import get_weather
        w = get_weather()
        if w:
            _log(f"   Weather: {w.get('city', '?')} — {w.get('description', '?')}, "
                 f"{w.get('temperature', '?')}°C, humidity {w.get('humidity', '?')}%", args.silent)
        else:
            _log("   Weather: no data (will use simulation fallback)", args.silent)
    except Exception as e:
        _log(f"   Weather: failed to fetch ({e}), using simulation fallback", args.silent)

    _log("   Waiting for Modbus requests... (Ctrl+C to stop)\n", args.silent)

    ser = serial.Serial(
        port=args.port, baudrate=args.baud,
        bytesize=8, parity="N", stopbits=1, timeout=0.5,
    )

    read_count = 0

    try:
        while True:
            data = ser.read(8)  # Standard Modbus RTU request is 8 bytes
            if not data:
                continue

            field = _get_field()
            sensor_values = field.read()
            read_count += 1

            if not args.silent:
                weather = "Rain" if field.is_raining else "Clear"
                city = _get_active_city()
                print(f"[{read_count:04d}] {city} | {weather} | Request: {data.hex(' ')}")

            response = handle_request(data, sensor_values, args.silent)

            if response:
                ser.write(response)
                if not args.silent:
                    vals = sensor_values
                    print(f"       T={vals[0]/10:.1f}C  M={vals[1]/10:.1f}%  "
                          f"EC={vals[2]}uS  pH={vals[3]/100:.2f}  "
                          f"N={vals[4]}  P={vals[5]}  K={vals[6]}")
            else:
                _log("       No response (invalid request)", args.silent)

            _log("", args.silent)
    except KeyboardInterrupt:
        _log(f"\nSensor stopped. Total reads: {read_count}", args.silent)
    except Exception:
        pass  # Silent mode — don't crash on broken pipe etc.
    finally:
        ser.close()


if __name__ == "__main__":
    main()

