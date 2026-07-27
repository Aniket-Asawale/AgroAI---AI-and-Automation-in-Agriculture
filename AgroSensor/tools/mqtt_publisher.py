"""
AgroSensor — MQTT Cloud Publisher (Sensor Simulator)

Simulates sensor readings using FieldEnvironment and publishes them
to HiveMQ Cloud over MQTT/TLS. Each active city gets its own topic:
    agro/<city>/sensor-001

Usage:
    python tools/mqtt_publisher.py
    python tools/mqtt_publisher.py --interval 60

The FastAPI server subscribes to these topics and stores readings in PostgreSQL.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import paho.mqtt.client as mqtt

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from tools.sensor_conn import FieldEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mqtt_publisher")

# ─── API base for fetching active locations ───
_API_BASE = f"http://127.0.0.1:{settings.API_PORT}"


def _load_active_locations() -> list[dict]:
    """Fetch active locations from the running FastAPI server, fallback to Kolhapur."""
    try:
        req = Request(f"{_API_BASE}/api/weather/cities", headers={"Accept": "application/json"})
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # API returns {"cities": [...], "current": "..."}
            cities = data.get("cities", data) if isinstance(data, dict) else data
            active = [c for c in cities if c.get("is_active", False)]
            if active:
                return active
    except (URLError, OSError, json.JSONDecodeError) as e:
        logger.debug("Could not fetch cities from API: %s", e)
    # Fallback: just Kolhapur
    logger.info("Using fallback location: Kolhapur")
    return [{"name": "Kolhapur", "soil_type": "Alluvial", "is_active": True}]


def _build_payload(city: str, sensor_id: str, raw_regs: list[int]) -> dict:
    """Build a JSON payload matching the SensorData schema."""
    return {
        "sensor_id": sensor_id,
        "city": city,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": raw_regs[0] / 10.0,
        "moisture": raw_regs[1] / 10.0,
        "ec": float(raw_regs[2]),
        "ph": raw_regs[3] / 100.0,
        "nitrogen": float(raw_regs[4]),
        "phosphorus": float(raw_regs[5]),
        "potassium": float(raw_regs[6]),
        "raw_frame": str(raw_regs),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AgroSensor MQTT Cloud Publisher")
    parser.add_argument("--interval", type=int, default=int(settings.POLLING_INTERVAL_SECONDS),
                        help=f"Publish interval in seconds (default: {int(settings.POLLING_INTERVAL_SECONDS)})")
    args = parser.parse_args()

    # ─── MQTT Client Setup ───
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="agro-publisher",
        protocol=mqtt.MQTTv311,
    )
    client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)

    if settings.MQTT_USE_TLS:
        client.tls_set()  # uses system CA certs + best TLS version

    def on_connect(c, userdata, flags, rc, properties=None):
        success = (rc == 0 or str(rc) == "Success")
        if success:
            logger.info("✅ Connected to HiveMQ Cloud: %s:%d", settings.MQTT_BROKER, settings.MQTT_PORT)
        else:
            logger.error("❌ MQTT connection failed (rc=%s)", rc)

    def on_disconnect(c, userdata, flags_or_rc=None, rc=None, properties=None):
        logger.warning("⚠️  Disconnected from MQTT broker")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    logger.info("Connecting to %s:%d ...", settings.MQTT_BROKER, settings.MQTT_PORT)
    client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, keepalive=60)
    client.loop_start()
    time.sleep(2)  # Allow connection to establish

    # ─── Per-city FieldEnvironment instances ───
    environments: dict[str, FieldEnvironment] = {}
    publish_count = 0

    try:
        while True:
            locations = _load_active_locations()
            if not locations:
                logger.info("No active locations — sleeping %ds", args.interval)
                time.sleep(args.interval)
                continue

            for loc in locations:
                city = loc["name"]
                soil_type = loc.get("soil_type", "Alluvial")

                # Create environment on first use
                if city not in environments:
                    environments[city] = FieldEnvironment(soil_type=soil_type, sensor_index=0)
                    logger.info("Created simulator for %s (%s)", city, soil_type)

                env = environments[city]
                raw = env.read()
                sensor_id = f"AGRO-{city.upper()}-001"
                topic = f"{settings.MQTT_TOPIC_PREFIX}/{city.lower()}/sensor-001"
                payload = _build_payload(city, sensor_id, raw)

                result = client.publish(topic, json.dumps(payload), qos=1)
                publish_count += 1

                weather = "🌧️ Rain" if env.is_raining else "☀️ Clear"
                logger.info(
                    "[%04d] %s → %s | %s | T=%.1f°C M=%.1f%% EC=%.0f pH=%.2f N=%d P=%d K=%d",
                    publish_count, sensor_id, topic, weather,
                    payload["temperature"], payload["moisture"], payload["ec"],
                    payload["ph"], payload["nitrogen"], payload["phosphorus"], payload["potassium"],
                )

            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\nPublisher stopped. Total publishes: %d", publish_count)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()

