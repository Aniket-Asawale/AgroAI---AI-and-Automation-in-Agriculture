"""
AgroSensor — MQTT Cloud Subscriber

Connects to HiveMQ Cloud, subscribes to sensor topics, and stores
incoming readings in PostgreSQL via SQLAlchemy.

Used by main.py as a background task (replaces COM-port polling).
"""

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from config import settings

logger = logging.getLogger("mqtt_subscriber")


class MQTTSubscriber:
    """
    Manages the MQTT connection to HiveMQ Cloud.
    Calls `on_reading_callback` with parsed sensor data dict for each message.
    """

    def __init__(self, on_reading_callback: Optional[Callable[[dict], None]] = None):
        self._callback = on_reading_callback
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._message_count = 0
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def message_count(self) -> int:
        return self._message_count

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        # paho-mqtt v2.x: rc is a ReasonCode object, not int
        success = (rc == 0 or str(rc) == "Success")
        with self._lock:
            self._connected = success
        if success:
            topic = f"{settings.MQTT_TOPIC_PREFIX}/+/sensor-001"
            client.subscribe(topic, qos=1)
            logger.info("☁️  Connected to HiveMQ Cloud — subscribed to '%s'", topic)
        else:
            logger.error("❌ MQTT connection failed (rc=%s)", rc)

    def _on_disconnect(self, client, userdata, flags_or_rc=None, rc=None, properties=None):
        with self._lock:
            self._connected = False
        logger.warning("⚠️  Disconnected from MQTT broker — will auto-reconnect")

    def _on_message(self, client, userdata, msg):
        """Called when a sensor reading arrives from the cloud."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self._message_count += 1

            logger.debug(
                "📩 [%d] %s — %s | T=%.1f M=%.1f EC=%.0f pH=%.2f N=%d P=%d K=%d",
                self._message_count, msg.topic, payload.get("city", "?"),
                payload.get("temperature", 0), payload.get("moisture", 0),
                payload.get("ec", 0), payload.get("ph", 0),
                payload.get("nitrogen", 0), payload.get("phosphorus", 0),
                payload.get("potassium", 0),
            )

            if self._callback:
                self._callback(payload)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Invalid MQTT message on %s: %s", msg.topic, e)

    def start(self):
        """Connect to HiveMQ Cloud and start the network loop in a background thread."""
        if self._client is not None:
            logger.warning("MQTT subscriber already started")
            return

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="agro-server",
            protocol=mqtt.MQTTv311,
        )
        self._client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)

        if settings.MQTT_USE_TLS:
            self._client.tls_set()  # uses system CA certs + best TLS version

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Enable auto-reconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        logger.info("Connecting to MQTT broker %s:%d ...", settings.MQTT_BROKER, settings.MQTT_PORT)
        self._client.connect_async(settings.MQTT_BROKER, settings.MQTT_PORT, keepalive=60)
        self._client.loop_start()

    def stop(self):
        """Disconnect and stop the background network loop."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            with self._lock:
                self._connected = False
            logger.info("MQTT subscriber stopped (total messages: %d)", self._message_count)

