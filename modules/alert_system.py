"""
SafeGuard AI — Structured Alert Logger (P5)
============================================
Every alert is written as a JSON Lines record to logs/alerts.jsonl.
Fields: timestamp, camera_id, alert_type, confidence, frame_reference,
        person_id, duration_video_sec, consec_frames, message.

All past alerts are independently auditable by reading the JSONL file.
"""

import json
import os
import logging
import threading
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)


class StructuredAlertLogger:
    """
    Drop-in replacement for the old AlertSystem.
    - Writes every alert as a JSON line to a JSONL file.
    - Maintains an in-memory deque for the live dashboard.
    - Enforces per-type cooldown to suppress alert spam.
    """

    def __init__(self,
                 log_file: str = "logs/alerts.jsonl",
                 camera_id: str = "cam_unknown",
                 cooldown_sec: int = 10,
                 memory_limit: int = 200):
        self.log_file = log_file
        self.camera_id = camera_id
        self.cooldown_sec = cooldown_sec
        self._cooldown_map: dict = {}          # alert_type -> last fire epoch
        self._memory: deque = deque(maxlen=memory_limit)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)

    # ── Core API ──────────────────────────────────────────────────

    def log_alert(self,
                  alert_type: str,
                  message: str,
                  *,
                  confidence: float = None,
                  frame_reference: int = None,
                  person_id: str = None,
                  duration_video_sec: float = None,
                  consec_frames: int = None,
                  extra: dict = None) -> dict:
        """
        Write one structured alert. Returns the alert dict.
        Does NOT enforce cooldown — callers use should_fire() for that.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": self.camera_id,
            "alert_type": alert_type,
            "message": message,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "frame_reference": frame_reference,
            "person_id": person_id,
            "duration_video_sec": round(duration_video_sec, 3) if duration_video_sec is not None else None,
            "consec_frames": consec_frames,
        }
        if extra:
            record.update(extra)

        with self._lock:
            # Write to JSONL file
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as e:
                logger.error(f"Failed to write alert to {self.log_file}: {e}")

            # Keep in-memory copy for dashboard
            self._memory.append(record)

        logger.info(f"ALERT [{self.camera_id}] {alert_type}: {message} "
                    f"(conf={confidence}, frame={frame_reference})")
        return record

    def should_fire(self, alert_type: str, track_scope: str = "") -> bool:
        """
        Returns True and resets cooldown clock if the alert is past cooldown.
        Use track_scope to differentiate per-person cooldowns (e.g., person_id).
        """
        import time
        key = f"{alert_type}_{track_scope}"
        now = time.time()
        with self._lock:
            last = self._cooldown_map.get(key, 0)
            if now - last < self.cooldown_sec:
                return False
            self._cooldown_map[key] = now
        return True

    def fire(self, alert_type: str, message: str, **kwargs) -> dict | None:
        """
        Convenience: check cooldown + log in one call.
        Returns the logged record, or None if suppressed by cooldown.
        """
        person_id = kwargs.get("person_id", "")
        if not self.should_fire(alert_type, track_scope=str(person_id)):
            return None
        return self.log_alert(alert_type, message, **kwargs)

    # ── Dashboard / API helpers ────────────────────────────────────

    def get_recent(self, limit: int = 20) -> list:
        """Return the most recent alerts (for dashboard polling)."""
        with self._lock:
            return list(self._memory)[-limit:]

    def get_recent_by_type(self, alert_type: str, limit: int = 10) -> list:
        with self._lock:
            return [a for a in self._memory if a["alert_type"] == alert_type][-limit:]

    # ── Legacy compatibility shims (drop-in for old AlertSystem) ───

    def should_send_alert(self, alert_type: str) -> bool:
        return self.should_fire(alert_type)

    def log_alert_legacy(self, alert_type: str, message: str):
        """Compatibility with old log_alert(type, msg) signature."""
        self.log_alert(alert_type, message)

    def get_recent_alerts(self, limit: int = 10) -> list:
        return self.get_recent(limit)


# Backward-compatibility alias — old code importing AlertSystem keeps working
AlertSystem = StructuredAlertLogger