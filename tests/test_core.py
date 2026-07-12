"""
SafeGuard AI — Unit Tests
=========================
Tests core logic that requires no model/camera/GPU:
  - Dwell-time calculation (video-frame-based, never wall-clock)
  - Consecutive-frame false-positive guard
  - Centroid distance / zone membership
  - ZoneAnalyzer grid assignment
  - RiskScorer output range and factor contribution
  - AlertSystem cooldown and JSON logging
  - Config loading and validation

Run with:  pytest tests/test_core.py -v
"""

import pytest
import math
import json
import os
import time
import tempfile
import numpy as np
import cv2
from collections import deque
from datetime import datetime


# ──────────────────────────────────────────────
# Helpers (pure functions, no model needed)
# ──────────────────────────────────────────────

def compute_dwell_video_sec(entry_frame: int, current_frame: int, fps: float) -> float:
    """Dwell time in VIDEO seconds (frame-count based, never wall-clock)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (current_frame - entry_frame) / fps


def point_in_polygon(pt, poly_np: np.ndarray) -> bool:
    """True if (x,y) point is inside the polygon."""
    return cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0


def centroid_distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class FPGuard:
    """Minimal false-positive guard used in generate_demo.py / Detector.py"""
    def __init__(self, dwell_threshold_sec: float, min_consec_frames: int, fps: float):
        self.dwell_threshold_sec = dwell_threshold_sec
        self.min_consec_frames = min_consec_frames
        self.fps = fps
        self._entry_frame: dict = {}
        self._consec: dict = {}

    def observe(self, track_id: int, in_zone: bool, frame_idx: int) -> bool:
        """
        Call once per frame per track.
        Returns True only when alert should fire (both conditions met).
        """
        if not in_zone:
            self._entry_frame.pop(track_id, None)
            self._consec.pop(track_id, None)
            return False

        if track_id not in self._entry_frame:
            self._entry_frame[track_id] = frame_idx
            self._consec[track_id] = 1
        else:
            self._consec[track_id] += 1

        dwell = compute_dwell_video_sec(self._entry_frame[track_id], frame_idx, self.fps)
        consec = self._consec[track_id]
        return dwell >= self.dwell_threshold_sec and consec >= self.min_consec_frames


# ──────────────────────────────────────────────
# 1. Dwell-time calculation
# ──────────────────────────────────────────────

class TestDwellTime:
    def test_zero_frames_elapsed(self):
        assert compute_dwell_video_sec(100, 100, 30.0) == pytest.approx(0.0)

    def test_one_second_at_30fps(self):
        assert compute_dwell_video_sec(0, 30, 30.0) == pytest.approx(1.0)

    def test_video_duration_boundary(self):
        """Dwell of last frame equals total video duration."""
        total_frames = 473
        fps = 30.0
        dwell = compute_dwell_video_sec(0, total_frames - 1, fps)
        video_duration = total_frames / fps
        assert dwell < video_duration  # must not exceed duration

    def test_dwell_never_exceeds_video_duration(self):
        """Even if entry_frame=0 and current=last, dwell < total duration."""
        fps = 30.0
        total = 473
        dwell = compute_dwell_video_sec(0, total - 1, fps)
        assert dwell < total / fps

    def test_fractional_fps(self):
        result = compute_dwell_video_sec(0, 25, 25.0)
        assert result == pytest.approx(1.0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError):
            compute_dwell_video_sec(0, 10, -1.0)

    def test_mid_video_dwell(self):
        """Person enters at frame 100, current frame 160, fps=30 → 2.0s."""
        assert compute_dwell_video_sec(100, 160, 30.0) == pytest.approx(2.0)

    def test_wall_clock_independence(self):
        """Dwell should return same value regardless of when it's called."""
        t0 = time.time()
        d1 = compute_dwell_video_sec(50, 80, 30.0)
        time.sleep(0.1)
        d2 = compute_dwell_video_sec(50, 80, 30.0)
        assert d1 == d2  # identical — wall-clock plays no role


# ──────────────────────────────────────────────
# 2. False-positive consecutive-frame guard
# ──────────────────────────────────────────────

class TestFPGuard:
    def setup_method(self):
        # threshold=1.0s, guard=8 frames, fps=30
        self.guard = FPGuard(dwell_threshold_sec=1.0, min_consec_frames=8, fps=30.0)

    def test_no_alert_before_consec_threshold(self):
        """Alert must NOT fire before 8 consecutive in-zone frames."""
        for f in range(1, 8):  # frames 1–7
            fired = self.guard.observe(track_id=1, in_zone=True, frame_idx=f)
            assert not fired, f"Alert fired early at frame {f}"

    def test_alert_fires_at_exact_threshold(self):
        """Alert fires at exactly frame 8 (8th consecutive, dwell=0.23s — below 1s).
        But at frame 30+8=38 (dwell=1.0s + 8 consec) both conditions met."""
        guard = FPGuard(dwell_threshold_sec=1.0, min_consec_frames=8, fps=30.0)
        # enter zone at frame 0
        for f in range(0, 30):   # frames 0-29: in zone, dwell < 1s
            fired = guard.observe(1, True, f)
            assert not fired, f"Fired too early at frame {f}"
        # frame 30 = 1.0s dwell, 30 consec (>=8) → should fire
        fired = guard.observe(1, True, 30)
        assert fired, "Alert should fire at frame 30 (1.0s dwell, 30 consec >= 8)"

    def test_reset_on_zone_exit(self):
        """After leaving the zone, consecutive count and entry frame must reset."""
        guard = FPGuard(dwell_threshold_sec=1.0, min_consec_frames=8, fps=30.0)
        for f in range(0, 35):
            guard.observe(1, True, f)
        # Leave zone
        guard.observe(1, False, 35)
        # Re-enter — must start fresh (not carry over old consec/dwell)
        fired = guard.observe(1, True, 36)
        assert not fired, "After zone exit, first re-entry frame must not fire"

    def test_different_track_ids_independent(self):
        """Two different track IDs must not share state."""
        guard = FPGuard(dwell_threshold_sec=0.5, min_consec_frames=4, fps=30.0)
        for f in range(0, 20):
            guard.observe(1, True, f)
        # track_id=2 enters late — should NOT fire yet
        fired = guard.observe(2, True, 20)
        assert not fired

    def test_single_frame_never_fires(self):
        """A single in-zone frame must never trigger an alert."""
        guard = FPGuard(dwell_threshold_sec=0.0, min_consec_frames=8, fps=30.0)
        fired = guard.observe(1, True, 100)
        assert not fired


# ──────────────────────────────────────────────
# 3. Centroid distance calculations
# ──────────────────────────────────────────────

class TestCentroidDistance:
    def test_same_point(self):
        assert centroid_distance((100, 200), (100, 200)) == pytest.approx(0.0)

    def test_horizontal(self):
        assert centroid_distance((0, 0), (50, 0)) == pytest.approx(50.0)

    def test_vertical(self):
        assert centroid_distance((0, 0), (0, 80)) == pytest.approx(80.0)

    def test_diagonal(self):
        # 3-4-5 right triangle
        assert centroid_distance((0, 0), (30, 40)) == pytest.approx(50.0)

    def test_within_tolerance(self):
        """Two centroids within 80px tolerance are considered 'same person'."""
        d = centroid_distance((100, 100), (160, 140))
        assert d < 80  # sqrt(60^2+40^2) = 72 < 80

    def test_outside_tolerance(self):
        d = centroid_distance((0, 0), (200, 200))
        assert d > 80


# ──────────────────────────────────────────────
# 4. Zone polygon membership
# ──────────────────────────────────────────────

class TestZonePolygon:
    def setup_method(self):
        # Rectangle from (100,100) to (500,400)
        self.zone = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)

    def test_center_inside(self):
        assert point_in_polygon((300, 250), self.zone)

    def test_outside_left(self):
        assert not point_in_polygon((50, 250), self.zone)

    def test_outside_right(self):
        assert not point_in_polygon((600, 250), self.zone)

    def test_outside_top(self):
        assert not point_in_polygon((300, 50), self.zone)

    def test_outside_bottom(self):
        assert not point_in_polygon((300, 450), self.zone)

    def test_corner_is_inside(self):
        assert point_in_polygon((100, 100), self.zone)

    def test_normalised_zone_scales_correctly(self):
        """Zone defined in [0,1] normalised coords, scaled to pixel space."""
        norm = np.array([[0.1, 0.3], [0.6, 0.3], [0.6, 0.9], [0.1, 0.9]], dtype=np.float32)
        w, h = 1280, 720
        px = (norm * np.array([w, h])).astype(np.int32)
        # Point at (0.35, 0.6) should be inside
        pt = (int(0.35 * w), int(0.6 * h))
        assert point_in_polygon(pt, px)
        # Point at (0.05, 0.5) should be outside (x < 0.1 * 1280 = 128)
        pt2 = (50, int(0.5 * h))
        assert not point_in_polygon(pt2, px)


# ──────────────────────────────────────────────
# 5. ZoneAnalyzer grid assignment
# ──────────────────────────────────────────────

class TestZoneAnalyzer:
    def setup_method(self):
        import sys
        sys.path.insert(0, '.')
        from modules.zone_analyzer import ZoneAnalyzer
        self.za = ZoneAnalyzer(width=640, height=480, grid_size=(6, 6))

    def test_top_left_object_maps_to_zone_0_0(self):
        objects = {"obj1": {"centroid": (5, 5)}}
        zones = self.za.update_zones(objects)
        assert "0_0" in zones

    def test_bottom_right_maps_to_last_zone(self):
        objects = {"obj1": {"centroid": (639, 479)}}
        zones = self.za.update_zones(objects)
        assert "5_5" in zones

    def test_zone_counts_accumulate(self):
        for _ in range(3):
            self.za.update_zones({"obj1": {"centroid": (5, 5)}})
        assert self.za.zone_counts["0_0"] == 3

    def test_heatmap_grows_with_updates(self):
        initial = float(self.za.heatmap[0, 0])
        self.za.update_zones({"obj1": {"centroid": (5, 5)}})
        assert self.za.heatmap[0, 0] > initial


# ──────────────────────────────────────────────
# 6. AlertSystem cooldown + JSON logging
# ──────────────────────────────────────────────

class TestAlertSystem:
    def setup_method(self):
        import sys
        sys.path.insert(0, '.')
        from modules.alert_system import AlertSystem
        self.as_ = AlertSystem()

    def test_first_alert_passes_cooldown(self):
        assert self.as_.should_send_alert("zone_intrusion") is True

    def test_repeat_alert_within_cooldown_blocked(self):
        self.as_.should_send_alert("fall")
        assert self.as_.should_send_alert("fall") is False

    def test_different_types_independent(self):
        self.as_.should_send_alert("zone_intrusion")
        # Different type should still pass immediately
        assert self.as_.should_send_alert("ppe_violation") is True

    def test_log_alert_stores_entry(self):
        self.as_.log_alert("zone_intrusion", "test message")
        logs = self.as_.get_recent_alerts(limit=5)
        assert len(logs) >= 1
        last = logs[-1]
        assert last["alert_type"] == "zone_intrusion"
        assert "timestamp" in last
        assert "message" in last


# ──────────────────────────────────────────────
# 7. RiskScorer (imported once module is created)
# ──────────────────────────────────────────────

class TestRiskScorer:
    def setup_method(self):
        import sys
        sys.path.insert(0, '.')
        from modules.risk_scorer import RiskScorer
        self.rs = RiskScorer()

    def test_score_in_valid_range(self):
        score, _ = self.rs.compute(dwell_sec=0, proximity_m=None, hour=12)
        assert 0 <= score <= 100

    def test_high_dwell_increases_score(self):
        low, _ = self.rs.compute(dwell_sec=0, proximity_m=None, hour=12)
        high, _ = self.rs.compute(dwell_sec=30, proximity_m=None, hour=12)
        assert high > low

    def test_close_proximity_increases_score(self):
        far, _ = self.rs.compute(dwell_sec=5, proximity_m=10.0, hour=12)
        close, _ = self.rs.compute(dwell_sec=5, proximity_m=0.5, hour=12)
        assert close > far

    def test_night_shift_higher_than_day(self):
        day, _ = self.rs.compute(dwell_sec=5, proximity_m=2.0, hour=14)
        night, _ = self.rs.compute(dwell_sec=5, proximity_m=2.0, hour=2)
        assert night >= day

    def test_factors_list_not_empty(self):
        _, factors = self.rs.compute(dwell_sec=10, proximity_m=1.0, hour=3)
        assert isinstance(factors, list)
        assert len(factors) > 0

    def test_max_inputs_give_score_100(self):
        score, _ = self.rs.compute(dwell_sec=120, proximity_m=0.0, hour=3)
        assert score == 100

    def test_zero_inputs_give_low_score(self):
        score, _ = self.rs.compute(dwell_sec=0, proximity_m=None, hour=12)
        assert score < 30
