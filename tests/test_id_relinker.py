# -*- coding: utf-8 -*-
"""
tests/test_id_relinker.py

Unit tests for IDRelinker in modules/zone_analyzer.py.

Tests:
  PASS-1  (positive case): track disappears, new track appears within 100px
          and within 2s -> should be re-linked to same canonical ID ->
          dwell timer carries over.

  PASS-2  (positive case edge): new track at exactly 99px, just under 1.9s ->
          still within thresholds -> should relink.

  FAIL-1  (negative case - too far): new track appears 150px away within
          2s -> should NOT relink -> new canonical ID assigned -> dwell reset.

  FAIL-2  (negative case - too late): new track appears at 50px but after
          2.5s -> should NOT relink -> ghost evicted -> dwell reset.

  FAIL-3  (negative case - two separate people): two ghosts present, new
          track closer to ghost-B than ghost-A, should relink to ghost-B
          only, ghost-A remains separate.
"""

import sys
import os
import time

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.zone_analyzer import IDRelinker

PASS = "PASS"
FAIL = "FAIL"

results = []


def check(test_name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((test_name, status, detail))
    symbol = "[OK]" if condition else "[FAIL]"
    print(f"  {symbol} {test_name}" + (f": {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────
# Helper: simulate a dwell-time tracker on top of IDRelinker.
# Accumulates time_in_zone per canonical_id across frames.
# ─────────────────────────────────────────────────────────────
class MockDwellTracker:
    def __init__(self, relinker: IDRelinker):
        self.relinker = relinker
        self.dwell: dict = {}  # canonical_id -> cumulative seconds

    def tick(self, raw_detections: dict, frame_duration_s: float = 1/30):
        """Feed a frame's raw detections and accumulate dwell time."""
        remapped = self.relinker.remap(raw_detections)
        for canon_id in remapped:
            self.dwell[canon_id] = self.dwell.get(canon_id, 0.0) + frame_duration_s
        return remapped


# ─────────────────────────────────────────────────────────────
# TEST PASS-1: within 100px, within 2s -> relink, dwell carries over
# ─────────────────────────────────────────────────────────────
print("\n=== PASS-1: Within thresholds — should relink ===")
relinker = IDRelinker(spatial_threshold_px=100, time_window_s=2.0)
tracker = MockDwellTracker(relinker)

# Frame 1-5: raw_id=1 at (500, 500), accumulate 5 frames of dwell
for _ in range(5):
    tracker.tick({1: {'centroid': (500, 500)}})

dwell_before_drop = tracker.dwell.get(1, 0.0)

# Frame 6: raw_id=1 disappears (goes to ghost buffer)
tracker.tick({})

# Immediately (< 2s): raw_id=2 appears at (550, 520) — 72px away
time.sleep(0.05)  # 50ms elapsed, well within 2s window
remapped = tracker.tick({2: {'centroid': (550, 520)}})

# raw_id=2 should have been re-linked to canonical_id=1
canon_ids_in_frame = list(remapped.keys())
check(
    "PASS-1a: new raw_id=2 re-linked to canonical_id=1",
    canon_ids_in_frame == [1],
    f"got canon_ids={canon_ids_in_frame}"
)

# Dwell should have carried over (canonical_id=1 still accumulating)
# Canonical 2 should NOT exist since it was re-linked
check(
    "PASS-1b: canonical_id=2 NOT created (no reset)",
    2 not in tracker.dwell,
    f"tracker.dwell keys={list(tracker.dwell.keys())}"
)
check(
    "PASS-1c: canonical_id=1 dwell continued (not reset to 0)",
    tracker.dwell.get(1, 0) > dwell_before_drop,
    f"dwell_before={dwell_before_drop:.3f}s, dwell_after={tracker.dwell.get(1,0):.3f}s"
)


# ─────────────────────────────────────────────────────────────
# TEST PASS-2: edge of threshold (99px, 1.9s) — should still relink
# ─────────────────────────────────────────────────────────────
print("\n=== PASS-2: Edge of thresholds (99px / 1.9s) — should relink ===")
relinker2 = IDRelinker(spatial_threshold_px=100, time_window_s=2.0)
tracker2 = MockDwellTracker(relinker2)

for _ in range(3):
    tracker2.tick({10: {'centroid': (200, 200)}})

tracker2.tick({})  # drop

# Manually set the ghost's lost_at to 1.9s ago so we're under window
relinker2.ghost_tracks[10]['lost_at'] = time.monotonic() - 1.9

# 99px away from (200,200): (200+70, 200+70) ≈ 98.99px
remapped2 = tracker2.tick({11: {'centroid': (270, 270)}})
canon_ids_2 = list(remapped2.keys())
check(
    "PASS-2: 99px/1.9s -> still relinked to canon_id=10",
    10 in canon_ids_2 and 11 not in tracker2.dwell,
    f"got canon_ids={canon_ids_2}"
)


# ─────────────────────────────────────────────────────────────
# TEST FAIL-1: too far (150px) — should NOT relink
# ─────────────────────────────────────────────────────────────
print("\n=== FAIL-1: Too far (150px) — should NOT relink ===")
relinker3 = IDRelinker(spatial_threshold_px=100, time_window_s=2.0)
tracker3 = MockDwellTracker(relinker3)

for _ in range(3):
    tracker3.tick({20: {'centroid': (300, 300)}})

tracker3.tick({})  # drop

time.sleep(0.05)
# 150px away from (300,300): (300+106, 300+106) ≈ 150px
remapped3 = tracker3.tick({21: {'centroid': (406, 406)}})
canon_ids_3 = list(remapped3.keys())
check(
    "FAIL-1a: too far -> raw_id=21 gets NEW canonical_id (not 20)",
    20 not in canon_ids_3 and 21 in canon_ids_3,
    f"got canon_ids={canon_ids_3}"
)
check(
    "FAIL-1b: canonical_id=20 dwell NOT transferred to canonical_id=21",
    tracker3.dwell.get(21, 0) < tracker3.dwell.get(20, 0),
    f"dwell[20]={tracker3.dwell.get(20,0):.3f}, dwell[21]={tracker3.dwell.get(21,0):.3f}"
)


# ─────────────────────────────────────────────────────────────
# TEST FAIL-2: too late (>2s) — ghost evicted, should NOT relink
# ─────────────────────────────────────────────────────────────
print("\n=== FAIL-2: Too late (2.5s elapsed) — should NOT relink ===")
relinker4 = IDRelinker(spatial_threshold_px=100, time_window_s=2.0)
tracker4 = MockDwellTracker(relinker4)

for _ in range(3):
    tracker4.tick({30: {'centroid': (400, 400)}})

tracker4.tick({})  # drop

# Manually backdate the ghost's lost_at to 2.5s ago (past the 2.0s window)
relinker4.ghost_tracks[30]['lost_at'] = time.monotonic() - 2.5

# 50px away — within spatial threshold — but time window exceeded
remapped4 = tracker4.tick({31: {'centroid': (435, 435)}})
canon_ids_4 = list(remapped4.keys())
check(
    "FAIL-2: expired ghost -> raw_id=31 gets NEW canonical (not 30)",
    30 not in canon_ids_4 and 31 in canon_ids_4,
    f"got canon_ids={canon_ids_4}"
)


# ─────────────────────────────────────────────────────────────
# TEST FAIL-3: two ghosts present, new track links to closest only
# ─────────────────────────────────────────────────────────────
print("\n=== FAIL-3: Two ghosts — new track links to closest, other stays separate ===")
relinker5 = IDRelinker(spatial_threshold_px=100, time_window_s=2.0)
tracker5 = MockDwellTracker(relinker5)

# Two separate workers
for _ in range(3):
    tracker5.tick({
        40: {'centroid': (100, 100)},
        41: {'centroid': (700, 700)},
    })

tracker5.tick({})  # both drop

time.sleep(0.05)
# New track appears at (130,130) — 42px from ghost-40, 805px from ghost-41
remapped5 = tracker5.tick({42: {'centroid': (130, 130)}})
canon_ids_5 = list(remapped5.keys())

check(
    "FAIL-3a: new track links to ghost-40 (closest), NOT ghost-41",
    40 in canon_ids_5 and 41 not in canon_ids_5,
    f"got canon_ids={canon_ids_5}"
)
check(
    "FAIL-3b: ghost-41 still in ghost buffer (not accidentally merged)",
    41 in relinker5.ghost_tracks,
    f"ghost_tracks keys={list(relinker5.ghost_tracks.keys())}"
)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Test':<45} {'Result':>6}")
print("-" * 60)
passed = 0
failed = 0
for name, status, detail in results:
    print(f"  {name:<43} {status:>6}")
    if status == PASS:
        passed += 1
    else:
        failed += 1
print("=" * 60)
print(f"  {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
