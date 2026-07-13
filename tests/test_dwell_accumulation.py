"""
tests/test_dwell_accumulation.py

Integration test: does IDRelinker actually allow dwell-time to accumulate
continuously toward the alert threshold in test_proximity.mp4 despite
the raw tracker generating 14-15 unique IDs per 100 frames?

Approach
--------
- Define a danger zone polygon covering the central region of the frame
  where the proximity clip workers spend most of their time.
- Run test_proximity.mp4 through YOLOv8 + BoT-SORT tracker (default config).
- Feed raw detections into IDRelinker -> ZoneAnalyzer.
- Track dwell-time per CANONICAL id (after re-linking).
- Simultaneously track dwell-time per RAW tracker id (no re-linking).
- Compare: does at least one canonical ID cross the 3-second alert threshold
  that would have been missed if we relied purely on raw IDs?

Reports
-------
- Per-frame raw_id → canonical_id mapping events (re-links)
- Maximum raw-id dwell accumulated
- Maximum canonical-id dwell accumulated
- Whether 3s threshold was crossed with / without IDRelinker
- Number of raw-ID resets vs relinks observed
"""

import cv2
import sys
import time
import numpy as np
import pytest
import importlib.util
from pathlib import Path

# Resolve paths relative to this file so the test runs from any machine / working directory.
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))
from modules.zone_analyzer import IDRelinker

VIDEO   = str(BASE_DIR / "test_proximity.mp4")
TRACKER = str(BASE_DIR / "safeguard_botsort.yaml")

PERSON_CLASS = 0
MAX_FRAMES   = 200          # ~6-7s at 30fps
ALERT_THRESHOLD_S = 3.0     # seconds of continuous zone presence to fire alert
FPS_ESTIMATE      = 30.0

# Danger zone polygon: central 40% of frame
# Defined as fraction of (width, height); resolved to pixels after cap open.
ZONE_POLYGON_FRAC = np.array([
    [0.3, 0.2],
    [0.7, 0.2],
    [0.7, 0.8],
    [0.3, 0.8],
], dtype=np.float32)


def point_in_polygon(cx: float, cy: float, polygon: np.ndarray) -> bool:
    """Returns True if (cx, cy) is inside the polygon."""
    point = (int(cx), int(cy))
    poly  = polygon.astype(np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def run(video_path: str, use_relinker: bool):
    """
    Run detection + tracking on video_path.
    If use_relinker=True, route detections through IDRelinker before dwell tracking.
    Returns:
        dwell_per_id: dict {id -> total_seconds_in_zone}
        relink_events: list of (frame_num, raw_id, canonical_id) for relinks
        total_unique_ids: int
    """
    from ultralytics import YOLO  # imported here so CI collection never triggers this

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or FPS_ESTIMATE
    dt     = 1.0 / fps

    zone_polygon = (ZONE_POLYGON_FRAC * np.array([width, height])).astype(np.int32)

    model    = YOLO("yolov8s.pt")
    relinker = IDRelinker(spatial_threshold_px=100, time_window_s=2.0, fps=int(fps)) if use_relinker else None

    dwell_per_id:   dict = {}   # {canonical_id -> cumulative seconds in zone}
    all_raw_ids:    set  = set()
    relink_events:  list = []
    frame_num = 0

    while cap.isOpened() and frame_num < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, conf=0.25,
                               classes=[PERSON_CLASS], tracker=TRACKER)[0]

        raw_detections = {}
        if results.boxes.id is not None:
            boxes = results.boxes.cpu().numpy()
            for i, cls_id in enumerate(boxes.cls):
                if int(cls_id) != PERSON_CLASS:
                    continue
                raw_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                raw_detections[raw_id] = {'centroid': (cx, cy)}
                all_raw_ids.add(raw_id)

        if use_relinker:
            remapped = relinker.remap(raw_detections)
            # Capture relink events: when raw_id != canonical_id
            for canon_id, info in remapped.items():
                raw_id = info.get('raw_id', canon_id)
                if raw_id != canon_id:
                    relink_events.append((frame_num, raw_id, canon_id))
            effective = {cid: info for cid, info in remapped.items()}
        else:
            effective = raw_detections

        # Accumulate dwell for each effective ID that is inside the zone
        for eid, info in effective.items():
            cx, cy = info['centroid']
            if point_in_polygon(cx, cy, zone_polygon):
                dwell_per_id[eid] = dwell_per_id.get(eid, 0.0) + dt

        frame_num += 1

    cap.release()
    return dwell_per_id, relink_events, len(all_raw_ids)


# ─────────────────────────────────────────────────────────────
# pytest-compatible test function
# Skips gracefully if the video file or YOLO model is missing
# (e.g. CI environment without large binary assets).
# Run locally with: python -m pytest tests/test_dwell_accumulation.py -v -s
# Run standalone:   python tests/test_dwell_accumulation.py
# ─────────────────────────────────────────────────────────────

def test_dwell_accumulation_with_idrelinker():
    """
    Integration test: IDRelinker does not crash the pipeline and the
    canonical-ID dwell is at least as high as the raw-ID dwell.

    Skipped automatically in CI environments where ultralytics is not installed
    (the full torch/ultralytics stack is too large for a lightweight CI runner).
    Run locally with the full venv active:
        python -m pytest tests/test_dwell_accumulation.py -v -s
    """
    # Skip if ultralytics is not importable (lightweight CI environment)
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip(
            "ultralytics is not installed in this environment. "
            "This integration test requires the full local venv (ultralytics + torch). "
            "Run locally with: python -m pytest tests/test_dwell_accumulation.py -v -s"
        )

    # Skip if the video file is also missing (different machine / fresh clone)
    if not Path(VIDEO).exists():
        pytest.skip(
            f"test_proximity.mp4 not found at {VIDEO}. "
            "This test requires the real video file in the project root."
        )

    dwell_raw, _, n_raw_ids = run(VIDEO, use_relinker=False)
    dwell_canon, relink_events, n_raw_ids_r = run(VIDEO, use_relinker=True)

    max_raw_dwell   = max(dwell_raw.values(),   default=0.0)
    max_canon_dwell = max(dwell_canon.values(), default=0.0)

    print(f"\n  Unique raw IDs (no relinker):   {n_raw_ids}")
    print(f"  Unique raw IDs (with relinker): {n_raw_ids_r}")
    print(f"  Relink events fired:            {len(relink_events)}")
    print(f"  Max dwell WITHOUT IDRelinker:   {max_raw_dwell:.2f}s")
    print(f"  Max dwell WITH IDRelinker:      {max_canon_dwell:.2f}s")

    # Core assertion: IDRelinker must not make dwell *worse*
    assert max_canon_dwell >= max_raw_dwell, (
        f"IDRelinker degraded dwell continuity: "
        f"raw={max_raw_dwell:.2f}s > canon={max_canon_dwell:.2f}s"
    )

    # Structural assertion: pipeline completed without crashing
    assert n_raw_ids > 0, "No detections at all — check model or video path"


# ─────────────────────────────────────────────────────────────
# Standalone script entry-point (python tests/test_dwell_accumulation.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not Path(VIDEO).exists():
        print(f"ERROR: Video not found: {VIDEO}")
        sys.exit(1)

    print("=" * 65)
    print("Running WITHOUT IDRelinker (raw tracker IDs only)...")
    t0 = time.time()
    dwell_raw, _, n_raw_ids = run(VIDEO, use_relinker=False)
    print(f"  Done in {time.time()-t0:.1f}s")

    max_raw_dwell = max(dwell_raw.values(), default=0.0)
    raw_above_threshold = [iid for iid, d in dwell_raw.items() if d >= ALERT_THRESHOLD_S]
    print(f"  Unique raw IDs seen:       {n_raw_ids}")
    print(f"  Max dwell (any raw ID):    {max_raw_dwell:.2f}s")
    print(f"  IDs crossing {ALERT_THRESHOLD_S}s threshold: {raw_above_threshold}")

    print()
    print("Running WITH IDRelinker (canonical ID continuity)...")
    t0 = time.time()
    dwell_canon, relink_events, n_raw_ids_r = run(VIDEO, use_relinker=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    max_canon_dwell = max(dwell_canon.values(), default=0.0)
    canon_above_threshold = [iid for iid, d in dwell_canon.items() if d >= ALERT_THRESHOLD_S]
    print(f"  Unique raw IDs seen:       {n_raw_ids_r}")
    print(f"  Relink events (ID saved):  {len(relink_events)}")
    if relink_events[:5]:
        print(f"  First relinks: {relink_events[:5]}")
    print(f"  Max dwell (canonical ID):  {max_canon_dwell:.2f}s")
    print(f"  IDs crossing {ALERT_THRESHOLD_S}s threshold: {canon_above_threshold}")

    print()
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Alert threshold:                     {ALERT_THRESHOLD_S}s")
    print(f"  Max dwell WITHOUT IDRelinker:        {max_raw_dwell:.2f}s")
    print(f"  Max dwell WITH IDRelinker:           {max_canon_dwell:.2f}s")
    print(f"  Alert fires WITHOUT IDRelinker:      {'YES' if raw_above_threshold else 'NO'}")
    print(f"  Alert fires WITH IDRelinker:         {'YES' if canon_above_threshold else 'NO'}")
    print(f"  Relink events that saved continuity: {len(relink_events)}")
    print("=" * 65)

    alert_fired_with    = bool(canon_above_threshold)
    alert_fired_without = bool(raw_above_threshold)

    if alert_fired_with and not alert_fired_without:
        verdict = "[PASS] IDRelinker enabled alert that raw tracker missed"
    elif max_canon_dwell > max_raw_dwell + 0.5:
        verdict = f"[PASS] IDRelinker added >{max_canon_dwell - max_raw_dwell:.1f}s of dwell continuity"
    elif alert_fired_with and alert_fired_without:
        verdict = "[WARN] INCONCLUSIVE: Both fire threshold — clip dwell is long enough that relinker is not decisive in this window"
    else:
        verdict = f"[FAIL] IDRelinker did not improve dwell continuity (raw={max_raw_dwell:.2f}s, canon={max_canon_dwell:.2f}s)"

    print(f"\n  Integration test verdict: {verdict}")
