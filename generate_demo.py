"""
SafeGuard AI - Demo Video Generator
====================================
Uses YOLOv8s with ByteTrack for persistent IDs across frames.
Dwell time is computed from frame_index / fps — never wall-clock time.
All overlay numbers come directly from YOLO+ByteTrack output on that frame.
False-positive guard: zone breach only fires after MIN_BREACH_FRAMES consecutive frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# ─── Configuration ────────────────────────────────────────────────
VIDEO_IN  = "3078511-hd_1920_1080_30fps.mp4"
VIDEO_OUT = "annotated_construction_demo.mp4"
MODEL_PATH = "yolov8s.pt"
CONF_THRESHOLD = 0.30
PERSON_CLASS_ID = 0

# Danger-zone polygon in normalised [0,1] coords (x,y)
DANGER_ZONE_NORM = np.array([
    [0.05, 0.35],
    [0.50, 0.35],
    [0.50, 0.95],
    [0.05, 0.95],
], dtype=np.float32)

# Dwell-time threshold for alert (seconds of VIDEO time, not wall-clock)
DWELL_THRESHOLD_SEC = 1.0

# False-positive guard: person must be in zone for this many consecutive
# frames before the alert banner fires. At 30fps, 8 frames = ~0.27s.
MIN_BREACH_FRAMES = 8

# ─── Helpers ──────────────────────────────────────────────────────
COCO_NAMES = {
    0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',
    6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
    11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',
    16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',21:'bear',
    22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',
    27:'tie',28:'suitcase',56:'chair',57:'couch',62:'tv',63:'laptop',
    67:'cell phone',79:'toothbrush',
}

def point_in_polygon(pt, poly):
    """Return True if 2-D point (x,y) is inside poly (Nx2 int array)."""
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def draw_danger_zone(frame, zone_px):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_px], (0, 0, 180))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [zone_px], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "DANGER ZONE", (zone_px[0][0] + 5, zone_px[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

def draw_overlay(frame, frame_idx, src_fps, n_people, n_objects, zone_breach_ids):
    h, w = frame.shape[:2]
    # Video-position timestamp (not wall clock)
    video_sec = frame_idx / src_fps
    ts = f"Video time: {video_sec:.2f}s  (frame {frame_idx})"

    lines = [
        ("SafeGuard AI | Construction Site", (10, 28), 0.65, (0, 230, 255), 2),
        (f"Model: YOLOv8s+ByteTrack | Conf>={CONF_THRESHOLD}", (10, 52), 0.45, (200, 200, 200), 1),
        (ts, (10, 70), 0.45, (200, 200, 200), 1),
        # Real detection counts from YOLO output for THIS frame
        (f"Detected: {n_people} person(s), {n_objects} other object(s)", (10, 96), 0.50,
         (0, 255, 0) if n_people > 0 else (200, 200, 200), 1),
    ]
    for text, pos, scale, color, thick in lines:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    if zone_breach_ids:
        cv2.putText(frame,
            f"!! DANGER ZONE INTRUSION ({len(zone_breach_ids)} person(s), IDs: {zone_breach_ids})",
            (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)


def main():
    # ── Load model ──
    print(f"Loading model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  => running on {device.upper()}")

    # ── Open video ──
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_IN}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_duration = total_frames / src_fps
    out_w, out_h = 1280, 720
    print(f"  => source: {src_w}x{src_h} @ {src_fps:.1f} fps  ({total_frames} frames = {video_duration:.2f}s)")
    print(f"  => dwell threshold: {DWELL_THRESHOLD_SEC}s video-time | FP guard: {MIN_BREACH_FRAMES} consec frames")

    # ── Prepare output ──
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, src_fps, (out_w, out_h))

    # ── Danger zone in pixel space ──
    zone_px = (DANGER_ZONE_NORM * np.array([out_w, out_h])).astype(np.int32)

    # ── Per-track state (keyed by ByteTrack integer ID) ──
    # entry_frame: frame index when this track first entered the zone
    # consec_frames: how many consecutive frames it has been inside the zone
    track_zone_entry_frame = {}   # track_id -> frame_idx when first entered zone
    track_consec_in_zone   = {}   # track_id -> consecutive in-zone frame count

    frame_idx = 0
    total_people_seen = 0

    print(f"\nProcessing {total_frames} frames -> {VIDEO_OUT}")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = cv2.resize(frame, (out_w, out_h))

        # ── ByteTrack inference: persist=True keeps IDs across frames ──
        results = model.track(
            frame,
            verbose=False,
            conf=CONF_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
        )
        if results is None or len(results) == 0:
            writer.write(frame)
            print(f"  Frame {frame_idx:04d}: no results")
            continue
        result = results[0]

        n_people  = 0
        n_objects = 0
        zone_breach_ids = []          # track IDs confirmed in breach THIS frame
        active_zone_ids = set()       # track IDs inside zone this frame (for consec counter)

        boxes = result.boxes
        # ByteTrack assigns .id; fall back to None if tracking not initialised
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, track_ids):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            label  = COCO_NAMES.get(cls_id, f"cls{cls_id}")
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cls_id == PERSON_CLASS_ID:
                n_people += 1
                box_color = (0, 200, 0)

                # ── Danger zone: dwell time in VIDEO seconds ──
                if point_in_polygon((cx, cy), zone_px) and track_id is not None:
                    active_zone_ids.add(track_id)

                    # Record entry frame if first time in zone
                    if track_id not in track_zone_entry_frame:
                        track_zone_entry_frame[track_id] = frame_idx
                        track_consec_in_zone[track_id] = 1
                    else:
                        track_consec_in_zone[track_id] = track_consec_in_zone.get(track_id, 0) + 1

                    # Dwell = video frames elapsed / fps — NEVER wall-clock
                    entry_frame = track_zone_entry_frame[track_id]
                    dwell_video_sec = (frame_idx - entry_frame) / src_fps
                    consec = track_consec_in_zone[track_id]

                    # Clamp: dwell can never exceed total video duration
                    dwell_video_sec = min(dwell_video_sec, video_duration)

                    # Alert fires only when BOTH conditions are met:
                    #   1. dwell >= threshold in video-time seconds
                    #   2. consecutive in-zone frames >= MIN_BREACH_FRAMES (FP guard)
                    if dwell_video_sec >= DWELL_THRESHOLD_SEC and consec >= MIN_BREACH_FRAMES:
                        zone_breach_ids.append(track_id)
                        box_color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame,
                            f"ZONE BREACH {dwell_video_sec:.1f}s vid | {consec}f consec | conf={conf:.2f}",
                            (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    else:
                        # In zone but not yet breaching threshold — show progress
                        cv2.putText(frame,
                            f"in zone {dwell_video_sec:.1f}s/{DWELL_THRESHOLD_SEC}s ({consec}/{MIN_BREACH_FRAMES}f)",
                            (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)

                # Draw person bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                id_str = f"ID:{track_id}" if track_id is not None else "ID:?"
                cv2.putText(frame, f"person {conf:.2f} {id_str}",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, box_color, 2)
            else:
                n_objects += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 150, 0), 1)

        # Reset consecutive counter for tracks that LEFT the zone this frame
        for tid in list(track_consec_in_zone.keys()):
            if tid not in active_zone_ids:
                # Track has left zone — reset consecutive counter and entry frame
                track_consec_in_zone.pop(tid, None)
                track_zone_entry_frame.pop(tid, None)

        total_people_seen = max(total_people_seen, n_people)

        # ── Draw zone and overlays ──
        draw_danger_zone(frame, zone_px)
        draw_overlay(frame, frame_idx, src_fps, n_people, n_objects, zone_breach_ids)

        # ── Write frame ──
        writer.write(frame)

        # ── Per-frame terminal report ──
        zone_str = f" | ZONE BREACH IDs={zone_breach_ids}" if zone_breach_ids else ""
        print(f"  Frame {frame_idx:04d} ({frame_idx/src_fps:.2f}s): "
              f"{n_people} person(s), {n_objects} other(s){zone_str}")

    cap.release()
    writer.release()

    print("-" * 60)
    print(f"Done.  Output: {VIDEO_OUT}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Video duration: {video_duration:.2f}s")
    print(f"Max simultaneous people in any frame: {total_people_seen}")
    print("Dwell times are video-time (frame/fps) — wall-clock is not used.")
    print("FP guard: breach requires both dwell>=threshold AND consec frames>=MIN_BREACH_FRAMES.")

if __name__ == "__main__":
    main()
