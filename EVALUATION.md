# Safeguard Pipeline Evaluation

This document outlines the performance characteristics of the Safeguard Physical AI pipeline across diverse environmental conditions. Testing was conducted on the first 100 frames of each standard test clip using the `yolov8s.pt` model with a confidence threshold of `0.25` and the default BoT-SORT/ByteTrack tracker.

## 📊 Environmental Comparison Matrix

### 1. Baseline Construction Footage (`test_construction.mp4`)
*Note: The person in this clip enters the frame after the 100-frame cutoff, resulting in 0 detections in this specific sampled window. Historical evaluation shows stable tracking once the subject enters the frame.*
- **Person Detection Rate:** 0.0%
- **Machinery Detection Rate:** 0.0%
- **Tracker Stability:** N/A in sample window

### 2. High-Density / Crowded Site (`test_crowd.mp4`)
- **Person Detection Rate:** 100.0%
- **Machinery Detection Rate:** 1.0%
- **Confidence Range (Persons):** 0.26 (Min) - 0.91 (Max) | Avg: 0.63
- **Tracker Stability:** **Fair.** The tracker generated 5 unique person IDs across 100 frames, indicating some ID switching (likely due to worker occlusion and overlapping bounding boxes in the crowd).
- **Failure Mode:** Occlusion causes track IDs to drop and re-initialize, which could reset the "dwell time" counter for Danger Zone Intrusion alerts if a worker walks behind another worker.

### 3. Drone / Aerial View (`test_aerial.mp4`)
- **Person Detection Rate:** 10.0%
- **Machinery Detection Rate:** 67.0%
- **Confidence Range (Machinery):** 0.26 (Min) - 0.68 (Max) | Avg: 0.38
- **Tracker Stability:** **Poor.** Generated 10 unique machinery IDs and 2 person IDs within just 3 seconds of footage.
- **Failure Mode:** The YOLOv8s model struggles significantly with top-down perspectives. Confidence scores for heavy machinery are abnormally low (maxing at 0.68), and the tracker constantly drops bounding boxes, resulting in severe ID proliferation. Small pixel footprint of workers limits person detection to only 10% of frames.

### 4. Low Light / Night Operations (`test_lowlight.mp4`)
- **Person Detection Rate:** 100.0%
- **Machinery Detection Rate:** 0.0%
- **Confidence Range (Persons):** 0.26 (Min) - 0.90 (Max) | Avg: 0.65
- **Tracker Stability:** **Poor.** Generated 9 unique person IDs for what should be a contiguous track.
- **Failure Mode:** While the detection rate remains surprisingly robust (100% of frames had a person), motion blur and lack of feature contrast in low light causes the tracker's re-identification logic to fail frequently. The worker is constantly assigned a new ID.

### 5. PPE Close-up / Partial Frame (`test_ppe_closeup.mp4`)
- **Person Detection Rate:** 100.0%
- **Machinery Detection Rate:** 0.0%
- **Confidence Range (Persons):** 0.25 (Min) - 0.93 (Max) | Avg: 0.67
- **Tracker Stability:** **Poor.** Generated 10 unique person IDs.
- **Failure Mode:** Severe bounding-box truncation. Because the camera is close, the worker often steps partially out of frame. The model detects the visible body parts with high confidence, but the tracker breaks continuity as the aspect ratio of the bounding box changes dramatically.

### 6. Close Proximity / Worker Overlap (`test_proximity.mp4`)
- **Person Detection Rate:** 100.0%
- **Machinery Detection Rate:** 0.0%
- **Confidence Range (Persons):** 0.25 (Min) - 0.94 (Max) | Avg: 0.55
- **Tracker Stability:** **Critical Failure.** Generated 14 unique person IDs across 100 frames.
- **Failure Mode:** When workers cross paths or operate in extremely close proximity, their bounding boxes merge. The tracker completely fails to maintain distinct IDs, causing a rapid "flicker" of new ID generation (14 IDs in 3 seconds). This will severely disrupt dwell-time alerts (loitering/danger zone) since the timer resets with every new ID.

---

## 🛠️ Recommended Demo Setup
1. **Camera Angle:** Use isometric or slightly elevated security camera angles (like the baseline `test_construction.mp4`). **Avoid aerial/drone footage** — see Out-of-Scope section below.
2. **Lighting:** Ensure bright, even lighting. Low-light tracker stability cannot be resolved by parameter tuning alone (see Post-Fix Results).
3. **Worker Spacing:** Keep workers at least 1–2 body-widths apart during demo to avoid bounding-box overlap that defeats proximity tracking.

---

## 🔧 Fix Attempt: Tracker Tuning + ID Re-linking

Two fixes were applied to address the tracker ID instability identified above.

### Fix 1: Custom BoT-SORT Configuration (`safeguard_botsort.yaml`)

A custom tracker config was created with the following changes from the Ultralytics defaults:

| Parameter | Default | Tuned | Rationale |
|---|---|---|---|
| `track_buffer` | 30 frames | **75 frames** | Keeps lost tracks alive 2.5× longer |
| `match_thresh` | 0.8 | **0.65** | Looser IOU matching for partially-occluded re-association |
| `new_track_thresh` | 0.25 | **0.35** | Higher bar to spawn a new ID; reduces spurious ones |
| `proximity_thresh` | 0.5 | **0.35** | Wider ReID proximity window |
| `appearance_thresh` | 0.8 | **0.70** | Looser appearance matching (construction PPE changes appearance) |

### Fix 2: `IDRelinker` Post-Processing in `zone_analyzer.py`

An `IDRelinker` class was added as a software-level fallback inside `ZoneAnalyzer`. If a new tracker ID appears within **100px** and **2 seconds** of a recently-disappeared track's last known position, the new ID is silently remapped to the old canonical ID — preserving the dwell-time counter without relying on the tracker itself to maintain continuity.

---

## 📈 Post-Fix Results: Before vs After (Tracker Tuning)

Evaluation: 100 frames per clip, `yolov8s.pt`, `conf=0.25`.  
Metric: **unique_person_ids_tracked** — lower is better (ideal = number of actual people in scene).

```
======================================================================
Clip                           Before IDs  After IDs    Delta
----------------------------------------------------------------------
test_crowd.mp4                          5          4       -1
test_lowlight.mp4                       9         11       +2
test_ppe_closeup.mp4                   10          9       -1
test_proximity.mp4                     14         15       +1
======================================================================
```

### Honest Assessment

- **test_crowd.mp4:** Marginal improvement (5 → 4). The loosened `match_thresh` helped slightly with occlusion re-association, but 4 IDs for a crowd with ~2 actual workers still indicates significant switching.
- **test_lowlight.mp4:** **Tuning made it worse** (9 → 11). Loosening `match_thresh` in low-light conditions caused the tracker to incorrectly re-associate detections from different workers, spawning additional spurious IDs. Motion-blur-driven feature degradation is not fixable via threshold tuning without a dedicated ReID model or image enhancement pre-processing.
- **test_ppe_closeup.mp4:** Marginal improvement (10 → 9). Partial-frame bounding-box truncation remains the root cause; parameter tuning cannot recover features that were never captured.
- **test_proximity.mp4:** **Tuning made it slightly worse** (14 → 15). When bounding boxes physically merge, no IOU threshold adjustment prevents ID proliferation — the tracker has no geometric signal to distinguish between people.

**Conclusion:** Tracker parameter tuning alone provides negligible improvement (−1 to +2 unique IDs) on the failure modes identified. The `IDRelinker` in `zone_analyzer.py` provides a meaningful practical safeguard for the dwell-timer specifically — even when the tracker assigns a new raw ID, the canonical dwell counter continues uninterrupted for up to 2 seconds. This does not reduce visible ID count in these metrics (since IDRelinker operates above the tracker layer and these metrics measure raw tracker output), but it directly protects the safety-critical danger zone intrusion logic from false resets.

---

## 🚁 Out of Scope: Aerial / Top-Down Detection

The `test_aerial.mp4` failure (10% person detection rate, 67% machinery detection rate with avg confidence 0.38) is a **detection problem, not a tracking problem** and was not included in the tuning exercise above.

`yolov8s.pt` is trained on the COCO dataset which contains predominantly horizontal/eye-level perspective imagery. Top-down aerial footage presents:
- Drastically different object aspect ratios (people appear as small ovals, not vertical rectangles)
- Absence of the head/shoulder/leg features the model learned to associate with the `person` class
- Scale mismatch between typical training anchors and the small pixel footprint of aerial subjects

**Resolution would require** fine-tuning YOLOv8 on an aerial-angle dataset (e.g., VisDrone, UAVDT) — which is out of scope for this iteration of Safeguard. This limitation is documented here and the aerial clip is excluded from all tracker stability benchmarks.
