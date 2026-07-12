import numpy as np
import cv2
from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)


class IDRelinker:
    """
    Post-processing fallback: if a new tracker ID appears within a short time window
    AND close spatial proximity to a recently-disappeared track, treat it as a
    continuation of that track rather than a fresh one.

    This protects dwell-time counters in ZoneAnalyzer from resetting when the tracker
    momentarily drops and reassigns an ID (common in occlusion, low-light, and
    close-proximity scenarios).

    Parameters
    ----------
    spatial_threshold_px : int
        Max pixel distance between last known centroid of a lost track and
        the centroid of the candidate new track to be considered the same person.
        Default 100px — covers ~0.5 body-width at typical security-cam distance.
    time_window_s : float
        Max seconds a lost track will be held in the "ghost" buffer before being
        permanently evicted. Default 2.0s — covers short occlusions.
    fps : int
        Estimated video FPS, used to convert track_buffer frames to seconds.
    """

    def __init__(self, spatial_threshold_px: int = 100, time_window_s: float = 2.0, fps: int = 30):
        self.spatial_threshold_px = spatial_threshold_px
        self.time_window_s = time_window_s
        self.fps = fps

        # ghost_tracks: {canonical_id -> {'centroid': (x,y), 'lost_at': timestamp}}
        self.ghost_tracks: dict = {}
        # raw_to_canonical: {raw_tracker_id -> canonical_id}
        self.raw_to_canonical: dict = {}
        # last seen centroid for each canonical id
        self.canonical_centroids: dict = {}

    def remap(self, raw_detections: dict) -> dict:
        """
        Takes a dict of {raw_track_id: {'centroid': (cx, cy), ...}} from the tracker
        and returns a remapped dict using stable canonical IDs.

        Steps:
        1. For each raw ID that is NEW (not seen before), check if it matches
           any ghost track within spatial_threshold_px and time_window_s.
           If yes → remap to that ghost's canonical_id.
           If no  → assign a new canonical ID (same as raw_id for simplicity).
        2. For each canonical ID that was alive last frame but is MISSING now,
           move it to the ghost buffer with its last centroid and timestamp.
        3. Evict ghosts older than time_window_s.
        """
        now = time.monotonic()
        remapped = {}
        active_canonical = set()

        for raw_id, info in raw_detections.items():
            cx, cy = info['centroid']

            if raw_id in self.raw_to_canonical:
                # Known raw ID — use existing canonical
                canon_id = self.raw_to_canonical[raw_id]
            else:
                # New raw ID — try to match against ghost tracks
                matched_canon = None
                best_dist = float('inf')

                for ghost_canon, ghost_info in self.ghost_tracks.items():
                    age = now - ghost_info['lost_at']
                    if age > self.time_window_s:
                        continue  # Too old — will be evicted below
                    gx, gy = ghost_info['centroid']
                    dist = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                    if dist < self.spatial_threshold_px and dist < best_dist:
                        best_dist = dist
                        matched_canon = ghost_canon

                if matched_canon is not None:
                    # Re-link to ghost → continue its dwell timer
                    canon_id = matched_canon
                    del self.ghost_tracks[matched_canon]
                    logger.debug(
                        f"IDRelinker: raw_id={raw_id} re-linked to canon_id={canon_id} "
                        f"(dist={best_dist:.1f}px)"
                    )
                else:
                    # Genuinely new track
                    canon_id = raw_id

                self.raw_to_canonical[raw_id] = canon_id

            # Update canonical centroid and mark as active
            self.canonical_centroids[canon_id] = (cx, cy)
            active_canonical.add(canon_id)

            remapped[canon_id] = {**info, 'raw_id': raw_id}

        # Move newly lost tracks to ghost buffer
        for canon_id, centroid in list(self.canonical_centroids.items()):
            if canon_id not in active_canonical:
                if canon_id not in self.ghost_tracks:
                    self.ghost_tracks[canon_id] = {
                        'centroid': centroid,
                        'lost_at': now,
                    }
                    logger.debug(f"IDRelinker: canon_id={canon_id} moved to ghost buffer")

        # Evict expired ghosts and clean up mappings
        expired = [cid for cid, g in self.ghost_tracks.items()
                   if now - g['lost_at'] > self.time_window_s]
        for cid in expired:
            del self.ghost_tracks[cid]
            if cid in self.canonical_centroids:
                del self.canonical_centroids[cid]
            # Clean raw→canonical mappings for expired ghost
            dead_raws = [r for r, c in self.raw_to_canonical.items() if c == cid]
            for r in dead_raws:
                del self.raw_to_canonical[r]

        return remapped


class ZoneAnalyzer:
    """Accurate zone-based analytics and heatmap generation.
    
    Now includes ID re-linking via IDRelinker to maintain dwell-time continuity
    even when the upstream tracker drops and reassigns IDs during occlusion,
    low-light, or close-proximity scenarios.
    """

    def __init__(self, width, height, grid_size=(6, 6),
                 relink_spatial_px: int = 100,
                 relink_time_window_s: float = 2.0,
                 fps: int = 30):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.zone_width = width // grid_size[0]
        self.zone_height = height // grid_size[1]
        self.heatmap = np.zeros(grid_size, dtype=np.float32)
        self.zone_counts = defaultdict(int)

        # ID re-linking fallback
        self.relinker = IDRelinker(
            spatial_threshold_px=relink_spatial_px,
            time_window_s=relink_time_window_s,
            fps=fps,
        )

    def update_zones(self, objects_info):
        """Update zone occupancy based on tracked objects.

        objects_info expected format:
            {object_id_str: {'centroid': (cx, cy), ...}, ...}

        Internally remaps IDs through IDRelinker before computing zones,
        so dwell-time counters are stable across brief tracker dropouts.
        """
        # Convert string keys to int keys for relinker then remap
        int_keyed = {}
        for k, v in objects_info.items():
            try:
                int_keyed[int(k)] = v
            except (ValueError, TypeError):
                int_keyed[k] = v

        remapped = self.relinker.remap(int_keyed)

        current_zones = defaultdict(int)
        for canon_id, info in remapped.items():
            centroid = info['centroid']
            zone_x = min(int(centroid[0] // self.zone_width), self.grid_size[0] - 1)
            zone_y = min(int(centroid[1] // self.zone_height), self.grid_size[1] - 1)

            zone_x = max(0, zone_x)
            zone_y = max(0, zone_y)

            zone_key = f"{zone_x}_{zone_y}"
            current_zones[zone_key] += 1
            self.zone_counts[zone_key] += 1
            self.heatmap[zone_y, zone_x] += 0.1

        return dict(current_zones)

    def get_heatmap_overlay(self, image):
        """Generate accurate heatmap overlay for visualization"""
        try:
            overlay = image.copy()

            if np.max(self.heatmap) > 0:
                normalized_heatmap = (self.heatmap / np.max(self.heatmap) * 255).astype(np.uint8)
                colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
                resized_heatmap = cv2.resize(colored_heatmap, (self.width, self.height))
                overlay = cv2.addWeighted(image, 0.6, resized_heatmap, 0.4, 0)

            for i in range(1, self.grid_size[0]):
                x = i * self.zone_width
                cv2.line(overlay, (x, 0), (x, self.height), (255, 255, 255), 1)

            for i in range(1, self.grid_size[1]):
                y = i * self.zone_height
                cv2.line(overlay, (0, y), (self.width, y), (255, 255, 255), 1)

            return overlay
        except Exception as e:
            logger.error(f"Error generating heatmap overlay: {e}")
            return image