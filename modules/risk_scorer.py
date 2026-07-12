"""
SafeGuard AI — Risk Scorer (P6)
================================
Per-worker risk score (0–100) combining:
  - Zone dwell time
  - Proximity to heavy machinery
  - Time-of-day (night shift = higher baseline risk)

Usage:
    from modules.risk_scorer import RiskScorer
    rs = RiskScorer()
    score, factors = rs.compute(dwell_sec=12.5, proximity_m=1.2, hour=14)
    # score: 0-100 int
    # factors: list of human-readable strings explaining the score

Can be wired into the dashboard or logged with each alert.
"""

import math
from typing import Optional


class RiskScorer:
    """
    Computes a 0–100 risk score per tracked person.

    Weights and thresholds are read from config if available,
    otherwise fall back to the built-in defaults below.
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get('risk_scorer', {}) or {}

        self.dwell_weight      = float(cfg.get('dwell_weight',      0.40))
        self.proximity_weight  = float(cfg.get('proximity_weight',  0.35))
        self.time_weight       = float(cfg.get('time_of_day_weight', 0.25))

        self.max_dwell_sec     = float(cfg.get('max_dwell_sec',     120.0))
        self.danger_prox_m     = float(cfg.get('proximity_danger_m', 0.5))
        self.night_hours       = set(cfg.get('night_hours', [0,1,2,3,4,5,22,23]))

        # Verify weights sum to ~1.0
        total = self.dwell_weight + self.proximity_weight + self.time_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Risk weights must sum to 1.0, got {total:.3f}")

    # ── Public API ────────────────────────────────────────────────

    def compute(self,
                dwell_sec: float,
                proximity_m: Optional[float],
                hour: int) -> tuple[int, list[str]]:
        """
        Returns (score: int 0-100, factors: list[str]).

        Parameters
        ----------
        dwell_sec    : seconds the person has been in the danger zone (video-time)
        proximity_m  : distance to nearest machinery in metres; None = no machinery nearby
        hour         : hour of day (0-23) for time-of-day factor
        """
        factors = []

        # ── 1. Dwell contribution ────────────────────────────────
        dwell_norm = min(dwell_sec / self.max_dwell_sec, 1.0)
        dwell_score = dwell_norm * 100
        if dwell_sec > 0:
            factors.append(
                f"Dwell {dwell_sec:.1f}s in danger zone "
                f"({dwell_norm*100:.0f}% of {self.max_dwell_sec:.0f}s max)"
            )

        # ── 2. Proximity contribution ────────────────────────────
        if proximity_m is not None:
            if proximity_m <= 0:
                prox_norm = 1.0
            else:
                # Inverse-distance: closer = higher score, decays with distance
                # At danger_prox_m → 1.0, at 10x danger distance → ~0.0
                prox_norm = min(1.0, self.danger_prox_m / max(proximity_m, 0.01))
            prox_score = prox_norm * 100
            factors.append(
                f"Machinery proximity {proximity_m:.1f}m "
                f"({prox_norm*100:.0f}% of max proximity risk)"
            )
        else:
            prox_score = 0.0
            prox_norm = 0.0

        # ── 3. Time-of-day contribution ──────────────────────────
        if hour in self.night_hours:
            time_norm = 1.0
            factors.append(f"Night-shift hour ({hour:02d}:xx) — elevated baseline risk")
        else:
            # Mild elevation during low-traffic hours (06:00–07:00, 20:00–21:00)
            border_hours = {6, 7, 20, 21}
            time_norm = 0.4 if hour in border_hours else 0.0
            if time_norm > 0:
                factors.append(f"Low-light/border hour ({hour:02d}:xx)")
        time_score = time_norm * 100

        # ── Weighted sum ─────────────────────────────────────────
        raw = (
            self.dwell_weight     * dwell_score +
            self.proximity_weight * prox_score  +
            self.time_weight      * time_score
        )
        score = int(min(100, max(0, round(raw))))

        if not factors:
            factors.append("No significant risk factors detected")

        return score, factors

    def score_label(self, score: int) -> str:
        """Human-readable label for dashboard display."""
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "LOW"

    def compute_labeled(self,
                        dwell_sec: float,
                        proximity_m: Optional[float],
                        hour: int) -> dict:
        """Returns full dict suitable for logging / JSON serialisation."""
        score, factors = self.compute(dwell_sec, proximity_m, hour)
        return {
            "risk_score": score,
            "risk_label": self.score_label(score),
            "factors": factors,
            "inputs": {
                "dwell_sec": dwell_sec,
                "proximity_m": proximity_m,
                "hour": hour,
            }
        }
