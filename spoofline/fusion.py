"""Score level fusion of the two calibrated streams.

Both streams are mapped to a probability by their own calibrator, so a convex
combination is meaningful. The weight is fitted on the calibration split by
maximising recall subject to the same target precision that fixes the single
stream thresholds, which means the fused operating point is chosen the same way
as the ones it is compared against. The AND and OR rules over the two single
stream decisions are kept as a reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import OperatingThreshold, StreamCalibration, threshold_at_precision


@dataclass(frozen=True)
class FusionModel:
    """Weighted sum fusion with its own operating threshold."""

    weight: float
    operating: OperatingThreshold

    def fuse(self, p_video, p_audio) -> np.ndarray:
        pv = np.asarray(p_video, dtype=np.float64)
        pa = np.asarray(p_audio, dtype=np.float64)
        return self.weight * pv + (1.0 - self.weight) * pa

    def decide(self, p_video, p_audio) -> np.ndarray:
        return self.fuse(p_video, p_audio) >= self.operating.threshold

    def as_dict(self) -> dict:
        return {"weight": float(self.weight), "operating": self.operating.as_dict()}


def fit_fusion(
    p_video,
    p_audio,
    labels,
    target_precision: float,
    grid: int = 101,
) -> FusionModel:
    """Grid search the fusion weight on the calibration split.

    The grid includes 0 and 1, so the fused operating point can never be worse
    than the better single stream on the calibration split.
    """
    pv = np.asarray(p_video, dtype=np.float64)
    pa = np.asarray(p_audio, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    best_weight, best_point = 0.5, None
    for weight in np.linspace(0.0, 1.0, grid):
        fused = weight * pv + (1.0 - weight) * pa
        point = threshold_at_precision(fused, y, target_precision)
        if best_point is None:
            best_weight, best_point = float(weight), point
            continue
        better = (point.reached_target, point.recall, point.achieved_precision) > (
            best_point.reached_target,
            best_point.recall,
            best_point.achieved_precision,
        )
        if better:
            best_weight, best_point = float(weight), point
    assert best_point is not None
    return FusionModel(weight=best_weight, operating=best_point)


def and_rule(video: StreamCalibration, audio: StreamCalibration, p_video, p_audio) -> np.ndarray:
    """Both streams must flag the clip. Inputs are already calibrated probabilities."""
    return _flags(video, p_video) & _flags(audio, p_audio)


def or_rule(video: StreamCalibration, audio: StreamCalibration, p_video, p_audio) -> np.ndarray:
    """Either stream flagging the clip is enough. Inputs are already calibrated probabilities."""
    return _flags(video, p_video) | _flags(audio, p_audio)


def _flags(calibration: StreamCalibration, probabilities) -> np.ndarray:
    return np.asarray(probabilities, dtype=np.float64) >= calibration.operating.threshold


def rule_metrics(decisions: np.ndarray, labels) -> dict[str, float]:
    """Precision, recall and F1 for a boolean decision rule."""
    y = np.asarray(labels, dtype=np.int64)
    tp = int(np.sum(decisions & (y == 1)))
    fp = int(np.sum(decisions & (y == 0)))
    fn = int(np.sum(~decisions & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
