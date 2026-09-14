"""Score level fusion of the two calibrated streams.

Both streams are mapped to a probability by their own calibrator, so a convex
combination is meaningful. The weight is fitted on the calibration split by
maximising recall subject to the same target precision that fixes the single
stream thresholds, which means the fused operating point is chosen the same way
as the ones it is compared against. A logistic fusion over both probabilities and
their disagreement is fitted on the same split and given its operating point the
same way. The AND and OR rules over the two single stream decisions are kept as
a reference, and `attribute` names the stream that triggered each decision.
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


LOGISTIC_FEATURES = ("p_video", "p_audio", "disagreement")
ATTRIBUTIONS = ("none", "video", "audio", "either", "joint")


def fusion_features(p_video, p_audio) -> np.ndarray:
    """Both calibrated probabilities plus their absolute disagreement, one row per clip.

    ``max(pv, pa) = (pv + pa) / 2 + |pv - pa| / 2``, so with the disagreement term a
    linear model can express an OR of the two streams as well as an average.
    """
    pv = np.asarray(p_video, dtype=np.float64).ravel()
    pa = np.asarray(p_audio, dtype=np.float64).ravel()
    return np.column_stack([pv, pa, np.abs(pv - pa)])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass(frozen=True)
class LogisticFusion:
    """Logistic regression over the fusion features with its own operating threshold."""

    coefficients: tuple[float, ...]
    intercept: float
    operating: OperatingThreshold

    def fuse(self, p_video, p_audio) -> np.ndarray:
        z = fusion_features(p_video, p_audio) @ np.asarray(self.coefficients) + self.intercept
        return _sigmoid(z)

    def decide(self, p_video, p_audio) -> np.ndarray:
        return self.fuse(p_video, p_audio) >= self.operating.threshold

    def as_dict(self) -> dict:
        coefficients = zip(LOGISTIC_FEATURES, map(float, self.coefficients), strict=True)
        return {
            "coefficients": dict(coefficients),
            "intercept": float(self.intercept),
            "operating": self.operating.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> LogisticFusion:
        return cls(
            coefficients=tuple(float(payload["coefficients"][name]) for name in LOGISTIC_FEATURES),
            intercept=float(payload["intercept"]),
            operating=OperatingThreshold(**payload["operating"]),
        )


def fit_logistic_weights(
    features: np.ndarray, labels, ridge: float = 1.0, iterations: int = 100
) -> tuple[np.ndarray, float]:
    """Newton fit of an L2 penalised logistic regression, intercept unpenalised."""
    x = np.column_stack([np.asarray(features, dtype=np.float64), np.ones(len(features))])
    y = np.asarray(labels, dtype=np.float64).ravel()
    penalty = np.full(x.shape[1], float(ridge))
    penalty[-1] = 1e-9
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + penalty * w
        hess = (x * (p * (1.0 - p))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return w[:-1], float(w[-1])


def fit_logistic_fusion(
    p_video, p_audio, labels, target_precision: float, ridge: float = 1.0
) -> LogisticFusion:
    """Fit the weights on the calibration split, then search for the operating point there.

    The operating point is the threshold with the most recall among those whose
    precision on the calibration split reaches the target.
    """
    features = fusion_features(p_video, p_audio)
    coefficients, intercept = fit_logistic_weights(features, labels, ridge)
    scores = _sigmoid(features @ coefficients + intercept)
    operating = threshold_at_precision(scores, labels, target_precision)
    return LogisticFusion(tuple(float(c) for c in coefficients), intercept, operating)


def attribute(decide, p_video, p_audio) -> list[str]:
    """Which stream triggered each decision, by silencing one stream at a time.

    A silenced stream reports probability 0. ``video`` or ``audio`` means that
    stream alone keeps the clip flagged and the other alone does not, ``either``
    means each alone would flag it, ``joint`` means only the two together do, and
    ``none`` means the clip was not flagged.
    """
    pv = np.asarray(p_video, dtype=np.float64).ravel()
    pa = np.asarray(p_audio, dtype=np.float64).ravel()
    silent = np.zeros_like(pv)
    flagged = np.asarray(decide(pv, pa), dtype=bool)
    video_alone = np.asarray(decide(pv, silent), dtype=bool)
    audio_alone = np.asarray(decide(silent, pa), dtype=bool)
    labels = np.select(
        [~flagged, video_alone & audio_alone, video_alone, audio_alone],
        ["none", "either", "video", "audio"],
        default="joint",
    )
    return labels.tolist()
