"""Per-stream score calibration and threshold selection.

Raw logits from the two streams are not comparable: they come from different
networks trained on different modality labels. A Platt scaling fitted on the
calibration split maps each stream's logit to an estimate of P(clip is an
attack), which puts both streams on the same axis and lets a single target
precision define an operating point for each of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlattCalibrator:
    """Monotonic logistic map ``p = sigmoid(a * score + b)`` with ``a >= 0``."""

    a: float
    b: float

    def predict(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(self.a * s + self.b)))

    def as_dict(self) -> dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> PlattCalibrator:
        return cls(a=float(payload["a"]), b=float(payload["b"]))

    @classmethod
    def fit(cls, scores, labels, iterations: int = 100, ridge: float = 1e-6) -> PlattCalibrator:
        """Newton fit of a two parameter logistic on the raw scores.

        Platt's smoothed targets are used so that a separable calibration set does
        not push the parameters to infinity.
        """
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        hi = (n_pos + 1.0) / (n_pos + 2.0) if n_pos else 0.5
        lo = 1.0 / (n_neg + 2.0) if n_neg else 0.5
        target = np.where(y == 1, hi, lo)

        scale = float(np.std(s)) or 1.0
        z = s / scale
        a, b = 0.0, 0.0
        for _ in range(iterations):
            p = 1.0 / (1.0 + np.exp(-(a * z + b)))
            w = np.clip(p * (1.0 - p), 1e-9, None)
            residual = p - target
            grad = np.array([float(np.dot(residual, z)), float(residual.sum())])
            hess = np.array(
                [
                    [float(np.dot(w * z, z)) + ridge, float(np.dot(w, z))],
                    [float(np.dot(w, z)), float(w.sum()) + ridge],
                ]
            )
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:  # pragma: no cover - ridge keeps this invertible
                break
            a -= float(step[0])
            b -= float(step[1])
            if float(np.max(np.abs(step))) < 1e-10:
                break
        a_scaled = a / scale
        if a_scaled < 0.0:
            # A stream whose score runs backwards would break the fusion axis; keep
            # the map non decreasing and fall back to the base rate offset.
            a_scaled = 0.0
            b = float(np.log((n_pos + 1.0) / (n_neg + 1.0)))
        return cls(a=a_scaled, b=b)


@dataclass(frozen=True)
class OperatingThreshold:
    """A threshold chosen to hit a target precision on the calibration set."""

    threshold: float
    target_precision: float
    achieved_precision: float
    recall: float
    reached_target: bool

    def as_dict(self) -> dict:
        return {
            "threshold": float(self.threshold),
            "target_precision": float(self.target_precision),
            "achieved_precision": float(self.achieved_precision),
            "recall": float(self.recall),
            "reached_target": bool(self.reached_target),
        }


def threshold_at_precision(scores, labels, target_precision: float) -> OperatingThreshold:
    """Lowest threshold whose precision reaches the target, which maximises recall.

    If no threshold reaches the target, the threshold with the highest precision is
    returned and ``reached_target`` is False, so the caller can report it honestly.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if len(s) == 0:
        raise ValueError("cannot pick a threshold from an empty calibration set")
    candidates = np.unique(s)
    best: OperatingThreshold | None = None
    fallback: OperatingThreshold | None = None
    n_pos = max(1, int(np.sum(y == 1)))
    for threshold in candidates:
        flagged = s >= threshold
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / n_pos
        point = OperatingThreshold(
            threshold=float(threshold),
            target_precision=float(target_precision),
            achieved_precision=float(precision),
            recall=float(recall),
            reached_target=bool(precision >= target_precision),
        )
        if point.reached_target and (best is None or point.recall > best.recall):
            best = point
        if fallback is None or (
            point.achieved_precision > fallback.achieved_precision
            or (
                point.achieved_precision == fallback.achieved_precision
                and point.recall > fallback.recall
            )
        ):
            fallback = point
    return best or fallback  # type: ignore[return-value]


@dataclass(frozen=True)
class StreamCalibration:
    """Calibrator plus operating threshold for one stream."""

    stream: str
    calibrator: PlattCalibrator
    operating: OperatingThreshold

    def probabilities(self, scores) -> np.ndarray:
        return self.calibrator.predict(scores)

    def decide(self, scores) -> np.ndarray:
        return self.probabilities(scores) >= self.operating.threshold

    def as_dict(self) -> dict:
        return {
            "stream": self.stream,
            "calibrator": self.calibrator.as_dict(),
            "operating": self.operating.as_dict(),
        }


def calibrate_stream(
    stream: str,
    calib_scores,
    modality_labels,
    clip_labels,
    target_precision: float,
) -> StreamCalibration:
    """Fit the calibration map and pick the operating threshold on the calibration split.

    The map is fitted against the stream's own modality label, so the probability
    means "this modality was attacked" and does not silently absorb the corpus wide
    attack prior. The threshold is then chosen against the clip label, because that
    is the decision a deployed detector actually makes, and because it puts the
    single stream operating points and the fused one on the same footing.
    """
    calibrator = PlattCalibrator.fit(calib_scores, modality_labels)
    probabilities = calibrator.predict(calib_scores)
    operating = threshold_at_precision(probabilities, clip_labels, target_precision)
    return StreamCalibration(stream=stream, calibrator=calibrator, operating=operating)
