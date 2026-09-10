"""Detection metrics, implemented locally so the numbers are auditable.

Everything takes a score array (higher means more likely to be an attack) and a
binary label array where 1 means attack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class OperatingPoint:
    """Metrics at a fixed threshold plus the threshold free summaries."""

    threshold: float
    precision: float
    recall: float
    f1: float
    eer: float
    auc: float
    counts: Counts
    n: int
    n_positive: int

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["counts"] = asdict(self.counts)
        return payload


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores and labels differ in length: {s.shape} vs {y.shape}")
    return s, y


def counts_at(scores, labels, threshold: float) -> Counts:
    """Confusion counts for the rule ``score >= threshold``."""
    s, y = _as_arrays(scores, labels)
    predicted = s >= threshold
    return Counts(
        tp=int(np.sum(predicted & (y == 1))),
        fp=int(np.sum(predicted & (y == 0))),
        tn=int(np.sum(~predicted & (y == 0))),
        fn=int(np.sum(~predicted & (y == 1))),
    )


def precision_recall_f1(counts: Counts) -> tuple[float, float, float]:
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve via the rank statistic, with tie correction."""
    s, y = _as_arrays(scores, labels)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False positive rate, true positive rate and thresholds, sorted by threshold desc."""
    s, y = _as_arrays(scores, labels)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = max(1, int(np.sum(y == 0)))
    tpr = np.r_[0.0, tps[idx] / n_pos]
    fpr = np.r_[0.0, fps[idx] / n_neg]
    thresholds = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thresholds


def equal_error_rate(scores, labels) -> tuple[float, float]:
    """Equal error rate and the threshold where it is reached."""
    s, y = _as_arrays(scores, labels)
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(s, y)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(crossings) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])
    i = int(crossings[0])
    d0, d1 = diff[i], diff[i + 1]
    alpha = 0.0 if d1 == d0 else float(-d0 / (d1 - d0))
    eer = float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))
    nearest = i + 1 if alpha >= 0.5 else i
    threshold = float(thresholds[nearest])
    if np.isinf(threshold):
        threshold = float(thresholds[min(nearest + 1, len(thresholds) - 1)])
    return eer, threshold


def evaluate(scores, labels, threshold: float) -> OperatingPoint:
    """Full metric block at a threshold."""
    s, y = _as_arrays(scores, labels)
    counts = counts_at(s, y, threshold)
    precision, recall, f1 = precision_recall_f1(counts)
    eer, _ = equal_error_rate(s, y)
    return OperatingPoint(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        eer=eer,
        auc=roc_auc(s, y),
        counts=counts,
        n=len(s),
        n_positive=int(np.sum(y == 1)),
    )


def family_breakdown(
    scores, labels, families: list[tuple[str, ...]], threshold: float
) -> dict[str, dict[str, float]]:
    """Detection rate per attack family, plus the bona fide false alarm rate.

    A clip that carries two families counts once for each.
    """
    s, _ = _as_arrays(scores, labels)
    flagged = s >= threshold
    rows: dict[str, dict[str, float]] = {}
    bona = [i for i, fam in enumerate(families) if not fam]
    if bona:
        rows["bonafide"] = {
            "n": float(len(bona)),
            "detected": float(np.sum(flagged[bona])),
            "rate": float(np.mean(flagged[bona])),
        }
    names = sorted({name for fam in families for name in fam})
    for name in names:
        idx = [i for i, fam in enumerate(families) if name in fam]
        rows[name] = {
            "n": float(len(idx)),
            "detected": float(np.sum(flagged[idx])),
            "rate": float(np.mean(flagged[idx])),
        }
    return rows
