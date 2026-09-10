"""Calibration is monotonic and its threshold hits the requested precision."""

from __future__ import annotations

import numpy as np

from spoofline.calibrate import PlattCalibrator, calibrate_stream, threshold_at_precision


def _separable(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n // 2)
    scores = np.where(labels == 1, rng.normal(2.0, 1.0, n), rng.normal(-2.0, 1.0, n))
    return scores, labels


def test_calibrator_is_monotonic_in_the_raw_score():
    scores, labels = _separable()
    calibrator = PlattCalibrator.fit(scores, labels)
    sweep = np.linspace(-8, 8, 200)
    probabilities = calibrator.predict(sweep)
    assert np.all(np.diff(probabilities) >= 0)
    assert calibrator.a >= 0


def test_calibrated_probabilities_stay_in_range():
    scores, labels = _separable()
    probabilities = PlattCalibrator.fit(scores, labels).predict(scores)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_calibrated_mean_is_close_to_the_base_rate():
    scores, labels = _separable(n=400, seed=3)
    probabilities = PlattCalibrator.fit(scores, labels).predict(scores)
    assert abs(float(probabilities.mean()) - float(labels.mean())) < 0.05


def test_calibrator_falls_back_to_the_base_rate_when_the_score_runs_backwards():
    scores, labels = _separable()
    calibrator = PlattCalibrator.fit(-scores, labels)
    assert calibrator.a == 0.0
    assert np.allclose(calibrator.predict([0.0, 5.0]), calibrator.predict([0.0, -5.0]))


def test_calibrator_fit_is_deterministic():
    scores, labels = _separable()
    first = PlattCalibrator.fit(scores, labels)
    second = PlattCalibrator.fit(scores, labels)
    assert first.as_dict() == second.as_dict()


def test_threshold_reaches_the_requested_precision():
    scores, labels = _separable(n=400, seed=1)
    for target in (0.8, 0.9, 0.95):
        point = threshold_at_precision(scores, labels, target)
        assert point.reached_target
        assert point.achieved_precision >= target
        flagged = scores >= point.threshold
        precision = float(np.sum(flagged & (labels == 1)) / max(1, np.sum(flagged)))
        assert precision >= target


def test_threshold_maximises_recall_subject_to_the_target():
    scores, labels = _separable(n=400, seed=2)
    point = threshold_at_precision(scores, labels, 0.9)
    lower = point.threshold - 1e-6
    flagged = scores >= lower
    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    if tp + fp > 0 and tp / (tp + fp) >= 0.9:
        assert tp / max(1, int(np.sum(labels == 1))) <= point.recall


def test_unreachable_target_is_reported_rather_than_faked():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = rng.normal(0, 1, 200)
    point = threshold_at_precision(scores, labels, 0.999)
    assert not point.reached_target
    assert point.achieved_precision < 0.999


def test_calibrate_stream_bundles_map_and_threshold():
    scores, labels = _separable(n=300, seed=4)
    calibration = calibrate_stream("video", scores, labels, 0.9)
    assert calibration.stream == "video"
    assert calibration.operating.reached_target
    decisions = calibration.decide(scores)
    flagged = int(decisions.sum())
    assert flagged > 0
    precision = float(np.sum(decisions & (labels == 1)) / flagged)
    assert precision >= 0.9


def test_calibration_round_trips_through_a_dict():
    scores, labels = _separable()
    calibrator = PlattCalibrator.fit(scores, labels)
    restored = PlattCalibrator.from_dict(calibrator.as_dict())
    assert np.allclose(restored.predict(scores), calibrator.predict(scores))
