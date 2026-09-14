"""Learned logistic fusion, its operating point search and per clip attribution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spoofline.calibrate import OperatingThreshold, calibrate_stream
from spoofline.config import FIXTURE_DIR, profile
from spoofline.fusion import (
    ATTRIBUTIONS,
    FusionModel,
    LogisticFusion,
    attribute,
    fit_logistic_fusion,
    rule_metrics,
)
from spoofline.pipeline import run_pipeline


def _calibrated_streams(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    video_attacked = rng.integers(0, 2, n)
    audio_attacked = rng.integers(0, 2, n)
    labels = ((video_attacked + audio_attacked) > 0).astype(int)
    video_scores = np.where(video_attacked == 1, rng.normal(2.0, 1.2, n), rng.normal(-2.0, 1.2, n))
    audio_scores = np.where(audio_attacked == 1, rng.normal(2.0, 1.2, n), rng.normal(-2.0, 1.2, n))
    video = calibrate_stream("video", video_scores, video_attacked, labels, 0.9)
    audio = calibrate_stream("audio", audio_scores, audio_attacked, labels, 0.9)
    return video.probabilities(video_scores), audio.probabilities(audio_scores), labels


def _operating(threshold: float) -> OperatingThreshold:
    return OperatingThreshold(threshold, 0.9, 1.0, 1.0, True)


def test_logistic_fit_is_deterministic_and_independent_of_row_order():
    pv, pa, labels = _calibrated_streams()
    first = fit_logistic_fusion(pv, pa, labels, 0.9)
    second = fit_logistic_fusion(pv, pa, labels, 0.9)
    assert first == second
    order = np.random.default_rng(1).permutation(len(labels))
    shuffled = fit_logistic_fusion(pv[order], pa[order], labels[order], 0.9)
    assert np.allclose(shuffled.coefficients, first.coefficients, atol=1e-8)
    assert shuffled.intercept == pytest.approx(first.intercept, abs=1e-8)
    assert shuffled.operating.threshold == pytest.approx(first.operating.threshold, abs=1e-9)


def test_precision_constrained_search_meets_its_target_on_the_calibration_split():
    pv, pa, labels = _calibrated_streams(seed=4)
    fusion = fit_logistic_fusion(pv, pa, labels, 0.95)
    measured = rule_metrics(fusion.decide(pv, pa), labels)
    assert fusion.operating.reached_target
    assert measured["precision"] >= 0.95
    assert measured["precision"] == pytest.approx(fusion.operating.achieved_precision)
    assert measured["recall"] == pytest.approx(fusion.operating.recall)
    scores = fusion.fuse(pv, pa)
    for threshold in np.unique(scores):
        other = rule_metrics(scores >= threshold, labels)
        if other["precision"] >= 0.95:
            assert other["recall"] <= fusion.operating.recall + 1e-12


def test_disagreement_weight_is_positive_when_the_label_is_an_or_of_the_streams():
    pv, pa, labels = _calibrated_streams(seed=9)
    fusion = fit_logistic_fusion(pv, pa, labels, 0.9)
    assert fusion.coefficients[2] > 0.0


def test_logistic_fusion_round_trips_through_its_dict():
    pv, pa, labels = _calibrated_streams(seed=2)
    fusion = fit_logistic_fusion(pv, pa, labels, 0.9)
    assert LogisticFusion.from_dict(fusion.as_dict()) == fusion


def test_attribution_on_constructed_weighted_sum_cases():
    fusion = FusionModel(weight=0.5, operating=_operating(0.4))
    p_video = [0.90, 0.05, 0.90, 0.50, 0.10]
    p_audio = [0.00, 0.95, 0.90, 0.50, 0.10]
    assert attribute(fusion.decide, p_video, p_audio) == [
        "video",
        "audio",
        "either",
        "joint",
        "none",
    ]


def test_attribution_on_constructed_logistic_cases():
    fusion = LogisticFusion(coefficients=(4.0, 4.0, 4.0), intercept=-3.0, operating=_operating(0.5))
    p_video = [0.9, 0.0, 0.9, 0.0]
    p_audio = [0.0, 0.9, 0.9, 0.0]
    assert attribute(fusion.decide, p_video, p_audio) == ["video", "audio", "either", "none"]


def test_pipeline_reports_logistic_fusion_and_attributes_every_test_clip(tmp_path):
    config = replace(profile("tiny"), corpus_dir=FIXTURE_DIR, run_dir=tmp_path / "run")
    result = run_pipeline(config)
    for split in ("seen_test", "unseen_test"):
        assert "logistic" in result.results["metrics"][split]
        assert "logistic" in result.results["rules"][split]
        n_clips = result.results["metrics"][split]["fused"]["n"]
        for detector in ("fused", "logistic"):
            rows = result.results["attribution"][split][detector]
            assert sum(sum(row[a] for a in ATTRIBUTIONS) for row in rows.values()) == n_clips
    assert "logistic" in result.results["calibration"]
