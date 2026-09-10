"""Fusion must never be worse than the best single stream on the calibration set."""

from __future__ import annotations

import numpy as np

from spoofline.calibrate import calibrate_stream, threshold_at_precision
from spoofline.fusion import and_rule, fit_fusion, or_rule, rule_metrics


def _two_streams(n: int = 400, seed: int = 0):
    """Video catches half the attacks, audio catches the other half."""
    rng = np.random.default_rng(seed)
    video_attacked = rng.integers(0, 2, n)
    audio_attacked = rng.integers(0, 2, n)
    labels = ((video_attacked + audio_attacked) > 0).astype(int)
    video_scores = np.where(video_attacked == 1, rng.normal(2.5, 1.0, n), rng.normal(-2.5, 1.0, n))
    audio_scores = np.where(audio_attacked == 1, rng.normal(2.5, 1.0, n), rng.normal(-2.5, 1.0, n))
    return video_scores, audio_scores, labels, video_attacked, audio_attacked


def test_fusion_ties_or_beats_the_best_stream_on_the_calibration_set():
    video_scores, audio_scores, labels, video_attacked, audio_attacked = _two_streams()
    video = calibrate_stream("video", video_scores, video_attacked, labels, 0.9)
    audio = calibrate_stream("audio", audio_scores, audio_attacked, labels, 0.9)
    pv = video.probabilities(video_scores)
    pa = audio.probabilities(audio_scores)
    fusion = fit_fusion(pv, pa, labels, 0.9)
    best_single = max(video.operating.recall, audio.operating.recall)
    assert fusion.operating.recall >= best_single - 1e-12
    assert fusion.operating.achieved_precision >= 0.9


def test_fusion_weight_stays_in_range():
    video_scores, audio_scores, labels, _, _ = _two_streams(seed=2)
    fusion = fit_fusion(video_scores, audio_scores, labels, 0.85, grid=21)
    assert 0.0 <= fusion.weight <= 1.0


def test_fused_score_is_a_convex_combination():
    fusion = fit_fusion([0.9, 0.1], [0.1, 0.9], [1, 1], 0.5, grid=3)
    fused = fusion.fuse([1.0, 0.0], [0.0, 1.0])
    assert np.all(fused >= 0.0) and np.all(fused <= 1.0)


def test_or_rule_flags_at_least_as_much_as_and_rule():
    video_scores, audio_scores, labels, video_attacked, audio_attacked = _two_streams(seed=5)
    video = calibrate_stream("video", video_scores, video_attacked, labels, 0.9)
    audio = calibrate_stream("audio", audio_scores, audio_attacked, labels, 0.9)
    pv, pa = video.probabilities(video_scores), audio.probabilities(audio_scores)
    both = and_rule(video, audio, pv, pa)
    either = or_rule(video, audio, pv, pa)
    assert int(either.sum()) >= int(both.sum())
    assert np.all(both <= either)


def test_or_rule_recovers_the_attacks_a_single_stream_cannot_see():
    video_scores, audio_scores, labels, video_attacked, audio_attacked = _two_streams(seed=7)
    video = calibrate_stream("video", video_scores, video_attacked, labels, 0.9)
    audio = calibrate_stream("audio", audio_scores, audio_attacked, labels, 0.9)
    pv, pa = video.probabilities(video_scores), audio.probabilities(audio_scores)
    single = rule_metrics(video.decide(video_scores), labels)
    either = rule_metrics(or_rule(video, audio, pv, pa), labels)
    assert either["recall"] > single["recall"]


def test_rule_metrics_match_hand_computation():
    decisions = np.array([True, True, False, False])
    labels = [1, 0, 1, 0]
    metrics = rule_metrics(decisions, labels)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1


def test_fusion_grid_includes_the_single_stream_endpoints():
    video_scores, audio_scores, labels, _, _ = _two_streams(seed=11)
    pv = np.asarray(video_scores)
    pa = np.asarray(audio_scores)
    fusion = fit_fusion(pv, pa, labels, 0.9)
    video_only = threshold_at_precision(pv, labels, 0.9)
    audio_only = threshold_at_precision(pa, labels, 0.9)
    assert fusion.operating.recall >= max(video_only.recall, audio_only.recall) - 1e-12
