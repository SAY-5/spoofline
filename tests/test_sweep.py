"""The repeated seed, leave two families out sweep."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spoofline.config import CorpusConfig, StreamTrainConfig, profile
from spoofline.data.dataset import make_splits
from spoofline.data.generate import plan_corpus
from spoofline.data.sources import ClipRecord
from spoofline.families import AUDIO_FAMILIES, VIDEO_FAMILIES
from spoofline.sweep import (
    aggregate,
    bootstrap_interval,
    derived_metrics,
    leave_two_out_pairs,
    run_sweep,
    summarise,
)


def _planned_records(n_clips: int = 640, n_identities: int = 80, seed: int = 7):
    plans = plan_corpus(CorpusConfig(n_clips=n_clips, n_identities=n_identities), seed)
    return [
        ClipRecord(
            clip_id=plan.clip_id,
            identity=f"id_{plan.identity_index:03d}",
            video_family=plan.video_family,
            audio_family=plan.audio_family,
            label=plan.label,
        )
        for plan in plans
    ]


def test_pairs_are_every_video_family_with_every_audio_family():
    pairs = leave_two_out_pairs()
    assert len(pairs) == 16
    assert len(set(pairs)) == 16
    assert {video for video, _ in pairs} == set(VIDEO_FAMILIES)
    assert {audio for _, audio in pairs} == set(AUDIO_FAMILIES)


def test_no_held_out_family_leaks_into_train_or_calib_for_any_pair():
    records = _planned_records()
    by_id = {r.clip_id: r for r in records}
    for pair in leave_two_out_pairs():
        splits = make_splits(records, pair, seed=7)
        for name in ("train", "calib"):
            for clip_id in getattr(splits, name):
                assert not set(by_id[clip_id].families) & set(pair), (pair, name, clip_id)
        attacks = [by_id[c] for c in splits.unseen_test if by_id[c].label == 1]
        assert attacks, pair
        assert all(set(r.families) & set(pair) for r in attacks)
        assert {by_id[c].label for c in splits.calib} == {0, 1}


def test_bootstrap_interval_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(3)
    true_mean, hits, trials = 2.0, 0, 300
    for trial in range(trials):
        sample = rng.normal(true_mean, 1.5, size=60)
        low, high = bootstrap_interval(sample, n_resamples=1000, seed=trial)
        hits += int(low <= true_mean <= high)
    assert 0.90 <= hits / trials <= 0.98


def test_bootstrap_interval_width_matches_the_normal_approximation():
    sample = np.random.default_rng(11).normal(5.0, 2.0, size=400)
    low, high = bootstrap_interval(sample)
    expected = 2 * 1.96 * sample.std(ddof=1) / np.sqrt(len(sample))
    assert high - low == pytest.approx(expected, rel=0.1)
    assert low < sample.mean() < high


def test_bootstrap_interval_of_a_constant_collapses_to_the_constant():
    assert bootstrap_interval([0.75] * 12) == (0.75, 0.75)


def test_summarise_matches_hand_computed_values_and_ignores_nan():
    summary = summarise([0.8, 0.9, float("nan"), 1.0, 0.7])
    assert summary.n == 4
    assert summary.mean == pytest.approx(0.85)
    assert summary.std == pytest.approx(np.sqrt(0.05 / 3))
    assert summary.ci_low <= summary.mean <= summary.ci_high


def _fake_run(video_p, audio_p, fused_p, seen_fused_p):
    def block(precision):
        return {"precision": precision, "recall": 0.5, "f1": 0.5, "eer": 0.2, "auc": 0.9}

    metrics = {
        "seen_test": {"video": block(0.9), "audio": block(0.9), "fused": block(seen_fused_p)},
        "unseen_test": {"video": block(video_p), "audio": block(audio_p), "fused": block(fused_p)},
    }
    return {"metrics": metrics, "derived": derived_metrics(metrics)}


def test_derived_metrics_compare_fused_precision_with_the_best_stream_and_seen_split():
    run = _fake_run(video_p=1.0, audio_p=0.8, fused_p=0.94, seen_fused_p=0.97)
    assert run["derived"]["unseen_precision_gap"] == pytest.approx(-0.06)
    assert run["derived"]["seen_to_unseen_precision_drop"] == pytest.approx(0.03)


def test_aggregate_arithmetic_over_runs():
    runs = [
        _fake_run(1.0, 0.8, 0.90, 0.95),
        _fake_run(0.9, 0.7, 0.80, 0.95),
        _fake_run(0.8, 0.9, 0.85, 0.95),
    ]
    table = aggregate(runs)
    assert table["n_runs"] == 3
    fused = table["metrics"]["unseen_test"]["fused"]["precision"]
    assert fused["mean"] == pytest.approx(0.85)
    assert fused["std"] == pytest.approx(0.05)
    assert fused["n"] == 3
    gap = table["derived"]["unseen_precision_gap"]
    assert gap["mean"] == pytest.approx(np.mean([-0.10, -0.10, -0.05]))
    video_recall = table["metrics"]["seen_test"]["video"]["recall"]
    assert video_recall["mean"] == pytest.approx(0.5) and video_recall["std"] == 0.0


def test_sweep_runs_every_pair_on_the_tiny_profile_and_reuses_cached_scores(tmp_path):
    fast = StreamTrainConfig(epochs=1, batch_size=4, hidden_size=16, embed_size=16)
    config = replace(profile("tiny"), video=fast, audio=fast)
    messages: list[str] = []
    first = run_sweep(config, 1, 1, tmp_path / "corpus", tmp_path / "out", messages.append)
    assert first.results["aggregate"]["n_runs"] == 16
    assert len(first.results["per_pair"]) == 16
    assert (tmp_path / "out" / "sweep.json").exists()
    assert "mean, sample std and bootstrap 95% interval" in first.summary

    messages.clear()
    second = run_sweep(config, 1, 1, tmp_path / "corpus", tmp_path / "out", messages.append)
    assert any("16 runs cached, 0 to train" in message for message in messages)
    assert second.results["aggregate"] == first.results["aggregate"]
