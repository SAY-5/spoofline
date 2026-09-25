"""Reading a finished run back: saved splits, its seed, and the headline verdict."""

from __future__ import annotations

from dataclasses import replace

import pytest

from spoofline.config import profile
from spoofline.data.dataset import load_splits, make_splits, save_splits
from spoofline.data.generate import generate_corpus
from spoofline.data.sources import NpzCorpusSource
from spoofline.pipeline import precision_verdict, run_pipeline
from spoofline.robustness import run_robustness


def _splits(source, config):
    return make_splits(
        source.records,
        config.unseen_families,
        config.seed,
        config.train_fraction,
        config.calib_fraction,
    )


def _unseen(video: float, audio: float, fused: float) -> dict[str, dict]:
    def block(precision: float) -> dict:
        return {"precision": precision, "recall": 0.5, "f1": 0.6, "auc": 0.7, "eer": 0.3}

    return {"video": block(video), "audio": block(audio), "fused": block(fused)}


def test_saved_splits_round_trip(tmp_path, source, config):
    splits = _splits(source, config)
    save_splits(tmp_path / "splits.json", splits)
    assert load_splits(tmp_path / "splits.json") == splits


def test_counts_account_for_every_clip_including_the_dropped_ones(source, config):
    splits = _splits(source, config)
    counts = splits.counts()
    assert counts["dropped"] == len(splits.dropped)
    assert sum(counts.values()) == len(source.records)
    by_id = {r.clip_id: r for r in source.records}
    for clip_id in splits.dropped:
        assert set(by_id[clip_id].families) & set(splits.unseen_families)


def test_precision_verdict_names_each_relation_to_the_best_single_stream():
    above = precision_verdict(_unseen(0.90, 0.80, 0.95), "fused")
    matches = precision_verdict(_unseen(0.90, 0.80, 0.90), "fused")
    below = precision_verdict(_unseen(0.90, 0.80, 0.85), "fused")
    assert "is above the best single stream (video 0.900)" in above
    assert "matches the best single stream (video 0.900)" in matches
    assert "is BELOW the best single stream (video 0.900)" in below


def test_a_corpus_of_another_shape_is_refused(tmp_path):
    config = profile("tiny")
    generate_corpus(config.corpus, config.seed, tmp_path / "corpus")
    taller = replace(config.corpus, n_frames=config.corpus.n_frames + 1)
    with pytest.raises(ValueError, match="different shape"):
        generate_corpus(taller, config.seed, tmp_path / "corpus")


def test_robustness_follows_the_seed_of_the_run_not_the_profile(tmp_path):
    seed = profile("tiny").seed + 5
    trained = replace(
        profile("tiny"), seed=seed, corpus_dir=tmp_path / "corpus", run_dir=tmp_path / "run"
    )
    run_pipeline(trained)
    # Post-run commands build their config from the profile, whose seed is the default.
    from_profile = replace(
        profile("tiny"), corpus_dir=tmp_path / "corpus", run_dir=tmp_path / "run"
    )
    result = run_robustness(from_profile, tmp_path / "run")
    assert result.results["seed"] == seed
    kept = load_splits(tmp_path / "run" / "splits.json")
    labels = {r.clip_id: r.label for r in NpzCorpusSource(tmp_path / "corpus").records}
    bonafide = sum(1 for c in kept.seen_test + kept.unseen_test if labels[c] == 0)
    assert sum(result.results["bonafide_clips"].values()) == bonafide
