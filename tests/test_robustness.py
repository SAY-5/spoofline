"""Benign perturbations of bona fide clips and the abstain rule."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from spoofline.config import FIXTURE_DIR, profile
from spoofline.data.sources import NpzCorpusSource
from spoofline.perturb import PERTURBATIONS, perturb_clip
from spoofline.pipeline import run_pipeline
from spoofline.robustness import abstain_curve, abstain_mask, run_robustness

CASES = [(p, s) for p in PERTURBATIONS for s in p.severities]


@pytest.fixture(scope="module")
def bonafide_clips():
    source = NpzCorpusSource(FIXTURE_DIR)
    return [source.load(r.clip_id) for r in source.records if r.label == 0]


def _stream(clip, stream):
    return clip.frames if stream == "video" else clip.audio


@pytest.mark.parametrize(
    ("perturbation", "severity"), CASES, ids=lambda v: str(getattr(v, "name", v))
)
def test_perturbation_changes_its_stream_and_keeps_the_label(
    bonafide_clips, perturbation, severity
):
    other = "audio" if perturbation.stream == "video" else "video"
    for clip in bonafide_clips:
        out = perturb_clip(clip, perturbation, severity, seed=5)
        changed, original = _stream(out, perturbation.stream), _stream(clip, perturbation.stream)
        assert changed.shape == original.shape and changed.dtype == original.dtype
        assert not np.array_equal(changed, original)
        assert np.array_equal(_stream(out, other), _stream(clip, other))
        assert out.record == clip.record and out.record.label == 0


def test_heavier_severities_change_the_signal_more(bonafide_clips):
    for perturbation in PERTURBATIONS:
        for clip in bonafide_clips:
            original = _stream(clip, perturbation.stream).astype(np.float64)
            change = [
                np.abs(
                    _stream(perturb_clip(clip, perturbation, s, seed=5), perturbation.stream)
                    - original
                ).mean()
                for s in perturbation.severities
            ]
            assert all(a <= b + 1e-12 for a, b in pairwise(change)), (
                perturbation.name,
                change,
            )


def test_perturbations_are_deterministic_for_a_seed(bonafide_clips):
    clip = bonafide_clips[0]
    for perturbation in PERTURBATIONS:
        severity = perturbation.severities[-1]
        first = perturb_clip(clip, perturbation, severity, seed=9)
        second = perturb_clip(clip, perturbation, severity, seed=9)
        assert np.array_equal(
            _stream(first, perturbation.stream), _stream(second, perturbation.stream)
        )


def test_perturbing_an_attacked_clip_is_refused():
    source = NpzCorpusSource(FIXTURE_DIR)
    attacked = next(r for r in source.records if r.label == 1)
    with pytest.raises(ValueError, match="bona fide clips only"):
        perturb_clip(source.load(attacked.clip_id), PERTURBATIONS[0], 50, seed=1)


def test_abstain_coverage_arithmetic():
    p_video = [0.9, 0.1, 0.8, 0.2, 0.5]
    p_audio = [0.1, 0.9, 0.7, 0.3, 0.5]
    decisions = [True, True, True, False, False]
    labels = [1, 1, 1, 0, 1]
    assert abstain_mask(p_video, p_audio, 0.5).tolist() == [True, True, False, False, False]
    loose, strict = abstain_curve(p_video, p_audio, decisions, labels, margins=(1.0, 0.5))
    assert loose["coverage"] == 1.0 and loose["kept"] == 5
    assert loose["precision"] == 1.0 and loose["recall"] == pytest.approx(0.75)
    assert strict["coverage"] == pytest.approx(0.6) and strict["kept"] == 3
    assert strict["abstained_attacks"] == 2 and strict["abstained_bonafide"] == 0
    assert strict["precision"] == 1.0 and strict["recall"] == pytest.approx(0.5)


def test_robustness_report_on_a_tiny_run(tmp_path):
    config = replace(profile("tiny"), corpus_dir=FIXTURE_DIR, run_dir=tmp_path / "run")
    run_pipeline(config)
    result = run_robustness(config, tmp_path / "run")
    rows = result.results["false_alarms"]
    assert len(rows) == 1 + sum(len(p.severities) for p in PERTURBATIONS)
    n_bonafide = sum(result.results["bonafide_clips"].values())
    for row in rows:
        assert row["n"] == n_bonafide
        for detector in ("video", "audio", "fused", "logistic"):
            assert 0.0 <= row[detector] <= 1.0
    assert set(result.results["abstain"]) == {"seen_test", "unseen_test"}
    assert (tmp_path / "run" / "robustness.json").exists()
    assert "false alarm rate" in result.summary
