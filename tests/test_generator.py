"""Generator determinism, corpus balance and manifest integrity."""

from __future__ import annotations

import numpy as np

from spoofline.data.generate import combo_counts, family_counts, plan_corpus, render_clip
from spoofline.families import ALL_FAMILIES, AUDIO_FAMILIES, BONAFIDE, VIDEO_FAMILIES


def test_render_clip_is_deterministic(config):
    plan = plan_corpus(config.corpus, config.seed)[0]
    first = render_clip(plan, config.corpus, config.seed)
    second = render_clip(plan, config.corpus, config.seed)
    assert np.array_equal(first.frames, second.frames)
    assert np.array_equal(first.audio, second.audio)


def test_render_clip_changes_with_seed(config):
    plan = plan_corpus(config.corpus, config.seed)[0]
    first = render_clip(plan, config.corpus, config.seed)
    other = render_clip(plan, config.corpus, config.seed + 1)
    assert not np.array_equal(first.frames, other.frames)
    assert not np.array_equal(first.audio, other.audio)


def test_committed_fixture_matches_a_fresh_render(config, source):
    """The committed corpus is exactly what the generator produces for this seed."""
    plans = {p.clip_id: p for p in plan_corpus(config.corpus, config.seed)}
    for clip_id in ("clip_00000", "clip_00007", "clip_00019"):
        stored = source.load(clip_id)
        fresh = render_clip(plans[clip_id], config.corpus, config.seed)
        assert np.array_equal(stored.frames, fresh.frames)
        assert np.allclose(stored.audio, fresh.audio, atol=1e-4)
        assert stored.record == fresh.record


def test_plan_is_balanced_across_combinations():
    from spoofline.config import CorpusConfig

    corpus = CorpusConfig(n_clips=800, n_identities=20)
    plans = plan_corpus(corpus, 7)
    combos = {
        "bonafide": 0,
        "video_only": 0,
        "audio_only": 0,
        "both": 0,
    }
    for plan in plans:
        video = plan.video_family != BONAFIDE
        audio = plan.audio_family != BONAFIDE
        key = (
            "both"
            if video and audio
            else "video_only"
            if video
            else "audio_only"
            if audio
            else "bonafide"
        )
        combos[key] += 1
    assert set(combos.values()) == {200}


def test_plan_covers_every_family_evenly():
    from spoofline.config import CorpusConfig

    plans = plan_corpus(CorpusConfig(n_clips=800, n_identities=20), 7)
    video = [p.video_family for p in plans if p.video_family != BONAFIDE]
    audio = [p.audio_family for p in plans if p.audio_family != BONAFIDE]
    assert {video.count(f) for f in VIDEO_FAMILIES} == {100}
    assert {audio.count(f) for f in AUDIO_FAMILIES} == {100}


def test_plan_never_uses_the_same_identity_as_donor(config):
    for plan in plan_corpus(config.corpus, config.seed):
        assert plan.donor_index != plan.identity_index


def test_labels_follow_the_either_stream_rule(source):
    for record in source.records:
        expected = int(record.video_family != BONAFIDE or record.audio_family != BONAFIDE)
        assert record.label == expected


def test_fixture_corpus_has_every_combination(source):
    counts = combo_counts(source.records)
    assert set(counts) == {"bonafide", "video_only", "audio_only", "both"}
    assert all(value > 0 for value in counts.values())


def test_fixture_corpus_family_names_are_known(source):
    counts = family_counts(source.records)
    for key in counts:
        stream, family = key.split(":", 1)
        assert stream in ("video", "audio")
        assert family == BONAFIDE or family in ALL_FAMILIES


def test_clip_shapes_match_the_corpus_config(config, source):
    clip = source.load("clip_00000")
    assert clip.frames.shape == (
        config.corpus.n_frames,
        config.corpus.frame_size,
        config.corpus.frame_size,
        3,
    )
    assert clip.frames.dtype == np.uint8
    assert clip.audio.shape == (config.corpus.n_samples,)
    assert np.abs(clip.audio).max() <= 1.0
