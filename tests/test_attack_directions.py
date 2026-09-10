"""Each attack family must move the signal in the direction it claims to.

These are the tests that keep the generator honest: an attack is a real signal
transformation, so it has to leave a measurable fingerprint.
"""

from __future__ import annotations

import cv2
import numpy as np

from spoofline.data import attacks_audio as aa
from spoofline.data import attacks_video as av
from spoofline.data import signal_stats as st

SR = 16000
SEEDS = range(5)


def test_video_replay_adds_scrolling_refresh_banding(bona_clip, attack_rng):
    _, frames = bona_clip
    base = st.row_profile_temporal(frames)
    for seed in SEEDS:
        attacked = av.video_replay(frames, attack_rng(f"replay/{seed}"))
        assert st.row_profile_temporal(attacked) > 1.5 * base


def test_video_replay_shifts_gamma(bona_clip, attack_rng):
    _, frames = bona_clip
    attacked = av.video_replay(frames, attack_rng("replay/gamma"))
    assert not np.array_equal(attacked, frames)
    assert attacked.shape == frames.shape


def test_video_print_raises_local_contrast(bona_clip, attack_rng):
    _, frames = bona_clip
    base = st.spatial_gradient_energy(frames)
    for seed in SEEDS:
        attacked = av.video_print(frames, attack_rng(f"print/{seed}"))
        assert st.spatial_gradient_energy(attacked) > 3.0 * base


def test_video_print_flattens_motion(bona_clip, attack_rng):
    _, frames = bona_clip
    base = st.temporal_energy(frames)
    for seed in SEEDS:
        attacked = av.video_print(frames, attack_rng(f"print/{seed}"))
        assert st.temporal_energy(attacked) < base


def test_video_splice_changes_only_the_face_region(bona_clip, donor_clip, attack_rng):
    _, frames = bona_clip
    _, donor = donor_clip
    size = frames.shape[1]
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (size // 2, int(size * 0.52)),
        (int(size * 0.24), int(size * 0.32)),
        0,
        0,
        360,
        1,
        -1,
    )
    attacked = av.video_splice(frames, donor, attack_rng("splice"))
    inside, outside = st.region_change(attacked, frames, mask)
    assert inside > 10.0 * max(outside, 1e-3)


def test_video_recompress_raises_blockiness(bona_clip, attack_rng):
    _, frames = bona_clip
    base = st.blockiness(frames)
    for seed in SEEDS:
        attacked = av.video_recompress(frames, attack_rng(f"recompress/{seed}"))
        assert st.blockiness(attacked) > 1.5 * base


def test_audio_replay_loses_high_frequency_energy(bona_clip, attack_rng):
    audio, _ = bona_clip
    base = st.hf_energy_ratio(audio, SR)
    for seed in SEEDS:
        attacked = aa.audio_replay(audio, SR, attack_rng(f"areplay/{seed}"))
        assert st.hf_energy_ratio(attacked, SR) < base


def test_audio_vocoder_destroys_phase_but_keeps_the_envelope(bona_clip, attack_rng):
    audio, _ = bona_clip
    for seed in SEEDS:
        attacked = aa.audio_vocoder(audio, SR, attack_rng(f"voc/{seed}"))
        assert abs(st.waveform_correlation(audio, attacked)) < 0.5
        assert st.mel_cosine(audio, attacked, SR) > 0.9


def test_audio_conversion_shifts_pitch(bona_clip, attack_rng):
    audio, _ = bona_clip
    base = st.f0_estimate(audio, SR)
    for seed in SEEDS:
        attacked = aa.audio_conversion(audio, SR, attack_rng(f"conv/{seed}"))
        ratio = st.f0_estimate(attacked, SR) / base
        assert abs(ratio - 1.0) > 0.04


def test_audio_conversion_preserves_length(bona_clip, attack_rng):
    audio, _ = bona_clip
    attacked = aa.audio_conversion(audio, SR, attack_rng("conv/len"))
    assert len(attacked) == len(audio)


def test_audio_splice_adds_discontinuities(bona_clip, donor_clip, attack_rng):
    audio, _ = bona_clip
    donor, _ = donor_clip
    base = st.max_normalised_jump(audio)
    for seed in SEEDS:
        attacked = aa.audio_splice(audio, donor, SR, attack_rng(f"asplice/{seed}"))
        assert st.max_normalised_jump(attacked) > 1.15 * base


def test_attacks_preserve_shape_and_range(bona_clip, donor_clip, attack_rng):
    audio, frames = bona_clip
    donor_audio, donor_frames = donor_clip
    outputs = [
        av.video_replay(frames, attack_rng("s1")),
        av.video_print(frames, attack_rng("s2")),
        av.video_splice(frames, donor_frames, attack_rng("s3")),
        av.video_recompress(frames, attack_rng("s4")),
    ]
    for out in outputs:
        assert out.shape == frames.shape
        assert out.dtype == np.uint8
    waves = [
        aa.audio_replay(audio, SR, attack_rng("s5")),
        aa.audio_vocoder(audio, SR, attack_rng("s6")),
        aa.audio_conversion(audio, SR, attack_rng("s7")),
        aa.audio_splice(audio, donor_audio, SR, attack_rng("s8")),
    ]
    for wave in waves:
        assert wave.shape == audio.shape
        assert wave.dtype == np.float32
        assert np.abs(wave).max() <= 1.0


def test_attacks_are_deterministic_for_a_seed(bona_clip, attack_rng):
    audio, frames = bona_clip
    assert np.array_equal(
        av.video_print(frames, attack_rng("same")), av.video_print(frames, attack_rng("same"))
    )
    assert np.array_equal(
        aa.audio_replay(audio, SR, attack_rng("same")),
        aa.audio_replay(audio, SR, attack_rng("same")),
    )
