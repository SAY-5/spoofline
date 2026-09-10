"""Feature extraction shapes and normalisation."""

from __future__ import annotations

import numpy as np
import torch

from spoofline.data.features import (
    N_MELS,
    PATCH_WIDTH,
    Normalizer,
    log_mel,
    mel_patches,
    video_steps,
)


def test_video_steps_are_channel_first_and_scaled(source):
    clip = source.load("clip_00000")
    steps = video_steps(clip.frames)
    assert steps.shape == (clip.frames.shape[0], 3, clip.frames.shape[1], clip.frames.shape[2])
    assert steps.dtype == np.float32
    assert steps.min() >= -1.0 and steps.max() <= 1.0


def test_log_mel_shape_matches_the_hop(source):
    clip = source.load("clip_00000")
    spectrogram = log_mel(clip.audio, clip.sample_rate)
    assert spectrogram.shape[0] == N_MELS
    assert spectrogram.shape[1] > 1


def test_mel_patches_form_a_step_sequence(source):
    clip = source.load("clip_00000")
    patches = mel_patches(log_mel(clip.audio, clip.sample_rate))
    assert patches.ndim == 4
    assert patches.shape[1:] == (1, N_MELS, PATCH_WIDTH)


def test_normalizer_standardises_and_round_trips():
    samples = [np.random.default_rng(0).normal(3.0, 2.0, size=(4, 2, 5, 5)).astype(np.float32)]
    normalizer = Normalizer.fit(samples)
    out = normalizer.apply(torch.from_numpy(samples[0]))
    assert abs(float(out.mean())) < 0.1
    assert abs(float(out.std()) - 1.0) < 0.2
    restored = Normalizer.from_state_dict(normalizer.state_dict())
    assert torch.allclose(restored.mean, normalizer.mean)
    assert torch.allclose(restored.std, normalizer.std)
