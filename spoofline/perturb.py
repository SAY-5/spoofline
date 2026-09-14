"""Benign degradations of bona fide clips, and the abstain rule on stream disagreement.

A deployed detector sees compression, sensor noise, lighting drift, resampled and
reverberant audio and short dropouts on genuine clips. Each perturbation here is
applied to a bona fide clip only, changes one stream's signal and keeps the clip
label, so any flag it causes is a false alarm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
import torchaudio.functional as AF

from .data.sources import Clip
from .seeding import rng as make_rng


def jpeg(frames: np.ndarray, quality: float, _r: np.random.Generator, _rate: int) -> np.ndarray:
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    out = np.empty_like(frames)
    for i, frame in enumerate(frames):
        ok, encoded = cv2.imencode(".jpg", frame, params)
        if not ok:  # pragma: no cover - OpenCV always encodes uint8 BGR frames
            raise RuntimeError("JPEG encoding failed")
        out[i] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out


def video_noise(frames: np.ndarray, sigma: float, r: np.random.Generator, _rate: int) -> np.ndarray:
    noisy = frames.astype(np.float32) + r.normal(0.0, sigma, frames.shape).astype(np.float32)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def brightness_contrast_drift(
    frames: np.ndarray, strength: float, _r: np.random.Generator, _rate: int
) -> np.ndarray:
    """Contrast falls by ``strength`` and brightness rises by ``60 * strength`` over the clip."""
    ramp = np.linspace(0.0, 1.0, len(frames), dtype=np.float32)[:, None, None, None]
    x = frames.astype(np.float32)
    drifted = (x - 128.0) * (1.0 - strength * ramp) + 128.0 + 60.0 * strength * ramp
    return np.clip(np.rint(drifted), 0, 255).astype(np.uint8)


def audio_noise(audio: np.ndarray, snr_db: float, r: np.random.Generator, _rate: int) -> np.ndarray:
    power = float(np.mean(audio.astype(np.float64) ** 2)) or 1e-12
    sigma = np.sqrt(power / 10.0 ** (snr_db / 10.0))
    return (audio + r.normal(0.0, sigma, audio.shape)).astype(np.float32)


def resample_roundtrip(
    audio: np.ndarray, rate: float, _r: np.random.Generator, sample_rate: int
) -> np.ndarray:
    """Down to ``rate`` Hz and back, which removes everything above the lower Nyquist."""
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    down = AF.resample(wav, sample_rate, int(rate))
    up = AF.resample(down, int(rate), sample_rate).numpy()
    return _fit_length(up, len(audio))


def reverb(audio: np.ndarray, rt60: float, r: np.random.Generator, sample_rate: int) -> np.ndarray:
    """A direct path plus an exponentially decaying noise tail, level matched by RMS.

    The tail falls by 60 dB over ``rt60`` seconds, so a longer ``rt60`` adds more
    reverberant energy on top of the same direct sound.
    """
    length = int(sample_rate * rt60)
    n = np.arange(length, dtype=np.float64)
    tail = r.standard_normal(length) * np.exp(-6.9 * n / length) * 0.05
    tail[0] = 1.0
    wet = np.convolve(audio.astype(np.float64), tail)[: len(audio)]
    scale = np.sqrt(np.mean(audio.astype(np.float64) ** 2) / (np.mean(wet**2) or 1e-12))
    return (wet * scale).astype(np.float32)


def audio_dropout(
    audio: np.ndarray, gaps: float, r: np.random.Generator, sample_rate: int
) -> np.ndarray:
    """Zero ``gaps`` separate 30 ms stretches at random positions."""
    width = int(0.03 * sample_rate)
    out = audio.astype(np.float32, copy=True)
    starts = r.permutation(np.arange(0, len(audio) - width, width))[: int(gaps)]
    for start in starts:
        out[start : start + width] = 0.0
    return out


def video_dropout(
    frames: np.ndarray, dropped: float, r: np.random.Generator, _rate: int
) -> np.ndarray:
    """Replace ``dropped`` frames with the frame before them, as a stalled capture does."""
    out = frames.copy()
    picks = np.sort(r.permutation(np.arange(1, len(frames)))[: int(dropped)])
    for index in picks:
        out[index] = out[index - 1]
    return out


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


Transform = Callable[[np.ndarray, float, np.random.Generator, int], np.ndarray]


@dataclass(frozen=True)
class Perturbation:
    """A benign degradation of one stream, with severities ordered mildest first."""

    name: str
    stream: str
    unit: str
    severities: tuple[float, ...]
    transform: Transform


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("jpeg", "video", "quality", (90, 70, 50, 30, 15), jpeg),
    Perturbation("video_noise", "video", "sigma", (2, 5, 10, 20), video_noise),
    Perturbation(
        "brightness_contrast", "video", "drift", (0.1, 0.2, 0.35, 0.5), brightness_contrast_drift
    ),
    Perturbation("video_dropout", "video", "frames", (1, 2, 4), video_dropout),
    Perturbation("audio_noise", "audio", "snr_db", (40, 30, 20, 10), audio_noise),
    Perturbation("resample", "audio", "hz", (12000, 8000, 6000, 4000), resample_roundtrip),
    Perturbation("reverb", "audio", "rt60_s", (0.1, 0.2, 0.35), reverb),
    Perturbation("audio_dropout", "audio", "gaps", (1, 3, 6), audio_dropout),
)


def perturb_clip(clip: Clip, perturbation: Perturbation, severity: float, seed: int) -> Clip:
    """Apply one perturbation to a bona fide clip; the record and its label are untouched."""
    if clip.record.label != 0:
        raise ValueError(
            f"perturbations apply to bona fide clips only, {clip.record.clip_id} is not"
        )
    r = make_rng(seed, f"perturb/{perturbation.name}/{clip.record.clip_id}")
    if perturbation.stream == "video":
        frames = perturbation.transform(clip.frames, severity, r, clip.sample_rate)
        return replace(clip, frames=frames)
    audio = perturbation.transform(clip.audio, severity, r, clip.sample_rate)
    return replace(clip, audio=audio)
