"""Cheap signal statistics used to characterise attack fingerprints.

These are the measurements the attack direction tests in
``tests/test_attack_directions.py`` assert on: each family has to move at least
one of them in the direction it claims to.
"""

from __future__ import annotations

import numpy as np


def _luma(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32).mean(axis=-1)


def spatial_gradient_energy(frames: np.ndarray) -> float:
    """Mean absolute spatial gradient; halftone dithering raises it sharply."""
    y = _luma(frames)
    return float(np.abs(np.diff(y, axis=-1)).mean() + np.abs(np.diff(y, axis=-2)).mean())


def temporal_energy(frames: np.ndarray) -> float:
    """Mean absolute frame to frame difference; a flat print collapses it."""
    return float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean())


def row_profile_temporal(frames: np.ndarray) -> float:
    """Frame to frame change of the row luminance profile; scrolling refresh banding raises it."""
    profile = _luma(frames).mean(axis=-1)
    return float(np.abs(np.diff(profile, axis=0)).mean())


def blockiness(frames: np.ndarray, block: int = 8) -> float:
    """Ratio of gradient energy on block boundaries to gradient energy inside blocks."""
    y = _luma(frames)
    diff = np.abs(np.diff(y, axis=-1))
    cols = np.arange(diff.shape[-1])
    edge = diff[..., cols % block == block - 1].mean()
    inner = diff[..., cols % block != block - 1].mean()
    return float(edge / (inner + 1e-6))


def region_change(frames: np.ndarray, other: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Mean absolute change inside and outside a spatial mask."""
    delta = np.abs(frames.astype(np.float32) - other.astype(np.float32)).mean(axis=(0, 3))
    return float(delta[mask > 0].mean()), float(delta[mask == 0].mean())


def hf_energy_ratio(audio: np.ndarray, sample_rate: int, cutoff: float = 6000.0) -> float:
    """Fraction of magnitude spectrum above a cutoff; band limited replay loses it."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    return float(spectrum[freqs > cutoff].sum() / (spectrum.sum() + 1e-9))


def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    return float((spectrum * freqs).sum() / (spectrum.sum() + 1e-9))


def f0_estimate(audio: np.ndarray, sample_rate: int, low: float = 70.0, high: float = 320.0):
    """Autocorrelation pitch estimate, used to verify a conversion attack shifted pitch."""
    x = audio - audio.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo_lag, hi_lag = int(sample_rate / high), int(sample_rate / low)
    lag = int(np.argmax(ac[lo_lag:hi_lag])) + lo_lag
    return float(sample_rate / lag)


def waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def mel_cosine(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    """Cosine similarity of log mel spectra, used to show a vocoder kept the envelope."""
    import torch
    import torchaudio.transforms as AT

    mel = AT.MelSpectrogram(sample_rate, n_fft=512, hop_length=128, n_mels=64)
    n = min(len(a), len(b))
    fa = torch.log1p(mel(torch.from_numpy(np.ascontiguousarray(a[:n])))).flatten()
    fb = torch.log1p(mel(torch.from_numpy(np.ascontiguousarray(b[:n])))).flatten()
    return float(torch.nn.functional.cosine_similarity(fa, fb, dim=0))


def max_normalised_jump(audio: np.ndarray, frame: int = 160) -> float:
    """Largest sample step relative to the local RMS.

    Band limited speech has a bounded slope, so a hard splice join between two
    recordings with different level and phase stands out here.
    """
    x = audio.astype(np.float64)
    steps = np.abs(np.diff(x))
    local = np.sqrt(np.convolve(x**2, np.ones(frame) / frame, mode="same"))[1:]
    global_rms = float(np.sqrt((x**2).mean()))
    keep = local > 0.25 * global_rms
    if not np.any(keep):  # pragma: no cover - silent input only
        return 0.0
    return float(np.max(steps[keep] / local[keep]))
