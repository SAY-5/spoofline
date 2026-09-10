"""Audio attack families.

Each function takes a float32 waveform in [-1, 1] and returns a waveform of the
same length. torchaudio provides the mel filterbank, Griffin-Lim, resampling and
the phase vocoder, so the vocoder and conversion artefacts are the real thing.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT

_N_FFT = 512
_HOP = 128
_N_MELS = 80

# Small integer pitch ratios keep torchaudio's polyphase resampling kernel tiny.
_SHIFT_RATIOS: tuple[tuple[int, int], ...] = (
    (9, 8),
    (8, 9),
    (7, 6),
    (6, 7),
    (10, 9),
    (9, 10),
    (11, 10),
    (10, 11),
    (6, 5),
    (5, 6),
)


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = int(2 ** np.ceil(np.log2(len(x) + len(h) - 1)))
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)
    return y[: len(x)].astype(np.float32)


def _band_filter(x: np.ndarray, sample_rate: int, low: float, high: float, rolloff: float = 0.35):
    """Zero phase band limiting with smooth skirts."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    gain = np.ones_like(freqs)
    with np.errstate(divide="ignore"):
        gain *= 1.0 / (1.0 + (np.maximum(freqs, 1e-6) / high) ** (2 * (1.0 / rolloff)))
        gain *= 1.0 / (1.0 + (low / np.maximum(freqs, 1e-6)) ** (2 * (1.0 / rolloff)))
    return np.fft.irfft(spec * gain, len(x)).astype(np.float32)


def _normalise(x: np.ndarray, peak: float = 0.72) -> np.ndarray:
    top = float(np.max(np.abs(x))) or 1.0
    return (x / top * peak).astype(np.float32)


def _room_impulse(r: np.random.Generator, sample_rate: int) -> np.ndarray:
    length = int(sample_rate * float(r.uniform(0.09, 0.18)))
    n = np.arange(length, dtype=np.float32)
    decay = np.exp(-n / (sample_rate * float(r.uniform(0.020, 0.045))))
    h = r.standard_normal(length).astype(np.float32) * decay * 0.35
    h[0] = 1.0
    for _ in range(int(r.integers(3, 7))):
        idx = int(r.integers(int(sample_rate * 0.002), max(2, length - 1)))
        h[idx] += float(r.uniform(0.18, 0.55)) * float(np.exp(-idx / (sample_rate * 0.03)))
    return h


def audio_replay(x: np.ndarray, sample_rate: int, r: np.random.Generator) -> np.ndarray:
    """Loudspeaker plus microphone in a room: RIR, band limiting, device noise floor."""
    y = _fft_convolve(x, _room_impulse(r, sample_rate))
    low = float(r.uniform(180.0, 320.0))
    high = float(r.uniform(4200.0, 6200.0))
    y = _band_filter(y, sample_rate, low, high)
    y = np.tanh(y * float(r.uniform(1.1, 1.6))).astype(np.float32)

    n = len(y)
    t = np.arange(n, dtype=np.float32) / sample_rate
    hum = np.zeros(n, dtype=np.float32)
    for k in (1, 2, 3):
        hum += (0.4 / k) * np.sin(2 * np.pi * 50.0 * k * t + float(r.uniform(0, 6.28)))
    noise = r.standard_normal(n).astype(np.float32)
    y = y + float(r.uniform(0.004, 0.012)) * noise + float(r.uniform(0.002, 0.006)) * hum
    return _normalise(y)


def _mel_transform(sample_rate: int) -> tuple[torch.Tensor, AT.Spectrogram]:
    fb = AF.melscale_fbanks(
        n_freqs=_N_FFT // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=_N_MELS,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )
    spec = AT.Spectrogram(n_fft=_N_FFT, hop_length=_HOP, power=1.0)
    return fb, spec


def audio_vocoder(x: np.ndarray, sample_rate: int, r: np.random.Generator) -> np.ndarray:
    """Mel analysis, pseudo-inverse back to linear magnitude, Griffin-Lim phase resynthesis."""
    wav = torch.from_numpy(np.ascontiguousarray(x))
    fb, spec_fn = _mel_transform(sample_rate)
    mag = spec_fn(wav)
    mel = fb.T @ mag
    recovered = torch.clamp(torch.linalg.pinv(fb.T) @ mel, min=0.0)
    griffin = AT.GriffinLim(
        n_fft=_N_FFT,
        hop_length=_HOP,
        power=1.0,
        n_iter=int(r.integers(18, 34)),
        rand_init=False,
    )
    y = griffin(recovered).numpy()
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return _normalise(y[: len(x)])


def audio_conversion(x: np.ndarray, sample_rate: int, r: np.random.Generator) -> np.ndarray:
    """Pitch and formant shift by resampling, with a phase vocoder stretch back to length."""
    num, den = _SHIFT_RATIOS[int(r.integers(0, len(_SHIFT_RATIOS)))]
    shift = num / den
    wav = torch.from_numpy(np.ascontiguousarray(x))
    window = torch.hann_window(_N_FFT)
    spec = torch.stft(wav, _N_FFT, _HOP, window=window, return_complex=True)
    phase_advance = torch.linspace(0, np.pi * _HOP, spec.shape[0])[..., None]
    # Stretch by `shift` with the phase vocoder, then decimate by the same ratio:
    # the length comes back to N and every harmonic and formant moves by `shift`.
    # Passing the ratio itself to resample keeps the polyphase kernel small.
    stretched = AF.phase_vocoder(spec, 1.0 / shift, phase_advance)
    stretched_wav = torch.istft(stretched, _N_FFT, _HOP, window=window)
    shifted = AF.resample(stretched_wav, num, den).numpy()
    if len(shifted) < len(x):
        shifted = np.pad(shifted, (0, len(x) - len(shifted)))
    return _normalise(shifted[: len(x)])


def audio_splice(
    x: np.ndarray, donor: np.ndarray, sample_rate: int, r: np.random.Generator
) -> np.ndarray:
    """Concatenate segments from two utterances with hard joins and level mismatch."""
    n = len(x)
    n_segments = int(r.integers(4, 7))
    # Cut inside voiced material rather than in the pauses: a join in silence is
    # inaudible, and real splices are made mid utterance.
    frame = 160
    usable = n // frame * frame
    energy = np.sqrt((x[:usable].reshape(-1, frame) ** 2).mean(axis=1))
    loud = np.flatnonzero(energy > np.median(energy))
    loud = loud[(loud * frame > 0.1 * n) & (loud * frame < 0.9 * n)]
    candidates = loud * frame if len(loud) >= n_segments else np.arange(int(0.1 * n), int(0.9 * n))
    cuts = np.sort(r.choice(candidates, size=n_segments - 1, replace=False))
    bounds = [0, *cuts.tolist(), n]
    donor = donor[:n] if len(donor) >= n else np.pad(donor, (0, n - len(donor)))
    out = np.array(x, dtype=np.float32, copy=True)
    for k in range(len(bounds) - 1):
        if k % 2 == 0:
            continue
        a, b = bounds[k], bounds[k + 1]
        gain = float(r.uniform(1.45, 2.2)) if r.random() < 0.5 else float(r.uniform(0.28, 0.52))
        offset = int(r.integers(0, max(1, n - (b - a))))
        out[a:b] = donor[offset : offset + (b - a)] * gain
    return _normalise(out)
