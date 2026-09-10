"""Feature extraction for the two streams.

Video frames become a sequence of small RGB tensors, one step per frame. Audio
becomes a log mel spectrogram sliced into fixed width patches along time, one
step per patch, so both streams present the same shape to the shared CNN-LSTM.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as AT

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
PATCH_WIDTH = 20
_MEL_CACHE: dict[int, AT.MelSpectrogram] = {}


def mel_transform(sample_rate: int) -> AT.MelSpectrogram:
    """A cached mel filterbank for a sample rate."""
    if sample_rate not in _MEL_CACHE:
        _MEL_CACHE[sample_rate] = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
    return _MEL_CACHE[sample_rate]


def log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log mel spectrogram of shape (N_MELS, frames)."""
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    mel = mel_transform(sample_rate)(wav)
    return torch.log(mel + 1e-6).numpy().astype(np.float32)


def mel_patches(spectrogram: np.ndarray, patch_width: int = PATCH_WIDTH) -> np.ndarray:
    """Slice a log mel spectrogram into a sequence of (1, N_MELS, patch_width) patches."""
    n_frames = spectrogram.shape[1]
    n_patches = max(1, n_frames // patch_width)
    usable = n_patches * patch_width
    if usable > n_frames:  # pragma: no cover - only for very short clips
        spectrogram = np.pad(spectrogram, ((0, 0), (0, usable - n_frames)))
    trimmed = spectrogram[:, :usable]
    patches = trimmed.reshape(spectrogram.shape[0], n_patches, patch_width)
    return np.ascontiguousarray(patches.transpose(1, 0, 2)[:, None, :, :])


def video_steps(frames: np.ndarray) -> np.ndarray:
    """Frames (T, H, W, 3) uint8 to a sequence of (3, H, W) float32 steps in [-1, 1]."""
    x = frames.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class Normalizer:
    """Per-channel standardisation fitted on the training split."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.float()
        self.std = torch.clamp(std.float(), min=1e-4)

    @classmethod
    def fit(cls, samples: list[np.ndarray]) -> Normalizer:
        stacked = np.concatenate([s.reshape(s.shape[0], s.shape[1], -1) for s in samples], axis=0)
        mean = torch.from_numpy(stacked.mean(axis=(0, 2)))
        std = torch.from_numpy(stacked.std(axis=(0, 2)))
        return cls(mean, std)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a (..., C, H, W) tensor."""
        shape = [1] * (x.dim() - 3) + [-1, 1, 1]
        return (x - self.mean.view(shape)) / self.std.view(shape)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> Normalizer:
        return cls(state["mean"], state["std"])
