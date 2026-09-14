"""Scoring clips with a finished run: both checkpoints, their calibration and both fusions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .calibrate import StreamCalibration
from .data.features import Normalizer, log_mel, mel_patches, video_steps
from .data.sources import Clip
from .models.cnn_lstm import CnnLstmDetector, load_checkpoint
from .pipeline import Fusions, load_calibration

STREAMS = ("video", "audio")


def clip_steps(stream: str, clip: Clip) -> np.ndarray:
    """The step sequence one stream consumes for a clip."""
    if stream == "video":
        return video_steps(clip.frames)
    return mel_patches(log_mel(clip.audio, clip.sample_rate))


def batch_logits(
    model: CnnLstmDetector,
    normalizer: Normalizer,
    steps: Sequence[np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    """Raw logits for step sequences, padded and packed in batches."""
    logits: list[float] = []
    with torch.no_grad():
        for start in range(0, len(steps), batch_size):
            chunk = steps[start : start + batch_size]
            lengths = torch.tensor([len(item) for item in chunk], dtype=torch.long)
            padded = torch.zeros((len(chunk), int(lengths.max()), *chunk[0].shape[1:]))
            for i, item in enumerate(chunk):
                padded[i, : len(item)] = torch.from_numpy(item)
            logits.extend(model(normalizer.apply(padded), lengths).tolist())
    return np.asarray(logits, dtype=np.float64)


@dataclass
class RunScorer:
    """Everything a finished run needs to score new clips."""

    models: dict[str, tuple[CnnLstmDetector, Normalizer]]
    calibrations: dict[str, StreamCalibration]
    fusions: Fusions

    @classmethod
    def from_run(cls, run_dir: Path) -> RunScorer:
        models = {}
        for stream in STREAMS:
            model, normalizer, _ = load_checkpoint(Path(run_dir) / f"{stream}.pt")
            models[stream] = (model, normalizer)
        calibrations, fusions = load_calibration(Path(run_dir))
        return cls(models, calibrations, fusions)

    def logits(
        self, clips: Sequence[Clip], streams: Sequence[str] = STREAMS
    ) -> dict[str, np.ndarray]:
        return {
            stream: batch_logits(*self.models[stream], [clip_steps(stream, c) for c in clips])
            for stream in streams
        }

    def probabilities(self, logits: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.calibrations["video"].probabilities(logits["video"]),
            self.calibrations["audio"].probabilities(logits["audio"]),
        )

    def flags(self, logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Decision of each single stream and each fusion at its calibrated threshold."""
        p_video, p_audio = self.probabilities(logits)
        flags = {
            stream: probabilities >= self.calibrations[stream].operating.threshold
            for stream, probabilities in (("video", p_video), ("audio", p_audio))
        }
        for name, model in self.fusions.items():
            flags[name] = np.asarray(model.decide(p_video, p_audio), dtype=bool)
        return flags
