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
from .fusion import attribute
from .models.cnn_lstm import CnnLstmDetector, load_checkpoint
from .pipeline import Fusions, load_calibration

STREAMS = ("video", "audio")


def clip_steps(stream: str, clip: Clip | MediaClip) -> np.ndarray:
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
        self, clips: Sequence[Clip | MediaClip], streams: Sequence[str] = STREAMS
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

    def describe(self, logits: dict[str, np.ndarray]) -> list[dict]:
        """Per clip probabilities, flags, decisions and the stream that triggered them."""
        p_video, p_audio = self.probabilities(logits)
        flags = self.flags(logits)
        weighted, logistic = self.fusions["fused"], self.fusions["logistic"]
        fused = weighted.fuse(p_video, p_audio)
        learned = logistic.fuse(p_video, p_audio)
        triggered = attribute(weighted.decide, p_video, p_audio)
        return [
            {
                "video_logit": float(logits["video"][i]),
                "audio_logit": float(logits["audio"][i]),
                "video_probability": float(p_video[i]),
                "audio_probability": float(p_audio[i]),
                "fused_probability": float(fused[i]),
                "logistic_probability": float(learned[i]),
                "video_flags": bool(flags["video"][i]),
                "audio_flags": bool(flags["audio"][i]),
                "decision": "attack" if flags["fused"][i] else "bonafide",
                "logistic_decision": "attack" if flags["logistic"][i] else "bonafide",
                "triggered_by": triggered[i],
            }
            for i in range(len(p_video))
        ]


SCORE_FIELDS = (
    "clip",
    "video_logit",
    "audio_logit",
    "video_probability",
    "audio_probability",
    "fused_probability",
    "logistic_probability",
    "video_flags",
    "audio_flags",
    "decision",
    "logistic_decision",
    "triggered_by",
)


@dataclass(frozen=True)
class MediaClip:
    """Decoded frames and audio with no label, as a deployment receives them."""

    name: str
    frames: np.ndarray
    audio: np.ndarray
    sample_rate: int


def read_npz_clip(path: Path, sample_rate: int) -> MediaClip:
    """Read a clip npz in the generator's format: uint8 video and int16 audio."""
    with np.load(path) as data:
        frames = data["video"]
        audio = data["audio"].astype(np.float32) / 32767.0
    return MediaClip(name=str(path), frames=frames, audio=audio, sample_rate=sample_rate)


def score_clip_paths(run_dir: Path, paths: Sequence[Path], sample_rate: int) -> list[dict]:
    """Score clip npz files in batches and describe every clip with SCORE_FIELDS."""
    scorer = RunScorer.from_run(run_dir)
    clips = [read_npz_clip(path, sample_rate) for path in paths]
    entries = scorer.describe(scorer.logits(clips))
    return [{"clip": clip.name, **entry} for clip, entry in zip(clips, entries, strict=True)]
