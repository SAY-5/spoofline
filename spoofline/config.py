"""Configuration objects and the two run profiles used by the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CorpusConfig:
    """Shape of a generated corpus."""

    name: str = "full"
    n_clips: int = 1600
    n_identities: int = 80
    n_frames: int = 16
    frame_size: int = 64
    sample_rate: int = 16000
    duration_s: float = 2.0

    @property
    def n_samples(self) -> int:
        return round(self.sample_rate * self.duration_s)


@dataclass(frozen=True)
class StreamTrainConfig:
    """Optimisation settings for one stream."""

    epochs: int = 8
    batch_size: int = 32
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    hidden_size: int = 96
    embed_size: int = 96
    dropout: float = 0.1
    val_fraction: float = 0.15


@dataclass(frozen=True)
class SpooflineConfig:
    """Everything a pipeline run needs."""

    profile: str = "full"
    seed: int = 20250117
    threads: int = 8
    target_precision: float = 0.95
    unseen_families: tuple[str, ...] = ("video_splice", "audio_vocoder")
    train_fraction: float = 0.6
    calib_fraction: float = 0.2
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    video: StreamTrainConfig = field(default_factory=lambda: StreamTrainConfig(epochs=10))
    audio: StreamTrainConfig = field(default_factory=lambda: StreamTrainConfig(epochs=8))
    corpus_dir: Path = REPO_ROOT / "data" / "full"
    run_dir: Path = REPO_ROOT / "runs" / "full"

    def with_dirs(self, corpus_dir: Path, run_dir: Path) -> SpooflineConfig:
        return replace(self, corpus_dir=Path(corpus_dir), run_dir=Path(run_dir))


FIXTURE_DIR = REPO_ROOT / "fixtures" / "tiny"

TINY_CORPUS = CorpusConfig(
    name="tiny",
    n_clips=24,
    n_identities=12,
    n_frames=8,
    frame_size=64,
    sample_rate=16000,
    duration_s=1.0,
)

_TINY_TRAIN = StreamTrainConfig(epochs=2, batch_size=4, hidden_size=32, embed_size=32)

REDUCED_CORPUS = CorpusConfig(
    name="reduced",
    n_clips=960,
    n_identities=80,
    n_frames=8,
    frame_size=64,
    sample_rate=16000,
    duration_s=1.0,
)

_REDUCED_VIDEO = StreamTrainConfig(epochs=8, batch_size=16)
_REDUCED_AUDIO = StreamTrainConfig(epochs=6, batch_size=16)


def profile(name: str) -> SpooflineConfig:
    """Return the configuration for a named run profile."""
    if name == "full":
        return SpooflineConfig()
    if name == "tiny":
        return SpooflineConfig(
            profile="tiny",
            threads=4,
            train_fraction=0.5,
            calib_fraction=0.25,
            corpus=TINY_CORPUS,
            video=_TINY_TRAIN,
            audio=_TINY_TRAIN,
            corpus_dir=FIXTURE_DIR,
            run_dir=REPO_ROOT / "runs" / "tiny",
        )
    if name == "reduced":
        return SpooflineConfig(
            profile="reduced",
            threads=2,
            corpus=REDUCED_CORPUS,
            video=_REDUCED_VIDEO,
            audio=_REDUCED_AUDIO,
            corpus_dir=REPO_ROOT / "data" / "reduced",
            run_dir=REPO_ROOT / "runs" / "reduced",
        )
    raise ValueError(f"unknown profile: {name!r} (expected one of {PROFILE_NAMES})")


PROFILE_NAMES = ("full", "reduced", "tiny")
