"""Splits and torch datasets for the leave-one-attack-family-out protocol.

Four disjoint splits come out of a corpus:

``train``       seen families only, training identities
``calib``       seen families only, calibration identities, never trained on
``seen_test``   seen families only, test identities
``unseen_test`` every clip carrying a held out family, test identities, plus the
                half of the test bona fide clips not used by ``seen_test``

Identities never appear in more than one pool, so nothing about a speaker leaks
across a split boundary, and no held out family appears in train or calib.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..families import validate_families
from ..seeding import rng as make_rng
from .features import Normalizer, log_mel, mel_patches, video_steps
from .sources import ClipRecord, ClipSource

SPLIT_NAMES = ("train", "calib", "seen_test", "unseen_test")


@dataclass(frozen=True)
class Splits:
    """Clip ids per split, plus the identity pools they came from."""

    train: tuple[str, ...]
    calib: tuple[str, ...]
    seen_test: tuple[str, ...]
    unseen_test: tuple[str, ...]
    identity_pools: dict[str, tuple[str, ...]]
    unseen_families: tuple[str, ...]
    dropped: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.train,
            "calib": self.calib,
            "seen_test": self.seen_test,
            "unseen_test": self.unseen_test,
        }

    def counts(self) -> dict[str, int]:
        """Clips per split, plus the train and calib clips dropped for an unseen family."""
        counts = {name: len(ids) for name, ids in self.as_dict().items()}
        counts["dropped"] = len(self.dropped)
        return counts


def make_splits(
    records: Sequence[ClipRecord],
    unseen_families: Iterable[str],
    seed: int,
    train_fraction: float = 0.6,
    calib_fraction: float = 0.2,
) -> Splits:
    """Assign identities to pools, then clips to splits."""
    unseen = validate_families(tuple(unseen_families))
    identities = sorted({r.identity for r in records})
    order = make_rng(seed, "identity-pools").permutation(len(identities))
    shuffled = [identities[i] for i in order]
    n_train = max(1, round(len(shuffled) * train_fraction))
    n_calib = max(1, round(len(shuffled) * calib_fraction))
    n_train = min(n_train, len(shuffled) - 2)
    n_calib = min(n_calib, len(shuffled) - n_train - 1)
    pools = {
        "train": tuple(sorted(shuffled[:n_train])),
        "calib": tuple(sorted(shuffled[n_train : n_train + n_calib])),
        "test": tuple(sorted(shuffled[n_train + n_calib :])),
    }
    pool_of = {ident: name for name, members in pools.items() for ident in members}

    train: list[str] = []
    calib: list[str] = []
    seen_test: list[str] = []
    unseen_test: list[str] = []
    test_bonafide: list[str] = []
    dropped: list[str] = []

    for record in sorted(records, key=lambda r: r.clip_id):
        pool = pool_of[record.identity]
        touches_unseen = any(f in unseen for f in record.families)
        if pool == "train":
            # A held out family is dropped from the training view rather than
            # relabelled, so it cannot leak into the weights or the calibration.
            (dropped if touches_unseen else train).append(record.clip_id)
        elif pool == "calib":
            (dropped if touches_unseen else calib).append(record.clip_id)
        elif touches_unseen:
            unseen_test.append(record.clip_id)
        elif record.label == 0:
            test_bonafide.append(record.clip_id)
        else:
            seen_test.append(record.clip_id)

    # Split the test bona fide clips between the two test sets so the splits stay
    # disjoint while both can measure precision.
    seen_test.extend(test_bonafide[0::2])
    unseen_test.extend(test_bonafide[1::2])

    return Splits(
        train=tuple(sorted(train)),
        calib=tuple(sorted(calib)),
        seen_test=tuple(sorted(seen_test)),
        unseen_test=tuple(sorted(unseen_test)),
        identity_pools=pools,
        unseen_families=unseen,
        dropped=tuple(sorted(dropped)),
    )


@dataclass
class LoadedCorpus:
    """Whole corpus held in memory as frames and log mel spectrograms."""

    records: list[ClipRecord]
    video: dict[str, np.ndarray]
    mel: dict[str, np.ndarray]
    sample_rate: int

    def record(self, clip_id: str) -> ClipRecord:
        return self._by_id[clip_id]

    def __post_init__(self) -> None:
        self._by_id = {r.clip_id: r for r in self.records}

    def __len__(self) -> int:
        return len(self.records)


def load_corpus(source: ClipSource, progress: Callable[[str], None] | None = None) -> LoadedCorpus:
    """Decode every clip once and keep frames plus log mel in memory."""
    records = source.records
    video: dict[str, np.ndarray] = {}
    mel: dict[str, np.ndarray] = {}
    step = max(1, len(records) // 5)
    for i, record in enumerate(records):
        clip = source.load(record.clip_id)
        video[record.clip_id] = clip.frames
        mel[record.clip_id] = log_mel(clip.audio, clip.sample_rate)
        if progress and ((i + 1) % step == 0 or i + 1 == len(records)):
            progress(f"  loaded {i + 1}/{len(records)} clips")
    return LoadedCorpus(records=list(records), video=video, mel=mel, sample_rate=source.sample_rate)


class StreamDataset(Dataset):
    """A single stream's view of a list of clips."""

    def __init__(self, corpus: LoadedCorpus, clip_ids: Sequence[str], stream: str) -> None:
        if stream not in ("video", "audio"):
            raise ValueError(f"stream must be 'video' or 'audio', got {stream!r}")
        self.corpus = corpus
        self.clip_ids = list(clip_ids)
        self.stream = stream

    def __len__(self) -> int:
        return len(self.clip_ids)

    def steps(self, clip_id: str) -> np.ndarray:
        if self.stream == "video":
            return video_steps(self.corpus.video[clip_id])
        return mel_patches(self.corpus.mel[clip_id])

    def modality_label(self, record: ClipRecord) -> float:
        return float(record.video_attacked if self.stream == "video" else record.audio_attacked)

    def __getitem__(self, index: int) -> dict:
        clip_id = self.clip_ids[index]
        record = self.corpus.record(clip_id)
        x = torch.from_numpy(self.steps(clip_id))
        return {
            "x": x,
            "length": x.shape[0],
            "y": self.modality_label(record),
            "y_clip": float(record.label),
            "clip_id": clip_id,
        }

    def fit_normalizer(self) -> Normalizer:
        return Normalizer.fit([self.steps(cid) for cid in self.clip_ids])


def collate(batch: list[dict]) -> dict:
    """Pad a batch of variable length step sequences."""
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max())
    sample = batch[0]["x"]
    padded = torch.zeros((len(batch), max_len, *sample.shape[1:]), dtype=sample.dtype)
    for i, item in enumerate(batch):
        padded[i, : item["length"]] = item["x"]
    return {
        "x": padded,
        "lengths": lengths,
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.float32),
        "y_clip": torch.tensor([item["y_clip"] for item in batch], dtype=torch.float32),
        "clip_ids": [item["clip_id"] for item in batch],
    }


def save_splits(path: Path, splits: Splits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unseen_families": list(splits.unseen_families),
        "identity_pools": {k: list(v) for k, v in splits.identity_pools.items()},
        "splits": {k: list(v) for k, v in splits.as_dict().items()},
        "dropped": list(splits.dropped),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_splits(path: Path) -> Splits:
    """The splits a finished run wrote, so post-run commands score the same clips."""
    payload = json.loads(Path(path).read_text())
    ids = payload["splits"]
    return Splits(
        train=tuple(ids["train"]),
        calib=tuple(ids["calib"]),
        seen_test=tuple(ids["seen_test"]),
        unseen_test=tuple(ids["unseen_test"]),
        identity_pools={k: tuple(v) for k, v in payload["identity_pools"].items()},
        unseen_families=tuple(payload["unseen_families"]),
        dropped=tuple(payload.get("dropped", ())),
    )
