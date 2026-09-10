"""Clip records, the `ClipSource` protocol and the two concrete sources.

`NpzCorpusSource` reads the corpus this repo generates. `DirectoryClipSource` is
the adapter for a real corpus: point it at a directory laid out as

    root/
      manifest.csv            clip_id,identity,video_family,audio_family,label
      video/<clip_id>.mp4     any container OpenCV can decode
      audio/<clip_id>.wav     mono or stereo, any sample rate

PCM wav is decoded with the standard library, so no extra codec package is
needed. Any other audio extension falls back to ``torchaudio.load``, which on
recent torchaudio releases needs a decoding backend installed separately.

and it will decode, resample and reshape clips into exactly the tensors the two
streams consume. Family names must come from `spoofline.families`, with
``bonafide`` for a stream that was not attacked.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..families import BONAFIDE


@dataclass(frozen=True)
class ClipRecord:
    """One manifest row."""

    clip_id: str
    identity: str
    video_family: str
    audio_family: str
    label: int

    @property
    def video_attacked(self) -> bool:
        return self.video_family != BONAFIDE

    @property
    def audio_attacked(self) -> bool:
        return self.audio_family != BONAFIDE

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(f for f in (self.video_family, self.audio_family) if f != BONAFIDE)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Clip:
    """A decoded clip: frames as uint8 (T, H, W, 3) and audio as float32."""

    record: ClipRecord
    frames: np.ndarray
    audio: np.ndarray
    sample_rate: int


@runtime_checkable
class ClipSource(Protocol):
    """Anything the trainer can read clips from."""

    @property
    def records(self) -> list[ClipRecord]: ...

    @property
    def sample_rate(self) -> int: ...

    def load(self, clip_id: str) -> Clip: ...

    def __len__(self) -> int: ...


def write_manifest(path: Path, records: list[ClipRecord], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "records": [r.as_dict() for r in records]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path) -> tuple[list[ClipRecord], dict]:
    payload = json.loads(path.read_text())
    records = [ClipRecord(**row) for row in payload["records"]]
    return records, payload["meta"]


class NpzCorpusSource:
    """The generated corpus: one compressed npz per clip plus a json manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"no corpus at {self.root} (missing manifest.json); run `spoofline generate` first"
            )
        self._records, self.meta = read_manifest(manifest)
        self._sample_rate = int(self.meta["sample_rate"])

    @property
    def records(self) -> list[ClipRecord]:
        return list(self._records)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __len__(self) -> int:
        return len(self._records)

    def path_for(self, clip_id: str) -> Path:
        return self.root / "clips" / f"{clip_id}.npz"

    def load(self, clip_id: str) -> Clip:
        record = next(r for r in self._records if r.clip_id == clip_id)
        with np.load(self.path_for(clip_id)) as data:
            frames = data["video"]
            audio = data["audio"].astype(np.float32) / 32767.0
        return Clip(record=record, frames=frames, audio=audio, sample_rate=self._sample_rate)


class DirectoryClipSource:
    """Adapter for a real corpus of media files on disk.

    Frames are uniformly sampled to ``n_frames`` and resized to
    ``frame_size``; audio is downmixed to mono, resampled to ``sample_rate`` and
    trimmed or zero padded to ``n_samples``.
    """

    def __init__(
        self,
        root: Path,
        n_frames: int,
        frame_size: int,
        sample_rate: int,
        n_samples: int,
        video_ext: str = ".mp4",
        audio_ext: str = ".wav",
    ) -> None:
        self.root = Path(root)
        self.n_frames = n_frames
        self.frame_size = frame_size
        self._sample_rate = sample_rate
        self.n_samples = n_samples
        self.video_ext = video_ext
        self.audio_ext = audio_ext
        self._records = self._read_csv(self.root / "manifest.csv")

    @staticmethod
    def _read_csv(path: Path) -> list[ClipRecord]:
        if not path.exists():
            raise FileNotFoundError(f"expected a manifest at {path}")
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        required = {"clip_id", "identity", "video_family", "audio_family", "label"}
        for row in rows:
            missing = required - set(row)
            if missing:
                raise ValueError(f"manifest row missing columns {sorted(missing)}: {row}")
        return [
            ClipRecord(
                clip_id=row["clip_id"],
                identity=row["identity"],
                video_family=row["video_family"],
                audio_family=row["audio_family"],
                label=int(row["label"]),
            )
            for row in rows
        ]

    @property
    def records(self) -> list[ClipRecord]:
        return list(self._records)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __len__(self) -> int:
        return len(self._records)

    def _read_video(self, clip_id: str) -> np.ndarray:
        import cv2

        path = self.root / "video" / f"{clip_id}{self.video_ext}"
        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"no decodable frames in {path}")
        idx = np.linspace(0, len(frames) - 1, self.n_frames).round().astype(int)
        size = (self.frame_size, self.frame_size)
        picked = [cv2.resize(frames[i], size, interpolation=cv2.INTER_AREA) for i in idx]
        return np.stack(picked).astype(np.uint8)

    @staticmethod
    def _decode_wav(path: Path) -> tuple[np.ndarray, int]:
        """Read a PCM wav with the standard library, so no codec package is needed."""
        import wave

        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise ValueError(f"unsupported wav sample width {width} bytes in {path}")
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        # 8 bit wav is unsigned and centred on 128, wider widths are signed.
        offset = 128.0 if dtype is np.uint8 else 0.0
        scale = 128.0 if dtype is np.uint8 else float(np.iinfo(dtype).max)
        data = (data - offset) / scale
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, rate

    def _read_audio(self, clip_id: str) -> np.ndarray:
        import torch
        import torchaudio.functional as AF

        path = self.root / "audio" / f"{clip_id}{self.audio_ext}"
        if path.suffix.lower() == ".wav":
            data, sr = self._decode_wav(path)
        else:
            import torchaudio

            wav, sr = torchaudio.load(str(path))
            data = wav.mean(dim=0).numpy().astype(np.float32)
        if sr != self._sample_rate:
            data = AF.resample(torch.from_numpy(data), sr, self._sample_rate).numpy()
        audio = np.asarray(data, dtype=np.float32)
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)))
        return audio[: self.n_samples]

    def load(self, clip_id: str) -> Clip:
        record = next(r for r in self._records if r.clip_id == clip_id)
        return Clip(
            record=record,
            frames=self._read_video(clip_id),
            audio=self._read_audio(clip_id),
            sample_rate=self._sample_rate,
        )
