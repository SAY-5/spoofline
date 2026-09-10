"""Manifest handling and the real corpus adapter."""

from __future__ import annotations

import csv
import wave

import cv2
import numpy as np
import pytest

from spoofline.data.sources import (
    ClipRecord,
    ClipSource,
    DirectoryClipSource,
    NpzCorpusSource,
    read_manifest,
    write_manifest,
)


def test_npz_source_satisfies_the_protocol(source):
    assert isinstance(source, ClipSource)
    assert len(source) == len(source.records)


def test_missing_corpus_raises_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="spoofline generate"):
        NpzCorpusSource(tmp_path / "nothing")


def test_manifest_round_trip(tmp_path):
    records = [ClipRecord("clip_1", "id_1", "video_replay", "bonafide", 1)]
    path = tmp_path / "manifest.json"
    write_manifest(path, records, {"sample_rate": 16000})
    restored, meta = read_manifest(path)
    assert restored == records
    assert meta["sample_rate"] == 16000


def test_record_helpers():
    record = ClipRecord("clip_1", "id_1", "video_replay", "bonafide", 1)
    assert record.video_attacked
    assert not record.audio_attacked
    assert record.families == ("video_replay",)


def test_directory_source_reads_media_files(tmp_path, source):
    """Write one clip out as media files and read it back through the adapter."""
    clip = source.load("clip_00000")
    (tmp_path / "video").mkdir()
    (tmp_path / "audio").mkdir()

    writer = cv2.VideoWriter(
        str(tmp_path / "video" / "clip_00000.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (clip.frames.shape[2], clip.frames.shape[1]),
    )
    if not writer.isOpened():  # pragma: no cover - depends on the local codec set
        pytest.skip("no MJPG writer available")
    for frame in clip.frames:
        writer.write(frame)
    writer.release()

    pcm = np.clip(clip.audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(tmp_path / "audio" / "clip_00000.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(clip.sample_rate)
        handle.writeframes(pcm.tobytes())

    with (tmp_path / "manifest.csv").open("w", newline="") as fh:
        writer_csv = csv.DictWriter(
            fh, fieldnames=["clip_id", "identity", "video_family", "audio_family", "label"]
        )
        writer_csv.writeheader()
        writer_csv.writerow(clip.record.as_dict())

    adapter = DirectoryClipSource(
        tmp_path,
        n_frames=clip.frames.shape[0],
        frame_size=clip.frames.shape[1],
        sample_rate=clip.sample_rate,
        n_samples=len(clip.audio),
        video_ext=".avi",
    )
    assert len(adapter) == 1
    loaded = adapter.load("clip_00000")
    assert loaded.frames.shape == clip.frames.shape
    assert loaded.audio.shape == clip.audio.shape
    assert np.corrcoef(loaded.audio, clip.audio)[0, 1] > 0.99


def test_directory_source_requires_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        DirectoryClipSource(tmp_path, 8, 64, 16000, 16000)
