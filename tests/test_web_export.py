"""The browser demo's export script must keep working against the package it imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spoofline import export as package_export
from spoofline import scoring as package_scoring
from spoofline.data.sources import ClipRecord

SCRIPT = Path(__file__).resolve().parents[1] / "web" / "scripts" / "export.py"


def _record(video_family: str, audio_family: str) -> ClipRecord:
    attacked = video_family != "bonafide" or audio_family != "bonafide"
    return ClipRecord(
        clip_id="clip_00000",
        identity="id_000",
        video_family=video_family,
        audio_family=audio_family,
        label=int(attacked),
    )


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("spoofline_web_export", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_imports_against_the_current_package(script):
    assert callable(script.main)


def test_script_uses_the_package_export_and_scorer(script):
    assert script.export_stream is package_export.export_stream
    assert script.onnx_logits is package_export.onnx_logits
    assert script.RunScorer is package_scoring.RunScorer


def test_clip_encoding_round_trips_losslessly(script):
    generator = np.random.default_rng(0)
    frames = generator.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    audio = generator.integers(-3000, 3000, size=512).astype("<i2")
    encoded = script.encode_clip(frames, audio)
    assert script.decode_clip(encoded, frames.shape) == (frames.tobytes(), audio.tobytes())


def test_combo_names_the_attacked_modalities(script):
    assert script.combo(_record("bonafide", "bonafide")) == "bonafide"
    assert script.combo(_record("video_splice", "bonafide")) == "video_only"
    assert script.combo(_record("bonafide", "audio_vocoder")) == "audio_only"
    assert script.combo(_record("video_splice", "audio_vocoder")) == "both"
