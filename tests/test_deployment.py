"""ONNX export parity, the model card, batch JSON scoring and the latency benchmark."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from click.testing import CliRunner

from spoofline.cli import main
from spoofline.config import FIXTURE_DIR, profile
from spoofline.export import PARITY_TOLERANCE, export_run, onnx_logits
from spoofline.model_card import REQUIRED_SECTIONS, write_model_card
from spoofline.pipeline import run_pipeline
from spoofline.scoring import SCORE_FIELDS, RunScorer, clip_steps, read_npz_clip

CLIPS = sorted((FIXTURE_DIR / "clips").glob("*.npz"))


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("deploy") / "run"
    config = replace(profile("tiny"), corpus_dir=FIXTURE_DIR, run_dir=run_dir)
    run_pipeline(config)
    return run_dir


@pytest.fixture(scope="module")
def exported(finished_run):
    return export_run(finished_run, finished_run / "onnx", CLIPS[:6])


def test_onnx_matches_pytorch_within_tolerance_for_both_streams(finished_run, exported):
    scorer = RunScorer.from_run(finished_run)
    clips = [read_npz_clip(path, 16000) for path in CLIPS]
    for stream in ("video", "audio"):
        torch_logits = scorer.logits(clips, streams=(stream,))[stream]
        steps = [clip_steps(stream, clip) for clip in clips]
        onnx = onnx_logits(exported["files"][stream], steps)
        assert np.max(np.abs(onnx - torch_logits)) <= PARITY_TOLERANCE
        assert exported["parity"][stream]["max_abs_diff"] <= PARITY_TOLERANCE


def test_onnx_accepts_other_batch_sizes_and_step_counts(finished_run, exported):
    scorer = RunScorer.from_run(finished_run)
    clip = read_npz_clip(CLIPS[0], 16000)
    steps = clip_steps("video", clip)[:5]
    model, normalizer = scorer.models["video"]
    from spoofline.scoring import batch_logits

    expected = batch_logits(model, normalizer, [steps])
    assert np.allclose(onnx_logits(exported["files"]["video"], [steps]), expected, atol=1e-4)


def test_model_card_carries_the_run_numbers(finished_run):
    path = write_model_card(finished_run)
    card = path.read_text()
    results = json.loads((finished_run / "results.json").read_text())
    for section in REQUIRED_SECTIONS:
        assert section in card
    for detector in ("video", "audio", "fused", "logistic"):
        threshold = results["calibration"][detector]["operating"]["threshold"]
        assert f"{threshold:.4f}" in card
        precision = results["metrics"]["unseen_test"][detector]["precision"]
        assert f"{precision:.3f}" in card
    for name, count in results["splits"]["counts"].items():
        assert f"| {name} | {count} |" in card
    for family in results["unseen_families"]:
        assert family in card


def test_batch_score_emits_json_with_the_documented_schema(finished_run):
    runner = CliRunner()
    paths = [str(path) for path in CLIPS[:4]]
    result = runner.invoke(
        main, ["score", *paths, "--json", "--profile", "tiny", "--run-dir", str(finished_run)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["run_dir"] == str(finished_run)
    assert len(payload["clips"]) == 4
    for path, entry in zip(paths, payload["clips"], strict=True):
        assert set(entry) == set(SCORE_FIELDS)
        assert entry["clip"] == path
        assert entry["decision"] in {"attack", "bonafide"}
        assert entry["triggered_by"] in {"none", "video", "audio", "either", "joint"}
        assert 0.0 <= entry["fused_probability"] <= 1.0
        assert isinstance(entry["video_flags"], bool)


def test_latency_benchmark_reports_p50_and_p95_for_every_engine_and_stage(finished_run, exported):
    from spoofline.bench import run_benchmark

    result = run_benchmark(finished_run, CLIPS[:4], finished_run / "onnx", threads=1, warmup=1)
    stages = {(row["engine"], row["stage"]) for row in result["rows"]}
    assert stages == {
        (engine, stage)
        for engine in ("torch", "onnx")
        for stage in ("video", "audio", "end_to_end")
    }
    for row in result["rows"]:
        assert row["n"] == 4
        assert 0.0 < row["p50_ms"] <= row["p95_ms"]
