"""CLI smoke tests through the click runner."""

from __future__ import annotations

import json

from click.testing import CliRunner

from spoofline.cli import main
from spoofline.config import FIXTURE_DIR


def test_help_lists_every_command():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in (
        "generate",
        "train",
        "calibrate",
        "eval",
        "score",
        "pipeline",
        "sweep",
        "robustness",
    ):
        assert command in result.output


def test_generate_writes_a_corpus(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["generate", "--profile", "tiny", "--corpus-dir", str(tmp_path / "corpus"), "--force"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "corpus" / "manifest.json").exists()
    assert "24 clips" in result.output


def test_pipeline_runs_end_to_end_and_writes_artifacts(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "pipeline",
            "--profile",
            "tiny",
            "--corpus-dir",
            str(FIXTURE_DIR),
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "spoofline pipeline summary" in result.output
    for name in ("results.json", "summary.txt", "calibration.json", "video.pt", "audio.pt"):
        assert (tmp_path / "run" / name).exists()
    results = json.loads((tmp_path / "run" / "results.json").read_text())
    assert results["profile"] == "tiny"
    assert "unseen_test" in results["metrics"]


def test_eval_and_score_reuse_a_finished_run(tmp_path):
    runner = CliRunner()
    run_dir = tmp_path / "run"
    common = ["--profile", "tiny", "--corpus-dir", str(FIXTURE_DIR), "--run-dir", str(run_dir)]
    assert runner.invoke(main, ["pipeline", *common]).exit_code == 0

    evaluated = runner.invoke(main, ["eval", *common])
    assert evaluated.exit_code == 0, evaluated.output
    assert "clip level metrics" in evaluated.output

    clip = FIXTURE_DIR / "clips" / "clip_00000.npz"
    scored = runner.invoke(
        main, ["score", str(clip), "--profile", "tiny", "--run-dir", str(run_dir)]
    )
    assert scored.exit_code == 0, scored.output
    assert "fused_probability" in scored.output
    assert "decision" in scored.output


def test_train_and_calibrate_commands_produce_checkpoints(tmp_path):
    runner = CliRunner()
    run_dir = tmp_path / "run"
    common = ["--profile", "tiny", "--corpus-dir", str(FIXTURE_DIR), "--run-dir", str(run_dir)]
    for stream in ("video", "audio"):
        result = runner.invoke(main, ["train", "--stream", stream, *common])
        assert result.exit_code == 0, result.output
        assert (run_dir / f"{stream}.pt").exists()
    result = runner.invoke(main, ["calibrate", *common])
    assert result.exit_code == 0, result.output
    payload = json.loads((run_dir / "calibration.json").read_text())
    assert set(payload) == {"video", "audio", "fused", "logistic"}
