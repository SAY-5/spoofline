"""Command line interface."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import click
import numpy as np

from .calibrate import OperatingThreshold, PlattCalibrator, StreamCalibration
from .config import PROFILE_NAMES, REPO_ROOT, SpooflineConfig
from .config import profile as load_profile
from .data.dataset import load_corpus, make_splits, save_splits
from .data.generate import combo_counts, family_counts, generate_corpus
from .fusion import FusionModel
from .models.cnn_lstm import load_checkpoint, save_checkpoint
from .pipeline import (
    calibrate_all,
    evaluate_splits,
    modality_labels,
    run_pipeline,
    score_single_clip,
)
from .report import render_evaluation
from .seeding import seed_everything
from .sweep import run_sweep
from .train import score_clips, train_stream

PROFILE_OPTION = click.option(
    "--profile", type=click.Choice(PROFILE_NAMES), default="full", show_default=True
)
SEED_OPTION = click.option("--seed", type=int, default=None, help="Override the run seed.")


def _config(profile: str, seed: int | None, corpus_dir: str | None, run_dir: str | None):
    config = load_profile(profile)
    if seed is not None:
        config = replace(config, seed=seed)
    if corpus_dir:
        config = replace(config, corpus_dir=Path(corpus_dir))
    if run_dir:
        config = replace(config, run_dir=Path(run_dir))
    return config


def _prepare(config: SpooflineConfig, force: bool = False):
    seed_everything(config.seed, config.threads)
    source = generate_corpus(config.corpus, config.seed, config.corpus_dir, force=force)
    corpus = load_corpus(source)
    splits = make_splits(
        corpus.records,
        config.unseen_families,
        config.seed,
        config.train_fraction,
        config.calib_fraction,
    )
    return source, corpus, splits


def _load_calibration(run_dir: Path):
    payload = json.loads((run_dir / "calibration.json").read_text())
    calibrations = {
        stream: StreamCalibration(
            stream=stream,
            calibrator=PlattCalibrator.from_dict(payload[stream]["calibrator"]),
            operating=OperatingThreshold(
                threshold=payload[stream]["operating"]["threshold"],
                target_precision=payload[stream]["operating"]["target_precision"],
                achieved_precision=payload[stream]["operating"]["achieved_precision"],
                recall=payload[stream]["operating"]["recall"],
                reached_target=payload[stream]["operating"]["reached_target"],
            ),
        )
        for stream in ("video", "audio")
    }
    fused = payload["fused"]
    fusion = FusionModel(
        weight=fused["weight"],
        operating=OperatingThreshold(
            threshold=fused["operating"]["threshold"],
            target_precision=fused["operating"]["target_precision"],
            achieved_precision=fused["operating"]["achieved_precision"],
            recall=fused["operating"]["recall"],
            reached_target=fused["operating"]["reached_target"],
        ),
    )
    return calibrations, fusion


def _score_everything(config: SpooflineConfig, corpus, splits):
    raw: dict[str, dict[str, np.ndarray]] = {}
    for stream in ("video", "audio"):
        model, normalizer, _ = load_checkpoint(Path(config.run_dir) / f"{stream}.pt")
        raw[stream] = {
            name: score_clips(model, normalizer, corpus, ids, stream)
            for name, ids in splits.as_dict().items()
        }
    labels = {
        name: np.array([corpus.record(cid).label for cid in ids], dtype=np.int64)
        for name, ids in splits.as_dict().items()
    }
    return raw, labels


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="spoofline")
def main() -> None:
    """Two-stream audio and video spoof detection."""


@main.command()
@PROFILE_OPTION
@SEED_OPTION
@click.option("--corpus-dir", type=click.Path(), default=None)
@click.option("--force/--no-force", default=False, help="Regenerate even if a corpus is cached.")
def generate(profile: str, seed: int | None, corpus_dir: str | None, force: bool) -> None:
    """Render the deterministic corpus."""
    config = _config(profile, seed, corpus_dir, None)
    source = generate_corpus(
        config.corpus, config.seed, config.corpus_dir, progress=click.echo, force=force
    )
    click.echo(f"corpus at {config.corpus_dir}: {len(source)} clips")
    click.echo(f"combinations: {combo_counts(source.records)}")
    for key, value in family_counts(source.records).items():
        click.echo(f"  {key:<26}{value}")


@main.command()
@click.option("--stream", type=click.Choice(["video", "audio"]), required=True)
@PROFILE_OPTION
@SEED_OPTION
@click.option("--corpus-dir", type=click.Path(), default=None)
@click.option("--run-dir", type=click.Path(), default=None)
def train(
    stream: str, profile: str, seed: int | None, corpus_dir: str | None, run_dir: str | None
) -> None:
    """Train one stream and write its checkpoint."""
    config = _config(profile, seed, corpus_dir, run_dir)
    _, corpus, splits = _prepare(config)
    Path(config.run_dir).mkdir(parents=True, exist_ok=True)
    save_splits(Path(config.run_dir) / "splits.json", splits)
    trained = train_stream(corpus, splits, stream, config, progress=click.echo)
    path = Path(config.run_dir) / f"{stream}.pt"
    save_checkpoint(path, trained.model, trained.normalizer, {"history": trained.history})
    click.echo(f"wrote {path}")


@main.command()
@PROFILE_OPTION
@SEED_OPTION
@click.option("--corpus-dir", type=click.Path(), default=None)
@click.option("--run-dir", type=click.Path(), default=None)
def calibrate(profile: str, seed: int | None, corpus_dir: str | None, run_dir: str | None) -> None:
    """Fit both stream calibrations and the fusion weight on the calibration split."""
    config = _config(profile, seed, corpus_dir, run_dir)
    _, corpus, splits = _prepare(config)
    raw, labels = _score_everything(config, corpus, splits)
    calib_modality = {
        stream: modality_labels(corpus, splits.calib, stream) for stream in ("video", "audio")
    }
    calibrations, _, fusion = calibrate_all(raw, labels, config.target_precision, calib_modality)
    payload = {
        "video": calibrations["video"].as_dict(),
        "audio": calibrations["audio"].as_dict(),
        "fused": fusion.as_dict(),
    }
    path = Path(config.run_dir) / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for stream in ("video", "audio"):
        operating = calibrations[stream].operating
        click.echo(
            f"{stream}: threshold {operating.threshold:.4f} "
            f"precision {operating.achieved_precision:.3f} recall {operating.recall:.3f}"
        )
    click.echo(
        f"fused: weight {fusion.weight:.2f} threshold {fusion.operating.threshold:.4f} "
        f"precision {fusion.operating.achieved_precision:.3f} recall {fusion.operating.recall:.3f}"
    )
    click.echo(f"wrote {path}")


@main.command("eval")
@PROFILE_OPTION
@SEED_OPTION
@click.option("--corpus-dir", type=click.Path(), default=None)
@click.option("--run-dir", type=click.Path(), default=None)
def eval_command(
    profile: str, seed: int | None, corpus_dir: str | None, run_dir: str | None
) -> None:
    """Evaluate the calibrated streams and their fusion on both test splits."""
    config = _config(profile, seed, corpus_dir, run_dir)
    _, corpus, splits = _prepare(config)
    raw, labels = _score_everything(config, corpus, splits)
    calibrations, fusion = _load_calibration(Path(config.run_dir))
    probabilities = {
        stream: {
            name: calibrations[stream].calibrator.predict(scores)
            for name, scores in raw[stream].items()
        }
        for stream in ("video", "audio")
    }
    metrics, rules, family_rates = evaluate_splits(
        corpus, splits, probabilities, labels, calibrations, fusion
    )
    click.echo(
        render_evaluation({"metrics": metrics, "rules": rules, "family_rates": family_rates})
    )


@main.command()
@click.argument("clip", type=click.Path(exists=True))
@PROFILE_OPTION
@click.option("--run-dir", type=click.Path(), default=None)
def score(clip: str, profile: str, run_dir: str | None) -> None:
    """Score a single clip npz with a finished run."""
    config = _config(profile, None, None, run_dir)
    result = score_single_clip(
        Path(config.run_dir), Path(clip), sample_rate=config.corpus.sample_rate
    )
    for key, value in result.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        click.echo(f"{key:<20}{formatted}")


@main.command()
@PROFILE_OPTION
@SEED_OPTION
@click.option("--corpus-dir", type=click.Path(), default=None)
@click.option("--run-dir", type=click.Path(), default=None)
@click.option("--force/--no-force", default=False, help="Regenerate the corpus first.")
def pipeline(
    profile: str, seed: int | None, corpus_dir: str | None, run_dir: str | None, force: bool
) -> None:
    """Run generation, training, calibration, fusion and evaluation end to end."""
    config = _config(profile, seed, corpus_dir, run_dir)
    result = run_pipeline(config, progress=click.echo, force_generate=force)
    click.echo("")
    click.echo(result.summary)
    click.echo(f"artifacts in {result.run_dir}")


@main.command("sweep")
@click.option("--profile", type=click.Choice(PROFILE_NAMES), default="reduced", show_default=True)
@SEED_OPTION
@click.option(
    "--seeds",
    "n_seeds",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Number of consecutive run seeds, starting at the profile seed.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Training processes run in parallel.",
)
@click.option("--corpus-root", type=click.Path(), default=None)
@click.option("--out-dir", type=click.Path(), default=None)
def sweep_command(
    profile: str,
    seed: int | None,
    n_seeds: int,
    workers: int,
    corpus_root: str | None,
    out_dir: str | None,
) -> None:
    """Repeat the evaluation over seeds and all 16 leave two families out splits."""
    config = _config(profile, seed, None, None)
    corpus = Path(corpus_root) if corpus_root else REPO_ROOT / "data" / "sweep" / config.profile
    out = Path(out_dir) if out_dir else REPO_ROOT / "runs" / "sweep" / config.profile
    result = run_sweep(config, n_seeds, workers, corpus, out, progress=click.echo)
    click.echo("")
    click.echo(result.summary)
    click.echo(f"artifacts in {result.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
