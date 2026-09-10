"""The end to end pipeline: generate, train both streams, calibrate, fuse, evaluate."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibrate import StreamCalibration, calibrate_stream
from .config import SpooflineConfig
from .data.dataset import (
    LoadedCorpus,
    Splits,
    load_corpus,
    make_splits,
    save_splits,
)
from .data.generate import combo_counts, family_counts, generate_corpus
from .data.sources import NpzCorpusSource
from .fusion import FusionModel, and_rule, fit_fusion, or_rule, rule_metrics
from .metrics import evaluate, family_breakdown
from .models.cnn_lstm import load_checkpoint, save_checkpoint
from .report import render_summary
from .seeding import seed_everything
from .train import TrainedStream, score_clips, train_stream

Progress = Callable[[str], None] | None


@dataclass
class PipelineResult:
    """Machine readable results plus the rendered summary block."""

    results: dict
    summary: str
    run_dir: Path
    calibration: dict[str, object] = field(default_factory=dict)


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = now - self._start
        self._start = now


def _stage_scores(
    trained: TrainedStream, corpus: LoadedCorpus, splits: Splits
) -> dict[str, np.ndarray]:
    return {
        name: score_clips(trained.model, trained.normalizer, corpus, ids, trained.stream)
        for name, ids in splits.as_dict().items()
    }


def _labels(corpus: LoadedCorpus, clip_ids) -> np.ndarray:
    return np.array([corpus.record(cid).label for cid in clip_ids], dtype=np.int64)


def _families(corpus: LoadedCorpus, clip_ids) -> list[tuple[str, ...]]:
    return [corpus.record(cid).families for cid in clip_ids]


def _training_block(trained: TrainedStream) -> dict:
    return {
        "epochs": len(trained.history),
        "history": trained.history,
        "final": trained.final,
        "train_clips": len(trained.train_ids),
        "val_clips": len(trained.val_ids),
    }


def calibrate_all(
    raw: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    target_precision: float,
) -> tuple[dict[str, StreamCalibration], dict[str, dict[str, np.ndarray]], FusionModel]:
    """Fit both stream calibrations and the fusion weight on the calibration split."""
    calibrations = {
        stream: calibrate_stream(stream, raw[stream]["calib"], labels["calib"], target_precision)
        for stream in ("video", "audio")
    }
    probabilities = {
        stream: {
            name: calibrations[stream].calibrator.predict(scores)
            for name, scores in raw[stream].items()
        }
        for stream in ("video", "audio")
    }
    fusion = fit_fusion(
        probabilities["video"]["calib"],
        probabilities["audio"]["calib"],
        labels["calib"],
        target_precision,
    )
    return calibrations, probabilities, fusion


def evaluate_splits(
    corpus: LoadedCorpus,
    splits: Splits,
    probabilities: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    calibrations: dict[str, StreamCalibration],
    fusion: FusionModel,
    split_names: tuple[str, ...] = ("seen_test", "unseen_test"),
) -> tuple[dict, dict, dict]:
    """Clip level metrics, rule comparison and per family rates for the test splits."""
    metrics: dict[str, dict[str, dict]] = {}
    rules: dict[str, dict[str, dict]] = {}
    family_rates: dict[str, dict[str, dict[str, float]]] = {}
    for split in split_names:
        ids = splits.as_dict()[split]
        y = labels[split]
        pv = probabilities["video"][split]
        pa = probabilities["audio"][split]
        fused = fusion.fuse(pv, pa)
        metrics[split] = {
            "video": evaluate(pv, y, calibrations["video"].operating.threshold).as_dict(),
            "audio": evaluate(pa, y, calibrations["audio"].operating.threshold).as_dict(),
            "fused": evaluate(fused, y, fusion.operating.threshold).as_dict(),
        }
        rules[split] = {
            "and": rule_metrics(and_rule(calibrations["video"], calibrations["audio"], pv, pa), y),
            "or": rule_metrics(or_rule(calibrations["video"], calibrations["audio"], pv, pa), y),
            "weighted": rule_metrics(fused >= fusion.operating.threshold, y),
        }
        family_rates[split] = family_breakdown(
            fused, y, _families(corpus, ids), fusion.operating.threshold
        )
    return metrics, rules, family_rates


def run_pipeline(
    config: SpooflineConfig,
    progress: Progress = None,
    force_generate: bool = False,
) -> PipelineResult:
    """Run every stage and return the results plus the printable summary."""
    say = progress or (lambda _msg: None)
    seed_everything(config.seed, config.threads)
    timer = _Timer()
    started = time.perf_counter()

    say(f"[1/6] generating corpus '{config.corpus.name}' into {config.corpus_dir}")
    source = generate_corpus(
        config.corpus, config.seed, config.corpus_dir, progress=say, force=force_generate
    )
    timer.mark("generate")

    say(f"[2/6] loading {len(source)} clips into memory")
    corpus = load_corpus(source, progress=say)
    splits = make_splits(
        corpus.records,
        config.unseen_families,
        config.seed,
        config.train_fraction,
        config.calib_fraction,
    )
    say(f"  splits {splits.counts()}")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_splits(run_dir / "splits.json", splits)
    timer.mark("load")

    say("[3/6] training the video stream")
    video = train_stream(corpus, splits, "video", config, progress=say)
    save_checkpoint(run_dir / "video.pt", video.model, video.normalizer, {"history": video.history})
    timer.mark("train_video")

    say("[4/6] training the audio stream")
    audio = train_stream(corpus, splits, "audio", config, progress=say)
    save_checkpoint(run_dir / "audio.pt", audio.model, audio.normalizer, {"history": audio.history})
    timer.mark("train_audio")

    say("[5/6] scoring every split and calibrating")
    raw = {
        "video": _stage_scores(video, corpus, splits),
        "audio": _stage_scores(audio, corpus, splits),
    }
    labels = {name: _labels(corpus, ids) for name, ids in splits.as_dict().items()}

    calibrations, probabilities, fusion = calibrate_all(raw, labels, config.target_precision)
    say(
        f"  video threshold {calibrations['video'].operating.threshold:.4f}, "
        f"audio threshold {calibrations['audio'].operating.threshold:.4f}, "
        f"fusion weight {fusion.weight:.2f} threshold {fusion.operating.threshold:.4f}"
    )
    timer.mark("calibrate")

    say("[6/6] evaluating on the seen and unseen test splits")
    metrics, rules, family_rates = evaluate_splits(
        corpus, splits, probabilities, labels, calibrations, fusion
    )
    timer.mark("evaluate")

    unseen = metrics["unseen_test"]
    best_single = max(unseen["video"]["precision"], unseen["audio"]["precision"])
    if unseen["fused"]["precision"] > best_single + 1e-9:
        verdict = "fusion holds a higher precision than either single stream on unseen families"
    elif unseen["fused"]["precision"] >= best_single - 1e-9:
        verdict = "fusion matches the best single stream precision on unseen families"
    else:
        verdict = (
            "fusion does NOT hold precision against the best single stream on unseen families "
            f"({unseen['fused']['precision']:.3f} vs {best_single:.3f})"
        )

    results = {
        "profile": config.profile,
        "seed": config.seed,
        "target_precision": config.target_precision,
        "unseen_families": list(config.unseen_families),
        "corpus": dict(source.meta),
        "combo_counts": combo_counts(corpus.records),
        "family_counts": family_counts(corpus.records),
        "splits": {
            "counts": splits.counts(),
            "identity_pools": {k: len(v) for k, v in splits.identity_pools.items()},
        },
        "training": {"video": _training_block(video), "audio": _training_block(audio)},
        "calibration": {
            "video": calibrations["video"].as_dict(),
            "audio": calibrations["audio"].as_dict(),
            "fused": fusion.as_dict(),
        },
        "metrics": metrics,
        "rules": rules,
        "family_rates": family_rates,
        "headline": {
            "fused_precision": unseen["fused"]["precision"],
            "video_precision": unseen["video"]["precision"],
            "audio_precision": unseen["audio"]["precision"],
            "fused_recall": unseen["fused"]["recall"],
            "video_recall": unseen["video"]["recall"],
            "audio_recall": unseen["audio"]["recall"],
            "verdict": verdict,
        },
        "timings": {**timer.timings, "total": time.perf_counter() - started},
    }
    summary = render_summary(results)
    (run_dir / "calibration.json").write_text(
        json.dumps(results["calibration"], indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (run_dir / "summary.txt").write_text(summary + "\n")
    return PipelineResult(results=results, summary=summary, run_dir=run_dir)


def score_single_clip(run_dir: Path, clip_path: Path, sample_rate: int | None = None) -> dict:
    """Score one npz clip with the checkpoints and calibration from a finished run."""
    import torch

    from .data.features import log_mel, mel_patches, video_steps

    calibration = json.loads((Path(run_dir) / "calibration.json").read_text())
    video_model, video_norm, _ = load_checkpoint(Path(run_dir) / "video.pt")
    audio_model, audio_norm, _ = load_checkpoint(Path(run_dir) / "audio.pt")
    rate = sample_rate or 16000

    with np.load(clip_path) as data:
        frames = data["video"]
        audio = data["audio"].astype(np.float32) / 32767.0

    with torch.no_grad():
        v_steps = torch.from_numpy(video_steps(frames))[None]
        v_logit = float(video_model(video_norm.apply(v_steps), torch.tensor([v_steps.shape[1]]))[0])
        a_steps = torch.from_numpy(mel_patches(log_mel(audio, rate)))[None]
        a_logit = float(audio_model(audio_norm.apply(a_steps), torch.tensor([a_steps.shape[1]]))[0])

    from .calibrate import PlattCalibrator

    pv = float(PlattCalibrator.from_dict(calibration["video"]["calibrator"]).predict([v_logit])[0])
    pa = float(PlattCalibrator.from_dict(calibration["audio"]["calibrator"]).predict([a_logit])[0])
    weight = float(calibration["fused"]["weight"])
    fused = weight * pv + (1.0 - weight) * pa
    return {
        "clip": str(clip_path),
        "video_logit": v_logit,
        "audio_logit": a_logit,
        "video_probability": pv,
        "audio_probability": pa,
        "fused_probability": fused,
        "video_flags": pv >= calibration["video"]["operating"]["threshold"],
        "audio_flags": pa >= calibration["audio"]["operating"]["threshold"],
        "decision": "attack"
        if fused >= calibration["fused"]["operating"]["threshold"]
        else "bonafide",
    }


def load_corpus_source(corpus_dir: Path) -> NpzCorpusSource:
    return NpzCorpusSource(Path(corpus_dir))
