"""The end to end pipeline: generate, train both streams, calibrate, fuse, evaluate."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibrate import OperatingThreshold, PlattCalibrator, StreamCalibration, calibrate_stream
from .config import SpooflineConfig
from .data.dataset import (
    LoadedCorpus,
    Splits,
    load_corpus,
    make_splits,
    save_splits,
)
from .data.generate import combo_counts, family_counts, generate_corpus
from .fusion import (
    ATTRIBUTIONS,
    FusionModel,
    LogisticFusion,
    and_rule,
    attribute,
    fit_fusion,
    fit_logistic_fusion,
    or_rule,
    rule_metrics,
)
from .metrics import evaluate, family_breakdown
from .models.cnn_lstm import save_checkpoint
from .report import render_summary
from .seeding import seed_everything
from .train import TrainedStream, score_clips, train_stream

Progress = Callable[[str], None] | None
Fusions = dict[str, FusionModel | LogisticFusion]
FUSION_DETECTORS = ("fused", "logistic")
COMBOS = ("bonafide", "video_only", "audio_only", "both")


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


def modality_labels(corpus: LoadedCorpus, clip_ids, stream: str) -> np.ndarray:
    """1 where this stream's own modality was attacked."""
    field_name = "video_attacked" if stream == "video" else "audio_attacked"
    return np.array(
        [int(getattr(corpus.record(cid), field_name)) for cid in clip_ids], dtype=np.int64
    )


def _families(corpus: LoadedCorpus, clip_ids) -> list[tuple[str, ...]]:
    return [corpus.record(cid).families for cid in clip_ids]


def _training_block(trained: TrainedStream) -> dict:
    return {
        "epochs": len(trained.history),
        "best_epoch": trained.best_epoch,
        "history": trained.history,
        "final": trained.final,
        "train_clips": len(trained.train_ids),
        "val_clips": len(trained.val_ids),
    }


def calibrate_all(
    raw: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    target_precision: float,
    modality_labels: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, StreamCalibration], dict[str, dict[str, np.ndarray]], Fusions]:
    """Fit both stream calibrations and both fusions on the calibration split."""
    modality = modality_labels or {}
    calibrations = {
        stream: calibrate_stream(
            stream,
            raw[stream]["calib"],
            modality.get(stream, labels["calib"]),
            labels["calib"],
            target_precision,
        )
        for stream in ("video", "audio")
    }
    probabilities = {
        stream: {
            name: calibrations[stream].calibrator.predict(scores)
            for name, scores in raw[stream].items()
        }
        for stream in ("video", "audio")
    }
    calib = (probabilities["video"]["calib"], probabilities["audio"]["calib"], labels["calib"])
    fusions: Fusions = {
        "fused": fit_fusion(*calib, target_precision),
        "logistic": fit_logistic_fusion(*calib, target_precision),
    }
    return calibrations, probabilities, fusions


def evaluate_splits(
    corpus: LoadedCorpus,
    splits: Splits,
    probabilities: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    calibrations: dict[str, StreamCalibration],
    fusions: Fusions,
    split_names: tuple[str, ...] = ("seen_test", "unseen_test"),
) -> dict[str, dict]:
    """Clip level metrics, rules, per family rates and attribution for the test splits."""
    blocks: dict[str, dict] = {"metrics": {}, "rules": {}, "family_rates": {}, "attribution": {}}
    weighted = fusions["fused"]
    for split in split_names:
        ids = splits.as_dict()[split]
        y = labels[split]
        pv = probabilities["video"][split]
        pa = probabilities["audio"][split]
        blocks["metrics"][split], blocks["rules"][split] = split_metrics(
            pv, pa, y, calibrations, fusions
        )
        blocks["family_rates"][split] = family_breakdown(
            weighted.fuse(pv, pa), y, _families(corpus, ids), weighted.operating.threshold
        )
        combos = [corpus.record(cid).combo for cid in ids]
        blocks["attribution"][split] = attribution_table(combos, pv, pa, fusions)
    return blocks


def attribution_table(
    combos: list[str], p_video: np.ndarray, p_audio: np.ndarray, fusions: Fusions
) -> dict[str, dict[str, dict[str, int]]]:
    """Counts of which stream triggered each fusion decision, by attacked modality."""
    table: dict[str, dict[str, dict[str, int]]] = {}
    for name, model in fusions.items():
        rows = {combo: dict.fromkeys(ATTRIBUTIONS, 0) for combo in COMBOS}
        for combo, label in zip(combos, attribute(model.decide, p_video, p_audio), strict=True):
            rows[combo][label] += 1
        table[name] = rows
    return table


def split_metrics(
    p_video: np.ndarray,
    p_audio: np.ndarray,
    labels: np.ndarray,
    calibrations: dict[str, StreamCalibration],
    fusions: Fusions,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Detector metrics and the decision rule comparison for one split."""
    video, audio = calibrations["video"], calibrations["audio"]
    metrics = {
        "video": evaluate(p_video, labels, video.operating.threshold).as_dict(),
        "audio": evaluate(p_audio, labels, audio.operating.threshold).as_dict(),
    }
    for name, model in fusions.items():
        scores = model.fuse(p_video, p_audio)
        metrics[name] = evaluate(scores, labels, model.operating.threshold).as_dict()
    rules = {
        "and": rule_metrics(and_rule(video, audio, p_video, p_audio), labels),
        "or": rule_metrics(or_rule(video, audio, p_video, p_audio), labels),
        "weighted": rule_metrics(fusions["fused"].decide(p_video, p_audio), labels),
        "logistic": rule_metrics(fusions["logistic"].decide(p_video, p_audio), labels),
    }
    return metrics, rules


def precision_verdict(unseen: dict[str, dict], detector: str) -> str:
    """Compare a fusion detector's unseen precision with the best single stream."""
    best_stream = max(("video", "audio"), key=lambda s: unseen[s]["precision"])
    best_single = unseen[best_stream]["precision"]
    precision = unseen[detector]["precision"]
    if precision > best_single + 1e-9:
        relation = "is above"
    elif precision >= best_single - 1e-9:
        relation = "matches"
    else:
        relation = "is BELOW"
    return (
        f"{detector} precision {precision:.3f} {relation} the best single stream "
        f"({best_stream} {best_single:.3f}); {detector} F1 {unseen[detector]['f1']:.3f} and "
        f"AUC {unseen[detector]['auc']:.3f} against {best_stream} F1 "
        f"{unseen[best_stream]['f1']:.3f} and AUC {unseen[best_stream]['auc']:.3f}"
    )


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

    calib_modality = {
        stream: modality_labels(corpus, splits.calib, stream) for stream in ("video", "audio")
    }
    calibrations, probabilities, fusions = calibrate_all(
        raw, labels, config.target_precision, calib_modality
    )
    fusion = fusions["fused"]
    say(
        f"  video threshold {calibrations['video'].operating.threshold:.4f}, "
        f"audio threshold {calibrations['audio'].operating.threshold:.4f}, "
        f"fusion weight {fusion.weight:.2f} threshold {fusion.operating.threshold:.4f}, "
        f"logistic threshold {fusions['logistic'].operating.threshold:.4f}"
    )
    timer.mark("calibrate")

    say("[6/6] evaluating on the seen and unseen test splits")
    evaluation = evaluate_splits(corpus, splits, probabilities, labels, calibrations, fusions)
    timer.mark("evaluate")

    unseen = evaluation["metrics"]["unseen_test"]
    best_single = max(unseen["video"]["precision"], unseen["audio"]["precision"])

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
            "logistic": fusions["logistic"].as_dict(),
        },
        **evaluation,
        "headline": {
            **{
                f"{detector}_{metric}": unseen[detector][metric]
                for detector in ("fused", "logistic", "video", "audio")
                for metric in ("precision", "recall")
            },
            **{
                f"{detector}_precision_gap": unseen[detector]["precision"] - best_single
                for detector in FUSION_DETECTORS
            },
            "verdict": precision_verdict(unseen, "fused"),
            "logistic_verdict": precision_verdict(unseen, "logistic"),
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


def load_calibration(run_dir: Path) -> tuple[dict[str, StreamCalibration], Fusions]:
    """Stream calibrations and both fusions from a finished run's calibration.json."""
    payload = json.loads((Path(run_dir) / "calibration.json").read_text())
    calibrations = {
        stream: StreamCalibration(
            stream=stream,
            calibrator=PlattCalibrator.from_dict(payload[stream]["calibrator"]),
            operating=OperatingThreshold(**payload[stream]["operating"]),
        )
        for stream in ("video", "audio")
    }
    fusions: Fusions = {
        "fused": FusionModel(
            weight=float(payload["fused"]["weight"]),
            operating=OperatingThreshold(**payload["fused"]["operating"]),
        ),
        "logistic": LogisticFusion.from_dict(payload["logistic"]),
    }
    return calibrations, fusions
