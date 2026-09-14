"""Repeated seed sweep over every leave two families out split.

Each run holds out one video family and one audio family, trains both streams on
what is left, calibrates and fuses on the calibration split, and scores the seen
and unseen test splits. The raw logits of every run are cached on disk, so the
calibration, fusion and metrics are recomputed from them without retraining. The
aggregate reports the mean, the standard deviation and a percentile bootstrap
interval of the mean over runs for every detector and metric.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .config import SpooflineConfig
from .data.dataset import LoadedCorpus, load_corpus, make_splits
from .data.generate import generate_corpus
from .families import AUDIO_FAMILIES, VIDEO_FAMILIES
from .pipeline import calibrate_all, modality_labels, split_metrics
from .seeding import seed_everything
from .train import score_clips, train_stream

Pair = tuple[str, str]
Progress = Callable[[str], None] | None

STREAMS = ("video", "audio")
SCORED_SPLITS = ("calib", "seen_test", "unseen_test")
TEST_SPLITS = ("seen_test", "unseen_test")
DETECTORS = ("video", "audio", "fused", "logistic")
METRICS = ("precision", "recall", "f1", "eer", "auc")
RULES = ("and", "or")
RULE_METRICS = ("precision", "recall", "f1")
DERIVED = (
    "unseen_precision_gap",
    "logistic_unseen_precision_gap",
    "seen_to_unseen_precision_drop",
    "logistic_seen_to_unseen_precision_drop",
)


def leave_two_out_pairs() -> tuple[Pair, ...]:
    """Every pairing of one held out video family with one held out audio family."""
    return tuple(itertools.product(VIDEO_FAMILIES, AUDIO_FAMILIES))


def sweep_seeds(base_seed: int, n_seeds: int) -> tuple[int, ...]:
    return tuple(base_seed + offset for offset in range(n_seeds))


def _finite(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def bootstrap_interval(
    values, confidence: float = 0.95, n_resamples: int = 4000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of the finite values."""
    x = _finite(values)
    if x.size == 0:
        return float("nan"), float("nan")
    draws = np.random.default_rng(seed).integers(0, x.size, size=(n_resamples, x.size))
    means = x[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


def summarise(values, confidence: float = 0.95) -> MetricSummary:
    """Mean, sample standard deviation and bootstrap interval of the finite values."""
    x = _finite(values)
    if x.size == 0:
        nan = float("nan")
        return MetricSummary(nan, nan, nan, nan, 0)
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    low, high = bootstrap_interval(x, confidence)
    return MetricSummary(float(x.mean()), std, low, high, int(x.size))


def derived_metrics(metrics: dict) -> dict[str, float]:
    """How far each fusion's precision sits from the best single stream and from seen families."""
    seen, unseen = metrics["seen_test"], metrics["unseen_test"]
    best_single = max(unseen["video"]["precision"], unseen["audio"]["precision"])
    derived = {}
    for detector, prefix in (("fused", ""), ("logistic", "logistic_")):
        precision = unseen[detector]["precision"]
        derived[f"{prefix}unseen_precision_gap"] = precision - best_single
        derived[f"{prefix}seen_to_unseen_precision_drop"] = seen[detector]["precision"] - precision
    return derived


def aggregate(runs: Sequence[dict]) -> dict:
    """Summaries per split, detector and metric, plus the derived precision gaps."""
    table = {
        split: {
            detector: {
                metric: asdict(summarise([run["metrics"][split][detector][metric] for run in runs]))
                for metric in METRICS
            }
            for detector in DETECTORS
        }
        for split in TEST_SPLITS
    }
    rules = {
        split: {
            rule: {
                metric: asdict(summarise([run["rules"][split][rule][metric] for run in runs]))
                for metric in RULE_METRICS
            }
            for rule in RULES
        }
        for split in TEST_SPLITS
    }
    derived = {name: asdict(summarise([run["derived"][name] for run in runs])) for name in DERIVED}
    return {"n_runs": len(runs), "metrics": table, "rules": rules, "derived": derived}


def per_pair(runs: Sequence[dict]) -> list[dict]:
    """Unseen precision and recall per held out pair, averaged over seeds."""
    rows = []
    for pair in leave_two_out_pairs():
        subset = [run for run in runs if tuple(run["pair"]) == pair]
        if not subset:
            continue
        row: dict = {"pair": list(pair), "n": len(subset)}
        for detector in DETECTORS:
            for metric in ("precision", "recall"):
                values = [run["metrics"]["unseen_test"][detector][metric] for run in subset]
                row[f"{detector}_{metric}"] = float(np.mean(values))
        rows.append(row)
    return rows


def score_pair(corpus: LoadedCorpus, config: SpooflineConfig, pair: Pair) -> dict[str, np.ndarray]:
    """Train both streams with the pair held out and return raw logits and labels."""
    seed_everything(config.seed, config.threads)
    splits = make_splits(
        corpus.records, pair, config.seed, config.train_fraction, config.calib_fraction
    )
    ids = splits.as_dict()
    arrays: dict[str, np.ndarray] = {}
    for stream in STREAMS:
        trained = train_stream(corpus, splits, stream, config)
        for split in SCORED_SPLITS:
            arrays[f"{stream}_{split}"] = score_clips(
                trained.model, trained.normalizer, corpus, ids[split], stream
            )
        arrays[f"modality_{stream}_calib"] = modality_labels(corpus, splits.calib, stream)
    for split in SCORED_SPLITS:
        arrays[f"label_{split}"] = np.array(
            [corpus.record(cid).label for cid in ids[split]], dtype=np.int64
        )
    return arrays


def evaluate_pair(arrays: dict[str, np.ndarray], target_precision: float) -> dict:
    """Calibrate, fuse and score one run from its cached logits."""
    raw = {
        stream: {split: arrays[f"{stream}_{split}"] for split in SCORED_SPLITS}
        for stream in STREAMS
    }
    labels = {split: arrays[f"label_{split}"] for split in SCORED_SPLITS}
    modality = {stream: arrays[f"modality_{stream}_calib"] for stream in STREAMS}
    calibrations, probabilities, fusions = calibrate_all(raw, labels, target_precision, modality)
    metrics, rules = {}, {}
    for split in TEST_SPLITS:
        metrics[split], split_rules = split_metrics(
            probabilities["video"][split],
            probabilities["audio"][split],
            labels[split],
            calibrations,
            fusions,
        )
        rules[split] = {rule: split_rules[rule] for rule in RULES}
    return {
        "metrics": metrics,
        "rules": rules,
        "derived": derived_metrics(metrics),
        "fusion_weight": fusions["fused"].weight,
        "logistic": fusions["logistic"].as_dict(),
    }


def config_fingerprint(config: SpooflineConfig) -> str:
    """A short hash of everything that changes the trained models, directories excluded."""
    fields = asdict(replace(config, corpus_dir=Path(), run_dir=Path()))
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def load_scores(path: Path, fingerprint: str) -> dict[str, np.ndarray] | None:
    """Cached logits for a run, or None when missing or trained under other settings."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if str(data["fingerprint"]) != fingerprint:
            return None
        return {key: data[key] for key in data.files if key != "fingerprint"}


@dataclass(frozen=True)
class SweepTask:
    """One seed and held out pair, with the seed's corpus directory already set."""

    config: SpooflineConfig
    pair: Pair
    scores_path: Path


_CORPUS_CACHE: dict[Path, LoadedCorpus] = {}


def _corpus_for(config: SpooflineConfig) -> LoadedCorpus:
    key = Path(config.corpus_dir)
    if key not in _CORPUS_CACHE:
        _CORPUS_CACHE.clear()
        _CORPUS_CACHE[key] = load_corpus(generate_corpus(config.corpus, config.seed, key))
    return _CORPUS_CACHE[key]


def _generate(config: SpooflineConfig) -> str:
    generate_corpus(config.corpus, config.seed, Path(config.corpus_dir))
    return str(config.corpus_dir)


def run_task(task: SweepTask) -> tuple[SweepTask, float]:
    """Train and score one run unless its logits are already cached."""
    started = time.perf_counter()
    fingerprint = config_fingerprint(task.config)
    if load_scores(task.scores_path, fingerprint) is None:
        arrays = score_pair(_corpus_for(task.config), task.config, task.pair)
        task.scores_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task.scores_path, fingerprint=np.array(fingerprint), **arrays)
    return task, time.perf_counter() - started


def _execute(tasks: Sequence[SweepTask], workers: int, say: Callable[[str], None]) -> None:
    pending = [t for t in tasks if load_scores(t.scores_path, config_fingerprint(t.config)) is None]
    say(f"  {len(tasks) - len(pending)} runs cached, {len(pending)} to train")
    if not pending:
        return
    seed_configs = list({t.config.seed: t.config for t in pending}.values())

    def report(done: int, task: SweepTask, seconds: float) -> None:
        say(
            f"  run {done}/{len(pending)} seed {task.config.seed} "
            f"held out {task.pair[0]} + {task.pair[1]} ({seconds:.1f}s)"
        )

    if workers <= 1:
        for config in seed_configs:
            say(f"  corpus ready at {_generate(config)}")
        for done, task in enumerate(pending, start=1):
            report(done, *run_task(task))
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for directory in pool.map(_generate, seed_configs):
            say(f"  corpus ready at {directory}")
        futures = [pool.submit(run_task, task) for task in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            report(done, *future.result())


@dataclass
class SweepResult:
    results: dict
    summary: str
    out_dir: Path


def describe_profile(config: SpooflineConfig) -> dict:
    return {
        "corpus": asdict(config.corpus),
        "video_epochs": config.video.epochs,
        "audio_epochs": config.audio.epochs,
        "batch_size": config.video.batch_size,
        "threads_per_run": config.threads,
    }


def run_sweep(
    config: SpooflineConfig,
    n_seeds: int,
    workers: int,
    corpus_root: Path,
    out_dir: Path,
    progress: Progress = None,
) -> SweepResult:
    """Train every seed and held out pair, then aggregate the cached logits."""
    from .report import render_sweep

    say = progress or (lambda _msg: None)
    started = time.perf_counter()
    seeds = sweep_seeds(config.seed, n_seeds)
    pairs = leave_two_out_pairs()
    out_dir = Path(out_dir)
    tasks = [
        SweepTask(
            config=replace(config, seed=seed, corpus_dir=Path(corpus_root) / f"seed_{seed}"),
            pair=pair,
            scores_path=out_dir / f"seed_{seed}" / f"{pair[0]}+{pair[1]}.npz",
        )
        for seed in seeds
        for pair in pairs
    ]
    say(
        f"sweep: {len(seeds)} seeds x {len(pairs)} held out pairs = {len(tasks)} runs, "
        f"profile {config.profile}, {workers} workers"
    )
    _execute(tasks, workers, say)

    runs = []
    for task in tasks:
        arrays = load_scores(task.scores_path, config_fingerprint(task.config))
        assert arrays is not None
        runs.append(
            {
                "seed": task.config.seed,
                "pair": list(task.pair),
                **evaluate_pair(arrays, config.target_precision),
            }
        )
    results = {
        "profile": config.profile,
        "profile_settings": describe_profile(config),
        "target_precision": config.target_precision,
        "seeds": list(seeds),
        "pairs": [list(pair) for pair in pairs],
        "runs": runs,
        "aggregate": aggregate(runs),
        "per_pair": per_pair(runs),
        "wall_clock_s": time.perf_counter() - started,
    }
    summary = render_sweep(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (out_dir / "sweep.txt").write_text(summary + "\n")
    return SweepResult(results=results, summary=summary, out_dir=out_dir)
