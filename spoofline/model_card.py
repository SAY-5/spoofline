"""A model card rendered from the results of the last evaluation run."""

from __future__ import annotations

import json
from pathlib import Path

from .report import POOL_ORDER, SPLIT_ORDER, ordered_keys

DETECTORS = ("video", "audio", "fused", "logistic")
METRIC_COLUMNS = ("precision", "recall", "f1", "eer", "auc")
REQUIRED_SECTIONS = (
    "## Model details",
    "## Data note",
    "## Splits",
    "## Thresholds",
    "## Metrics",
    "## Limitations",
)


def _metrics_table(split_metrics: dict) -> list[str]:
    lines = [
        "| detector | precision | recall | F1 | EER | AUC | clips | attacks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for detector in DETECTORS:
        row = split_metrics[detector]
        cells = " | ".join(f"{row[key]:.3f}" for key in METRIC_COLUMNS)
        lines.append(f"| {detector} | {cells} | {row['n']} | {row['n_positive']} |")
    return lines


def _score_formulas(calibration: dict) -> dict[str, str]:
    formulas = {}
    for stream in ("video", "audio"):
        platt = calibration[stream]["calibrator"]
        formulas[stream] = f"sigmoid({platt['a']:.4f} logit {platt['b']:+.4f})"
    weight = calibration["fused"]["weight"]
    formulas["fused"] = f"{weight:.2f} p_video + {1.0 - weight:.2f} p_audio"
    logistic = calibration["logistic"]
    c = logistic["coefficients"]
    formulas["logistic"] = (
        f"sigmoid({c['p_video']:.3f} p_video {c['p_audio']:+.3f} p_audio "
        f"{c['disagreement']:+.3f} disagreement {logistic['intercept']:+.3f})"
    )
    return formulas


def _thresholds(calibration: dict, target: float) -> list[str]:
    formulas = _score_formulas(calibration)
    lines = [
        f"Every operating point was chosen on the calibration split at target precision "
        f"{target:.2f}. Disagreement is the absolute difference of the two stream probabilities.",
        "",
        "| detector | score | threshold | calib precision | calib recall | target met |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for detector in DETECTORS:
        op = calibration[detector]["operating"]
        lines.append(
            f"| {detector} | {formulas[detector]} | {op['threshold']:.4f} | "
            f"{op['achieved_precision']:.3f} | {op['recall']:.3f} | "
            f"{str(op['reached_target']).lower()} |"
        )
    return lines


def _robustness_lines(robustness: dict) -> list[str]:
    rows = robustness["false_alarms"]
    clean = rows[0]
    heaviest = {row["perturbation"]: row for row in rows[1:]}
    lines = [
        f"False alarm rate on {clean['n']} bona fide test clips, clean and at the heaviest "
        "severity of each perturbation.",
        "",
        "| perturbation | severity | video | audio | fused | logistic |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in (clean, *heaviest.values()):
        severity = "none" if row["severity"] is None else f"{row['unit']} {row['severity']:g}"
        cells = " | ".join(f"{row[detector]:.3f}" for detector in DETECTORS)
        lines.append(f"| {row['perturbation']} | {severity} | {cells} |")
    return lines


def _limitations(results: dict) -> list[str]:
    corpus = results["corpus"]
    unseen = results["metrics"]["unseen_test"]
    best = max(("video", "audio"), key=lambda stream: unseen[stream]["precision"])
    return [
        "* The corpus is synthetic, rendered by the repository's deterministic generator. "
        "These metrics describe that corpus, not FaceForensics++, ASVspoof or real capture.",
        f"* Clips are {corpus['n_frames']} frames of {corpus['frame_size']}x"
        f"{corpus['frame_size']} with {corpus['duration_s']} s of audio at "
        f"{corpus['sample_rate']} Hz, far smaller than real forensic inputs.",
        f"* One seed and one held out pair ({', '.join(results['unseen_families'])}). "
        "`spoofline sweep` measures variance over seeds and all 16 held out pairs.",
        "* Thresholds are the lowest that reach the target precision on the calibration "
        "split, the most optimistic choice on a finite set.",
        f"* On unseen families the weighted fusion precision is "
        f"{unseen['fused']['precision']:.3f} and the logistic fusion precision is "
        f"{unseen['logistic']['precision']:.3f}, against {best} alone at "
        f"{unseen[best]['precision']:.3f}.",
        "* Each stream only sees its own modality, so single stream recall is capped by the "
        "share of attacks that touch that modality.",
        "* No benign channel variation is in training; `spoofline robustness` shows modest "
        "audio noise, resampling or reverb raising the fused false alarm rate sharply.",
    ]


def render_model_card(results: dict, robustness: dict | None = None) -> str:
    """Markdown model card for a finished pipeline run."""
    corpus, splits, training = results["corpus"], results["splits"], results["training"]
    lines = [
        "# Spoofline model card",
        "",
        f"Rendered from the evaluation run with profile `{results['profile']}` and seed "
        f"{results['seed']} by `spoofline model-card`.",
        "",
        "## Model details",
        "",
        "Two CNN-LSTM detectors, one per stream: a per step CNN encoder, a packed "
        "bidirectional LSTM, masked attention pooling and a single logit head. Each stream "
        "is trained on its own modality label, calibrated with a Platt map, and the two "
        "probabilities are fused by a weighted sum (the primary decision) and by a logistic "
        "regression over both probabilities and their disagreement.",
        "",
    ]
    for stream in ("video", "audio"):
        block = training[stream]
        lines.append(
            f"* {stream}: {block['epochs']} epochs, kept epoch {block['best_epoch']}, "
            f"{block['train_clips']} training clips, {block['val_clips']} validation clips."
        )
    lines += [
        "",
        "## Data note",
        "",
        "No public spoofing corpus is used, because the public sets need signed licences. "
        f"The corpus is {corpus['n_clips']} generated clips from {corpus['n_identities']} "
        "identities, each with a bona fide or attacked video stream and a bona fide or "
        "attacked audio stream. The eight attack families are real signal transformations. "
        "A clip is labelled an attack if either stream was attacked.",
        "",
        "## Splits",
        "",
        f"Unseen families: {', '.join(results['unseen_families'])}. Identity pools never mix.",
        "",
        "| split | clips |",
        "| --- | --- |",
        *[
            f"| {name} | {splits['counts'][name]} |"
            for name in ordered_keys(splits["counts"], SPLIT_ORDER)
        ],
        "",
        "| identity pool | identities |",
        "| --- | --- |",
        *[
            f"| {name} | {splits['identity_pools'][name]} |"
            for name in ordered_keys(splits["identity_pools"], POOL_ORDER)
        ],
        "",
        "## Thresholds",
        "",
        *_thresholds(results["calibration"], results["target_precision"]),
        "",
        "## Metrics",
        "",
        "Seen attack families:",
        "",
        *_metrics_table(results["metrics"]["seen_test"]),
        "",
        "Unseen attack families:",
        "",
        *_metrics_table(results["metrics"]["unseen_test"]),
        "",
    ]
    if robustness is not None:
        lines += ["## Robustness", "", *_robustness_lines(robustness), ""]
    lines += ["## Limitations", "", *_limitations(results)]
    return "\n".join(lines) + "\n"


def write_model_card(run_dir: Path, out_path: Path | None = None) -> Path:
    """Render the card from results.json, and robustness.json when present."""
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    robustness_path = run_dir / "robustness.json"
    robustness = json.loads(robustness_path.read_text()) if robustness_path.exists() else None
    path = Path(out_path) if out_path else run_dir / "MODEL_CARD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_card(results, robustness))
    return path
