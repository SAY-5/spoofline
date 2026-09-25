"""False alarm rates under benign degradation, and coverage when the streams disagree."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import SpooflineConfig
from .data.dataset import load_splits
from .data.generate import generate_corpus
from .data.sources import Clip
from .fusion import rule_metrics
from .perturb import PERTURBATIONS, perturb_clip
from .pipeline import FUSION_DETECTORS
from .report import render_robustness
from .scoring import RunScorer
from .seeding import seed_everything

Progress = Callable[[str], None] | None

TEST_SPLITS = ("seen_test", "unseen_test")
DETECTORS = ("video", "audio", *FUSION_DETECTORS)
ABSTAIN_MARGINS = (1.0, 0.9, 0.75, 0.5, 0.25)


def abstain_mask(p_video, p_audio, margin: float) -> np.ndarray:
    """True where the calibrated streams disagree by more than ``margin``, so it abstains."""
    pv = np.asarray(p_video, dtype=np.float64)
    pa = np.asarray(p_audio, dtype=np.float64)
    return np.abs(pv - pa) > margin


def abstain_curve(
    p_video, p_audio, decisions, labels, margins: Sequence[float] = ABSTAIN_MARGINS
) -> list[dict]:
    """Coverage, and precision and recall on the clips kept, for each abstain margin."""
    flags = np.asarray(decisions, dtype=bool)
    y = np.asarray(labels, dtype=np.int64)
    rows = []
    for margin in margins:
        kept = ~abstain_mask(p_video, p_audio, margin)
        scored = rule_metrics(flags[kept], y[kept])
        rows.append(
            {
                "margin": float(margin),
                "coverage": float(kept.mean()) if len(kept) else float("nan"),
                "kept": int(kept.sum()),
                "abstained_attacks": int(np.sum(~kept & (y == 1))),
                "abstained_bonafide": int(np.sum(~kept & (y == 0))),
                "precision": scored["precision"],
                "recall": scored["recall"],
            }
        )
    return rows


def false_alarm_rates(flags: dict[str, np.ndarray]) -> dict[str, float]:
    return {detector: float(np.mean(flags[detector])) for detector in DETECTORS}


def false_alarm_table(scorer: RunScorer, clips: Sequence[Clip], seed: int, say) -> list[dict]:
    """False alarm rate per detector on the clean clips and under every perturbation."""
    clean = scorer.logits(clips)
    rows = [
        {"perturbation": "clean", "stream": "none", "unit": "", "severity": None, "n": len(clips)}
        | false_alarm_rates(scorer.flags(clean))
    ]
    for perturbation in PERTURBATIONS:
        for severity in perturbation.severities:
            perturbed = [perturb_clip(clip, perturbation, severity, seed) for clip in clips]
            logits = dict(clean) | scorer.logits(perturbed, streams=(perturbation.stream,))
            rows.append(
                {
                    "perturbation": perturbation.name,
                    "stream": perturbation.stream,
                    "unit": perturbation.unit,
                    "severity": float(severity),
                    "n": len(clips),
                }
                | false_alarm_rates(scorer.flags(logits))
            )
        say(f"  {perturbation.name}: {len(perturbation.severities)} severities scored")
    return rows


def abstain_table(scorer: RunScorer, clips_by_split: dict[str, list[Clip]]) -> dict:
    """Abstain curves for both fusion detectors on the seen and unseen test splits."""
    table: dict[str, dict[str, list[dict]]] = {}
    for split, clips in clips_by_split.items():
        logits = scorer.logits(clips)
        pv, pa = scorer.probabilities(logits)
        flags = scorer.flags(logits)
        labels = [clip.record.label for clip in clips]
        table[split] = {
            detector: abstain_curve(pv, pa, flags[detector], labels)
            for detector in FUSION_DETECTORS
        }
    return table


@dataclass
class RobustnessResult:
    results: dict
    summary: str
    run_dir: Path


def run_robustness(
    config: SpooflineConfig, run_dir: Path, progress: Progress = None
) -> RobustnessResult:
    """Score the bona fide test clips under every perturbation with a finished run.

    The seed, the held out families and the split membership are read from the run
    itself rather than from the profile, so a run trained with another seed or
    another pair held out is scored on exactly the clips it was evaluated on.
    """
    say = progress or (lambda _msg: None)
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "results.json").read_text())
    seed = int(run["seed"])
    config = replace(config, seed=seed, unseen_families=tuple(run["unseen_families"]))
    seed_everything(seed, config.threads)
    source = generate_corpus(config.corpus, seed, config.corpus_dir)
    splits = load_splits(run_dir / "splits.json")
    scorer = RunScorer.from_run(run_dir)
    clips_by_split = {
        split: [source.load(cid) for cid in splits.as_dict()[split]] for split in TEST_SPLITS
    }
    bonafide = [
        clip for split in TEST_SPLITS for clip in clips_by_split[split] if clip.record.label == 0
    ]
    say(f"robustness: {len(bonafide)} bona fide test clips, {len(PERTURBATIONS)} perturbations")
    results = {
        "profile": config.profile,
        # Recorded as the repository sees it (for example runs/full). report.py prints this
        # field, so an absolute path from the producing machine would be meaningless to a
        # reader of the committed artifact.
        "run_dir": str(Path(*Path(run_dir).resolve().parts[-2:])),
        "seed": seed,
        "unseen_families": list(config.unseen_families),
        "bonafide_clips": {
            split: sum(clip.record.label == 0 for clip in clips_by_split[split])
            for split in TEST_SPLITS
        },
        "false_alarms": false_alarm_table(scorer, bonafide, config.seed, say),
        "abstain": abstain_table(scorer, clips_by_split),
    }
    summary = render_robustness(results)
    run_dir = Path(run_dir)
    (run_dir / "robustness.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (run_dir / "robustness.txt").write_text(summary + "\n")
    return RobustnessResult(results=results, summary=summary, run_dir=run_dir)
