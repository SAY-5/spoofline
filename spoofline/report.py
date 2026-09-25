"""Rendering of the pipeline summary block."""

from __future__ import annotations

RULE = "=" * 78
DETECTOR_ORDER = ("video", "audio", "fused", "logistic")
RULE_ORDER = ("and", "or", "weighted", "logistic")
ATTRIBUTION_ORDER = ("none", "video", "audio", "either", "joint")
SPLIT_ORDER = ("train", "calib", "seen_test", "unseen_test", "dropped")
POOL_ORDER = ("train", "calib", "test")
COMBO_ORDER = ("bonafide", "video_only", "audio_only", "both")
FUSION_ORDER = ("fused", "logistic")
TEST_SPLITS = ("seen_test", "unseen_test")
SWEEP_METRIC_ORDER = ("precision", "recall", "f1", "eer", "auc")
SWEEP_RULE_ORDER = ("and", "or")
SWEEP_RULE_METRIC_ORDER = ("precision", "recall", "f1")
TIMING_ORDER = (
    "generate",
    "load",
    "train_video",
    "train_audio",
    "calibrate",
    "evaluate",
    "total",
)


def ordered_keys(mapping: dict, order: tuple[str, ...] = ()) -> list[str]:
    """Keys of ``mapping`` in ``order`` first, then the rest alphabetically.

    The report must not inherit the key order of the dict it is handed: a run read
    back from ``results.json`` has its keys sorted, so rendering a committed run
    would otherwise print its rows in a different order from the run that wrote it.
    """
    known = [name for name in order if name in mapping]
    return known + sorted(name for name in mapping if name not in order)


def _counts_line(counts: dict[str, int], order: tuple[str, ...] = ()) -> str:
    return " | ".join(f"{name} {counts[name]}" for name in ordered_keys(counts, order))


def _fmt(value: float, digits: int = 3) -> str:
    if value != value:  # NaN
        return "  n/a"
    return f"{value:.{digits}f}"


def _wrap_verdict(text: str, label: str = "verdict") -> list[str]:
    """Break the verdict at the semicolon so the block stays inside 78 columns."""
    parts = [part.strip() for part in text.split(";")]
    return [f"  {label:<12}{parts[0]}"] + [f"              {part}" for part in parts[1:]]


def render_evaluation(results: dict) -> str:
    """The metrics, rule comparison and per family sections on their own."""
    lines = [
        "",
        "clip level metrics at the calibrated operating points",
        f"  {'split':<12}{'detector':<10}{'P':>7}{'R':>7}{'F1':>7}{'EER':>7}{'AUC':>7}"
        f"{'n':>6}{'attacks':>9}",
    ]
    for split in ("seen_test", "unseen_test"):
        for detector in DETECTOR_ORDER:
            point = results["metrics"][split][detector]
            lines.append(
                f"  {split:<12}{detector:<10}"
                f"{_fmt(point['precision']):>7}{_fmt(point['recall']):>7}{_fmt(point['f1']):>7}"
                f"{_fmt(point['eer']):>7}{_fmt(point['auc']):>7}"
                f"{point['n']:>6}{point['n_positive']:>9}"
            )

    lines += ["", "decision rule comparison at the same thresholds"]
    lines.append(f"  {'split':<12}{'rule':<10}{'P':>7}{'R':>7}{'F1':>7}")
    for split in ("seen_test", "unseen_test"):
        for rule in RULE_ORDER:
            entry = results["rules"][split][rule]
            lines.append(
                f"  {split:<12}{rule:<10}"
                f"{_fmt(entry['precision']):>7}{_fmt(entry['recall']):>7}{_fmt(entry['f1']):>7}"
            )

    lines += ["", "per family detection rate, fused detector at its calibrated threshold"]
    lines.append(f"  {'split':<12}{'family':<20}{'n':>6}{'detected':>10}{'rate':>8}")
    for split in TEST_SPLITS:
        rates = results["family_rates"][split]
        for family in ordered_keys(rates, ("bonafide",)):
            row = rates[family]
            lines.append(
                f"  {split:<12}{family:<20}{int(row['n']):>6}{int(row['detected']):>10}"
                f"{_fmt(row['rate']):>8}"
            )

    lines += [
        "",
        "which stream triggered each fusion decision, silencing one stream at a time",
        f"  {'split':<12}{'detector':<10}{'clips':<12}"
        + "".join(f"{name:>7}" for name in ATTRIBUTION_ORDER),
    ]
    for split in TEST_SPLITS:
        detectors = results["attribution"][split]
        for detector in ordered_keys(detectors, FUSION_ORDER):
            rows = detectors[detector]
            for combo in ordered_keys(rows, COMBO_ORDER):
                counts = rows[combo]
                lines.append(
                    f"  {split:<12}{detector:<10}{combo:<12}"
                    + "".join(f"{counts[name]:>7}" for name in ATTRIBUTION_ORDER)
                )

    return "\n".join(lines)


def render_summary(results: dict) -> str:
    """Render the block that `spoofline pipeline` prints and the README quotes."""
    corpus = results["corpus"]
    splits = results["splits"]
    lines: list[str] = [
        RULE,
        f"spoofline pipeline summary   profile={results['profile']}  seed={results['seed']}",
        RULE,
        f"corpus            {corpus['n_clips']} clips, {corpus['n_identities']} identities, "
        f"{corpus['n_frames']} frames of {corpus['frame_size']}x{corpus['frame_size']}, "
        f"{corpus['duration_s']} s at {corpus['sample_rate']} Hz",
        f"combinations      {_counts_line(results['combo_counts'])}",
    ]
    video_families = {
        k.split(":", 1)[1]: v for k, v in results["family_counts"].items() if k.startswith("video:")
    }
    audio_families = {
        k.split(":", 1)[1]: v for k, v in results["family_counts"].items() if k.startswith("audio:")
    }
    lines += [
        f"video families    {_counts_line(video_families)}",
        f"audio families    {_counts_line(audio_families)}",
        f"unseen families   {', '.join(results['unseen_families'])}",
        f"splits            {_counts_line(splits['counts'], SPLIT_ORDER)}",
        f"identity pools    {_counts_line(splits['identity_pools'], POOL_ORDER)}",
        "",
        "training",
    ]
    for stream in ("video", "audio"):
        training = results["training"][stream]
        final = training["final"]
        lines.append(
            f"  {stream:<6} {training['epochs']} epochs, kept epoch {training['best_epoch']}  "
            f"train_loss {final['train_loss']:.4f} acc {final['train_acc']:.3f}  |  "
            f"val_loss {final['val_loss']:.4f} acc {final['val_acc']:.3f}  "
            f"({training['train_clips']} train / {training['val_clips']} val clips)"
        )

    calibration = results["calibration"]
    lines += [
        "",
        f"calibration on the calib split, target precision {results['target_precision']:.2f}",
    ]
    for stream in ("video", "audio"):
        entry = calibration[stream]
        operating = entry["operating"]
        platt = entry["calibrator"]
        lines.append(
            f"  {stream:<6} platt a={platt['a']:+.3f} b={platt['b']:+.3f}"
            f"   threshold {operating['threshold']:.4f}"
            f"   calib precision {operating['achieved_precision']:.3f}"
            f"   recall {operating['recall']:.3f}"
            f"   target met {str(operating['reached_target']).lower()}"
        )
    fused = calibration["fused"]
    lines.append(
        f"  {'fused':<6} weight {fused['weight']:.2f} on video"
        f"           threshold {fused['operating']['threshold']:.4f}"
        f"   calib precision {fused['operating']['achieved_precision']:.3f}"
        f"   recall {fused['operating']['recall']:.3f}"
        f"   target met {str(fused['operating']['reached_target']).lower()}"
    )
    logistic = calibration["logistic"]
    coefficients = logistic["coefficients"]
    lines += [
        f"  {'logistic':<8} p_video {coefficients['p_video']:+.3f}"
        f"  p_audio {coefficients['p_audio']:+.3f}"
        f"  disagreement {coefficients['disagreement']:+.3f}"
        f"  intercept {logistic['intercept']:+.3f}",
        f"           threshold {logistic['operating']['threshold']:.4f}"
        f"   calib precision {logistic['operating']['achieved_precision']:.3f}"
        f"   recall {logistic['operating']['recall']:.3f}"
        f"   target met {str(logistic['operating']['reached_target']).lower()}",
    ]

    lines += render_evaluation(results).split("\n")

    headline = results["headline"]
    lines += [
        "",
        "headline, unseen attack families",
        f"  precision   fused {_fmt(headline['fused_precision'])}"
        f"   logistic {_fmt(headline['logistic_precision'])}"
        f"   video {_fmt(headline['video_precision'])}"
        f"   audio {_fmt(headline['audio_precision'])}",
        f"  recall      fused {_fmt(headline['fused_recall'])}"
        f"   logistic {_fmt(headline['logistic_recall'])}"
        f"   video {_fmt(headline['video_recall'])}"
        f"   audio {_fmt(headline['audio_recall'])}",
        f"  gap         precision minus best single stream: fused "
        f"{headline['fused_precision_gap']:+.3f}   logistic "
        f"{headline['logistic_precision_gap']:+.3f}",
        *_wrap_verdict(headline["verdict"]),
        *_wrap_verdict(headline["logistic_verdict"], label="logistic"),
        "",
        "wall clock",
    ]
    timings = results["timings"]
    for name in ordered_keys(timings, TIMING_ORDER):
        lines.append(f"  {name:<16}{timings[name]:8.1f}s")
    lines.append(RULE)
    return "\n".join(lines)


DERIVED_LABELS = {
    "unseen_precision_gap": "unseen fused precision minus best stream",
    "logistic_unseen_precision_gap": "unseen logistic precision minus best stream",
    "seen_to_unseen_precision_drop": "seen minus unseen fused precision",
    "logistic_seen_to_unseen_precision_drop": "seen minus unseen logistic precision",
}


def _summary_cells(entry: dict) -> str:
    return (
        f"{_fmt(entry['mean']):>7}{_fmt(entry['std']):>7}"
        f"{_fmt(entry['ci_low']):>9}{_fmt(entry['ci_high']):>9}"
    )


def render_sweep(results: dict) -> str:
    """Render the variance table that `spoofline sweep` prints."""
    settings = results["profile_settings"]
    corpus = settings["corpus"]
    aggregate = results["aggregate"]
    lines = [
        RULE,
        f"spoofline sweep   profile={results['profile']}   seeds={len(results['seeds'])}"
        f"   held out pairs={len(results['pairs'])}   runs={aggregate['n_runs']}",
        RULE,
        f"corpus            {corpus['n_clips']} clips, {corpus['n_identities']} identities, "
        f"{corpus['n_frames']} frames of {corpus['frame_size']}x{corpus['frame_size']}, "
        f"{corpus['duration_s']} s at {corpus['sample_rate']} Hz",
        f"training          video {settings['video_epochs']} epochs, audio "
        f"{settings['audio_epochs']} epochs, batch {settings['batch_size']}, "
        f"{settings['threads_per_run']} threads per run",
        f"seeds             {', '.join(str(seed) for seed in results['seeds'])}",
        f"target precision  {results['target_precision']:.2f} on the calib split of every run",
        "",
        "mean, sample std and bootstrap 95% interval of the mean over runs",
        f"  {'split':<12}{'detector':<10}{'metric':<11}{'mean':>7}{'std':>7}"
        f"{'ci low':>9}{'ci high':>9}",
    ]
    for split in ordered_keys(aggregate["metrics"], TEST_SPLITS):
        detectors = aggregate["metrics"][split]
        for detector in ordered_keys(detectors, DETECTOR_ORDER):
            metrics = detectors[detector]
            for metric in ordered_keys(metrics, SWEEP_METRIC_ORDER):
                cells = _summary_cells(metrics[metric])
                lines.append(f"  {split:<12}{detector:<10}{metric:<11}{cells}")

    lines += [
        "",
        "decision rules at the single stream thresholds, mean over runs",
        f"  {'split':<12}{'rule':<10}{'metric':<11}{'mean':>7}{'std':>7}"
        f"{'ci low':>9}{'ci high':>9}",
    ]
    for split in ordered_keys(aggregate["rules"], TEST_SPLITS):
        rules = aggregate["rules"][split]
        for rule in ordered_keys(rules, SWEEP_RULE_ORDER):
            metrics = rules[rule]
            for metric in ordered_keys(metrics, SWEEP_RULE_METRIC_ORDER):
                cells = _summary_cells(metrics[metric])
                lines.append(f"  {split:<12}{rule:<10}{metric:<11}{cells}")

    lines += ["", "derived, per run then summarised"]
    for name in ordered_keys(aggregate["derived"], tuple(DERIVED_LABELS)):
        entry = aggregate["derived"][name]
        lines.append(f"  {DERIVED_LABELS.get(name, name):<48}{_summary_cells(entry)}")

    lines += [
        "",
        "unseen test precision and recall per held out pair, mean over seeds",
        f"  {'video family':<18}{'audio family':<18}{'video P':>8}{'audio P':>8}"
        f"{'fused P':>8}{'logis P':>8}{'fused R':>8}",
    ]
    for row in results["per_pair"]:
        video_family, audio_family = row["pair"]
        lines.append(
            f"  {video_family:<18}{audio_family:<18}{_fmt(row['video_precision']):>8}"
            f"{_fmt(row['audio_precision']):>8}{_fmt(row['fused_precision']):>8}"
            f"{_fmt(row['logistic_precision']):>8}{_fmt(row['fused_recall']):>8}"
        )
    lines += ["", f"wall clock        {results['wall_clock_s']:.1f}s", RULE]
    return "\n".join(lines)


def _severity_label(row: dict) -> str:
    if row["severity"] is None:
        return "none"
    value = row["severity"]
    number = f"{value:g}"
    return f"{row['unit']} {number}"


def render_robustness(results: dict) -> str:
    """Render the false alarm and abstain tables that `spoofline robustness` prints."""
    counts = results["bonafide_clips"]
    detectors = ("video", "audio", "fused", "logistic")
    lines = [
        RULE,
        f"spoofline robustness   profile={results['profile']}   run={results['run_dir']}",
        RULE,
        f"bona fide test clips  {sum(counts.values())} ({_counts_line(counts)}), "
        "each degraded, label kept",
        "",
        "false alarm rate at the calibrated thresholds, per perturbation and severity",
        f"  {'perturbation':<21}{'stream':<7}{'severity':<14}"
        + "".join(f"{name:>9}" for name in detectors),
    ]
    for row in results["false_alarms"]:
        lines.append(
            f"  {row['perturbation']:<21}{row['stream']:<7}{_severity_label(row):<14}"
            + "".join(f"{_fmt(row[name]):>9}" for name in detectors)
        )
    lines += [
        "",
        "abstain when |p_video - p_audio| exceeds the margin, coverage against precision",
        f"  {'split':<12}{'detector':<10}{'margin':>7}{'coverage':>10}{'P':>7}{'R':>7}"
        f"{'abst att':>10}{'abst bona':>10}",
    ]
    for split, by_detector in results["abstain"].items():
        for detector, rows in by_detector.items():
            for row in rows:
                lines.append(
                    f"  {split:<12}{detector:<10}{row['margin']:>7.2f}{_fmt(row['coverage']):>10}"
                    f"{_fmt(row['precision']):>7}{_fmt(row['recall']):>7}"
                    f"{row['abstained_attacks']:>10}{row['abstained_bonafide']:>10}"
                )
    lines.append(RULE)
    return "\n".join(lines)


def render_latency(results: dict) -> str:
    """Render the per clip latency table that `spoofline bench` prints."""
    lines = [
        RULE,
        f"spoofline bench   clips={results['clips']}   threads={results['threads']}   CPU",
        RULE,
        "per clip wall time; a stream includes its feature extraction, end to end adds",
        "npz decode, calibration, both fusions and attribution",
        f"  {'engine':<8}{'stage':<12}{'p50 ms':>9}{'p95 ms':>9}{'mean ms':>9}{'n':>6}",
    ]
    for row in results["rows"]:
        lines.append(
            f"  {row['engine']:<8}{row['stage']:<12}{row['p50_ms']:>9.2f}{row['p95_ms']:>9.2f}"
            f"{row['mean_ms']:>9.2f}{row['n']:>6}"
        )
    lines.append(RULE)
    return "\n".join(lines)
