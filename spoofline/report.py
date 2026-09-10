"""Rendering of the pipeline summary block."""

from __future__ import annotations

RULE = "=" * 78
DETECTOR_ORDER = ("video", "audio", "fused")


def _counts_line(counts: dict[str, int]) -> str:
    return " | ".join(f"{name} {value}" for name, value in counts.items())


def _fmt(value: float, digits: int = 3) -> str:
    if value != value:  # NaN
        return "  n/a"
    return f"{value:.{digits}f}"


def _wrap_verdict(text: str) -> list[str]:
    """Break the verdict at the semicolon so the block stays inside 78 columns."""
    parts = [part.strip() for part in text.split(";")]
    return [f"  verdict     {parts[0]}"] + [f"              {part}" for part in parts[1:]]


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
        for rule in ("and", "or", "weighted"):
            entry = results["rules"][split][rule]
            lines.append(
                f"  {split:<12}{rule:<10}"
                f"{_fmt(entry['precision']):>7}{_fmt(entry['recall']):>7}{_fmt(entry['f1']):>7}"
            )

    lines += ["", "per family detection rate, fused detector at its calibrated threshold"]
    lines.append(f"  {'split':<12}{'family':<20}{'n':>6}{'detected':>10}{'rate':>8}")
    for split in ("seen_test", "unseen_test"):
        for family, row in results["family_rates"][split].items():
            lines.append(
                f"  {split:<12}{family:<20}{int(row['n']):>6}{int(row['detected']):>10}"
                f"{_fmt(row['rate']):>8}"
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
        f"splits            {_counts_line(splits['counts'])}",
        f"identity pools    {_counts_line(splits['identity_pools'])}",
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

    lines += render_evaluation(results).split("\n")

    headline = results["headline"]
    lines += [
        "",
        "headline, unseen attack families",
        f"  precision   fused {_fmt(headline['fused_precision'])}"
        f"   video {_fmt(headline['video_precision'])}"
        f"   audio {_fmt(headline['audio_precision'])}",
        f"  recall      fused {_fmt(headline['fused_recall'])}"
        f"   video {_fmt(headline['video_recall'])}"
        f"   audio {_fmt(headline['audio_recall'])}",
        *_wrap_verdict(headline["verdict"]),
        "",
        "wall clock",
    ]
    timings = results["timings"]
    for name, seconds in timings.items():
        lines.append(f"  {name:<16}{seconds:8.1f}s")
    lines.append(RULE)
    return "\n".join(lines)
