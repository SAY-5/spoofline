"""Every figure the README quotes must match the committed artifacts of the run it quotes.

The blocks in README.md are pasted output. These tests re-render them from the JSON
committed under docs/runs/ and compare, so a stale table fails instead of drifting.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from spoofline.model_card import render_model_card
from spoofline.report import render_latency, render_robustness, render_summary, render_sweep

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
DEMO_RUN = REPO / "docs" / "runs" / "demo-full-seed20250117"
SWEEP_RUN = REPO / "docs" / "runs" / "sweep-full-default-pair"

DETECTOR_ROWS = {
    "video only": "video",
    "audio only": "audio",
    "fused": "fused",
    "weighted sum": "fused",
    "logistic": "logistic",
}
RULE_ROWS = {"AND rule": "and", "OR rule": "or"}
FUSION_NAMES = {"weighted sum": "fused", "logistic": "logistic"}


def _artifact(run: Path, name: str) -> dict:
    return json.loads((run / name).read_text())


def _fence_containing(needle: str) -> str:
    """The fenced block that contains ``needle``."""
    for match in re.finditer(r"```(?:\w+)?\n(.*?)```", README, re.S):
        if needle in match.group(1):
            return match.group(1)
    raise AssertionError(f"no fenced block contains {needle!r}")


def _tables(header: str) -> list[list[list[str]]]:
    """Every markdown table under ``header``, as rows of stripped cells."""
    lines = README.splitlines()
    tables = []
    for index, line in enumerate(lines):
        if line != header:
            continue
        rows = []
        for row in lines[index + 2 :]:
            if not row.startswith("|"):
                break
            rows.append([cell.strip() for cell in row.strip("|").split("|")])
        tables.append(rows)
    return tables


def _severity_label(row: dict) -> str:
    if row["severity"] is None:
        return "none"
    return f"{row['unit']} {row['severity']:g}"


def test_the_demo_block_is_the_rendered_summary_of_the_committed_run():
    block = _fence_containing("spoofline pipeline summary")
    rule = "=" * 78
    pasted = block[block.index(rule) :].rstrip("\n")
    assert pasted == render_summary(_artifact(DEMO_RUN, "results.json")).rstrip("\n")


def test_the_model_card_is_the_card_of_the_committed_run():
    card = render_model_card(
        _artifact(DEMO_RUN, "results.json"), _artifact(DEMO_RUN, "robustness.json")
    )
    assert (REPO / "docs" / "MODEL_CARD.md").read_text() == card


@pytest.mark.parametrize(
    ("header", "split"),
    [
        ("| unseen families | P | R | F1 | EER | AUC |", "unseen_test"),
        ("| seen families | P | R | F1 | EER | AUC |", "seen_test"),
    ],
)
def test_the_metric_tables_match_the_committed_run(header, split):
    results = _artifact(DEMO_RUN, "results.json")
    metrics, rules = results["metrics"][split], results["rules"][split]
    tables = _tables(header)
    assert tables, header
    checked = 0
    for rows in tables:
        for name, precision, recall, f1, eer, auc in rows:
            if name in DETECTOR_ROWS:
                point = metrics[DETECTOR_ROWS[name]]
                assert precision == f"{point['precision']:.3f}", (header, name)
                assert recall == f"{point['recall']:.3f}", (header, name)
                assert f1 == f"{point['f1']:.3f}", (header, name)
                assert eer == f"{point['eer']:.3f}", (header, name)
                assert auc == f"{point['auc']:.3f}", (header, name)
            elif name in RULE_ROWS:
                rule = rules[RULE_ROWS[name]]
                assert precision == f"{rule['precision']:.3f}", (header, name)
                assert recall == f"{rule['recall']:.3f}", (header, name)
                assert f1 == f"{rule['f1']:.3f}", (header, name)
                assert (eer, auc) == ("n/a", "n/a"), (header, name)
            else:
                continue
            checked += 1
    assert checked >= 6


def test_the_false_alarm_table_matches_the_committed_robustness_run():
    rows = {
        (row["perturbation"], _severity_label(row)): row
        for row in _artifact(DEMO_RUN, "robustness.json")["false_alarms"]
    }
    header = "| perturbation | stream | severity | video | audio | weighted sum | logistic |"
    tables = _tables(header)
    assert len(tables) == 1
    table = tables[0]
    assert len(table) == len(rows)
    for perturbation, stream, severity, video, audio, weighted, logistic in table:
        row = rows[(perturbation, severity)]
        assert stream == row["stream"], perturbation
        assert video == f"{row['video']:.3f}", (perturbation, severity)
        assert audio == f"{row['audio']:.3f}", (perturbation, severity)
        assert weighted == f"{row['fused']:.3f}", (perturbation, severity)
        assert logistic == f"{row['logistic']:.3f}", (perturbation, severity)


def test_the_abstain_table_matches_the_committed_robustness_run():
    abstain = _artifact(DEMO_RUN, "robustness.json")["abstain"]
    header = (
        "| split | fusion | margin | coverage | precision | recall | attacks abstained | "
        "bona fide abstained |"
    )
    tables = _tables(header)
    assert len(tables) == 1
    for split, fusion, margin, coverage, precision, recall, attacks, bona in tables[0]:
        rows = abstain[split][FUSION_NAMES[fusion]]
        row = next(r for r in rows if f"{r['margin']:.2f}" == margin)
        assert coverage == f"{row['coverage']:.3f}", (split, fusion, margin)
        assert precision == f"{row['precision']:.3f}", (split, fusion, margin)
        assert recall == f"{row['recall']:.3f}", (split, fusion, margin)
        assert attacks == str(row["abstained_attacks"]), (split, fusion, margin)
        assert bona == str(row["abstained_bonafide"]), (split, fusion, margin)


def test_the_robustness_block_headings_come_from_the_committed_run():
    rendered = render_robustness(_artifact(DEMO_RUN, "robustness.json"))
    assert (
        f"bona fide test clips  {sum(_artifact(DEMO_RUN, 'robustness.json')['bonafide_clips'].values())}"
        in rendered
    )
    assert "false alarm rate" in rendered


def test_the_latency_block_is_the_rendered_benchmark():
    pasted = _fence_containing("spoofline bench   clips=").rstrip("\n")
    assert pasted == render_latency(_artifact(DEMO_RUN, "latency.json")).rstrip("\n")


def test_the_export_parity_block_matches_the_committed_export():
    report = _artifact(DEMO_RUN, "export.json")
    pasted = _fence_containing("max |onnx - torch|")
    assert f"over {report['clips']} clips" in pasted
    for stream in ("video", "audio"):
        assert f"{report['parity'][stream]['max_abs_diff']:.2e}" in pasted


def test_the_full_profile_sweep_block_is_the_rendered_sweep():
    block = _fence_containing("spoofline sweep   profile=full")
    rule = "=" * 78
    pasted = block[block.index(rule) :].rstrip("\n")
    assert pasted == render_sweep(_artifact(SWEEP_RUN, "sweep.json")).rstrip("\n")
