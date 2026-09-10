"""The same seed must reproduce the same summary on the tiny fixture corpus."""

from __future__ import annotations

import json
from dataclasses import replace

from spoofline.config import FIXTURE_DIR, profile
from spoofline.pipeline import run_pipeline


def _tiny(run_dir):
    config = profile("tiny")
    return replace(config, corpus_dir=FIXTURE_DIR, run_dir=run_dir)


def _strip_timings(text: str) -> str:
    lines = text.split("\n")
    cut = lines.index("wall clock")
    return "\n".join(lines[:cut])


def test_same_seed_reproduces_the_same_summary(tmp_path):
    first = run_pipeline(_tiny(tmp_path / "a"))
    second = run_pipeline(_tiny(tmp_path / "b"))
    assert _strip_timings(first.summary) == _strip_timings(second.summary)


def test_same_seed_reproduces_the_same_numbers(tmp_path):
    first = run_pipeline(_tiny(tmp_path / "a"))
    second = run_pipeline(_tiny(tmp_path / "b"))
    for key in ("metrics", "calibration", "rules", "family_rates", "headline", "training"):
        assert json.dumps(first.results[key], sort_keys=True) == json.dumps(
            second.results[key], sort_keys=True
        )


def test_a_different_seed_changes_the_result(tmp_path):
    first = run_pipeline(_tiny(tmp_path / "a"))
    other = replace(
        profile("tiny"),
        seed=profile("tiny").seed + 1,
        corpus_dir=tmp_path / "corpus",
        run_dir=tmp_path / "c",
    )
    second = run_pipeline(other)
    assert first.results["calibration"] != second.results["calibration"]


def test_generating_over_a_corpus_from_another_seed_is_refused(tmp_path):
    import pytest

    from spoofline.data.generate import generate_corpus

    config = profile("tiny")
    generate_corpus(config.corpus, config.seed, tmp_path / "corpus")
    with pytest.raises(ValueError, match="already holds a corpus"):
        generate_corpus(config.corpus, config.seed + 1, tmp_path / "corpus")


def test_summary_contains_the_sections_the_readme_quotes(tmp_path):
    result = run_pipeline(_tiny(tmp_path / "a"))
    for heading in (
        "spoofline pipeline summary",
        "training",
        "calibration on the calib split",
        "clip level metrics at the calibrated operating points",
        "decision rule comparison",
        "per family detection rate",
        "headline, unseen attack families",
        "wall clock",
    ):
        assert heading in result.summary
