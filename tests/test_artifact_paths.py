"""Committed run artifacts must not carry paths from the machine that produced them."""

import json
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "docs" / "runs"


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_committed_artifacts_carry_no_absolute_paths():
    offenders = []
    for path in sorted(ARTIFACT_DIR.rglob("*.json")):
        for text in _strings(json.loads(path.read_text())):
            if text.startswith("/") or text.startswith("~"):
                offenders.append(f"{path.relative_to(ARTIFACT_DIR.parent.parent)}: {text}")
    assert offenders == [], "absolute paths in committed artifacts: " + "; ".join(offenders)
