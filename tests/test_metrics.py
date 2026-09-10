"""Metric implementations checked against values computed by hand."""

from __future__ import annotations

import math

import numpy as np

from spoofline.metrics import (
    counts_at,
    equal_error_rate,
    evaluate,
    family_breakdown,
    precision_recall_f1,
    roc_auc,
)


def test_counts_and_prf_match_hand_computation():
    scores = [0.9, 0.8, 0.4, 0.2]
    labels = [1, 0, 1, 0]
    counts = counts_at(scores, labels, 0.5)
    assert (counts.tp, counts.fp, counts.tn, counts.fn) == (1, 1, 1, 1)
    precision, recall, f1 = precision_recall_f1(counts)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_auc_matches_the_textbook_example():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75


def test_auc_of_a_perfect_ranking_is_one():
    assert roc_auc([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1]) == 1.0


def test_auc_with_all_ties_is_a_half():
    assert roc_auc([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1]) == 0.5


def test_auc_is_nan_without_both_classes():
    assert math.isnan(roc_auc([0.1, 0.2], [1, 1]))


def test_eer_of_a_separable_set_is_zero():
    eer, threshold = equal_error_rate([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])
    assert eer == 0.0
    assert threshold == 2.0


def test_eer_matches_a_hand_built_quarter_error_case():
    scores = [0.9, 0.8, 0.7, 0.2, 0.6, 0.5, 0.4, 0.95]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    eer, threshold = equal_error_rate(scores, labels)
    assert eer == 0.25
    assert threshold == 0.7


def test_evaluate_bundles_the_operating_point():
    point = evaluate([0.9, 0.8, 0.4, 0.2], [1, 0, 1, 0], 0.5)
    assert point.threshold == 0.5
    assert point.n == 4
    assert point.n_positive == 2
    assert point.counts.tp == 1


def test_family_breakdown_counts_each_family_once_per_clip():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 1, 1, 0])
    families = [("video_replay",), ("audio_replay",), ("video_replay", "audio_replay"), ()]
    rows = family_breakdown(scores, labels, families, 0.5)
    assert rows["video_replay"]["n"] == 2
    assert rows["video_replay"]["detected"] == 2
    assert rows["audio_replay"]["n"] == 2
    assert rows["audio_replay"]["rate"] == 0.5
    assert rows["bonafide"]["n"] == 1
    assert rows["bonafide"]["rate"] == 0.0


def test_precision_is_zero_when_nothing_is_flagged():
    point = evaluate([0.1, 0.2], [1, 0], 0.9)
    assert point.precision == 0.0
    assert point.recall == 0.0
    assert point.f1 == 0.0
