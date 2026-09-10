"""The leave-one-attack-family-out protocol must not leak."""

from __future__ import annotations

from spoofline.data.dataset import SPLIT_NAMES, make_splits
from spoofline.families import BONAFIDE


def test_splits_are_disjoint_by_clip_id(splits):
    seen = set()
    for name in SPLIT_NAMES:
        ids = set(getattr(splits, name))
        assert not (ids & seen), f"{name} overlaps an earlier split"
        seen |= ids


def test_identity_pools_are_disjoint(splits):
    train, calib, test = (set(splits.identity_pools[k]) for k in ("train", "calib", "test"))
    assert not train & calib
    assert not train & test
    assert not calib & test


def test_splits_are_disjoint_by_identity(source, splits):
    by_id = {r.clip_id: r.identity for r in source.records}
    identities = {name: {by_id[c] for c in getattr(splits, name)} for name in SPLIT_NAMES}
    assert not identities["train"] & identities["calib"]
    assert not identities["train"] & identities["seen_test"]
    assert not identities["train"] & identities["unseen_test"]
    assert not identities["calib"] & identities["seen_test"]
    assert not identities["calib"] & identities["unseen_test"]


def test_no_unseen_family_reaches_train_or_calib(source, splits):
    by_id = {r.clip_id: r for r in source.records}
    for name in ("train", "calib"):
        for clip_id in getattr(splits, name):
            assert not set(by_id[clip_id].families) & set(splits.unseen_families)


def test_unseen_test_only_holds_unseen_attacks_and_bona_fide(source, splits):
    by_id = {r.clip_id: r for r in source.records}
    for clip_id in splits.unseen_test:
        record = by_id[clip_id]
        if record.label == 1:
            assert set(record.families) & set(splits.unseen_families)
        else:
            assert record.video_family == BONAFIDE and record.audio_family == BONAFIDE


def test_seen_test_never_holds_an_unseen_family(source, splits):
    by_id = {r.clip_id: r for r in source.records}
    for clip_id in splits.seen_test:
        assert not set(by_id[clip_id].families) & set(splits.unseen_families)


def test_every_split_is_populated(splits):
    for name in SPLIT_NAMES:
        assert len(getattr(splits, name)) > 0, f"{name} is empty"


def test_splits_are_deterministic(source, config):
    args = (source.records, config.unseen_families, config.seed)
    first = make_splits(*args, config.train_fraction, config.calib_fraction)
    second = make_splits(*args, config.train_fraction, config.calib_fraction)
    assert first.as_dict() == second.as_dict()


def test_changing_the_unseen_family_moves_clips(source, config):
    default = make_splits(
        source.records,
        ("video_splice", "audio_vocoder"),
        config.seed,
        config.train_fraction,
        config.calib_fraction,
    )
    other = make_splits(
        source.records,
        ("video_replay", "audio_replay"),
        config.seed,
        config.train_fraction,
        config.calib_fraction,
    )
    assert set(default.unseen_test) != set(other.unseen_test)
