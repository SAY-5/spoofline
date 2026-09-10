"""Training plumbing on the tiny fixture corpus."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from spoofline.config import StreamTrainConfig
from spoofline.train import score_clips, split_train_val, train_stream


def _fast(config):
    tiny = StreamTrainConfig(epochs=1, batch_size=4, hidden_size=16, embed_size=16)
    return replace(config, video=tiny, audio=tiny)


def test_validation_split_is_identity_disjoint(corpus, splits, config):
    train_ids, val_ids = split_train_val(corpus, splits.train, 0.25, config.seed, "video")
    train_identities = {corpus.record(c).identity for c in train_ids}
    val_identities = {corpus.record(c).identity for c in val_ids}
    assert not train_identities & val_identities
    assert set(train_ids) | set(val_ids) == set(splits.train)


def test_train_stream_reports_one_history_row_per_epoch(corpus, splits, config):
    trained = train_stream(corpus, splits, "video", _fast(config))
    assert len(trained.history) == 1
    assert set(trained.history[0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    assert np.isfinite(trained.history[0]["train_loss"])


def test_training_is_deterministic_for_a_seed(corpus, splits, config):
    fast = _fast(config)
    first = train_stream(corpus, splits, "audio", fast)
    second = train_stream(corpus, splits, "audio", fast)
    ids = splits.calib
    a = score_clips(first.model, first.normalizer, corpus, ids, "audio")
    b = score_clips(second.model, second.normalizer, corpus, ids, "audio")
    assert np.allclose(a, b, atol=1e-9)


def test_scores_come_back_in_the_order_requested(corpus, splits, config):
    trained = train_stream(corpus, splits, "video", _fast(config))
    ids = list(splits.seen_test)
    forward = score_clips(trained.model, trained.normalizer, corpus, ids, "video")
    backward = score_clips(trained.model, trained.normalizer, corpus, ids[::-1], "video")
    assert np.allclose(forward, backward[::-1], atol=1e-9)


def test_stream_labels_use_the_modality_not_the_clip_label(corpus, splits):
    from spoofline.data.dataset import StreamDataset

    dataset = StreamDataset(corpus, splits.train, "video")
    for clip_id in splits.train:
        record = corpus.record(clip_id)
        assert dataset.modality_label(record) == float(record.video_attacked)
        if record.audio_attacked and not record.video_attacked:
            assert record.label == 1
            assert dataset.modality_label(record) == 0.0
