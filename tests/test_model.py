"""Model shapes, variable length packing and checkpoint round trips."""

from __future__ import annotations

import numpy as np
import torch

from spoofline.data.dataset import StreamDataset, collate
from spoofline.data.features import Normalizer
from spoofline.models.cnn_lstm import build_model, load_checkpoint, save_checkpoint
from spoofline.seeding import seed_everything


def _model(stream: str):
    seed_everything(11, 1)
    model = build_model(stream, 3 if stream == "video" else 1, 32, 32, 0.0)
    model.eval()
    return model


def test_video_stream_returns_one_logit_per_clip():
    model = _model("video")
    x = torch.zeros(5, 7, 3, 64, 64)
    out = model(x, torch.full((5,), 7))
    assert out.shape == (5,)


def test_audio_stream_returns_one_logit_per_clip():
    model = _model("audio")
    x = torch.zeros(3, 6, 1, 64, 20)
    out = model(x, torch.tensor([6, 4, 2]))
    assert out.shape == (3,)


def test_padding_does_not_change_a_short_sequence():
    model = _model("video")
    torch.manual_seed(3)
    x = torch.randn(3, 8, 3, 64, 64)
    lengths = torch.tensor([8, 5, 2])
    with torch.no_grad():
        baseline = model(x, lengths)
    polluted = x.clone()
    polluted[1, 5:] = 12.5
    polluted[2, 2:] = -9.0
    with torch.no_grad():
        after = model(polluted, lengths)
    assert torch.allclose(baseline, after, atol=1e-6)


def test_batching_matches_single_clip_scoring():
    model = _model("audio")
    torch.manual_seed(5)
    x = torch.randn(4, 9, 1, 64, 20)
    lengths = torch.tensor([9, 7, 5, 3])
    with torch.no_grad():
        batched = model(x, lengths)
        singles = torch.stack(
            [model(x[i : i + 1, : lengths[i]], lengths[i : i + 1])[0] for i in range(4)]
        )
    assert torch.allclose(batched, singles, atol=1e-5)


def test_collate_pads_variable_length_batches(corpus, splits):
    dataset = StreamDataset(corpus, splits.train, "audio")
    items = [dataset[i] for i in range(min(3, len(dataset)))]
    items[0]["x"] = items[0]["x"][:2]
    items[0]["length"] = 2
    batch = collate(items)
    assert batch["x"].shape[0] == len(items)
    assert int(batch["lengths"][0]) == 2
    assert torch.count_nonzero(batch["x"][0, 2:]) == 0


def test_checkpoint_round_trip_reproduces_scores(tmp_path):
    model = _model("video")
    normalizer = Normalizer(torch.zeros(3), torch.ones(3))
    torch.manual_seed(1)
    x = torch.randn(2, 4, 3, 64, 64)
    lengths = torch.tensor([4, 3])
    with torch.no_grad():
        before = model(normalizer.apply(x), lengths)
    path = tmp_path / "video.pt"
    save_checkpoint(path, model, normalizer, {"note": "round trip"})
    restored, restored_norm, extra = load_checkpoint(path)
    with torch.no_grad():
        after = restored(restored_norm.apply(x), lengths)
    assert torch.allclose(before, after, atol=1e-7)
    assert extra["note"] == "round trip"


def test_attention_pooling_ignores_masked_steps():
    from spoofline.models.cnn_lstm import AttentionPool

    torch.manual_seed(0)
    pool = AttentionPool(8)
    sequence = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        pooled = pool(sequence, mask)
        polluted = sequence.clone()
        polluted[0, 3:] = 100.0
        pooled_polluted = pool(polluted, mask)
    assert torch.allclose(pooled, pooled_polluted, atol=1e-6)


def test_model_is_deterministic_for_a_seed():
    first = _model("video")
    second = _model("video")
    x = torch.zeros(2, 3, 3, 64, 64)
    lengths = torch.tensor([3, 3])
    with torch.no_grad():
        assert torch.allclose(first(x, lengths), second(x, lengths))


def test_parameter_count_is_modest():
    model = build_model("video", 3, 96, 96, 0.1)
    total = sum(p.numel() for p in model.parameters())
    assert 50_000 < total < 2_000_000


def test_scores_change_when_the_input_changes():
    model = _model("audio")
    torch.manual_seed(2)
    a = torch.randn(1, 5, 1, 64, 20)
    b = torch.randn(1, 5, 1, 64, 20)
    with torch.no_grad():
        assert not np.isclose(
            float(model(a, torch.tensor([5]))), float(model(b, torch.tensor([5])))
        )
