"""The shared CNN-LSTM detector used by both streams.

Per step a small CNN encodes one input (an RGB frame for the video stream, a mel
patch for the audio stream), a bidirectional LSTM runs over the step sequence
with packing so padded batches are handled exactly, and attention pooling
collapses the sequence to a single score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..data.features import Normalizer


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to rebuild the network."""

    stream: str
    in_channels: int
    embed_size: int = 96
    hidden_size: int = 96
    dropout: float = 0.1
    channels: tuple[int, ...] = (24, 40, 64, 64)

    def as_dict(self) -> dict:
        return asdict(self)


class StepEncoder(nn.Module):
    """Small strided CNN applied to every step independently."""

    def __init__(self, in_channels: int, channels: tuple[int, ...], embed_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for width in channels:
            layers += [
                nn.Conv2d(prev, width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
            prev = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(prev, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.project(h)


class AttentionPool(nn.Module):
    """Masked additive attention over the LSTM outputs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(sequence).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (sequence * weights).sum(dim=1)


class CnnLstmDetector(nn.Module):
    """CNN encoder, temporal LSTM, attention pooling, single logit head."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = StepEncoder(spec.in_channels, spec.channels, spec.embed_size)
        self.lstm = nn.LSTM(
            input_size=spec.embed_size,
            hidden_size=spec.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionPool(2 * spec.hidden_size)
        self.dropout = nn.Dropout(spec.dropout)
        self.head = nn.Linear(2 * spec.hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, steps = x.shape[0], x.shape[1]
        flat = x.reshape(batch * steps, *x.shape[2:])
        embedded = self.encoder(flat).view(batch, steps, -1)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        sequence, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=steps)

        mask = torch.arange(steps, device=x.device)[None, :] < lengths[:, None].to(x.device)
        pooled = self.attention(sequence, mask)
        return self.head(self.dropout(pooled)).squeeze(-1)


def build_model(stream: str, in_channels: int, embed_size: int, hidden_size: int, dropout: float):
    return CnnLstmDetector(
        ModelSpec(
            stream=stream,
            in_channels=in_channels,
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    )


def save_checkpoint(
    path: Path, model: CnnLstmDetector, normalizer: Normalizer, extra: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "spec": model.spec.as_dict(),
            "state_dict": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[CnnLstmDetector, Normalizer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec_fields = dict(payload["spec"])
    spec_fields["channels"] = tuple(spec_fields["channels"])
    model = CnnLstmDetector(ModelSpec(**spec_fields))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, Normalizer.from_state_dict(payload["normalizer"]), payload.get("extra", {})
