"""Training and scoring for one stream."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import SpooflineConfig, StreamTrainConfig
from .data.dataset import LoadedCorpus, Splits, StreamDataset, collate
from .data.features import Normalizer
from .models.cnn_lstm import CnnLstmDetector, build_model
from .seeding import derive_seed
from .seeding import rng as make_rng

Progress = Callable[[str], None] | None


@dataclass
class TrainedStream:
    """A trained stream plus the bookkeeping the report needs."""

    stream: str
    model: CnnLstmDetector
    normalizer: Normalizer
    history: list[dict[str, float]]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    best_epoch: int = 0

    @property
    def final(self) -> dict[str, float]:
        """The epoch whose weights were kept, which is the best validation loss."""
        if not self.history:
            return {}
        return self.history[self.best_epoch - 1] if self.best_epoch else self.history[-1]


def split_train_val(
    corpus: LoadedCorpus, clip_ids: Sequence[str], val_fraction: float, seed: int, stream: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Hold out whole identities from the training pool to monitor validation loss."""
    identities = sorted({corpus.record(cid).identity for cid in clip_ids})
    n_val = max(1, round(len(identities) * val_fraction)) if len(identities) > 1 else 0
    order = make_rng(seed, f"val-split/{stream}").permutation(len(identities))
    val_identities = {identities[i] for i in order[:n_val]}
    train_ids = tuple(cid for cid in clip_ids if corpus.record(cid).identity not in val_identities)
    val_ids = tuple(cid for cid in clip_ids if corpus.record(cid).identity in val_identities)
    if not train_ids:  # pragma: no cover - only possible for degenerate corpora
        return tuple(clip_ids), ()
    return train_ids, val_ids


def _loader(dataset: StreamDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _epoch_loss(
    model: CnnLstmDetector,
    normalizer: Normalizer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total, count, correct = 0.0, 0, 0
    for batch in loader:
        x = normalizer.apply(batch["x"])
        with torch.set_grad_enabled(training):
            logits = model(x, batch["lengths"])
            loss = criterion(logits, batch["y"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        size = len(batch["y"])
        total += float(loss.detach()) * size
        count += size
        correct += int(((logits.detach() > 0).float() == batch["y"]).sum())
    if count == 0:  # pragma: no cover - empty validation pool
        return float("nan"), float("nan")
    return total / count, correct / count


def train_stream(
    corpus: LoadedCorpus,
    splits: Splits,
    stream: str,
    config: SpooflineConfig,
    progress: Progress = None,
) -> TrainedStream:
    """Train one stream on its own modality label."""
    stream_config: StreamTrainConfig = config.video if stream == "video" else config.audio
    torch.manual_seed(derive_seed(config.seed, f"train/{stream}"))

    train_ids, val_ids = split_train_val(
        corpus, splits.train, stream_config.val_fraction, config.seed, stream
    )
    train_set = StreamDataset(corpus, train_ids, stream)
    val_set = StreamDataset(corpus, val_ids, stream)
    normalizer = train_set.fit_normalizer()

    in_channels = 3 if stream == "video" else 1
    model = build_model(
        stream,
        in_channels,
        stream_config.embed_size,
        stream_config.hidden_size,
        stream_config.dropout,
    )

    labels = np.array([train_set.modality_label(corpus.record(c)) for c in train_ids])
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    pos_weight = torch.tensor([negatives / positives if positives else 1.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=stream_config.lr, weight_decay=stream_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, stream_config.epochs)
    )

    train_loader = _loader(
        train_set, stream_config.batch_size, True, derive_seed(config.seed, f"loader/{stream}")
    )
    val_loader = _loader(val_set, stream_config.batch_size, False, 0) if val_ids else None

    if progress:
        progress(
            f"  {stream}: {len(train_ids)} train clips, {len(val_ids)} val clips, "
            f"{int(positives)} attacked, {stream_config.epochs} epochs"
        )

    history: list[dict[str, float]] = []
    best_epoch, best_val = 0, float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, stream_config.epochs + 1):
        train_loss, train_acc = _epoch_loss(model, normalizer, train_loader, criterion, optimizer)
        if val_loader is not None:
            val_loss, val_acc = _epoch_loss(model, normalizer, val_loader, criterion, None)
        else:  # pragma: no cover - degenerate corpora only
            val_loss, val_acc = float("nan"), float("nan")
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        # Keep the weights from the best validation epoch rather than the last one.
        score = val_loss if val_loader is not None and np.isfinite(val_loss) else train_loss
        if score < best_val:
            best_val, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if progress:
            progress(
                f"    epoch {epoch}/{stream_config.epochs} "
                f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val_loss {val_loss:.4f} acc {val_acc:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        if progress:
            progress(f"    kept epoch {best_epoch} (best validation loss {best_val:.4f})")
    model.eval()
    return TrainedStream(
        stream=stream,
        model=model,
        normalizer=normalizer,
        history=history,
        train_ids=tuple(train_ids),
        val_ids=tuple(val_ids),
        best_epoch=best_epoch,
    )


def score_clips(
    model: CnnLstmDetector,
    normalizer: Normalizer,
    corpus: LoadedCorpus,
    clip_ids: Sequence[str],
    stream: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Raw logits for a list of clips, in the order given."""
    dataset = StreamDataset(corpus, clip_ids, stream)
    loader = _loader(dataset, batch_size, False, 0)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(normalizer.apply(batch["x"]), batch["lengths"])
            scores.extend(logits.tolist())
    return np.asarray(scores, dtype=np.float64)
