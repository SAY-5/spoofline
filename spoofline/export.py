"""ONNX export of both streams, checked against PyTorch on real clips."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data.features import Normalizer
from .models.cnn_lstm import CnnLstmDetector
from .scoring import STREAMS, RunScorer, batch_logits, clip_steps, read_npz_clip

PARITY_TOLERANCE = 1e-4


class ExportableDetector(nn.Module):
    """The detector with its normaliser folded in, for full length step sequences.

    Every clip a deployment scores has the same number of steps, so a batch holds no
    padding, packing is the identity and the LSTM runs on the dense tensor, which is
    what ONNX can express. The attention mask is all true for the same reason.
    """

    def __init__(self, model: CnnLstmDetector, normalizer: Normalizer) -> None:
        super().__init__()
        self.model = model.eval()
        channels = normalizer.mean.numel()
        self.register_buffer("mean", normalizer.mean.reshape(1, 1, channels, 1, 1).clone())
        self.register_buffer("std", normalizer.std.reshape(1, 1, channels, 1, 1).clone())

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        x = (steps - self.mean) / self.std
        batch, length = x.shape[0], x.shape[1]
        embedded = self.model.encoder(x.reshape(batch * length, *x.shape[2:]))
        sequence, _ = self.model.lstm(embedded.reshape(batch, length, -1))
        weights = torch.softmax(self.model.attention.score(sequence).squeeze(-1), dim=1)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return self.model.head(pooled).squeeze(-1)


def export_stream(
    model: CnnLstmDetector, normalizer: Normalizer, example: np.ndarray, path: Path
) -> Path:
    """Write one stream as ONNX with a dynamic batch and step axis.

    The trace uses a batch of two copies of the example, because an axis of size one
    would be specialised to a constant.
    """
    wrapper = ExportableDetector(model, normalizer).eval()
    sample = torch.from_numpy(np.ascontiguousarray(np.stack([example, example]), dtype=np.float32))
    batch, steps = torch.export.Dim("batch", min=1), torch.export.Dim("steps", min=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample,),
        str(path),
        input_names=["steps"],
        output_names=["logit"],
        dynamic_shapes={"steps": {0: batch, 1: steps}},
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    return path


def onnx_logits(path: Path, steps: Sequence[np.ndarray]) -> np.ndarray:
    """Logits from an exported stream for equal length step sequences."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = np.stack(steps).astype(np.float32)
    return session.run(["logit"], {"steps": batch})[0].astype(np.float64)


def export_run(
    run_dir: Path, out_dir: Path, clip_paths: Sequence[Path], sample_rate: int = 16000
) -> dict:
    """Export both streams of a run and check their parity with PyTorch on real clips.

    Raises if any clip's ONNX logit differs from the PyTorch logit by more than
    ``PARITY_TOLERANCE``.
    """
    scorer = RunScorer.from_run(run_dir)
    clips = [read_npz_clip(path, sample_rate) for path in clip_paths]
    report: dict = {"tolerance": PARITY_TOLERANCE, "clips": len(clips), "files": {}, "parity": {}}
    for stream in STREAMS:
        model, normalizer = scorer.models[stream]
        steps = [clip_steps(stream, clip) for clip in clips]
        path = export_stream(model, normalizer, steps[0], Path(out_dir) / f"{stream}.onnx")
        difference = float(
            np.max(np.abs(onnx_logits(path, steps) - batch_logits(model, normalizer, steps)))
        )
        if difference > PARITY_TOLERANCE:
            raise RuntimeError(
                f"{stream} ONNX logits differ from PyTorch by {difference:.2e}, "
                f"above the {PARITY_TOLERANCE:.0e} tolerance"
            )
        report["files"][stream] = path
        report["parity"][stream] = {"max_abs_diff": difference}
    written = {**report, "files": {name: str(path) for name, path in report["files"].items()}}
    (Path(out_dir) / "export.json").write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
    return report
