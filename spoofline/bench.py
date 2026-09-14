"""CPU latency of scoring one clip, per stream and end to end, with PyTorch and ONNX Runtime."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from .scoring import STREAMS, MediaClip, RunScorer, batch_logits, clip_steps, read_npz_clip

ENGINES = ("torch", "onnx")
STAGES = (*STREAMS, "end_to_end")


def latency_summary(seconds: Sequence[float]) -> dict:
    samples = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(samples.mean()),
        "n": int(samples.size),
    }


def per_call_seconds(fn: Callable, items: Sequence, warmup: int) -> list[float]:
    """Wall time of ``fn`` on each item, after ``warmup`` untimed calls on the first item."""
    for _ in range(warmup):
        fn(items[0])
    samples = []
    for item in items:
        started = time.perf_counter()
        fn(item)
        samples.append(time.perf_counter() - started)
    return samples


def stream_logit_fn(scorer: RunScorer, sessions: dict | None, stream: str) -> Callable:
    """One clip's features plus one forward pass of one stream, on either engine."""

    def run(clip: MediaClip) -> float:
        steps = clip_steps(stream, clip)
        if sessions is None:
            return float(batch_logits(*scorer.models[stream], [steps])[0])
        return float(sessions[stream].run(["logit"], {"steps": steps[None]})[0][0])

    return run


def end_to_end_fn(scorer: RunScorer, sessions: dict | None, sample_rate: int) -> Callable:
    """Decode an npz, run both streams, calibrate, fuse and attribute."""
    streams = {stream: stream_logit_fn(scorer, sessions, stream) for stream in STREAMS}

    def run(path: Path) -> dict:
        clip = read_npz_clip(path, sample_rate)
        logits = {stream: np.array([fn(clip)]) for stream, fn in streams.items()}
        return scorer.describe(logits)[0]

    return run


def onnx_sessions(onnx_dir: Path, threads: int) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return {
        stream: ort.InferenceSession(
            str(Path(onnx_dir) / f"{stream}.onnx"), options, providers=["CPUExecutionProvider"]
        )
        for stream in STREAMS
    }


def run_benchmark(
    run_dir: Path,
    clip_paths: Sequence[Path],
    onnx_dir: Path,
    threads: int = 1,
    warmup: int = 5,
    sample_rate: int = 16000,
) -> dict:
    """p50 and p95 per clip latency for each stream and end to end, on both engines."""
    torch.set_num_threads(threads)
    scorer = RunScorer.from_run(run_dir)
    sessions = onnx_sessions(onnx_dir, threads)
    clips = [read_npz_clip(path, sample_rate) for path in clip_paths]
    rows = []
    for engine in ENGINES:
        engine_sessions = sessions if engine == "onnx" else None
        for stream in STREAMS:
            fn = stream_logit_fn(scorer, engine_sessions, stream)
            rows.append(
                {
                    "engine": engine,
                    "stage": stream,
                    **latency_summary(per_call_seconds(fn, clips, warmup)),
                }
            )
        fn = end_to_end_fn(scorer, engine_sessions, sample_rate)
        samples = per_call_seconds(fn, list(clip_paths), warmup)
        rows.append({"engine": engine, "stage": "end_to_end", **latency_summary(samples)})
    return {"threads": threads, "clips": len(clips), "warmup": warmup, "rows": rows}
