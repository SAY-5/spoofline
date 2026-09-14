"""Export a finished spoofline run for the browser demo.

Run from the repository root after `make demo`:

    uv run --with onnx --with onnxruntime python web/scripts/export.py --trained-from <sha>

torch.onnx.export needs the onnx package, which is not a project dependency, hence
the --with flags. Pass the commit the checkpoints were trained from; without it the
last commit touching spoofline/, pyproject.toml, uv.lock or the Makefile is used.

Everything lands in web/public/data:

    models/video.onnx, models/audio.onnx   both streams, normaliser folded into the graph
    clips/<clip_id>.zlib                    lossless clip: deflate of uint8 frame residuals (left
                                            then temporal difference) and int16 audio differences
    mel_fbank.bin                           float32 mel filterbank, shape (n_fft // 2 + 1, n_mels)
    logmel_check.bin                        float32 log mel of the first clip, (n_mels, frames)
    manifest.json                           shapes, features, calibration, fusion, clip index
    calib_scores.json                       calibration split logits and labels, in pipeline order
    test_scores.json                        seen and unseen test split logits and labels
    reference.json                          PyTorch logits, probabilities and decisions per exported
                                            clip, plus the measured metrics of the run

The script checks its own output: the calibration split logits must refit to the
exact Platt parameters and thresholds in calibration.json, the test split scores
must reproduce the metrics in results.json, and when onnxruntime is importable
every exported clip is scored through the ONNX graphs and compared with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spoofline.calibrate import PlattCalibrator, threshold_at_precision
from spoofline.config import profile
from spoofline.data.dataset import load_corpus
from spoofline.data.features import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    PATCH_WIDTH,
    log_mel,
    mel_patches,
    mel_transform,
    video_steps,
)
from spoofline.data.sources import Clip, ClipRecord, NpzCorpusSource
from spoofline.families import ALL_FAMILIES, FAMILY_DESCRIPTIONS
from spoofline.fusion import fit_fusion
from spoofline.metrics import evaluate
from spoofline.models.cnn_lstm import CnnLstmDetector, load_checkpoint
from spoofline.pipeline import score_single_clip
from spoofline.seeding import seed_everything
from spoofline.train import score_clips

REPO = Path(__file__).resolve().parents[2]
MAX_CLIPS = 28
STREAMS = ("video", "audio")


class InferenceGraph(nn.Module):
    """Batch of one, every step valid: normaliser, step encoder, LSTM, attention, head.

    With no padding, packing and the attention mask change nothing, so this is the
    same computation as CnnLstmDetector.forward on a single full length clip.
    """

    def __init__(self, model: CnnLstmDetector, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        channels = mean.numel()
        self.register_buffer("mean", mean.float().view(1, 1, channels, 1, 1).clone())
        self.register_buffer("std", std.float().view(1, 1, channels, 1, 1).clone())

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        x = (steps - self.mean) / self.std
        batch, n_steps = x.shape[0], x.shape[1]
        embedded = self.model.encoder(x.reshape(batch * n_steps, *x.shape[2:]))
        sequence, _ = self.model.lstm(embedded.view(batch, n_steps, -1))
        weights = torch.softmax(self.model.attention.score(sequence).squeeze(-1), dim=1)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return self.model.head(pooled).squeeze(-1)


class SubsetSource:
    """A ClipSource restricted to a list of clip ids, in that order."""

    def __init__(self, source: NpzCorpusSource, clip_ids: Sequence[str]) -> None:
        by_id = {r.clip_id: r for r in source.records}
        self._records = [by_id[c] for c in clip_ids]
        self._source = source

    @property
    def records(self) -> list[ClipRecord]:
        return list(self._records)

    @property
    def sample_rate(self) -> int:
        return self._source.sample_rate

    def load(self, clip_id: str) -> Clip:
        return self._source.load(clip_id)

    def __len__(self) -> int:
        return len(self._records)


def encode_clip(frames: np.ndarray, audio: np.ndarray) -> bytes:
    """Lossless packing: left then temporal prediction on frames, first difference on audio."""
    f = frames.astype(np.int16)
    left = f.copy()
    left[:, :, 1:] = f[:, :, 1:] - f[:, :, :-1]
    residual = left.copy()
    residual[1:] = left[1:] - left[:-1]
    a = audio.astype(np.int32)
    diff = a.copy()
    diff[1:] = a[1:] - a[:-1]
    payload = (residual % 256).astype(np.uint8).tobytes() + diff.astype("<i2").tobytes()
    return zlib.compress(payload, 9)


def decode_clip(blob: bytes, shape: tuple[int, ...]) -> tuple[bytes, bytes]:
    raw = zlib.decompress(blob)
    size = int(np.prod(shape))
    residual = np.frombuffer(raw[:size], np.uint8).reshape(shape).astype(np.int64)
    left = np.cumsum(residual, axis=0) % 256
    frames = (np.cumsum(left, axis=2) % 256).astype(np.uint8)
    diff = np.frombuffer(raw[size:], "<i2").astype(np.int64)
    audio = ((np.cumsum(diff) + 32768) % 65536 - 32768).astype("<i2")
    return frames.tobytes(), audio.tobytes()


def combo(record: ClipRecord) -> str:
    if record.video_attacked and record.audio_attacked:
        return "both"
    if record.video_attacked:
        return "video_only"
    if record.audio_attacked:
        return "audio_only"
    return "bonafide"


def training_commit() -> str:
    paths = ["spoofline", "pyproject.toml", "uv.lock", "Makefile"]
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def split_scores(run_dir, corpus_dir, clip_ids, config) -> dict[str, np.ndarray]:
    """Logits through the exact batched path the pipeline used, so thresholds reproduce."""
    source = SubsetSource(NpzCorpusSource(corpus_dir), clip_ids)
    corpus = load_corpus(source)
    scores = {}
    for stream in STREAMS:
        model, normalizer, _ = load_checkpoint(run_dir / f"{stream}.pt")
        scores[stream] = score_clips(model, normalizer, corpus, list(clip_ids), stream)
    del config
    return scores


def check_close(label: str, got: float, want: float, tol: float) -> None:
    if not abs(got - want) <= tol:
        raise SystemExit(f"export check failed: {label}: got {got!r}, want {want!r}")


def select_clips(records: dict[str, ClipRecord], split_of: dict[str, str], fused_flag, unseen):
    """Deterministic pick: a gallery identity plus clips covering every combination."""
    test = [records[c] for c in sorted(split_of)]
    identities = sorted({r.identity for r in test})

    def coverage(identity: str) -> tuple[int, int, int]:
        rows = [r for r in test if r.identity == identity]
        present = {f for r in rows for f in r.families}
        single = {f for r in rows if combo(r) in ("video_only", "audio_only") for f in r.families}
        return len(present), len(single), int(any(r.label == 0 for r in rows))

    gallery_identity = max(identities, key=coverage)
    own = [r for r in test if r.identity == gallery_identity]
    gallery = {"bonafide": next(r.clip_id for r in own if r.label == 0)}
    for family in ALL_FAMILIES:
        single = [r for r in own if family in r.families and combo(r) != "both"]
        both = [r for r in own if family in r.families and combo(r) == "both"]
        gallery[family] = (single or both)[0].clip_id

    picks: list[str] = list(dict.fromkeys(gallery.values()))

    def add(candidates) -> None:
        for record in candidates:
            if record.clip_id not in picks and len(picks) < MAX_CLIPS:
                picks.append(record.clip_id)
                return

    others = [r for r in test if r.identity != gallery_identity]
    for family in ALL_FAMILIES:
        single = [r for r in others if family in r.families and combo(r) != "both"]
        if family in unseen:
            add([r for r in single if fused_flag[r.clip_id]])
            add([r for r in single if not fused_flag[r.clip_id]])
        else:
            add(single)
    bona = [r for r in others if r.label == 0]
    add([r for r in bona if fused_flag[r.clip_id]])
    add([r for r in bona if split_of[r.clip_id] == "seen_test" and not fused_flag[r.clip_id]])
    add([r for r in bona if split_of[r.clip_id] == "unseen_test" and not fused_flag[r.clip_id]])
    both = [r for r in others if combo(r) == "both"]
    add([r for r in both if all(f in unseen for f in r.families)])
    add([r for r in both if not any(f in unseen for f in r.families)])
    add([r for r in both if sum(f in unseen for f in r.families) == 1])
    return gallery_identity, gallery, picks


def export_graph(graph: InferenceGraph, shape: tuple[int, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(shape, dtype=torch.float32)
    torch.onnx.export(
        graph,
        (dummy,),
        str(path),
        input_names=["steps"],
        output_names=["logit"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs" / "full")
    parser.add_argument("--corpus-dir", type=Path, default=REPO / "data" / "full")
    parser.add_argument("--out", type=Path, default=REPO / "web" / "public" / "data")
    parser.add_argument("--trained-from", default=None, help="commit the run was trained from")
    args = parser.parse_args()

    config = profile("full")
    seed_everything(config.seed, config.threads)
    run_dir, out = args.run_dir, args.out
    results = json.loads((run_dir / "results.json").read_text())
    calibration = json.loads((run_dir / "calibration.json").read_text())
    splits = json.loads((run_dir / "splits.json").read_text())["splits"]
    source = NpzCorpusSource(args.corpus_dir)
    records = {r.clip_id: r for r in source.records}
    meta = source.meta
    unseen = tuple(results["unseen_families"])
    target = float(results["target_precision"])
    commit = args.trained_from or training_commit()

    print("[1/5] rescoring the calibration and test splits through the pipeline path")
    calib_ids = splits["calib"]
    calib = split_scores(run_dir, args.corpus_dir, calib_ids, config)
    clip_labels = np.array([records[c].label for c in calib_ids], dtype=np.int64)
    for stream in STREAMS:
        modality = np.array(
            [int(getattr(records[c], f"{stream}_attacked")) for c in calib_ids], dtype=np.int64
        )
        fitted = PlattCalibrator.fit(calib[stream], modality)
        saved = calibration[stream]
        check_close(f"{stream} platt a", fitted.a, saved["calibrator"]["a"], 1e-12)
        check_close(f"{stream} platt b", fitted.b, saved["calibrator"]["b"], 1e-12)
        point = threshold_at_precision(fitted.predict(calib[stream]), clip_labels, target)
        check_close(f"{stream} threshold", point.threshold, saved["operating"]["threshold"], 1e-12)
    probs = {}
    for stream in STREAMS:
        platt_map = PlattCalibrator.from_dict(calibration[stream]["calibrator"])
        probs[stream] = platt_map.predict(calib[stream])
    fusion = fit_fusion(probs["video"], probs["audio"], clip_labels, target)
    check_close("fusion weight", fusion.weight, calibration["fused"]["weight"], 1e-12)
    check_close(
        "fusion threshold",
        fusion.operating.threshold,
        calibration["fused"]["operating"]["threshold"],
        1e-12,
    )

    test_ids = sorted(splits["seen_test"] + splits["unseen_test"])
    split_of = {c: n for n in ("seen_test", "unseen_test") for c in splits[n]}
    test = split_scores(run_dir, args.corpus_dir, test_ids, config)
    platt = {s: PlattCalibrator.from_dict(calibration[s]["calibrator"]) for s in STREAMS}
    weight = float(calibration["fused"]["weight"])
    thresholds = {s: float(calibration[s]["operating"]["threshold"]) for s in STREAMS}
    thresholds["fused"] = float(calibration["fused"]["operating"]["threshold"])
    test_probs = {s: platt[s].predict(test[s]) for s in STREAMS}
    test_probs["fused"] = weight * test_probs["video"] + (1.0 - weight) * test_probs["audio"]
    test_labels = np.array([records[c].label for c in test_ids], dtype=np.int64)
    for split in ("seen_test", "unseen_test"):
        mask = np.array([split_of[c] == split for c in test_ids])
        for detector in ("video", "audio", "fused"):
            point = evaluate(
                test_probs[detector][mask], test_labels[mask], thresholds[detector]
            ).as_dict()
            saved = results["metrics"][split][detector]
            for key in ("precision", "recall", "f1", "auc", "eer"):
                check_close(f"{split} {detector} {key}", point[key], saved[key], 1e-9)
    fused_flag = {
        c: bool(test_probs["fused"][i] >= thresholds["fused"]) for i, c in enumerate(test_ids)
    }

    print("[2/5] choosing clips")
    gallery_identity, gallery, picks = select_clips(records, split_of, fused_flag, unseen)
    print(f"  gallery identity {gallery_identity}, {len(picks)} clips")

    print("[3/5] writing clips, filterbank and scores")
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for stale in (out / "clips").iterdir():
        if stale.suffix != ".zlib" or stale.stem not in picks:
            stale.unlink()
    rate = int(meta["sample_rate"])
    reference_clips: dict[str, dict] = {}
    steps_by_clip: dict[str, dict[str, np.ndarray]] = {}
    roles = {clip_id: [] for clip_id in picks}
    for family, clip_id in gallery.items():
        roles[clip_id].append(f"gallery:{family}")
    for clip_id in picks:
        with np.load(source.path_for(clip_id)) as data:
            frames = np.ascontiguousarray(data["video"], dtype=np.uint8)
            audio = np.ascontiguousarray(data["audio"], dtype="<i2")
        encoded = encode_clip(frames, audio)
        if decode_clip(encoded, frames.shape) != (frames.tobytes(), audio.tobytes()):
            raise SystemExit(f"export check failed: {clip_id} does not round trip")
        (out / "clips" / f"{clip_id}.zlib").write_bytes(encoded)
        wave = audio.astype(np.float32) / 32767.0
        steps_by_clip[clip_id] = {
            "video": video_steps(frames)[None],
            "audio": mel_patches(log_mel(wave, rate))[None],
        }
        scored = score_single_clip(run_dir, source.path_for(clip_id), sample_rate=rate)
        reference_clips[clip_id] = {
            "video_logit": scored["video_logit"],
            "audio_logit": scored["audio_logit"],
            "video_probability": scored["video_probability"],
            "audio_probability": scored["audio_probability"],
            "fused_probability": scored["fused_probability"],
            "video_flags": bool(scored["video_flags"]),
            "audio_flags": bool(scored["audio_flags"]),
            "decision": scored["decision"],
        }

    fbank = mel_transform(rate).mel_scale.fb.numpy().astype("<f4")
    (out / "mel_fbank.bin").write_bytes(np.ascontiguousarray(fbank).tobytes())
    first = picks[0]
    with np.load(source.path_for(first)) as data:
        check_wave = data["audio"].astype(np.float32) / 32767.0
    check_mel = log_mel(check_wave, rate).astype("<f4")
    (out / "logmel_check.bin").write_bytes(np.ascontiguousarray(check_mel).tobytes())

    def rows(ids: Sequence[str], scores: dict[str, np.ndarray]) -> dict:
        return {
            "clip_ids": list(ids),
            "label": [records[c].label for c in ids],
            "video_attacked": [int(records[c].video_attacked) for c in ids],
            "audio_attacked": [int(records[c].audio_attacked) for c in ids],
            "video_logit": [float(v) for v in scores["video"]],
            "audio_logit": [float(v) for v in scores["audio"]],
        }

    (out / "calib_scores.json").write_text(json.dumps(rows(calib_ids, calib)) + "\n")
    test_payload = rows(test_ids, test)
    test_payload["split"] = [split_of[c] for c in test_ids]
    test_payload["families"] = [list(records[c].families) for c in test_ids]
    (out / "test_scores.json").write_text(json.dumps(test_payload) + "\n")

    print("[4/5] exporting both streams to ONNX")
    graphs: dict[str, InferenceGraph] = {}
    shapes: dict[str, list[int]] = {}
    for stream in STREAMS:
        model, normalizer, _ = load_checkpoint(run_dir / f"{stream}.pt")
        graph = InferenceGraph(model, normalizer.mean, normalizer.std).eval()
        shape = tuple(steps_by_clip[first][stream].shape)
        shapes[stream] = list(shape)
        with torch.no_grad():
            for clip_id in picks:
                got = float(graph(torch.from_numpy(steps_by_clip[clip_id][stream]))[0])
                want = reference_clips[clip_id][f"{stream}_logit"]
                check_close(f"{clip_id} {stream} graph", got, want, 1e-5)
        export_graph(graph, shape, out / "models" / f"{stream}.onnx")
        graphs[stream] = graph

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not importable, skipping the ONNX parity pass")
    else:
        worst = 0.0
        for stream in STREAMS:
            session = ort.InferenceSession(str(out / "models" / f"{stream}.onnx"))
            for clip_id in picks:
                got = float(session.run(None, {"steps": steps_by_clip[clip_id][stream]})[0][0])
                want = reference_clips[clip_id][f"{stream}_logit"]
                check_close(f"{clip_id} {stream} onnx", got, want, 1e-4)
                worst = max(worst, abs(got - want))
        print(f"  onnxruntime matches PyTorch on {len(picks)} clips, worst logit gap {worst:.2e}")

    print("[5/5] writing manifest and reference")
    clip_index = []
    for clip_id in picks:
        record = records[clip_id]
        clip_index.append(
            {
                "id": clip_id,
                "identity": record.identity,
                "video_family": record.video_family,
                "audio_family": record.audio_family,
                "label": record.label,
                "combo": combo(record),
                "split": split_of[clip_id],
                "touches_unseen": any(f in unseen for f in record.families),
                "roles": roles[clip_id],
            }
        )
    manifest = {
        "trained_from_commit": commit,
        "seed": results["seed"],
        "profile": results["profile"],
        "target_precision": target,
        "unseen_families": list(unseen),
        "families": {f: FAMILY_DESCRIPTIONS[f] for f in ALL_FAMILIES},
        "corpus": {
            "n_frames": int(meta["n_frames"]),
            "frame_size": int(meta["frame_size"]),
            "sample_rate": rate,
            "n_samples": int(meta["n_samples"]),
            "n_clips": int(meta["n_clips"]),
            "n_identities": int(meta["n_identities"]),
        },
        "features": {
            "audio_scale": 32767.0,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "patch_width": PATCH_WIDTH,
            "window": "hann_periodic",
            "center_pad": "reflect",
            "log_epsilon": 1e-6,
            "fbank_shape": list(fbank.shape),
            "logmel_check_clip": first,
            "logmel_check_shape": list(check_mel.shape),
        },
        "clip_encoding": (
            "zlib deflate of uint8 frame residuals (T, H, W, 3), left difference along W then "
            "temporal difference along T, modulo 256, followed by int16 little endian first "
            "differences of the audio, modulo 65536"
        ),
        "models": {
            s: {"path": f"models/{s}.onnx", "input": "steps", "output": "logit", "shape": shapes[s]}
            for s in STREAMS
        },
        "calibration": calibration,
        "gallery": {"identity": gallery_identity, "clips": gallery},
        "clips": clip_index,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reference = {
        "trained_from_commit": commit,
        "clips": reference_clips,
        "run": {
            key: results[key]
            for key in (
                "calibration",
                "metrics",
                "rules",
                "family_rates",
                "headline",
                "splits",
                "combo_counts",
                "family_counts",
                "corpus",
                "training",
                "target_precision",
                "unseen_families",
                "seed",
            )
        },
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"  wrote {out} ({total / 1e6:.2f} MB, trained from {commit[:12]})")


if __name__ == "__main__":
    main()
