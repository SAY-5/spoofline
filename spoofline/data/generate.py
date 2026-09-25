"""Deterministic corpus generation.

The plan (which clip gets which identity and which attack combination) is fixed
by the run seed, and every clip is rendered from its own derived seed, so a clip
can be regenerated in isolation and the corpus is byte identical across runs.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CorpusConfig
from ..families import AUDIO_FAMILIES, BONAFIDE, VIDEO_FAMILIES
from ..seeding import rng as make_rng
from . import attacks_audio as aa
from . import attacks_video as av
from .sources import Clip, ClipRecord, NpzCorpusSource, write_manifest
from .synth import audio_envelope, make_identity, synthesize_audio, synthesize_video

COMBOS = ("none", "video", "audio", "both")


@dataclass(frozen=True)
class ClipPlan:
    """What a clip should contain, decided before any signal is rendered."""

    index: int
    clip_id: str
    identity_index: int
    donor_index: int
    video_family: str
    audio_family: str

    @property
    def label(self) -> int:
        return int(self.video_family != BONAFIDE or self.audio_family != BONAFIDE)


def plan_corpus(corpus: CorpusConfig, seed: int) -> list[ClipPlan]:
    """Balanced, shuffled assignment of attack combinations to clips."""
    n = corpus.n_clips
    combos: list[tuple[str, str]] = []
    for i in range(n):
        combo = COMBOS[i % 4]
        vf = VIDEO_FAMILIES[(i // 4) % len(VIDEO_FAMILIES)]
        af = AUDIO_FAMILIES[((i // 4) + (i // (4 * len(AUDIO_FAMILIES)))) % len(AUDIO_FAMILIES)]
        combos.append(
            {
                "none": (BONAFIDE, BONAFIDE),
                "video": (vf, BONAFIDE),
                "audio": (BONAFIDE, af),
                "both": (vf, af),
            }[combo]
        )
    order = make_rng(seed, "plan").permutation(n)
    plans: list[ClipPlan] = []
    for position, source in enumerate(order):
        vf, af = combos[int(source)]
        identity = position % corpus.n_identities
        donor = (identity + 1 + position % max(1, corpus.n_identities - 1)) % corpus.n_identities
        if donor == identity:
            donor = (identity + 1) % corpus.n_identities
        plans.append(
            ClipPlan(
                index=position,
                clip_id=f"clip_{position:05d}",
                identity_index=identity,
                donor_index=donor,
                video_family=vf,
                audio_family=af,
            )
        )
    return plans


def render_clip(plan: ClipPlan, corpus: CorpusConfig, seed: int) -> Clip:
    """Render one clip: bona fide signals first, then the assigned attacks."""
    identity = make_identity(seed, plan.identity_index)
    audio = synthesize_audio(identity, seed, plan.clip_id, corpus.sample_rate, corpus.n_samples)
    envelope = audio_envelope(audio, corpus.n_frames)
    frames = synthesize_video(
        identity, seed, plan.clip_id, corpus.n_frames, corpus.frame_size, envelope
    )

    if plan.video_family != BONAFIDE:
        r = make_rng(seed, f"attack/{plan.clip_id}/video")
        if plan.video_family == "video_replay":
            frames = av.video_replay(frames, r)
        elif plan.video_family == "video_print":
            frames = av.video_print(frames, r)
        elif plan.video_family == "video_recompress":
            frames = av.video_recompress(frames, r)
        elif plan.video_family == "video_splice":
            donor_identity = make_identity(seed, plan.donor_index)
            donor_frames = synthesize_video(
                donor_identity,
                seed,
                f"{plan.clip_id}/donor",
                corpus.n_frames,
                corpus.frame_size,
                envelope,
            )
            frames = av.video_splice(frames, donor_frames, r)
        else:  # pragma: no cover - guarded by families.validate_families
            raise ValueError(f"unknown video family {plan.video_family!r}")

    if plan.audio_family != BONAFIDE:
        r = make_rng(seed, f"attack/{plan.clip_id}/audio")
        if plan.audio_family == "audio_replay":
            audio = aa.audio_replay(audio, corpus.sample_rate, r)
        elif plan.audio_family == "audio_vocoder":
            audio = aa.audio_vocoder(audio, corpus.sample_rate, r)
        elif plan.audio_family == "audio_conversion":
            audio = aa.audio_conversion(audio, corpus.sample_rate, r)
        elif plan.audio_family == "audio_splice":
            donor_identity = make_identity(seed, plan.donor_index)
            donor_audio = synthesize_audio(
                donor_identity,
                seed,
                f"{plan.clip_id}/donor",
                corpus.sample_rate,
                corpus.n_samples,
            )
            audio = aa.audio_splice(audio, donor_audio, corpus.sample_rate, r)
        else:  # pragma: no cover - guarded by families.validate_families
            raise ValueError(f"unknown audio family {plan.audio_family!r}")

    record = ClipRecord(
        clip_id=plan.clip_id,
        identity=identity.ident,
        video_family=plan.video_family,
        audio_family=plan.audio_family,
        label=plan.label,
    )
    return Clip(record=record, frames=frames, audio=audio, sample_rate=corpus.sample_rate)


def family_counts(records: Iterable[ClipRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[f"video:{record.video_family}"] += 1
        counts[f"audio:{record.audio_family}"] += 1
    return dict(sorted(counts.items()))


def combo_counts(records: Iterable[ClipRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.combo for record in records).items()))


def corpus_meta(corpus: CorpusConfig, seed: int) -> dict:
    """Everything that decides what a rendered corpus contains."""
    return {
        "seed": seed,
        "name": corpus.name,
        "n_clips": corpus.n_clips,
        "n_identities": corpus.n_identities,
        "n_frames": corpus.n_frames,
        "frame_size": corpus.frame_size,
        "sample_rate": corpus.sample_rate,
        "duration_s": corpus.duration_s,
        "n_samples": corpus.n_samples,
    }


def generate_corpus(
    corpus: CorpusConfig,
    seed: int,
    out_dir: Path,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> NpzCorpusSource:
    """Render the whole corpus to ``out_dir`` and return a source over it."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    meta = corpus_meta(corpus, seed)
    if manifest_path.exists() and not force:
        source = NpzCorpusSource(out_dir)
        cached = {key: source.meta.get(key) for key in meta}
        if cached == meta:
            if progress:
                progress(f"corpus cache hit: {len(source)} clips at {out_dir}")
            return source
        differences = ", ".join(
            f"{key}={cached[key]!r} on disk against {meta[key]!r} requested"
            for key in meta
            if cached[key] != meta[key]
        )
        raise ValueError(
            f"{out_dir} already holds a corpus generated with a different shape "
            f"({differences}). Point at another directory or pass force to overwrite it."
        )

    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    plans = plan_corpus(corpus, seed)
    records: list[ClipRecord] = []
    started = time.perf_counter()
    step = max(1, len(plans) // 10)
    for i, plan in enumerate(plans):
        clip = render_clip(plan, corpus, seed)
        np.savez_compressed(
            out_dir / "clips" / f"{plan.clip_id}.npz",
            video=clip.frames,
            audio=np.clip(clip.audio * 32767.0, -32768, 32767).astype(np.int16),
        )
        records.append(clip.record)
        if progress and ((i + 1) % step == 0 or i + 1 == len(plans)):
            elapsed = time.perf_counter() - started
            progress(f"  generated {i + 1}/{len(plans)} clips ({elapsed:.1f}s)")

    meta = {
        "seed": seed,
        "name": corpus.name,
        "n_clips": corpus.n_clips,
        "n_identities": corpus.n_identities,
        "n_frames": corpus.n_frames,
        "frame_size": corpus.frame_size,
        "sample_rate": corpus.sample_rate,
        "duration_s": corpus.duration_s,
        "n_samples": corpus.n_samples,
    }
    write_manifest(manifest_path, records, meta)
    if progress:
        progress(f"  wrote manifest for {len(records)} clips to {manifest_path}")
    return NpzCorpusSource(out_dir)
