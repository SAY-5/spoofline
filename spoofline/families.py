"""The attack family registry.

A family is a real signal transformation applied to one stream. A clip carries a
video family and an audio family; either may be ``BONAFIDE``.
"""

from __future__ import annotations

from typing import Literal

BONAFIDE = "bonafide"

VIDEO_FAMILIES: tuple[str, ...] = (
    "video_replay",
    "video_print",
    "video_splice",
    "video_recompress",
)

AUDIO_FAMILIES: tuple[str, ...] = (
    "audio_replay",
    "audio_vocoder",
    "audio_conversion",
    "audio_splice",
)

ALL_FAMILIES: tuple[str, ...] = VIDEO_FAMILIES + AUDIO_FAMILIES

Stream = Literal["video", "audio"]

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "video_replay": (
        "screen re-capture: resampling moire, refresh banding, gamma shift, bezel crop, rotation"
    ),
    "video_print": (
        "print attack: ordered halftone dither, static paper grain, flattened motion parallax"
    ),
    "video_splice": (
        "face region swapped in from a second identity with alpha blending and seam jitter"
    ),
    "video_recompress": (
        "aggressive JPEG round trip through cv2 plus 8x8 DC quantisation blocking"
    ),
    "audio_replay": (
        "room impulse response, loudspeaker and microphone band limiting, device noise floor"
    ),
    "audio_vocoder": (
        "mel analysis, pseudo-inverse back to linear magnitude, Griffin-Lim phase resynthesis"
    ),
    "audio_conversion": (
        "pitch and formant shift by resampling, then phase vocoder time stretch back"
    ),
    "audio_splice": "segments from two different utterances concatenated with hard joins",
}


def stream_of(family: str) -> Stream:
    """Which stream a family attacks."""
    if family in VIDEO_FAMILIES:
        return "video"
    if family in AUDIO_FAMILIES:
        return "audio"
    raise ValueError(f"unknown attack family: {family!r}")


def validate_families(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Check that every name is a known family and return them as a tuple."""
    unknown = [n for n in names if n not in ALL_FAMILIES]
    if unknown:
        raise ValueError(f"unknown attack families: {unknown}; known: {list(ALL_FAMILIES)}")
    return tuple(names)
