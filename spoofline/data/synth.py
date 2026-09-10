"""Procedural bona fide clip synthesis.

The video side renders a talking-head-like sequence: a textured face region on a
parallax background, with blink events, a mouth whose opening follows the audio
envelope, a slow lighting drift and camera noise. The audio side is an additive
harmonic model with a prosody contour, time varying formants, unvoiced bursts and
room tone. Both are fully determined by an identity and a clip seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..seeding import rng as make_rng

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Identity:
    """Per-speaker parameters shared by every clip of that speaker."""

    ident: str
    f0_base: float
    formants: tuple[float, float, float]
    formant_alt: tuple[float, float, float]
    timbre_alpha: float
    speaking_rate: float
    skin: tuple[float, float, float]
    background: tuple[float, float, float]
    face_scale: float
    face_aspect: float
    eye_spacing: float
    texture_seed: int


def make_identity(seed: int, index: int) -> Identity:
    """Build a deterministic identity from the run seed and an index."""
    r = make_rng(seed, f"identity/{index}")
    f0 = float(r.uniform(95.0, 205.0))
    f1 = float(r.uniform(420.0, 700.0))
    f2 = float(r.uniform(1100.0, 1950.0))
    f3 = float(r.uniform(2400.0, 3100.0))
    return Identity(
        ident=f"id_{index:03d}",
        f0_base=f0,
        formants=(f1, f2, f3),
        formant_alt=(f1 * float(r.uniform(0.7, 1.4)), f2 * float(r.uniform(0.7, 1.4)), f3),
        timbre_alpha=float(r.uniform(1.1, 1.9)),
        speaking_rate=float(r.uniform(3.0, 5.0)),
        skin=(
            float(r.uniform(120, 210)),
            float(r.uniform(95, 170)),
            float(r.uniform(80, 150)),
        ),
        background=(
            float(r.uniform(30, 110)),
            float(r.uniform(30, 110)),
            float(r.uniform(30, 110)),
        ),
        face_scale=float(r.uniform(0.30, 0.38)),
        face_aspect=float(r.uniform(1.15, 1.40)),
        eye_spacing=float(r.uniform(0.30, 0.40)),
        texture_seed=int(r.integers(0, 2**31 - 1)),
    )


def _smooth_noise(r: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Low pass filtered gaussian field, used for skin and paper texture."""
    field = r.standard_normal(shape).astype(np.float32)
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(field, (k, k), sigma)


def _random_walk(r: np.random.Generator, n: int, step: float, smooth: int = 3) -> np.ndarray:
    """A short smoothed random walk used for head motion and drift."""
    walk = np.cumsum(r.normal(0.0, step, size=n).astype(np.float32))
    walk -= walk.mean()
    if smooth > 1 and n >= smooth:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        walk = np.convolve(walk, kernel, mode="same").astype(np.float32)
    return walk


def synthesize_audio(
    identity: Identity,
    seed: int,
    label: str,
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    """Synthesize a speech-like waveform for one clip.

    Returns float32 in [-1, 1].
    """
    r = make_rng(seed, f"audio/{label}")
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    duration = n_samples / sample_rate

    # Prosody: declining f0 with a slow contour plus vibrato.
    rate = identity.speaking_rate * float(r.uniform(0.85, 1.15))
    contour = 1.0 + 0.07 * np.sin(TWO_PI * 0.6 * t + r.uniform(0, TWO_PI))
    declination = 1.0 - 0.12 * (t / max(duration, 1e-6))
    vibrato = 1.0 + 0.012 * np.sin(TWO_PI * 5.2 * t + r.uniform(0, TWO_PI))
    f0 = identity.f0_base * float(r.uniform(0.92, 1.08)) * contour * declination * vibrato
    phase = np.cumsum(TWO_PI * f0 / sample_rate).astype(np.float32)

    # Syllabic envelope and voiced / unvoiced segmentation.
    syllable = 0.5 * (1.0 + np.sin(TWO_PI * rate * t + r.uniform(0, TWO_PI)))
    envelope = (syllable**1.6).astype(np.float32)
    voiced = (syllable > 0.35).astype(np.float32)
    voiced = np.convolve(voiced, np.ones(64, dtype=np.float32) / 64, mode="same")

    # Time varying vowel target, blending two formant sets.
    blend = 0.5 * (1.0 + np.sin(TWO_PI * rate * 0.5 * t + r.uniform(0, TWO_PI)))
    formants = [
        identity.formants[i] * (1.0 - blend) + identity.formant_alt[i] * blend for i in range(3)
    ]
    bandwidths = (90.0, 130.0, 190.0)
    gains = (1.0, 0.62, 0.32)

    nyquist = sample_rate / 2.0
    n_harm = int(min(48, max(8, np.floor(0.92 * nyquist / float(np.max(f0))))))
    signal = np.zeros(n_samples, dtype=np.float32)
    for k in range(1, n_harm + 1):
        freq = k * f0
        gain = np.zeros(n_samples, dtype=np.float32)
        for fc, bw, g in zip(formants, bandwidths, gains, strict=True):
            gain += g / (1.0 + ((freq - fc) / bw) ** 2)
        amp = gain * (k ** (-identity.timbre_alpha))
        signal += (amp * np.sin(k * phase)).astype(np.float32)

    signal *= envelope * voiced

    # Unvoiced bursts: band limited noise where voicing is weak.
    burst = r.standard_normal(n_samples).astype(np.float32)
    burst = burst - np.convolve(burst, np.ones(9, dtype=np.float32) / 9, mode="same")
    signal += 0.06 * burst * envelope * (1.0 - voiced)

    # Room tone: pink-ish noise floor.
    tone = r.standard_normal(n_samples).astype(np.float32)
    tone = np.convolve(tone, np.ones(24, dtype=np.float32) / 24, mode="same")
    signal += 0.004 * tone

    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.72).astype(np.float32)


def audio_envelope(audio: np.ndarray, n_frames: int) -> np.ndarray:
    """Frame level amplitude envelope in [0, 1], used to drive the mouth."""
    chunks = np.array_split(np.abs(audio), n_frames)
    env = np.array([float(c.mean()) if c.size else 0.0 for c in chunks], dtype=np.float32)
    top = float(env.max()) or 1.0
    return env / top


def _face_layer(
    identity: Identity,
    size: int,
    mouth_open: float,
    blink: float,
    texture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the face as a BGR layer plus its alpha mask."""
    layer = np.zeros((size, size, 3), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size / 2.0, size * 0.52
    rx = size * identity.face_scale
    ry = rx * identity.face_aspect

    cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 1.2)
    skin = np.array(identity.skin, dtype=np.float32)
    layer[:] = skin[None, None, :]
    layer *= (1.0 + 0.22 * texture)[:, :, None]

    eye_dx = rx * identity.eye_spacing * 2.0
    eye_y = int(cy - ry * 0.25)
    eye_h = max(1, int(ry * 0.14 * max(blink, 0.12)))
    eye_w = max(2, int(rx * 0.26))
    for sign in (-1, 1):
        ex = int(cx + sign * eye_dx / 2.0)
        cv2.ellipse(layer, (ex, eye_y), (eye_w, eye_h), 0, 0, 360, (35.0, 32.0, 30.0), -1)
        pupil = (max(1, eye_w // 3), eye_h)
        cv2.ellipse(layer, (ex, eye_y), pupil, 0, 0, 360, (12.0, 10.0, 9.0), -1)

    nose_y = int(cy + ry * 0.08)
    cv2.ellipse(
        layer,
        (int(cx), nose_y),
        (max(1, int(rx * 0.12)), max(1, int(ry * 0.16))),
        0,
        0,
        360,
        tuple((skin * 0.86).tolist()),
        -1,
    )

    mouth_y = int(cy + ry * 0.45)
    mouth_w = max(2, int(rx * 0.42))
    mouth_h = max(1, int(ry * (0.05 + 0.24 * mouth_open)))
    cv2.ellipse(layer, (int(cx), mouth_y), (mouth_w, mouth_h), 0, 0, 360, (48.0, 40.0, 62.0), -1)
    if mouth_open > 0.45:
        cv2.ellipse(
            layer,
            (int(cx), mouth_y),
            (int(mouth_w * 0.7), max(1, int(mouth_h * 0.55))),
            0,
            0,
            360,
            (22.0, 18.0, 30.0),
            -1,
        )
    return layer, mask


def synthesize_video(
    identity: Identity,
    seed: int,
    label: str,
    n_frames: int,
    size: int,
    envelope: np.ndarray,
) -> np.ndarray:
    """Render a bona fide clip as uint8 frames of shape (T, H, W, 3)."""
    r = make_rng(seed, f"video/{label}")
    tex_rng = np.random.default_rng(identity.texture_seed)
    texture = _smooth_noise(tex_rng, (size, size), 1.6)
    paper = _smooth_noise(tex_rng, (size, size), 3.5)

    grad = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    bg = np.array(identity.background, dtype=np.float32)[None, None, :]
    background = bg * (1.0 + 0.25 * grad[:, None, None]) + 18.0 * paper[:, :, None]

    dx = _random_walk(r, n_frames, 0.9)
    dy = _random_walk(r, n_frames, 0.7)
    angle = _random_walk(r, n_frames, 0.9)
    light = 1.0 + 0.06 * _random_walk(r, n_frames, 0.6)
    noise_sigma = float(r.uniform(1.5, 3.5))

    blink_frames = set()
    if n_frames >= 6:
        start = int(r.integers(1, max(2, n_frames - 2)))
        blink_frames = {start, min(start + 1, n_frames - 1)}

    frames = np.empty((n_frames, size, size, 3), dtype=np.uint8)
    for i in range(n_frames):
        blink = 0.15 if i in blink_frames else 1.0
        face, mask = _face_layer(identity, size, float(envelope[i]), blink, texture)

        m_face = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), float(angle[i]), 1.0)
        m_face[0, 2] += float(dx[i])
        m_face[1, 2] += float(dy[i])
        m_bg = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), float(angle[i]) * 0.25, 1.0)
        m_bg[0, 2] += float(dx[i]) * 0.25
        m_bg[1, 2] += float(dy[i]) * 0.25

        face_w = cv2.warpAffine(face, m_face, (size, size), borderMode=cv2.BORDER_REFLECT)
        mask_w = cv2.warpAffine(mask, m_face, (size, size), borderMode=cv2.BORDER_CONSTANT)
        bg_w = cv2.warpAffine(background, m_bg, (size, size), borderMode=cv2.BORDER_REFLECT)

        shade = 1.0 + 0.18 * np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :, None]
        composed = bg_w * (1.0 - mask_w[:, :, None]) + face_w * mask_w[:, :, None] * shade
        composed *= float(light[i])
        composed += r.normal(0.0, noise_sigma, composed.shape).astype(np.float32)
        frames[i] = np.clip(composed, 0, 255).astype(np.uint8)
    return frames
