"""Video attack families.

Each function takes bona fide frames of shape (T, H, W, 3) uint8 and returns
attacked frames of the same shape. The transformations are real signal
operations, never label flips, and each one leaves a measurable fingerprint that
`tests/test_attack_directions.py` asserts on.
"""

from __future__ import annotations

import cv2
import numpy as np

BAYER4 = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    + 0.5
) / 16.0


def _tile(pattern: np.ndarray, h: int, w: int) -> np.ndarray:
    reps = (h // pattern.shape[0] + 1, w // pattern.shape[1] + 1)
    return np.tile(pattern, reps)[:h, :w]


def video_replay(frames: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """Screen re-capture: resampling moire, refresh banding, gamma shift, crop, rotation."""
    t, h, w, _ = frames.shape
    scale = float(r.uniform(0.52, 0.68))
    small = (max(8, int(w * scale)), max(8, int(h * scale)))
    gamma = float(r.uniform(0.72, 0.88))
    contrast = float(r.uniform(1.05, 1.22))
    brightness = float(r.uniform(-12.0, 6.0))
    grid_freq = float(r.uniform(0.30, 0.45))
    grid_amp = float(r.uniform(0.05, 0.11))
    band_cycles = float(r.uniform(1.5, 3.5))
    band_amp = float(r.uniform(0.06, 0.13))
    band_speed = float(r.uniform(0.15, 0.45))
    margin = int(r.integers(2, 6))
    angle = float(r.uniform(-2.2, 2.2))

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    pixel_grid = 1.0 + grid_amp * np.sin(2 * np.pi * grid_freq * xs)[None, :, None]

    out = np.empty_like(frames)
    for i in range(t):
        f = frames[i].astype(np.float32)
        # Display resampling: down to panel resolution and back up.
        f = cv2.resize(f, small, interpolation=cv2.INTER_AREA)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
        f *= pixel_grid
        band = 1.0 + band_amp * np.sin(2 * np.pi * (band_cycles * ys / h + band_speed * i)).astype(
            np.float32
        )
        f *= band[:, None, None]
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, gamma)
        f = f * contrast + brightness
        f = np.clip(f, 0, 255)
        # Bezel crop then rescale, plus a small hand-held rotation.
        f = f[margin : h - margin, margin : w - margin]
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        f = cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        out[i] = np.clip(f, 0, 255).astype(np.uint8)
    return out


def video_print(frames: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """Print attack: halftone dither, static paper grain, flattened motion parallax."""
    t, h, w, _ = frames.shape
    flatten = float(r.uniform(0.55, 0.80))
    levels = float(r.choice([3.0, 4.0, 5.0]))
    grain_amp = float(r.uniform(0.05, 0.10))
    tint = np.array(
        [r.uniform(1.00, 1.06), r.uniform(0.97, 1.02), r.uniform(0.88, 0.96)], dtype=np.float32
    )
    desat = float(r.uniform(0.25, 0.45))

    grain_rng = np.random.default_rng(int(r.integers(0, 2**31 - 1)))
    grain = grain_rng.standard_normal((h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0.7)
    dither = _tile(BAYER4, h, w)[:, :, None]

    mean_frame = frames.astype(np.float32).mean(axis=0)
    out = np.empty_like(frames)
    for i in range(t):
        f = frames[i].astype(np.float32)
        # A print has no parallax: the sheet is flat, so motion collapses.
        f = (1.0 - flatten) * f + flatten * mean_frame
        grey = f.mean(axis=2, keepdims=True)
        f = f * (1.0 - desat) + grey * desat
        f *= tint[None, None, :]
        f *= 1.0 + grain_amp * grain[:, :, None]
        # Ordered halftone: quantise with a Bayer threshold matrix.
        q = f / 255.0 * levels
        f = np.floor(q + dither) / levels * 255.0
        out[i] = np.clip(f, 0, 255).astype(np.uint8)
    return out


def video_splice(frames: np.ndarray, donor: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """Swap the face region in from a second identity with alpha blending and seam jitter."""
    t, h, w, _ = frames.shape
    cx, cy = w / 2.0, h * 0.52
    rx = int(w * float(r.uniform(0.26, 0.32)))
    ry = int(h * float(r.uniform(0.34, 0.42)))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), float(r.uniform(0.8, 1.8)))
    alpha = float(r.uniform(0.80, 0.96))

    out = np.empty_like(frames)
    for i in range(t):
        jx, jy = int(r.integers(-1, 2)), int(r.integers(-1, 2))
        m = np.float32([[1, 0, jx], [0, 1, jy]])
        donor_i = cv2.warpAffine(
            donor[i % donor.shape[0]].astype(np.float32), m, (w, h), borderMode=cv2.BORDER_REFLECT
        )
        mask_i = cv2.warpAffine(mask, m, (w, h), borderMode=cv2.BORDER_CONSTANT)[:, :, None]
        host = frames[i].astype(np.float32)
        # Match the donor to the host global colour so the swap is not trivial.
        donor_i = donor_i - donor_i.mean(axis=(0, 1)) + host.mean(axis=(0, 1))
        blended = host * (1.0 - alpha * mask_i) + donor_i * (alpha * mask_i)
        out[i] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def video_recompress(frames: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """Aggressive re-encode: real JPEG round trips plus 8x8 DC quantisation blocking."""
    t, h, w, _ = frames.shape
    q1 = int(r.integers(6, 16))
    q2 = int(r.integers(8, 20))
    dc_step = float(r.uniform(14.0, 26.0))
    out = np.empty_like(frames)
    bh, bw = h // 8, w // 8
    for i in range(t):
        f = frames[i]
        for quality in (q1, q2):
            ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:  # pragma: no cover - encoder always succeeds for 8 bit frames
                raise RuntimeError("jpeg encoding failed")
            f = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        g = f.astype(np.float32)
        # Coarse DC term per 8x8 block, the classic block boundary artefact.
        blocks = g[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8, 3)
        dc = blocks.mean(axis=(1, 3), keepdims=True)
        dc_q = np.round(dc / dc_step) * dc_step
        g[: bh * 8, : bw * 8] = (blocks - dc + dc_q).reshape(bh * 8, bw * 8, 3)
        out[i] = np.clip(g, 0, 255).astype(np.uint8)
    return out
