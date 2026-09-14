/**
 * Feature extraction that mirrors spoofline/data/features.py.
 *
 * video_steps: uint8 frames (T, H, W, 3) to float32 steps (T, 3, H, W) in [-1, 1].
 * log_mel:     torchaudio MelSpectrogram (n_fft 512, hop 160, periodic Hann window,
 *              centred with reflect padding, power 2, HTK mel, no norm), then
 *              log(mel + 1e-6).
 * mel_patches: non overlapping 20 frame patches, (P, 1, n_mels, 20).
 */
import { Fft } from "./fft.ts";

export const N_FFT = 512;
export const HOP_LENGTH = 160;
export const N_MELS = 64;
export const PATCH_WIDTH = 20;
export const N_FREQS = N_FFT / 2 + 1;
const LOG_EPSILON = 1e-6;
const AUDIO_SCALE = 32767;

const f32 = Math.fround;

export interface ClipShape {
  frames: number;
  size: number;
  samples: number;
}

export function videoSteps(frames: Uint8Array, shape: ClipShape): Float32Array {
  const { frames: t, size } = shape;
  const plane = size * size;
  const out = new Float32Array(t * 3 * plane);
  for (let i = 0; i < t; i++) {
    for (let c = 0; c < 3; c++) {
      const dst = (i * 3 + c) * plane;
      const src = i * plane * 3;
      for (let p = 0; p < plane; p++) {
        out[dst + p] = f32(f32(frames[src + p * 3 + c]! / 127.5) - 1);
      }
    }
  }
  return out;
}

let hannCache: Float64Array | null = null;
function hannPeriodic(): Float64Array {
  if (!hannCache) {
    hannCache = new Float64Array(N_FFT);
    for (let k = 0; k < N_FFT; k++) {
      hannCache[k] = f32(0.5 - 0.5 * Math.cos((2 * Math.PI * k) / N_FFT));
    }
  }
  return hannCache;
}

let fftCache: Fft | null = null;

export interface LogMel {
  /** Row major (n_mels, frames). */
  data: Float32Array;
  frames: number;
}

export function waveform(audio: Int16Array): Float32Array {
  const out = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) out[i] = f32(audio[i]! / AUDIO_SCALE);
  return out;
}

/** fbank is float32, row major (N_FREQS, N_MELS), exported from torchaudio. */
export function logMel(audio: Int16Array, fbank: Float32Array): LogMel {
  if (fbank.length !== N_FREQS * N_MELS) {
    throw new Error(`mel filterbank has ${fbank.length} values, expected ${N_FREQS * N_MELS}`);
  }
  const x = waveform(audio);
  const n = x.length;
  const pad = N_FFT >> 1;
  if (n <= pad) {
    throw new Error(`reflect padding needs more than ${pad} samples, got ${n}`);
  }
  const frames = 1 + Math.floor(n / HOP_LENGTH);
  const window = hannPeriodic();
  fftCache ??= new Fft(N_FFT);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_FREQS);
  const data = new Float32Array(N_MELS * frames);

  for (let t = 0; t < frames; t++) {
    const origin = t * HOP_LENGTH - pad;
    for (let k = 0; k < N_FFT; k++) {
      let src = origin + k;
      if (src < 0) src = -src;
      else if (src >= n) src = 2 * (n - 1) - src;
      re[k] = x[src]! * window[k]!;
      im[k] = 0;
    }
    fftCache.transform(re, im);
    for (let f = 0; f < N_FREQS; f++) {
      const mag = Math.hypot(f32(re[f]!), f32(im[f]!));
      power[f] = f32(mag) * f32(mag);
    }
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      for (let f = 0; f < N_FREQS; f++) sum += f32(power[f]!) * fbank[f * N_MELS + m]!;
      data[m * frames + t] = f32(Math.log(f32(sum) + LOG_EPSILON));
    }
  }
  return { data, frames };
}

/** (P, 1, N_MELS, PATCH_WIDTH), flattened row major. */
export function melPatches(mel: LogMel): { data: Float32Array; patches: number } {
  const patches = Math.max(1, Math.floor(mel.frames / PATCH_WIDTH));
  const out = new Float32Array(patches * N_MELS * PATCH_WIDTH);
  for (let p = 0; p < patches; p++) {
    for (let m = 0; m < N_MELS; m++) {
      for (let k = 0; k < PATCH_WIDTH; k++) {
        const col = p * PATCH_WIDTH + k;
        const value = col < mel.frames ? mel.data[m * mel.frames + col]! : 0;
        out[(p * N_MELS + m) * PATCH_WIDTH + k] = value;
      }
    }
  }
  return { data: out, patches };
}
