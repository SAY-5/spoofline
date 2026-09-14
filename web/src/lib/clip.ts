import type { CorpusShape } from "./types.ts";

export interface ClipData {
  frames: Uint8Array;
  audio: Int16Array;
}

/**
 * Undo the lossless packing written by web/scripts/export.py. The inflated buffer holds
 * uint8 frame residuals (T, H, W, 3), a left difference along W then a temporal difference
 * along T, modulo 256, followed by int16 little endian first differences of the audio.
 */
export function decodeClip(inflated: ArrayBuffer, corpus: CorpusShape): ClipData {
  const size = corpus.frame_size;
  const plane = size * size * 3;
  const frameBytes = corpus.n_frames * plane;
  const expected = frameBytes + corpus.n_samples * 2;
  if (inflated.byteLength !== expected) {
    throw new Error(`clip holds ${inflated.byteLength} bytes, expected ${expected}`);
  }
  const residual = new Uint8Array(inflated, 0, frameBytes);
  const left = new Uint8Array(frameBytes);
  for (let i = 0; i < frameBytes; i++) {
    left[i] = i < plane ? residual[i]! : (residual[i]! + left[i - plane]!) & 255;
  }
  const frames = new Uint8Array(frameBytes);
  for (let t = 0; t < corpus.n_frames; t++) {
    for (let y = 0; y < size; y++) {
      const row = (t * size + y) * size * 3;
      for (let c = 0; c < 3; c++) frames[row + c] = left[row + c]!;
      for (let x = 1; x < size; x++) {
        for (let c = 0; c < 3; c++) {
          const idx = row + x * 3 + c;
          frames[idx] = (left[idx]! + frames[idx - 3]!) & 255;
        }
      }
    }
  }
  const view = new DataView(inflated, frameBytes);
  const audio = new Int16Array(corpus.n_samples);
  let previous = 0;
  for (let i = 0; i < corpus.n_samples; i++) {
    audio[i] = view.getInt16(i * 2, true) + previous;
    previous = audio[i]!;
  }
  return { frames, audio };
}

export function clipShape(corpus: CorpusShape) {
  return { frames: corpus.n_frames, size: corpus.frame_size, samples: corpus.n_samples };
}
