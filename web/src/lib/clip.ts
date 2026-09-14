import type { CorpusShape } from "./types.ts";

export interface ClipData {
  frames: Uint8Array;
  audio: Int16Array;
}

/** A clip file is uint8 frames (T, H, W, 3) followed by int16 little endian samples. */
export function decodeClip(buffer: ArrayBuffer, corpus: CorpusShape): ClipData {
  const frameBytes = corpus.n_frames * corpus.frame_size * corpus.frame_size * 3;
  const expected = frameBytes + corpus.n_samples * 2;
  if (buffer.byteLength !== expected) {
    throw new Error(`clip is ${buffer.byteLength} bytes, expected ${expected}`);
  }
  const frames = new Uint8Array(buffer, 0, frameBytes);
  const view = new DataView(buffer, frameBytes);
  const audio = new Int16Array(corpus.n_samples);
  for (let i = 0; i < corpus.n_samples; i++) audio[i] = view.getInt16(i * 2, true);
  return { frames, audio };
}

export function clipShape(corpus: CorpusShape) {
  return { frames: corpus.n_frames, size: corpus.frame_size, samples: corpus.n_samples };
}
