import { useEffect, useRef } from "react";
import type { LogMel } from "../lib/features.ts";

const LOW = -14;
const HIGH = 4;

function buildRamp(): Uint8ClampedArray {
  const stops: [number, [number, number, number]][] = [
    [0, [16, 17, 18]],
    [0.45, [96, 64, 18]],
    [0.8, [242, 179, 61]],
    [1, [255, 244, 214]],
  ];
  const ramp = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1]![0]) k++;
    const [t0, c0] = stops[k]!;
    const [t1, c1] = stops[k + 1]!;
    const u = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c++) ramp[i * 3 + c] = c0[c]! + (c1[c]! - c0[c]!) * u;
  }
  return ramp;
}
const RAMP = buildRamp();

interface SpectrogramProps {
  mel: LogMel;
  nMels: number;
  label: string;
  playhead?: number | null;
  className?: string;
}

/** Log mel spectrogram on one fixed scale for every clip, low frequencies at the bottom. */
export function Spectrogram({ mel, nMels, label, playhead = null, className }: SpectrogramProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const image = ctx.createImageData(mel.frames, nMels);
    for (let m = 0; m < nMels; m++) {
      const row = nMels - 1 - m;
      for (let t = 0; t < mel.frames; t++) {
        const v = (mel.data[m * mel.frames + t]! - LOW) / (HIGH - LOW);
        const idx = Math.max(0, Math.min(255, Math.round(v * 255)));
        const p = (row * mel.frames + t) * 4;
        image.data[p] = RAMP[idx * 3]!;
        image.data[p + 1] = RAMP[idx * 3 + 1]!;
        image.data[p + 2] = RAMP[idx * 3 + 2]!;
        image.data[p + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [mel, nMels]);
  return (
    <div className={`spectro ${className ?? ""}`}>
      <canvas ref={ref} width={mel.frames} height={nMels} role="img" aria-label={label} />
      {playhead !== null && (
        <span className="spectro-playhead" aria-hidden="true" style={{ left: `${Math.min(1, Math.max(0, playhead)) * 100}%` }} />
      )}
    </div>
  );
}
