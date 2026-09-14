import { fuse, plattProbability } from "./calibration.ts";
import type { CalibrationJson, Stream } from "./types.ts";

export interface StageResult {
  logit: number;
  probability: number;
  threshold: number;
  flags: boolean;
}

export interface ClipScore {
  video: StageResult;
  audio: StageResult;
  fused: { probability: number; threshold: number; weight: number; decision: "attack" | "bonafide" };
}

export function stage(calibration: CalibrationJson, stream: Stream, logit: number): StageResult {
  const entry = calibration[stream];
  const probability = plattProbability(entry.calibrator, logit);
  return { logit, probability, threshold: entry.operating.threshold, flags: probability >= entry.operating.threshold };
}

/** The decision spoofline.pipeline.score_single_clip makes from the two raw logits. */
export function scoreClip(calibration: CalibrationJson, videoLogit: number, audioLogit: number): ClipScore {
  const video = stage(calibration, "video", videoLogit);
  const audio = stage(calibration, "audio", audioLogit);
  const weight = calibration.fused.weight;
  const threshold = calibration.fused.operating.threshold;
  const probability = fuse(weight, video.probability, audio.probability);
  return {
    video,
    audio,
    fused: { probability, threshold, weight, decision: probability >= threshold ? "attack" : "bonafide" },
  };
}
