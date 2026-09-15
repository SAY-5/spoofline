import { attribute, fuse, logisticFuse, plattProbability } from "./calibration.ts";
import type { Attribution, CalibrationJson, Stream } from "./types.ts";

export interface StageResult {
  logit: number;
  probability: number;
  threshold: number;
  flags: boolean;
}

export interface FusionResult {
  probability: number;
  threshold: number;
  decision: "attack" | "bonafide";
}

export interface ClipScore {
  video: StageResult;
  audio: StageResult;
  fused: FusionResult & { weight: number };
  logistic: FusionResult;
  triggeredBy: Attribution;
}

export function stage(calibration: CalibrationJson, stream: Stream, logit: number): StageResult {
  const entry = calibration[stream];
  const probability = plattProbability(entry.calibrator, logit);
  return { logit, probability, threshold: entry.operating.threshold, flags: probability >= entry.operating.threshold };
}

/** Everything spoofline.scoring.RunScorer.describe reports for one clip, from the two logits. */
export function scoreClip(calibration: CalibrationJson, videoLogit: number, audioLogit: number): ClipScore {
  const video = stage(calibration, "video", videoLogit);
  const audio = stage(calibration, "audio", audioLogit);
  const weight = calibration.fused.weight;
  const threshold = calibration.fused.operating.threshold;
  const probability = fuse(weight, video.probability, audio.probability);
  const learned = calibration.logistic;
  const logisticProbability = logisticFuse(learned, video.probability, audio.probability);
  const decide = (pv: number, pa: number) => fuse(weight, pv, pa) >= threshold;
  return {
    video,
    audio,
    fused: { probability, threshold, weight, decision: probability >= threshold ? "attack" : "bonafide" },
    logistic: {
      probability: logisticProbability,
      threshold: learned.operating.threshold,
      decision: logisticProbability >= learned.operating.threshold ? "attack" : "bonafide",
    },
    triggeredBy: attribute(decide, video.probability, audio.probability),
  };
}
