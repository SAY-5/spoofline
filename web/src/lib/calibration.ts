/**
 * Calibration and fusion, mirroring spoofline/calibrate.py and spoofline/fusion.py.
 */

export interface Platt {
  a: number;
  b: number;
}

export interface OperatingPoint {
  threshold: number;
  targetPrecision: number;
  achievedPrecision: number;
  recall: number;
  reachedTarget: boolean;
}

export function plattProbability(platt: Platt, score: number): number {
  return 1 / (1 + Math.exp(-(platt.a * score + platt.b)));
}

export function plattProbabilities(platt: Platt, scores: ArrayLike<number>): Float64Array {
  const out = new Float64Array(scores.length);
  for (let i = 0; i < scores.length; i++) out[i] = plattProbability(platt, scores[i]!);
  return out;
}

function populationStd(values: ArrayLike<number>): number {
  let mean = 0;
  for (let i = 0; i < values.length; i++) mean += values[i]!;
  mean /= values.length;
  let acc = 0;
  for (let i = 0; i < values.length; i++) acc += (values[i]! - mean) ** 2;
  return Math.sqrt(acc / values.length);
}

/** Newton fit with Platt's smoothed targets and a non negative slope, as in PlattCalibrator.fit. */
export function fitPlatt(
  scores: ArrayLike<number>,
  labels: ArrayLike<number>,
  iterations = 100,
  ridge = 1e-6,
): Platt {
  const n = scores.length;
  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] === 1) nPos++;
    else if (labels[i] === 0) nNeg++;
  }
  const hi = nPos ? (nPos + 1) / (nPos + 2) : 0.5;
  const lo = nNeg ? 1 / (nNeg + 2) : 0.5;
  const scale = populationStd(scores) || 1;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = scores[i]! / scale;
  let a = 0;
  let b = 0;
  for (let it = 0; it < iterations; it++) {
    let g0 = 0;
    let g1 = 0;
    let h00 = ridge;
    let h01 = 0;
    let h11 = ridge;
    for (let i = 0; i < n; i++) {
      const p = 1 / (1 + Math.exp(-(a * z[i]! + b)));
      const w = Math.max(p * (1 - p), 1e-9);
      const r = p - (labels[i] === 1 ? hi : lo);
      g0 += r * z[i]!;
      g1 += r;
      h00 += w * z[i]! * z[i]!;
      h01 += w * z[i]!;
      h11 += w;
    }
    const det = h00 * h11 - h01 * h01;
    if (det === 0) break;
    const s0 = (h11 * g0 - h01 * g1) / det;
    const s1 = (h00 * g1 - h01 * g0) / det;
    a -= s0;
    b -= s1;
    if (Math.max(Math.abs(s0), Math.abs(s1)) < 1e-10) break;
  }
  let aScaled = a / scale;
  if (aScaled < 0) {
    aScaled = 0;
    b = Math.log((nPos + 1) / (nNeg + 1));
  }
  return { a: aScaled, b };
}

/** Lowest threshold whose precision reaches the target, as in threshold_at_precision. */
export function thresholdAtPrecision(
  scores: ArrayLike<number>,
  labels: ArrayLike<number>,
  targetPrecision: number,
): OperatingPoint {
  const n = scores.length;
  if (n === 0) throw new Error("cannot pick a threshold from an empty calibration set");
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => scores[i]! - scores[j]!);
  let totalPos = 0;
  for (let i = 0; i < n; i++) if (labels[i] === 1) totalPos++;
  const nPos = Math.max(1, totalPos);
  let best: OperatingPoint | null = null;
  let fallback: OperatingPoint | null = null;
  // Walking the sorted scores upward, everything at or above index k is flagged.
  let tpAbove = totalPos;
  let fpAbove = n - totalPos;
  let k = 0;
  while (k < n) {
    const threshold = scores[order[k]!]!;
    const tp = tpAbove;
    const fp = fpAbove;
    let j = k;
    while (j < n && scores[order[j]!]! === threshold) {
      if (labels[order[j]!] === 1) tpAbove--;
      else fpAbove--;
      j++;
    }
    k = j;
    if (tp + fp === 0) continue;
    const precision = tp / (tp + fp);
    const point: OperatingPoint = {
      threshold,
      targetPrecision,
      achievedPrecision: precision,
      recall: tp / nPos,
      reachedTarget: precision >= targetPrecision,
    };
    if (point.reachedTarget && (best === null || point.recall > best.recall)) best = point;
    if (
      fallback === null ||
      point.achievedPrecision > fallback.achievedPrecision ||
      (point.achievedPrecision === fallback.achievedPrecision && point.recall > fallback.recall)
    ) {
      fallback = point;
    }
  }
  return (best ?? fallback)!;
}

export function fuse(weight: number, pVideo: number, pAudio: number): number {
  return weight * pVideo + (1 - weight) * pAudio;
}

/** Grid search of the fusion weight, as in fit_fusion (np.linspace(0, 1, grid)). */
export function fitFusion(
  pVideo: ArrayLike<number>,
  pAudio: ArrayLike<number>,
  labels: ArrayLike<number>,
  targetPrecision: number,
  grid = 101,
): { weight: number; operating: OperatingPoint } {
  const step = 1 / (grid - 1);
  const fused = new Float64Array(pVideo.length);
  let bestWeight = 0.5;
  let bestPoint: OperatingPoint | null = null;
  for (let g = 0; g < grid; g++) {
    const weight = g === grid - 1 ? 1 : g * step;
    for (let i = 0; i < fused.length; i++) fused[i] = weight * pVideo[i]! + (1 - weight) * pAudio[i]!;
    const point = thresholdAtPrecision(fused, labels, targetPrecision);
    if (bestPoint === null) {
      bestWeight = weight;
      bestPoint = point;
      continue;
    }
    const better =
      Number(point.reachedTarget) !== Number(bestPoint.reachedTarget)
        ? point.reachedTarget
        : point.recall !== bestPoint.recall
          ? point.recall > bestPoint.recall
          : point.achievedPrecision > bestPoint.achievedPrecision;
    if (better) {
      bestWeight = weight;
      bestPoint = point;
    }
  }
  return { weight: bestWeight, operating: bestPoint! };
}

export interface Counts {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export function countsAt(scores: ArrayLike<number>, labels: ArrayLike<number>, threshold: number): Counts {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (let i = 0; i < scores.length; i++) {
    const flagged = scores[i]! >= threshold;
    const positive = labels[i] === 1;
    if (flagged && positive) c.tp++;
    else if (flagged) c.fp++;
    else if (positive) c.fn++;
    else c.tn++;
  }
  return c;
}

export function precisionRecall(c: Counts): { precision: number; recall: number } {
  return {
    precision: c.tp + c.fp ? c.tp / (c.tp + c.fp) : 0,
    recall: c.tp + c.fn ? c.tp / (c.tp + c.fn) : 0,
  };
}

export interface PrCurve {
  thresholds: Float64Array;
  precision: Float64Array;
  recall: Float64Array;
}

/** Precision and recall of `score >= t` at every distinct score, ascending in t. */
export function precisionRecallCurve(scores: ArrayLike<number>, labels: ArrayLike<number>): PrCurve {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => scores[i]! - scores[j]!);
  let totalPos = 0;
  for (let i = 0; i < n; i++) if (labels[i] === 1) totalPos++;
  const thresholds: number[] = [];
  const precision: number[] = [];
  const recall: number[] = [];
  let tp = totalPos;
  let fp = n - totalPos;
  let k = 0;
  while (k < n) {
    const t = scores[order[k]!]!;
    if (tp + fp > 0) {
      thresholds.push(t);
      precision.push(tp / (tp + fp));
      recall.push(totalPos ? tp / totalPos : 0);
    }
    while (k < n && scores[order[k]!]! === t) {
      if (labels[order[k]!] === 1) tp--;
      else fp--;
      k++;
    }
  }
  return {
    thresholds: Float64Array.from(thresholds),
    precision: Float64Array.from(precision),
    recall: Float64Array.from(recall),
  };
}
