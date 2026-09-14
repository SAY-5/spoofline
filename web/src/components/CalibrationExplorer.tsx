import { useMemo, useRef, useState, type PointerEvent } from "react";
import {
  countsAt,
  fuse,
  plattProbabilities,
  plattProbability,
  precisionRecall,
  precisionRecallCurve,
  thresholdAtPrecision,
  type OperatingPoint,
  type Platt,
  type PrCurve,
} from "../lib/calibration.ts";
import type { DemoData } from "../lib/data.ts";
import type { Detector, TestSplit } from "../lib/types.ts";
import { fixed } from "../format.ts";
import { Axis, ChartFrame, linear, logOdds, type Frame } from "./charts.tsx";

const TARGET_MIN = 0.8;
const TARGET_MAX = 1;
const P_TICKS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99];

function tickLabel(p: number): string {
  return p < 0.01 ? String(p) : p.toString();
}

interface PlattChartProps {
  stream: "video" | "audio";
  logits: number[];
  labels: number[];
  platt: Platt;
  threshold: number;
}

function PlattChart({ stream, logits, labels, platt, threshold }: PlattChartProps) {
  const frame: Frame = { width: 480, height: 300, left: 52, right: 16, top: 16, bottom: 44 };
  const lo = Math.floor(Math.min(...logits)) - 1;
  const hi = Math.ceil(Math.max(...logits)) + 1;
  const x = linear(lo, hi, frame.left, frame.width - frame.right);
  const y = linear(0, 1, frame.height - frame.bottom, frame.top);
  const steps = 160;
  let path = "";
  for (let i = 0; i <= steps; i++) {
    const s = lo + ((hi - lo) * i) / steps;
    path += `${i ? "L" : "M"}${x(s).toFixed(1)},${y(plattProbability(platt, s)).toFixed(1)}`;
  }
  const span = hi - lo;
  const every = span > 30 ? 10 : span > 12 ? 5 : 2;
  const xTicks = [];
  for (let v = Math.ceil(lo / every) * every; v <= hi; v += every) xTicks.push({ value: v, label: String(v) });
  const flagged = logits.filter((s) => plattProbability(platt, s) >= threshold).length;
  return (
    <figure className="figure">
      <figcaption className="figure-cap">
        <strong>{stream} stream</strong> Platt map p = sigmoid({fixed(platt.a)} s {platt.b < 0 ? "-" : "+"} {fixed(Math.abs(platt.b))})
      </figcaption>
      <ChartFrame
        frame={frame}
        label={`${stream} Platt calibration curve over ${logits.length} calibration clips; ${flagged} sit at or above the threshold ${fixed(threshold, 4)}`}
      >
        <Axis frame={frame} orient="bottom" scale={x} ticks={xTicks} title="raw logit s" />
        <Axis
          frame={frame}
          orient="left"
          scale={y}
          ticks={[0, 0.25, 0.5, 0.75, 1].map((v) => ({ value: v, label: String(v) }))}
          title="calibrated probability"
        />
        <path className="curve" d={path} />
        {logits.map((s, i) =>
          labels[i] === 1 ? (
            <circle key={i} className="pt pt-attack" cx={x(s)} cy={y(plattProbability(platt, s))} r={3.2} />
          ) : (
            <circle key={i} className="pt pt-bona" cx={x(s)} cy={y(plattProbability(platt, s))} r={3.2} />
          ),
        )}
        <line className="threshold-line" x1={frame.left} x2={frame.width - frame.right} y1={y(threshold)} y2={y(threshold)} />
        <text className="threshold-label" x={frame.left + 6} y={y(threshold) - 6}>
          threshold {fixed(threshold, 4)}
        </text>
      </ChartFrame>
    </figure>
  );
}

interface PrChartProps {
  title: string;
  curve: PrCurve;
  target: number;
  point: OperatingPoint;
  onTarget: (value: number) => void;
}

function PrChart({ title, curve, target, point, onTarget }: PrChartProps) {
  const frame: Frame = { width: 480, height: 300, left: 52, right: 16, top: 16, bottom: 44 };
  const lo = -8;
  const hi = 8;
  const x = linear(lo, hi, frame.left, frame.width - frame.right);
  const y = linear(0, 1, frame.height - frame.bottom, frame.top);
  const svgRef = useRef<SVGGElement>(null);
  const dragging = useRef(false);
  const clampX = (p: number) => Math.min(hi, Math.max(lo, logOdds(p)));
  const line = (values: Float64Array) => {
    let d = "";
    for (let i = 0; i < curve.thresholds.length; i++) {
      const px = x(clampX(curve.thresholds[i]!)).toFixed(1);
      const py = y(values[i]!).toFixed(1);
      d += `${i ? "L" : "M"}${px},${py}`;
    }
    return d;
  };
  const move = (event: PointerEvent<SVGRectElement>) => {
    if (!dragging.current) return;
    const svg = svgRef.current?.ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const py = ((event.clientY - rect.top) / rect.height) * frame.height;
    const value = Math.round(Math.min(TARGET_MAX, Math.max(TARGET_MIN, y.invert(py))) * 1000) / 1000;
    onTarget(value);
  };
  const tx = x(clampX(point.threshold));
  return (
    <figure className="figure">
      <figcaption className="figure-cap">
        <strong>{title}</strong> precision and recall against the threshold, calibration split
      </figcaption>
      <ChartFrame
        frame={frame}
        label={`${title}: at target precision ${fixed(target)} the threshold is ${fixed(point.threshold, 4)}, precision ${fixed(point.achievedPrecision)}, recall ${fixed(point.recall)}`}
      >
        <g ref={svgRef}>
          <Axis
            frame={frame}
            orient="bottom"
            scale={(p) => x(logOdds(p))}
            ticks={P_TICKS.map((p) => ({ value: p, label: tickLabel(p) }))}
            title="threshold on calibrated probability (log odds spacing)"
          />
          <Axis
            frame={frame}
            orient="left"
            scale={y}
            ticks={[0, 0.25, 0.5, 0.75, 1].map((v) => ({ value: v, label: String(v) }))}
            title="precision, recall"
          />
          <path className="series series-precision" d={line(curve.precision)} />
          <path className="series series-recall" d={line(curve.recall)} />
          <text className="series-label" x={frame.left + 6} y={y(curve.precision[0] ?? 0) + 18}>
            precision
          </text>
          <text
            className="series-label series-label-muted"
            x={frame.width - frame.right - 6}
            y={y(curve.recall[curve.recall.length - 1] ?? 0) - 10}
            textAnchor="end"
          >
            recall
          </text>
          <line className="operating-line" x1={tx} x2={tx} y1={frame.top} y2={frame.height - frame.bottom} />
          <circle className="operating-dot" cx={tx} cy={y(point.recall)} r={4.5} />
          <line className="target-line" x1={frame.left} x2={frame.width - frame.right} y1={y(target)} y2={y(target)} />
          <text className="target-label" x={frame.width - frame.right - 32} y={y(target) + 17} textAnchor="end">
            target {fixed(target)}
          </text>
          <rect
            className="target-handle"
            x={frame.left}
            width={frame.width - frame.left - frame.right}
            y={y(target) - 12}
            height={24}
            onPointerDown={(e) => {
              dragging.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
            }}
            onPointerMove={move}
            onPointerUp={(e) => {
              dragging.current = false;
              e.currentTarget.releasePointerCapture(e.pointerId);
            }}
            onPointerCancel={() => {
              dragging.current = false;
            }}
          />
          <g className="target-grip" transform={`translate(${frame.width - frame.right - 14},${y(target)})`} aria-hidden="true">
            <rect x={-10} y={-7} width={20} height={14} rx={3} />
            <line x1={-4} x2={-4} y1={-3} y2={3} />
            <line x1={0} x2={0} y1={-3} y2={3} />
            <line x1={4} x2={4} y1={-3} y2={3} />
          </g>
        </g>
      </ChartFrame>
    </figure>
  );
}

export function CalibrationExplorer({ data }: { data: DemoData }) {
  const { manifest, calib, test } = data;
  const cal = manifest.calibration;
  const [target, setTarget] = useState(manifest.target_precision);

  const base = useMemo(() => {
    const labels = calib.label;
    const pv = plattProbabilities(cal.video.calibrator, calib.video_logit);
    const pa = plattProbabilities(cal.audio.calibrator, calib.audio_logit);
    const pf = pv.map((v, i) => fuse(cal.fused.weight, v, pa[i]!));
    const tv = plattProbabilities(cal.video.calibrator, test.video_logit);
    const ta = plattProbabilities(cal.audio.calibrator, test.audio_logit);
    const tf = tv.map((v, i) => fuse(cal.fused.weight, v, ta[i]!));
    return {
      labels,
      calibScores: { video: pv, audio: pa, fused: pf } as Record<Detector, Float64Array>,
      testScores: { video: tv, audio: ta, fused: tf } as Record<Detector, Float64Array>,
      curves: {
        video: precisionRecallCurve(pv, labels),
        audio: precisionRecallCurve(pa, labels),
        fused: precisionRecallCurve(pf, labels),
      } as Record<Detector, PrCurve>,
    };
  }, [cal, calib, test]);

  const points = useMemo(() => {
    const out = {} as Record<Detector, OperatingPoint>;
    for (const d of ["video", "audio", "fused"] as Detector[]) {
      out[d] = thresholdAtPrecision(base.calibScores[d], base.labels, target);
    }
    return out;
  }, [base, target]);

  const testAt = (detector: Detector, split: TestSplit) => {
    const scores: number[] = [];
    const labels: number[] = [];
    test.split.forEach((s, i) => {
      if (s !== split) return;
      scores.push(base.testScores[detector][i]!);
      labels.push(test.label[i]!);
    });
    const counts = countsAt(scores, labels, points[detector].threshold);
    return { ...precisionRecall(counts), counts, positives: counts.tp + counts.fn };
  };

  const atDefault = Math.abs(target - manifest.target_precision) < 1e-9;
  const names: Record<Detector, string> = { video: "Video", audio: "Audio", fused: `Fused, w = ${fixed(cal.fused.weight, 2)}` };

  return (
    <div className="calib">
      <div className="calib-control">
        <label htmlFor="target-precision" className="control-label">
          Target precision <output htmlFor="target-precision">{fixed(target)}</output>
        </label>
        <input
          id="target-precision"
          type="range"
          min={TARGET_MIN}
          max={TARGET_MAX}
          step={0.001}
          value={target}
          onChange={(e) => setTarget(Number(e.target.value))}
        />
        <button type="button" className="btn btn-ghost" onClick={() => setTarget(manifest.target_precision)} disabled={atDefault}>
          Reset to {fixed(manifest.target_precision, 2)}
        </button>
        <p className="control-hint">Drag the amber target line on any precision chart, or use the slider. The fusion weight stays at the fitted {fixed(cal.fused.weight, 2)}.</p>
      </div>

      <div className="calib-grid">
        <PlattChart stream="video" logits={calib.video_logit} labels={calib.label} platt={cal.video.calibrator} threshold={points.video.threshold} />
        <PrChart title="Video" curve={base.curves.video} target={target} point={points.video} onTarget={setTarget} />
        <PlattChart stream="audio" logits={calib.audio_logit} labels={calib.label} platt={cal.audio.calibrator} threshold={points.audio.threshold} />
        <PrChart title="Audio" curve={base.curves.audio} target={target} point={points.audio} onTarget={setTarget} />
        <PrChart title="Fused" curve={base.curves.fused} target={target} point={points.fused} onTarget={setTarget} />
        <div className="legend-block">
          <p className="legend">
            <span className="key key-attack" aria-hidden="true" /> attack clip
            <span className="key key-bona" aria-hidden="true" /> bona fide clip
          </p>
          <p className="legend">
            <span className="key key-line" aria-hidden="true" /> precision
            <span className="key key-line key-line-muted" aria-hidden="true" /> recall
            <span className="key key-line key-line-accent" aria-hidden="true" /> target and threshold
          </p>
          <p className="note">
            Thresholds are chosen the way calibrate.py chooses them: the lowest calibrated probability whose precision on the
            {" "}{calib.clip_ids.length} calibration clips reaches the target. Lowest means the most recall available at that
            precision. The Platt map itself is fitted against each stream's own modality label, so a vocoded voice on a real face
            counts as a negative for the video map.
          </p>
        </div>
      </div>

      <div className="table-wrap">
        <table className="data-table">
          <caption>
            Operating points at target {fixed(target)}
            {atDefault ? ", the exported run" : ", recomputed in the browser"}
          </caption>
          <thead>
            <tr>
              <th scope="col">detector</th>
              <th scope="col">threshold</th>
              <th scope="col">calib P</th>
              <th scope="col">calib R</th>
              <th scope="col">seen P</th>
              <th scope="col">seen R</th>
              <th scope="col">unseen P</th>
              <th scope="col">unseen R</th>
            </tr>
          </thead>
          <tbody aria-live="polite">
            {(["video", "audio", "fused"] as Detector[]).map((d) => {
              const seen = testAt(d, "seen_test");
              const unseen = testAt(d, "unseen_test");
              return (
                <tr key={d}>
                  <th scope="row">{names[d]}</th>
                  <td>{fixed(points[d].threshold, 4)}</td>
                  <td>
                    {fixed(points[d].achievedPrecision)}
                    {points[d].reachedTarget ? "" : " (target not met)"}
                  </td>
                  <td>{fixed(points[d].recall)}</td>
                  <td>{fixed(seen.precision)}</td>
                  <td>{fixed(seen.recall)}</td>
                  <td>{fixed(unseen.precision)}</td>
                  <td>{fixed(unseen.recall)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
