import type { DemoData } from "../lib/data.ts";
import type { Detector, RunJson } from "../lib/types.ts";
import { fixed } from "../format.ts";
import { Axis, ChartFrame, linear, spread, type Frame } from "./charts.tsx";

const DETECTORS: Detector[] = ["video", "audio", "fused"];
const NAME: Record<Detector, string> = { video: "video", audio: "audio", fused: "fused" };

function Marker({ detector, x, y }: { detector: Detector; x: number; y: number }) {
  if (detector === "video") return <rect className="mk mk-video" x={x - 4.5} y={y - 4.5} width={9} height={9} />;
  if (detector === "audio") return <path className="mk mk-audio" d={`M${x},${y - 5.5}L${x + 5.5},${y + 4.5}L${x - 5.5},${y + 4.5}Z`} />;
  return <circle className="mk mk-fused" cx={x} cy={y} r={5} />;
}

function SlopeChart({ run, metric, domain, target }: { run: RunJson; metric: "precision" | "recall"; domain: [number, number]; target?: number }) {
  const frame: Frame = { width: 400, height: 320, left: 56, right: 16, top: 20, bottom: 44 };
  const xSeen = 130;
  const xUnseen = 300;
  const y = linear(domain[0], domain[1], frame.height - frame.bottom, frame.top);
  const step = (domain[1] - domain[0]) / 4;
  const ticks = Array.from({ length: 5 }, (_, i) => domain[0] + i * step).map((v) => ({ value: v, label: v.toFixed(2) }));
  const seen = DETECTORS.map((d) => run.metrics.seen_test[d][metric]);
  const unseen = DETECTORS.map((d) => run.metrics.unseen_test[d][metric]);
  const leftLabels = spread(seen.map((v) => y(v)), 15);
  const rightLabels = spread(unseen.map((v) => y(v)), 15);
  const summary = DETECTORS.map((d, i) => `${d} ${fixed(seen[i]!)} to ${fixed(unseen[i]!)}`).join(", ");
  return (
    <figure className="figure">
      <figcaption className="figure-cap">
        <strong>{metric}</strong> seen families to unseen families
      </figcaption>
      <ChartFrame frame={frame} label={`${metric} from the seen to the unseen test split: ${summary}`}>
        <Axis frame={frame} orient="left" scale={y} ticks={ticks} title={metric} />
        <g className="axis">
          <line className="axis-base" x1={frame.left} x2={frame.width - frame.right} y1={frame.height - frame.bottom} y2={frame.height - frame.bottom} />
          <text className="tick-label" x={xSeen} y={frame.height - frame.bottom + 16} textAnchor="middle">
            seen
          </text>
          <text className="tick-label" x={xUnseen} y={frame.height - frame.bottom + 16} textAnchor="middle">
            unseen
          </text>
          <text className="axis-title" x={(xSeen + xUnseen) / 2} y={frame.height - 6} textAnchor="middle">
            test split
          </text>
        </g>
        {target !== undefined && (
          <g>
            <line className="target-line" x1={frame.left} x2={frame.width - frame.right} y1={y(target)} y2={y(target)} />
            <text className="target-label" x={frame.width - frame.right - 2} y={y(target) - 6} textAnchor="end">
              target {fixed(target, 2)}
            </text>
          </g>
        )}
        {DETECTORS.map((d, i) => (
          <g key={d} className={`slope slope-${d}`}>
            <line x1={xSeen} x2={xUnseen} y1={y(seen[i]!)} y2={y(unseen[i]!)} />
            <Marker detector={d} x={xSeen} y={y(seen[i]!)} />
            <Marker detector={d} x={xUnseen} y={y(unseen[i]!)} />
            <text className="slope-label" x={xSeen - 12} y={leftLabels[i]!} dy="0.32em" textAnchor="end">
              {fixed(seen[i]!)}
            </text>
            <text className="slope-label" x={xUnseen + 12} y={rightLabels[i]!} dy="0.32em">
              {NAME[d]} {fixed(unseen[i]!)}
            </text>
          </g>
        ))}
      </ChartFrame>
    </figure>
  );
}

export function UnseenSection({ data }: { data: DemoData }) {
  const run = data.reference.run;
  const u = run.metrics.unseen_test;
  const s = run.metrics.seen_test;
  const target = data.manifest.target_precision;
  const bonaUnseen = u.fused.n - u.fused.n_positive;
  const rates = run.family_rates.unseen_test;
  const splice = rates["video_splice"];
  const vocoder = rates["audio_vocoder"];
  return (
    <div className="unseen">
      <div className="unseen-charts">
        <SlopeChart run={run} metric="precision" domain={[0.8, 1]} target={target} />
        <SlopeChart run={run} metric="recall" domain={[0, 1]} />
        <p className="legend legend-wide">
          <span className="key-shape key-shape-video" aria-hidden="true" /> video
          <span className="key-shape key-shape-audio" aria-hidden="true" /> audio
          <span className="key-shape key-shape-fused" aria-hidden="true" /> fused
          <span className="legend-note">precision axis starts at 0.80</span>
        </p>
      </div>
      <div className="unseen-copy">
        <p className="callout">
          On unseen families the fused precision, <strong>{fixed(u.fused.precision)}</strong>, is below video alone at{" "}
          <strong>{fixed(u.video.precision)}</strong>, and below the {fixed(target, 2)} target.
        </p>
        <p>
          The video stream earns its perfect precision by staying quiet. It flags {u.video.counts.tp} of {u.video.n_positive} unseen
          family attacks with {u.video.counts.fp} false alarms on {bonaUnseen} bona fide clips, because it is structurally blind to a
          genuine face with a vocoded voice.
        </p>
        <p>
          Fusion adds the audio stream, which can hear those voices. Recall rises to {u.fused.counts.tp} of {u.fused.n_positive}, but the
          fused score also inherits the audio stream's false alarms: {u.fused.counts.fp} of {bonaUnseen} bona fide clips. That is{" "}
          {u.fused.counts.tp} / ({u.fused.counts.tp} + {u.fused.counts.fp}) = {fixed(u.fused.precision)}. The fused weight is{" "}
          {fixed(data.manifest.calibration.fused.weight, 2)} on video, a soft OR, so a moderate audio score clears the threshold on its own.
        </p>
        <p>
          Across the seen to unseen boundary the fused operating point moves from {fixed(s.fused.precision)} to{" "}
          {fixed(u.fused.precision)}, and F1 goes from {fixed(s.fused.f1)} to {fixed(u.fused.f1)} against {fixed(u.video.f1)} for video alone.
          {splice && vocoder
            ? ` The held out families themselves are the hard part: ${splice.detected} of ${splice.n} face splices and ${vocoder.detected} of ${vocoder.n} vocoded voices are caught.`
            : ""}
        </p>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <caption>Measured on the test identities at the calibrated operating points</caption>
          <thead>
            <tr>
              <th scope="col">detector</th>
              <th scope="col">seen P</th>
              <th scope="col">seen R</th>
              <th scope="col">unseen P</th>
              <th scope="col">unseen R</th>
              <th scope="col">unseen caught</th>
              <th scope="col">unseen false alarms</th>
              <th scope="col">unseen AUC</th>
            </tr>
          </thead>
          <tbody>
            {DETECTORS.map((d) => (
              <tr key={d} className={d === "fused" ? "row-accent" : ""}>
                <th scope="row">{d}</th>
                <td>{fixed(s[d].precision)}</td>
                <td>{fixed(s[d].recall)}</td>
                <td>{fixed(u[d].precision)}</td>
                <td>{fixed(u[d].recall)}</td>
                <td>
                  {u[d].counts.tp} / {u[d].n_positive}
                </td>
                <td>
                  {u[d].counts.fp} / {u[d].n - u[d].n_positive}
                </td>
                <td>{fixed(u[d].auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
