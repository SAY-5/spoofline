import type { ReactNode } from "react";

export interface Frame {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export function linear(d0: number, d1: number, r0: number, r1: number) {
  const k = (r1 - r0) / (d1 - d0);
  const scale = (v: number) => r0 + (v - d0) * k;
  scale.invert = (px: number) => d0 + (px - r0) / k;
  return scale;
}

export function logOdds(p: number): number {
  const c = Math.min(1 - 1e-9, Math.max(1e-9, p));
  return Math.log(c / (1 - c));
}

interface AxisProps {
  frame: Frame;
  ticks: { value: number; label: string }[];
  scale: (v: number) => number;
  orient: "bottom" | "left";
  title: string;
  grid?: boolean;
}

export function Axis({ frame, ticks, scale, orient, title, grid = true }: AxisProps) {
  const { width, height, left, right, top, bottom } = frame;
  if (orient === "bottom") {
    const y = height - bottom;
    return (
      <g className="axis">
        <line className="axis-base" x1={left} x2={width - right} y1={y} y2={y} />
        {ticks.map((t) => (
          <g key={t.label} transform={`translate(${scale(t.value)},0)`}>
            {grid && <line className="grid" y1={top} y2={y} />}
            <line className="axis-tick" y1={y} y2={y + 4} />
            <text className="tick-label" y={y + 16} textAnchor="middle">
              {t.label}
            </text>
          </g>
        ))}
        <text className="axis-title" x={left + (width - left - right) / 2} y={height - 6} textAnchor="middle">
          {title}
        </text>
      </g>
    );
  }
  return (
    <g className="axis">
      <line className="axis-base" x1={left} x2={left} y1={top} y2={height - bottom} />
      {ticks.map((t) => (
        <g key={t.label} transform={`translate(0,${scale(t.value)})`}>
          {grid && <line className="grid" x1={left} x2={width - right} />}
          <text className="tick-label" x={left - 8} dy="0.32em" textAnchor="end">
            {t.label}
          </text>
        </g>
      ))}
      <text
        className="axis-title"
        transform={`translate(14,${top + (height - top - bottom) / 2}) rotate(-90)`}
        textAnchor="middle"
      >
        {title}
      </text>
    </g>
  );
}

export function ChartFrame({ frame, label, children, className }: { frame: Frame; label: string; children: ReactNode; className?: string }) {
  return (
    <svg className={`chart ${className ?? ""}`} viewBox={`0 0 ${frame.width} ${frame.height}`} role="img" aria-label={label}>
      {children}
    </svg>
  );
}

/**
 * Baseline for a label drawn 6px above a horizontal reference line. The baseline is held
 * at least `clearance` px above the bottom axis, where points with probability near zero
 * gather, so a line close to the axis does not put its label over that cluster.
 */
export function lineLabelY(frame: Frame, lineY: number, clearance = 14): number {
  return Math.min(lineY, frame.height - frame.bottom - clearance) - 6;
}

/** Spread label positions so that no two sit closer than `gap` pixels. */
export function spread(positions: number[], gap: number): number[] {
  const order = positions.map((p, i) => [p, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(positions.length);
  let last = -Infinity;
  for (const [p, i] of order) {
    const placed = Math.max(p, last + gap);
    out[i] = placed;
    last = placed;
  }
  return out;
}
