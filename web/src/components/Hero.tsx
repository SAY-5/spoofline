import { useMemo } from "react";
import { plattProbability, fuse } from "../lib/calibration.ts";
import type { DemoData } from "../lib/data.ts";
import { fixed } from "../format.ts";

function CatchStrip({ label, flags, accent }: { label: string; flags: boolean[]; accent: boolean }) {
  const cols = 20;
  const cell = 11;
  const pitch = 14;
  const rows = Math.ceil(flags.length / cols);
  const caught = flags.filter(Boolean).length;
  return (
    <figure className="strip">
      <figcaption>
        <span className="strip-label">{label}</span>
        <span className="strip-count">
          <strong>{caught}</strong> of {flags.length}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${cols * pitch} ${rows * pitch}`}
        role="img"
        aria-label={`${label}: ${caught} of ${flags.length} unseen family attacks flagged`}
      >
        {flags.map((hit, i) => (
          <rect
            key={i}
            className={`cell ${hit ? (accent ? "cell-accent" : "cell-ink") : "cell-miss"}`}
            style={{ animationDelay: `${i * 9}ms` }}
            x={(i % cols) * pitch + 1.5}
            y={Math.floor(i / cols) * pitch + 1.5}
            width={cell}
            height={cell}
            rx={1.5}
          />
        ))}
      </svg>
    </figure>
  );
}

export function Hero({ data }: { data: DemoData | null }) {
  const view = useMemo(() => {
    if (!data) return null;
    const { reference, test, manifest } = data;
    const cal = manifest.calibration;
    const metrics = reference.run.metrics;
    const video: boolean[] = [];
    const fused: boolean[] = [];
    test.clip_ids.forEach((_, i) => {
      if (test.split[i] !== "unseen_test" || test.label[i] !== 1) return;
      const pv = plattProbability(cal.video.calibrator, test.video_logit[i]!);
      const pa = plattProbability(cal.audio.calibrator, test.audio_logit[i]!);
      video.push(pv >= cal.video.operating.threshold);
      fused.push(fuse(cal.fused.weight, pv, pa) >= cal.fused.operating.threshold);
    });
    return { metrics, target: manifest.target_precision, video, fused };
  }, [data]);

  const m = view?.metrics;
  const manifest = data?.manifest;
  const target = view?.target ?? 0.95;
  const seenP = m?.seen_test.fused.precision;
  const unseenP = m?.unseen_test.fused.precision;
  const scale = (p: number) => ((p - 0.9) / 0.1) * 100;

  return (
    <section className="hero" aria-labelledby="hero-title">
      <p className="eyebrow">Two stream audio and video spoof detection</p>
      <h1 id="hero-title">
        <span className="sr-only">Fused precision on seen and unseen attack families</span>
        <span className="h1-line" aria-hidden="true">
          Fused precision
        </span>
        <span className="h1-figures" aria-hidden="true">
          <span className="fig">
            <span className="fig-num">{seenP !== undefined ? fixed(seenP) : "0.000"}</span>
            <span className="fig-cap">seen attack families</span>
          </span>
          <span className="fig fig-accent">
            <span className="fig-num">{unseenP !== undefined ? fixed(unseenP) : "0.000"}</span>
            <span className="fig-cap">families it never trained on</span>
          </span>
        </span>
      </h1>
      {seenP !== undefined && unseenP !== undefined && (
        <p className="sr-only">
          Fused precision {fixed(seenP)} on seen attack families and {fixed(unseenP)} on families it never trained on.
        </p>
      )}
      {manifest && (
        <p className="hero-provenance">
          Measured offline on the synthetic {manifest.corpus.n_clips} clip corpus, seed {manifest.seed}, one held out
          pair ({manifest.unseen_families.join(" and ")}), weights from commit{" "}
          {manifest.trained_from_commit.slice(0, 7)}. Single run, no variance estimate.
        </p>
      )}
      <div className="gauge" aria-hidden={m ? undefined : true}>
        <div className="gauge-track">
          <span className="gauge-target" style={{ left: `${scale(target)}%` }}>
            <span>target {fixed(target, 2)}</span>
          </span>
          {m && (
            <>
              <span className="gauge-dot" style={{ left: `${scale(m.seen_test.fused.precision)}%` }} title="seen" />
              <span className="gauge-dot gauge-dot-accent" style={{ left: `${scale(m.unseen_test.fused.precision)}%` }} title="unseen" />
            </>
          )}
        </div>
        <div className="gauge-axis">
          <span>0.90</span>
          <span>0.95</span>
          <span>1.00</span>
        </div>
      </div>
      <div className="hero-grid">
        <p className="hero-copy">
          Two CNN-LSTM detectors, one for frames and one for audio, each Platt calibrated on held out clips with its
          threshold set for a {fixed(target, 2)} precision target, then fused. Evaluation leaves attack families out of
          training entirely. {m ? (
            <>
              Video alone holds <strong>{fixed(m.unseen_test.video.precision)}</strong> precision on the unseen families
              but flags only <strong>{m.unseen_test.video.counts.tp} of {m.unseen_test.video.n_positive}</strong> attacks.
              The fused score flags <strong>{m.unseen_test.fused.counts.tp} of {m.unseen_test.fused.n_positive}</strong>.
            </>
          ) : null}
        </p>
        {view && (
          <div className="strips">
            <CatchStrip label="Video alone" flags={view.video} accent={false} />
            <CatchStrip label="Fused" flags={view.fused} accent />
            <p className="strips-note">
              Each cell is one of the {view.fused.length} attacked clips in the unseen family test split. The cells
              replay the exported PyTorch logits through the same Platt maps and thresholds; they are not scored in
              this tab. The clip lab below is what runs here.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
