import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ClipData } from "../lib/clip.ts";
import { loadClip, type DemoData } from "../lib/data.ts";
import type { LogMel } from "../lib/features.ts";
import type { BrowserDetector, StreamRun } from "../lib/inference.ts";
import { playSamples, type Playback } from "../lib/playback.ts";
import { scoreClip, type ClipScore, type StageResult } from "../lib/score.ts";
import type { ClipEntry, Combo, ReferenceClip, Stream } from "../lib/types.ts";
import { FAMILY_LABEL, fixed, gap, ms, signed } from "../format.ts";
import { useReducedMotion } from "../hooks.ts";
import { FrameCanvas } from "./FrameCanvas.tsx";
import { Spectrogram } from "./Spectrogram.tsx";

const GROUPS: { combo: Combo; title: string }[] = [
  { combo: "bonafide", title: "Bona fide" },
  { combo: "video_only", title: "Video attacked" },
  { combo: "audio_only", title: "Audio attacked" },
  { combo: "both", title: "Both attacked" },
];

interface LabResult {
  id: string;
  runs: Record<Stream, StreamRun>;
  score: ClipScore;
}

function clipTitle(entry: ClipEntry): string {
  if (entry.combo === "bonafide") return "Bona fide";
  const parts = [entry.video_family, entry.audio_family].filter((f) => f !== "bonafide");
  return parts.map((f) => FAMILY_LABEL[f] ?? f).join(" + ");
}

function outcome(label: 0 | 1, decision: "attack" | "bonafide"): string {
  if (label === 1) return decision === "attack" ? "caught" : "missed";
  return decision === "attack" ? "false alarm" : "correctly passed";
}

interface StageCardProps {
  name: string;
  stream: Stream;
  stage: StageResult | null;
  run: StreamRun | null;
  reference: ReferenceClip | undefined;
  attacked: boolean;
  family: string;
}

function StageCard({ name, stream, stage, run, reference, attacked, family }: StageCardProps) {
  const refLogit = reference ? (stream === "video" ? reference.video_logit : reference.audio_logit) : null;
  return (
    <article className="stage" aria-label={`${name} result`} data-stream={stream} data-logit={stage?.logit ?? ""}>
      <header className="stage-head">
        <h4>{name}</h4>
        <span className="stage-truth">{attacked ? `truth: ${FAMILY_LABEL[family]}` : "truth: untouched"}</span>
      </header>
      <dl className="readout">
        <div>
          <dt>raw logit</dt>
          <dd>{stage ? signed(stage.logit, 4) : "..."}</dd>
        </div>
        <div>
          <dt>PyTorch gap</dt>
          <dd>{stage && refLogit !== null ? gap(Math.abs(stage.logit - refLogit)) : "..."}</dd>
        </div>
        <div>
          <dt>calibrated p</dt>
          <dd>{stage ? fixed(stage.probability, 4) : "..."}</dd>
        </div>
        <div>
          <dt>threshold</dt>
          <dd>{stage ? fixed(stage.threshold, 4) : "..."}</dd>
        </div>
      </dl>
      <ProbabilityBar value={stage?.probability ?? 0} threshold={stage?.threshold ?? 0} active={stage !== null} />
      <footer className="stage-foot">
        <span className={`verdict ${stage?.flags ? "is-attack" : ""}`}>
          {stage ? (stage.flags ? "Flags attack" : "Passes") : "Waiting"}
        </span>
        <span className="timing">{run ? `${ms(run.featureMs)} ms features, ${ms(run.inferenceMs)} ms model` : ""}</span>
      </footer>
    </article>
  );
}

function ProbabilityBar({ value, threshold, active }: { value: number; threshold: number; active: boolean }) {
  return (
    <div className="pbar" aria-hidden="true">
      <span className="pbar-fill" style={{ transform: `scaleX(${active ? value : 0})` }} />
      <span className="pbar-tick" style={{ left: `${threshold * 100}%` }} />
      <span className="pbar-scale">
        <span>0</span>
        <span>0.5</span>
        <span>1</span>
      </span>
    </div>
  );
}

export function ClipLab({ data, detector, modelError }: { data: DemoData; detector: BrowserDetector | null; modelError: string | null }) {
  const { manifest, reference } = data;
  const corpus = manifest.corpus;
  const reducedMotion = useReducedMotion();
  const initial = manifest.gallery.clips["audio_vocoder"] ?? manifest.clips[0]!.id;
  const [selected, setSelected] = useState(initial);
  const [clip, setClip] = useState<{ id: string; data: ClipData; mel: LogMel } | null>(null);
  const [result, setResult] = useState<LabResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState(Math.floor(corpus.n_frames / 2));
  const [playhead, setPlayhead] = useState<number | null>(null);
  const playback = useRef<Playback | null>(null);
  const scrubbed = useRef(false);
  const request = useRef(0);

  const entry = manifest.clips.find((c) => c.id === selected)!;
  const ref = reference.clips[selected];

  const stopAudio = useCallback(() => {
    playback.current?.stop();
    playback.current = null;
    setPlayhead(null);
  }, []);

  const run = useCallback(
    async (id: string) => {
      if (!detector) return;
      const token = ++request.current;
      setBusy(true);
      setError(null);
      try {
        const clipData = await loadClip(manifest, id);
        const mel = detector.logMel(clipData);
        if (token !== request.current) return;
        setClip({ id, data: clipData, mel });
        const video = await detector.run("video", clipData);
        const audio = await detector.run("audio", clipData);
        if (token !== request.current) return;
        setResult({ id, runs: { video, audio }, score: scoreClip(manifest.calibration, video.logit, audio.logit) });
      } catch (err) {
        if (token === request.current) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (token === request.current) setBusy(false);
      }
    },
    [detector, manifest],
  );

  useEffect(() => {
    stopAudio();
    scrubbed.current = false;
    setResult(null);
    void run(selected);
  }, [selected, run, stopAudio]);

  useEffect(() => () => stopAudio(), [stopAudio]);

  useEffect(() => {
    let raf = 0;
    const started = performance.now();
    const tick = () => {
      const active = playback.current;
      if (active) {
        const elapsed = active.context.currentTime - active.startedAt;
        const t = Math.max(0, Math.min(0.9999, elapsed / active.duration));
        setFrame(Math.floor(t * corpus.n_frames));
        setPlayhead(t);
      } else if (!reducedMotion && !scrubbed.current) {
        const elapsed = (performance.now() - started) / 1000;
        const duration = corpus.n_samples / corpus.sample_rate;
        setFrame(Math.floor(((elapsed % duration) / duration) * corpus.n_frames));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [corpus, reducedMotion]);

  const togglePlay = () => {
    if (playback.current) {
      stopAudio();
      return;
    }
    if (!clip) return;
    scrubbed.current = false;
    playback.current = playSamples(clip.data.audio, corpus.sample_rate, () => {
      playback.current = null;
      setPlayhead(null);
    });
  };

  const score = result?.id === selected ? result.score : null;
  const runs = result?.id === selected ? result.runs : null;
  const shown = clip?.id === selected ? clip : null;
  const summary = score
    ? `${selected}: video ${fixed(score.video.probability, 3)} ${score.video.flags ? "flags" : "passes"}, audio ${fixed(
        score.audio.probability,
        3,
      )} ${score.audio.flags ? "flags" : "passes"}, fused ${fixed(score.fused.probability, 4)} so ${
        score.fused.decision === "attack" ? "attack" : "bona fide"
      }, ${outcome(entry.label, score.fused.decision)}.`
    : busy
      ? `Scoring ${selected} in the browser.`
      : "";

  const grouped = useMemo(
    () => GROUPS.map((g) => ({ ...g, clips: manifest.clips.filter((c) => c.combo === g.combo) })),
    [manifest],
  );

  return (
    <div className="lab">
      <fieldset className="picker">
        <legend className="picker-legend">Pick a clip from the held out test identities</legend>
        {grouped.map((group) => (
          <div className="picker-group" key={group.combo}>
            <p className="picker-title">{group.title}</p>
            <div className="picker-options">
              {group.clips.map((c) => (
                <label key={c.id} className={`pick ${c.id === selected ? "is-selected" : ""}`}>
                  <input
                    type="radio"
                    name="lab-clip"
                    value={c.id}
                    checked={c.id === selected}
                    onChange={() => setSelected(c.id)}
                  />
                  <span className="pick-title">{clipTitle(c)}</span>
                  <span className="pick-meta">
                    {c.id.replace("clip_", "#")}
                    {c.touches_unseen ? <span className="tag">unseen</span> : null}
                  </span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </fieldset>

      <div className="bench">
        <div className="monitor">
          <div className="monitor-video">
            {shown ? (
              <FrameCanvas
                frames={shown.data.frames}
                size={corpus.frame_size}
                index={Math.min(frame, corpus.n_frames - 1)}
                label={`Frame ${frame + 1} of ${corpus.n_frames} from ${selected}`}
                className="frame frame-lab"
              />
            ) : (
              <div className="frame frame-lab frame-empty" aria-hidden="true" />
            )}
            {busy && <span className="scanline" aria-hidden="true" />}
          </div>
          <div className="monitor-side">
            <p className="monitor-id">
              <span>{selected}</span>
              <span>{entry.identity}</span>
              <span>{entry.split.replace("_", " ")}</span>
            </p>
            <p className="monitor-title">{clipTitle(entry)}</p>
            <div className="monitor-controls">
              <button type="button" className="btn" onClick={togglePlay} disabled={!shown} aria-pressed={playhead !== null}>
                {playhead !== null ? "Stop audio" : "Play audio"}
              </button>
              <button type="button" className="btn btn-ghost" onClick={() => void run(selected)} disabled={!detector || busy}>
                Run again
              </button>
            </div>
            <label className="scrub">
              <span>frame {frame + 1} / {corpus.n_frames}</span>
              <input
                type="range"
                min={0}
                max={corpus.n_frames - 1}
                value={Math.min(frame, corpus.n_frames - 1)}
                onChange={(e) => {
                  stopAudio();
                  scrubbed.current = true;
                  setFrame(Number(e.target.value));
                }}
                disabled={playhead !== null}
              />
            </label>
          </div>
          {shown ? (
            <Spectrogram
              mel={shown.mel}
              nMels={manifest.features.n_mels}
              label={`Log mel spectrogram of ${selected}, 64 mel bands over 2 seconds`}
              playhead={playhead}
              className="spectro-lab"
            />
          ) : (
            <div className="spectro spectro-lab spectro-empty" aria-hidden="true" />
          )}
        </div>

        <div className="stages">
          <StageCard
            name="Video stream"
            stream="video"
            stage={score?.video ?? null}
            run={runs?.video ?? null}
            reference={ref}
            attacked={entry.video_family !== "bonafide"}
            family={entry.video_family}
          />
          <StageCard
            name="Audio stream"
            stream="audio"
            stage={score?.audio ?? null}
            run={runs?.audio ?? null}
            reference={ref}
            attacked={entry.audio_family !== "bonafide"}
            family={entry.audio_family}
          />
          <article className="stage stage-fused" aria-label="Fused result" data-fused={score?.fused.probability ?? ""} data-decision={score?.fused.decision ?? ""}>
            <header className="stage-head">
              <h4>Fusion</h4>
              <span className="stage-truth">truth: {entry.label ? "attack" : "bona fide"}</span>
            </header>
            <p className="formula">
              {score ? (
                <>
                  {fixed(score.fused.weight, 2)} x {fixed(score.video.probability, 4)} + {fixed(1 - score.fused.weight, 2)} x{" "}
                  {fixed(score.audio.probability, 4)} = <strong>{fixed(score.fused.probability, 4)}</strong>
                </>
              ) : (
                "w x p_video + (1 - w) x p_audio"
              )}
            </p>
            <ProbabilityBar value={score?.fused.probability ?? 0} threshold={manifest.calibration.fused.operating.threshold} active={score !== null} />
            <footer className="stage-foot">
              <span className={`verdict verdict-big ${score?.fused.decision === "attack" ? "is-attack" : ""}`}>
                {score ? (score.fused.decision === "attack" ? "Attack" : "Bona fide") : busy ? "Scoring" : "Waiting"}
              </span>
              <span className="timing">
                {score ? `threshold ${fixed(score.fused.threshold, 4)}, ${outcome(entry.label, score.fused.decision)}` : ""}
                {score && ref ? `, PyTorch said ${ref.decision === "attack" ? "attack" : "bona fide"}` : ""}
              </span>
            </footer>
          </article>
          <p className="lab-live" aria-live="polite" role="status">
            {modelError ? `The models could not load: ${modelError}` : error ? `Scoring failed: ${error}` : !detector ? "Loading the two ONNX models into onnxruntime-web." : summary}
          </p>
        </div>
      </div>
    </div>
  );
}
