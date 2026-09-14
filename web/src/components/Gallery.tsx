import { useEffect, useState } from "react";
import type { ClipData } from "../lib/clip.ts";
import { loadClip, type DemoData } from "../lib/data.ts";
import { logMel, type LogMel } from "../lib/features.ts";
import { FAMILY_LABEL, FAMILY_ORDER, FAMILY_PLAIN, streamOf } from "../format.ts";
import { FrameCanvas } from "./FrameCanvas.tsx";
import { Spectrogram } from "./Spectrogram.tsx";

const STRIP_FRAMES = [0, 5, 10, 15];

export function Gallery({ data }: { data: DemoData }) {
  const { manifest, fbank } = data;
  const [clips, setClips] = useState<Record<string, { clip: ClipData; mel: LogMel }>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const ids = [...new Set(Object.values(manifest.gallery.clips))];
    Promise.all(ids.map(async (id) => [id, await loadClip(manifest, id)] as const))
      .then((loaded) => {
        if (!live) return;
        const next: Record<string, { clip: ClipData; mel: LogMel }> = {};
        for (const [id, clip] of loaded) next[id] = { clip, mel: logMel(clip.audio, fbank) };
        setClips(next);
      })
      .catch((err: unknown) => live && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      live = false;
    };
  }, [manifest, fbank]);

  const unseen = new Set(manifest.unseen_families);
  const n = manifest.corpus.n_frames;

  return (
    <>
      {error && <p role="alert">Could not load the gallery clips: {error}</p>}
      <ul className="gallery" aria-label={`Identity ${manifest.gallery.identity}, bona fide and all eight attack families`}>
        {FAMILY_ORDER.map((family) => {
          const id = manifest.gallery.clips[family];
          if (!id) return null;
          const entry = manifest.clips.find((c) => c.id === id)!;
          const loaded = clips[id];
          const stream = streamOf(family);
          const other = stream === "video" ? entry.audio_family : stream === "audio" ? entry.video_family : "bonafide";
          return (
            <li key={family} className={`card ${family === "bonafide" ? "card-bona" : ""}`}>
              <div className="card-media">
                <div className="card-strip">
                  {STRIP_FRAMES.filter((f) => f < n).map((f) =>
                    loaded ? (
                      <FrameCanvas
                        key={f}
                        frames={loaded.clip.frames}
                        size={manifest.corpus.frame_size}
                        index={f}
                        label={`${FAMILY_LABEL[family]} clip ${id}, frame ${f + 1}`}
                        className="frame frame-thumb"
                      />
                    ) : (
                      <div key={f} className="frame frame-thumb frame-empty" aria-hidden="true" />
                    ),
                  )}
                </div>
                {loaded ? (
                  <Spectrogram
                    mel={loaded.mel}
                    nMels={manifest.features.n_mels}
                    label={`Log mel spectrogram of ${FAMILY_LABEL[family]} clip ${id}`}
                    className="spectro-thumb"
                  />
                ) : (
                  <div className="spectro spectro-thumb spectro-empty" aria-hidden="true" />
                )}
              </div>
              <div className="card-body">
                <p className="card-tags">
                  <span className={`tag tag-stream ${stream ? "" : "tag-quiet"}`}>{stream ?? "no attack"}</span>
                  {unseen.has(family) && <span className="tag tag-accent">held out of training</span>}
                </p>
                <h3>{FAMILY_LABEL[family]}</h3>
                <p className="card-text">{FAMILY_PLAIN[family]}</p>
                <p className="card-meta">
                  {id}
                  {other !== "bonafide" ? `, also carries ${FAMILY_LABEL[other]}` : ""}
                </p>
              </div>
            </li>
          );
        })}
      </ul>
    </>
  );
}
