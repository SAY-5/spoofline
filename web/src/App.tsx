import { useEffect, useState } from "react";
import { CalibrationExplorer } from "./components/CalibrationExplorer.tsx";
import { ClipLab } from "./components/ClipLab.tsx";
import { DataNote } from "./components/DataNote.tsx";
import { Gallery } from "./components/Gallery.tsx";
import { Hero } from "./components/Hero.tsx";
import { Pending, Section } from "./components/Section.tsx";
import { SiteFooter } from "./components/SiteFooter.tsx";
import { UnseenSection } from "./components/UnseenSection.tsx";
import { DATA_BASE, loadClip, loadDemoData, type DemoData } from "./lib/data.ts";
import { BrowserDetector } from "./lib/inference.ts";
import { scoreClip, type ClipScore } from "./lib/score.ts";

declare global {
  interface Window {
    spooflineScore?: (id: string) => Promise<ClipScore>;
  }
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function App() {
  const [data, setData] = useState<DemoData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [detector, setDetector] = useState<BrowserDetector | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    loadDemoData()
      .then((loaded) => {
        if (!live) return;
        setData(loaded);
        return BrowserDetector.load(loaded.manifest, DATA_BASE, loaded.fbank)
          .then((ready) => live && setDetector(ready))
          .catch((err: unknown) => live && setModelError(message(err)));
      })
      .catch((err: unknown) => live && setLoadError(message(err)));
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!data || !detector) return;
    window.spooflineScore = async (id: string) => {
      const clip = await loadClip(data.manifest, id);
      const video = await detector.run("video", clip);
      const audio = await detector.run("audio", clip);
      return scoreClip(data.manifest.calibration, video.logit, audio.logit);
    };
    return () => {
      delete window.spooflineScore;
    };
  }, [data, detector]);

  const status = modelError ? "models failed to load" : detector ? "2 ONNX graphs ready, wasm" : "loading models";

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="masthead">
        <div className="masthead-inner">
          <a className="wordmark" href="#top" aria-label="spoofline, back to top">
            spoofline
          </a>
          <nav aria-label="Sections">
            <ul className="nav">
              <li>
                <a href="#lab">Lab</a>
              </li>
              <li>
                <a href="#families">Families</a>
              </li>
              <li>
                <a href="#calibration">Calibration</a>
              </li>
              <li>
                <a href="#unseen">Unseen</a>
              </li>
              <li>
                <a href="#data">Data</a>
              </li>
            </ul>
          </nav>
          <p className={`status ${detector ? "is-ready" : ""} ${modelError ? "is-error" : ""}`} role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" />
            {status}
          </p>
        </div>
      </header>
      <main id="main" tabIndex={-1}>
        <div id="top" />
        <Hero data={data} />
        {loadError && (
          <p className="alert" role="alert">
            The exported run could not be loaded: {loadError}
          </p>
        )}
        <Section
          id="lab"
          index="01"
          title="Clip lab"
          lede="Pick a clip. Both trained networks score it here in the browser, and every stage of the decision is shown next to the PyTorch reference."
        >
          {data ? <ClipLab data={data} detector={detector} modelError={modelError} /> : <Pending label="Loading the exported run." />}
        </Section>
        <Section
          id="families"
          index="02"
          title="Attack families"
          lede="One test identity, bona fide and under each of the eight transformations. Frames and spectrograms are drawn from the clip data itself."
        >
          {data ? <Gallery data={data} /> : <Pending label="Loading clips." />}
        </Section>
        <Section
          id="calibration"
          index="03"
          title="Calibration explorer"
          lede="Each stream's logits are mapped to a probability by a Platt fit, then thresholded at a target precision. Move the target and watch the threshold and recall follow."
        >
          {data ? <CalibrationExplorer data={data} /> : <Pending label="Loading calibration scores." />}
        </Section>
        <Section
          id="unseen"
          index="04"
          title="Unseen families"
          lede="The video_splice and audio_vocoder families were held out of training and calibration. This is where the claim gets tested, and where it partly fails."
        >
          {data ? <UnseenSection data={data} /> : <Pending label="Loading metrics." />}
        </Section>
        <Section id="data" index="05" title="Data note" lede="What the numbers are measured on, and what exactly this page executes.">
          {data ? <DataNote data={data} /> : <Pending label="Loading." />}
        </Section>
      </main>
      <SiteFooter commit={data?.manifest.trained_from_commit ?? null} />
    </>
  );
}
