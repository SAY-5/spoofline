import type { DemoData } from "../lib/data.ts";

export function DataNote({ data }: { data: DemoData }) {
  const { manifest } = data;
  const c = manifest.corpus;
  return (
    <div className="note-grid">
      <div>
        <h3>Where the clips come from</h3>
        <p>
          The corpus is a deterministic generator in this repository, not a public benchmark. FaceForensics++ and ASVspoof need
          signed licences, so spoofline renders {c.n_clips} clips of {c.n_identities} synthetic identities from seed {manifest.seed}:
          {" "}{c.n_frames} frames of {c.frame_size}x{c.frame_size} video drawn with OpenCV and {c.n_samples / c.sample_rate} s of{" "}
          {c.sample_rate / 1000} kHz speech-like audio built with numpy and torchaudio. The eight attack families are real signal
          transformations applied to those renders.
        </p>
        <p>
          Every number on this page describes that synthetic corpus. It is one seed and one held out pair of families, with no
          variance estimate, so gaps of a point or two between detectors should not be read as real.
        </p>
        <p>
          The headline figures, the catch strips and every table are the measured results of that offline run, replayed here from
          the exported logits. The clip lab is the part that scores clips in this tab.
        </p>
      </div>
      <div>
        <h3>What runs in this tab</h3>
        <p>
          The {manifest.clips.length} clips here come from the test identities, which neither network trained on and which were not
          used for calibration. For each clip the page unpacks the exact uint8 frames and int16 samples, computes the video steps and the
          log mel patches in TypeScript, and runs both trained networks as ONNX graphs under onnxruntime-web on WebAssembly. Platt
          scaling, the two stream thresholds, the weighted sum, the logistic fusion and the attribution of a decision to a stream are
          applied in TypeScript with the exported parameters. Nothing is sent to a server.
        </p>
        <p>
          A node self-check scores every exported clip the same way and holds the raw logits to within 1e-4 of PyTorch, with identical
          per stream flags, identical weighted and logistic decisions, and the same triggering stream.
        </p>
      </div>
    </div>
  );
}
