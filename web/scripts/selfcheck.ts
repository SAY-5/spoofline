/**
 * Node parity check for the browser demo.
 *
 * Scores every exported clip with the same TypeScript feature code the page uses and
 * the exported ONNX graphs under onnxruntime-node, then compares against the PyTorch
 * reference written by web/scripts/export.py.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import {
  countsAt,
  fitFusion,
  fitPlatt,
  fuse,
  plattProbabilities,
  plattProbability,
  precisionRecall,
  thresholdAtPrecision,
} from "../src/lib/calibration.ts";
import { clipShape, decodeClip } from "../src/lib/clip.ts";
import { logMel, melPatches, N_FREQS, N_MELS, videoSteps } from "../src/lib/features.ts";
import { scoreClip } from "../src/lib/score.ts";
import type { Detector, Manifest, Reference, ScoreRows, Stream, TestScores, TestSplit } from "../src/lib/types.ts";

const LOGIT_TOL = 1e-4;
const DATA = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data");

let passed = 0;
const failures: string[] = [];
function check(name: string, ok: boolean, detail = ""): void {
  if (ok) passed++;
  else failures.push(`${name}${detail ? `: ${detail}` : ""}`);
}
function close(name: string, got: number, want: number, tol: number): void {
  check(name, Math.abs(got - want) <= tol, `got ${got}, want ${want}, tol ${tol}`);
}

function readJson<T>(name: string): T {
  return JSON.parse(readFileSync(join(DATA, name), "utf8")) as T;
}
function readBuffer(name: string): ArrayBuffer {
  const bytes = readFileSync(join(DATA, name));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function main(): Promise<void> {
  const manifest = readJson<Manifest>("manifest.json");
  const reference = readJson<Reference>("reference.json");
  const calib = readJson<ScoreRows>("calib_scores.json");
  const test = readJson<TestScores>("test_scores.json");
  const fbank = new Float32Array(readBuffer("mel_fbank.bin"));
  const cal = manifest.calibration;
  const streams: Stream[] = ["video", "audio"];

  // Exported artefacts agree with each other.
  check("manifest and reference name the same training commit", manifest.trained_from_commit === reference.trained_from_commit);
  check("mel filterbank has (n_freqs, n_mels) values", fbank.length === N_FREQS * N_MELS, `${fbank.length}`);
  check("every exported clip has a PyTorch reference", manifest.clips.every((c) => c.id in reference.clips));
  check("between 24 and 40 clips are exported", manifest.clips.length >= 24 && manifest.clips.length <= 40, `${manifest.clips.length}`);
  for (const family of Object.keys(manifest.families)) {
    check(`clip set covers ${family}`, manifest.clips.some((c) => c.video_family === family || c.audio_family === family));
  }
  for (const combo of ["bonafide", "video_only", "audio_only", "both"] as const) {
    check(`clip set covers ${combo}`, manifest.clips.some((c) => c.combo === combo));
  }
  for (const stream of streams) {
    const run = reference.run.calibration[stream];
    check(`${stream} Platt parameters equal the run`, run.calibrator.a === cal[stream].calibrator.a && run.calibrator.b === cal[stream].calibrator.b);
    check(`${stream} threshold equals the run`, run.operating.threshold === cal[stream].operating.threshold);
  }
  check("fusion weight equals the run", reference.run.calibration.fused.weight === cal.fused.weight);
  check("fusion threshold equals the run", reference.run.calibration.fused.operating.threshold === cal.fused.operating.threshold);

  // Log mel front end against torchaudio.
  const checkClip = manifest.features.logmel_check_clip;
  const torchMel = new Float32Array(readBuffer("logmel_check.bin"));
  const clipForMel = decodeClip(readBuffer(`clips/${checkClip}.bin`), manifest.corpus);
  const mel = logMel(clipForMel.audio, fbank);
  const [nMels, nFrames] = manifest.features.logmel_check_shape;
  check("log mel shape matches torchaudio", mel.frames === nFrames && nMels === N_MELS, `${mel.frames} frames`);
  let melGap = 0;
  for (let i = 0; i < torchMel.length; i++) melGap = Math.max(melGap, Math.abs(torchMel[i]! - mel.data[i]!));
  check("log mel matches torchaudio within 1e-3", melGap <= 1e-3, `worst ${melGap}`);

  // Calibration is monotonic and reproducible from the exported calibration scores.
  const calibLabels = Float64Array.from(calib.label);
  for (const stream of streams) {
    const platt = cal[stream].calibrator;
    check(`${stream} Platt slope is non negative`, platt.a >= 0);
    let monotonic = true;
    let previous = -Infinity;
    for (let s = -40; s <= 40; s += 0.05) {
      const p = plattProbability(platt, s);
      if (p < previous) monotonic = false;
      previous = p;
    }
    check(`${stream} calibration is monotonic over logits -40..40`, monotonic);
    const logits = stream === "video" ? calib.video_logit : calib.audio_logit;
    const modality = stream === "video" ? calib.video_attacked : calib.audio_attacked;
    const refit = fitPlatt(logits, modality);
    close(`${stream} Platt a refits from calibration scores`, refit.a, platt.a, 1e-9);
    close(`${stream} Platt b refits from calibration scores`, refit.b, platt.b, 1e-9);
    const point = thresholdAtPrecision(plattProbabilities(platt, logits), calibLabels, manifest.target_precision);
    close(`${stream} threshold reproduces at target precision`, point.threshold, cal[stream].operating.threshold, 1e-12);
    close(`${stream} calibration precision reproduces`, point.achievedPrecision, cal[stream].operating.achieved_precision, 1e-12);
    close(`${stream} calibration recall reproduces`, point.recall, cal[stream].operating.recall, 1e-12);
  }
  const calibPv = plattProbabilities(cal.video.calibrator, calib.video_logit);
  const calibPa = plattProbabilities(cal.audio.calibrator, calib.audio_logit);
  const fusion = fitFusion(calibPv, calibPa, calibLabels, manifest.target_precision);
  close("fusion weight reproduces from the grid search", fusion.weight, cal.fused.weight, 1e-12);
  close("fusion threshold reproduces from the grid search", fusion.operating.threshold, cal.fused.operating.threshold, 1e-12);

  // Test split metrics table.
  const testPv = plattProbabilities(cal.video.calibrator, test.video_logit);
  const testPa = plattProbabilities(cal.audio.calibrator, test.audio_logit);
  const testPf = testPv.map((pv, i) => fuse(cal.fused.weight, pv, testPa[i]!));
  const scoresFor: Record<Detector, Float64Array> = { video: testPv, audio: testPa, fused: testPf };
  const thresholdFor: Record<Detector, number> = {
    video: cal.video.operating.threshold,
    audio: cal.audio.operating.threshold,
    fused: cal.fused.operating.threshold,
  };
  for (const split of ["seen_test", "unseen_test"] as TestSplit[]) {
    const idx = test.split.flatMap((s, i) => (s === split ? [i] : []));
    const labels = idx.map((i) => test.label[i]!);
    for (const detector of ["video", "audio", "fused"] as Detector[]) {
      const scores = idx.map((i) => scoresFor[detector][i]!);
      const counts = countsAt(scores, labels, thresholdFor[detector]);
      const pr = precisionRecall(counts);
      const want = reference.run.metrics[split][detector];
      check(`${split} ${detector} confusion counts match the run`, counts.tp === want.counts.tp && counts.fp === want.counts.fp, JSON.stringify(counts));
      close(`${split} ${detector} precision matches the run`, pr.precision, want.precision, 1e-12);
      close(`${split} ${detector} recall matches the run`, pr.recall, want.recall, 1e-12);
    }
  }

  // Every exported clip through onnxruntime-node.
  const sessions = {
    video: await ort.InferenceSession.create(join(DATA, manifest.models.video.path)),
    audio: await ort.InferenceSession.create(join(DATA, manifest.models.audio.path)),
  };
  let worstLogit = 0;
  for (const entry of manifest.clips) {
    const clip = decodeClip(readBuffer(`clips/${entry.id}.bin`), manifest.corpus);
    const want = reference.clips[entry.id]!;
    const logits: Record<Stream, number> = { video: 0, audio: 0 };
    for (const stream of streams) {
      const spec = manifest.models[stream];
      const data = stream === "video" ? videoSteps(clip.frames, clipShape(manifest.corpus)) : melPatches(logMel(clip.audio, fbank)).data;
      const out = await sessions[stream].run({ [spec.input]: new ort.Tensor("float32", data, spec.shape) });
      logits[stream] = Number((out[spec.output]!.data as Float32Array)[0]);
      const gap = Math.abs(logits[stream] - (stream === "video" ? want.video_logit : want.audio_logit));
      worstLogit = Math.max(worstLogit, gap);
      check(`${entry.id} ${stream} raw logit within ${LOGIT_TOL}`, gap <= LOGIT_TOL, `gap ${gap}`);
    }
    const got = scoreClip(cal, logits.video, logits.audio);
    check(`${entry.id} video verdict matches`, got.video.flags === want.video_flags);
    check(`${entry.id} audio verdict matches`, got.audio.flags === want.audio_flags);
    check(`${entry.id} fused decision matches`, got.fused.decision === want.decision);
    close(`${entry.id} fused probability within 1e-5`, got.fused.probability, want.fused_probability, 1e-5);
    const fromReference = scoreClip(cal, want.video_logit, want.audio_logit);
    close(`${entry.id} fusion rule reproduces the Python fused probability`, fromReference.fused.probability, want.fused_probability, 1e-12);
    const label = want.decision === "attack" ? 1 : 0;
    check(`${entry.id} Python decision agrees with the fused threshold`, (want.fused_probability >= cal.fused.operating.threshold ? 1 : 0) === label);
  }

  const total = passed + failures.length;
  console.log(`selfcheck: ${passed}/${total} assertions passed over ${manifest.clips.length} clips`);
  console.log(`selfcheck: worst raw logit gap ${worstLogit.toExponential(2)}, worst log mel gap ${melGap.toExponential(2)}`);
  if (failures.length) {
    for (const failure of failures) console.error(`FAIL ${failure}`);
    process.exit(1);
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
