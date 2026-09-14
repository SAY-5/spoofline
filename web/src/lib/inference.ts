import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { logMel, melPatches, videoSteps, type LogMel } from "./features.ts";
import { clipShape, type ClipData } from "./clip.ts";
import type { Manifest, Stream } from "./types.ts";

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

export interface StreamRun {
  logit: number;
  featureMs: number;
  inferenceMs: number;
}

export class BrowserDetector {
  private constructor(
    private readonly manifest: Manifest,
    private readonly sessions: Record<Stream, ort.InferenceSession>,
    private readonly fbank: Float32Array,
  ) {}

  static async load(manifest: Manifest, base: string, fbank: Float32Array): Promise<BrowserDetector> {
    const create = async (stream: Stream) => {
      const response = await fetch(base + manifest.models[stream].path);
      if (!response.ok) throw new Error(`could not fetch the ${stream} model (${response.status})`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    };
    const [video, audio] = await Promise.all([create("video"), create("audio")]);
    return new BrowserDetector(manifest, { video, audio }, fbank);
  }

  logMel(clip: ClipData): LogMel {
    return logMel(clip.audio, this.fbank);
  }

  async run(stream: Stream, clip: ClipData): Promise<StreamRun> {
    const spec = this.manifest.models[stream];
    const t0 = performance.now();
    const data =
      stream === "video"
        ? videoSteps(clip.frames, clipShape(this.manifest.corpus))
        : melPatches(logMel(clip.audio, this.fbank)).data;
    const t1 = performance.now();
    const feeds = { [spec.input]: new ort.Tensor("float32", data, spec.shape) };
    const outputs = await this.sessions[stream].run(feeds);
    const t2 = performance.now();
    const logit = Number((outputs[spec.output]!.data as Float32Array)[0]);
    return { logit, featureMs: t1 - t0, inferenceMs: t2 - t1 };
  }
}
