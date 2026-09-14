import { decodeClip, type ClipData } from "./clip.ts";
import type { Manifest, Reference, ScoreRows, TestScores } from "./types.ts";

export const DATA_BASE = `${import.meta.env.BASE_URL}data/`;

async function fetchOk(path: string): Promise<Response> {
  const response = await fetch(DATA_BASE + path);
  if (!response.ok) throw new Error(`could not fetch ${path} (${response.status})`);
  return response;
}

export async function loadJson<T>(path: string): Promise<T> {
  return (await fetchOk(path)).json() as Promise<T>;
}

async function inflate(response: Response): Promise<ArrayBuffer> {
  const body = response.body ?? new Blob([await response.arrayBuffer()]).stream();
  return new Response(body.pipeThrough(new DecompressionStream("deflate"))).arrayBuffer();
}

export interface DemoData {
  manifest: Manifest;
  reference: Reference;
  calib: ScoreRows;
  test: TestScores;
  fbank: Float32Array;
}

export async function loadDemoData(): Promise<DemoData> {
  const [manifest, reference, calib, test, fbankBuffer] = await Promise.all([
    loadJson<Manifest>("manifest.json"),
    loadJson<Reference>("reference.json"),
    loadJson<ScoreRows>("calib_scores.json"),
    loadJson<TestScores>("test_scores.json"),
    fetchOk("mel_fbank.bin").then((r) => r.arrayBuffer()),
  ]);
  return { manifest, reference, calib, test, fbank: new Float32Array(fbankBuffer) };
}

const clipCache = new Map<string, Promise<ClipData>>();

export function loadClip(manifest: Manifest, id: string): Promise<ClipData> {
  let pending = clipCache.get(id);
  if (!pending) {
    pending = fetchOk(`clips/${id}.zlib`)
      .then(inflate)
      .then((buffer) => decodeClip(buffer, manifest.corpus));
    pending.catch(() => clipCache.delete(id));
    clipCache.set(id, pending);
  }
  return pending;
}
