# spoofline browser demo

A static page that runs the trained spoofline detectors in the browser. There is no
backend: everything below happens in the visitor's tab.

## Where the weights come from

The checkpoints were trained by `make demo` at commit
`c4952e77d45f4deeac6c0c92cad8a02cecc28445` (`git describe`: `v5.0.0-2-gc4952e7`),
profile `full`, seed 20250117, with `video_splice` and `audio_vocoder` held out.
`web/scripts/export.py` then wrote everything under `public/data`, and both
`manifest.json` and `reference.json` record that commit and its describe string.
The same run's summary block is the one pasted in the top level README, and its
`results.json` is committed under `docs/runs/demo-full-seed20250117/`.

## What the page shows, and what it computes

The headline figures, the catch strips, the calibration table and the unseen family
section are the measured results of that offline run, replayed from the exported
logits. The clip lab is the part that scores clips in the tab: pick a clip and both
graphs run locally, and the page then applies the calibration, both fusion rules and
the attribution itself.

## What runs in the browser

* **Clip decoding.** Each clip comes from the test identities only and is stored
  losslessly as a deflate stream of prediction residuals. The page inflates it with
  `DecompressionStream` and undoes the prediction, which gives back the exact uint8
  frames of shape (16, 64, 64, 3) and 32000 int16 samples at 16 kHz.
* **Features, in TypeScript.** `src/lib/features.ts` mirrors
  `spoofline/data/features.py`: frames scaled to [-1, 1], and a log mel spectrogram
  (512 point FFT, hop 160, periodic Hann window, reflect padding, 64 HTK mel bands
  from the exported torchaudio filterbank) cut into ten 20 frame patches.
* **Both networks, in onnxruntime-web.** `models/video.onnx` and
  `models/audio.onnx` come from `spoofline.export.export_stream`: the trained
  CNN-LSTMs with their normalisers folded in and dynamic batch and step axes. The
  page feeds one clip at a time. They run on the WebAssembly backend, bundled with
  the page rather than fetched from a CDN.
* **Calibration and both fusions, in TypeScript.** `src/lib/calibration.ts` applies
  the exported Platt parameters, the per stream thresholds, the weighted sum and the
  logistic fusion, attributes each decision to a stream the way
  `spoofline.fusion.attribute` does, and reimplements threshold selection and the
  fusion grid search so the calibration explorer can move the target precision live.

## Checking it

```bash
uv run --with onnx --with onnxruntime python web/scripts/export.py --trained-from <sha>
cd web
npm ci
npm run selfcheck   # onnxruntime-node parity against PyTorch for every exported clip
npm run build       # the production bundle, which Vercel runs
```

The export refuses to write if the calibration scores do not refit to the exact
Platt parameters, thresholds and fusion weight of the run, or if the test scores do
not reproduce its metrics table. Measured on the export of the run above,
`npm run selfcheck` passed 370 assertions over 25 clips: worst raw logit gap
3.53e-5 against PyTorch, worst log mel gap 1.50e-4 against torchaudio, and
identical per stream flags, weighted and logistic decisions and triggering stream
on every clip. The same three commands run in CI on every push.

## Deploying

`vercel.json` builds with `npm run build` and serves `dist/`. Set the project root
to `web/`. The bundle is 20 MB on disk, 14 MB of which is the onnxruntime-web
WebAssembly binary.
