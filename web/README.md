# spoofline browser demo

A static page that runs the trained spoofline detectors in the browser. There is no
backend: everything below happens in the visitor's tab.

## Where the weights come from

The checkpoints were trained with `make demo` at commit
`7181cf425af215ebadfee6506e50f70899b66ac5`. That run printed a summary block
identical to the one in the top level README, timings aside, and
`web/scripts/export.py` then wrote everything under `public/data`.
`manifest.json` and `reference.json` record the same commit.

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
  `models/audio.onnx` are the trained CNN-LSTMs with their normalisers folded in,
  exported at a fixed length and batch of one, where packing and the attention mask
  have no effect. They run on the WebAssembly backend, bundled with the page rather
  than fetched from a CDN.
* **Calibration and fusion, in TypeScript.** `src/lib/calibration.ts` applies the
  exported Platt parameters, per stream thresholds, fusion weight and fused
  threshold, and reimplements threshold selection and the fusion grid search so the
  calibration explorer can move the target precision live.

## Checking it

```bash
uv run --with onnx --with onnxruntime python web/scripts/export.py --trained-from <sha>
cd web
npm install
npm run selfcheck   # onnxruntime-node parity against PyTorch for every exported clip
npm run bundle      # the same as npm run build, which Vercel uses
```

The export refuses to write if the calibration scores do not refit to the exact
Platt parameters, thresholds and fusion weight of the run, or if the test scores do
not reproduce its metrics table. `npm run selfcheck` then holds every raw logit to
within 1e-4 of PyTorch and requires identical per stream and fused decisions.

## Deploying

`vercel.json` builds with `npm run build` and serves `dist/`. Set the project root
to `web/`.
