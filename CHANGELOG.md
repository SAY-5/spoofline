# Changelog

All notable changes to spoofline. Versions follow semantic versioning and each
one is an annotated git tag with a matching GitHub release.

## Unreleased

The browser demo as a first class deliverable, and every README figure gated on a
committed artifact.

* `web/scripts/export.py` scores its reference through
  `spoofline.scoring.RunScorer` and exports both graphs through
  `spoofline.export.export_stream`. It had kept a second copy of the ONNX wrapper
  and imported `score_single_clip`, which the v4 series removed, so the documented
  way to rebuild the page had been broken at HEAD.
* The page shows what the repository computes: the logistic fusion beside the
  weighted sum in the calibration table and in the clip lab, and the stream each
  fusion decision is attributed to. The hero names the corpus, the seed, the held
  out pair and the training commit, and says that the catch strips replay exported
  logits rather than scoring in the tab.
* CI runs the page: `npm ci`, `npm run typecheck`, `npm run selfcheck` and the
  production bundle, and `ruff` now lints `web/scripts` as well. The self-check grew
  from 258 to 370 assertions over 25 clips, because both fusion probabilities, both
  decisions and the attribution now have to match PyTorch.
* `docs/runs/` holds the JSON of the runs the README quotes, and
  `tests/test_readme_numbers.py` re-renders every pasted block from it, so a stale
  table fails the suite instead of drifting. The demo run was re-measured on this
  branch: every metric reproduced, and the latency table was re-measured on a
  quieter machine.
* `spoofline sweep --pairs default` restricts the sweep to the profile's held out
  pair, which is how the full profile variance table was produced.
* `spoofline robustness` takes its seed, held out families and split membership
  from the run's `results.json` and `splits.json` instead of recomputing them from
  the profile, so a run trained with another seed can be post-processed.
* `Splits.counts()` reports the clips dropped for carrying a held out family, so
  the four split sizes and the dropped count add up to the corpus.
* The report orders its rows itself instead of inheriting the key order of the
  dict it is handed, so a run rendered from `results.json` prints what the run
  printed.
* On Linux torch and torchaudio resolve from the PyTorch CPU index, so a CPU only
  test run no longer installs the CUDA toolkit, and CI installs with
  `uv sync --locked`.

## 5.0.0

Deployment path.

* `spoofline export --onnx` writes both streams as ONNX with a dynamic batch and
  step axis, with the normaliser folded into the graph, and refuses to finish if
  any clip's ONNX logit differs from the PyTorch logit by more than 1e-4. Measured
  on the demo run: video 3.81e-06, audio 3.81e-06 over 32 clips.
* `spoofline model-card` renders a model card from the last evaluation run: model
  details, data note, splits, thresholds with their score formulas, seen and unseen
  metrics for every detector, robustness at the heaviest severities, limitations.
  `docs/MODEL_CARD.md` is the card of the demo run.
* `spoofline score` takes any number of clips and `--json` emits one document with
  `schema_version`, `run_dir` and a `clips` list whose entries carry every logit,
  probability, flag, both decisions and `triggered_by`.
* `spoofline bench` reports per clip p50 and p95 latency for each stream and end to
  end on PyTorch and ONNX Runtime. Measured on the demo run: PyTorch p50 10.07 ms video, 2.54 ms audio, 17.04 ms end to end; ONNX
  Runtime 4.50 ms, 2.05 ms and 8.44 ms, one thread on a busy 10 core CPU.
* 5 new tests: ONNX parity within 1e-4 for both streams, ONNX on other batch and
  step sizes, model card sections and numbers, batch JSON schema, latency rows.

## 4.0.0

Robustness to benign degradation.

* `spoofline robustness` degrades every bona fide test clip of a finished run with
  eight perturbations at three to five severities each: JPEG quality ladder, video
  gaussian noise, brightness and contrast drift, frame dropout, audio gaussian
  noise, resampling round trip, mild reverb and short audio dropouts. It reports
  the false alarm rate per perturbation and severity for the video and audio
  streams and both fusions.
* Abstain option: a fusion abstains when `|p_video - p_audio|` exceeds a margin,
  and the report gives coverage, precision and recall on the kept clips, plus the
  attacks and bona fide clips abstained on, for a ladder of margins on the seen
  and unseen test splits.
* `spoofline.scoring.RunScorer` loads a finished run's checkpoints, calibration and
  fusions once and scores decoded clips in batches.
* Measured on the demo run's 86 bona fide test clips: clean false alarm rate 0.116
  for the weighted sum; 40 dB SNR audio noise raises it to 0.849, a 12 kHz resample
  to 1.000, RT60 0.1 s reverb to 0.709, JPEG quality 30 to 0.174 and brightness and
  contrast drift 0.35 to 0.244. Video noise and frame dropout have no effect.
* Abstaining at margin 0.9 reaches unseen precision 1.000 at coverage 0.821 but
  abstains on 19 of 80 attacks, and lowers seen precision to 0.881 at coverage
  0.487.
* 35 new tests: every one of the 30 perturbation severities changes its own stream,
  leaves the other stream and the label untouched, heavier severities change the
  signal more, perturbations are deterministic and refuse attacked clips, the
  abstain coverage arithmetic, and a robustness report on a tiny run.

## 3.0.0

Logistic fusion and per clip attribution.

* Logistic fusion over both calibrated probabilities and their absolute
  disagreement, fitted by a penalised Newton solve on the calibration split only,
  with its operating point chosen by the precision constrained threshold search
  at the target precision.
* Per clip attribution of every fusion decision to `video`, `audio`, `either`,
  `joint` or `none` by silencing one stream at a time, printed by `spoofline score`
  as `triggered_by` and tabulated against the attacked modality in the summary.
* The pipeline summary and the sweep compare the weighted sum, the logistic fusion
  and the AND and OR rules on seen and unseen families, with the gap of each fusion
  to the best single stream.
* Measured over the 48 sweep runs: logistic unseen precision 0.946 against 0.944
  for the weighted sum; the gap to the best single stream is -0.036 against
  -0.038, a paired difference of +0.002 (95 percent interval -0.011 to +0.017).
  It does not measurably narrow the gap, so the weighted sum stays primary.
* 7 new tests: fitting determinism and row order independence, the precision
  search meeting its target on the calibration split with maximal recall, the
  disagreement weight sign, dict round trip, attribution on constructed weighted
  and logistic cases, and attribution covering every test clip in a pipeline run.

## 2.0.0

Seed and held out pair sweep with bootstrap intervals.

* `spoofline sweep` runs every leave two families out split, one held out video
  family paired with one held out audio family (16 splits), for several seeds in
  parallel worker processes, and caches each run's raw logits so calibration,
  fusion and metrics are recomputed without retraining.
* Mean, sample standard deviation and a percentile bootstrap 95 percent interval
  of the mean for precision, recall, F1, EER and AUC per detector on the seen and
  unseen test splits, plus two per run derived gaps and a per pair table.
* New `reduced` profile (960 clips, 8 frames of 64x64, 1.0 s of audio, 8 video and
  6 audio epochs) sized for the sweep. The README variance table comes from it.
* Measured over 3 seeds and 16 splits: fused unseen precision 0.944 (std 0.048),
  on average 0.038 below the better single stream of the same run.
* 9 new tests: pair enumeration, no held out family in train or calibration for
  any pair, bootstrap coverage and width on a known normal distribution,
  aggregation arithmetic, and a tiny end to end sweep that reuses its cache.

## 1.0.0

Baseline release of the two-stream detector as it stood on main.

* CNN-LSTM detector per stream (per-step CNN encoder, packed bidirectional LSTM,
  masked attention pooling), about 244k parameters each.
* Deterministic corpus generator with eight attack families, four per modality,
  each a real signal transformation.
* Leave-one-attack-family-out protocol with identity-disjoint train, calibration
  and test pools.
* Per-stream Platt calibration against the modality label, threshold selection
  at a target precision on the calibration split, weighted score fusion, AND and
  OR reference rules.
* Precision, recall, F1, EER, AUC and a per-family breakdown on the seen and
  unseen test splits, printed by `spoofline pipeline` and pasted in the README.
* `DirectoryClipSource` adapter for a real corpus of media files.
* 95 tests on a committed 24 clip fixture corpus.
