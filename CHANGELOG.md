# Changelog

All notable changes to spoofline. Versions follow semantic versioning and each
one is an annotated git tag with a matching GitHub release.

## 3.0.0

Fusion that earns its place, measured.

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

Evaluation you can trust.

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
