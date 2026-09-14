# Changelog

All notable changes to spoofline. Versions follow semantic versioning and each
one is an annotated git tag with a matching GitHub release.

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
