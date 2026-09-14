# Architecture

## Why two streams

A spoofed clip can be spoofed in the video, in the audio, or in both. A detector
that only looks at frames cannot see a vocoded voice on top of a genuine face,
and a detector that only listens cannot see a printed photograph held in front of
a camera. Spoofline trains one CNN-LSTM per modality, calibrates each one
separately, and fuses their calibrated scores, which is where the interesting
question sits: does the fused score keep its precision when it meets an attack
family that was never in the training set?

## Data flow

```
                       spoofline generate
                              |
        identity parameters --+-- bona fide video (frames)   bona fide audio (waveform)
                              |            |                          |
                              |     video attack family        audio attack family
                              |     (or bona fide)             (or bona fide)
                              v            v                          v
                          clip npz:  video uint8 (T,H,W,3)   audio int16 (N,)
                                           |                          |
                                    video_steps()               log_mel() -> mel_patches()
                                           |                          |
                                  (T, 3, 64, 64)              (P, 1, 64, 20)
                                           |                          |
                                  +--------+--------+       +---------+--------+
                                  | video CNN-LSTM  |       | audio CNN-LSTM   |
                                  +--------+--------+       +---------+--------+
                                           |                          |
                                     raw logit s_v               raw logit s_a
                                           |                          |
                                  Platt map on calib          Platt map on calib
                                           |                          |
                                        p_v in [0,1]              p_a in [0,1]
                                           +------------+-------------+
                                                        |
                                       w * p_v + (1 - w) * p_a  >=  t_fused
```

A clip is labelled an attack if either stream was attacked. Each stream is
trained on its own modality label (was *this* stream attacked), because a video
network cannot be expected to answer for a vocoded waveform. Calibration and
everything downstream work with the clip level label, which is what a deployed
system has to decide.

## The two networks

Both streams use the same shape, the CNN-LSTM arrangement from the published
anti-spoofing line of work:

1. **Step encoder.** Four strided 3x3 convolutions (24, 40, 64, 64 channels), each
   with batch norm and ReLU, then global average pooling and a linear projection
   to a 96 dimensional step embedding. The video stream feeds it one RGB frame at
   a time; the audio stream feeds it one 64 mel by 20 frame patch at a time.
2. **Temporal LSTM.** A single bidirectional layer with 96 hidden units per
   direction. Batches are packed with `pack_padded_sequence`, so clips of
   different length share a batch without the padding ever reaching the
   recurrence.
3. **Attention pooling.** A two layer additive attention scores every step, the
   scores are masked to the true length, softmaxed and used to average the LSTM
   outputs.
4. **Head.** Dropout and a linear layer to a single logit.

About 244k parameters per stream. Both run on CPU.

### Video features

16 frames of 64x64 RGB, scaled to [-1, 1], then standardised per channel with
statistics fitted on the training split and stored in the checkpoint.

### Audio features

Log mel spectrogram, 64 mel bands, 512 point FFT, hop 160 at 16 kHz. The
spectrogram is cut into non overlapping 20 frame patches (200 ms each), so a two
second clip becomes a sequence of 10 steps. Same normaliser treatment.

## The leave-one-attack-family-out protocol

Eight attack families exist, four per modality. A run nominates one or more as
**unseen**; the default is `video_splice` and `audio_vocoder`, one per stream, so
both networks meet a new attack at test time.

Identities are shuffled and partitioned into three pools that never mix: train,
calib and test. Clips are then assigned:

| split | identity pool | families allowed | used for |
| --- | --- | --- | --- |
| `train` | train | seen only | fitting network weights |
| `calib` | calib | seen only | Platt maps, thresholds, fusion weight |
| `seen_test` | test | seen only, plus half the test bona fide clips | in-distribution report |
| `unseen_test` | test | at least one unseen family, plus the other half of the test bona fide clips | the headline claim |

Any train or calib clip that carries an unseen family is dropped from the corpus
view entirely rather than being relabelled, so the held out families never leak.
The two test splits share no clips: the test bona fide clips are dealt
alternately between them, so precision is measurable on both without double
counting a negative.

Inside the training split a further fraction of *identities* is held out to
report a validation loss.

## The sweep: every leave two families out split

One held out pair is one draw. `spoofline sweep` enumerates all 16 pairings of a
held out video family with a held out audio family and runs each of them under
several seeds. For every (seed, pair) run:

1. the corpus for that seed is generated once and shared by its 16 runs,
2. `make_splits` is called with the pair as the unseen families, so neither held
   out family can reach the train or calib split,
3. both streams are trained and their raw logits on calib, seen_test and
   unseen_test are cached with a hash of the profile,
4. calibration, fusion and metrics are recomputed from the cached logits.

Runs execute in a spawn process pool, each with the profile's fixed thread count,
so a run's numbers do not depend on how many workers ran beside it. The aggregate
is the mean and sample standard deviation over runs, and a percentile bootstrap
interval of the mean from 4000 resamples of the runs. Two derived quantities are
computed per run before summarising: fused unseen precision minus the better
single stream of that run, and seen minus unseen fused precision.

## Calibration

Raw logits from two independently trained networks are not comparable. For each
stream a Platt map

```
p = sigmoid(a * s + b)
```

is fitted on the calibration split against the clip level label, with Platt's
smoothed targets so that a separable calibration set cannot drive the parameters
to infinity, and with `a` clamped to be non negative so the map stays monotonic.
Newton's method with a small ridge term solves the two parameter problem in a few
iterations.

The operating threshold for a stream is the **lowest** calibrated probability
whose precision on the calibration split reaches the target (0.95 by default).
Lowest means the most recall available at that precision. If no threshold reaches
the target, the threshold with the highest achievable precision is returned and
the run reports `target met false` rather than pretending.

## Fusion

The fused score is a convex combination of the two calibrated probabilities:

```
p_fused = w * p_video + (1 - w) * p_audio
```

`w` is grid searched over 101 points on the calibration split, scoring each
candidate exactly the way the single streams are scored: pick the threshold that
maximises recall subject to precision >= target, then compare recall. Because the
grid contains `w = 0` and `w = 1`, the fused operating point can never be worse
than the better single stream *on the calibration split*. Whether that survives
the move to unseen families is the empirical question, and the README reports the
measured answer rather than an assumed one.

Two reference rules are reported next to the weighted sum, both using the single
stream thresholds:

* **AND** flags a clip only if both streams flag it. High precision, low recall.
* **OR** flags a clip if either stream flags it. High recall, and the natural
  baseline given that a clip counts as an attack if either modality was attacked.

## Metrics

Precision, recall and F1 at the calibrated threshold; EER and AUC over the whole
score range; and a per family detection rate at the operating point, with the
bona fide row doubling as the false alarm rate. All of them are implemented in
`spoofline/metrics.py` and checked against hand computed values in the tests.

## Determinism

Every stochastic component draws from a generator derived from `(run seed, label)`
by a blake2b hash, so components are independent and a single clip can be
regenerated without rendering the ones before it. `seed_everything` seeds python,
numpy and torch, pins the intra-op thread count, and turns on deterministic torch
kernels. Data loaders run in the main process with seeded shuffling generators.
The result is that the same seed on the same machine reproduces the same summary
block, which `tests/test_determinism.py` asserts on the tiny fixture corpus.
