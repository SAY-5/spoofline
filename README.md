# Spoofline

Two-stream audio and video spoof detection. A CNN-LSTM scores video frames, a
second CNN-LSTM scores the audio, each stream's threshold is calibrated on a held
out attack set, and the two calibrated scores are fused so that a clip attacked in
either modality is caught. The question the repo is built to answer honestly:
**does the fused score hold its precision on attack families it has never seen?**

## The data note, up front

No public spoofing corpus is bundled. FaceForensics++, ASVspoof and the rest need
signed licences, so this repo ships a **deterministic generator** instead. It
renders talking-head-like video with OpenCV and speech-like audio with numpy and
torchaudio, then applies eight real signal transformations as attack families.
Every number in this README came from running that generator and the pipeline on
one CPU machine. Nothing here is a benchmark result on a public dataset, and the
absolute numbers should be read as a measurement of this synthetic corpus, not as
a claim about FaceForensics++ or ASVspoof.

To point the same pipeline at a real corpus, implement the `ClipSource` protocol
in `spoofline/data/sources.py`. `DirectoryClipSource` is a working adapter: give
it a directory of media files and a `manifest.csv` and it decodes, resamples and
reshapes clips into exactly the tensors the two streams consume. See
[plugging in a real corpus](#plugging-in-a-real-corpus).

## What is in the box

* Two CNN-LSTM detectors (per-step CNN encoder, bidirectional LSTM with packing,
  masked attention pooling, single logit head), about 244k parameters each.
* Eight attack families, each a real transformation of the signal:

  | video | what it does |
  | --- | --- |
  | `video_replay` | screen re-capture: resampling moire, scrolling refresh banding, gamma and contrast shift, bezel crop, small rotation |
  | `video_print` | ordered halftone dither, static paper grain, flattened motion parallax, desaturation and paper tint |
  | `video_splice` | face region swapped in from a second identity with alpha blending and per-frame seam jitter |
  | `video_recompress` | two real JPEG round trips through OpenCV plus 8x8 DC quantisation blocking |

  | audio | what it does |
  | --- | --- |
  | `audio_replay` | room impulse response, loudspeaker and microphone band limiting, device noise floor and mains hum |
  | `audio_vocoder` | mel analysis, pseudo-inverse back to linear magnitude, Griffin-Lim phase resynthesis |
  | `audio_conversion` | pitch and formant shift by rational resampling with a phase vocoder stretch back to length |
  | `audio_splice` | segments from a second speaker concatenated with hard joins and level mismatch |

* A leave-one-attack-family-out protocol with identity-disjoint train, calibration
  and test pools.
* Per-stream Platt calibration, threshold selection at a target precision, and a
  fused operating point chosen the same way, plus AND and OR rules for reference.
* Precision, recall, F1, EER, AUC and a per-family breakdown, on the seen-family
  test split and the unseen-family test split separately.

`docs/ARCHITECTURE.md` has the data flow, both stream diagrams, the calibration
and fusion maths and the split table.

## Quick start

```bash
make setup      # uv sync, Python 3.12, CPU PyTorch
make lint       # ruff check and ruff format --check
make test       # pytest on the committed tiny fixture corpus
make demo       # the whole pipeline on the full corpus, prints the summary below
make demo-fast  # the same pipeline on the 24 clip fixture corpus, a few seconds
```

Individual stages:

```bash
uv run spoofline generate --profile full
uv run spoofline train --stream video
uv run spoofline train --stream audio
uv run spoofline calibrate
uv run spoofline eval
uv run spoofline score data/full/clips/clip_00000.npz
```

### Browser demo

`web/` is a static page that runs both trained detectors in the browser: clip features are computed in
TypeScript, the two CNN-LSTMs run as ONNX graphs under onnxruntime-web, and the exported Platt maps,
thresholds and fusion rule decide each clip. `web/scripts/export.py` writes the models and a small set of
test clips from a `make demo` run, and `npm run selfcheck` holds every logit to within 1e-4 of PyTorch.
`web/README.md` names the commit the weights were trained from and how to rebuild the page.

## Releases

| version | feature |
| --- | --- |
| [v1.0.0](https://github.com/SAY-5/spoofline/releases/tag/v1.0.0) | baseline: two CNN-LSTM streams, per-stream Platt calibration at a target precision, weighted fusion, leave-one-family-out evaluation |
| [v2.0.0](https://github.com/SAY-5/spoofline/releases/tag/v2.0.0) | evaluation you can trust: `spoofline sweep` over 3 seeds and all 16 leave-two-families-out splits, mean, std and bootstrap 95% intervals per detector |
| [v3.0.0](https://github.com/SAY-5/spoofline/releases/tag/v3.0.0) | fusion that earns its place: logistic fusion over both streams plus their disagreement, compared with weighted sum, AND and OR, and per clip attribution of the triggering stream |
| [v4.0.0](https://github.com/SAY-5/spoofline/releases/tag/v4.0.0) | robustness to benign degradation: false alarm rate per perturbation and severity for every detector, and an abstain option with coverage against precision |
| [v5.0.0](https://github.com/SAY-5/spoofline/releases/tag/v5.0.0) | deployment path: `spoofline export --onnx` with a 1e-4 parity check, a model card from the last run, batch `spoofline score --json`, and per clip p50 and p95 CPU latency |

`CHANGELOG.md` has the detail for every version.

## Measured demo run

Output of one `make demo` on a 10 core Apple silicon CPU, pasted verbatim:

```
$ make demo
[1/6] generating corpus 'full' into data/full
  generated 160/1600 clips (8.5s)
  generated 320/1600 clips (20.5s)
  generated 480/1600 clips (31.6s)
  generated 640/1600 clips (39.4s)
  generated 800/1600 clips (46.4s)
  generated 960/1600 clips (56.0s)
  generated 1120/1600 clips (64.1s)
  generated 1280/1600 clips (71.0s)
  generated 1440/1600 clips (78.0s)
  generated 1600/1600 clips (84.4s)
  wrote manifest for 1600 clips to data/full/manifest.json
[2/6] loading 1600 clips into memory
  loaded 320/1600 clips
  loaded 640/1600 clips
  loaded 960/1600 clips
  loaded 1280/1600 clips
  loaded 1600/1600 clips
  splits {'train': 751, 'calib': 234, 'seen_test': 197, 'unseen_test': 123}
[3/6] training the video stream
  video: 639 train clips, 112 val clips, 278 attacked, 10 epochs
    epoch 1/10 train_loss 0.5827 acc 0.740 | val_loss 0.4322 acc 0.795
    epoch 2/10 train_loss 0.4588 acc 0.831 | val_loss 0.4267 acc 0.839
    epoch 3/10 train_loss 0.3233 acc 0.894 | val_loss 0.1586 acc 0.955
    epoch 4/10 train_loss 0.1115 acc 0.966 | val_loss 1.5692 acc 0.554
    epoch 5/10 train_loss 0.0551 acc 0.986 | val_loss 0.0820 acc 0.964
    epoch 6/10 train_loss 0.0253 acc 0.989 | val_loss 0.0049 acc 1.000
    epoch 7/10 train_loss 0.0093 acc 0.998 | val_loss 0.0030 acc 1.000
    epoch 8/10 train_loss 0.0036 acc 1.000 | val_loss 0.0110 acc 0.991
    epoch 9/10 train_loss 0.0029 acc 1.000 | val_loss 0.0117 acc 0.991
    epoch 10/10 train_loss 0.0044 acc 1.000 | val_loss 0.0106 acc 0.991
    kept epoch 7 (best validation loss 0.0030)
[4/6] training the audio stream
  audio: 631 train clips, 120 val clips, 284 attacked, 8 epochs
    epoch 1/8 train_loss 0.6469 acc 0.634 | val_loss 0.5973 acc 0.725
    epoch 2/8 train_loss 0.3368 acc 0.868 | val_loss 0.2895 acc 0.908
    epoch 3/8 train_loss 0.2271 acc 0.919 | val_loss 0.1536 acc 0.958
    epoch 4/8 train_loss 0.1391 acc 0.959 | val_loss 0.2035 acc 0.917
    epoch 5/8 train_loss 0.0881 acc 0.975 | val_loss 0.0901 acc 0.975
    epoch 6/8 train_loss 0.0501 acc 0.992 | val_loss 0.0905 acc 0.975
    epoch 7/8 train_loss 0.0336 acc 0.994 | val_loss 0.0727 acc 0.975
    epoch 8/8 train_loss 0.0259 acc 0.995 | val_loss 0.0741 acc 0.975
    kept epoch 7 (best validation loss 0.0727)
[5/6] scoring every split and calibrating
  video threshold 0.0179, audio threshold 0.6029, fusion weight 0.05 threshold 0.0434, logistic threshold 0.0869
[6/6] evaluating on the seen and unseen test splits

==============================================================================
spoofline pipeline summary   profile=full  seed=20250117
==============================================================================
corpus            1600 clips, 80 identities, 16 frames of 64x64, 2.0 s at 16000 Hz
combinations      audio_only 400 | bonafide 400 | both 400 | video_only 400
video families    bonafide 800 | video_print 200 | video_recompress 200 | video_replay 200 | video_splice 200
audio families    audio_conversion 200 | audio_replay 200 | audio_splice 200 | audio_vocoder 200 | bonafide 800
unseen families   video_splice, audio_vocoder
splits            train 751 | calib 234 | seen_test 197 | unseen_test 123
identity pools    train 48 | calib 16 | test 16

training
  video  10 epochs, kept epoch 7  train_loss 0.0093 acc 0.998  |  val_loss 0.0030 acc 1.000  (639 train / 112 val clips)
  audio  8 epochs, kept epoch 7  train_loss 0.0336 acc 0.994  |  val_loss 0.0727 acc 0.975  (631 train / 120 val clips)

calibration on the calib split, target precision 0.95
  video  platt a=+0.774 b=-0.228   threshold 0.0179   calib precision 0.953   recall 0.631   target met true
  audio  platt a=+0.783 b=-1.763   threshold 0.6029   calib precision 0.957   recall 0.562   target met true
  fused  weight 0.05 on video           threshold 0.0434   calib precision 0.952   recall 1.000   target met true
  logistic p_video +3.252  p_audio +2.474  disagreement +3.016  intercept -2.650
           threshold 0.0869   calib precision 0.952   recall 1.000   target met true

clip level metrics at the calibrated operating points
  split       detector        P      R     F1    EER    AUC     n  attacks
  seen_test   video       0.990  0.643  0.780  0.256  0.848   197      154
  seen_test   audio       0.969  0.604  0.744  0.292  0.785   197      154
  seen_test   fused       0.956  0.994  0.975  0.140  0.947   197      154
  seen_test   logistic    0.963  1.000  0.981  0.023  0.998   197      154
  unseen_test video       1.000  0.475  0.644  0.387  0.705   123       80
  unseen_test audio       0.864  0.237  0.373  0.279  0.806   123       80
  unseen_test fused       0.941  0.600  0.733  0.188  0.869   123       80
  unseen_test logistic    0.948  0.688  0.797  0.125  0.883   123       80

decision rule comparison at the same thresholds
  split       rule            P      R     F1
  seen_test   and         1.000  0.260  0.412
  seen_test   or          0.974  0.987  0.981
  seen_test   weighted    0.956  0.994  0.975
  seen_test   logistic    0.963  1.000  0.981
  unseen_test and         1.000  0.113  0.202
  unseen_test or          0.941  0.600  0.733
  unseen_test weighted    0.941  0.600  0.733
  unseen_test logistic    0.948  0.688  0.797

per family detection rate, fused detector at its calibrated threshold
  split       family                   n  detected    rate
  seen_test   bonafide                43         7   0.163
  seen_test   audio_conversion        35        35   1.000
  seen_test   audio_replay            30        30   1.000
  seen_test   audio_splice            33        33   1.000
  seen_test   video_print             34        34   1.000
  seen_test   video_recompress        27        26   0.963
  seen_test   video_replay            37        37   1.000
  unseen_test bonafide                43         3   0.070
  unseen_test audio_conversion         8         8   1.000
  unseen_test audio_replay             7         7   1.000
  unseen_test audio_splice             2         2   1.000
  unseen_test audio_vocoder           41        27   0.659
  unseen_test video_print              6         6   1.000
  unseen_test video_recompress         6         6   1.000
  unseen_test video_replay             1         1   1.000
  unseen_test video_splice            44        23   0.523

which stream triggered each fusion decision, silencing one stream at a time
  split       detector  clips          none  video  audio either  joint
  seen_test   fused     bonafide         36      0      6      0      1
  seen_test   fused     video_only        1     51      0      3      1
  seen_test   fused     audio_only        0      0     56      0      0
  seen_test   fused     both              0      1      1     40      0
  seen_test   logistic  bonafide         37      0      6      0      0
  seen_test   logistic  video_only        0     53      0      3      0
  seen_test   logistic  audio_only        0      0     56      0      0
  seen_test   logistic  both              0      1      0     41      0
  unseen_test fused     bonafide         40      0      3      0      0
  unseen_test fused     video_only       18      2      2      0      0
  unseen_test fused     audio_only       11      0     12      0      0
  unseen_test fused     both              3      7     13     11      1
  unseen_test logistic  bonafide         40      0      3      0      0
  unseen_test logistic  video_only       12      8      2      0      0
  unseen_test logistic  audio_only       11      0     12      0      0
  unseen_test logistic  both              2      9      8     16      0

headline, unseen attack families
  precision   fused 0.941   logistic 0.948   video 1.000   audio 0.864
  recall      fused 0.600   logistic 0.688   video 0.475   audio 0.237
  gap         precision minus best single stream: fused -0.059   logistic -0.052
  verdict     fused precision 0.941 is BELOW the best single stream (video 1.000)
              fused F1 0.733 and AUC 0.869 against video F1 0.644 and AUC 0.705
  logistic    logistic precision 0.948 is BELOW the best single stream (video 1.000)
              logistic F1 0.797 and AUC 0.883 against video F1 0.644 and AUC 0.705

wall clock
  generate            84.5s
  load                 4.5s
  train_video        331.0s
  train_audio        107.6s
  calibrate           49.8s
  evaluate             0.0s
  total              577.4s
==============================================================================
```


### Scoring one clip

```
$ uv run spoofline score data/full/clips/clip_00003.npz    # bona fide in both streams
clip                  data/full/clips/clip_00003.npz
video_logit           -6.9419
audio_logit           -5.0318
video_probability     0.0037
audio_probability     0.0033
fused_probability     0.0033
logistic_probability  0.0673
video_flags           False
audio_flags           False
decision              bonafide
logistic_decision     bonafide
triggered_by          none

$ uv run spoofline score data/full/clips/clip_00004.npz    # audio_conversion, video untouched
clip                  data/full/clips/clip_00004.npz
video_logit           -6.8527
audio_logit           7.4499
video_probability     0.0039
audio_probability     0.9832
fused_probability     0.9342
logistic_probability  0.9399
video_flags           False
audio_flags           True
decision              attack
logistic_decision     attack
triggered_by          audio

$ uv run spoofline score data/full/clips/clip_00000.npz    # video_recompress, audio untouched
clip                  data/full/clips/clip_00000.npz
video_logit           5.5226
audio_logit           -4.3175
video_probability     0.9828
audio_probability     0.0058
fused_probability     0.0547
logistic_probability  0.9709
video_flags           True
audio_flags           False
decision              attack
logistic_decision     attack
triggered_by          video
```

The third case is the weighted fusion doing its job: the video term alone carries the fused score of 0.0547 over the 0.0434 threshold while the audio stream, correctly, sees nothing wrong, and `triggered_by` names video as the stream that did it.

### Reproducibility

Two separate `make demo` processes on the same machine and seed produce a byte
identical summary block, timings aside. `tests/test_determinism.py` asserts the
same property on the 24 clip fixture corpus, where a full pipeline run takes about
a second.

## Reading the numbers

### Does fusion hold its precision on unseen attack families?

Two readings of the same table, both worth stating.

**Across the seen to unseen boundary the fused detector holds up.** Its precision
goes 0.956 on families it trained on to 0.941 on families it has never seen, a
drop of 1.5 points, while the target set on the calibration split was 0.95. The
calibrated operating point transfers.

**Against the best single stream on unseen families it does not.** Video alone
reaches precision 1.000 there, fused 0.941, so the strict claim fails and the
summary block says `BELOW`. What the video stream buys that precision with is
recall: it flags 38 of the 80 attacks, because it is structurally blind to a clip
whose audio was vocoded and whose video is genuine. Fused catches 48 of 80 at
0.941 precision, which is 3 false positives out of 43 bona fide clips.

| unseen families | P | R | F1 | EER | AUC |
| --- | --- | --- | --- | --- | --- |
| video only | 1.000 | 0.475 | 0.644 | 0.387 | 0.705 |
| audio only | 0.864 | 0.237 | 0.373 | 0.279 | 0.806 |
| fused | 0.941 | 0.600 | 0.733 | 0.188 | 0.869 |

| seen families | P | R | F1 | EER | AUC |
| --- | --- | --- | --- | --- | --- |
| video only | 0.990 | 0.643 | 0.780 | 0.256 | 0.848 |
| audio only | 0.969 | 0.604 | 0.744 | 0.292 | 0.785 |
| fused | 0.956 | 0.994 | 0.975 | 0.140 | 0.947 |

So the accurate one line version is: fusion keeps precision within 6 points of the
best single stream on unseen attacks while catching 26 percent more of them, and
it is clearly ahead on F1 and AUC on both splits. It does not dominate on
precision, and this repo says so rather than quietly reporting F1.

### What the per-family rows show

Held out families are the hard ones, exactly as the protocol intends. At the fused
operating point, every seen family is detected at 0.963 to 1.000, while the two
held out families sit at 0.523 for `video_splice` and 0.659 for `audio_vocoder`.
The bona fide false alarm rate is 0.163 on the seen split and 0.070 on the unseen
split.

### What the fusion weight converged to

The grid search picked `w = 0.05` with a threshold of 0.0434 on
`0.05 * p_video + 0.95 * p_audio`. Because the video term alone can contribute
0.05, that is a soft OR: a confident video score clears the threshold on its own,
and so does a moderate audio score. That is the right shape for a label that says
"attack if either stream was attacked", and the explicit OR rule lands on the same
numbers on the unseen split (0.941 / 0.600). The AND rule is the other extreme:
precision 1.000 at recall 0.113.


## Variance over seeds and held out families

The demo above is one seed with one held out pair. `spoofline sweep` repeats the
whole protocol for every pairing of one held out video family with one held out
audio family (4 x 4 = 16 splits) and for several seeds, then reports the mean,
the sample standard deviation and a percentile bootstrap 95 percent interval of
the mean over all runs.

```bash
uv run spoofline sweep                          # reduced profile, 3 seeds, 4 worker processes
uv run spoofline sweep --seeds 5 --workers 6
```

**The variance table below comes from the reduced profile, not from the full demo
profile.** Sixteen splits times three seeds of full size training do not fit on a
CPU, so the sweep runs the `reduced` profile: 960 clips from 80 identities, 8
frames of 64x64 and 1.0 s of audio per clip, 8 video epochs and 6 audio epochs at
batch 16, 2 threads per run with 4 runs in parallel. The profile was sized on wall
clock and on the validation loss inside the training identities only; no test
split, seen or unseen, was looked at while choosing it. Its streams are weaker
than the full profile's, so compare its numbers with each other rather than with
the demo run. This sweep took 1125 s on the same 10 core machine while another CPU
heavy job was running on it.

```
$ uv run spoofline sweep
==============================================================================
spoofline sweep   profile=reduced   seeds=3   held out pairs=16   runs=48
==============================================================================
corpus            960 clips, 80 identities, 8 frames of 64x64, 1.0 s at 16000 Hz
training          video 8 epochs, audio 6 epochs, batch 16, 2 threads per run
seeds             20250117, 20250118, 20250119
target precision  0.95 on the calib split of every run

mean, sample std and bootstrap 95% interval of the mean over runs
  split       detector  metric        mean    std   ci low  ci high
  seen_test   video     precision    0.979  0.028    0.971    0.986
  seen_test   video     recall       0.560  0.091    0.535    0.585
  seen_test   video     f1           0.708  0.076    0.687    0.729
  seen_test   video     eer          0.306  0.059    0.290    0.323
  seen_test   video     auc          0.762  0.066    0.743    0.780
  seen_test   audio     precision    0.991  0.014    0.987    0.995
  seen_test   audio     recall       0.494  0.108    0.465    0.525
  seen_test   audio     f1           0.652  0.097    0.626    0.679
  seen_test   audio     eer          0.269  0.045    0.257    0.282
  seen_test   audio     auc          0.807  0.048    0.794    0.821
  seen_test   fused     precision    0.985  0.016    0.980    0.989
  seen_test   fused     recall       0.834  0.088    0.810    0.859
  seen_test   fused     f1           0.901  0.052    0.886    0.916
  seen_test   fused     eer          0.119  0.055    0.103    0.134
  seen_test   fused     auc          0.939  0.040    0.928    0.950
  unseen_test video     precision    0.934  0.095    0.906    0.958
  unseen_test video     recall       0.412  0.166    0.367    0.458
  unseen_test video     f1           0.556  0.172    0.507    0.603
  unseen_test video     eer          0.366  0.113    0.336    0.399
  unseen_test video     auc          0.690  0.127    0.654    0.725
  unseen_test audio     precision    0.939  0.067    0.921    0.958
  unseen_test audio     recall       0.268  0.121    0.235    0.302
  unseen_test audio     f1           0.403  0.142    0.364    0.442
  unseen_test audio     eer          0.363  0.101    0.335    0.391
  unseen_test audio     auc          0.684  0.126    0.648    0.718
  unseen_test fused     precision    0.944  0.048    0.930    0.957
  unseen_test fused     recall       0.566  0.132    0.529    0.602
  unseen_test fused     f1           0.699  0.113    0.666    0.729
  unseen_test fused     eer          0.274  0.092    0.248    0.301
  unseen_test fused     auc          0.794  0.094    0.767    0.820

derived, per run then summarised
  unseen fused precision minus best single stream  -0.038  0.043   -0.050   -0.026
  seen minus unseen fused precision                 0.041  0.041    0.029    0.052

unseen test precision and recall per held out pair, mean over seeds
  video family      audio family       video P audio P fused P fused R
  video_replay      audio_replay         0.949   0.965   0.960   0.560
  video_replay      audio_vocoder        0.920   0.964   0.952   0.668
  video_replay      audio_conversion     0.917   0.954   0.922   0.507
  video_replay      audio_splice         0.973   1.000   0.970   0.744
  video_print       audio_replay         0.871   0.926   0.913   0.487
  video_print       audio_vocoder        0.871   0.985   0.943   0.566
  video_print       audio_conversion     0.803   0.969   0.920   0.619
  video_print       audio_splice         0.883   0.924   0.924   0.591
  video_splice      audio_replay         1.000   0.907   0.974   0.573
  video_splice      audio_vocoder        0.973   0.910   0.959   0.578
  video_splice      audio_conversion     0.978   0.970   0.986   0.624
  video_splice      audio_splice         0.979   0.884   0.943   0.526
  video_recompress  audio_replay         0.881   0.912   0.920   0.424
  video_recompress  audio_vocoder        0.938   0.923   0.899   0.575
  video_recompress  audio_conversion     1.000   0.922   0.964   0.508
  video_recompress  audio_splice         1.000   0.916   0.954   0.498

wall clock        1124.8s
==============================================================================
```

Each run's raw logits are cached in `runs/sweep/<profile>/seed_<seed>/`, keyed by
a hash of the profile, so a second `spoofline sweep` recomputes calibration, fusion
and every metric from the cache in seconds instead of retraining.

### What the sweep says

* **Fused precision mostly holds across the seen to unseen boundary.** Over 48
  runs it is 0.985 on seen families and 0.944 on unseen ones, a drop of 0.041 with
  an interval of 0.029 to 0.052, which leaves the unseen mean just under the 0.95
  calibration target.
* **It is below the best single stream of the same run, and that is the typical
  outcome.** Per run, fused unseen precision minus the better of the two streams
  averages -0.038 (interval -0.050 to -0.026). The fused mean of 0.944 is above
  both stream means (video 0.934, audio 0.939) only because which stream wins
  changes from run to run. The demo's `BELOW` verdict was not bad luck.
* **What fusion buys is recall.** On unseen families fused recall is 0.566
  against 0.412 for video and 0.268 for audio, F1 0.699 against 0.556 and 0.403,
  and AUC 0.794 against 0.690 and 0.684.
* **Single runs are noisy.** The standard deviation over runs is 0.095 for unseen
  video precision and 0.132 for unseen fused recall, so differences of a few
  points in the single run demo are inside the noise.
* **Some held out pairs are much harder than others.** Fused unseen recall ranges
  from 0.424 with `video_recompress` and `audio_replay` held out to 0.744 with
  `video_replay` and `audio_splice`; fused unseen precision ranges from 0.899 to
  0.986.
* **The intervals are a lower bound on the uncertainty.** The bootstrap resamples
  runs as if they were independent, but runs that share a seed share a corpus and
  identity pools.

## Learned fusion against the weighted sum

v3 adds a second fusion. A logistic regression takes three features per clip, both
calibrated probabilities and their absolute disagreement `|p_video - p_audio|`, and
is fitted with a small L2 penalty on the calibration split only. Its operating
point comes from the same precision constrained search as every other detector:
the threshold with the most recall among those whose calibration precision
reaches 0.95. The disagreement feature is there for a reason:
`max(p_video, p_audio) = (p_video + p_audio) / 2 + |p_video - p_audio| / 2`, so
with it a linear model can express an OR of the two streams, which is what an
"attacked in either modality" label calls for. Its coefficient came out positive
in all 48 sweep runs.

Every fusion decision is also attributed to a stream by silencing one stream at a
time (setting its probability to 0): `video` or `audio` if that stream alone keeps
the clip flagged, `either` if each alone would, `joint` if only the two together
do, and `none` if the clip is not flagged. `spoofline score` prints it as
`triggered_by`, and the pipeline summary tabulates it against the modality that
was actually attacked.

### The comparison, over the 48 sweep runs

Same cached logits as the v2 sweep, so the stream numbers are unchanged and every
row is the mean over 48 runs of the reduced profile (bootstrap 95 percent interval
of the mean in brackets).

| unseen families | precision | recall | F1 | AUC |
| --- | --- | --- | --- | --- |
| video only | 0.934 [0.906, 0.958] | 0.412 | 0.556 | 0.690 |
| audio only | 0.939 [0.921, 0.958] | 0.268 | 0.403 | 0.684 |
| AND rule | 0.916 [0.836, 0.977] | 0.107 | 0.186 | n/a |
| OR rule | 0.934 [0.916, 0.951] | 0.574 | 0.703 | n/a |
| weighted sum | 0.944 [0.930, 0.957] | 0.566 | 0.699 | 0.794 |
| logistic | 0.946 [0.930, 0.960] | 0.555 | 0.689 | 0.795 |

| seen families | precision | recall | F1 | AUC |
| --- | --- | --- | --- | --- |
| AND rule | 0.998 | 0.220 | 0.356 | n/a |
| OR rule | 0.980 | 0.834 | 0.899 | n/a |
| weighted sum | 0.985 | 0.834 | 0.901 | 0.939 |
| logistic | 0.984 | 0.834 | 0.901 | 0.941 |

**Does the learned fusion narrow the gap to the best single stream on unseen
precision? Not measurably.** Per run, unseen precision minus the better single
stream of that run is -0.038 [-0.050, -0.026] for the weighted sum and -0.036
[-0.050, -0.023] for the logistic fusion. Paired run by run, logistic minus
weighted is +0.002 [-0.011, +0.017] on unseen precision, -0.011 [-0.025, +0.003]
on unseen recall and -0.009 [-0.020, +0.001] on unseen F1; the logistic fusion is
higher on unseen precision in 19 runs, equal in 12 and lower in 17. On seen
families the two are within 0.002 on every metric. The honest reading is that the
learned fusion is a wash: it matches the weighted sum, it does not close the
precision gap, and it gives up about a point of unseen recall. The weighted sum
stays the primary `decision`; the logistic fusion is reported beside it.
The AND rule's unseen precision has a standard deviation of 0.248 because in some
runs it flags nothing, which counts as precision 0.

### The comparison on the single demo run

The v3 demo run (full profile, seed 20250117, `video_splice` and `audio_vocoder`
held out) reproduces the v1 numbers for both streams and the weighted sum exactly
and adds the logistic rows. The logistic fusion fitted `p_video +3.252`,
`p_audio +2.474`, `disagreement +3.016`, intercept `-2.650` and a threshold of
0.0869 on the calibration split alone.

| unseen families | P | R | F1 | EER | AUC |
| --- | --- | --- | --- | --- | --- |
| video only | 1.000 | 0.475 | 0.644 | 0.387 | 0.705 |
| audio only | 0.864 | 0.237 | 0.373 | 0.279 | 0.806 |
| AND rule | 1.000 | 0.113 | 0.202 | n/a | n/a |
| OR rule | 0.941 | 0.600 | 0.733 | n/a | n/a |
| weighted sum | 0.941 | 0.600 | 0.733 | 0.188 | 0.869 |
| logistic | 0.948 | 0.688 | 0.797 | 0.125 | 0.883 |

| seen families | P | R | F1 | EER | AUC |
| --- | --- | --- | --- | --- | --- |
| AND rule | 1.000 | 0.260 | 0.412 | n/a | n/a |
| OR rule | 0.974 | 0.987 | 0.981 | n/a | n/a |
| weighted sum | 0.956 | 0.994 | 0.975 | 0.140 | 0.947 |
| logistic | 0.963 | 1.000 | 0.981 | 0.023 | 0.998 |

On this one run the logistic fusion looks clearly better: unseen precision 0.948
against 0.941, the gap to video alone shrinks from -0.059 to -0.052, and it
catches 55 of 80 unseen attacks instead of 48. The sweep is why the README does not
claim that. Across 48 runs the same paired comparison averages +0.002 on unseen
precision and -0.011 on unseen recall, so this run is a favourable draw, which is
exactly the kind of single run difference the v2 sweep was built to check.

Attribution on the same run says where decisions come from:

* Every audio only attack the fusion flags is attributed to audio (56 of 56 on the
  seen split, 12 of 12 flagged on the unseen split), and 51 of the 56 seen video
  only attacks to video, with 3 more that either stream alone would flag.
* The bona fide false alarms are audio triggered: all 3 on the unseen split for
  both fusions, and 6 of the 7 on the seen split for the weighted sum (the seventh
  needs both streams together).
* The misses sit on the held out families: the weighted sum leaves 18 of the 22
  unseen video only clips (all `video_splice`) and 11 of the 23 unseen audio only
  clips (all `audio_vocoder`) unflagged.

## Robustness to benign degradation

A deployed detector sees genuine clips that have been compressed, filmed on a
noisy sensor, relit, resampled, recorded in a room or cut by a network glitch. If
any of that trips the detector, its precision in the field is lower than the
calibration split promised. `spoofline robustness` takes the bona fide clips of the
two test splits of a finished run, degrades each one, keeps its label, and counts
how often each detector now calls it an attack. Every flag in this table is a
false alarm.

```bash
uv run spoofline robustness            # uses runs/full and data/full from make demo
```

| perturbation | stream | severities, mildest first |
| --- | --- | --- |
| `jpeg` | video | quality 90, 70, 50, 30, 15, one OpenCV encode and decode per frame |
| `video_noise` | video | gaussian pixel noise, sigma 2, 5, 10, 20 |
| `brightness_contrast` | video | contrast falls by s and brightness rises by 60 s over the clip, s = 0.1, 0.2, 0.35, 0.5 |
| `video_dropout` | video | 1, 2, 4 frames replaced by the frame before, as a stalled capture does |
| `audio_noise` | audio | gaussian noise at SNR 40, 30, 20, 10 dB |
| `resample` | audio | down to 12000, 8000, 6000, 4000 Hz and back to 16000 Hz |
| `reverb` | audio | direct path plus a decaying noise tail with RT60 0.1, 0.2, 0.35 s, RMS matched |
| `audio_dropout` | audio | 1, 3, 6 separate 30 ms stretches set to zero |

The severity ladders and the abstain margins were fixed before the run and not
adjusted after seeing the table. The perturbed stream is rescored; the other
stream keeps its clean score.

### False alarm rate on degraded bona fide clips

Measured on the v3 demo run (full profile, seed 20250117) over its 86 bona fide
test clips, 43 from each test split, so one clip moves a rate by 0.012. This is a
single run; the table has no variance estimate.

| perturbation | stream | severity | video | audio | weighted sum | logistic |
| --- | --- | --- | --- | --- | --- | --- |
| clean | none | none | 0.012 | 0.070 | 0.116 | 0.105 |
| jpeg | video | quality 90 | 0.023 | 0.070 | 0.116 | 0.116 |
| jpeg | video | quality 70 | 0.256 | 0.070 | 0.116 | 0.163 |
| jpeg | video | quality 50 | 0.477 | 0.070 | 0.116 | 0.349 |
| jpeg | video | quality 30 | 0.895 | 0.070 | 0.174 | 0.698 |
| jpeg | video | quality 15 | 1.000 | 0.070 | 0.884 | 1.000 |
| video_noise | video | sigma 2 | 0.012 | 0.070 | 0.116 | 0.105 |
| video_noise | video | sigma 5 | 0.000 | 0.070 | 0.116 | 0.105 |
| video_noise | video | sigma 10 | 0.000 | 0.070 | 0.116 | 0.105 |
| video_noise | video | sigma 20 | 0.000 | 0.070 | 0.116 | 0.105 |
| brightness_contrast | video | drift 0.1 | 0.093 | 0.070 | 0.116 | 0.128 |
| brightness_contrast | video | drift 0.2 | 0.372 | 0.070 | 0.140 | 0.372 |
| brightness_contrast | video | drift 0.35 | 0.942 | 0.070 | 0.244 | 0.616 |
| brightness_contrast | video | drift 0.5 | 1.000 | 0.070 | 0.302 | 0.895 |
| video_dropout | video | frames 1 | 0.012 | 0.070 | 0.116 | 0.105 |
| video_dropout | video | frames 2 | 0.012 | 0.070 | 0.116 | 0.105 |
| video_dropout | video | frames 4 | 0.012 | 0.070 | 0.116 | 0.105 |
| audio_noise | audio | snr_db 40 | 0.012 | 0.419 | 0.849 | 0.837 |
| audio_noise | audio | snr_db 30 | 0.012 | 1.000 | 1.000 | 1.000 |
| audio_noise | audio | snr_db 20 | 0.012 | 1.000 | 1.000 | 1.000 |
| audio_noise | audio | snr_db 10 | 0.012 | 1.000 | 1.000 | 1.000 |
| resample | audio | hz 12000 | 0.012 | 0.953 | 1.000 | 1.000 |
| resample | audio | hz 8000 | 0.012 | 0.977 | 1.000 | 1.000 |
| resample | audio | hz 6000 | 0.012 | 1.000 | 1.000 | 1.000 |
| resample | audio | hz 4000 | 0.012 | 1.000 | 1.000 | 1.000 |
| reverb | audio | rt60_s 0.1 | 0.012 | 0.326 | 0.709 | 0.686 |
| reverb | audio | rt60_s 0.2 | 0.012 | 0.977 | 1.000 | 1.000 |
| reverb | audio | rt60_s 0.35 | 0.012 | 1.000 | 1.000 | 1.000 |
| audio_dropout | audio | gaps 1 | 0.012 | 0.058 | 0.151 | 0.140 |
| audio_dropout | audio | gaps 3 | 0.012 | 0.093 | 0.291 | 0.291 |
| audio_dropout | audio | gaps 6 | 0.012 | 0.233 | 0.581 | 0.558 |

### What the table says

* **Clean capture is the only condition the calibration covers.** With no
  degradation the false alarm rate is 0.012 for video, 0.070 for audio, 0.116 for
  the weighted sum and 0.105 for the logistic fusion.
* **Benign audio channel changes break the audio stream, and fusion makes it
  worse.** Gaussian noise at 40 dB SNR already flags 0.419 of genuine clips on the
  audio stream and 0.849 on the weighted sum; at 30 dB and below every clip is
  flagged. A round trip through 12 kHz flags 0.953 on audio and every clip on both
  fusions, and a small room (RT60 0.1 s) flags 0.326 on audio and 0.709 on the
  weighted sum. The likely reason is that the training data has attacks that are
  channel effects (`audio_replay` is a room response, band limiting and a noise
  floor) and no benign channel variation, so the stream learned that any channel
  change means an attack. The fusions sit above the audio stream because their
  operating points are low on the audio axis: the weighted sum puts 0.95 on audio
  with a threshold of 0.0434, so an audio probability near 0.046 flags a clip,
  while the audio stream's own threshold is 0.6029.
* **Compression and lighting trip the video stream, and the weighted sum mostly
  shields against them.** JPEG quality 70 flags 0.256 of clips on video and quality
  30 flags 0.895, but the weighted sum stays at 0.116 and 0.174 because it puts only
  0.05 on video; only quality 15 gets through (0.884). Brightness and contrast drift
  of 0.35 flags 0.942 on video and 0.244 on the weighted sum. The logistic fusion
  weights video more and follows the video stream further (0.698 at quality 30,
  0.616 at drift 0.35). Heavy JPEG is also close to a trained attack family,
  `video_recompress`.
* **Pixel noise and frame dropout do nothing.** Video noise up to sigma 20 and up
  to 4 frozen frames leave every rate at or below its clean value.
* **What this does to the headline claim.** The fused precision that holds on
  unseen attack families holds for clean capture. A modest benign audio change
  makes the fused detector flag most genuine clips, at least for the synthetic
  degradations measured here. A deployment would need benign channel augmentation in
  training and recalibration on the target channel; this repo has neither.

### Abstaining when the streams disagree

A fusion can abstain on a clip when `|p_video - p_audio|` exceeds a margin. The
margin ladder 1.0 (never abstain), 0.9, 0.75, 0.5 and 0.25 was fixed before the run.
Coverage is the share of clips kept; precision and recall are measured on the kept
clips only.

| split | fusion | margin | coverage | precision | recall | attacks abstained | bona fide abstained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seen_test | weighted sum | 1.00 | 1.000 | 0.956 | 0.994 | 0 | 0 |
| seen_test | weighted sum | 0.90 | 0.487 | 0.881 | 0.981 | 101 | 0 |
| seen_test | weighted sum | 0.75 | 0.452 | 0.885 | 0.979 | 107 | 1 |
| seen_test | weighted sum | 0.50 | 0.416 | 0.913 | 1.000 | 112 | 3 |
| seen_test | weighted sum | 0.25 | 0.391 | 0.927 | 1.000 | 116 | 4 |
| seen_test | logistic | 1.00 | 1.000 | 0.963 | 1.000 | 0 | 0 |
| seen_test | logistic | 0.90 | 0.487 | 0.898 | 1.000 | 101 | 0 |
| seen_test | logistic | 0.75 | 0.452 | 0.904 | 1.000 | 107 | 1 |
| seen_test | logistic | 0.50 | 0.416 | 0.933 | 1.000 | 112 | 3 |
| seen_test | logistic | 0.25 | 0.391 | 0.950 | 1.000 | 116 | 4 |
| unseen_test | weighted sum | 1.00 | 1.000 | 0.941 | 0.600 | 0 | 0 |
| unseen_test | weighted sum | 0.90 | 0.821 | 1.000 | 0.475 | 19 | 3 |
| unseen_test | weighted sum | 0.75 | 0.772 | 1.000 | 0.418 | 25 | 3 |
| unseen_test | weighted sum | 0.50 | 0.748 | 1.000 | 0.385 | 28 | 3 |
| unseen_test | weighted sum | 0.25 | 0.691 | 1.000 | 0.333 | 35 | 3 |
| unseen_test | logistic | 1.00 | 1.000 | 0.948 | 0.688 | 0 | 0 |
| unseen_test | logistic | 0.90 | 0.821 | 1.000 | 0.590 | 19 | 3 |
| unseen_test | logistic | 0.75 | 0.772 | 1.000 | 0.545 | 25 | 3 |
| unseen_test | logistic | 0.50 | 0.748 | 1.000 | 0.519 | 28 | 3 |
| unseen_test | logistic | 0.25 | 0.691 | 1.000 | 0.444 | 35 | 3 |

Abstention is not a general precision fix here, because an attack on one modality
is exactly a clip where the two streams disagree. On the unseen split the weighted
sum at margin 0.9 keeps 0.821 of clips at precision 1.000: the 3 abstained bona
fide clips are its 3 audio triggered false alarms. The price is 19 of the 80
attacks abstained on and recall on the kept clips falling from 0.600 to 0.475. On
the seen split the same margin keeps 0.487 of clips, abstains on 101 attacks and on
no bona fide clip, and precision on what is kept falls from 0.956 to 0.881. Whether
abstention helps depends on whether the false alarms or the true detections are the
ones with disagreeing streams, and that differs between the two splits of the same
run.

## Deployment path

v5 turns a finished run into something a service can load: both streams as ONNX,
a model card, batch scoring with a stable JSON schema, and measured CPU latency.
Every number in this section comes from the demo run above.

```bash
uv run spoofline export --onnx          # runs/full/onnx/video.onnx, audio.onnx, export.json
uv run spoofline model-card             # runs/full/MODEL_CARD.md
uv run spoofline score data/full/clips/clip_00000.npz data/full/clips/clip_00003.npz --json
uv run spoofline bench                  # runs/full/latency.json
```

### ONNX export and parity

Each stream is exported with its normaliser folded into the graph and a dynamic
batch and step axis. Deployed clips all have the same number of steps, so the
exported graph runs the LSTM on the dense tensor instead of a packed sequence;
with no padding the two are the same computation. Export fails unless every
clip's ONNX Runtime logit is within 1e-4 of the PyTorch logit.

```
$ uv run spoofline export --onnx --parity-clips 32
video: runs/full/onnx/video.onnx  max |onnx - torch| 3.81e-06 over 32 clips
audio: runs/full/onnx/audio.onnx  max |onnx - torch| 3.81e-06 over 32 clips
wrote runs/full/onnx/export.json
```

### Model card

`spoofline model-card` renders the data note, splits, every threshold with its
score formula, seen and unseen metrics for all four detectors, the robustness
table at the heaviest severity of each perturbation, and the limitations, all read
from `results.json` and `robustness.json` of the run. The card of the demo run is
committed as [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

### Batch scoring as JSON

`spoofline score` takes any number of clips and scores them in batches. With
`--json` it emits one document, `schema_version` 1:

```json
{
  "schema_version": 1,
  "run_dir": "runs/full",
  "clips": [
    {
      "clip": "data/full/clips/clip_00004.npz",
      "video_logit": -6.852695465087891,
      "audio_logit": 7.449930667877197,
      "video_probability": 0.003949245872857415,
      "audio_probability": 0.9832120510611272,
      "fused_probability": 0.9342489108017137,
      "logistic_probability": 0.9398798501923926,
      "video_flags": false,
      "audio_flags": true,
      "decision": "attack",
      "logistic_decision": "attack",
      "triggered_by": "audio"
    }
  ]
}
```

### CPU latency

Per clip wall time on one thread, after 5 warm up calls. A stream row is feature
extraction plus one forward pass; end to end adds reading the npz, calibration,
both fusions and attribution.

```
==============================================================================
spoofline bench   clips=200   threads=1   CPU
==============================================================================
per clip wall time; a stream includes its feature extraction, end to end adds
npz decode, calibration, both fusions and attribution
  engine  stage          p50 ms   p95 ms  mean ms     n
  torch   video           10.07    15.03    11.91   200
  torch   audio            2.54     7.20     3.96   200
  torch   end_to_end      17.04    42.82    20.45   200
  onnx    video            4.50     6.35     4.81   200
  onnx    audio            2.05     3.16     2.20   200
  onnx    end_to_end       8.44    17.35    10.44   200
==============================================================================
```

The machine was running another CPU heavy job throughout, at a load average near
17 on 10 cores, so the p95 figures in particular are inflated; treat these as
numbers from a busy laptop, not a quiet benchmark host. ONNX Runtime is about
twice as fast as PyTorch here on both streams, and the end to end row adds the
npz decode and the feature extraction that both engines share.

## Plugging in a real corpus

`spoofline/data/sources.py` defines the `ClipSource` protocol:

```python
class ClipSource(Protocol):
    @property
    def records(self) -> list[ClipRecord]: ...
    @property
    def sample_rate(self) -> int: ...
    def load(self, clip_id: str) -> Clip: ...
    def __len__(self) -> int: ...
```

`DirectoryClipSource` implements it for media files on disk:

```
root/
  manifest.csv            clip_id,identity,video_family,audio_family,label
  video/<clip_id>.mp4     any container OpenCV can decode
  audio/<clip_id>.wav     mono or stereo, any sample rate
```

Family names must come from `spoofline.families`, with `bonafide` for a stream
that was not attacked, and `label` must be 1 if either stream was attacked. Frames
are uniformly sampled to `n_frames` and resized; audio is downmixed, resampled and
trimmed or padded. Everything downstream, including the split logic, works off the
manifest, so a real corpus gets the same leave-one-family-out treatment as the
generated one. `tests/test_sources.py` exercises the adapter on a written out clip.

## Limitations

* **The corpus is synthetic.** The absolute numbers describe this generator, not
  FaceForensics++ or ASVspoof. What transfers is the protocol and the code, not
  the precision figures.
* **The clips are small.** 16 frames of 64x64 and two seconds of 16 kHz audio, so
  the whole demo fits in seven minutes of CPU. Real face forensics works at much
  higher resolution and a print or replay attack is far subtler there.
* **The demo is one seed and one held out pair.** The demo block is a single run
  with `video_splice` and `audio_vocoder` held out. The sweep adds variance over
  3 seeds and all 16 held out pairs, but only on the reduced profile; no full
  profile sweep has been run.
* **The threshold is fitted on 234 clips.** Choosing the lowest threshold that
  reaches the target precision is the most optimistic choice available on a finite
  calibration set, so some of the seen to unseen drop is threshold sampling noise
  rather than family novelty.
* **Small cells in the per family table.** The unseen split is dominated by the two
  held out families by construction, so the seen family rows there have as few as
  one clip and their rates carry no weight.
* **Each stream is blind to the other modality by design.** A single stream is
  trained on its own modality label, so its clip level recall is capped near the
  fraction of attacks that touch its modality. That is the premise of the
  experiment, not a defect, but it is why single stream recall looks low.
* **Benign audio degradation breaks the detector.** See the robustness section:
  40 dB SNR noise or a 12 kHz resample makes the fused detector flag most genuine
  clips. Nothing in training covers benign channel variation.
* **No augmentation, no speaker or channel robustness work.** An earlier version of
  this repo used 40 identities and the audio stream memorised them: modality AUC
  1.00 on the calibration identities and 0.74 on the test identities. Widening the
  corpus to 80 identities fixed it. A real deployment would need far more.


## Layout

```
spoofline/
  config.py          run profiles (full, reduced, tiny) and all hyperparameters
  seeding.py         seed derivation, torch and numpy seeding
  families.py        the eight attack families
  metrics.py         precision, recall, F1, AUC, EER, per-family breakdown
  calibrate.py       Platt scaling and threshold selection at a target precision
  fusion.py          weighted and logistic fusion, AND and OR rules, per clip attribution
  train.py           per-stream training and scoring
  pipeline.py        the end to end run, calibration loading and evaluation
  sweep.py           repeated seed, leave two families out sweep with bootstrap summaries
  scoring.py         run scorer: checkpoints, calibration and fusions loaded once
  perturb.py         benign degradations of bona fide clips
  robustness.py      false alarms under degradation and the abstain rule
  export.py          ONNX export of both streams with a parity check
  model_card.py      model card rendered from a finished run
  bench.py           per clip CPU latency on PyTorch and ONNX Runtime
  report.py          the summary block
  cli.py             click commands
  data/
    synth.py         bona fide video and audio synthesis
    attacks_video.py four video attack families
    attacks_audio.py four audio attack families
    signal_stats.py  the statistics the attack direction tests assert on
    generate.py      corpus planning and rendering
    sources.py       ClipSource protocol, npz source, real corpus adapter
    features.py      video steps, log mel patches, normalisation
    dataset.py       splits and torch datasets
  models/
    cnn_lstm.py      the shared detector
tests/               pytest suite on the committed tiny fixture corpus
fixtures/tiny/       24 clip corpus, committed so tests never generate
```
