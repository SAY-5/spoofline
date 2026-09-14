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

`CHANGELOG.md` has the detail for every version.

## Measured demo run

Output of one `make demo` on a 10 core Apple silicon CPU, pasted verbatim:

```
$ make demo
[1/6] generating corpus 'full' into data/full
  generated 160/1600 clips (4.8s)
  generated 320/1600 clips (9.4s)
  generated 480/1600 clips (14.1s)
  generated 640/1600 clips (18.7s)
  generated 800/1600 clips (23.3s)
  generated 960/1600 clips (28.6s)
  generated 1120/1600 clips (34.3s)
  generated 1280/1600 clips (40.7s)
  generated 1440/1600 clips (45.7s)
  generated 1600/1600 clips (50.5s)
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
  video threshold 0.0179, audio threshold 0.6029, fusion weight 0.05 threshold 0.0434
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

clip level metrics at the calibrated operating points
  split       detector        P      R     F1    EER    AUC     n  attacks
  seen_test   video       0.990  0.643  0.780  0.256  0.848   197      154
  seen_test   audio       0.969  0.604  0.744  0.292  0.785   197      154
  seen_test   fused       0.956  0.994  0.975  0.140  0.947   197      154
  unseen_test video       1.000  0.475  0.644  0.387  0.705   123       80
  unseen_test audio       0.864  0.237  0.373  0.279  0.806   123       80
  unseen_test fused       0.941  0.600  0.733  0.188  0.869   123       80

decision rule comparison at the same thresholds
  split       rule            P      R     F1
  seen_test   and         1.000  0.260  0.412
  seen_test   or          0.974  0.987  0.981
  seen_test   weighted    0.956  0.994  0.975
  unseen_test and         1.000  0.113  0.202
  unseen_test or          0.941  0.600  0.733
  unseen_test weighted    0.941  0.600  0.733

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

headline, unseen attack families
  precision   fused 0.941   video 1.000   audio 0.864
  recall      fused 0.600   video 0.475   audio 0.237
  verdict     fused precision 0.941 is BELOW the best single stream (video 1.000)
              fused F1 0.733 and AUC 0.869 against video F1 0.644 and AUC 0.705

wall clock
  generate            50.7s
  load                 3.3s
  train_video        231.3s
  train_audio         75.3s
  calibrate           57.6s
  evaluate             0.0s
  total              418.2s
==============================================================================
```


### Scoring one clip

```
$ uv run spoofline score data/full/clips/clip_00003.npz    # bona fide in both streams
video_probability   0.0037
audio_probability   0.0033
fused_probability   0.0033
video_flags         False
audio_flags         False
decision            bonafide

$ uv run spoofline score data/full/clips/clip_00004.npz    # audio_conversion, video untouched
video_probability   0.0039
audio_probability   0.9832
fused_probability   0.9342
video_flags         False
audio_flags         True
decision            attack

$ uv run spoofline score data/full/clips/clip_00000.npz    # video_recompress, audio untouched
video_probability   0.9828
audio_probability   0.0058
fused_probability   0.0547
video_flags         True
audio_flags         False
decision            attack
```

The third case is the fusion rule doing its job: the video term alone carries the
fused score of 0.0547 over the 0.0434 threshold while the audio stream, correctly,
sees nothing wrong.

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
  fusion.py          weighted score fusion plus AND and OR rules
  train.py           per-stream training and scoring
  pipeline.py        the end to end run and the single clip scorer
  sweep.py           repeated seed, leave two families out sweep with bootstrap summaries
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
