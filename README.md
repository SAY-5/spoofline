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

## Measured demo run

<!-- DEMO BLOCK -->

## Reading the numbers

<!-- READING -->

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

<!-- LIMITATIONS -->

## Layout

```
spoofline/
  config.py          run profiles (full, tiny) and all hyperparameters
  seeding.py         seed derivation, torch and numpy seeding
  families.py        the eight attack families
  metrics.py         precision, recall, F1, AUC, EER, per-family breakdown
  calibrate.py       Platt scaling and threshold selection at a target precision
  fusion.py          weighted score fusion plus AND and OR rules
  train.py           per-stream training and scoring
  pipeline.py        the end to end run and the single clip scorer
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
