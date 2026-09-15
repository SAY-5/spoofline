# Spoofline model card

Rendered from the evaluation run with profile `full` and seed 20250117 by `spoofline model-card`.

## Model details

Two CNN-LSTM detectors, one per stream: a per step CNN encoder, a packed bidirectional LSTM, masked attention pooling and a single logit head. Each stream is trained on its own modality label, calibrated with a Platt map, and the two probabilities are fused by a weighted sum (the primary decision) and by a logistic regression over both probabilities and their disagreement.

* video: 10 epochs, kept epoch 7, 639 training clips, 112 validation clips.
* audio: 8 epochs, kept epoch 7, 631 training clips, 120 validation clips.

## Data note

No public spoofing corpus is used, because the public sets need signed licences. The corpus is 1600 generated clips from 80 identities, each with a bona fide or attacked video stream and a bona fide or attacked audio stream. The eight attack families are real signal transformations. A clip is labelled an attack if either stream was attacked.

## Splits

Unseen families: video_splice, audio_vocoder. Identity pools never mix.

| split | clips |
| --- | --- |
| train | 751 |
| calib | 234 |
| seen_test | 197 |
| unseen_test | 123 |
| dropped | 295 |

| identity pool | identities |
| --- | --- |
| train | 48 |
| calib | 16 |
| test | 16 |

## Thresholds

Every operating point was chosen on the calibration split at target precision 0.95. Disagreement is the absolute difference of the two stream probabilities.

| detector | score | threshold | calib precision | calib recall | target met |
| --- | --- | --- | --- | --- | --- |
| video | sigmoid(0.7737 logit -0.2284) | 0.0179 | 0.953 | 0.631 | true |
| audio | sigmoid(0.7830 logit -1.7632) | 0.6029 | 0.957 | 0.562 | true |
| fused | 0.05 p_video + 0.95 p_audio | 0.0434 | 0.952 | 1.000 | true |
| logistic | sigmoid(3.252 p_video +2.474 p_audio +3.016 disagreement -2.650) | 0.0869 | 0.952 | 1.000 | true |

## Metrics

Seen attack families:

| detector | precision | recall | F1 | EER | AUC | clips | attacks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video | 0.990 | 0.643 | 0.780 | 0.256 | 0.848 | 197 | 154 |
| audio | 0.969 | 0.604 | 0.744 | 0.292 | 0.785 | 197 | 154 |
| fused | 0.956 | 0.994 | 0.975 | 0.140 | 0.947 | 197 | 154 |
| logistic | 0.963 | 1.000 | 0.981 | 0.023 | 0.998 | 197 | 154 |

Unseen attack families:

| detector | precision | recall | F1 | EER | AUC | clips | attacks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video | 1.000 | 0.475 | 0.644 | 0.387 | 0.705 | 123 | 80 |
| audio | 0.864 | 0.237 | 0.373 | 0.279 | 0.806 | 123 | 80 |
| fused | 0.941 | 0.600 | 0.733 | 0.188 | 0.869 | 123 | 80 |
| logistic | 0.948 | 0.688 | 0.797 | 0.125 | 0.883 | 123 | 80 |

## Robustness

False alarm rate on 86 bona fide test clips, clean and at the heaviest severity of each perturbation.

| perturbation | severity | video | audio | fused | logistic |
| --- | --- | --- | --- | --- | --- |
| clean | none | 0.012 | 0.070 | 0.116 | 0.105 |
| jpeg | quality 15 | 1.000 | 0.070 | 0.884 | 1.000 |
| video_noise | sigma 20 | 0.000 | 0.070 | 0.116 | 0.105 |
| brightness_contrast | drift 0.5 | 1.000 | 0.070 | 0.302 | 0.895 |
| video_dropout | frames 4 | 0.012 | 0.070 | 0.116 | 0.105 |
| audio_noise | snr_db 10 | 0.012 | 1.000 | 1.000 | 1.000 |
| resample | hz 4000 | 0.012 | 1.000 | 1.000 | 1.000 |
| reverb | rt60_s 0.35 | 0.012 | 1.000 | 1.000 | 1.000 |
| audio_dropout | gaps 6 | 0.012 | 0.233 | 0.581 | 0.558 |

## Limitations

* The corpus is synthetic, rendered by the repository's deterministic generator. These metrics describe that corpus, not FaceForensics++, ASVspoof or real capture.
* Clips are 16 frames of 64x64 with 2.0 s of audio at 16000 Hz, far smaller than real forensic inputs.
* One seed and one held out pair (video_splice, audio_vocoder). `spoofline sweep` measures variance over seeds and all 16 held out pairs.
* Thresholds are the lowest that reach the target precision on the calibration split, the most optimistic choice on a finite set.
* On unseen families the weighted fusion precision is 0.941 and the logistic fusion precision is 0.948, against video alone at 1.000.
* Each stream only sees its own modality, so single stream recall is capped by the share of attacks that touch that modality.
* No benign channel variation is in training; `spoofline robustness` shows modest audio noise, resampling or reverb raising the fused false alarm rate sharply.
