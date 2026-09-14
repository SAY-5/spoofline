export type Stream = "video" | "audio";
export type Detector = Stream | "fused";
export type TestSplit = "seen_test" | "unseen_test";
export type Combo = "bonafide" | "video_only" | "audio_only" | "both";

export interface OperatingJson {
  threshold: number;
  target_precision: number;
  achieved_precision: number;
  recall: number;
  reached_target: boolean;
}

export interface StreamCalibrationJson {
  stream: Stream;
  calibrator: { a: number; b: number };
  operating: OperatingJson;
}

export interface CalibrationJson {
  video: StreamCalibrationJson;
  audio: StreamCalibrationJson;
  fused: { weight: number; operating: OperatingJson };
}

export interface ClipEntry {
  id: string;
  identity: string;
  video_family: string;
  audio_family: string;
  label: 0 | 1;
  combo: Combo;
  split: TestSplit;
  touches_unseen: boolean;
  roles: string[];
}

export interface CorpusShape {
  n_frames: number;
  frame_size: number;
  sample_rate: number;
  n_samples: number;
  n_clips: number;
  n_identities: number;
}

export interface Manifest {
  trained_from_commit: string;
  seed: number;
  profile: string;
  target_precision: number;
  unseen_families: string[];
  families: Record<string, string>;
  corpus: CorpusShape;
  features: {
    audio_scale: number;
    n_fft: number;
    hop_length: number;
    n_mels: number;
    patch_width: number;
    window: string;
    center_pad: string;
    log_epsilon: number;
    fbank_shape: [number, number];
    logmel_check_clip: string;
    logmel_check_shape: [number, number];
  };
  models: Record<Stream, { path: string; input: string; output: string; shape: number[] }>;
  calibration: CalibrationJson;
  gallery: { identity: string; clips: Record<string, string> };
  clips: ClipEntry[];
}

export interface ReferenceClip {
  video_logit: number;
  audio_logit: number;
  video_probability: number;
  audio_probability: number;
  fused_probability: number;
  video_flags: boolean;
  audio_flags: boolean;
  decision: "attack" | "bonafide";
}

export interface MetricPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  eer: number;
  auc: number;
  counts: { tp: number; fp: number; tn: number; fn: number };
  n: number;
  n_positive: number;
}

export interface RuleMetrics {
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface RunJson {
  calibration: CalibrationJson;
  metrics: Record<TestSplit, Record<Detector, MetricPoint>>;
  rules: Record<TestSplit, Record<"and" | "or" | "weighted", RuleMetrics>>;
  family_rates: Record<TestSplit, Record<string, { n: number; detected: number; rate: number }>>;
  headline: {
    fused_precision: number;
    video_precision: number;
    audio_precision: number;
    fused_recall: number;
    video_recall: number;
    audio_recall: number;
    verdict: string;
  };
  splits: { counts: Record<string, number>; identity_pools: Record<string, number> };
  combo_counts: Record<string, number>;
  family_counts: Record<string, number>;
  corpus: Record<string, number | string>;
  training: Record<
    Stream,
    {
      epochs: number;
      best_epoch: number;
      train_clips: number;
      val_clips: number;
      final: { train_loss: number; train_acc: number; val_loss: number; val_acc: number };
    }
  >;
  target_precision: number;
  unseen_families: string[];
  seed: number;
}

export interface Reference {
  trained_from_commit: string;
  clips: Record<string, ReferenceClip>;
  run: RunJson;
}

export interface ScoreRows {
  clip_ids: string[];
  label: number[];
  video_attacked: number[];
  audio_attacked: number[];
  video_logit: number[];
  audio_logit: number[];
}

export interface TestScores extends ScoreRows {
  split: TestSplit[];
  families: string[][];
}
