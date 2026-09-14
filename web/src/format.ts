export const FAMILY_ORDER = [
  "bonafide",
  "video_replay",
  "video_print",
  "video_splice",
  "video_recompress",
  "audio_replay",
  "audio_vocoder",
  "audio_conversion",
  "audio_splice",
] as const;

export const FAMILY_LABEL: Record<string, string> = {
  bonafide: "Bona fide",
  video_replay: "Screen replay",
  video_print: "Print",
  video_splice: "Face splice",
  video_recompress: "Recompression",
  audio_replay: "Room replay",
  audio_vocoder: "Vocoder",
  audio_conversion: "Voice conversion",
  audio_splice: "Speaker splice",
};

export const FAMILY_PLAIN: Record<string, string> = {
  bonafide: "The untouched render: a synthetic talking head and its speech-like voice.",
  video_replay:
    "The face shown on a screen and filmed again: moire, scrolling refresh bands, a gamma shift, a bezel crop and a slight tilt.",
  video_print:
    "A printed photo held up to the camera: halftone dots, paper grain and tint, with head motion flattened because paper has no depth.",
  video_splice:
    "The face region swapped in from another identity, alpha blended, with a seam that jitters from frame to frame.",
  video_recompress: "Two harsh JPEG round trips, then 8x8 block quantisation of the DC term.",
  audio_replay:
    "The voice played through a loudspeaker in a room and recorded again: echo, band limiting, device noise and mains hum.",
  audio_vocoder:
    "The voice reduced to a mel spectrogram and rebuilt with Griffin-Lim, which has to invent the phase.",
  audio_conversion:
    "Pitch and formants shifted by resampling, then stretched back to length with a phase vocoder.",
  audio_splice: "Segments from a second speaker cut in with hard joins and mismatched levels.",
};

export function streamOf(family: string): "video" | "audio" | null {
  if (family.startsWith("video_")) return "video";
  if (family.startsWith("audio_")) return "audio";
  return null;
}

export function fixed(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

export function signed(value: number, digits = 3): string {
  const text = value.toFixed(digits);
  return value >= 0 ? `+${text}` : text;
}

export function gap(value: number): string {
  if (value === 0) return "0";
  return value.toExponential(1).replace("e-", "e-").replace("e+", "e+");
}

export function ms(value: number): string {
  return value < 10 ? value.toFixed(1) : value.toFixed(0);
}
