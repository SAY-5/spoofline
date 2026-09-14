let context: AudioContext | null = null;

export interface Playback {
  context: AudioContext;
  startedAt: number;
  duration: number;
  stop: () => void;
}

/** Plays int16 samples through WebAudio at their own sample rate. */
export function playSamples(samples: Int16Array, sampleRate: number, onEnded: () => void): Playback {
  context ??= new AudioContext();
  void context.resume();
  const buffer = context.createBuffer(1, samples.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) channel[i] = samples[i]! / 32767;
  const source = context.createBufferSource();
  source.buffer = buffer;
  const gain = context.createGain();
  gain.gain.value = 0.8;
  source.connect(gain).connect(context.destination);
  source.onended = onEnded;
  const startedAt = context.currentTime + 0.03;
  source.start(startedAt);
  return {
    context,
    startedAt,
    duration: samples.length / sampleRate,
    stop: () => {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // already stopped
      }
    },
  };
}
