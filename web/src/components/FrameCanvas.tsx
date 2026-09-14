import { useEffect, useRef } from "react";

interface FrameCanvasProps {
  frames: Uint8Array;
  size: number;
  index: number;
  label: string;
  className?: string;
}

/** Draws one frame from the clip buffer, channel 0 as red, scaled with nearest neighbour sampling. */
export function FrameCanvas({ frames, size, index, label, className }: FrameCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const image = ctx.createImageData(size, size);
    const offset = index * size * size * 3;
    for (let p = 0; p < size * size; p++) {
      image.data[p * 4] = frames[offset + p * 3]!;
      image.data[p * 4 + 1] = frames[offset + p * 3 + 1]!;
      image.data[p * 4 + 2] = frames[offset + p * 3 + 2]!;
      image.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [frames, size, index]);
  return <canvas ref={ref} width={size} height={size} className={className ?? "frame"} role="img" aria-label={label} />;
}
