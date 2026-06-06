import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Before/after compare slider. Drag the handle (or move the pointer) to wipe
 * between the original and the upscaled output. Both images are drawn at the
 * same box size; `after` is clipped to the handle position.
 */
export function ImageCompareSlider({
  before,
  after,
  className,
  beforeLabel = "Input",
  afterLabel = "Upscaled",
}: {
  before: string;
  after: string;
  className?: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const [pos, setPos] = React.useState(50);
  const ref = React.useRef<HTMLDivElement>(null);
  const dragging = React.useRef(false);

  const move = React.useCallback((clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
    setPos((x / rect.width) * 100);
  }, []);

  React.useEffect(() => {
    const onMove = (e: PointerEvent) => dragging.current && move(e.clientX);
    const onUp = () => (dragging.current = false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [move]);

  return (
    <div
      ref={ref}
      className={cn(
        "relative select-none overflow-hidden rounded-lg border bg-[repeating-conic-gradient(#1f1f23_0%_25%,#17171a_0%_50%)] bg-[length:24px_24px]",
        className
      )}
      onPointerDown={(e) => {
        dragging.current = true;
        move(e.clientX);
      }}
    >
      <img src={before} alt={beforeLabel} className="block h-full w-full object-contain" draggable={false} />
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <img
          src={after}
          alt={afterLabel}
          className="absolute inset-0 h-full w-full object-contain"
          style={{ width: ref.current?.clientWidth ?? "100%" }}
          draggable={false}
        />
        <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white">
          {afterLabel}
        </span>
      </div>
      <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white">
        {beforeLabel}
      </span>
      <div
        className="absolute inset-y-0 z-10 w-0.5 -translate-x-1/2 bg-white/90 shadow"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 left-1/2 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/80 bg-black/60 text-white">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m9 18-6-6 6-6" /><path d="m15 6 6 6-6 6" />
          </svg>
        </div>
      </div>
    </div>
  );
}
