import { useEffect, useMemo, useRef } from "react";
import { useTheme, type Palette } from "../lib/theme";
import type { ClassMeta, ConveyorItem, Telemetry } from "../lib/types";

interface Props {
  conveyor: ConveyorItem[];
  telemetry: Telemetry | null;
  classes: ClassMeta[];
}

/**
 * Card geometry. The image area is square because the frames are square;
 * stretching a 224×224 render would misrepresent what the classifier sees.
 */
const CARD_W = 104;
const LABEL_H = 20;
const CARD_H = CARD_W + LABEL_H;
const GAP = 12;
const PITCH = CARD_W + GAP;

/**
 * The inspection belt: a canvas showing the latest frames, newest on the
 * right, each boxed by outcome — green correct, warn wrong class, alarm for a
 * missed defect or a dropped frame. Canvas rather than DOM because ~14 cards
 * redraw every animation frame beside a React tree updating at 25 Hz.
 *
 * Server thumbnails are real synthesised rack imagery. The in-browser copy has
 * no renderer, so the canvas draws a schematic rack from the frame id instead.
 */
export default function InspectionConveyor({ conveyor, telemetry, classes }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const imageCache = useRef(new Map<number, HTMLImageElement>());
  const { palette } = useTheme();

  const stateRef = useRef({ conveyor, telemetry, palette });
  stateRef.current = { conveyor, telemetry, palette };

  const shortNames = useMemo(
    () => Object.fromEntries(classes.map((c) => [c.id, c.short.toLowerCase()])) as Record<number, string>,
    [classes],
  );

  useEffect(() => {
    const cache = imageCache.current;
    for (const item of conveyor) {
      if (!item.thumbnail || cache.has(item.frame_id)) continue;
      const img = new Image();
      img.src = item.thumbnail;
      img.decode?.().catch(() => {});
      cache.set(item.frame_id, img);
    }
    const live = new Set(conveyor.map((i) => i.frame_id));
    for (const key of cache.keys()) if (!live.has(key)) cache.delete(key);
  }, [conveyor]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const frame = (now: number) => {
      const { conveyor: items, palette: p } = stateRef.current;
      const rect = canvas.getBoundingClientRect();
      const W = rect.width;
      const H = rect.height;

      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, W, H);

      const cardY = (H - CARD_H) / 2;
      items.forEach((item, idx) => {
        const fromRight = items.length - 1 - idx;
        const x = W - CARD_W - fromRight * PITCH;
        if (x + CARD_W < 0) return;
        drawCard(ctx, item, x, cardY, p, shortNames, imageCache.current, now);
      });

      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [shortNames]);

  const latest = conveyor[conveyor.length - 1];
  const queue = telemetry?.queue;

  const verdict = !latest
    ? { text: "Waiting for the first frame.", tone: "text-muted" }
    : latest.dropped
      ? { text: `Frame ${latest.frame_id} dropped. No model saw this ${classes[latest.true_class]?.label ?? "rack"}.`, tone: "text-alarm" }
      : latest.outcome === "missed_defect"
        ? { text: `Frame ${latest.frame_id}: ${classes[latest.true_class]?.label} missed, called clean at ${(latest.confidence * 100).toFixed(0)}%.`, tone: "text-alarm" }
        : latest.pred_class === latest.true_class
          ? { text: `Frame ${latest.frame_id}: ${classes[latest.true_class]?.label}, correct at ${(latest.confidence * 100).toFixed(0)}%.`, tone: "text-ink" }
          : { text: `Frame ${latest.frame_id}: ${classes[latest.true_class]?.label} called ${classes[latest.pred_class ?? 0]?.label} at ${(latest.confidence * 100).toFixed(0)}%.`, tone: "text-warn" };

  return (
    <section aria-labelledby="conveyor-title" className="section">
      <div className="section__head">
        <h2 id="conveyor-title" className="section__title">
          Camera feed
        </h2>
        <span className="meta">
          {queue?.arrival_rate_fps ?? 25} fps in, {(queue?.throughput_fps ?? 0).toFixed(1)} fps inspected
        </span>
      </div>
      <canvas ref={canvasRef} className="block h-[140px] w-full" aria-hidden="true" />
      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 text-xs">
        <p className={verdict.tone} aria-live="polite">
          {verdict.text}
        </p>
        {latest && !latest.dropped ? (
          <p className="text-muted">
            waited {(latest.queue_delay_ms ?? 0).toFixed(1)} ms, {(latest.system_time_ms ?? 0).toFixed(1)} ms in total
          </p>
        ) : null}
      </div>
    </section>
  );
}

// ─── canvas drawing ──────────────────────────────────────────────────────────

function drawCard(
  ctx: CanvasRenderingContext2D,
  item: ConveyorItem,
  x: number,
  y: number,
  p: Palette,
  shortNames: Record<number, string>,
  cache: Map<number, HTMLImageElement>,
  now: number,
) {
  const correct = !item.dropped && item.pred_class === item.true_class;
  const bad = item.dropped || item.outcome === "missed_defect";
  const box = bad ? p.alarm : correct ? p.accent : p.warn;

  ctx.save();
  if (item.dropped) ctx.globalAlpha = 0.35;

  const img = cache.get(item.frame_id);
  if (img && img.complete && img.naturalWidth > 0) {
    ctx.drawImage(img, x, y, CARD_W, CARD_W);
  } else {
    drawProceduralRack(ctx, x, y, CARD_W, CARD_W, item.true_class, item.frame_id, now);
  }

  ctx.strokeStyle = box;
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 1, y + 1, CARD_W - 2, CARD_W - 2);

  if (item.outcome === "missed_defect" || item.dropped) {
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + 16, y + 16);
    ctx.lineTo(x + CARD_W - 16, y + CARD_W - 16);
    ctx.moveTo(x + CARD_W - 16, y + 16);
    ctx.lineTo(x + 16, y + CARD_W - 16);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.font = "500 11px 'IBM Plex Mono', monospace";
  ctx.textBaseline = "middle";
  const ly = y + CARD_W + LABEL_H / 2 + 1;
  if (item.dropped) {
    ctx.fillStyle = p.alarm;
    ctx.fillText("dropped", x, ly);
  } else {
    ctx.fillStyle = bad ? p.alarm : correct ? p.ink : p.warn;
    ctx.fillText(shortNames[item.pred_class ?? 0] ?? "?", x, ly);
    ctx.fillStyle = p.muted;
    const conf = `${(item.confidence * 100).toFixed(0)}%`;
    ctx.fillText(conf, x + CARD_W - ctx.measureText(conf).width, ly);
  }
  ctx.restore();
}

/**
 * Client-side stand-in for a server-rendered frame.
 *
 * Deliberately schematic rather than photographic: a chassis, a vent block, an
 * LED column, and the one visual cue that defines the class. It is drawn from
 * the frame id so a given frame always looks the same, which matters because
 * the belt redraws 60 times a second and a frame that shimmered would be
 * unreadable.
 */
function drawProceduralRack(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  classId: number,
  seed: number,
  now: number,
) {
  // Cheap deterministic hash → [0,1)
  const rand = (n: number) => {
    const v = Math.sin(seed * 12.9898 + n * 78.233) * 43758.5453;
    return v - Math.floor(v);
  };

  ctx.fillStyle = "#121824";
  ctx.fillRect(x, y, w, h);

  const blades = 4;
  const bh = (h - 10) / blades;

  for (let i = 0; i < blades; i++) {
    const by = y + 5 + i * bh;
    ctx.fillStyle = "#1c2636";
    ctx.fillRect(x + 6, by, w - 12, bh - 3);

    // Vent mesh
    ctx.fillStyle = "#0e141f";
    ctx.fillRect(x + 12, by + 3, (w - 12) * 0.56, bh - 9);
    ctx.fillStyle = "#273243";
    for (let vy = by + 5; vy < by + bh - 7; vy += 3) {
      for (let vx = x + 14; vx < x + 12 + (w - 12) * 0.56 - 2; vx += 3) {
        ctx.fillRect(vx, vy, 1, 1);
      }
    }

    // Activity LED
    const lit = rand(i) > 0.3;
    ctx.fillStyle = classId === 0 ? (lit ? "#10b981" : "#16432f") : "#10b981";
    ctx.fillRect(x + 8, by + bh / 2 - 2, 3, 3);
  }

  // Class signature
  switch (classId) {
    case 1: {
      // Thermal plume — a radial ironbow blob
      const cx = x + w * (0.3 + rand(1) * 0.4);
      const cy = y + h * (0.3 + rand(2) * 0.4);
      const r = w * 0.3;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, "rgba(255,240,180,0.95)");
      g.addColorStop(0.4, "rgba(255,140,40,0.75)");
      g.addColorStop(1, "rgba(200,30,20,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case 2: {
      // Fiber looped below its minimum bend radius
      const cx = x + w * 0.52;
      const cy = y + h * 0.5;
      const r = w * 0.13;
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x + 10, cy + h * 0.28);
      ctx.lineTo(cx - r, cy + r * 0.6);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI * 0.6, Math.PI * 2.5);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx + r * 0.4, cy - r * 0.6);
      ctx.lineTo(x + w - 10, cy - h * 0.22);
      ctx.stroke();
      break;
    }
    case 3: {
      // Port row with a fault
      const py = y + h * 0.56;
      const bad = Math.floor(rand(3) * 8);
      for (let p = 0; p < 8; p++) {
        const px = x + 12 + p * ((w - 24) / 8);
        ctx.fillStyle = "#0e141f";
        ctx.fillRect(px, py, (w - 24) / 8 - 2, 8);
        const blink = p === bad && Math.floor(now / 260) % 2 === 0;
        ctx.fillStyle = p === bad ? (blink ? "#ef4444" : "#7f1d1d") : "#10b981";
        ctx.fillRect(px + 1, py + 2, 3, 4);
      }
      break;
    }
    case 4: {
      // Particulate occlusion over the intake
      ctx.fillStyle = "rgba(150,140,120,0.55)";
      ctx.fillRect(x + 10, y + 6, (w - 12) * 0.56, h - 14);
      ctx.fillStyle = "rgba(190,180,160,0.7)";
      for (let d = 0; d < 46; d++) {
        const dx = x + 10 + rand(d + 10) * (w - 12) * 0.56;
        const dy = y + 6 + rand(d + 60) * (h - 14);
        ctx.fillRect(dx, dy, 2, 2);
      }
      break;
    }
    default:
      break;
  }

  // Scanline tint, so canvas-drawn and server-rendered cards sit together
  ctx.fillStyle = "rgba(255,255,255,0.03)";
  for (let sy = y; sy < y + h; sy += 3) ctx.fillRect(x, sy, w, 1);
}
