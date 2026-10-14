import { useEffect, useRef } from "react";
import type { Telemetry } from "../lib/types";

interface Props {
  telemetry: Telemetry | null;
}

/**
 * The edge buffer — the one instrument the whole project exists to show.
 *
 * Drawn large, as discrete slots (a buffer holds frames, not a percentage),
 * with the load factor ρ and the closed-form M/D/1 predictions beside the live
 * measurement.
 *
 * Audio: at 8 of 10 slots a short square-wave tone plays, pitched by depth and
 * rate-limited. Synthesised with WebAudio and only armed after a user gesture.
 */
export default function QueueBufferMeter({ telemetry }: Props) {
  const audioRef = useRef<AudioContext | null>(null);
  const lastBeepRef = useRef(0);
  const armedRef = useRef(false);

  useEffect(() => {
    const arm = () => {
      armedRef.current = true;
      window.removeEventListener("pointerdown", arm);
      window.removeEventListener("keydown", arm);
    };
    window.addEventListener("pointerdown", arm);
    window.addEventListener("keydown", arm);
    return () => {
      window.removeEventListener("pointerdown", arm);
      window.removeEventListener("keydown", arm);
      audioRef.current?.close().catch(() => {});
    };
  }, []);

  const queue = telemetry?.queue;
  const theory = telemetry?.theory;
  const service = telemetry?.service;
  const depth = queue?.depth ?? 0;
  const capacity = queue?.capacity ?? 10;
  const critical = depth >= 8;

  useEffect(() => {
    if (!critical || !armedRef.current) return;
    const now = performance.now();
    if (now - lastBeepRef.current < 700) return;
    lastBeepRef.current = now;
    try {
      const ctx = (audioRef.current ??= new AudioContext());
      if (ctx.state === "suspended") void ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = 660 + (depth - 8) * 110;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.05, ctx.currentTime + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.16);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.18);
    } catch {
      /* audio is optional */
    }
  }, [critical, depth]);

  const rho = theory?.rho ?? 0;
  const saturated = rho >= 1;
  const total = service?.total_ms ?? 0;
  const budget = service?.budget_ms ?? 40;
  const headroom = budget - total;

  const slotTone = (i: number) =>
    i >= depth ? "bg-sunk" : i >= 7 ? "bg-alarm" : i >= 5 ? "bg-warn" : "bg-ink";

  const fmt = (v: number | null | undefined, digits = 2) =>
    v === null || v === undefined ? "∞" : v.toFixed(digits);

  return (
    <section aria-labelledby="buffer-title" className="py-6">
      <div className="flex flex-wrap items-end justify-between gap-x-10 gap-y-5">
        <div>
          <h1 id="buffer-title" className="text-sm text-muted">
            Edge buffer
          </h1>
          <div className="mt-2 flex items-center gap-3 sm:gap-4">
            <div
              className="flex items-center gap-[3px] text-3xl font-medium leading-none text-muted sm:gap-[5px] sm:text-4xl"
              role="meter"
              aria-valuemin={0}
              aria-valuemax={capacity}
              aria-valuenow={depth}
              aria-label="Frames waiting in the buffer"
            >
              <span aria-hidden="true">[</span>
              {Array.from({ length: capacity }, (_, i) => (
                <span key={i} className={`h-7 w-3.5 rounded-[2px] sm:h-9 sm:w-5 transition-colors duration-100 ${slotTone(i)}`} />
              ))}
              <span aria-hidden="true">]</span>
            </div>
            <div className="text-3xl font-medium leading-none sm:text-4xl">
              <span className={critical ? "text-alarm" : ""}>{depth}</span>
              <span className="text-muted">/{capacity}</span>
            </div>
          </div>
          <p className={`mt-3 text-sm ${queue?.overflow ? "font-semibold text-alarm" : "text-muted"}`}>
            {queue?.overflow
              ? "Buffer full. New frames are being dropped, so those racks are never inspected."
              : saturated
                ? "Frames arrive faster than the model can process them. The buffer is filling."
                : "The model keeps up with the camera. Frames wait briefly, then clear."}
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm sm:grid-cols-[auto_auto_auto_auto]">
          <dt className="text-muted">load ρ = λ/μ</dt>
          <dd className={`text-right font-semibold ${saturated ? "text-alarm" : "text-accent-ink"}`}>
            {rho.toFixed(3)}
          </dd>
          <dt className="text-muted">service time</dt>
          <dd className={`text-right ${service?.over_budget ? "font-semibold text-alarm" : ""}`}>
            {total.toFixed(1)} / {budget} ms
          </dd>

          <dt className="text-muted">mean time in system</dt>
          <dd className="text-right">{(queue?.mean_system_time_ms ?? 0).toFixed(1)} ms</dd>
          <dt className="text-muted">headroom</dt>
          <dd className={`text-right ${headroom < 0 ? "font-semibold text-alarm" : ""}`}>
            {headroom >= 0 ? "+" : ""}
            {headroom.toFixed(1)} ms
          </dd>

          <dt className="text-muted">p95 time in system</dt>
          <dd className="text-right">{(queue?.p95_system_time_ms ?? 0).toFixed(1)} ms</dd>
          <dt className="text-muted" title="Pollaczek–Khinchine: L_q = ρ²/(2(1−ρ))">
            predicted queue L_q
          </dt>
          <dd className="text-right">
            {fmt(theory?.lq_md1)} <span className="text-muted">(M/M/1 {fmt(theory?.lq_mm1)})</span>
          </dd>

          <dt className="text-muted">SLA breaches</dt>
          <dd className="text-right">{queue?.sla_breaches ?? 0}</dd>
          <dt className="text-muted" title="M/M/1/K blocking probability, an upper bound on loss">
            loss bound
          </dt>
          <dd className="text-right">{((theory?.blocking_probability_bound ?? 0) * 100).toFixed(1)}%</dd>
        </dl>
      </div>

      {/* ρ on a 0–1.5 scale; the tick at 1 is the stability boundary. */}
      <div className="mt-5">
        <div className="track h-1">
          <div
            className={`track__fill ${saturated ? "bg-alarm" : rho > 0.8 ? "bg-warn" : "bg-accent"}`}
            style={{ width: `${Math.min(100, (rho / 1.5) * 100)}%` }}
          />
          <div className="absolute inset-y-0 left-2/3 w-px bg-ink" />
        </div>
        <div className="relative mt-1 h-4 text-xs text-muted">
          <span className="absolute left-0">ρ 0</span>
          <span className="absolute left-2/3 -translate-x-1/2 whitespace-nowrap">1, stability limit</span>
          <span className="absolute right-0">1.5</span>
        </div>
      </div>
    </section>
  );
}
