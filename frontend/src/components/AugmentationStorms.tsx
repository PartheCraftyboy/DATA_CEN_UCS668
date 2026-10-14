import { useOptimistic, useState, useTransition } from "react";
import type { FilterKey, FilterSpec, StormKey, StormSpec, Telemetry } from "../lib/types";

interface Props {
  storms: StormSpec[];
  filters: FilterSpec[];
  telemetry: Telemetry | null;
  onStorm: (key: StormKey, intensity: number) => void;
  onFilter: (key: FilterKey, enabled: boolean) => void;
}

/**
 * Conditions (storms the camera suffers) against filters (preprocessing the
 * operator deploys). Filters bill their cost to the same per-frame budget as
 * the model, and the panel keeps that trade visible.
 */
export default function AugmentationStorms({ storms, filters, telemetry, onStorm, onFilter }: Props) {
  const [, startTransition] = useTransition();
  const [detail, setDetail] = useState<string | null>(null);

  const serverStorms = telemetry?.storms.storms ?? ({} as Record<StormKey, number>);
  const serverFilters = telemetry?.storms.filters ?? [];
  const [optStorms, setOptStorms] = useOptimistic(serverStorms);
  const [optFilters, setOptFilters] = useOptimistic(serverFilters);

  const setStorm = (key: StormKey, intensity: number) =>
    startTransition(() => {
      setOptStorms({ ...optStorms, [key]: intensity });
      onStorm(key, intensity);
    });

  const toggleFilter = (key: FilterKey) => {
    const enabled = !optFilters.includes(key);
    startTransition(() => {
      setOptFilters(enabled ? [...optFilters, key] : optFilters.filter((f) => f !== key));
      onFilter(key, enabled);
    });
  };

  const severity = telemetry?.severity ?? 0;
  const headroom = (telemetry?.service.budget_ms ?? 40) - (telemetry?.service.total_ms ?? 0);
  const activeStorms = storms.filter((s) => (optStorms[s.key] ?? 0) > 0);

  return (
    <section aria-labelledby="conditions-title" className="section">
      <div className="section__head">
        <h2 id="conditions-title" className="section__title">
          Conditions
        </h2>
        <span className={`meta ${severity > 0.5 ? "text-alarm" : ""}`}>
          image degradation {(severity * 100).toFixed(0)}%
        </span>
      </div>

      <ul className="space-y-3">
        {storms.map((storm) => {
          const intensity = optStorms[storm.key] ?? 0;
          const on = intensity > 0;
          const open = detail === storm.key;
          return (
            <li key={storm.key}>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  role="switch"
                  aria-checked={on}
                  onClick={() => setStorm(storm.key, on ? 0 : 0.75)}
                  className="flex flex-1 items-baseline gap-3 text-left"
                >
                  <span className={on ? "text-warn" : "text-muted"} aria-hidden="true">
                    {on ? "[x]" : "[ ]"}
                  </span>
                  <span className={on ? "font-semibold" : ""}>{storm.display_name}</span>
                </button>
                {on ? (
                  <input
                    type="range"
                    min={0.05}
                    max={1}
                    step={0.05}
                    value={intensity}
                    onChange={(e) => setStorm(storm.key, Number(e.target.value))}
                    className="w-24"
                    aria-label={`${storm.display_name} intensity`}
                  />
                ) : null}
                <span className="w-9 text-right text-xs text-muted">{on ? `${(intensity * 100).toFixed(0)}%` : "off"}</span>
              </div>
              <div className="ml-9 mt-0.5 text-xs text-muted">
                {storm.blurb}{" "}
                <button
                  type="button"
                  className="underline decoration-rule underline-offset-2 hover:text-ink"
                  aria-expanded={open}
                  onClick={() => setDetail(open ? null : storm.key)}
                >
                  {open ? "less" : "physics"}
                </button>
              </div>
              {open ? (
                <div className="ml-9 mt-1.5 max-w-[62ch] space-y-1 text-xs leading-relaxed text-muted">
                  <code className="block text-ink">{storm.formula}</code>
                  <p>{storm.physics}</p>
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>

      <h3 className="mb-2 mt-6 flex items-baseline justify-between text-sm font-semibold">
        Filters
        <span className="text-xs font-normal text-muted">each one adds time per frame</span>
      </h3>
      <ul className="space-y-1.5">
        {filters.map((f) => {
          const on = optFilters.includes(f.key);
          const countered = activeStorms
            .map((s) => ({ s, k: f.counters[s.key] ?? 0 }))
            .filter((x) => x.k > 0)
            .sort((a, b) => b.k - a.k);
          return (
            <li key={f.key}>
              <button
                type="button"
                role="switch"
                aria-checked={on}
                onClick={() => toggleFilter(f.key)}
                title={`${f.formula}. ${f.tradeoff}`}
                className="flex w-full items-baseline gap-3 text-left"
              >
                <span className={on ? "text-accent-ink" : "text-muted"} aria-hidden="true">
                  {on ? "[x]" : "[ ]"}
                </span>
                <span className={`flex-1 ${on ? "font-semibold" : ""}`}>{f.display_name}</span>
                <span className="text-xs text-muted">+{f.nominal_cost_ms.toFixed(1)} ms</span>
              </button>
              {on ? (
                <p className="ml-9 text-xs text-muted">
                  {countered.length
                    ? `Reduces ${countered.map((x) => `${x.s.short.toLowerCase()} by ${(x.k * 100).toFixed(0)}%`).join(", ")}.`
                    : "No active condition to counter. This is costing time for nothing."}
                </p>
              ) : null}
            </li>
          );
        })}
      </ul>

      <p className={`mt-4 text-xs ${headroom < 0 ? "font-semibold text-alarm" : "text-muted"}`}>
        {headroom < 0
          ? `Over budget by ${(-headroom).toFixed(1)} ms. The buffer will fill and frames will be dropped.`
          : `${headroom.toFixed(1)} ms of the frame budget left.`}
      </p>
    </section>
  );
}
