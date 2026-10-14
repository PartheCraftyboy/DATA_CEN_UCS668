import { useOptimistic, useState, useTransition } from "react";
import type { ModelCard, ModelKey, Telemetry } from "../lib/types";

interface Props {
  models: ModelCard[];
  active: ModelKey;
  telemetry: Telemetry | null;
  onSelect: (key: ModelKey) => void;
}

/**
 * Backbone selector. `useOptimistic` flips the selection immediately; the next
 * telemetry tick (≤40 ms) reconciles it with the model the server is running.
 */
export default function ModelSwitcher({ models, active, telemetry, onSelect }: Props) {
  const [, startTransition] = useTransition();
  const [optimisticKey, setOptimisticKey] = useOptimistic(active);
  const [expanded, setExpanded] = useState<ModelKey | null>(null);

  const select = (key: ModelKey) => {
    startTransition(() => {
      setOptimisticKey(key);
      onSelect(key);
    });
  };

  const budget = telemetry?.service.budget_ms ?? 40;
  const service = telemetry?.service;
  const scale = Math.max(service?.total_ms ?? 0, budget) * 1.15;

  return (
    <section aria-labelledby="models-title" className="section">
      <div className="section__head">
        <h2 id="models-title" className="section__title">
          Model
        </h2>
        <span className="meta">press 1, 2 or 3</span>
      </div>

      <ul className="space-y-1">
        {models.map((model, idx) => {
          const isActive = optimisticKey === model.key;
          const over = model.nominal_latency_ms >= budget;
          const isOpen = expanded === model.key;
          const a = model.architecture;

          return (
            <li key={model.key}>
              <button
                type="button"
                className="row"
                onClick={() => select(model.key)}
                aria-pressed={isActive}
              >
                <div className="flex items-baseline gap-3">
                  <span className={`w-3 ${isActive ? "text-accent-ink" : "text-muted"}`} aria-hidden="true">
                    {isActive ? "●" : idx + 1}
                  </span>
                  <span className={`flex-1 ${isActive ? "font-semibold" : ""}`}>{model.display_name}</span>
                  <span className={over ? "font-semibold text-alarm" : ""}>
                    {model.nominal_latency_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="ml-6 mt-1.5 flex items-center gap-3">
                  <div className="track flex-1">
                    <div
                      className={`track__fill ${over ? "bg-alarm" : isActive ? "bg-accent" : "bg-muted/60"}`}
                      style={{ width: `${Math.min(100, (model.nominal_latency_ms / (budget * 1.15)) * 100)}%` }}
                    />
                    <div className="absolute inset-y-0 w-px bg-ink" style={{ left: `${100 / 1.15}%` }} />
                  </div>
                  <span className={`w-28 text-right text-xs ${over ? "text-alarm" : "text-muted"}`}>
                    {over ? "over budget" : `${(budget - model.nominal_latency_ms).toFixed(1)} ms spare`}
                  </span>
                </div>
                <div className="ml-6 mt-1 text-xs text-muted">
                  {(model.clean_accuracy * 100).toFixed(1)}% accurate, {a.params_millions.toFixed(2)}M params,{" "}
                  {model.power_watts.toFixed(1)} W
                </div>
              </button>

              <button
                type="button"
                className="ml-6 mt-0.5 text-xs text-muted hover:text-ink"
                aria-expanded={isOpen}
                onClick={() => setExpanded(isOpen ? null : model.key)}
              >
                {isOpen ? "− hide architecture" : "+ architecture"}
              </button>

              {isOpen ? (
                <div className="mb-2 ml-6 mt-2 space-y-2 border-l border-rule pl-3 text-xs">
                  <p className="max-w-[62ch] leading-relaxed text-muted">{model.architecture_note}</p>
                  <code className="block text-ink">{model.signature_block}</code>
                  <p className="text-muted">
                    resolution {a.resolution_cascade.join(" → ")}
                  </p>
                  <dl className="grid max-w-sm grid-cols-2 gap-x-4">
                    <dt className="text-muted">MACs</dt>
                    <dd>{a.macs_millions.toFixed(0)} M</dd>
                    <dt className="text-muted">GFLOPs</dt>
                    <dd>{a.gflops.toFixed(2)}</dd>
                    <dt className="text-muted">weights</dt>
                    <dd>{a.weight_mb.toFixed(1)} MB</dd>
                    <dt className="text-muted">capacity</dt>
                    <dd>{(model.capacity * 100).toFixed(0)}%</dd>
                    <dt className="text-muted">measured here</dt>
                    <dd>
                      {model.measured_latency.samples
                        ? `${model.measured_latency.median_ms.toFixed(1)} ms (p95 ${model.measured_latency.p95_ms.toFixed(1)})`
                        : "not measured"}
                    </dd>
                  </dl>
                  {model.measured_latency.samples ? (
                    <p className="max-w-[62ch] leading-relaxed text-muted">
                      The queue uses the {model.nominal_latency_ms.toFixed(1)} ms reference figure so every
                      machine produces the same simulation.
                    </p>
                  ) : null}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>

      {service ? (
        <div className="mt-4">
          <div className="mb-1.5 flex justify-between text-xs">
            <span className="text-muted">time per frame</span>
            <span>
              {service.inference_ms.toFixed(1)} model + {service.preprocess_ms.toFixed(1)} filters ={" "}
              <span className={service.over_budget ? "font-semibold text-alarm" : "font-semibold"}>
                {service.total_ms.toFixed(1)} ms
              </span>
            </span>
          </div>
          <div className="track h-2">
            <div className="track__fill bg-ink" style={{ width: `${(service.inference_ms / scale) * 100}%` }} />
            <div
              className="track__fill bg-warn"
              style={{
                left: `${(service.inference_ms / scale) * 100}%`,
                width: `${(service.preprocess_ms / scale) * 100}%`,
              }}
            />
            <div className="absolute inset-y-0 w-0.5 bg-alarm" style={{ left: `${(budget / scale) * 100}%` }} />
          </div>
          <div className="mt-1 text-right text-xs text-muted">{budget} ms budget</div>
        </div>
      ) : null}
    </section>
  );
}
