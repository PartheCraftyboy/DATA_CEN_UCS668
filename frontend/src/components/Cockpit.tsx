import { useCallback, useEffect, useState } from "react";
import CockpitHeader from "./CockpitHeader";
import InspectionConveyor from "./InspectionConveyor";
import ModelSwitcher from "./ModelSwitcher";
import QueueBufferMeter from "./QueueBufferMeter";
import ConfusionMatrix from "./ConfusionMatrix";
import ParetoPlot from "./ParetoPlot";
import AugmentationStorms from "./AugmentationStorms";
import { useTelemetry } from "../lib/useTelemetry";
import type { CockpitEvent, FilterKey, ModelKey, StormKey } from "../lib/types";

/**
 * Root island. One WebSocket, one telemetry object, fanned out to the panels.
 *
 * All the instrument panels are children of this single island rather than
 * seven separately-hydrated islands. That is deliberate: they all read the
 * same 25 Hz stream, so hydrating them separately would mean seven copies of
 * the socket or a cross-island store, and no less JavaScript. The Astro page
 * around this — layout, chrome, the static briefing — still ships zero JS.
 */
export default function Cockpit() {
  const { connection, handshake, telemetry, conveyor, send, reconnect } = useTelemetry();
  const [log, setLog] = useState<(CockpitEvent & { at: number })[]>([]);

  // Rolling event log, newest first.
  useEffect(() => {
    if (!telemetry?.events?.length) return;
    const stamped = telemetry.events.map((e) => ({ ...e, at: Date.now() }));
    setLog((prev) => [...stamped, ...prev].slice(0, 18));
  }, [telemetry]);

  const setModel = useCallback((model: ModelKey) => send({ action: "set_model", model }), [send]);
  const setStorm = useCallback(
    (storm: StormKey, intensity: number) => send({ action: "set_storm", storm, intensity }),
    [send],
  );
  const setFilter = useCallback(
    (filter: FilterKey, enabled: boolean) => send({ action: "set_filter", filter, enabled }),
    [send],
  );
  const reset = useCallback(() => {
    setLog([]);
    send({ action: "reset" });
  }, [send]);
  const pause = useCallback((paused: boolean) => send({ action: "pause", paused }), [send]);

  // Keyboard shortcuts — 1/2/3 switch backbones, space holds, R resets.
  // An arcade cabinet should be playable without the mouse.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

      switch (e.key) {
        case "1":
          setModel("squeezenet");
          break;
        case "2":
          setModel("mobilenet");
          break;
        case "3":
          setModel("resnet");
          break;
        case " ":
          e.preventDefault();
          pause(!(telemetry?.paused ?? false));
          break;
        case "r":
        case "R":
          reset();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setModel, pause, reset, telemetry?.paused]);

  if (!handshake) {
    return (
      <p className="mx-auto max-w-[1440px] px-4 py-16 text-muted sm:px-6" aria-live="polite">
        Connecting to the edge node<span className="animate-pulse">_</span>
      </p>
    );
  }

  const activeModel = (telemetry?.model.key ?? handshake.default_model) as ModelKey;

  return (
    <>
      <CockpitHeader
        telemetry={telemetry}
        connection={connection}
        onReset={reset}
        onPause={pause}
        onReconnect={reconnect}
      />

      <main className="mx-auto max-w-[1440px] px-4 sm:px-6">
        <QueueBufferMeter telemetry={telemetry} />

        <InspectionConveyor conveyor={conveyor} telemetry={telemetry} classes={handshake.classes} />

        <div className="grid gap-x-10 lg:grid-cols-2 xl:grid-cols-3">
          <ModelSwitcher models={handshake.models} active={activeModel} telemetry={telemetry} onSelect={setModel} />
          <ConfusionMatrix telemetry={telemetry} classes={handshake.classes} />
          <AugmentationStorms
            storms={handshake.augmentations.storms}
            filters={handshake.augmentations.filters}
            telemetry={telemetry}
            onStorm={setStorm}
            onFilter={setFilter}
          />
        </div>

        <div className="grid gap-x-10 lg:grid-cols-[2fr_1fr]">
          <ParetoPlot models={handshake.models} telemetry={telemetry} />

          <section aria-labelledby="log-title" className="section">
            <div className="section__head">
              <h2 id="log-title" className="section__title">
                Incidents
              </h2>
              {log.length ? <span className="meta">latest {log.length}</span> : null}
            </div>
            {log.length === 0 ? (
              <p className="text-xs text-muted">No dropped frames or missed defects yet.</p>
            ) : (
              <ol className="max-h-[300px] space-y-1 overflow-y-auto text-xs">
                {log.map((e, i) => (
                  <li key={`${e.frame_id}-${e.type}-${i}`} className="flex gap-3">
                    <time className="shrink-0 text-muted">
                      {new Date(e.at).toLocaleTimeString("en-GB", { hour12: false })}
                    </time>
                    <span className={e.type === "drop" ? "text-alarm" : "text-warn"}>
                      {e.type === "drop" ? `dropped ${e.label.replace(/_/g, " ").toLowerCase()}` : `missed ${e.label.toLowerCase()}`}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>
      </main>
    </>
  );
}
