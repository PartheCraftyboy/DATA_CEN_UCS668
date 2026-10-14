import { useEffect, useRef, useState } from "react";
import { PALETTES, THEME_ORDER, setTheme, useTheme } from "../lib/theme";
import type { ConnectionState, Telemetry } from "../lib/types";

interface Props {
  telemetry: Telemetry | null;
  connection: ConnectionState;
  onReset: () => void;
  onPause: (paused: boolean) => void;
  onReconnect: () => void;
}

const CONNECTION: Record<ConnectionState, { label: string; note: string; tone: string }> = {
  connecting: { label: "connecting", note: "Reaching the backend on :8000", tone: "text-warn" },
  live: { label: "live", note: "Backend connected; latencies measured with PyTorch", tone: "text-accent-ink" },
  simulated: { label: "simulated", note: "No backend found; running the in-browser copy", tone: "text-muted" },
  offline: { label: "reconnecting", note: "Backend dropped; in-browser copy running meanwhile", tone: "text-alarm" },
};

const Chevron = () => (
  <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
    <path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
  </svg>
);

const Check = () => (
  <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
    <path d="M2.5 8.5 6.2 12 13.5 4.5" fill="none" stroke="currentColor" strokeWidth="2.4" />
  </svg>
);

const Prompt = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
    <path d="M2.5 4 6.5 8l-4 4" fill="none" stroke="currentColor" strokeWidth="2" />
    <path d="M8 12.5h5.5" stroke="currentColor" strokeWidth="2" />
  </svg>
);

/** The three-dot pill: one swatch per theme, the active one ringed. */
function ThemePill() {
  const { name } = useTheme();
  return (
    <div
      role="radiogroup"
      aria-label="Theme"
      className="flex items-center gap-2 rounded-full border border-rule bg-[#1e201b] px-1.5 py-1"
    >
      {THEME_ORDER.map((t) => {
        const active = t === name;
        return (
          <button
            key={t}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={`${t} theme`}
            title={`${t} theme`}
            onClick={() => setTheme(t)}
            className="h-[18px] w-[18px] rounded-full transition-transform hover:scale-110"
            style={{
              background: PALETTES[t].swatch,
              boxShadow: active ? `0 0 0 1.5px #1e201b, 0 0 0 3px ${PALETTES[t].swatch}` : "none",
            }}
          />
        );
      })}
    </div>
  );
}

/**
 * The status bar. Reads left to right like a breadcrumb: the app, the active
 * backbone, then the vitals that say whether the run is going well.
 */
export default function CockpitHeader({ telemetry, connection, onReset, onPause, onReconnect }: Props) {
  const conn = CONNECTION[connection];
  const queue = telemetry?.queue;
  const paused = telemetry?.paused ?? false;

  // Browser render rate: answers "is the page keeping up?", which the server
  // cannot know.
  const [uiFps, setUiFps] = useState(0);
  const frames = useRef(0);
  const lastT = useRef(performance.now());
  useEffect(() => {
    frames.current += 1;
    const now = performance.now();
    const elapsed = now - lastT.current;
    if (elapsed >= 1000) {
      setUiFps(Math.round((frames.current * 1000) / elapsed));
      frames.current = 0;
      lastT.current = now;
    }
  }, [telemetry]);

  const health = queue?.health ?? 100;
  const drops = queue?.drops_total ?? 0;

  return (
    <header className="sticky top-0 z-20 border-b-2 border-ink bg-bg">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 sm:px-6">
        <nav aria-label="Location" className="flex min-w-0 items-center gap-2.5 text-base">
          <Prompt />
          <span>edge-vision</span>
          <Chevron />
          <Check />
          <span className="truncate">{telemetry?.model.display_name ?? "…"}</span>
        </nav>

        <dl className="flex flex-wrap items-baseline gap-x-5 gap-y-1 text-xs">
          <div className="flex gap-1.5">
            <dt className="text-muted">health</dt>
            <dd className={health < 30 ? "font-semibold text-alarm" : ""}>{health.toFixed(0)}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-muted">score</dt>
            <dd>{Math.round(queue?.score ?? 0).toLocaleString()}</dd>
          </div>
          <div className="flex gap-1.5">
            <dt className="text-muted">dropped</dt>
            <dd className={drops > 0 ? "font-semibold text-alarm" : ""}>{drops}</dd>
          </div>
          <div className="flex gap-1.5" title={`Browser rendering at ${uiFps} fps`}>
            <dt className="text-muted">throughput</dt>
            <dd>{(queue?.throughput_fps ?? 0).toFixed(1)} fps</dd>
          </div>
        </dl>

        <div className="ml-auto flex items-center gap-3">
          <button
            type="button"
            onClick={connection === "live" ? undefined : onReconnect}
            title={connection === "live" ? conn.note : `${conn.note}. Click to retry.`}
            className={`flex items-center gap-1.5 text-xs ${conn.tone}`}
          >
            <span aria-hidden="true">{connection === "live" ? "●" : "○"}</span>
            {conn.label}
          </button>
          <button type="button" className="btn" onClick={() => onPause(!paused)} aria-pressed={paused}>
            {paused ? "resume" : "hold"} <kbd className="border-0 px-0 text-muted">space</kbd>
          </button>
          <button type="button" className="btn" onClick={onReset}>
            reset <kbd className="border-0 px-0 text-muted">r</kbd>
          </button>
          <ThemePill />
        </div>
      </div>
    </header>
  );
}
