import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useTheme } from "../lib/theme";
import type { ModelCard, Telemetry } from "../lib/types";

interface Props {
  models: ModelCard[];
  telemetry: Telemetry | null;
}

type View = "tradeoff" | "trend" | "calibration";

const VIEWS: [View, string][] = [
  ["tradeoff", "speed vs accuracy"],
  ["trend", "trend"],
  ["calibration", "calibration"],
];

const FONT = "'IBM Plex Mono', monospace";

/**
 * Three views over the same run:
 *   speed vs accuracy  each backbone's latency against accuracy, with the
 *                      frame budget as a hard vertical line.
 *   trend              cross-entropy and accuracy over the run.
 *   calibration        the reliability diagram; points below the diagonal
 *                      are overconfidence, and the average gap is ECE.
 */
export default function ParetoPlot({ models, telemetry }: Props) {
  const [view, setView] = useState<View>("tradeoff");
  const { palette: p } = useTheme();

  const budget = telemetry?.service.budget_ms ?? 40;
  const live = telemetry?.metrics.pareto.live ?? [];
  const calibration = telemetry?.metrics.calibration;
  const curves = telemetry?.metrics.curves;

  const axis = { fill: p.muted, fontSize: 11, fontFamily: FONT };
  const tooltip = {
    contentStyle: { background: p.bg, border: `1px solid ${p.rule}`, borderRadius: 3, fontFamily: FONT, fontSize: 12 },
    labelStyle: { color: p.muted },
    itemStyle: { color: p.ink },
  };

  const specPoints = useMemo(
    () =>
      models.map((m) => ({
        x: m.nominal_latency_ms,
        y: m.clean_accuracy * 100,
        z: m.power_watts,
        name: m.display_name,
        meets: m.nominal_latency_ms < budget,
      })),
    [models, budget],
  );

  const livePoints = useMemo(
    () =>
      live
        .filter((pt) => pt.frames >= 10)
        .map((pt) => ({
          x: pt.latency_ms,
          y: pt.accuracy * 100,
          z: pt.power_watts,
          name: `${models.find((m) => m.key === pt.model_key)?.display_name ?? pt.model_key} (this run)`,
        })),
    [live, models],
  );

  const trendData = useMemo(() => {
    const loss = curves?.loss ?? [];
    const acc = curves?.accuracy ?? [];
    return Array.from({ length: Math.max(loss.length, acc.length) }, (_, i) => ({
      i,
      loss: loss[i] ?? null,
      accuracy: (acc[i] ?? 0) * 100,
    }));
  }, [curves]);

  const calData = useMemo(
    () =>
      (calibration?.bins ?? [])
        .filter((b) => b.count > 0)
        .map((b) => ({
          confidence: ((b.lo + b.hi) / 2) * 100,
          accuracy: b.accuracy * 100,
          ideal: ((b.lo + b.hi) / 2) * 100,
        })),
    [calibration],
  );

  return (
    <section aria-labelledby="analysis-title" className="section">
      <div className="section__head flex-wrap">
        <h2 id="analysis-title" className="section__title">
          Analysis
        </h2>
        <div className="flex gap-1.5" role="group" aria-label="Chart">
          {VIEWS.map(([key, label]) => (
            <button key={key} type="button" className="btn" aria-pressed={view === key} onClick={() => setView(key)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {view === "tradeoff" ? (
        <>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 0 }}>
                <CartesianGrid stroke={p.rule} strokeDasharray="2 4" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="time per frame"
                  unit=" ms"
                  domain={[0, Math.max(60, budget * 1.6)]}
                  tick={axis}
                  stroke={p.rule}
                  label={{ value: "time per frame (ms)", position: "insideBottom", offset: -14, fill: p.muted, fontSize: 11, fontFamily: FONT }}
                />
                <YAxis type="number" dataKey="y" name="accuracy" unit="%" domain={[60, 100]} tick={axis} stroke={p.rule} width={48} />
                <ZAxis type="number" dataKey="z" range={[60, 360]} name="power" unit=" W" />
                <ReferenceLine
                  x={budget}
                  stroke={p.alarm}
                  strokeDasharray="4 3"
                  label={{ value: `${budget} ms budget`, fill: p.alarm, fontSize: 11, fontFamily: FONT, position: "top" }}
                />
                <Tooltip cursor={{ stroke: p.rule }} {...tooltip} formatter={(v: number, n: string) => [typeof v === "number" ? v.toFixed(2) : v, n]} />
                <Scatter name="reference" data={specPoints} isAnimationActive={false}>
                  {specPoints.map((pt) => (
                    <Cell key={pt.name} fill={p.bg} stroke={pt.meets ? p.ink : p.alarm} strokeWidth={1.5} />
                  ))}
                  <LabelList dataKey="name" position="right" offset={10} style={{ fill: p.ink, stroke: "none", fontSize: 11, fontFamily: FONT }} />
                </Scatter>
                <Scatter name="this run" data={livePoints} fill={p.accent} shape="diamond" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 max-w-[70ch] text-xs leading-relaxed text-muted">
            Circles are reference figures, sized by power draw. Green diamonds are this run, and they move right as
            filters are added. Anything right of the budget line cannot keep up with a 25 fps camera, however
            accurate it is.
          </p>
        </>
      ) : null}

      {view === "trend" ? (
        <>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData} margin={{ top: 16, right: 4, bottom: 20, left: 0 }}>
                <CartesianGrid stroke={p.rule} strokeDasharray="2 4" />
                <XAxis dataKey="i" tick={axis} stroke={p.rule} label={{ value: "samples (every 5 frames)", position: "insideBottom", offset: -12, fill: p.muted, fontSize: 11, fontFamily: FONT }} />
                <YAxis yAxisId="loss" tick={axis} stroke={p.rule} width={40} domain={[0, "auto"]} />
                <YAxis yAxisId="acc" orientation="right" tick={axis} stroke={p.rule} width={44} domain={[0, 100]} unit="%" />
                <Tooltip {...tooltip} />
                <Line yAxisId="loss" type="monotone" dataKey="loss" name="cross-entropy" stroke={p.warn} strokeWidth={1.75} dot={false} isAnimationActive={false} />
                <Line yAxisId="acc" type="monotone" dataKey="accuracy" name="accuracy %" stroke={p.ink} strokeWidth={1.75} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 max-w-[70ch] text-xs leading-relaxed text-muted">
            <span className="text-warn">Cross-entropy</span> (left axis) and <span className="text-ink">accuracy</span>{" "}
            (right). Start a condition and the loss rises before accuracy falls: the model becomes less sure before
            it becomes wrong.
          </p>
        </>
      ) : null}

      {view === "calibration" ? (
        <>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={calData} margin={{ top: 16, right: 4, bottom: 20, left: 0 }}>
                <CartesianGrid stroke={p.rule} strokeDasharray="2 4" />
                <XAxis dataKey="confidence" type="number" domain={[0, 100]} tick={axis} stroke={p.rule} unit="%" label={{ value: "stated confidence", position: "insideBottom", offset: -12, fill: p.muted, fontSize: 11, fontFamily: FONT }} />
                <YAxis domain={[0, 100]} tick={axis} stroke={p.rule} width={44} unit="%" />
                <Tooltip {...tooltip} />
                <Line type="linear" dataKey="ideal" name="perfect" stroke={p.muted} strokeDasharray="4 4" strokeWidth={1} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="accuracy" name="actual" stroke={p.accent} strokeWidth={2} dot={{ r: 3, fill: p.accent, stroke: p.accent }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <dl className="mt-2 grid grid-cols-3 gap-4 text-sm">
            <div>
              <dt className="text-xs text-muted">ECE</dt>
              <dd>{(calibration?.ece ?? 0).toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">worst bin</dt>
              <dd>{(calibration?.mce ?? 0).toFixed(3)}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">bias</dt>
              <dd className={(calibration?.calibration_bias ?? 0) < 0 ? "text-alarm" : ""}>
                {(calibration?.calibration_bias ?? 0).toFixed(3)}
              </dd>
            </div>
          </dl>
          <p className="mt-2 max-w-[70ch] text-xs leading-relaxed text-muted">
            Points below the dashed line mean the model claims more confidence than it earns. ResNet-18 is the most
            accurate model here and the worst calibrated, so its confidence can't be used to decide which frames
            need a second look.
          </p>
        </>
      ) : null}
    </section>
  );
}
