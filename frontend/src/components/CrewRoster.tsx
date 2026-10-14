import { CREW } from "./mascots/Mascots";
import type { ClassMeta, Telemetry } from "../lib/types";

interface Props {
  classes: ClassMeta[];
  telemetry: Telemetry | null;
}

/**
 * The class legend, fronted by the crew.
 *
 * Each mascot's mood is driven by live recall for its own class: a character
 * whose defects are being caught looks pleased, one being missed looks
 * alarmed. It is a legend that also functions as a dashboard — you can read
 * which class is failing from across the room, before reading a single number.
 */
export default function CrewRoster({ classes, telemetry }: Props) {
  const perClass = telemetry?.metrics.confusion.per_class ?? [];

  return (
    <section className="panel panel--riveted">
      <div className="panel__header">
        <h2 className="panel__title">Inspection Crew</h2>
        <span className="font-mono text-[10px] text-slate-500">5 target classes · Module 1</span>
      </div>

      <div className="panel__body">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          {CREW.map((member) => {
            const meta = classes[member.classId];
            const stats = perClass[member.classId];
            const recall = stats?.recall ?? 0;
            const support = stats?.support ?? 0;

            // Mood from live recall, but only once there is enough support to
            // mean anything — a single missed frame should not set a character
            // screaming.
            const mood =
              support < 4
                ? "idle"
                : recall < 0.5
                  ? "alarm"
                  : recall < 0.8
                    ? "alert"
                    : member.classId === 0
                      ? "happy"
                      : "idle";

            const failing = support >= 4 && recall < 0.6;

            return (
              <div
                key={member.classId}
                className="flex flex-col items-center rounded-md border p-2 text-center transition-colors"
                style={{
                  borderColor: failing ? "#ef4444" : "#273243",
                  background: failing ? "rgba(239,68,68,0.07)" : "rgba(11,17,25,0.5)",
                }}
              >
                <member.Component size={64} mood={mood as never} />

                <div className="mt-1 font-arcade text-[8px]" style={{ color: member.accent }}>
                  {member.codename}
                </div>
                <div className="font-mono text-[9px] leading-tight text-slate-400">
                  {meta?.label ?? member.role}
                </div>

                <div className="mt-1.5 w-full">
                  <div className="h-1.5 overflow-hidden rounded-sm bg-void-900">
                    <div
                      className="h-full transition-all duration-500"
                      style={{
                        width: `${recall * 100}%`,
                        background: failing ? "#ef4444" : member.accent,
                      }}
                    />
                  </div>
                  <div className="mt-0.5 flex justify-between font-mono text-[8px] text-slate-600">
                    <span>recall</span>
                    <span
                      className="tabular-nums"
                      style={{ color: failing ? "#ef4444" : "#94a3b8" }}
                    >
                      {support ? `${(recall * 100).toFixed(0)}%` : "—"}
                    </span>
                  </div>
                </div>

                {meta && meta.miss_cost > 0 ? (
                  <div className="mt-1 font-mono text-[8px] text-slate-600">
                    miss cost <span className="text-alarm">{meta.miss_cost} HP</span>
                  </div>
                ) : (
                  <div className="mt-1 font-mono text-[8px] text-slate-600">baseline class</div>
                )}
              </div>
            );
          })}
        </div>

        <p className="mt-2 font-mono text-[9px] leading-relaxed text-slate-600">
          Severity is not uniform. A missed thermal hotspot melts a blade within minutes; a missed
          dust clog degrades airflow over weeks. The health bar is weighted accordingly, which is
          why macro-F1 — not accuracy — is the metric that matters here.
        </p>
      </div>
    </section>
  );
}
