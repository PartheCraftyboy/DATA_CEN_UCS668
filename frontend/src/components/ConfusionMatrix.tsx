import { useState } from "react";
import type { ClassMeta, Telemetry } from "../lib/types";

interface Props {
  telemetry: Telemetry | null;
  classes: ClassMeta[];
}

/**
 * Live 5×5 confusion matrix, row-normalised so the diagonal reads as recall.
 *
 * Colour encodes the kind of cell: green on the diagonal (correct), alarm in
 * column 0 below the diagonal (a real defect called clean — the expensive
 * miss), warn elsewhere (wrong defect, or a false alarm). Opacity encodes the
 * share.
 */
export default function ConfusionMatrix({ telemetry, classes }: Props) {
  const [selected, setSelected] = useState<{ row: number; col: number } | null>(null);
  const confusion = telemetry?.metrics.confusion;

  const head = (
    <div className="section__head">
      <h2 id="matrix-title" className="section__title">
        Confusion matrix
      </h2>
      {confusion ? (
        <span className="meta">last {Math.min(confusion.total, confusion.window)} frames</span>
      ) : null}
    </div>
  );

  if (!confusion) {
    return (
      <section aria-labelledby="matrix-title" className="section">
        {head}
        <p className="text-muted">Waiting for the first classified frames.</p>
      </section>
    );
  }

  const { normalised, counts, per_class, macro_f1, accuracy, false_negative_rate } = confusion;

  const tone = (row: number, col: number) =>
    row === col ? "--accent" : col === 0 && row !== 0 ? "--alarm" : "--warn";

  const cellStyle = (row: number, col: number, v: number) => {
    const alpha = v <= 0 ? 0 : 0.12 + Math.min(1, v) * 0.88;
    return {
      background: v > 0 ? `rgb(var(${tone(row, col)}) / ${alpha})` : undefined,
      color: alpha > 0.6 ? "rgb(var(--bg))" : undefined,
    };
  };

  const inspect = selected
    ? {
        rowMeta: classes[selected.row],
        colMeta: classes[selected.col],
        value: normalised[selected.row][selected.col],
        count: counts[selected.row][selected.col],
        rowTotal: counts[selected.row].reduce((a, b) => a + b, 0),
        stats: per_class[selected.row],
        diagonal: selected.row === selected.col,
        falseNegative: selected.col === 0 && selected.row !== 0,
      }
    : null;

  return (
    <section aria-labelledby="matrix-title" className="section">
      {head}

      <dl className="mb-4 grid grid-cols-3 gap-4">
        <div>
          <dt className="text-xs text-muted">accuracy</dt>
          <dd className="text-xl">{(accuracy * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-xs text-muted">macro F1</dt>
          <dd className="text-xl">{macro_f1.toFixed(3)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted">defects missed</dt>
          <dd className={`text-xl ${false_negative_rate > 0.15 ? "text-alarm" : ""}`}>
            {(false_negative_rate * 100).toFixed(1)}%
          </dd>
        </div>
      </dl>

      <div className="overflow-x-auto">
        <table className="w-full table-fixed border-separate border-spacing-[3px] text-xs">
          <caption className="sr-only">Rows are the true class, columns the predicted class.</caption>
          <thead>
            <tr>
              <th className="w-[4.5rem] text-left font-normal text-muted">true ↓</th>
              {classes.map((c) => (
                <th key={c.id} scope="col" className="pb-1 text-center font-normal text-muted" title={`Predicted ${c.label}`}>
                  {c.short.toLowerCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classes.map((r) => (
              <tr key={r.id}>
                <th scope="row" className="pr-1 text-left font-normal text-muted" title={r.label}>
                  {r.short.toLowerCase()}
                </th>
                {classes.map((c) => {
                  const v = normalised[r.id][c.id];
                  const isSel = selected?.row === r.id && selected?.col === c.id;
                  return (
                    <td key={c.id} className="p-0">
                      <button
                        type="button"
                        onClick={() => setSelected(isSel ? null : { row: r.id, col: c.id })}
                        aria-pressed={isSel}
                        title={`${r.label} called ${c.label}: ${counts[r.id][c.id]} frames`}
                        className={`h-10 w-full rounded-[2px] bg-sunk text-center transition-colors ${
                          isSel ? "outline outline-2 outline-offset-1 outline-ink" : ""
                        }`}
                        style={cellStyle(r.id, c.id, v)}
                      >
                        {v > 0 ? v.toFixed(2).replace(/^0/, "") : <span className="text-muted">·</span>}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 min-h-[4.5rem] text-xs">
        {inspect ? (
          <div className="space-y-1">
            <p className={`font-semibold ${inspect.diagonal ? "text-accent-ink" : inspect.falseNegative ? "text-alarm" : "text-warn"}`}>
              {inspect.diagonal
                ? `${inspect.rowMeta.label}: correct`
                : inspect.falseNegative
                  ? `${inspect.rowMeta.label} called clean (missed defect)`
                  : `${inspect.rowMeta.label} called ${inspect.colMeta.label}`}
            </p>
            <p className="max-w-[62ch] text-muted">
              {inspect.count} of {inspect.rowTotal} frames ({(inspect.value * 100).toFixed(1)}%).
              {inspect.falseNegative
                ? ` No technician is sent to these racks. Each miss costs ${inspect.rowMeta.miss_cost} health.`
                : ""}
            </p>
            <p className="text-muted">
              precision <span className="text-ink">{inspect.stats.precision.toFixed(3)}</span>
              {"   "}recall <span className="text-ink">{inspect.stats.recall.toFixed(3)}</span>
              {"   "}F1 <span className="text-ink">{inspect.stats.f1.toFixed(3)}</span>
            </p>
          </div>
        ) : (
          <p className="text-muted">Select a cell to see what it counts.</p>
        )}
      </div>

      <table className="mt-2 w-full text-xs">
        <caption className="sr-only">Recall per class</caption>
        <thead>
          <tr className="text-muted">
            <th className="pb-1 text-left font-normal">class</th>
            <th className="pb-1 text-left font-normal">recall</th>
            <th className="pb-1 text-right font-normal">frames</th>
            <th className="pb-1 text-right font-normal">miss cost</th>
          </tr>
        </thead>
        <tbody>
          {per_class.map((m) => {
            const meta = classes[m.class_id];
            const weak = m.support >= 4 && m.class_id !== 0 && m.recall < 0.8;
            return (
              <tr key={m.class_id} className="border-t border-rule">
                <td className="py-1 pr-3">{meta.label}</td>
                <td className="w-1/2 py-1">
                  <div className="flex items-center gap-2">
                    <div className="track flex-1">
                      <div
                        className={`track__fill ${weak ? "bg-alarm" : "bg-accent"}`}
                        style={{ width: `${m.recall * 100}%` }}
                      />
                    </div>
                    <span className={`w-9 text-right ${weak ? "text-alarm" : ""}`}>{(m.recall * 100).toFixed(0)}%</span>
                  </div>
                </td>
                <td className="py-1 text-right text-muted">{m.support}</td>
                <td className="py-1 text-right text-muted">{meta.miss_cost || "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
