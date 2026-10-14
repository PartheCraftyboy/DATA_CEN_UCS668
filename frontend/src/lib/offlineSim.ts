/**
 * OFFLINE SIMULATOR — the cockpit's fallback when no backend is reachable.
 *
 * Why this exists: the cockpit is a lab demo, and lab demos are given on
 * borrowed machines, on projectors, after someone's venv broke. If the FastAPI
 * process is not running, the honest options are a dead screen or a faithful
 * client-side mirror of the simulation. This is the mirror.
 *
 * It reimplements, in TypeScript, exactly the parts of the backend that drive
 * the UI — the M/D/1 queue, the capacity-limited decision model, the running
 * confusion matrix and ECE — using the same constants. It does NOT reimplement
 * PyTorch: there is no forward pass here, so the "measured latency" readout
 * reports the reference figures rather than this machine's, and the cockpit
 * labels the connection SIMULATED rather than LIVE so the distinction is never
 * hidden from the viewer.
 *
 * Everything the queueing argument depends on is reproduced exactly, so the
 * ResNet-18 buffer overflow demonstration works identically offline.
 */

import type {
  CalibrationBin,
  ClassId,
  ConveyorItem,
  CockpitEvent,
  FilterKey,
  Handshake,
  ModelKey,
  Outcome,
  StormKey,
  Telemetry,
} from "./types";

// ─── Constants, mirrored from the backend ────────────────────────────────────

const NUM_CLASSES = 5;
const TICK_HZ = 25;
const TICK_DT = 1 / TICK_HZ;
const ARRIVAL_FPS = 25;
const BUFFER_CAPACITY = 10;
const FRAME_BUDGET_MS = 40;
const MAX_HEALTH = 100;
const SLA_LATENCY_MS = 80;
const CONVEYOR_DEPTH = 14;
const WINDOW = 300;
const ECE_BINS = 10;

const CLASS_NAMES = [
  "Optimal_Rack",
  "Thermal_Hotspot",
  "Cable_Bend_Violation",
  "Port_LED_Error",
  "Intake_Dust_Clogged",
];

const CLASS_META = [
  { id: 0, key: "Optimal_Rack", label: "Optimal Rack", short: "OPTIMAL", severity: 0, miss_cost: 0, accent: "#10b981", blurb: "Nominal thermals, green activity LEDs, unobstructed intake." },
  { id: 1, key: "Thermal_Hotspot", label: "Thermal Hotspot", short: "THERMAL", severity: 1, miss_cost: 9, accent: "#ef4444", blurb: "PSU/blade over temperature. Infrared false-colour plume." },
  { id: 2, key: "Cable_Bend_Violation", label: "Cable Bend Violation", short: "BEND", severity: 0.75, miss_cost: 6, accent: "#a855f7", blurb: "Fiber patch cord below the 30 mm critical bend radius." },
  { id: 3, key: "Port_LED_Error", label: "Port LED Error", short: "PORT", severity: 0.6, miss_cost: 4, accent: "#f59e0b", blurb: "Switch port signalling amber/red link-down or CRC fault." },
  { id: 4, key: "Intake_Dust_Clogged", label: "Intake Dust Clogged", short: "DUST", severity: 0.5, miss_cost: 3, accent: "#94a3b8", blurb: "Perforated chassis intake occluded by particulate debris." },
] as const;

const MODELS = {
  squeezenet: { key: "squeezenet", display_name: "SqueezeNet v1.1", family: "Fire Module", nominal_latency_ms: 4.5, clean_accuracy: 0.885, capacity: 0.34, power_watts: 1.9, confidence_temperature: 0.78, accent: "#10b981", params_m: 0.73, gflops: 0.526, macs_m: 263.1 },
  mobilenet: { key: "mobilenet", display_name: "MobileNetV2", family: "Inverted Residual", nominal_latency_ms: 11.5, clean_accuracy: 0.951, capacity: 0.72, power_watts: 3.4, confidence_temperature: 1.05, accent: "#06b6d4", params_m: 2.23, gflops: 0.599, macs_m: 299.5 },
  resnet: { key: "resnet", display_name: "ResNet-18", family: "Residual Stage", nominal_latency_ms: 44.0, clean_accuracy: 0.976, capacity: 0.94, power_watts: 7.8, confidence_temperature: 1.45, accent: "#a855f7", params_m: 11.18, gflops: 3.627, macs_m: 1813.6 },
} as const;

const STORM_WEIGHT: Record<StormKey, number> = {
  heat_haze: 0.46,
  strobe_glare: 0.38,
  intake_dust: 0.34,
};

const FILTER_COST: Record<FilterKey, number> = {
  clahe: 1.8,
  unsharp: 1.1,
  bilateral: 4.6,
};

const FILTER_COUNTERS: Record<FilterKey, Record<StormKey, number>> = {
  clahe: { strobe_glare: 0.62, heat_haze: 0.12, intake_dust: 0.18 },
  unsharp: { heat_haze: 0.48, strobe_glare: 0.1, intake_dust: 0.08 },
  bilateral: { intake_dust: 0.66, heat_haze: 0.2, strobe_glare: 0.14 },
};

const CONFUSABILITY: number[][] = [
  [0.0, 0.08, 0.38, 0.14, 0.4],
  [0.34, 0.0, 0.08, 0.46, 0.12],
  [0.62, 0.05, 0.0, 0.19, 0.14],
  [0.3, 0.44, 0.12, 0.0, 0.14],
  [0.58, 0.07, 0.17, 0.18, 0.0],
];

const STORM_FRAGILITY = [0.22, 0.46, 0.78, 0.55, 0.4];
const CLASS_PRIOR = [0.55, 0.11, 0.12, 0.12, 0.1];

// ─── Small numerics ──────────────────────────────────────────────────────────

/** Knuth's method: adequate for λ·dt ≈ 1, which is all we ever ask of it. */
function poisson(lambda: number): number {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= Math.random();
  } while (p > L);
  return k - 1;
}

function sampleCategorical(probs: number[]): number {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))));
  return sorted[idx];
}

// ─── Queue ticket ────────────────────────────────────────────────────────────

interface Ticket {
  id: number;
  arrivedAt: number;
  trueClass: ClassId;
  startedAt: number | null;
  completedAt: number | null;
}

// ─── The simulator ───────────────────────────────────────────────────────────

export class OfflineSimulator {
  modelKey: ModelKey = "mobilenet";
  paused = false;
  storms: Record<StormKey, number> = { heat_haze: 0, strobe_glare: 0, intake_dust: 0 };
  filters = new Set<FilterKey>();

  private clock = 0;
  private frameCounter = 0;
  private buffer: Ticket[] = [];
  private inService: Ticket | null = null;
  private serviceRemaining = 0;

  private health = MAX_HEALTH;
  private score = 0;
  private dropsTotal = 0;
  private dropsRecent = 0;
  private processedTotal = 0;
  private slaBreaches = 0;
  private missedDefects = 0;
  private falseAlarms = 0;
  private gameOver = false;

  private systemTimes: number[] = [];
  private counts: number[][] = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
  private cumulative: number[][] = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
  private history: [number, number][] = [];
  private calSamples: { conf: number; correct: boolean }[] = [];
  private ceSamples: number[] = [];
  private framesByModel: Record<string, number> = {};
  private correctByModel: Record<string, number> = {};
  private lossCurve: number[] = [];
  private accCurve: number[] = [];
  private curveTick = 0;
  private conveyor: ConveyorItem[] = [];
  private conveyorDelta: ConveyorItem[] = [];
  private paretoTrails: Record<string, { latency_ms: number; accuracy: number }[]> = {};

  // ── configuration ──────────────────────────────────────────────────────────

  get serviceTimeMs(): number {
    let cost = MODELS[this.modelKey].nominal_latency_ms;
    this.filters.forEach((f) => (cost += FILTER_COST[f]));
    return cost;
  }

  get filterCostMs(): number {
    let cost = 0;
    this.filters.forEach((f) => (cost += FILTER_COST[f]));
    return cost;
  }

  /**
   * Effective severity: raw storm severity after the deployed kernels have
   * mitigated what they can. Combination is multiplicative on both sides, so
   * severity stays in [0,1) however many storms stack, and no stack of filters
   * can drive it to zero.
   */
  severity(): { raw: number; effective: number } {
    let rawSurvival = 1;
    let mitigatedSurvival = 1;

    (Object.keys(this.storms) as StormKey[]).forEach((key) => {
      const intensity = Math.min(1, Math.max(0, this.storms[key]));
      if (intensity <= 0) return;
      rawSurvival *= 1 - STORM_WEIGHT[key] * intensity;

      let mitigation = 1;
      this.filters.forEach((f) => {
        mitigation *= 1 - (FILTER_COUNTERS[f][key] ?? 0);
      });
      mitigatedSurvival *= 1 - STORM_WEIGHT[key] * intensity * mitigation;
    });

    return { raw: 1 - rawSurvival, effective: 1 - mitigatedSurvival };
  }

  reset(): void {
    this.clock = 0;
    this.frameCounter = 0;
    this.buffer = [];
    this.inService = null;
    this.serviceRemaining = 0;
    this.health = MAX_HEALTH;
    this.score = 0;
    this.dropsTotal = 0;
    this.dropsRecent = 0;
    this.processedTotal = 0;
    this.slaBreaches = 0;
    this.missedDefects = 0;
    this.falseAlarms = 0;
    this.gameOver = false;
    this.systemTimes = [];
    this.counts = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
    this.cumulative = Array.from({ length: NUM_CLASSES }, () => new Array(NUM_CLASSES).fill(0));
    this.history = [];
    this.calSamples = [];
    this.ceSamples = [];
    this.framesByModel = {};
    this.correctByModel = {};
    this.lossCurve = [];
    this.accCurve = [];
    this.conveyor = [];
    this.conveyorDelta = [];
    this.paretoTrails = {};
    this.storms = { heat_haze: 0, strobe_glare: 0, intake_dust: 0 };
    this.filters.clear();
  }

  // ── decision model ─────────────────────────────────────────────────────────

  private posterior(trueClass: number, severity: number): number[] {
    const spec = MODELS[this.modelKey];
    const headroom = 1 - spec.capacity;
    let recall = spec.clean_accuracy * (1 - severity * STORM_FRAGILITY[trueClass] * headroom);
    recall = Math.min(0.995, Math.max(0.2, recall));

    const row = CONFUSABILITY[trueClass];
    const total = row.reduce((a, b) => a + b, 0) || 1;
    const out = row.map((v) => (v / total) * (1 - recall));
    out[trueClass] = recall;
    return out;
  }

  private decide(trueClass: number, severity: number): { pred: number; probs: number[] } {
    const posterior = this.posterior(trueClass, severity);
    const pred = sampleCategorical(posterior);

    const t = MODELS[this.modelKey].confidence_temperature;
    let reported = posterior.slice();
    reported[pred] = Math.max(reported[pred], Math.max(...posterior));
    let sum = reported.reduce((a, b) => a + b, 0);
    reported = reported.map((v) => v / sum);

    let sharp = reported.map((v) => Math.pow(Math.max(v, 1e-9), t));
    sum = sharp.reduce((a, b) => a + b, 0);
    sharp = sharp.map((v) => v / sum);

    // Light per-frame jitter so successive gauges do not report identical
    // confidences, which reads as fake.
    sharp = sharp.map((v) => v * (0.94 + Math.random() * 0.12));
    sum = sharp.reduce((a, b) => a + b, 0);
    sharp = sharp.map((v) => v / sum);

    return { pred, probs: sharp };
  }

  // ── stepping ───────────────────────────────────────────────────────────────

  step(): Telemetry {
    if (this.paused) return this.telemetry([]);

    const events: CockpitEvent[] = [];
    const serviceMs = this.serviceTimeMs;
    const serviceS = serviceMs / 1000;
    const { effective: severity } = this.severity();

    this.clock += TICK_DT;

    // 1. Poisson arrivals into a finite buffer.
    const nArrivals = poisson(ARRIVAL_FPS * TICK_DT);
    for (let i = 0; i < nArrivals; i++) {
      const trueClass = sampleCategorical(CLASS_PRIOR) as ClassId;
      const arrivedAt = this.clock - TICK_DT + ((i + 1) / (nArrivals + 1)) * TICK_DT;
      const ticket: Ticket = { id: this.frameCounter++, arrivedAt, trueClass, startedAt: null, completedAt: null };

      const occupancy = this.buffer.length + (this.inService ? 1 : 0);
      if (occupancy >= BUFFER_CAPACITY) {
        this.dropsTotal++;
        this.dropsRecent++;
        this.health = Math.max(0, this.health - 2.6);
        this.score = Math.max(0, this.score - 6);
        if (this.health <= 0) this.gameOver = true;
        events.push({
          type: "drop",
          frame_id: ticket.id,
          true_class: trueClass,
          label: CLASS_NAMES[trueClass],
          message: "BUFFER OVERFLOW — rack not inspected",
        });
        this.pushConveyor({
          frame_id: ticket.id,
          true_class: trueClass,
          pred_class: null,
          dropped: true,
          thumbnail: null,
          confidence: 0,
          outcome: "dropped",
          model_key: this.modelKey,
        });
      } else {
        this.buffer.push(ticket);
      }
    }

    // 2. Deterministic single-server service.
    let remaining = TICK_DT;
    let guard = 0;
    while (remaining > 1e-9 && guard++ < 512) {
      if (!this.inService) {
        if (!this.buffer.length) break;
        this.inService = this.buffer.shift()!;
        // Service cannot begin before the frame arrived (arrivals are spread
        // across the slice), or queueing delay comes out negative.
        this.inService.startedAt = Math.max(this.clock - remaining, this.inService.arrivedAt);
        this.serviceRemaining = serviceS;
      }
      const stepS = Math.min(remaining, this.serviceRemaining);
      this.serviceRemaining -= stepS;
      remaining -= stepS;

      if (this.serviceRemaining <= 1e-9) {
        const ticket = this.inService;
        ticket.completedAt = Math.max(this.clock - remaining, ticket.startedAt ?? 0);
        this.inService = null;
        this.processedTotal++;

        const systemMs = (ticket.completedAt - ticket.arrivedAt) * 1000;
        this.systemTimes.push(systemMs);
        if (this.systemTimes.length > 240) this.systemTimes.shift();
        if (systemMs > SLA_LATENCY_MS) this.slaBreaches++;

        const { pred, probs } = this.decide(ticket.trueClass, severity);
        const confidence = probs[pred];
        const ce = -Math.log(Math.max(probs[ticket.trueClass], 1e-9));
        const outcome = this.scoreOutcome(ticket.trueClass, pred as ClassId);
        this.observe(ticket.trueClass, pred, confidence, ce);

        if (outcome === "missed_defect") {
          events.push({
            type: "missed_defect",
            frame_id: ticket.id,
            true_class: ticket.trueClass,
            label: CLASS_META[ticket.trueClass].label,
            message: `FALSE NEGATIVE — ${CLASS_META[ticket.trueClass].label} missed`,
          });
        }

        this.pushConveyor({
          frame_id: ticket.id,
          true_class: ticket.trueClass,
          pred_class: pred as ClassId,
          dropped: false,
          thumbnail: null,
          confidence: Number(confidence.toFixed(4)),
          probs: probs.map((p) => Number(p.toFixed(4))),
          outcome,
          model_key: this.modelKey,
          system_time_ms: Number(systemMs.toFixed(2)),
          queue_delay_ms: Number(((ticket.startedAt! - ticket.arrivedAt) * 1000).toFixed(2)),
          severity: Number(severity.toFixed(3)),
        });
      }
    }

    // 3. Health regeneration.
    if (this.health > 0) this.health = Math.min(MAX_HEALTH, this.health + 1.4 * TICK_DT);
    if (this.health <= 0) this.gameOver = true;

    // 4. Pareto trail.
    const acc = this.accuracy();
    const trail = (this.paretoTrails[this.modelKey] ??= []);
    const frames = this.framesByModel[this.modelKey] ?? 0;
    if (frames >= 25) {
      const point = { latency_ms: Number(serviceMs.toFixed(3)), accuracy: Number(acc.toFixed(4)) };
      const last = trail[trail.length - 1];
      if (!last || last.latency_ms !== point.latency_ms || last.accuracy !== point.accuracy) {
        trail.push(point);
        if (trail.length > 60) trail.shift();
      }
    }

    return this.telemetry(events);
  }

  private scoreOutcome(trueClass: ClassId, pred: ClassId): Outcome {
    const meta = CLASS_META[trueClass];
    const trueIsDefect = trueClass !== 0;
    const predIsDefect = pred !== 0;

    if (pred === trueClass) {
      if (trueIsDefect) {
        this.score += 12 * (0.5 + meta.severity);
        return "detected";
      }
      this.score += 1;
      return "nominal";
    }
    if (trueIsDefect && !predIsDefect) {
      this.missedDefects++;
      this.health = Math.max(0, this.health - meta.miss_cost);
      this.score = Math.max(0, this.score - 8);
      if (this.health <= 0) this.gameOver = true;
      return "missed_defect";
    }
    if (!trueIsDefect && predIsDefect) {
      this.falseAlarms++;
      this.health = Math.max(0, this.health - 0.35);
      this.score = Math.max(0, this.score - 2);
      return "false_alarm";
    }
    this.score = Math.max(0, this.score - 1);
    return "misclassified_defect";
  }

  private observe(trueClass: number, pred: number, confidence: number, ce: number): void {
    if (this.history.length >= WINDOW) {
      const [ot, op] = this.history.shift()!;
      this.counts[ot][op] -= 1;
    }
    this.history.push([trueClass, pred]);
    this.counts[trueClass][pred] += 1;
    this.cumulative[trueClass][pred] += 1;

    this.calSamples.push({ conf: confidence, correct: pred === trueClass });
    if (this.calSamples.length > WINDOW) this.calSamples.shift();
    this.ceSamples.push(ce);
    if (this.ceSamples.length > WINDOW) this.ceSamples.shift();

    this.framesByModel[this.modelKey] = (this.framesByModel[this.modelKey] ?? 0) + 1;
    if (pred === trueClass) {
      this.correctByModel[this.modelKey] = (this.correctByModel[this.modelKey] ?? 0) + 1;
    }

    if (++this.curveTick >= 5) {
      this.curveTick = 0;
      this.lossCurve.push(Number(this.meanCE().toFixed(4)));
      this.accCurve.push(Number(this.accuracy().toFixed(4)));
      if (this.lossCurve.length > 160) this.lossCurve.shift();
      if (this.accCurve.length > 160) this.accCurve.shift();
    }
  }

  private pushConveyor(item: ConveyorItem): void {
    this.conveyor.push(item);
    this.conveyorDelta.push(item);
    if (this.conveyor.length > CONVEYOR_DEPTH) this.conveyor.shift();
  }

  // ── metrics ────────────────────────────────────────────────────────────────

  private total(): number {
    return this.counts.reduce((a, row) => a + row.reduce((x, y) => x + y, 0), 0);
  }

  private accuracy(): number {
    const t = this.total();
    if (!t) return 0;
    let diag = 0;
    for (let i = 0; i < NUM_CLASSES; i++) diag += this.counts[i][i];
    return diag / t;
  }

  private meanCE(): number {
    if (!this.ceSamples.length) return 0;
    return this.ceSamples.reduce((a, b) => a + b, 0) / this.ceSamples.length;
  }

  private perClass() {
    return CLASS_META.map((meta, c) => {
      const tp = this.counts[c][c];
      const rowSum = this.counts[c].reduce((a, b) => a + b, 0);
      let colSum = 0;
      for (let i = 0; i < NUM_CLASSES; i++) colSum += this.counts[i][c];
      const fn = rowSum - tp;
      const fp = colSum - tp;
      const precision = tp + fp ? tp / (tp + fp) : 0;
      const recall = tp + fn ? tp / (tp + fn) : 0;
      const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;

      let topConfusion: { class_id: number; class_name: string; count: number; share: number } | null = null;
      let best = 0;
      for (let j = 0; j < NUM_CLASSES; j++) {
        if (j !== c && this.counts[c][j] > best) {
          best = this.counts[c][j];
          topConfusion = {
            class_id: j,
            class_name: CLASS_NAMES[j],
            count: best,
            share: Number((best / Math.max(rowSum, 1)).toFixed(4)),
          };
        }
      }

      return {
        class_id: c as ClassId,
        class_name: CLASS_NAMES[c],
        label: meta.label,
        accent: meta.accent,
        support: rowSum,
        tp,
        fp,
        fn,
        tn: this.total() - tp - fp - fn,
        precision: Number(precision.toFixed(4)),
        recall: Number(recall.toFixed(4)),
        f1: Number(f1.toFixed(4)),
        top_confusion: topConfusion,
      };
    });
  }

  private calibration() {
    const n = this.calSamples.length;
    const bins: CalibrationBin[] = [];
    let ece = 0;
    let mce = 0;

    for (let i = 0; i < ECE_BINS; i++) {
      const lo = i / ECE_BINS;
      const hi = (i + 1) / ECE_BINS;
      const inBin = this.calSamples.filter((s) => (i === 0 ? s.conf >= lo : s.conf > lo) && s.conf <= hi);
      const count = inBin.length;
      const accuracy = count ? inBin.filter((s) => s.correct).length / count : 0;
      const confidence = count ? inBin.reduce((a, s) => a + s.conf, 0) / count : 0;
      if (count) {
        const gap = Math.abs(accuracy - confidence);
        ece += (count / n) * gap;
        mce = Math.max(mce, gap);
      }
      bins.push({
        lo: Number(lo.toFixed(3)),
        hi: Number(hi.toFixed(3)),
        count,
        accuracy: Number(accuracy.toFixed(4)),
        confidence: Number(confidence.toFixed(4)),
        gap: Number((accuracy - confidence).toFixed(4)),
      });
    }

    const meanConf = n ? this.calSamples.reduce((a, s) => a + s.conf, 0) / n : 0;
    const meanAcc = n ? this.calSamples.filter((s) => s.correct).length / n : 0;

    return {
      ece: Number(ece.toFixed(5)),
      mce: Number(mce.toFixed(5)),
      bins,
      mean_cross_entropy: Number(this.meanCE().toFixed(5)),
      mean_confidence: Number(meanConf.toFixed(5)),
      samples: n,
      calibration_bias: Number((meanAcc - meanConf).toFixed(5)),
    };
  }

  private theory(serviceMs: number) {
    const mu = 1000 / serviceMs;
    const rho = ARRIVAL_FPS / mu;
    const k = BUFFER_CAPACITY;
    const stable = rho < 1;
    const lqMd1 = stable ? (rho * rho) / (2 * (1 - rho)) : null;
    const lqMm1 = stable ? (rho * rho) / (1 - rho) : null;
    const wqMs = lqMd1 !== null ? (lqMd1 / ARRIVAL_FPS) * 1000 : null;
    const pBlock =
      Math.abs(rho - 1) < 1e-9
        ? 1 / (k + 1)
        : ((1 - rho) * Math.pow(rho, k)) / (1 - Math.pow(rho, k + 1));

    return {
      rho: Number(rho.toFixed(4)),
      stable,
      service_rate_fps: Number(mu.toFixed(2)),
      arrival_rate_fps: ARRIVAL_FPS,
      lq_md1: lqMd1 === null ? null : Number(lqMd1.toFixed(3)),
      lq_mm1: lqMm1 === null ? null : Number(lqMm1.toFixed(3)),
      wq_ms: wqMs === null ? null : Number(wqMs.toFixed(2)),
      expected_system_ms: wqMs === null ? null : Number((wqMs + serviceMs).toFixed(2)),
      blocking_probability_bound: Number(Math.min(1, Math.max(0, pBlock)).toFixed(5)),
      buffer_capacity: k,
      frame_budget_ms: FRAME_BUDGET_MS,
      budget_headroom_ms: Number((FRAME_BUDGET_MS - serviceMs).toFixed(3)),
      note: "L_q for M/D/1 is exactly half the M/M/1 value — deterministic service halves the backlog at identical ρ.",
    };
  }

  // ── payload ────────────────────────────────────────────────────────────────

  private telemetry(events: CockpitEvent[]): Telemetry {
    const serviceMs = this.serviceTimeMs;
    const spec = MODELS[this.modelKey];
    const { effective: severity } = this.severity();

    const sorted = this.systemTimes.slice().sort((a, b) => a - b);
    const meanSt = sorted.length ? sorted.reduce((a, b) => a + b, 0) / sorted.length : 0;

    const perClass = this.perClass();
    const supported = perClass.filter((m) => m.support > 0);
    const macroF1 = supported.length ? supported.reduce((a, m) => a + m.f1, 0) / supported.length : 0;

    let defectTotal = 0;
    let defectMissed = 0;
    for (let i = 1; i < NUM_CLASSES; i++) {
      for (let j = 0; j < NUM_CLASSES; j++) defectTotal += this.counts[i][j];
      defectMissed += this.counts[i][0];
    }

    const delta = this.conveyorDelta;
    this.conveyorDelta = [];
    const dropsRecent = this.dropsRecent;
    this.dropsRecent = 0;

    const depth = this.buffer.length + (this.inService ? 1 : 0);
    const acc = this.accuracy();

    return {
      type: "telemetry",
      t: Date.now() / 1000,
      paused: this.paused,
      model: {
        key: this.modelKey,
        display_name: spec.display_name,
        accent: spec.accent,
        family: spec.family,
        nominal_latency_ms: spec.nominal_latency_ms,
        // No forward pass offline — the reference figure stands in, and the
        // cockpit badges the connection SIMULATED so this is never mistaken
        // for a measurement from this machine.
        measured_latency_ms: spec.nominal_latency_ms,
        power_watts: spec.power_watts,
        capacity: spec.capacity,
      },
      service: {
        total_ms: Number(serviceMs.toFixed(3)),
        inference_ms: spec.nominal_latency_ms,
        preprocess_ms: Number(this.filterCostMs.toFixed(3)),
        preprocess_measured_ms: 0,
        budget_ms: FRAME_BUDGET_MS,
        over_budget: serviceMs > FRAME_BUDGET_MS,
      },
      queue: {
        depth,
        capacity: BUFFER_CAPACITY,
        fill_ratio: Number((depth / BUFFER_CAPACITY).toFixed(4)),
        utilisation: Number((ARRIVAL_FPS / (1000 / serviceMs)).toFixed(4)),
        service_time_ms: Number(serviceMs.toFixed(3)),
        throughput_fps: Number(Math.min(ARRIVAL_FPS, 1000 / serviceMs).toFixed(2)),
        arrival_rate_fps: ARRIVAL_FPS,
        drops_total: this.dropsTotal,
        drops_recent: dropsRecent,
        processed_total: this.processedTotal,
        health: Number(this.health.toFixed(2)),
        score: Number(this.score.toFixed(1)),
        mean_system_time_ms: Number(meanSt.toFixed(2)),
        p95_system_time_ms: Number(percentile(sorted, 95).toFixed(2)),
        sla_breaches: this.slaBreaches,
        overflow: depth >= BUFFER_CAPACITY,
        saturated: serviceMs * ARRIVAL_FPS >= 1000,
        elapsed_s: Number(this.clock.toFixed(2)),
      },
      theory: this.theory(serviceMs),
      metrics: {
        confusion: {
          counts: this.counts.map((r) => r.slice()),
          normalised: this.counts.map((row) => {
            const s = row.reduce((a, b) => a + b, 0);
            return row.map((v) => (s ? Number((v / s).toFixed(5)) : 0));
          }),
          cumulative: this.cumulative.map((r) => r.slice()),
          per_class: perClass,
          macro_f1: Number(macroF1.toFixed(4)),
          micro_f1: Number(acc.toFixed(4)),
          accuracy: Number(acc.toFixed(4)),
          false_negative_rate: Number((defectTotal ? defectMissed / defectTotal : 0).toFixed(4)),
          window: WINDOW,
          total: this.total(),
          class_names: CLASS_NAMES,
        },
        calibration: this.calibration(),
        pareto: {
          live: (Object.keys(MODELS) as ModelKey[])
            .filter((k) => (this.framesByModel[k] ?? 0) > 0)
            .map((k) => {
              const frames = this.framesByModel[k] ?? 0;
              const a = frames ? (this.correctByModel[k] ?? 0) / frames : 0;
              const lat =
                k === this.modelKey ? serviceMs : MODELS[k].nominal_latency_ms;
              return {
                model_key: k,
                latency_ms: Number(lat.toFixed(3)),
                accuracy: Number(a.toFixed(4)),
                macro_f1: Number(macroF1.toFixed(4)),
                power_watts: MODELS[k].power_watts,
                severity: Number(severity.toFixed(3)),
                frames,
                dominated: false,
              };
            }),
          trails: this.paretoTrails,
        },
        curves: { loss: this.lossCurve.slice(), accuracy: this.accCurve.slice() },
        per_model: Object.fromEntries(
          Object.keys(this.framesByModel).map((k) => [
            k,
            {
              frames: this.framesByModel[k],
              accuracy: Number(
                ((this.correctByModel[k] ?? 0) / Math.max(this.framesByModel[k], 1)).toFixed(4),
              ),
            },
          ]),
        ),
      },
      storms: {
        storms: { ...this.storms },
        filters: Array.from(this.filters).sort(),
        nominal_filter_cost_ms: Number(this.filterCostMs.toFixed(3)),
      },
      severity: Number(severity.toFixed(4)),
      conveyor_new: delta,
      conveyor_depth: CONVEYOR_DEPTH,
      events,
      summary: {
        frames_offered: this.processedTotal + this.dropsTotal,
        frames_processed: this.processedTotal,
        frames_dropped: this.dropsTotal,
        missed_defects: this.missedDefects,
        false_alarms: this.falseAlarms,
        sla_breaches: this.slaBreaches,
        final_health: Number(this.health.toFixed(2)),
        final_score: Number(this.score.toFixed(1)),
        elapsed_s: Number(this.clock.toFixed(2)),
        game_over: this.gameOver,
      },
    };
  }
}

/** The handshake the offline simulator presents in place of the server's. */
export function offlineHandshake(): Handshake {
  return {
    type: "hello",
    version: "1.0.0-offline",
    tick_hz: TICK_HZ,
    classes: CLASS_META.map((c) => ({ ...c })) as Handshake["classes"],
    models: (Object.keys(MODELS) as ModelKey[]).map((k) => {
      const m = MODELS[k];
      return {
        key: k,
        display_name: m.display_name,
        family: m.family,
        accent: m.accent,
        blurb:
          k === "squeezenet"
            ? "Blazing fast, low capacity. Misses subtle defects under storms."
            : k === "mobilenet"
              ? "The architectural sweet spot. Fits the 40 ms budget with margin."
              : "Most accurate, and 44 ms > 40 ms budget. Overflows the buffer.",
        architecture_note:
          k === "squeezenet"
            ? "Fire modules: a 1x1 squeeze layer collapses channel depth, then a parallel 1x1 + 3x3 expand pair restores it."
            : k === "mobilenet"
              ? "Inverted residuals with linear bottlenecks: expand by t=6, filter with a depthwise 3x3, project back down linearly."
              : "Residual stages of two 3x3 convolutions with an identity skip. Dense convolutions at full width cost 1.8 GFLOPs.",
        signature_block:
          k === "squeezenet"
            ? "1x1 squeeze -> [1x1 expand || 3x3 expand] -> concat"
            : k === "mobilenet"
              ? "1x1 expand (t=6) -> 3x3 depthwise -> 1x1 linear project (+ skip)"
              : "3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+identity) -> ReLU",
        clean_accuracy: m.clean_accuracy,
        capacity: m.capacity,
        power_watts: m.power_watts,
        confidence_temperature: m.confidence_temperature,
        nominal_latency_ms: m.nominal_latency_ms,
        nominal_throughput_fps: Number((1000 / m.nominal_latency_ms).toFixed(2)),
        pretrained: false,
        measured_latency: {
          mean_ms: m.nominal_latency_ms,
          median_ms: m.nominal_latency_ms,
          p95_ms: m.nominal_latency_ms,
          min_ms: m.nominal_latency_ms,
          max_ms: m.nominal_latency_ms,
          stdev_ms: 0,
          samples: 0,
        },
        architecture: {
          model_key: k,
          total_params: Math.round(m.params_m * 1e6),
          trainable_params: Math.round(m.params_m * 1e6),
          total_macs: Math.round(m.macs_m * 1e6),
          total_flops: Math.round(m.gflops * 1e9),
          resolution_cascade: k === "squeezenet" ? [224, 111, 55, 27, 13] : [224, 112, 56, 28, 14, 7],
          weight_bytes: Math.round(m.params_m * 1e6 * 4),
          weight_mb: Number(((m.params_m * 1e6 * 4) / (1024 * 1024)).toFixed(3)),
          params_millions: m.params_m,
          macs_millions: m.macs_m,
          gflops: m.gflops,
          layer_count: 0,
        },
        mac_efficiency_m_per_ms: Number((m.macs_m / m.nominal_latency_ms).toFixed(1)),
      };
    }),
    augmentations: {
      storms: [
        { key: "heat_haze", display_name: "Hot Aisle Heat Haze", short: "HAZE", accent: "#f59e0b", icon: "waves", severity_weight: 0.46, formula: "x' = x + A·sin(2πf·y + φ)", blurb: "Refractive shimmer off the hot aisle. Warps thin geometry.", physics: "Air density varies with temperature, so refractive index does too. Light crossing a thermal gradient bends. Geometric defects are destroyed first, because the cue IS the geometry." },
        { key: "strobe_glare", display_name: "Strobe LED Glare", short: "GLARE", accent: "#38bdf8", icon: "flash", severity_weight: 0.38, formula: "I' = I + G·exp(-‖p - c‖² / 2σ²)", blurb: "Specular bloom from maintenance lighting. Clips highlights.", physics: "An additive Gaussian specular lobe with hard highlight clipping. It saturates channels rather than attenuating them, so colour-coded cues vanish." },
        { key: "intake_dust", display_name: "Intake Particulate Storm", short: "DUST", accent: "#94a3b8", icon: "grain", severity_weight: 0.34, formula: "I' = I·(1-α) + Poisson(λ)·α", blurb: "Airborne particulate. High-frequency occlusion noise.", physics: "Shot noise from particles crossing the sensor plane. Signal-dependent and spatially clumped, so low-pass filtering removes it only at the cost of fine texture." },
      ],
      filters: [
        { key: "clahe", display_name: "CLAHE", short: "CLAHE", accent: "#10b981", counters: FILTER_COUNTERS.clahe, nominal_cost_ms: 1.8, formula: "Contrast-Limited Adaptive Histogram Equalisation (8×8 tiles)", blurb: "Restores local contrast inside blown-out glare lobes.", tradeoff: "Clip-limited, but still lifts the noise floor in dark regions." },
        { key: "unsharp", display_name: "Unsharp Mask", short: "UNSHRP", accent: "#06b6d4", counters: FILTER_COUNTERS.unsharp, nominal_cost_ms: 1.1, formula: "I' = I + k·(I − G_σ * I)", blurb: "Re-sharpens edges smeared by refractive warping.", tradeoff: "Amplifies high frequencies indiscriminately — harmful under dust." },
        { key: "bilateral", display_name: "Bilateral Edge Filter", short: "BILAT", accent: "#a855f7", counters: FILTER_COUNTERS.bilateral, nominal_cost_ms: 4.6, formula: "I'(p) = (1/W)·Σ G_σs·G_σr·I(q)", blurb: "Edge-preserving denoise. Kills grain, keeps boundaries.", tradeoff: "The most expensive kernel in the bank — a tenth of the frame budget." },
      ],
      backends: { opencv: false, scipy: false },
    },
    config: {
      arrival_rate_fps: ARRIVAL_FPS,
      buffer_capacity: BUFFER_CAPACITY,
      frame_budget_ms: FRAME_BUDGET_MS,
      max_health: MAX_HEALTH,
      sla_latency_ms: SLA_LATENCY_MS,
    },
    default_model: "mobilenet",
  };
}
