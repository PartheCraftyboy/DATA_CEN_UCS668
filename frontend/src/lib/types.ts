/**
 * Wire types for the telemetry stream.
 *
 * These mirror the payloads built in `backend/app/main.py`. They are hand-
 * written rather than generated because the surface is small and stable, and a
 * hand-written type can carry the units in its field names — `latency_ms`, not
 * `latency` — which is the only reliable defence against the classic
 * seconds/milliseconds bug in a project where both appear.
 */

export type ClassId = 0 | 1 | 2 | 3 | 4;
export type ModelKey = "squeezenet" | "mobilenet" | "resnet";
export type StormKey = "heat_haze" | "strobe_glare" | "intake_dust";
export type FilterKey = "clahe" | "unsharp" | "bilateral";

export type Outcome =
  | "detected"
  | "nominal"
  | "missed_defect"
  | "false_alarm"
  | "misclassified_defect"
  | "dropped";

export interface ClassMeta {
  id: ClassId;
  key: string;
  label: string;
  short: string;
  severity: number;
  miss_cost: number;
  accent: string;
  blurb: string;
}

export interface LayerTrace {
  index: number;
  name: string;
  kind: string;
  in_shape: number[];
  out_shape: number[];
  params: number;
  macs: number;
  resolution: number | null;
  block_role: string;
}

export interface ArchitectureProfile {
  model_key: string;
  total_params: number;
  trainable_params: number;
  total_macs: number;
  total_flops: number;
  resolution_cascade: number[];
  weight_bytes: number;
  weight_mb: number;
  params_millions: number;
  macs_millions: number;
  gflops: number;
  layer_count: number;
  layers?: LayerTrace[];
}

export interface ModelCard {
  key: ModelKey;
  display_name: string;
  family: string;
  accent: string;
  blurb: string;
  architecture_note: string;
  signature_block: string;
  clean_accuracy: number;
  capacity: number;
  power_watts: number;
  confidence_temperature: number;
  nominal_latency_ms: number;
  nominal_throughput_fps: number;
  pretrained: boolean;
  measured_latency: {
    mean_ms: number;
    median_ms: number;
    p95_ms: number;
    min_ms: number;
    max_ms: number;
    stdev_ms: number;
    samples: number;
  };
  architecture: ArchitectureProfile;
  mac_efficiency_m_per_ms: number;
}

export interface StormSpec {
  key: StormKey;
  display_name: string;
  short: string;
  accent: string;
  icon: string;
  severity_weight: number;
  formula: string;
  blurb: string;
  physics: string;
}

export interface FilterSpec {
  key: FilterKey;
  display_name: string;
  short: string;
  accent: string;
  counters: Record<string, number>;
  nominal_cost_ms: number;
  formula: string;
  blurb: string;
  tradeoff: string;
}

export interface QueueState {
  depth: number;
  capacity: number;
  fill_ratio: number;
  utilisation: number;
  service_time_ms: number;
  throughput_fps: number;
  arrival_rate_fps: number;
  drops_total: number;
  drops_recent: number;
  processed_total: number;
  health: number;
  score: number;
  mean_system_time_ms: number;
  p95_system_time_ms: number;
  sla_breaches: number;
  overflow: boolean;
  saturated: boolean;
  elapsed_s: number;
}

export interface TheoryState {
  rho: number;
  stable: boolean;
  service_rate_fps: number;
  arrival_rate_fps: number;
  lq_md1: number | null;
  lq_mm1: number | null;
  wq_ms: number | null;
  expected_system_ms: number | null;
  blocking_probability_bound: number;
  buffer_capacity: number;
  frame_budget_ms: number;
  budget_headroom_ms: number;
  note: string;
}

export interface PerClassMetric {
  class_id: ClassId;
  class_name: string;
  label: string;
  accent: string;
  support: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  top_confusion: { class_id: number; class_name: string; count: number; share: number } | null;
}

export interface ConfusionState {
  counts: number[][];
  normalised: number[][];
  cumulative: number[][];
  per_class: PerClassMetric[];
  macro_f1: number;
  micro_f1: number;
  accuracy: number;
  false_negative_rate: number;
  window: number;
  total: number;
  class_names: string[];
}

export interface CalibrationBin {
  lo: number;
  hi: number;
  count: number;
  accuracy: number;
  confidence: number;
  gap: number;
}

export interface CalibrationState {
  ece: number;
  mce: number;
  bins: CalibrationBin[];
  mean_cross_entropy: number;
  mean_confidence: number;
  samples: number;
  calibration_bias: number;
}

export interface ParetoLivePoint {
  model_key: ModelKey;
  latency_ms: number;
  accuracy: number;
  macro_f1: number;
  power_watts: number;
  severity: number;
  frames: number;
  dominated: boolean;
}

export interface ParetoState {
  live: ParetoLivePoint[];
  trails: Record<string, { latency_ms: number; accuracy: number }[]>;
}

export interface MetricsState {
  confusion: ConfusionState;
  calibration: CalibrationState;
  pareto: ParetoState;
  curves: { loss: number[]; accuracy: number[] };
  per_model: Record<string, { frames: number; accuracy: number }>;
}

export interface ConveyorItem {
  frame_id: number;
  true_class: ClassId;
  pred_class: ClassId | null;
  dropped: boolean;
  thumbnail: string | null;
  confidence: number;
  probs?: number[];
  outcome: Outcome;
  model_key: ModelKey;
  system_time_ms?: number;
  queue_delay_ms?: number;
  severity?: number;
  descriptor?: Record<string, unknown>;
}

export interface CockpitEvent {
  type: "drop" | "missed_defect";
  frame_id: number;
  true_class: ClassId;
  label: string;
  message: string;
}

export interface Telemetry {
  type: "telemetry";
  t: number;
  paused: boolean;
  model: {
    key: ModelKey;
    display_name: string;
    accent: string;
    family: string;
    nominal_latency_ms: number;
    measured_latency_ms: number;
    power_watts: number;
    capacity: number;
  };
  service: {
    total_ms: number;
    inference_ms: number;
    preprocess_ms: number;
    preprocess_measured_ms: number;
    budget_ms: number;
    over_budget: boolean;
  };
  queue: QueueState;
  theory: TheoryState;
  metrics: MetricsState;
  storms: {
    storms: Record<StormKey, number>;
    filters: FilterKey[];
    nominal_filter_cost_ms: number;
  };
  severity: number;
  conveyor_new: ConveyorItem[];
  conveyor_depth: number;
  events: CockpitEvent[];
  summary: Record<string, number | boolean>;
}

export interface Handshake {
  type: "hello";
  version: string;
  tick_hz: number;
  classes: ClassMeta[];
  models: ModelCard[];
  augmentations: {
    storms: StormSpec[];
    filters: FilterSpec[];
    backends: { opencv: boolean; scipy: boolean };
  };
  config: {
    arrival_rate_fps: number;
    buffer_capacity: number;
    frame_budget_ms: number;
    max_health: number;
    sla_latency_ms: number;
  };
  default_model: ModelKey;
}

export type ConnectionState = "connecting" | "live" | "offline" | "simulated";

/** Commands the cockpit can send back up the socket. */
export type Command =
  | { action: "set_model"; model: ModelKey }
  | { action: "set_storm"; storm: StormKey; intensity: number }
  | { action: "toggle_storm"; storm: StormKey }
  | { action: "set_filter"; filter: FilterKey; enabled: boolean }
  | { action: "toggle_filter"; filter: FilterKey }
  | { action: "pause"; paused: boolean }
  | { action: "reset" }
  | { action: "ping" };
