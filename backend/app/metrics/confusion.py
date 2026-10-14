"""
Module 5 — Training, Optimization & Metrics.

A live metrics engine: a 5x5 confusion matrix, per-class precision/recall/F1,
macro and micro F1, running cross-entropy, and Expected Calibration Error.

Design note — why the matrix is *windowed*, not cumulative
───────────────────────────────────────────────────────────
The player switches backbones mid-run. A cumulative confusion matrix would
average SqueezeNet's behaviour under a dust storm together with ResNet-18's
behaviour in clean air, and the result would describe neither. We therefore
keep a sliding window of recent predictions (default 300 frames, ~12 s at
25 FPS) so the matrix tracks the *current* configuration, and report the
cumulative totals separately for the end-of-run report card.

That decision has a visible payoff in the demo: switch from ResNet-18 to
SqueezeNet with a heat haze running and the matrix visibly loses its diagonal
within a few seconds, instead of drifting imperceptibly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Sequence

import numpy as np

from app import CLASS_META, CLASS_NAMES, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────


class ConfusionTracker:
    """Sliding-window 5x5 confusion matrix with derived classification metrics.

    Convention: `matrix[true][pred]`. Rows sum to the number of frames of that
    true class. Row-normalising therefore gives **recall** on the diagonal,
    which is the normalisation the course brief asks for and the one that
    matters for an inspection system — "of the thermal hotspots that existed,
    what fraction did we catch?"
    """

    def __init__(self, window: int = 300) -> None:
        self.window = window
        self.counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self.cumulative = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self._history: Deque[tuple[int, int]] = deque(maxlen=window)

    def update(self, true_class: int, pred_class: int) -> None:
        if len(self._history) == self.window:
            old_t, old_p = self._history[0]
            self.counts[old_t, old_p] -= 1
        self._history.append((true_class, pred_class))
        self.counts[true_class, pred_class] += 1
        self.cumulative[true_class, pred_class] += 1

    def reset(self) -> None:
        self.counts[:] = 0
        self.cumulative[:] = 0
        self._history.clear()

    # -- derived metrics ------------------------------------------------------

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def normalised(self) -> np.ndarray:
        """Row-normalised matrix. Diagonal = per-class recall."""
        row_sums = self.counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(row_sums > 0, self.counts / np.maximum(row_sums, 1), 0.0)
        return norm

    def per_class(self) -> list[dict[str, Any]]:
        """Precision, recall and F1 for each class, plus raw TP/FP/FN counts."""
        out: list[dict[str, Any]] = []
        for c in range(NUM_CLASSES):
            tp = int(self.counts[c, c])
            fn = int(self.counts[c, :].sum() - tp)   # was class c, called something else
            fp = int(self.counts[:, c].sum() - tp)   # called class c, wasn't
            tn = int(self.counts.sum() - tp - fn - fp)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            out.append(
                {
                    "class_id": c,
                    "class_name": CLASS_NAMES[c],
                    "label": CLASS_META[c]["label"],
                    "accent": CLASS_META[c]["accent"],
                    "support": int(self.counts[c, :].sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    # The row entry most responsible for this class's misses —
                    # the frontend uses it to annotate the dominant confusion.
                    "top_confusion": self._top_confusion(c),
                }
            )
        return out

    def _top_confusion(self, c: int) -> dict[str, Any] | None:
        row = self.counts[c].copy()
        row[c] = 0
        if row.sum() == 0:
            return None
        j = int(row.argmax())
        return {
            "class_id": j,
            "class_name": CLASS_NAMES[j],
            "count": int(row[j]),
            "share": round(float(row[j] / max(self.counts[c].sum(), 1)), 4),
        }

    def macro_f1(self) -> float:
        """Unweighted mean F1 across classes.

        Macro, not micro, deliberately: our stream is 55% Optimal_Rack, so
        micro-F1 is dominated by the easy majority class and would read ~0.9
        even for a model that catches no thermal hotspots at all. Macro-F1
        gives the rare critical classes equal weight, which is the whole point
        of an inspection system.
        """
        scores = [m["f1"] for m in self.per_class() if m["support"] > 0]
        return float(np.mean(scores)) if scores else 0.0

    def micro_f1(self) -> float:
        """Equivalent to overall accuracy in a single-label multi-class setting."""
        total = self.counts.sum()
        return float(np.trace(self.counts) / total) if total else 0.0

    def accuracy(self) -> float:
        return self.micro_f1()

    def false_negative_rate(self) -> float:
        """Fraction of real defects that were called Optimal_Rack.

        The headline safety metric. Class 0 is 'no defect', so every entry in
        column 0 from a non-zero row is a fault that went uninspected.
        """
        defect_rows = self.counts[1:, :]
        defect_total = defect_rows.sum()
        if defect_total == 0:
            return 0.0
        return float(defect_rows[:, 0].sum() / defect_total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts.tolist(),
            "normalised": np.round(self.normalised(), 5).tolist(),
            "cumulative": self.cumulative.tolist(),
            "per_class": self.per_class(),
            "macro_f1": round(self.macro_f1(), 4),
            "micro_f1": round(self.micro_f1(), 4),
            "accuracy": round(self.accuracy(), 4),
            "false_negative_rate": round(self.false_negative_rate(), 4),
            "window": self.window,
            "total": self.total,
            "class_names": list(CLASS_NAMES),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────


class CalibrationTracker:
    """Expected Calibration Error over a sliding window of predictions.

    ECE partitions predictions into M equal-width confidence bins and measures
    the gap between confidence and accuracy within each:

        ECE = Σ_m (|B_m| / n) · |acc(B_m) − conf(B_m)|

    A perfectly calibrated model that says "90% confident" is right 90% of the
    time, and scores 0. Modern deep networks are famously *not* calibrated:
    they are systematically overconfident, and the effect grows with capacity
    (Guo et al., 2017). Our zoo reproduces this on purpose — ResNet-18 carries
    a confidence temperature of 1.45, SqueezeNet 0.78 — so the cockpit shows
    the counter-intuitive result that the *most accurate* model also has the
    *worst* calibration.

    That matters operationally: if you want to route only low-confidence frames
    to the heavy model, you need confidences you can trust. An overconfident
    model never asks for help.
    """

    def __init__(self, bins: int = 10, window: int = 300) -> None:
        self.bins = bins
        self.window = window
        self._samples: Deque[tuple[float, bool]] = deque(maxlen=window)
        self._ce_samples: Deque[float] = deque(maxlen=window)

    def update(self, confidence: float, correct: bool, cross_entropy: float) -> None:
        self._samples.append((float(confidence), bool(correct)))
        self._ce_samples.append(float(cross_entropy))

    def reset(self) -> None:
        self._samples.clear()
        self._ce_samples.clear()

    def bin_stats(self) -> list[dict[str, Any]]:
        edges = np.linspace(0.0, 1.0, self.bins + 1)
        stats: list[dict[str, Any]] = []
        if not self._samples:
            return [
                {
                    "lo": round(float(edges[i]), 3),
                    "hi": round(float(edges[i + 1]), 3),
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "gap": 0.0,
                }
                for i in range(self.bins)
            ]

        conf = np.array([c for c, _ in self._samples])
        corr = np.array([1.0 if ok else 0.0 for _, ok in self._samples])

        for i in range(self.bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            n = int(mask.sum())
            if n:
                acc = float(corr[mask].mean())
                avg_conf = float(conf[mask].mean())
            else:
                acc = avg_conf = 0.0
            stats.append(
                {
                    "lo": round(float(lo), 3),
                    "hi": round(float(hi), 3),
                    "count": n,
                    "accuracy": round(acc, 4),
                    "confidence": round(avg_conf, 4),
                    "gap": round(acc - avg_conf, 4),
                }
            )
        return stats

    def ece(self) -> float:
        n = len(self._samples)
        if n == 0:
            return 0.0
        total = 0.0
        for b in self.bin_stats():
            if b["count"]:
                total += (b["count"] / n) * abs(b["accuracy"] - b["confidence"])
        return float(total)

    def mce(self) -> float:
        """Maximum Calibration Error — the worst single bin."""
        gaps = [abs(b["accuracy"] - b["confidence"]) for b in self.bin_stats() if b["count"]]
        return float(max(gaps)) if gaps else 0.0

    def mean_cross_entropy(self) -> float:
        return float(np.mean(self._ce_samples)) if self._ce_samples else 0.0

    def mean_confidence(self) -> float:
        return float(np.mean([c for c, _ in self._samples])) if self._samples else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": round(self.ece(), 5),
            "mce": round(self.mce(), 5),
            "bins": self.bin_stats(),
            "mean_cross_entropy": round(self.mean_cross_entropy(), 5),
            "mean_confidence": round(self.mean_confidence(), 5),
            "samples": len(self._samples),
            # Positive = underconfident, negative = overconfident.
            "calibration_bias": round(
                float(
                    np.mean([1.0 if ok else 0.0 for _, ok in self._samples])
                    - self.mean_confidence()
                )
                if self._samples
                else 0.0,
                5,
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Pareto frontier accumulator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParetoSample:
    model_key: str
    latency_ms: float
    accuracy: float
    power_watts: float
    macro_f1: float
    severity: float
    frames: int


class ParetoAccumulator:
    """Collects live (latency, accuracy, power) operating points per backbone.

    The static Pareto frontier from the spec sheet is one curve. The *measured*
    operating points are another, and they move: deploy a bilateral filter and
    a backbone's point slides right (more latency) and up (more accuracy),
    tracing the actual trade-off surface the player is navigating. Showing the
    live points drifting off the spec-sheet curve is a much better argument
    than the curve alone.
    """

    def __init__(self, min_frames: int = 25) -> None:
        self.min_frames = min_frames
        self._live: dict[str, ParetoSample] = {}
        self._trail: dict[str, list[dict[str, float]]] = {}

    def update(
        self,
        model_key: str,
        latency_ms: float,
        accuracy: float,
        macro_f1: float,
        power_watts: float,
        severity: float,
        frames: int,
    ) -> None:
        self._live[model_key] = ParetoSample(
            model_key=model_key,
            latency_ms=latency_ms,
            accuracy=accuracy,
            power_watts=power_watts,
            macro_f1=macro_f1,
            severity=severity,
            frames=frames,
        )
        if frames >= self.min_frames:
            trail = self._trail.setdefault(model_key, [])
            point = {"latency_ms": round(latency_ms, 3), "accuracy": round(accuracy, 4)}
            if not trail or trail[-1] != point:
                trail.append(point)
                if len(trail) > 60:
                    trail.pop(0)

    def reset(self) -> None:
        self._live.clear()
        self._trail.clear()

    def is_dominated(self, sample: ParetoSample) -> bool:
        """True if another live point is at least as good on every axis.

        Pareto dominance: B dominates A if B is no worse on all objectives and
        strictly better on at least one. Here: lower latency, higher accuracy,
        lower power.
        """
        for other in self._live.values():
            if other.model_key == sample.model_key:
                continue
            no_worse = (
                other.latency_ms <= sample.latency_ms
                and other.accuracy >= sample.accuracy
                and other.power_watts <= sample.power_watts
            )
            strictly_better = (
                other.latency_ms < sample.latency_ms
                or other.accuracy > sample.accuracy
                or other.power_watts < sample.power_watts
            )
            if no_worse and strictly_better:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        live = []
        for sample in self._live.values():
            live.append(
                {
                    "model_key": sample.model_key,
                    "latency_ms": round(sample.latency_ms, 3),
                    "accuracy": round(sample.accuracy, 4),
                    "macro_f1": round(sample.macro_f1, 4),
                    "power_watts": sample.power_watts,
                    "severity": round(sample.severity, 3),
                    "frames": sample.frames,
                    "dominated": self.is_dominated(sample),
                }
            )
        return {
            "live": live,
            "trails": {k: v for k, v in self._trail.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Combined engine
# ─────────────────────────────────────────────────────────────────────────────


class MetricsEngine:
    """One object owning all Module 5 telemetry for a live session."""

    def __init__(self, window: int = 300, bins: int = 10) -> None:
        self.confusion = ConfusionTracker(window=window)
        self.calibration = CalibrationTracker(bins=bins, window=window)
        self.pareto = ParetoAccumulator()
        self.frames_by_model: dict[str, int] = {}
        self.correct_by_model: dict[str, int] = {}
        self._loss_curve: Deque[float] = deque(maxlen=160)
        self._accuracy_curve: Deque[float] = deque(maxlen=160)
        self._since_curve_sample = 0

    def observe(
        self,
        true_class: int,
        pred_class: int,
        confidence: float,
        cross_entropy: float,
        model_key: str,
    ) -> None:
        self.confusion.update(true_class, pred_class)
        self.calibration.update(confidence, pred_class == true_class, cross_entropy)

        self.frames_by_model[model_key] = self.frames_by_model.get(model_key, 0) + 1
        if pred_class == true_class:
            self.correct_by_model[model_key] = self.correct_by_model.get(model_key, 0) + 1

        # Sample the loss/accuracy curves every 5 frames — at 25 FPS a
        # per-frame curve is 25 points a second and renders as noise.
        self._since_curve_sample += 1
        if self._since_curve_sample >= 5:
            self._since_curve_sample = 0
            self._loss_curve.append(round(self.calibration.mean_cross_entropy(), 4))
            self._accuracy_curve.append(round(self.confusion.accuracy(), 4))

    def accuracy_for(self, model_key: str) -> float:
        n = self.frames_by_model.get(model_key, 0)
        if not n:
            return 0.0
        return self.correct_by_model.get(model_key, 0) / n

    def reset(self) -> None:
        self.confusion.reset()
        self.calibration.reset()
        self.pareto.reset()
        self.frames_by_model.clear()
        self.correct_by_model.clear()
        self._loss_curve.clear()
        self._accuracy_curve.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion": self.confusion.to_dict(),
            "calibration": self.calibration.to_dict(),
            "pareto": self.pareto.to_dict(),
            "curves": {
                "loss": list(self._loss_curve),
                "accuracy": list(self._accuracy_curve),
            },
            "per_model": {
                k: {
                    "frames": v,
                    "accuracy": round(self.accuracy_for(k), 4),
                }
                for k, v in self.frames_by_model.items()
            },
        }
