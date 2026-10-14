"""
Module 2 — Building Your First Vision Model (and then two more).

The switchable backbone zoo: SqueezeNet v1.1, MobileNetV2 and ResNet-18, each
loaded from torchvision, each re-headed for our 5 inspection classes, each
wrapped in a microsecond-resolution timer.

────────────────────────────────────────────────────────────────────────────────
 SIMULATION FIDELITY — read this before the viva
────────────────────────────────────────────────────────────────────────────────
Be precise about which numbers in this project are *measured* and which are
*modelled*, because an examiner will ask and the honest answer is the better
answer:

  MEASURED (real PyTorch, real CPU, real wall clock)
    · Parameter counts                — iterated from the live nn.Parameter set
    · MAC / FLOP counts               — forward hooks on real tensor shapes
    · Layer-by-layer tensor shapes    — the real 224→112→56→28→14→7 cascade
    · Inference latency T_infer       — time.perf_counter() around a real
                                        forward pass on this machine's CPU
    · Preprocessing kernel cost       — real NumPy/SciPy filter execution

  MODELLED (a documented decision model, not a trained classifier)
    · Which class the backbone predicts, and with what confidence.

Why: these backbones are initialised from ImageNet weights (or randomly, if
weights are unavailable offline) and have *not* been fine-tuned on a labelled
dataset of datacenter rack photographs — no such dataset exists for this lab.
An untrained 5-way head emits uniform noise, which would make the confusion
matrix a meaningless grey square and destroy the pedagogy.

So the zoo runs the real network — paying the real latency, which is the
quantity the whole M/D/1 argument depends on — and then resolves the *label*
through `CapacityDecisionModel`: a class-conditional confusion kernel whose
diagonal dominance is a function of (a) the backbone's representational
capacity and (b) the current environmental storm severity. The real network
logits are folded in as the entropy source, so the decision still varies with
the actual pixels rather than being a coin flip on the ground truth.

The result is a simulator that is *exactly* honest about the thing it is
teaching: latency is real, and accuracy degrades with capacity under
distribution shift in the way the literature reports. Nothing here claims to
be a trained detector.

Set DCV_PRETRAINED=1 to attempt downloading ImageNet weights (improves the
realism of the feature statistics; requires network access and is entirely
optional — the simulation is identical without them).
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app import CLASS_NAMES, NUM_CLASSES
from app.models.profiler import (
    ArchitectureProfile,
    LatencyProfile,
    benchmark_latency,
    profile_architecture,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Model specifications
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one backbone in the zoo.

    `nominal_latency_ms` is the reference figure from the course brief, used
    for the M/D/1 queue arithmetic so the simulation behaves identically on a
    fast lab desktop and a slow laptop. The *measured* latency from this
    machine is reported alongside it, and the frontend shows both — the gap
    between "the textbook says 44 ms" and "your CPU says 61 ms" is itself a
    good talking point about hardware-dependent deployment budgets.
    """

    key: str
    display_name: str
    family: str
    torchvision_fn: str
    # Reference latency from the course brief (ms), used for queue service time.
    nominal_latency_ms: float
    # Clean-condition top-1 accuracy this backbone achieves on the task.
    clean_accuracy: float
    # Representational capacity in [0, 1]. Drives how fast accuracy collapses
    # under augmentation storms.
    capacity: float
    # Estimated edge power draw at full duty cycle (W) — the third Pareto axis.
    power_watts: float
    # Confidence sharpening. >1 means the model is systematically overconfident,
    # which is what produces a non-zero Expected Calibration Error.
    confidence_temperature: float
    accent: str
    blurb: str
    architecture_note: str
    signature_block: str

    @property
    def nominal_throughput_fps(self) -> float:
        """Service rate mu = 1 / T_service, in frames per second."""
        return 1000.0 / self.nominal_latency_ms


MODEL_SPECS: dict[str, ModelSpec] = {
    "squeezenet": ModelSpec(
        key="squeezenet",
        display_name="SqueezeNet v1.1",
        family="Fire Module",
        torchvision_fn="squeezenet1_1",
        nominal_latency_ms=4.5,
        clean_accuracy=0.885,
        capacity=0.34,
        power_watts=1.9,
        confidence_temperature=0.78,  # under-confident: soft, hedged softmax
        accent="#10b981",
        blurb="Blazing fast, low capacity. Misses subtle defects under storms.",
        architecture_note=(
            "Fire modules: a 1x1 squeeze layer collapses channel depth, then a "
            "parallel 1x1 + 3x3 expand pair restores it. Dropping 3x3 filters "
            "to a third of the channels is where the 50x parameter reduction "
            "against AlexNet comes from."
        ),
        signature_block="1x1 squeeze -> [1x1 expand || 3x3 expand] -> concat",
    ),
    "mobilenet": ModelSpec(
        key="mobilenet",
        display_name="MobileNetV2",
        family="Inverted Residual",
        torchvision_fn="mobilenet_v2",
        nominal_latency_ms=11.5,
        clean_accuracy=0.951,
        capacity=0.72,
        power_watts=3.4,
        confidence_temperature=1.05,  # near-calibrated: the sweet spot
        accent="#06b6d4",
        blurb="The architectural sweet spot. Fits the 40 ms budget with margin.",
        architecture_note=(
            "Inverted residuals with linear bottlenecks: expand channels by "
            "t=6 with a 1x1, filter spatially with a cheap depthwise 3x3, then "
            "project back down with a linear 1x1. Separating spatial filtering "
            "from channel mixing cuts compute ~6x versus a dense convolution "
            "of equivalent receptive field."
        ),
        signature_block="1x1 expand (t=6) -> 3x3 depthwise -> 1x1 linear project (+ skip)",
    ),
    "resnet": ModelSpec(
        key="resnet",
        display_name="ResNet-18",
        family="Residual Stage",
        torchvision_fn="resnet18",
        nominal_latency_ms=44.0,
        clean_accuracy=0.976,
        capacity=0.94,
        power_watts=7.8,
        confidence_temperature=1.45,  # overconfident: the classic deep-net ECE
        accent="#a855f7",
        blurb="Most accurate, and 44 ms > 40 ms budget. Overflows the buffer.",
        architecture_note=(
            "Residual stages of two 3x3 convolutions with an identity skip. "
            "The skip makes the gradient path shallow regardless of depth, "
            "which is what let networks get deep in the first place — but "
            "dense 3x3 convolutions at full width cost 1.8 GFLOPs."
        ),
        signature_block="3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+identity) -> ReLU",
    ),
}

DEFAULT_MODEL_KEY = "mobilenet"


# ─────────────────────────────────────────────────────────────────────────────
#  Class confusability prior
# ─────────────────────────────────────────────────────────────────────────────
#
# Not all mistakes are equally likely. A weak model under heat haze does not
# confuse a thermal hotspot with a dust-clogged intake at random — it confuses
# it with whatever shares low-level statistics. This matrix encodes which
# confusions are *physically plausible* in a server aisle:
#
#   · Cable bend  <-> Optimal      a fiber loop against a dark chassis is a
#                                  thin low-contrast curve; the easiest defect
#                                  in the set to simply not notice.
#   · Thermal     <-> Port LED     both are warm-coloured blobs; strobe glare
#                                  turns an amber port into an orange plume.
#   · Dust        <-> Optimal      occlusion is textural, and dust noise
#                                  augmentation actively masks the signature.
#
# Rows are ground truth, columns are the *relative* likelihood of each wrong
# answer. Diagonals are set at runtime from the model's accuracy.

_CONFUSABILITY = np.array(
    [
        # true Optimal  -> most often mistaken for a subtle bend or dust
        [0.00, 0.08, 0.38, 0.14, 0.40],
        # true Thermal  -> leaks into Port LED (warm blob) and Optimal (missed)
        [0.34, 0.00, 0.08, 0.46, 0.12],
        # true Bend     -> overwhelmingly a false negative against Optimal
        [0.62, 0.05, 0.00, 0.19, 0.14],
        # true Port LED -> confused with thermal, or washed out to Optimal
        [0.30, 0.44, 0.12, 0.00, 0.14],
        # true Dust     -> a missed occlusion reads as a normal rack
        [0.58, 0.07, 0.17, 0.18, 0.00],
    ],
    dtype=np.float64,
)

# How much each class's detectability suffers per unit of storm severity,
# scaled by (1 - capacity). Cable bends are the most fragile: the defect is a
# geometric cue that heat-haze displacement directly destroys.
_STORM_FRAGILITY = np.array([0.22, 0.46, 0.78, 0.55, 0.40], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InferenceResult:
    """Everything one forward pass tells us."""

    model_key: str
    true_class: int
    pred_class: int
    probs: list[float]
    confidence: float
    # Real wall-clock time of the forward pass on this machine.
    measured_latency_ms: float
    # Reference latency used for queue service-time arithmetic.
    nominal_latency_ms: float
    correct: bool
    # Cross-entropy of this single prediction against the true label.
    cross_entropy: float
    # Raw logit vector from the real network, for the telemetry inspector.
    logits: list[float] = field(default_factory=list)
    storm_severity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "true_class": self.true_class,
            "pred_class": self.pred_class,
            "true_label": CLASS_NAMES[self.true_class],
            "pred_label": CLASS_NAMES[self.pred_class],
            "probs": [round(p, 5) for p in self.probs],
            "confidence": round(self.confidence, 5),
            "measured_latency_ms": round(self.measured_latency_ms, 4),
            "nominal_latency_ms": self.nominal_latency_ms,
            "correct": self.correct,
            "cross_entropy": round(self.cross_entropy, 5),
            "storm_severity": round(self.storm_severity, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Capacity-limited decision model
# ─────────────────────────────────────────────────────────────────────────────


class CapacityDecisionModel:
    """Maps (true class, backbone capacity, storm severity) -> (pred, softmax).

    This is the modelled half of the system, and it is deliberately simple
    enough to defend line by line:

      1.  Effective per-class recall starts at the backbone's clean accuracy
          and decays with storm severity, weighted by how fragile that class's
          visual signature is and by how much representational capacity the
          backbone has to spare:

              recall_c = clean_acc * (1 - severity * fragility_c * (1 - capacity))

          A high-capacity backbone (ResNet, capacity 0.94) barely moves. A
          low-capacity one (SqueezeNet, 0.34) collapses — which is exactly the
          false-negative spike the course brief describes.

      2.  The remaining probability mass is distributed over the wrong classes
          according to the physical confusability prior above, so the confusion
          matrix develops *structured* off-diagonal mass rather than uniform
          grey.

      3.  The sampled prediction is turned into a plausible softmax vector by
          sharpening with the backbone's confidence temperature. Temperature
          > 1 (ResNet) yields overconfident outputs and therefore a measurable
          Expected Calibration Error, which is the Module 5 payoff.
    """

    def __init__(self, spec: ModelSpec, rng: np.random.Generator | None = None) -> None:
        self.spec = spec
        self.rng = rng or np.random.default_rng()

    def class_posterior(self, true_class: int, severity: float) -> np.ndarray:
        """P(predicted = j | true = i) under the current storm severity."""
        severity = float(np.clip(severity, 0.0, 1.0))
        fragility = _STORM_FRAGILITY[true_class]
        headroom = 1.0 - self.spec.capacity

        recall = self.spec.clean_accuracy * (1.0 - severity * fragility * headroom)
        # Never let a model fall below chance; a 5-way chance floor is 0.20.
        recall = float(np.clip(recall, 0.20, 0.995))

        posterior = _CONFUSABILITY[true_class].copy()
        total = posterior.sum()
        if total <= 0:
            posterior = np.full(NUM_CLASSES, 1.0 / (NUM_CLASSES - 1))
            posterior[true_class] = 0.0
            total = posterior.sum()
        posterior = posterior / total * (1.0 - recall)
        posterior[true_class] = recall
        return posterior

    def decide(
        self,
        true_class: int,
        severity: float,
        logits: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Sample a prediction and synthesise its softmax distribution.

        The real network's logits are not thrown away: they are normalised and
        used both to break ties between equally-likely confusions and to add
        pixel-dependent jitter to the output distribution. Two frames of the
        same class therefore produce *different* softmax vectors, as they
        would from a real classifier.
        """
        posterior = self.class_posterior(true_class, severity)

        # Fold in the real network's response as a mild, pixel-dependent tilt.
        if logits.size == NUM_CLASSES:
            tilt = logits - logits.max()
            tilt = np.exp(tilt)
            tilt = tilt / max(tilt.sum(), 1e-9)
            # 12% weight: enough to make the output depend on the actual image,
            # small enough that the capacity model remains in charge.
            posterior = 0.88 * posterior + 0.12 * tilt
            posterior = posterior / posterior.sum()

        pred_class = int(self.rng.choice(NUM_CLASSES, p=posterior))

        # Build the reported softmax: the posterior, re-centred on whatever we
        # actually predicted, then sharpened by the calibration temperature.
        reported = posterior.copy()
        reported[pred_class] = max(reported[pred_class], posterior.max())
        reported = reported / reported.sum()

        t = self.spec.confidence_temperature
        sharpened = np.power(np.clip(reported, 1e-9, 1.0), t)
        sharpened = sharpened / sharpened.sum()

        # A small Dirichlet perturbation keeps successive frames from reporting
        # byte-identical confidences, which would look fake on a live gauge.
        noise = self.rng.dirichlet(np.full(NUM_CLASSES, 48.0))
        sharpened = 0.93 * sharpened + 0.07 * noise
        sharpened = sharpened / sharpened.sum()

        return pred_class, sharpened


# ─────────────────────────────────────────────────────────────────────────────
#  The zoo
# ─────────────────────────────────────────────────────────────────────────────


class EdgeModelZoo:
    """Lazily-loaded registry of the three switchable backbones.

    Models are built on first use rather than at import, because constructing
    all three plus profiling them takes a couple of seconds and we would rather
    the server start instantly and warm up on demand.
    """

    def __init__(self, pretrained: bool | None = None, seed: int | None = None) -> None:
        if pretrained is None:
            pretrained = os.environ.get("DCV_PRETRAINED", "0") == "1"
        self.pretrained = pretrained
        self.device = torch.device("cpu")

        self._models: dict[str, nn.Module] = {}
        self._profiles: dict[str, ArchitectureProfile] = {}
        self._latency: dict[str, LatencyProfile] = {}
        self._deciders: dict[str, CapacityDecisionModel] = {}
        self._weights_loaded: dict[str, bool] = {}
        self._lock = threading.Lock()

        self.rng = np.random.default_rng(seed)

        # Single-threaded inference. An inspection rover has one core to spare;
        # letting PyTorch fan out across 16 desktop cores would flatter the
        # heavy model and quietly dissolve the latency argument.
        try:
            torch.set_num_threads(1)
        except Exception:  # pragma: no cover - defensive
            pass
        torch.set_grad_enabled(False)

    # -- construction ---------------------------------------------------------

    def _build(self, spec: ModelSpec) -> tuple[nn.Module, bool]:
        """Instantiate a torchvision backbone and re-head it for 5 classes."""
        import torchvision.models as tvm

        factory = getattr(tvm, spec.torchvision_fn)
        loaded_pretrained = False

        if self.pretrained:
            try:
                model = factory(weights="DEFAULT")
                loaded_pretrained = True
            except Exception as exc:  # network unavailable, checksum, etc.
                print(
                    f"[zoo] pretrained weights unavailable for {spec.key} "
                    f"({type(exc).__name__}); falling back to random init"
                )
                model = factory(weights=None)
        else:
            model = factory(weights=None)

        model = self._replace_head(model, spec)
        model.eval().to(self.device)
        return model, loaded_pretrained

    @staticmethod
    def _replace_head(model: nn.Module, spec: ModelSpec) -> nn.Module:
        """Swap the ImageNet-1000 classifier for our 5-way inspection head.

        Each family hides its head in a different place, which is itself worth
        knowing: SqueezeNet classifies with a 1x1 convolution (no Linear layer
        at all), MobileNetV2 ends in a Dropout+Linear sequence, and ResNet
        exposes a bare `fc`.
        """
        if spec.key == "squeezenet":
            final_conv = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
            nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(final_conv.bias, 0.0)
            model.classifier[1] = final_conv
            model.num_classes = NUM_CLASSES
        elif spec.key == "mobilenet":
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        elif spec.key == "resnet":
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    # -- accessors ------------------------------------------------------------

    def get(self, key: str) -> nn.Module:
        """Return (building if needed) the backbone for `key`."""
        spec = self.spec(key)
        with self._lock:
            if key not in self._models:
                model, loaded = self._build(spec)
                self._models[key] = model
                self._weights_loaded[key] = loaded
                self._deciders[key] = CapacityDecisionModel(spec, self.rng)
            return self._models[key]

    @staticmethod
    def spec(key: str) -> ModelSpec:
        if key not in MODEL_SPECS:
            raise KeyError(
                f"unknown model '{key}'. Available: {', '.join(MODEL_SPECS)}"
            )
        return MODEL_SPECS[key]

    def decider(self, key: str) -> CapacityDecisionModel:
        self.get(key)
        return self._deciders[key]

    def profile(self, key: str) -> ArchitectureProfile:
        """Static architecture profile (params, MACs, layer shapes)."""
        with self._lock:
            cached = self._profiles.get(key)
        if cached is not None:
            return cached
        model = self.get(key)
        prof = profile_architecture(model, key)
        with self._lock:
            self._profiles[key] = prof
        return prof

    def latency(self, key: str, runs: int = 20, warmup: int = 5) -> LatencyProfile:
        """Measured latency distribution on *this* machine."""
        with self._lock:
            cached = self._latency.get(key)
        if cached is not None:
            return cached
        model = self.get(key)
        prof = benchmark_latency(model, runs=runs, warmup=warmup, device=self.device)
        with self._lock:
            self._latency[key] = prof
        return prof

    def weights_are_pretrained(self, key: str) -> bool:
        self.get(key)
        return self._weights_loaded.get(key, False)

    # -- inference ------------------------------------------------------------

    def infer(
        self,
        key: str,
        image: np.ndarray,
        true_class: int,
        storm_severity: float = 0.0,
    ) -> InferenceResult:
        """Run one real forward pass and resolve the decision.

        `image` is HxWx3 uint8 at 224x224. Normalisation uses the ImageNet
        statistics the backbones were designed around.
        """
        spec = self.spec(key)
        model = self.get(key)
        tensor = self.preprocess(image)

        start = time.perf_counter()
        with torch.no_grad():
            logits_t = model(tensor)
        measured_ms = (time.perf_counter() - start) * 1000.0

        logits = logits_t.detach().cpu().numpy().reshape(-1)[:NUM_CLASSES]
        if logits.size < NUM_CLASSES:  # pragma: no cover - defensive
            logits = np.pad(logits, (0, NUM_CLASSES - logits.size))

        pred_class, probs = self.decider(key).decide(true_class, storm_severity, logits)

        p_true = float(np.clip(probs[true_class], 1e-9, 1.0))
        cross_entropy = -math.log(p_true)

        return InferenceResult(
            model_key=key,
            true_class=int(true_class),
            pred_class=int(pred_class),
            probs=[float(p) for p in probs],
            confidence=float(probs[pred_class]),
            measured_latency_ms=measured_ms,
            nominal_latency_ms=spec.nominal_latency_ms,
            correct=bool(pred_class == true_class),
            cross_entropy=cross_entropy,
            logits=[float(x) for x in logits],
            storm_severity=float(storm_severity),
        )

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """uint8 HxWx3 -> normalised float32 1x3x224x224."""
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
        return (tensor - self._MEAN) / self._STD

    # -- summaries ------------------------------------------------------------

    def describe(self, key: str, include_layers: bool = False) -> dict[str, Any]:
        """Full telemetry card for one backbone, as sent to the frontend."""
        spec = self.spec(key)
        prof = self.profile(key)
        lat = self.latency(key)

        macs_per_ms = prof.total_macs / max(lat.mean_ms, 1e-6)

        return {
            "key": spec.key,
            "display_name": spec.display_name,
            "family": spec.family,
            "accent": spec.accent,
            "blurb": spec.blurb,
            "architecture_note": spec.architecture_note,
            "signature_block": spec.signature_block,
            "clean_accuracy": spec.clean_accuracy,
            "capacity": spec.capacity,
            "power_watts": spec.power_watts,
            "confidence_temperature": spec.confidence_temperature,
            "nominal_latency_ms": spec.nominal_latency_ms,
            "nominal_throughput_fps": round(spec.nominal_throughput_fps, 2),
            "pretrained": self.weights_are_pretrained(key),
            "measured_latency": lat.to_dict(),
            "architecture": prof.to_dict(include_layers=include_layers),
            # MACs actually retired per millisecond: the number that explains
            # why SqueezeNet beats MobileNetV2 on the clock despite doing MORE
            # arithmetic. Depthwise convolutions are bandwidth-bound.
            "mac_efficiency_m_per_ms": round(macs_per_ms / 1e6, 1),
        }

    def describe_all(self) -> list[dict[str, Any]]:
        return [self.describe(k) for k in MODEL_SPECS]

    def pareto_points(self) -> list[dict[str, Any]]:
        """The static Accuracy / Latency / Power frontier for Module 5."""
        points = []
        for key, spec in MODEL_SPECS.items():
            prof = self.profile(key)
            points.append(
                {
                    "key": key,
                    "display_name": spec.display_name,
                    "accent": spec.accent,
                    "latency_ms": spec.nominal_latency_ms,
                    "accuracy": spec.clean_accuracy,
                    "power_watts": spec.power_watts,
                    "params_m": round(prof.total_params / 1e6, 2),
                    "gflops": round(prof.total_flops / 1e9, 3),
                    "throughput_fps": round(spec.nominal_throughput_fps, 1),
                    # Does this model survive a 40 ms frame budget at all?
                    "meets_budget": spec.nominal_latency_ms < 40.0,
                }
            )
        return points


# ─────────────────────────────────────────────────────────────────────────────
#  Process-wide singleton
# ─────────────────────────────────────────────────────────────────────────────

_ZOO: EdgeModelZoo | None = None
_ZOO_LOCK = threading.Lock()


def get_zoo(seed: int | None = None) -> EdgeModelZoo:
    """Return the process-wide zoo, constructing it on first call."""
    global _ZOO
    if _ZOO is None:
        with _ZOO_LOCK:
            if _ZOO is None:
                _ZOO = EdgeModelZoo(seed=seed)
    return _ZOO


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary parts (for reproducible runs)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")
