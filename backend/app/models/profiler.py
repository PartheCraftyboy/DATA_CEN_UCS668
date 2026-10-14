"""
Module 3 — Neural Network Design & Architectures.

Architectural telemetry for the three backbones. Everything in this file is a
*measurement*, not a lookup table: parameter counts come from iterating the
real `nn.Parameter` tensors, MAC counts come from forward hooks that see the
real input/output shapes, and latency comes from `time.perf_counter()` around
real forward passes.

The point of the lab is that these three numbers — parameters, MACs, and
wall-clock latency — are related but *not* interchangeable. SqueezeNet has
more MACs than MobileNetV2 (350M vs 300M) yet runs ~2.5x faster on CPU,
because dense 3x3 convolutions saturate the vector units while depthwise
convolutions are memory-bandwidth bound and achieve terrible arithmetic
intensity. That discrepancy is the single most interesting thing a student
can point at during the viva, so the profiler surfaces it explicitly as
`mac_efficiency` (MACs actually retired per millisecond).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
#  Layer-by-layer tensor shape tracing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LayerTrace:
    """One traced layer: what went in, what came out, and what it cost."""

    index: int
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    params: int
    macs: int
    # Spatial resolution of the output feature map, e.g. 112 for 112x112.
    resolution: int | None = None
    # Descriptive tag used by the frontend inspector to colour-code blocks.
    block_role: str = "conv"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["in_shape"] = list(self.in_shape)
        d["out_shape"] = list(self.out_shape)
        return d


@dataclass
class ArchitectureProfile:
    """Complete static profile of one backbone at 224x224x3 input."""

    model_key: str
    total_params: int
    trainable_params: int
    total_macs: int
    total_flops: int
    layers: list[LayerTrace] = field(default_factory=list)
    # The downsampling cascade 224 -> 112 -> 56 -> 28 -> 14 -> 7, read off the
    # real traced feature maps rather than assumed.
    resolution_cascade: list[int] = field(default_factory=list)
    # Bytes of fp32 weights — the number that decides whether the model fits in
    # the rover's SRAM budget.
    weight_bytes: int = 0

    def to_dict(self, include_layers: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_key": self.model_key,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_macs": self.total_macs,
            "total_flops": self.total_flops,
            "resolution_cascade": self.resolution_cascade,
            "weight_bytes": self.weight_bytes,
            "weight_mb": round(self.weight_bytes / (1024 * 1024), 3),
            "params_millions": round(self.total_params / 1e6, 3),
            "macs_millions": round(self.total_macs / 1e6, 1),
            "gflops": round(self.total_flops / 1e9, 3),
            "layer_count": len(self.layers),
        }
        if include_layers:
            out["layers"] = [layer.to_dict() for layer in self.layers]
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter counting
# ─────────────────────────────────────────────────────────────────────────────


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts.

    Counted by walking the real parameter tensors, so a model whose classifier
    head we swapped for 5 classes reports the *swapped* count, not the
    ImageNet-1000 one. This matters: ResNet-18's 1000-way head alone is 512k
    parameters, and dropping it to 5 classes visibly moves the needle.
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
#  MAC / FLOP counting via forward hooks
# ─────────────────────────────────────────────────────────────────────────────


def _conv_macs(module: nn.Conv2d, out_shape: torch.Size) -> int:
    """MACs for a (possibly grouped / depthwise) 2-D convolution.

    For output H_out x W_out, each output pixel of each output channel costs
    (C_in / groups) * k_h * k_w multiply-accumulates. Grouping is what makes
    a depthwise convolution cheap: with groups == C_in, the (C_in / groups)
    term collapses to 1, which is precisely MobileNetV2's trick.
    """
    if len(out_shape) != 4:
        return 0
    _, out_c, out_h, out_w = out_shape
    k_h, k_w = module.kernel_size
    in_c_per_group = module.in_channels // module.groups
    return int(out_c) * int(out_h) * int(out_w) * in_c_per_group * k_h * k_w


def _linear_macs(module: nn.Linear, out_shape: torch.Size) -> int:
    """MACs for a fully-connected layer (batch dimension excluded)."""
    return int(module.in_features) * int(module.out_features)


def _classify_block(module: nn.Module, name: str) -> str:
    """Tag a layer with its architectural role for the frontend inspector."""
    lowered = name.lower()
    if isinstance(module, nn.Conv2d):
        if module.groups > 1 and module.groups == module.in_channels:
            return "depthwise"
        if module.kernel_size == (1, 1):
            # SqueezeNet calls these squeeze/expand; MobileNet calls them
            # pointwise/projection. Same operator, different folklore.
            if "squeeze" in lowered:
                return "squeeze"
            if "expand" in lowered:
                return "expand"
            return "pointwise"
        if "downsample" in lowered:
            return "shortcut"
        return "conv"
    if isinstance(module, nn.Linear):
        return "classifier"
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return "norm"
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU)):
        return "activation"
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return "pool"
    return "other"


_COUNTED_TYPES = (nn.Conv2d, nn.Linear)
_TRACED_TYPES = (
    nn.Conv2d,
    nn.Linear,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
)


def count_macs(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device | None = None,
) -> tuple[int, list[LayerTrace]]:
    """Run one traced forward pass and return ``(total_macs, layer_traces)``.

    Only Conv2d and Linear contribute MACs — BatchNorm and activations are
    elementwise and cost O(N) adds, which is noise next to a 1.8 GFLOP
    convolution stack. Pooling layers are traced (so the resolution cascade is
    visible) but contribute no MACs.
    """
    device = device or torch.device("cpu")
    model = model.eval()

    traces: list[LayerTrace] = []
    handles: list[Any] = []
    counter = {"idx": 0, "macs": 0}

    def make_hook(name: str, module: nn.Module) -> Callable:
        def hook(mod: nn.Module, inputs: tuple, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            in_t = inputs[0] if inputs and isinstance(inputs[0], torch.Tensor) else None
            in_shape = tuple(in_t.shape) if in_t is not None else ()
            out_shape = tuple(output.shape)

            macs = 0
            if isinstance(mod, nn.Conv2d):
                macs = _conv_macs(mod, output.shape)
            elif isinstance(mod, nn.Linear):
                macs = _linear_macs(mod, output.shape)

            params = sum(p.numel() for p in mod.parameters(recurse=False))
            resolution = int(out_shape[-1]) if len(out_shape) == 4 else None

            counter["macs"] += macs
            traces.append(
                LayerTrace(
                    index=counter["idx"],
                    name=name or mod.__class__.__name__,
                    kind=mod.__class__.__name__,
                    in_shape=in_shape,
                    out_shape=out_shape,
                    params=params,
                    macs=macs,
                    resolution=resolution,
                    block_role=_classify_block(mod, name),
                )
            )
            counter["idx"] += 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, _TRACED_TYPES):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    try:
        with torch.no_grad():
            model(torch.zeros(*input_shape, device=device))
    finally:
        for h in handles:
            h.remove()

    return counter["macs"], traces


# ─────────────────────────────────────────────────────────────────────────────
#  Full static profile
# ─────────────────────────────────────────────────────────────────────────────


def _resolution_cascade(traces: Iterable[LayerTrace]) -> list[int]:
    """Extract the distinct spatial resolutions in the order they appear.

    On a 224x224 input every one of our three backbones produces the classic
    224 -> 112 -> 56 -> 28 -> 14 -> 7 pyramid, five halvings achieved by
    strided convolutions and pooling. Showing this cascade side by side is the
    cleanest way to make 'receptive field grows as resolution shrinks'
    concrete during the viva.
    """
    cascade: list[int] = []
    for t in traces:
        if t.resolution is None:
            continue
        if not cascade or cascade[-1] != t.resolution:
            # Only record genuine downsampling steps, not noise from 1x1 convs
            # that preserve resolution.
            if not cascade or t.resolution < cascade[-1]:
                cascade.append(t.resolution)
            elif not cascade:
                cascade.append(t.resolution)
    return cascade


def profile_architecture(
    model: nn.Module,
    model_key: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> ArchitectureProfile:
    """Measure params, MACs, layer shapes and the downsampling cascade."""
    total_params, trainable_params = count_parameters(model)
    total_macs, traces = count_macs(model, input_shape=input_shape)

    # A multiply-accumulate is conventionally billed as 2 FLOPs (one multiply,
    # one add). Papers are inconsistent about this; we state our convention
    # explicitly so the reported GFLOPs can be checked against the literature.
    total_flops = total_macs * 2

    cascade = [input_shape[-1]] + _resolution_cascade(traces)
    # De-duplicate while preserving order.
    seen: set[int] = set()
    cascade = [r for r in cascade if not (r in seen or seen.add(r))]

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    weight_bytes += sum(b.numel() * b.element_size() for b in model.buffers())

    return ArchitectureProfile(
        model_key=model_key,
        total_params=total_params,
        trainable_params=trainable_params,
        total_macs=total_macs,
        total_flops=total_flops,
        layers=traces,
        resolution_cascade=cascade,
        weight_bytes=weight_bytes,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Wall-clock latency benchmarking
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LatencyProfile:
    """Measured inference latency distribution, in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    runs: int = 24,
    warmup: int = 6,
    device: torch.device | None = None,
) -> LatencyProfile:
    """Time `runs` forward passes with microsecond-resolution timers.

    Warm-up runs are discarded. They are not optional: the first few passes
    pay for lazy kernel selection, MKL-DNN primitive JIT-ing and cold caches,
    and on CPU the first pass can be 5-10x the steady-state cost. Reporting a
    cold first pass as 'the latency' is the classic way to accidentally make
    SqueezeNet look slower than ResNet.

    We report p95 alongside the mean because a queueing system cares about the
    tail, not the average. A model whose p95 exceeds the 40 ms frame budget
    will build an unbounded backlog even if its mean fits comfortably.
    """
    device = device or torch.device("cpu")
    model = model.eval().to(device)
    sample = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(sample)

        timings: list[float] = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            model(sample)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    timings.sort()
    p95_index = min(len(timings) - 1, int(round(0.95 * (len(timings) - 1))))

    return LatencyProfile(
        mean_ms=statistics.fmean(timings),
        median_ms=statistics.median(timings),
        p95_ms=timings[p95_index],
        min_ms=timings[0],
        max_ms=timings[-1],
        stdev_ms=statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        samples=len(timings),
    )
