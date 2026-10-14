"""Module 2 & 3 — the switchable backbone zoo and the architecture profiler."""

from app.models.zoo import (
    MODEL_SPECS,
    ModelSpec,
    EdgeModelZoo,
    InferenceResult,
    get_zoo,
)
from app.models.profiler import (
    ArchitectureProfile,
    LayerTrace,
    profile_architecture,
    benchmark_latency,
    count_parameters,
    count_macs,
)

__all__ = [
    "MODEL_SPECS",
    "ModelSpec",
    "EdgeModelZoo",
    "InferenceResult",
    "get_zoo",
    "ArchitectureProfile",
    "LayerTrace",
    "profile_architecture",
    "benchmark_latency",
    "count_parameters",
    "count_macs",
]
