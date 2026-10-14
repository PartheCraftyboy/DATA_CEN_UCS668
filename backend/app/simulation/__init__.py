"""Module 1 & 4 — frame synthesis, environmental storms, and the edge queue."""

from app.simulation.generator import RackFrameGenerator, SyntheticFrame
from app.simulation.augmentations import (
    STORM_SPECS,
    FILTER_SPECS,
    StormBank,
    apply_storms,
    apply_filters,
)
from app.simulation.queue_sim import (
    EdgeQueueSimulator,
    QueueConfig,
    QueueSnapshot,
    FrameTicket,
)

__all__ = [
    "RackFrameGenerator",
    "SyntheticFrame",
    "STORM_SPECS",
    "FILTER_SPECS",
    "StormBank",
    "apply_storms",
    "apply_filters",
    "EdgeQueueSimulator",
    "QueueConfig",
    "QueueSnapshot",
    "FrameTicket",
]
