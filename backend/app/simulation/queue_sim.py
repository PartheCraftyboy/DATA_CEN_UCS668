"""
The M/D/1/K edge queue — the mechanism the whole project exists to demonstrate.

A camera on an inspection rover emits frames at λ = 25 FPS. The edge compute
module has one worker and a finite FIFO buffer of N_max = 10 frames. Each frame
takes a *deterministic* service time:

    T_service = T_infer(model) + T_preprocess(filters)

M/D/1 in Kendall notation: Markovian (Poisson) arrivals, Deterministic service,
1 server. The /K suffix marks the finite buffer, which is what makes this a
*lossy* system rather than merely a slow one.

The load factor is

    ρ = λ / μ = λ · T_service

and the entire lesson is the discontinuity at ρ = 1:

    ρ < 1   the backlog is bounded; latency is stable
    ρ ≥ 1   the backlog grows without bound until the buffer saturates,
            after which every excess frame is DROPPED — not delayed, dropped.
            A dropped frame is a rack that was never inspected.

With ResNet-18 at 44 ms, ρ = 25 × 0.044 = 1.10. The model is 97.6% accurate on
the frames it sees, and it is a system-level failure, because it does not see
them. That sentence is the viva answer.

Note on Poisson arrivals: the frame rate is nominally fixed, but arrivals are
modelled as Poisson rather than periodic because jitter in capture, encode and
transport makes inter-arrival times effectively memoryless at the compute node.
It is also the pessimistic assumption, and queues should be sized against the
pessimistic case — periodic arrivals into a deterministic server would queue
nothing at all until ρ hit exactly 1, which would be a less interesting and
less realistic system.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

import numpy as np

from app import CLASS_META, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QueueConfig:
    """Tunables for the edge queue and the scoring game layered on top."""

    arrival_rate_fps: float = 25.0   # λ
    buffer_capacity: int = 10        # N_max
    frame_budget_ms: float = 40.0    # 1000 / λ — the SLA deadline per frame

    # Health economics. The rack health bar is the player's life total.
    max_health: float = 100.0
    health_per_drop: float = 2.6     # cost of an uninspected rack
    health_per_missed_defect: float = 1.0  # multiplied by the class miss_cost
    health_per_false_alarm: float = 0.35   # crying wolf wastes technician time
    health_regen_per_s: float = 1.4        # clean operation slowly recovers

    # Scoring.
    score_per_correct_defect: float = 12.0
    score_per_correct_optimal: float = 1.0
    score_penalty_missed: float = 8.0
    score_penalty_false_alarm: float = 2.0
    score_penalty_drop: float = 6.0

    # A frame whose total system time (queueing + service) exceeds this is an
    # SLA breach even though it was eventually processed.
    sla_latency_ms: float = 80.0


# ─────────────────────────────────────────────────────────────────────────────
#  Frame tickets
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameTicket:
    """One frame's journey through the edge pipeline."""

    frame_id: int
    arrived_at: float            # simulation clock, seconds
    true_class: int
    subtlety: float
    started_service_at: float | None = None
    completed_at: float | None = None
    service_time_ms: float = 0.0
    model_key: str = ""
    pred_class: int | None = None
    confidence: float = 0.0
    dropped: bool = False

    @property
    def queue_delay_ms(self) -> float:
        if self.started_service_at is None:
            return 0.0
        return (self.started_service_at - self.arrived_at) * 1000.0

    @property
    def system_time_ms(self) -> float:
        """Total time in system: queueing delay + service. The real latency."""
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.arrived_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "true_class": self.true_class,
            "pred_class": self.pred_class,
            "confidence": round(self.confidence, 4),
            "model_key": self.model_key,
            "dropped": self.dropped,
            "queue_delay_ms": round(self.queue_delay_ms, 2),
            "service_time_ms": round(self.service_time_ms, 3),
            "system_time_ms": round(self.system_time_ms, 2),
            "subtlety": round(self.subtlety, 3),
        }


@dataclass
class QueueSnapshot:
    """Instantaneous queue telemetry, pushed to the cockpit every tick."""

    depth: int
    capacity: int
    utilisation: float            # ρ, the load factor
    service_time_ms: float
    throughput_fps: float
    arrival_rate_fps: float
    drops_total: int
    drops_recent: int
    processed_total: int
    health: float
    score: float
    mean_system_time_ms: float
    p95_system_time_ms: float
    sla_breaches: int
    overflow: bool
    saturated: bool               # ρ >= 1: backlog grows without bound
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "capacity": self.capacity,
            "fill_ratio": round(self.depth / max(self.capacity, 1), 4),
            "utilisation": round(self.utilisation, 4),
            "service_time_ms": round(self.service_time_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
            "arrival_rate_fps": round(self.arrival_rate_fps, 2),
            "drops_total": self.drops_total,
            "drops_recent": self.drops_recent,
            "processed_total": self.processed_total,
            "health": round(self.health, 2),
            "score": round(self.score, 1),
            "mean_system_time_ms": round(self.mean_system_time_ms, 2),
            "p95_system_time_ms": round(self.p95_system_time_ms, 2),
            "sla_breaches": self.sla_breaches,
            "overflow": self.overflow,
            "saturated": self.saturated,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Simulator
# ─────────────────────────────────────────────────────────────────────────────


class EdgeQueueSimulator:
    """Finite-buffer M/D/1 queue with a health/score game layered over it.

    Driven by `tick(dt, service_time_ms, ...)`. The caller owns the clock, so
    the simulation runs identically whether it is stepped by a 25 Hz asyncio
    loop or fast-forwarded in a unit test.
    """

    def __init__(self, config: QueueConfig | None = None, seed: int | None = None) -> None:
        self.config = config or QueueConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        cfg = self.config
        self.clock: float = 0.0
        self.frame_counter: int = 0

        self.buffer: Deque[FrameTicket] = deque()
        self.in_service: FrameTicket | None = None
        self.service_remaining_s: float = 0.0

        self.health: float = cfg.max_health
        self.score: float = 0.0

        self.drops_total: int = 0
        self.drops_recent: int = 0
        self.processed_total: int = 0
        self.sla_breaches: int = 0
        self.missed_defects: int = 0
        self.false_alarms: int = 0

        # Rolling window of system times for the latency distribution readout.
        self._system_times: Deque[float] = deque(maxlen=240)
        self._completion_times: Deque[float] = deque(maxlen=240)
        self._last_service_ms: float = 0.0
        self.game_over: bool = False

    # -- stepping -------------------------------------------------------------

    def tick(
        self,
        dt: float,
        service_time_ms: float,
        model_key: str,
        sample_class: Any,
    ) -> dict[str, Any]:
        """Advance the simulation by `dt` seconds.

        `sample_class` is a zero-argument callable returning `(true_class,
        subtlety)` for each newly arrived frame — supplied by the caller so the
        queue never needs to know about image synthesis.

        Returns the events that occurred in this slice: arrivals accepted,
        frames dropped, and frames whose service completed (which the caller
        must then actually classify).
        """
        cfg = self.config
        self.clock += dt
        self._last_service_ms = service_time_ms
        service_time_s = max(service_time_ms, 1e-4) / 1000.0

        arrivals: list[FrameTicket] = []
        dropped: list[FrameTicket] = []
        completed: list[FrameTicket] = []

        # ---- 1. Arrivals: Poisson process with mean λ·dt in this slice ------
        n_arrivals = int(self.rng.poisson(cfg.arrival_rate_fps * dt))
        for i in range(n_arrivals):
            true_class, subtlety = sample_class()
            # Spread arrivals uniformly across the slice rather than stamping
            # them all at the slice boundary; otherwise queueing delay is
            # quantised to the tick rate and the latency histogram lies.
            arrival_t = self.clock - dt + (i + 1) / (n_arrivals + 1) * dt

            ticket = FrameTicket(
                frame_id=self.frame_counter,
                arrived_at=arrival_t,
                true_class=true_class,
                subtlety=subtlety,
                model_key=model_key,
            )
            self.frame_counter += 1

            occupancy = len(self.buffer) + (1 if self.in_service else 0)
            if occupancy >= cfg.buffer_capacity:
                # BUFFER OVERFLOW. The frame is gone; the rack it showed is
                # never inspected. This is the failure mode the project is
                # about, and it is silent — nothing errors, the pipeline just
                # stops seeing things.
                ticket.dropped = True
                dropped.append(ticket)
                self.drops_total += 1
                self.drops_recent += 1
                self._apply_drop_cost()
            else:
                self.buffer.append(ticket)
                arrivals.append(ticket)

        # ---- 2. Service: deterministic, single server -----------------------
        remaining_dt = dt
        guard = 0
        while remaining_dt > 1e-9 and guard < 512:
            guard += 1

            if self.in_service is None:
                if not self.buffer:
                    break
                self.in_service = self.buffer.popleft()
                # Service cannot begin before the frame arrived. Arrivals are
                # spread across the slice, so a frame that landed mid-slice can
                # reach the head of an empty queue while `remaining_dt` still
                # points at the slice boundary — without this clamp its queueing
                # delay comes out negative.
                self.in_service.started_service_at = max(
                    self.clock - remaining_dt, self.in_service.arrived_at
                )
                self.in_service.service_time_ms = service_time_ms
                self.in_service.model_key = model_key
                self.service_remaining_s = service_time_s

            step = min(remaining_dt, self.service_remaining_s)
            self.service_remaining_s -= step
            remaining_dt -= step

            if self.service_remaining_s <= 1e-9:
                ticket = self.in_service
                # Likewise, completion is at least service-start plus the
                # service time, never earlier than the frame entered service.
                ticket.completed_at = max(
                    self.clock - remaining_dt, (ticket.started_service_at or 0.0)
                )
                self.in_service = None
                self.processed_total += 1

                st = ticket.system_time_ms
                self._system_times.append(st)
                self._completion_times.append(ticket.completed_at)
                if st > cfg.sla_latency_ms:
                    self.sla_breaches += 1

                completed.append(ticket)

        # ---- 3. Health regeneration ----------------------------------------
        if self.health > 0:
            self.health = min(cfg.max_health, self.health + cfg.health_regen_per_s * dt)
        if self.health <= 0:
            self.health = 0.0
            self.game_over = True

        return {
            "arrivals": arrivals,
            "dropped": dropped,
            "completed": completed,
            "n_arrivals": n_arrivals,
        }

    # -- outcome accounting ---------------------------------------------------

    def register_classification(
        self, ticket: FrameTicket, pred_class: int, confidence: float
    ) -> dict[str, Any]:
        """Score one completed inference and apply its health consequences.

        Three outcome classes, priced very differently:

          · Correct detection of a defect — the system did its job. Scored in
            proportion to how expensive that defect would have been to miss.
          · MISSED DEFECT (false negative) — a real fault called Optimal. This
            is the expensive failure, and it is what a low-capacity backbone
            produces under storms.
          · False alarm (false positive) — a healthy rack flagged. Cheap, but
            not free: it dispatches a technician for nothing, and a system that
            cries wolf gets ignored.
        """
        cfg = self.config
        ticket.pred_class = pred_class
        ticket.confidence = confidence

        true_is_defect = ticket.true_class != 0
        pred_is_defect = pred_class != 0
        correct = pred_class == ticket.true_class

        meta = CLASS_META[ticket.true_class]
        outcome = "correct"
        health_delta = 0.0
        score_delta = 0.0

        if correct:
            if true_is_defect:
                score_delta = cfg.score_per_correct_defect * (0.5 + meta["severity"])
                outcome = "detected"
            else:
                score_delta = cfg.score_per_correct_optimal
                outcome = "nominal"
        elif true_is_defect and not pred_is_defect:
            # The one that hurts.
            self.missed_defects += 1
            health_delta = -cfg.health_per_missed_defect * meta["miss_cost"]
            score_delta = -cfg.score_penalty_missed
            outcome = "missed_defect"
        elif not true_is_defect and pred_is_defect:
            self.false_alarms += 1
            health_delta = -cfg.health_per_false_alarm
            score_delta = -cfg.score_penalty_false_alarm
            outcome = "false_alarm"
        else:
            # Defect detected, but mislabelled as the wrong defect. The rack
            # still gets a technician, so this is only mildly bad.
            score_delta = -cfg.score_penalty_false_alarm * 0.5
            outcome = "misclassified_defect"

        self.health = max(0.0, min(cfg.max_health, self.health + health_delta))
        self.score = max(0.0, self.score + score_delta)
        if self.health <= 0:
            self.game_over = True

        return {
            "outcome": outcome,
            "health_delta": round(health_delta, 3),
            "score_delta": round(score_delta, 2),
        }

    def _apply_drop_cost(self) -> None:
        cfg = self.config
        self.health = max(0.0, self.health - cfg.health_per_drop)
        self.score = max(0.0, self.score - cfg.score_penalty_drop)
        if self.health <= 0:
            self.game_over = True

    # -- telemetry ------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Current occupancy: buffered frames plus the one in service."""
        return len(self.buffer) + (1 if self.in_service else 0)

    def snapshot(self, service_time_ms: float | None = None) -> QueueSnapshot:
        cfg = self.config
        st_ms = service_time_ms if service_time_ms is not None else self._last_service_ms
        st_ms = max(st_ms, 1e-4)

        mu = 1000.0 / st_ms
        rho = cfg.arrival_rate_fps / mu

        times = list(self._system_times)
        mean_st = float(np.mean(times)) if times else 0.0
        p95_st = float(np.percentile(times, 95)) if len(times) >= 4 else mean_st

        # Observed throughput over the rolling completion window.
        if len(self._completion_times) >= 2:
            span = self._completion_times[-1] - self._completion_times[0]
            observed_fps = (len(self._completion_times) - 1) / span if span > 1e-6 else 0.0
        else:
            observed_fps = 0.0

        snap = QueueSnapshot(
            depth=self.depth,
            capacity=cfg.buffer_capacity,
            utilisation=rho,
            service_time_ms=st_ms,
            throughput_fps=observed_fps,
            arrival_rate_fps=cfg.arrival_rate_fps,
            drops_total=self.drops_total,
            drops_recent=self.drops_recent,
            processed_total=self.processed_total,
            health=self.health,
            score=self.score,
            mean_system_time_ms=mean_st,
            p95_system_time_ms=p95_st,
            sla_breaches=self.sla_breaches,
            overflow=self.depth >= cfg.buffer_capacity,
            saturated=rho >= 1.0,
            elapsed_s=self.clock,
        )
        self.drops_recent = 0
        return snap

    # -- closed-form theory ---------------------------------------------------

    def theory(self, service_time_ms: float | None = None) -> dict[str, Any]:
        """Closed-form queueing predictions to display beside the live gauge.

        The Pollaczek–Khinchine formula gives the mean queue length for M/G/1:

            L_q = (λ² · E[S²]) / (2 · (1 − ρ))

        For deterministic service, E[S²] = E[S]², so this collapses to the
        classic M/D/1 result:

            L_q = ρ² / (2 · (1 − ρ))

        — exactly half the M/M/1 backlog. Determinism is worth a factor of two,
        which is a nice secondary result: removing *variance* from service time
        halves your buffering requirement at no extra compute.

        For the finite buffer we also report the M/M/1/K blocking probability
        as an upper bound on loss, since the exact M/D/1/K blocking probability
        has no elementary closed form and M/M/1/K is pessimistic (more service
        variance means more blocking).
        """
        cfg = self.config
        st_ms = service_time_ms if service_time_ms is not None else self._last_service_ms
        st_ms = max(st_ms, 1e-4)

        mu = 1000.0 / st_ms
        rho = cfg.arrival_rate_fps / mu
        k = cfg.buffer_capacity

        if rho < 1.0:
            lq_md1 = (rho**2) / (2.0 * (1.0 - rho))
            lq_mm1 = (rho**2) / (1.0 - rho)
            wq_ms = (lq_md1 / cfg.arrival_rate_fps) * 1000.0
            stable = True
        else:
            lq_md1 = float("inf")
            lq_mm1 = float("inf")
            wq_ms = float("inf")
            stable = False

        # M/M/1/K blocking probability — an upper bound on our loss rate.
        if abs(rho - 1.0) < 1e-9:
            p_block = 1.0 / (k + 1)
        else:
            p_block = ((1 - rho) * rho**k) / (1 - rho ** (k + 1))

        return {
            "rho": round(rho, 4),
            "stable": stable,
            "service_rate_fps": round(mu, 2),
            "arrival_rate_fps": cfg.arrival_rate_fps,
            "lq_md1": None if math.isinf(lq_md1) else round(lq_md1, 3),
            "lq_mm1": None if math.isinf(lq_mm1) else round(lq_mm1, 3),
            "wq_ms": None if math.isinf(wq_ms) else round(wq_ms, 2),
            "expected_system_ms": None if math.isinf(wq_ms) else round(wq_ms + st_ms, 2),
            "blocking_probability_bound": round(float(np.clip(p_block, 0.0, 1.0)), 5),
            "buffer_capacity": k,
            "frame_budget_ms": cfg.frame_budget_ms,
            "budget_headroom_ms": round(cfg.frame_budget_ms - st_ms, 3),
            "note": (
                "L_q for M/D/1 is exactly half the M/M/1 value — deterministic "
                "service halves the backlog at identical ρ."
            ),
        }

    def summary(self) -> dict[str, Any]:
        """End-of-run report card."""
        total = self.processed_total + self.drops_total
        return {
            "frames_offered": total,
            "frames_processed": self.processed_total,
            "frames_dropped": self.drops_total,
            "drop_rate": round(self.drops_total / total, 4) if total else 0.0,
            "missed_defects": self.missed_defects,
            "false_alarms": self.false_alarms,
            "sla_breaches": self.sla_breaches,
            "final_health": round(self.health, 2),
            "final_score": round(self.score, 1),
            "elapsed_s": round(self.clock, 2),
            "game_over": self.game_over,
        }
