"""
FastAPI ASGI surface for the Data Center Edge-Vision Commander.

    WebSocket  /ws              25 FPS telemetry stream + bidirectional control
    GET        /api/health      liveness
    GET        /api/models      full telemetry card for all three backbones
    GET        /api/models/{k}  one backbone, including layer-by-layer traces
    GET        /api/classes     the five inspection targets
    GET        /api/augments    storm + filter catalogues
    GET        /api/pareto      static accuracy/latency/power frontier
    GET        /api/frame       one sample frame (optionally of a given class)
    GET        /api/samples     one blatant example of each class
    GET        /api/theory      closed-form M/D/1 queueing predictions

Run:
    cd backend && uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import CLASS_META, CLASS_NAMES, NUM_CLASSES, __version__
from app.metrics import MetricsEngine
from app.models import MODEL_SPECS, get_zoo
from app.models.zoo import DEFAULT_MODEL_KEY
from app.simulation import EdgeQueueSimulator, QueueConfig, RackFrameGenerator, StormBank
from app.simulation.augmentations import specs_payload

# ─────────────────────────────────────────────────────────────────────────────
#  Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

TICK_HZ = 25.0                 # λ — the camera's frame rate, and our tick rate
TICK_DT = 1.0 / TICK_HZ
THUMBNAIL_PX = 96              # conveyor thumbnails; see SyntheticFrame.to_png_b64
CONVEYOR_DEPTH = 14            # how many recent frames the belt shows

# ── Real-time budget ─────────────────────────────────────────────────────────
# The tick budget is 40 ms and the full pixel pipeline (synthesise → storm →
# filter → encode) costs ~15-50 ms per frame depending on storm load. Running
# it on every frame would blow the budget and the simulation would lag behind
# its own clock, which would invalidate every latency number on screen.
#
# The resolution is to separate the two things a frame is for:
#
#   DECISION path  — needs only (true_class, severity). Severity is a property
#                    of the storm configuration, not of any image, so it is
#                    computed analytically. Every frame goes through this, and
#                    the confusion matrix sees all of them.
#   VISUAL path    — needs real pixels. Only the frames the player actually
#                    sees on the conveyor get rendered, capped per tick.
#
# A sampled subset of the rendered frames additionally gets a real PyTorch
# forward pass, which is what refreshes the measured-latency readout. Queue
# arithmetic always uses the nominal spec latency, so none of this sampling
# changes the simulation's behaviour — only how often the "measured on this
# machine" numbers refresh.
RENDER_PER_TICK = 1            # frames fully rendered through the pixel pipeline
CALIBRATION_PERIOD_S = 2.0     # how often to re-measure real inference latency


app = FastAPI(
    title="Data Center Edge-Vision Commander",
    description=(
        "Edge inference latency vs. accuracy simulator. "
        "UCS668 — Edge AI and Robotics: Data Center Vision."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    # The Astro dev server runs on 4321; the preview server on 4322.
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "http://localhost:4322",
        "http://127.0.0.1:4322",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Session
# ─────────────────────────────────────────────────────────────────────────────


class CockpitSession:
    """All mutable state for one connected cockpit.

    Each WebSocket connection gets its own session: its own queue, its own
    metrics, its own storm bank. The zoo (the loaded PyTorch models) is shared
    process-wide, because loading three backbones per connection would be
    wasteful and they are stateless at inference time.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.zoo = get_zoo()
        self.generator = RackFrameGenerator(seed=seed)
        self.queue = EdgeQueueSimulator(QueueConfig(), seed=seed)
        self.metrics = MetricsEngine()
        self.storms = StormBank()
        self.rng = np.random.default_rng(seed)

        self.model_key: str = DEFAULT_MODEL_KEY
        self.paused: bool = False
        self.frame_index: int = 0

        # Most recent real measurement, reused between sampled inferences.
        self.last_measured_latency_ms: float = MODEL_SPECS[self.model_key].nominal_latency_ms
        self.last_filter_cost_measured_ms: float = 0.0
        self.last_severity: float = 0.0

        # Pending frames waiting to be classified when their service completes.
        self.conveyor: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        # Frames added since the last telemetry push. Only these go over the
        # wire — resending the whole 14-frame belt every tick would be ~250 KB
        # per frame of base64 PNG, or 6 MB/s, which stutters the animation it
        # is trying to drive. The cockpit keeps its own ring buffer.
        self._conveyor_delta: list[dict[str, Any]] = []

        # Most recently rendered frame, handed to the calibration task.
        self._calibration_frame: tuple[np.ndarray, int] | None = None

    # -- latency calibration --------------------------------------------------

    def calibrate(self) -> dict[str, Any] | None:
        """Run one real PyTorch forward pass to re-measure inference latency.

        Deliberately *not* on the tick path. A ResNet-18 forward pass costs
        ~50-100 ms on a lab CPU — more than two entire tick budgets — so
        running it inline would make the simulation loop miss its own deadline
        and every latency number on screen would be measuring the simulator's
        stutter rather than the model's cost. It is a calibration measurement,
        so it belongs on a calibration schedule.

        The queue uses nominal spec latency regardless, so this affects only
        the "measured on this machine" readout.
        """
        if self._calibration_frame is None:
            return None
        image, true_class = self._calibration_frame
        inference = self.zoo.infer(
            self.model_key, image, true_class=true_class, storm_severity=self.last_severity
        )
        self.last_measured_latency_ms = inference.measured_latency_ms
        return {
            "type": "calibration",
            "model_key": self.model_key,
            "measured_latency_ms": round(inference.measured_latency_ms, 3),
            "nominal_latency_ms": inference.nominal_latency_ms,
            "ratio": round(
                inference.measured_latency_ms / max(inference.nominal_latency_ms, 1e-6), 3
            ),
        }

    # -- configuration --------------------------------------------------------

    @property
    def service_time_ms(self) -> float:
        """T_service = T_infer(model) + T_preprocess(filters).

        Both terms use the *nominal* reference costs rather than this machine's
        measured times, so the queue dynamics are identical on every lab
        machine and the demo is reproducible. This matters more than it might
        sound: the bilateral filter costs 0.3 ms through OpenCV's SIMD path and
        16 ms through the NumPy fallback, so charging measured cost would mean
        the simulation overflowed its buffer on a machine without OpenCV and
        sailed through on one with it — same code, opposite conclusion.

        The real measured costs are still taken and reported separately
        (`last_filter_cost_measured_ms`), so the cockpit can show both and the
        gap between reference and reality stays visible.
        """
        return MODEL_SPECS[self.model_key].nominal_latency_ms + self.storms.nominal_filter_cost_ms

    def set_model(self, key: str) -> bool:
        if key in MODEL_SPECS:
            self.model_key = key
            return True
        return False

    def reset(self) -> None:
        self.queue.reset()
        self.metrics.reset()
        self.storms.clear()
        self.frame_index = 0
        self.conveyor.clear()
        self._conveyor_delta.clear()
        self.events.clear()
        self.last_filter_cost_measured_ms = 0.0
        self.last_severity = 0.0

    # -- frame lifecycle ------------------------------------------------------

    def _sample_class(self) -> tuple[int, float]:
        """Pick the ground truth for a newly arriving frame.

        We do NOT render the image here. Rendering every arrival, including the
        ones about to be dropped, would burn CPU on frames nobody ever sees —
        which is ironic given the subject matter, but also just wasteful. The
        image is synthesised lazily when the frame enters service.
        """
        weights = np.array([0.55, 0.11, 0.12, 0.12, 0.10])
        label = int(self.rng.choice(NUM_CLASSES, p=weights))
        subtlety = float(self.rng.beta(2.0, 3.0))
        return label, subtlety

    def step(self) -> dict[str, Any]:
        """Advance one tick and return the full telemetry payload."""
        if self.paused:
            return self.telemetry(events=[])

        service_ms = self.service_time_ms
        result = self.queue.tick(
            dt=TICK_DT,
            service_time_ms=service_ms,
            model_key=self.model_key,
            sample_class=self._sample_class,
        )

        events: list[dict[str, Any]] = []

        # ---- dropped frames -------------------------------------------------
        for ticket in result["dropped"]:
            events.append(
                {
                    "type": "drop",
                    "frame_id": ticket.frame_id,
                    "true_class": ticket.true_class,
                    "label": CLASS_NAMES[ticket.true_class],
                    "message": "BUFFER OVERFLOW — rack not inspected",
                }
            )
            self._push_conveyor(
                {
                    "frame_id": ticket.frame_id,
                    "true_class": ticket.true_class,
                    "pred_class": None,
                    "dropped": True,
                    "thumbnail": None,
                    "confidence": 0.0,
                    "outcome": "dropped",
                    "model_key": self.model_key,
                }
            )

        # ---- completed frames: classify, and render the visible ones --------
        # Severity is analytic — it depends on the storm configuration, not on
        # any particular image — so every frame can be decided whether or not
        # we spend the milliseconds to draw it.
        _, self.last_severity = self.storms.analytic_severity()
        rendered_this_tick = 0

        for ticket in result["completed"]:
            self.frame_index += 1
            # Render the most recent completions, so the belt shows what just
            # happened rather than a stale backlog.
            render = rendered_this_tick < RENDER_PER_TICK
            thumb: str | None = None
            descriptor: dict[str, Any] = {}

            if render:
                rendered_this_tick += 1
                frame = self.generator.generate(
                    label=ticket.true_class, subtlety=ticket.subtlety
                )
                processed, storm_info = self.storms.process(frame.image, self.rng)
                self.last_filter_cost_measured_ms = storm_info["filter_cost_ms"]
                thumb = frame.to_png_b64(quality_size=THUMBNAIL_PX)
                descriptor = frame.descriptor
            else:
                processed = None

            # Keep the most recent processed frame for the calibration task to
            # run a real forward pass over, off the hot path.
            if processed is not None:
                self._calibration_frame = (processed, ticket.true_class)

            decider = self.zoo.decider(self.model_key)
            pred_class, prob_vec = decider.decide(
                ticket.true_class, self.last_severity, np.zeros(NUM_CLASSES)
            )
            probs = [float(p) for p in prob_vec]
            confidence = probs[pred_class]
            cross_entropy = -float(np.log(max(probs[ticket.true_class], 1e-9)))

            scoring = self.queue.register_classification(ticket, pred_class, confidence)
            self.metrics.observe(
                true_class=ticket.true_class,
                pred_class=pred_class,
                confidence=confidence,
                cross_entropy=cross_entropy,
                model_key=self.model_key,
            )

            self._push_conveyor(
                {
                    "frame_id": ticket.frame_id,
                    "true_class": ticket.true_class,
                    "pred_class": pred_class,
                    "dropped": False,
                    "thumbnail": thumb,
                    "confidence": round(confidence, 4),
                    "probs": [round(p, 4) for p in probs],
                    "outcome": scoring["outcome"],
                    "model_key": self.model_key,
                    "system_time_ms": round(ticket.system_time_ms, 2),
                    "queue_delay_ms": round(ticket.queue_delay_ms, 2),
                    "severity": round(self.last_severity, 3),
                    "descriptor": descriptor,
                }
            )

            if scoring["outcome"] == "missed_defect":
                events.append(
                    {
                        "type": "missed_defect",
                        "frame_id": ticket.frame_id,
                        "true_class": ticket.true_class,
                        "label": CLASS_META[ticket.true_class]["label"],
                        "message": f"FALSE NEGATIVE — {CLASS_META[ticket.true_class]['label']} missed",
                    }
                )

        # ---- Pareto operating point ----------------------------------------
        spec = MODEL_SPECS[self.model_key]
        self.metrics.pareto.update(
            model_key=self.model_key,
            latency_ms=service_ms,
            accuracy=self.metrics.confusion.accuracy(),
            macro_f1=self.metrics.confusion.macro_f1(),
            power_watts=spec.power_watts,
            severity=self.last_severity,
            frames=self.metrics.frames_by_model.get(self.model_key, 0),
        )

        return self.telemetry(events=events)

    def _push_conveyor(self, item: dict[str, Any]) -> None:
        self.conveyor.append(item)
        self._conveyor_delta.append(item)
        if len(self.conveyor) > CONVEYOR_DEPTH:
            self.conveyor.pop(0)

    # -- payload --------------------------------------------------------------

    def telemetry(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        service_ms = self.service_time_ms
        snap = self.queue.snapshot(service_ms)
        spec = MODEL_SPECS[self.model_key]

        conveyor_delta = self._conveyor_delta
        self._conveyor_delta = []

        return {
            "type": "telemetry",
            "t": round(time.time(), 3),
            "paused": self.paused,
            "model": {
                "key": self.model_key,
                "display_name": spec.display_name,
                "accent": spec.accent,
                "family": spec.family,
                "nominal_latency_ms": spec.nominal_latency_ms,
                "measured_latency_ms": round(self.last_measured_latency_ms, 3),
                "power_watts": spec.power_watts,
                "capacity": spec.capacity,
            },
            "service": {
                "total_ms": round(service_ms, 3),
                "inference_ms": spec.nominal_latency_ms,
                "preprocess_ms": round(self.storms.nominal_filter_cost_ms, 3),
                "preprocess_measured_ms": round(self.last_filter_cost_measured_ms, 3),
                "budget_ms": self.queue.config.frame_budget_ms,
                "over_budget": service_ms > self.queue.config.frame_budget_ms,
            },
            "queue": snap.to_dict(),
            "theory": self.queue.theory(service_ms),
            "metrics": self.metrics.to_dict(),
            "storms": self.storms.to_dict(),
            "severity": round(self.last_severity, 4),
            # Delta, not the whole belt. The cockpit appends and trims to
            # `conveyor_depth` itself.
            "conveyor_new": conveyor_delta,
            "conveyor_depth": CONVEYOR_DEPTH,
            "events": events,
            "summary": self.queue.summary(),
        }

    def handshake(self) -> dict[str, Any]:
        """Everything the cockpit needs once, on connect."""
        return {
            "type": "hello",
            "version": __version__,
            "tick_hz": TICK_HZ,
            "classes": list(CLASS_META),
            "models": self.zoo.describe_all(),
            "augmentations": specs_payload(),
            "config": {
                "arrival_rate_fps": self.queue.config.arrival_rate_fps,
                "buffer_capacity": self.queue.config.buffer_capacity,
                "frame_budget_ms": self.queue.config.frame_budget_ms,
                "max_health": self.queue.config.max_health,
                "sla_latency_ms": self.queue.config.sla_latency_ms,
            },
            "default_model": self.model_key,
        }

    # -- control channel ------------------------------------------------------

    def handle_command(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Apply one control message from the cockpit."""
        action = msg.get("action")

        if action == "set_model":
            key = msg.get("model")
            if self.set_model(key):
                return {"type": "ack", "action": action, "model": key}
            return {"type": "error", "message": f"unknown model '{key}'"}

        if action == "set_storm":
            self.storms.set_storm(msg.get("storm", ""), float(msg.get("intensity", 0.0)))
            return {"type": "ack", "action": action, "storms": self.storms.to_dict()}

        if action == "toggle_storm":
            self.storms.toggle_storm(msg.get("storm", ""))
            return {"type": "ack", "action": action, "storms": self.storms.to_dict()}

        if action == "set_filter":
            self.storms.set_filter(msg.get("filter", ""), bool(msg.get("enabled", False)))
            return {"type": "ack", "action": action, "storms": self.storms.to_dict()}

        if action == "toggle_filter":
            self.storms.toggle_filter(msg.get("filter", ""))
            return {"type": "ack", "action": action, "storms": self.storms.to_dict()}

        if action == "pause":
            self.paused = bool(msg.get("paused", not self.paused))
            return {"type": "ack", "action": action, "paused": self.paused}

        if action == "reset":
            self.reset()
            return {"type": "ack", "action": action}

        if action == "ping":
            return {"type": "pong", "t": time.time()}

        return {"type": "error", "message": f"unknown action '{action}'"}


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket
# ─────────────────────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def telemetry_socket(websocket: WebSocket) -> None:
    """25 FPS telemetry stream with a bidirectional control channel.

    Two coroutines share one session: a producer pacing the simulation at the
    camera's frame rate, and a consumer applying control messages. They are not
    locked against each other because asyncio gives us cooperative scheduling —
    neither awaits in the middle of mutating shared state.
    """
    await websocket.accept()
    session = CockpitSession()

    await websocket.send_json(session.handshake())

    async def producer() -> None:
        # Absolute scheduling rather than `await sleep(TICK_DT)`: sleeping a
        # fixed interval accumulates drift, and a simulation whose tick rate
        # quietly sags to 23 FPS invalidates every latency number on screen.
        next_tick = time.perf_counter()
        while True:
            next_tick += TICK_DT
            payload = await asyncio.to_thread(session.step)
            await websocket.send_json(payload)

            delay = next_tick - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # We fell behind. Resynchronise rather than trying to catch up,
                # which would spiral.
                next_tick = time.perf_counter()

    async def consumer() -> None:
        while True:
            msg = await websocket.receive_json()
            reply = session.handle_command(msg)
            if reply is not None:
                await websocket.send_json(reply)

    async def calibrator() -> None:
        """Periodically re-measure real inference latency, off the tick path."""
        while True:
            await asyncio.sleep(CALIBRATION_PERIOD_S)
            if session.paused:
                continue
            reading = await asyncio.to_thread(session.calibrate)
            if reading is not None:
                await websocket.send_json(reading)

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    calibrator_task = asyncio.create_task(calibrator())

    try:
        done, pending = await asyncio.wait(
            {producer_task, consumer_task, calibrator_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except WebSocketDisconnect:
        pass
    finally:
        for task in (producer_task, consumer_task, calibrator_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ─────────────────────────────────────────────────────────────────────────────
#  REST
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "tick_hz": TICK_HZ}


@app.get("/api/classes")
async def classes() -> dict[str, Any]:
    return {"classes": list(CLASS_META), "names": list(CLASS_NAMES)}


@app.get("/api/models")
async def models() -> dict[str, Any]:
    zoo = get_zoo()
    return {"models": await asyncio.to_thread(zoo.describe_all)}


@app.get("/api/models/{key}")
async def model_detail(key: str) -> JSONResponse:
    zoo = get_zoo()
    if key not in MODEL_SPECS:
        return JSONResponse({"error": f"unknown model '{key}'"}, status_code=404)
    payload = await asyncio.to_thread(zoo.describe, key, True)
    return JSONResponse(payload)


@app.get("/api/augments")
async def augments() -> dict[str, Any]:
    return specs_payload()


@app.get("/api/pareto")
async def pareto() -> dict[str, Any]:
    zoo = get_zoo()
    return {"points": await asyncio.to_thread(zoo.pareto_points)}


@app.get("/api/theory")
async def theory(service_ms: float = Query(11.5, gt=0.0)) -> dict[str, Any]:
    """Closed-form M/D/1 predictions for an arbitrary service time."""
    sim = EdgeQueueSimulator(QueueConfig())
    return sim.theory(service_ms)


@app.get("/api/frame")
async def frame(
    label: int | None = Query(None, ge=0, le=NUM_CLASSES - 1),
    subtlety: float | None = Query(None, ge=0.0, le=1.0),
    size: int = Query(224, ge=48, le=448),
) -> dict[str, Any]:
    """Render one synthetic inspection frame."""
    gen = RackFrameGenerator()
    f = await asyncio.to_thread(gen.generate, label, subtlety)
    return {
        "label": f.label,
        "label_name": f.label_name,
        "subtlety": round(f.subtlety, 3),
        "descriptor": f.descriptor,
        "image": f.to_png_b64(quality_size=size),
    }


@app.get("/api/samples")
async def samples(size: int = Query(160, ge=48, le=448)) -> dict[str, Any]:
    """One blatant example of each class — the UI's class legend."""
    gen = RackFrameGenerator()
    sheet = await asyncio.to_thread(gen.sample_sheet)
    return {
        "samples": [
            {
                "label": f.label,
                "label_name": f.label_name,
                "meta": CLASS_META[f.label],
                "descriptor": f.descriptor,
                "image": f.to_png_b64(quality_size=size),
            }
            for f in sheet
        ]
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
