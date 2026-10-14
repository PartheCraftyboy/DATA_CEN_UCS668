"""
Data Center Edge-Vision Commander — Edge Vision Core.

UCS668 · Edge AI and Robotics: Data Center Vision
Project 1 — The Latency vs. Accuracy Simulator.

This package implements the server-side half of the system:

    app.models      Module 2 & 3 — the switchable backbone zoo and the
                    architectural profiler (params, MACs, tensor shapes).
    app.simulation  Module 1 & 4 — the procedural inspection-frame generator,
                    the environmental storm/augmentation bank, and the M/D/1
                    finite-buffer edge queue state machine.
    app.metrics     Module 5 — running confusion matrix, calibration error,
                    macro-F1 and the Pareto frontier accumulator.
    app.main        The FastAPI ASGI surface: a 25 FPS WebSocket telemetry
                    loop plus REST introspection endpoints.
"""

__version__ = "1.0.0"
__course__ = "UCS668 — Edge AI and Robotics: Data Center Vision"

# The five inspection targets. Index order is load-bearing: it defines the row
# and column order of the 5x5 confusion matrix all the way up to the frontend.
CLASS_NAMES: tuple[str, ...] = (
    "Optimal_Rack",
    "Thermal_Hotspot",
    "Cable_Bend_Violation",
    "Port_LED_Error",
    "Intake_Dust_Clogged",
)

NUM_CLASSES: int = len(CLASS_NAMES)

# Human-facing labels and the severity weighting used by the scoring engine.
# A missed thermal hotspot is far more expensive than a missed dust intake:
# one melts a blade within minutes, the other degrades airflow over weeks.
CLASS_META: tuple[dict, ...] = (
    {
        "id": 0,
        "key": "Optimal_Rack",
        "label": "Optimal Rack",
        "short": "OPTIMAL",
        "severity": 0.0,
        "miss_cost": 0.0,
        "accent": "#10b981",
        "blurb": "Nominal thermals, green activity LEDs, unobstructed intake.",
    },
    {
        "id": 1,
        "key": "Thermal_Hotspot",
        "label": "Thermal Hotspot",
        "short": "THERMAL",
        "severity": 1.0,
        "miss_cost": 9.0,
        "accent": "#ef4444",
        "blurb": "PSU/blade over temperature. Infrared false-colour plume.",
    },
    {
        "id": 2,
        "key": "Cable_Bend_Violation",
        "label": "Cable Bend Violation",
        "short": "BEND",
        "severity": 0.75,
        "miss_cost": 6.0,
        "accent": "#a855f7",
        "blurb": "Fiber patch cord below the 30 mm critical bend radius.",
    },
    {
        "id": 3,
        "key": "Port_LED_Error",
        "label": "Port LED Error",
        "short": "PORT",
        "severity": 0.6,
        "miss_cost": 4.0,
        "accent": "#f59e0b",
        "blurb": "Switch port signalling amber/red link-down or CRC fault.",
    },
    {
        "id": 4,
        "key": "Intake_Dust_Clogged",
        "label": "Intake Dust Clogged",
        "short": "DUST",
        "severity": 0.5,
        "miss_cost": 3.0,
        "accent": "#94a3b8",
        "blurb": "Perforated chassis intake occluded by particulate debris.",
    },
)
