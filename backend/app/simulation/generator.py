"""
Module 1 — Understanding the Vision Problem.

A procedural synthesiser for 224x224x3 datacenter rack inspection frames.

Every frame is drawn from scratch with PIL: a dark chassis, a stack of 1U
blades with bezels and perforated intake grilles, an activity LED column, a
fiber patch bundle down one side, and the camera's own artefacts (vignette,
sensor noise, rolling-shutter banding). On top of that base, exactly one of
five defect signatures is painted.

Why synthesise rather than use a photo dataset: there is no public labelled
corpus of rack-level defect imagery, and more usefully, a procedural generator
lets us dial the *difficulty* of each class continuously. The `subtlety`
parameter controls how faint a defect signature is drawn, which is what makes
the low-capacity backbone's false-negative rate a function of something real
in the pixels rather than a free parameter.

Each frame carries its own ground-truth label, so the confusion matrix
downstream is measuring something true.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app import CLASS_NAMES, NUM_CLASSES

FRAME_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
#  Palette — a dimly-lit cold-aisle at 3200K emergency lighting
# ─────────────────────────────────────────────────────────────────────────────

_BG_DARK = (9, 13, 22)
_CHASSIS = (28, 34, 46)
_CHASSIS_LIGHT = (44, 52, 68)
_BEZEL = (18, 22, 32)
_RAIL = (58, 66, 84)
_VENT = (14, 18, 26)

_LED_GREEN = (16, 185, 129)
_LED_AMBER = (245, 158, 11)
_LED_RED = (239, 68, 68)
_LED_BLUE = (56, 189, 248)

_FIBER_COLOURS = [
    (250, 204, 21),   # OS2 single-mode yellow
    (34, 211, 238),   # OM3 aqua
    (168, 85, 247),   # OM4 violet
    (248, 113, 113),  # patch red
]


@dataclass
class SyntheticFrame:
    """One generated inspection frame plus its ground truth."""

    image: np.ndarray           # HxWx3 uint8
    label: int                  # index into CLASS_NAMES
    subtlety: float             # 0 = blatant, 1 = barely visible
    seed: int
    descriptor: dict[str, Any]  # what was drawn, for the telemetry panel

    @property
    def label_name(self) -> str:
        return CLASS_NAMES[self.label]

    def to_png_b64(self, scale: int = 1, quality_size: int | None = None) -> str:
        """Encode as a base64 PNG data URI for the WebSocket payload.

        Thumbnails are downscaled before encoding — at 25 FPS, shipping full
        224x224 PNGs over a WebSocket is roughly 1.2 MB/s of base64, which
        stutters the conveyor animation on a lab laptop. 96px costs about a
        fifth of that and is indistinguishable at the size the UI draws it.
        """
        img = Image.fromarray(self.image)
        if quality_size and quality_size != img.width:
            img = img.resize((quality_size, quality_size), Image.BILINEAR)
        elif scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=1)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────────────────────────────────────


class RackFrameGenerator:
    """Procedural 5-class server-rack frame synthesiser."""

    def __init__(self, seed: int | None = None, size: int = FRAME_SIZE) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self._counter = 0

        # Static fields precomputed once. These were originally per-row PIL
        # draw calls inside the hot path; at 25 FPS that was 224 line draws and
        # a fresh meshgrid per frame, for a result that never changes.
        xs = np.linspace(-1, 1, size, dtype=np.float32)
        gx, gy = np.meshgrid(xs, xs)
        radial = np.sqrt(gx**2 + gy**2) / math.sqrt(2.0)
        self._vignette = (1.0 - 0.34 * radial**2).astype(np.float32)[..., None]

        # Aisle lighting gradient: brighter at the top, falling off to the floor.
        k = (1.0 - (np.arange(size, dtype=np.float32) / size) * 0.55)[:, None, None]
        self._ambient = (k * np.array([20, 26, 38], dtype=np.float32)).astype(np.uint8)
        self._ambient = np.broadcast_to(self._ambient, (size, size, 3)).copy()

    # -- public API -----------------------------------------------------------

    def generate(
        self,
        label: int | None = None,
        subtlety: float | None = None,
        class_weights: np.ndarray | None = None,
    ) -> SyntheticFrame:
        """Draw one frame.

        `class_weights` lets the caller skew the class distribution. The
        default is deliberately imbalanced toward Optimal_Rack (55%), because
        that is what a real inspection stream looks like: defects are rare.
        Training a student to read a confusion matrix on a balanced stream
        teaches the wrong instinct about precision.
        """
        if label is None:
            if class_weights is None:
                class_weights = np.array([0.55, 0.11, 0.12, 0.12, 0.10])
            class_weights = np.asarray(class_weights, dtype=np.float64)
            class_weights = class_weights / class_weights.sum()
            label = int(self.rng.choice(NUM_CLASSES, p=class_weights))

        if subtlety is None:
            # Beta(2, 3) skews toward clearly-visible defects with a long tail
            # of genuinely hard ones.
            subtlety = float(self.rng.beta(2.0, 3.0))

        seed = int(self.rng.integers(0, 2**31 - 1))
        local = np.random.default_rng(seed)

        img = Image.fromarray(self._ambient.copy())
        draw = ImageDraw.Draw(img, "RGBA")

        descriptor: dict[str, Any] = {"label": int(label), "subtlety": round(subtlety, 3)}

        geometry = self._draw_rack_base(draw, local, descriptor)

        painter = (
            self._paint_optimal,
            self._paint_thermal_hotspot,
            self._paint_cable_bend,
            self._paint_port_led_error,
            self._paint_dust_clog,
        )[label]
        painter(img, draw, local, geometry, subtlety, descriptor)

        self._apply_camera_artifacts(img, local)

        self._counter += 1
        return SyntheticFrame(
            image=np.asarray(img, dtype=np.uint8),
            label=int(label),
            subtlety=subtlety,
            seed=seed,
            descriptor=descriptor,
        )

    def batch(self, n: int, **kwargs: Any) -> list[SyntheticFrame]:
        return [self.generate(**kwargs) for _ in range(n)]

    def sample_sheet(self) -> list[SyntheticFrame]:
        """One blatant example of each class, for the UI's class legend."""
        return [self.generate(label=c, subtlety=0.08) for c in range(NUM_CLASSES)]

    # -- base rack ------------------------------------------------------------

    def _draw_rack_base(
        self,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        descriptor: dict[str, Any],
    ) -> dict[str, Any]:
        """Draw the chassis, blades, rails, vents and LED column.

        Returns the geometry so the defect painters can place their signature
        on a real blade rather than at a random pixel.
        """
        s = self.size

        # (The ambient aisle gradient is precomputed in __init__ and used as
        # the canvas, so there is nothing to draw for it here.)

        # Vertical mounting rails.
        rail_w = 9
        draw.rectangle([0, 0, rail_w, s], fill=_RAIL)
        draw.rectangle([s - rail_w, 0, s, s], fill=_RAIL)
        for y in range(6, s, 11):
            draw.rectangle([2, y, rail_w - 3, y + 4], fill=(22, 27, 38))
            draw.rectangle([s - rail_w + 2, y, s - 3, y + 4], fill=(22, 27, 38))

        n_blades = int(rng.integers(4, 7))
        gap = 3
        usable = s - 12
        blade_h = (usable - gap * (n_blades - 1)) // n_blades

        blades: list[dict[str, Any]] = []
        y = 6
        for i in range(n_blades):
            x0, x1 = rail_w + 2, s - rail_w - 2
            y0, y1 = y, y + blade_h

            # Chassis face with a subtle top-lit bevel.
            draw.rectangle([x0, y0, x1, y1], fill=_CHASSIS)
            draw.line([(x0, y0), (x1, y0)], fill=_CHASSIS_LIGHT, width=1)
            draw.line([(x0, y1), (x1, y1)], fill=_BEZEL, width=1)

            # Perforated intake grille across the left two-thirds.
            vent_x0 = x0 + 6
            vent_x1 = x0 + int((x1 - x0) * 0.62)
            vent_y0 = y0 + 4
            vent_y1 = y1 - 4
            self._draw_perforation(draw, vent_x0, vent_y0, vent_x1, vent_y1, rng)

            # Handle / latch on the right.
            hx0 = x1 - 26
            draw.rounded_rectangle(
                [hx0, y0 + 5, x1 - 6, y1 - 5], radius=2, fill=(52, 60, 76)
            )
            draw.line([(hx0 + 4, (y0 + y1) // 2), (x1 - 10, (y0 + y1) // 2)],
                      fill=(70, 80, 98), width=1)

            # Activity LED pair, left of the vent.
            led_x = x0 + 2
            led_y = y0 + blade_h // 2
            blades.append(
                {
                    "index": i,
                    "box": (x0, y0, x1, y1),
                    "vent": (vent_x0, vent_y0, vent_x1, vent_y1),
                    "led": (led_x, led_y),
                    "height": blade_h,
                }
            )
            y = y1 + gap

        descriptor["blades"] = n_blades

        # Fiber patch bundle running down the right-hand channel.
        bundle = self._draw_cable_bundle(draw, rng)
        descriptor["cables"] = len(bundle)

        return {"blades": blades, "rail_w": rail_w, "bundle": bundle}

    @staticmethod
    def _draw_perforation(
        draw: ImageDraw.ImageDraw,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        rng: np.random.Generator,
    ) -> None:
        """Hexagonal-ish perforated intake mesh."""
        draw.rectangle([x0, y0, x1, y1], fill=_VENT)
        step = 4
        row = 0
        for py in range(y0 + 2, y1 - 1, step):
            offset = 2 if row % 2 else 0
            for px in range(x0 + 2 + offset, x1 - 1, step):
                draw.point((px, py), fill=(34, 41, 55))
            row += 1

    def _draw_cable_bundle(
        self, draw: ImageDraw.ImageDraw, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Draw 2-4 slack fiber runs down the right channel."""
        s = self.size
        bundle = []
        n = int(rng.integers(2, 5))
        for i in range(n):
            colour = _FIBER_COLOURS[int(rng.integers(0, len(_FIBER_COLOURS)))]
            base_x = s - 22 - i * 4
            amp = float(rng.uniform(3, 8))
            phase = float(rng.uniform(0, math.tau))
            pts = []
            for y in range(4, s - 4, 4):
                x = base_x + amp * math.sin(y / 34.0 + phase)
                pts.append((x, y))
            draw.line(pts, fill=colour + (190,), width=2, joint="curve")
            bundle.append({"colour": colour, "base_x": base_x})
        return bundle

    # -- class signatures -----------------------------------------------------

    def _paint_optimal(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        geo: dict[str, Any],
        subtlety: float,
        descriptor: dict[str, Any],
    ) -> None:
        """Class 0 — everything nominal. Green activity, clean intake."""
        for blade in geo["blades"]:
            lx, ly = blade["led"]
            # Activity LEDs blink, so some are lit and some are mid-cycle.
            lit = rng.random() > 0.25
            colour = _LED_GREEN if lit else (20, 60, 48)
            self._glow_dot(draw, lx + 3, ly, 2, colour, intensity=0.85 if lit else 0.25)
            self._glow_dot(draw, lx + 3, ly + 6, 1, _LED_BLUE, intensity=0.5)
        descriptor["signature"] = "nominal_green_activity"

    def _paint_thermal_hotspot(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        geo: dict[str, Any],
        subtlety: float,
        descriptor: dict[str, Any],
    ) -> None:
        """Class 1 — infrared false-colour plume over an overheating blade.

        Drawn as a radial temperature field mapped through an ironbow-style
        colour ramp, composited additively so it reads as emitted heat rather
        than a painted sticker.
        """
        blade = geo["blades"][int(rng.integers(0, len(geo["blades"])))]
        x0, y0, x1, y1 = blade["box"]

        cx = float(rng.uniform(x0 + 20, x1 - 20))
        cy = float((y0 + y1) / 2 + rng.uniform(-4, 4))
        radius = float(rng.uniform(26, 44))
        peak = (1.0 - 0.72 * subtlety)  # subtle hotspots are cooler-looking

        xs = np.arange(self.size, dtype=np.float32)
        gx, gy = np.meshgrid(xs, xs)
        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        field = np.clip(1.0 - d / radius, 0.0, 1.0) ** 1.6 * peak

        base = np.asarray(img, dtype=np.float32)
        ramp = self._ironbow(field)
        alpha = (field * 0.92)[..., None]
        blended = base * (1 - alpha) + ramp * alpha
        img.paste(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)))

        # Re-draw the LED row on top, now amber: the BMC has noticed.
        draw2 = ImageDraw.Draw(img, "RGBA")
        lx, ly = blade["led"]
        self._glow_dot(draw2, lx + 3, ly, 2, _LED_AMBER, intensity=0.9)

        descriptor["signature"] = "ir_plume"
        descriptor["hotspot"] = {"cx": round(cx, 1), "cy": round(cy, 1),
                                 "radius": round(radius, 1), "peak": round(peak, 3)}

    @staticmethod
    def _ironbow(t: np.ndarray) -> np.ndarray:
        """Map [0,1] temperature to an ironbow-style thermographic ramp."""
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.65 * t, 0, 1)
        g = np.clip(2.1 * t - 0.62, 0, 1)
        b = np.clip(np.where(t < 0.5, 1.5 * t, 2.4 * t - 1.25), 0, 1) * 0.65
        return np.stack([r, g, b], axis=-1) * 255.0

    def _paint_cable_bend(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        geo: dict[str, Any],
        subtlety: float,
        descriptor: dict[str, Any],
    ) -> None:
        """Class 2 — a fiber patch cord looped below its critical bend radius.

        The defect is purely geometric: a tight circular arc whose radius, in
        the frame's pixel scale, corresponds to under 30 mm of real fiber. It
        is the lowest-contrast class in the set — a thin curve against a dark
        chassis — which is exactly why it is the one a low-capacity backbone
        drops first.
        """
        s = self.size
        colour = _FIBER_COLOURS[int(rng.integers(0, len(_FIBER_COLOURS)))]
        cx = float(rng.uniform(s * 0.35, s * 0.72))
        cy = float(rng.uniform(s * 0.28, s * 0.72))
        # Tighter loop = more severe violation = easier to see.
        loop_r = float(rng.uniform(9, 15)) + subtlety * 5.0
        width = 2 if subtlety > 0.55 else 3
        alpha = int(255 * (1.0 - 0.45 * subtlety))

        # Lead-in run, tight loop, lead-out run.
        pts: list[tuple[float, float]] = []
        for t in np.linspace(0, 1, 24):
            pts.append((cx - 52 + t * 52, cy + 34 - t * 34 * 0.85))
        for theta in np.linspace(math.pi * 0.72, math.pi * 2.55, 46):
            pts.append((cx + loop_r * math.cos(theta), cy + loop_r * math.sin(theta)))
        for t in np.linspace(0, 1, 24):
            pts.append((cx + 6 + t * 46, cy - 12 + t * 40))

        # Dark backing stroke so the cord reads as a physical object with depth.
        draw.line(pts, fill=(6, 8, 14, 210), width=width + 2, joint="curve")
        draw.line(pts, fill=colour + (alpha,), width=width, joint="curve")

        # Specular highlight along the top of the strand.
        hi = [(x, y - 0.8) for x, y in pts[::2]]
        draw.line(hi, fill=(255, 255, 255, int(70 * (1 - subtlety))), width=1, joint="curve")

        descriptor["signature"] = "sub_minimum_bend_radius"
        descriptor["bend"] = {"cx": round(cx, 1), "cy": round(cy, 1),
                              "radius_px": round(loop_r, 1)}

    def _paint_port_led_error(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        geo: dict[str, Any],
        subtlety: float,
        descriptor: dict[str, Any],
    ) -> None:
        """Class 3 — a switch faceplate with ports signalling amber/red."""
        blade = geo["blades"][int(rng.integers(0, len(geo["blades"])))]
        x0, y0, x1, y1 = blade["box"]
        mid = (y0 + y1) // 2

        # Draw a 2-row SFP cage array across the blade face.
        draw.rectangle([x0 + 6, y0 + 4, x1 - 30, y1 - 4], fill=(16, 20, 29))
        n_ports = 12
        pw = (x1 - 30 - (x0 + 6) - 6) / n_ports
        faulted = set(
            int(i) for i in rng.choice(n_ports, size=int(rng.integers(1, 4)), replace=False)
        )

        for i in range(n_ports):
            px = x0 + 9 + i * pw
            for row, py in enumerate((y0 + 7, mid + 2)):
                draw.rectangle([px, py, px + pw - 3, py + 7], fill=(30, 36, 48))
                draw.rectangle([px + 1, py + 1, px + pw - 4, py + 3], fill=(12, 15, 22))

                if i in faulted and row == 0:
                    colour = _LED_RED if rng.random() > 0.45 else _LED_AMBER
                    intensity = 1.0 - 0.55 * subtlety
                else:
                    colour = _LED_GREEN if rng.random() > 0.3 else (18, 48, 40)
                    intensity = 0.55
                self._glow_dot(draw, px + 2, py + 5.5, 1.4, colour, intensity)

        descriptor["signature"] = "port_link_fault"
        descriptor["faulted_ports"] = sorted(faulted)

    def _paint_dust_clog(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        rng: np.random.Generator,
        geo: dict[str, Any],
        subtlety: float,
        descriptor: dict[str, Any],
    ) -> None:
        """Class 4 — particulate debris occluding the perforated intake.

        Built as a low-frequency occlusion mask (where the dust mat has
        accumulated) modulated by high-frequency grain (the particles), so the
        texture survives downscaling the way real dust does.
        """
        blade = geo["blades"][int(rng.integers(0, len(geo["blades"])))]
        vx0, vy0, vx1, vy1 = blade["vent"]
        w, h = int(vx1 - vx0), int(vy1 - vy0)
        if w <= 2 or h <= 2:
            return

        coverage = 1.0 - 0.62 * subtlety

        # Low-frequency mat: upsampled coarse noise.
        coarse = rng.random((max(2, h // 5), max(2, w // 5))).astype(np.float32)
        mat = np.asarray(
            Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        mat = np.clip((mat - 0.34) * 2.3, 0, 1) * coverage

        # High-frequency particulate grain.
        grain = rng.random((h, w)).astype(np.float32)
        mask = np.clip(mat * (0.72 + 0.5 * grain), 0, 1)

        patch = np.asarray(img, dtype=np.float32)[vy0:vy1, vx0:vx1, :]
        dust_rgb = np.array([138, 126, 108], dtype=np.float32)
        a = (mask * 0.86)[..., None]
        patch = patch * (1 - a) + dust_rgb * a

        base = np.asarray(img).copy()
        base[vy0:vy1, vx0:vx1, :] = np.clip(patch, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(base))

        # Fine fibre lint catching the light at the grille edge.
        draw2 = ImageDraw.Draw(img, "RGBA")
        for _ in range(int(14 * coverage)):
            fx = float(rng.uniform(vx0, vx1))
            fy = float(rng.uniform(vy0, vy1))
            draw2.line(
                [(fx, fy), (fx + rng.uniform(-5, 5), fy + rng.uniform(-3, 3))],
                fill=(176, 166, 146, 150),
                width=1,
            )

        descriptor["signature"] = "intake_occlusion"
        descriptor["coverage"] = round(float(coverage), 3)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _glow_dot(
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        r: float,
        colour: tuple[int, int, int],
        intensity: float = 1.0,
    ) -> None:
        """An LED: a saturated core inside two halo rings."""
        for ring, falloff in ((r * 3.0, 0.10), (r * 1.9, 0.26)):
            a = int(255 * intensity * falloff)
            draw.ellipse([x - ring, y - ring, x + ring, y + ring], fill=colour + (a,))
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=tuple(min(255, int(c * (0.55 + 0.45 * intensity))) for c in colour)
            + (int(255 * intensity),),
        )

    def _apply_camera_artifacts(self, img: Image.Image, rng: np.random.Generator) -> None:
        """Vignette, sensor noise and rolling-shutter banding.

        These are not decoration. They are the baseline nuisance variation that
        makes the task non-trivial even before the augmentation storms hit, and
        they are what stops the classifier from latching onto a pixel-exact
        template.
        """
        arr = np.asarray(img, dtype=np.float32)
        s = self.size

        # Vignette (precomputed field).
        arr *= self._vignette

        # Rolling-shutter banding from the aisle's 50 Hz lighting.
        phase = float(rng.uniform(0, math.tau))
        band = 1.0 + 0.035 * np.sin(np.linspace(0, math.tau * 3.5, s) + phase)
        arr *= band[:, None, None]

        # Sensor read noise.
        arr += rng.normal(0.0, 3.1, arr.shape).astype(np.float32)

        img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
        if rng.random() > 0.72:
            img.paste(img.filter(ImageFilter.GaussianBlur(radius=0.4)))
