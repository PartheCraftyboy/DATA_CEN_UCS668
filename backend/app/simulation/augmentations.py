"""
Module 4 — Data Augmentation & Environmental Storms.

Two halves, deliberately opposed:

  STORMS  — environmental perturbations the player does *not* control, which
            destroy the defect signatures the classifier depends on. Heat haze
            from a hot aisle, strobe glare from the overhead maintenance LEDs,
            and airborne particulate from a filter change.

  FILTERS — edge preprocessing kernels the player *does* control, deployed to
            claw accuracy back. Each one costs real milliseconds, charged to
            the same service-time budget as inference, so stacking all three
            will overflow the queue just as surely as picking ResNet-18.

That tension is the Module 4 lesson: preprocessing is not free robustness. It
is a trade you pay for in latency, out of the same 40 ms budget.

Every filter here is timed with `time.perf_counter()` and its *real* measured
cost is charged to the queue. No lookup tables.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # OpenCV gives a hardware-tuned CLAHE and a true bilateral filter.
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    from scipy import ndimage as ndi  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    ndi = None  # type: ignore
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
#  Specifications (shipped to the frontend so the UI stays in sync)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StormSpec:
    key: str
    display_name: str
    short: str
    accent: str
    icon: str
    # How much this storm raises the effective severity fed to the decision
    # model at intensity 1.0.
    severity_weight: float
    formula: str
    blurb: str
    physics: str


STORM_SPECS: dict[str, StormSpec] = {
    "heat_haze": StormSpec(
        key="heat_haze",
        display_name="Hot Aisle Heat Haze",
        short="HAZE",
        accent="#f59e0b",
        icon="waves",
        severity_weight=0.46,
        formula="x' = x + A·sin(2πf·y + φ)",
        blurb="Refractive shimmer off the hot aisle. Warps thin geometry.",
        physics=(
            "Air density varies with temperature, so the refractive index "
            "varies with it too. Light crossing a thermal gradient bends. We "
            "model it as a sinusoidal horizontal displacement field whose "
            "amplitude grows toward the bottom of the frame, where the exhaust "
            "plume is densest. Geometric defects — a fiber bend radius — are "
            "destroyed first, because the cue IS the geometry."
        ),
    ),
    "strobe_glare": StormSpec(
        key="strobe_glare",
        display_name="Strobe LED Glare",
        short="GLARE",
        accent="#38bdf8",
        icon="flash",
        severity_weight=0.38,
        formula="I' = I + G·exp(-‖p - c‖² / 2σ²)",
        blurb="Specular bloom from maintenance lighting. Clips highlights.",
        physics=(
            "An additive Gaussian specular lobe with hard highlight clipping. "
            "Because it saturates channels rather than attenuating them, it "
            "wipes out colour-coded cues: an amber port LED and a thermal "
            "plume become the same blown-out white blob, which is precisely "
            "the Thermal ↔ Port_LED confusion visible on the matrix."
        ),
    ),
    "intake_dust": StormSpec(
        key="intake_dust",
        display_name="Intake Particulate Storm",
        short="DUST",
        accent="#94a3b8",
        icon="grain",
        severity_weight=0.34,
        formula="I' = I·(1-α) + Poisson(λ)·α",
        blurb="Airborne particulate. High-frequency occlusion noise.",
        physics=(
            "Shot noise from particles crossing the sensor plane, drawn from a "
            "Poisson process and composited as partial occlusion. Unlike "
            "Gaussian noise it is signal-dependent and spatially clumped, so "
            "low-pass filtering removes it only at the cost of the fine "
            "texture that distinguishes a dusty grille from a clean one."
        ),
    ),
}


@dataclass(frozen=True)
class FilterSpec:
    key: str
    display_name: str
    short: str
    accent: str
    # Fraction of storm severity this kernel neutralises, per storm key.
    counters: dict[str, float]
    # Typical cost in ms — displayed before deployment; the real measured cost
    # is what actually gets charged.
    nominal_cost_ms: float
    formula: str
    blurb: str
    tradeoff: str


FILTER_SPECS: dict[str, FilterSpec] = {
    "clahe": FilterSpec(
        key="clahe",
        display_name="CLAHE",
        short="CLAHE",
        accent="#10b981",
        counters={"strobe_glare": 0.62, "heat_haze": 0.12, "intake_dust": 0.18},
        nominal_cost_ms=1.8,
        formula="Contrast-Limited Adaptive Histogram Equalisation (8×8 tiles)",
        blurb="Restores local contrast inside blown-out glare lobes.",
        tradeoff=(
            "Clip-limited so it cannot amplify noise without bound — but it "
            "still lifts the noise floor in genuinely dark regions, which can "
            "manufacture texture where there is none."
        ),
    ),
    "unsharp": FilterSpec(
        key="unsharp",
        display_name="Unsharp Mask",
        short="UNSHRP",
        accent="#06b6d4",
        counters={"heat_haze": 0.48, "strobe_glare": 0.10, "intake_dust": 0.08},
        nominal_cost_ms=1.1,
        formula="I' = I + k·(I − G_σ * I)",
        blurb="Re-sharpens edges smeared by refractive warping.",
        tradeoff=(
            "Amplifies high frequencies indiscriminately, so under a dust "
            "storm it sharpens the particulate as enthusiastically as the "
            "defect. Stacking it with dust present is actively harmful."
        ),
    ),
    "bilateral": FilterSpec(
        key="bilateral",
        display_name="Bilateral Edge Filter",
        short="BILAT",
        accent="#a855f7",
        counters={"intake_dust": 0.66, "heat_haze": 0.20, "strobe_glare": 0.14},
        nominal_cost_ms=4.6,
        formula="I'(p) = (1/W)·Σ G_σs(‖p−q‖)·G_σr(|I(p)−I(q)|)·I(q)",
        blurb="Edge-preserving denoise. Kills grain, keeps boundaries.",
        tradeoff=(
            "By far the most expensive kernel in the bank — a 4-5 ms edge-"
            "preserving smooth is a tenth of the entire frame budget. Deploy "
            "it with ResNet-18 selected and the queue overflows immediately."
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Storm implementations
# ─────────────────────────────────────────────────────────────────────────────


def heat_haze(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal refractive displacement:  x' = x + A·sin(2πf·y + φ).

    Amplitude ramps with height so the shimmer is strongest near the exhaust,
    which is both physically right and visually legible.
    """
    if intensity <= 0:
        return img
    h, w = img.shape[:2]
    amp = 1.0 + 7.5 * intensity
    freq = 2.4 + 1.8 * float(rng.random())
    phase = float(rng.uniform(0, math.tau))

    ys = np.arange(h, dtype=np.float32)
    # Vertical ramp: 0.35 at the top of the frame, 1.0 at the bottom.
    ramp = 0.35 + 0.65 * (ys / max(h - 1, 1))
    dx = (amp * ramp * np.sin(2 * math.pi * freq * ys / h + phase)).astype(np.float32)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), ys)
    src_x = np.clip(gx + dx[:, None], 0, w - 1)

    if _HAS_CV2:
        # cv2.remap warps all three channels in one SIMD pass — roughly 15x
        # faster than three separate scipy.map_coordinates calls, which matters
        # because this runs inside a 40 ms real-time budget.
        return cv2.remap(
            img,
            src_x,
            gy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    if _HAS_SCIPY:
        out = np.empty_like(img, dtype=np.float32)
        coords = np.stack([gy, src_x])
        for c in range(img.shape[2]):
            out[..., c] = ndi.map_coordinates(
                img[..., c].astype(np.float32), coords, order=1, mode="nearest"
            )
        return np.clip(out, 0, 255).astype(np.uint8)

    # Nearest-neighbour fallback keeps the project runnable without SciPy.
    idx = np.clip(np.rint(src_x).astype(np.int32), 0, w - 1)
    rows = np.arange(h)[:, None]
    return img[rows, idx]


def strobe_glare(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian specular lobe with highlight clipping."""
    if intensity <= 0:
        return img
    h, w = img.shape[:2]
    cx = float(rng.uniform(w * 0.2, w * 0.8))
    cy = float(rng.uniform(h * 0.15, h * 0.85))
    sigma = float(rng.uniform(w * 0.14, w * 0.30))
    gain = 235.0 * intensity

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    lobe = np.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * sigma**2)))

    # A faint anamorphic streak — real lens flare is not radially symmetric.
    streak = np.exp(-((gy - cy) ** 2) / (2.0 * (sigma * 0.11) ** 2)) * 0.42

    mask = np.clip(lobe + streak * lobe, 0, 1)[..., None] * gain
    # Slightly cool tint: maintenance LEDs run around 5600K.
    tint = np.array([0.94, 0.97, 1.0], dtype=np.float32)
    return np.clip(img.astype(np.float32) + mask * tint, 0, 255).astype(np.uint8)


def intake_dust(
    img: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
    exact_poisson: bool = False,
) -> np.ndarray:
    """Signal-dependent Poisson shot noise plus clumped particulate occlusion.

    By default the Poisson sampling uses its Gaussian limit. For X ~ Poisson(λ),
    X → N(λ, λ) as λ grows, and the approximation is within about 1% of the
    true distribution for λ ≳ 10 — which holds everywhere except the darkest
    pixels of the frame. Sampling a Normal is roughly 20x cheaper than sampling
    a Poisson over 150k elements, and this kernel runs inside a 40 ms real-time
    budget, so the approximation is what makes the storm affordable live.

    Pass ``exact_poisson=True`` for the true distribution. The REST sample
    endpoints do this, since nothing there is on a clock.
    """
    if intensity <= 0:
        return img
    arr = img.astype(np.float32)
    h, w = arr.shape[:2]

    # Shot noise. Lower lambda scaling = noisier, so we invert.
    lam_scale = max(3.0, 42.0 * (1.0 - intensity))
    lam = np.clip(arr, 0, None) / 255.0 * lam_scale

    if exact_poisson:
        sampled = rng.poisson(lam).astype(np.float32)
    else:
        # N(λ, λ): mean λ, variance λ, hence standard deviation sqrt(λ).
        sampled = lam + rng.standard_normal(lam.shape).astype(np.float32) * np.sqrt(lam)
        sampled = np.clip(sampled, 0.0, None)
    noisy = sampled / lam_scale * 255.0

    # Clumped particulate: coarse noise upsampled, thresholded into specks.
    coarse = rng.random((max(2, h // 8), max(2, w // 8))).astype(np.float32)
    if _HAS_CV2:
        clump = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    elif _HAS_SCIPY:
        clump = ndi.zoom(coarse, (h / coarse.shape[0], w / coarse.shape[1]), order=1)
    else:
        clump = np.kron(coarse, np.ones((8, 8), dtype=np.float32))[:h, :w]
    clump = np.clip((clump - 0.58) * 3.4, 0, 1) * intensity

    speck = rng.random((h, w)).astype(np.float32)
    occlusion = np.clip(clump * (0.55 + 0.75 * speck), 0, 1)[..., None]
    dust_rgb = np.array([150, 141, 124], dtype=np.float32)

    blended = noisy * (1 - occlusion * 0.8) + dust_rgb * occlusion * 0.8
    mix = 0.45 + 0.55 * intensity
    out = arr * (1 - mix) + blended * mix
    return np.clip(out, 0, 255).astype(np.uint8)


_STORM_FNS: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "heat_haze": heat_haze,
    "strobe_glare": strobe_glare,
    "intake_dust": intake_dust,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing kernel implementations
# ─────────────────────────────────────────────────────────────────────────────


def clahe(img: np.ndarray) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation on the luma channel.

    Equalising RGB independently would shift hue; we convert to LAB (or a
    luma approximation without OpenCV), equalise L only, and convert back.
    """
    if _HAS_CV2:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        op = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        lab[..., 0] = op.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # NumPy fallback: tiled histogram equalisation with a clip limit,
    # bilinearly blended between tile centres to avoid blocking artefacts.
    arr = img.astype(np.float32)
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    h, w = luma.shape
    tiles = 8
    th, tw = max(1, h // tiles), max(1, w // tiles)
    out = np.empty_like(luma)

    for ty in range(tiles):
        for tx in range(tiles):
            y0, y1 = ty * th, (ty + 1) * th if ty < tiles - 1 else h
            x0, x1 = tx * tw, (tx + 1) * tw if tx < tiles - 1 else w
            block = luma[y0:y1, x0:x1]
            hist, edges = np.histogram(block, bins=64, range=(0, 255))
            clip = max(1, int(hist.sum() * 0.028))
            excess = np.clip(hist - clip, 0, None).sum()
            hist = np.minimum(hist, clip) + excess // 64
            cdf = np.cumsum(hist).astype(np.float32)
            cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1e-6) * 255.0
            out[y0:y1, x0:x1] = np.interp(block, edges[:-1], cdf)

    if _HAS_SCIPY:
        out = ndi.uniform_filter(out, size=5)

    gain = np.where(luma > 1e-3, out / np.maximum(luma, 1e-3), 1.0)[..., None]
    return np.clip(arr * np.clip(gain, 0.35, 2.6), 0, 255).astype(np.uint8)


def unsharp_mask(img: np.ndarray, amount: float = 1.15, sigma: float = 1.3) -> np.ndarray:
    """I' = I + k·(I − G_σ * I). Re-sharpens haze-smeared edges."""
    if _HAS_CV2:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    elif _HAS_SCIPY:
        blurred = np.stack(
            [ndi.gaussian_filter(img[..., c].astype(np.float32), sigma) for c in range(3)],
            axis=-1,
        )
    else:
        k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        k = k / k.sum()
        blurred = img.astype(np.float32)
        for axis in (0, 1):
            blurred = np.apply_along_axis(
                lambda m: np.convolve(m, k, mode="same"), axis, blurred
            )

    arr = img.astype(np.float32)
    return np.clip(arr + amount * (arr - np.asarray(blurred, dtype=np.float32)), 0, 255).astype(
        np.uint8
    )


def bilateral(img: np.ndarray, d: int = 5, sigma_colour: float = 48.0,
              sigma_space: float = 5.0) -> np.ndarray:
    """Edge-preserving denoise.

    The NumPy fallback is a shiftwise approximation: accumulate neighbourhood
    shifts weighted jointly by spatial distance and intensity difference. It is
    O(d²) passes over the image and genuinely slow, which is honest — a
    bilateral filter IS the expensive kernel in this bank, and the simulation
    should charge for it.
    """
    if _HAS_CV2:
        return cv2.bilateralFilter(img, d, sigma_colour, sigma_space)

    arr = img.astype(np.float32)
    radius = max(1, d // 2)
    luma = arr.mean(axis=2)

    acc = np.zeros_like(arr)
    wsum = np.zeros(arr.shape[:2], dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
            shifted_luma = np.roll(np.roll(luma, dy, axis=0), dx, axis=1)

            spatial = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_space**2))
            rng_w = np.exp(-((shifted_luma - luma) ** 2) / (2.0 * sigma_colour**2))
            weight = (spatial * rng_w).astype(np.float32)

            acc += shifted * weight[..., None]
            wsum += weight

    return np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0, 255).astype(np.uint8)


_FILTER_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clahe": clahe,
    "unsharp": unsharp_mask,
    "bilateral": bilateral,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def apply_storms(
    img: np.ndarray,
    active: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Apply every active storm. Returns (image, raw_severity, elapsed_ms).

    `active` maps storm key -> intensity in [0, 1].

    Severity does not add linearly. Two storms at full intensity are worse than
    either alone but not twice as bad, because they attack partly-overlapping
    cues. We combine them probabilistically:

        severity = 1 - Π(1 - w_i · intensity_i)

    which is the standard "probability that at least one thing went wrong"
    form, and keeps severity bounded in [0, 1) however many storms are stacked.
    """
    start = time.perf_counter()
    out = img
    survival = 1.0

    for key, intensity in active.items():
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity <= 0.0 or key not in _STORM_FNS:
            continue
        out = _STORM_FNS[key](out, intensity, rng)
        survival *= 1.0 - STORM_SPECS[key].severity_weight * intensity

    elapsed = (time.perf_counter() - start) * 1000.0
    return out, float(1.0 - survival), elapsed


def apply_filters(
    img: np.ndarray,
    active: Sequence[str] | Iterable[str],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Apply each enabled preprocessing kernel, timing it for real.

    Returns (image, total_ms, per_filter_ms). The total is charged directly to
    the queue's service time, which is the whole point: robustness costs
    latency out of the same 40 ms budget as inference.

    Order matters and is fixed: denoise before sharpen before equalise. Running
    unsharp before bilateral would amplify the grain the bilateral then has to
    remove, wasting both kernels.
    """
    order = ("bilateral", "unsharp", "clahe")
    active_set = {k for k in active if k in _FILTER_FNS}

    out = img
    per_filter: dict[str, float] = {}
    total = 0.0

    for key in order:
        if key not in active_set:
            continue
        start = time.perf_counter()
        out = _FILTER_FNS[key](out)
        ms = (time.perf_counter() - start) * 1000.0
        per_filter[key] = ms
        total += ms

    return out, total, per_filter


def residual_severity(raw_severity: float, active_filters: Iterable[str],
                      storm_intensities: dict[str, float]) -> float:
    """Severity remaining after the deployed kernels have done their work.

    Each filter neutralises a fraction of each storm's contribution, per its
    `counters` table. Mitigations compound multiplicatively rather than adding,
    so deploying all three kernels cannot drive severity to zero — there is no
    preprocessing stack that makes a weak backbone robust. That ceiling is the
    argument for capacity.
    """
    active = [k for k in active_filters if k in FILTER_SPECS]
    if not active:
        return float(np.clip(raw_severity, 0.0, 1.0))

    survival = 1.0
    for storm_key, intensity in storm_intensities.items():
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity <= 0 or storm_key not in STORM_SPECS:
            continue

        mitigation_survival = 1.0
        for fkey in active:
            mitigation_survival *= 1.0 - FILTER_SPECS[fkey].counters.get(storm_key, 0.0)

        effective = STORM_SPECS[storm_key].severity_weight * intensity * mitigation_survival
        survival *= 1.0 - effective

    return float(np.clip(1.0 - survival, 0.0, 1.0))


class StormBank:
    """Mutable storm/filter state for one live session."""

    def __init__(self) -> None:
        self.storms: dict[str, float] = {k: 0.0 for k in STORM_SPECS}
        self.filters: set[str] = set()

    # -- mutation -------------------------------------------------------------

    def set_storm(self, key: str, intensity: float) -> None:
        if key in self.storms:
            self.storms[key] = float(np.clip(intensity, 0.0, 1.0))

    def toggle_storm(self, key: str, default_intensity: float = 0.75) -> None:
        if key in self.storms:
            self.storms[key] = 0.0 if self.storms[key] > 0 else default_intensity

    def set_filter(self, key: str, enabled: bool) -> None:
        if key not in FILTER_SPECS:
            return
        if enabled:
            self.filters.add(key)
        else:
            self.filters.discard(key)

    def toggle_filter(self, key: str) -> None:
        if key in FILTER_SPECS:
            self.set_filter(key, key not in self.filters)

    def clear(self) -> None:
        self.storms = {k: 0.0 for k in STORM_SPECS}
        self.filters.clear()

    # -- queries --------------------------------------------------------------

    @property
    def active_storms(self) -> dict[str, float]:
        return {k: v for k, v in self.storms.items() if v > 0.0}

    @property
    def nominal_filter_cost_ms(self) -> float:
        return sum(FILTER_SPECS[k].nominal_cost_ms for k in self.filters)

    def analytic_severity(self) -> tuple[float, float]:
        """Effective severity without touching a single pixel.

        Returns ``(raw_severity, effective_severity)``.

        This exists because severity is a property of the storm *configuration*,
        not of any particular image: it is determined entirely by which storms
        are running, at what intensity, and which kernels are mitigating them.
        Computing it analytically lets the simulation classify frames it never
        renders, which is what keeps the 25 FPS loop real-time — at full storm
        load the pixel pipeline costs ~36 ms per frame, and the tick budget is
        40 ms total.

        The visual pipeline still runs, on the frames the player actually sees
        on the conveyor. Those get the real treatment. The rest are decided
        from the same severity number the rendered ones would have produced, so
        the confusion matrix is unaffected.
        """
        raw_survival = 1.0
        for key, intensity in self.storms.items():
            intensity = float(np.clip(intensity, 0.0, 1.0))
            if intensity > 0.0:
                raw_survival *= 1.0 - STORM_SPECS[key].severity_weight * intensity
        raw = 1.0 - raw_survival
        return float(raw), residual_severity(raw, self.filters, self.storms)

    def process(
        self, img: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run the full storm -> filter pipeline over one frame."""
        stormed, raw_sev, storm_ms = apply_storms(img, self.active_storms, rng)
        filtered, filter_ms, per_filter = apply_filters(stormed, self.filters)
        effective = residual_severity(raw_sev, self.filters, self.storms)

        return filtered, {
            "raw_severity": round(raw_sev, 4),
            "effective_severity": round(effective, 4),
            "mitigation": round(max(0.0, raw_sev - effective), 4),
            "storm_render_ms": round(storm_ms, 3),
            "filter_cost_ms": round(filter_ms, 3),
            "per_filter_ms": {k: round(v, 3) for k, v in per_filter.items()},
            "active_storms": {k: round(v, 3) for k, v in self.active_storms.items()},
            "active_filters": sorted(self.filters),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "storms": {k: round(v, 3) for k, v in self.storms.items()},
            "filters": sorted(self.filters),
            "nominal_filter_cost_ms": round(self.nominal_filter_cost_ms, 3),
        }


def specs_payload() -> dict[str, Any]:
    """Storm and filter catalogues, as sent to the frontend on connect."""
    return {
        "storms": [
            {
                "key": s.key,
                "display_name": s.display_name,
                "short": s.short,
                "accent": s.accent,
                "icon": s.icon,
                "severity_weight": s.severity_weight,
                "formula": s.formula,
                "blurb": s.blurb,
                "physics": s.physics,
            }
            for s in STORM_SPECS.values()
        ],
        "filters": [
            {
                "key": f.key,
                "display_name": f.display_name,
                "short": f.short,
                "accent": f.accent,
                "counters": f.counters,
                "nominal_cost_ms": f.nominal_cost_ms,
                "formula": f.formula,
                "blurb": f.blurb,
                "tradeoff": f.tradeoff,
            }
            for f in FILTER_SPECS.values()
        ],
        "backends": {"opencv": _HAS_CV2, "scipy": _HAS_SCIPY},
    }
