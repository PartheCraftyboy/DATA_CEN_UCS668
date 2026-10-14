# MASTER FRONTEND PROMPT FOR CLAUDE OPUS

> **Role:** Principal Creative Frontend Engineer & Pixel-Art Game UI Architect  
> **Framework:** **Astro (v5+) + React 19 + Tailwind CSS**  
> **Visual Reference:** 16-bit / 32-bit SNES / GBA Pixel-Art Valley (Lush rolling plains, snow-capped alpine mountains, drifting pixel clouds, rocky cliff ledge with a winding dirt path)  
> **Theme:** "The Secret Mountain Edge-AI Observatory" (Retro Cartoon NPCs + Data Center Vision Arcade Cockpit)

---

## OBJECTIVE
Build a breathtaking, standalone, high-FPS interactive frontend using **Astro 5** and **React 19**. The application must feature a **multi-layer 16-bit parallax pixel-art scrolling world**, animated **retro cartoon pixel NPCs with mouse-following eye tracking**, and an integrated **arcade data-center vision telemetry cockpit**.

The frontend must include a **built-in mock simulation engine** so it runs out-of-the-box with full 60 FPS animations, interactive models, and live metrics without requiring an external backend.

---

## 1. VISUAL & AESTHETIC BLUEPRINT

### A. The 4-Layer 16-Bit Parallax Scrolling World
Create a 60 FPS continuous horizontal parallax engine (via HTML5 Canvas or hardware-accelerated CSS layers) replicating the scenic alpine mountain valley:
* **Layer 1 (Far Sky & Clouds)**: Drifting fluffy pixel clouds on deep azure sky ($v = 0.2\text{ px/frame}$).
* **Layer 2 (Snow-Capped Peaks)**: Jagged alpine mountain range silhouette with snow highlights ($v = 0.5\text{ px/frame}$).
* **Layer 3 (Midground Pine Forests & Plains)**: Rolling emerald green hills with scattered pixel conifer trees ($v = 1.2\text{ px/frame}$).
* **Layer 4 (Foreground Cliff & Path)**: Rocky cliff ledge with lush moss, wildflowers, and a winding dirt path ($v = 2.4\text{ px/frame}$).
* **Critical CSS Rule**: Every canvas, image, and sprite MUST enforce:
  ```css
  image-rendering: pixelated;
  image-rendering: crisp-edges;
  ```

### B. Retro Cartoon Pixel NPCs (Patrolling the Cliff)
Implement pixel-art animated sprites (SVG or Canvas-drawn) for:
1. **Skipper & Kowalski (Madagascar Penguins)**: Stationed at a mini pixel command desk, saluting and pointing at the meters.
2. **Courage the Cowardly Dog**: Patrols the edge; triggers a hilarious comic panic-shake animation whenever a "Thermal Hotspot" or "Queue Overflow" occurs!
3. **Pink Panther**: Smooth 4-frame lateral sneaking walk cycle along the dirt path with a magnifying glass.
4. **Oggy**: Chasing pixel bugs across the cables.
5. **Interactive Eye-Tracking**: Every character has expressive cartoon eyes with pupils that dynamically follow the user's cursor across the entire screen using `Math.atan2(deltaY, deltaX)` clamped to the eye boundary.

### C. The Arcade Vision Cockpit (HUD Overlay)
Overlaid on the landscape is a clean, semi-transparent 16-bit arcade mission control:
1. **Server Inspection Conveyor Belt**: Pixel server blades roll along the path at 25 FPS with color-coded bounding boxes showing classification tags (`OPTIMAL`, `THERMAL HOTSPOT`, `BENT CABLE`, `LED ERROR`, `DUST CLOG`).
2. **Edge Queue Battery Gauge**: An arcade 10-segment energy bar displaying the $M/D/1$ queue status ($Q(t) \in [0, 10]$). At $\ge 8$, the meter pulses neon red with flashing `QUEUE FULL!` warnings.
3. **Model Selector Cartridge Deck (React 19 Actions / `useOptimistic`)**:
   * `[1] SQUEEZE-BOT`: 4.2ms latency | 1.2M params | High speed, higher false alarms
   * `[2] MOBI-NET V2`: 11.5ms latency | 3.4M params | Optimal balance
   * `[3] RES-TITAN 18`: 44.0ms latency | 11.2M params | 98% accuracy, but overflows the queue!
4. **CRT 5x5 Confusion Matrix**: Styled like a green-phosphor arcade CRT monitor with scanline overlay, displaying live percentage heatmaps for the 5 inspection classes.
5. **Environmental Storm Controls**: Interactive arcade buttons to trigger "Heat Haze", "Strobe Glare", and "Dust Storm" with toggleable edge preprocessing filters (CLAHE, Denoise).

---

## 2. REPOSITORY & COMPONENT ARCHITECTURE

Structure the project cleanly within `frontend/`:

```text
frontend/
├── astro.config.mjs
├── tailwind.config.mjs
├── package.json
└── src/
    ├── layouts/
    │   └── Layout.astro                 # Loads Google Pixel Fonts: 'Press Start 2P', 'Silkscreen'
    ├── pages/
    │   └── index.astro                  # Main page hosting the Cockpit Island
    └── components/
        ├── Cockpit.tsx                  # Master React 19 State Container & Mock Engine
        ├── ParallaxWorld.tsx            # 4-layer Canvas parallax rendering engine
        ├── CartoonNPCs.tsx              # Character sprites, walk loops & eye-tracking hooks
        ├── InspectionConveyor.tsx       # 25 FPS server rack stream with bounding boxes
        ├── ModelCartridges.tsx          # React 19 interactive model selector cards
        ├── QueueMeter.tsx               # 10-segment pixel-art buffer gauge
        ├── ConfusionMatrixCRT.tsx       # 5x5 green-phosphor CRT heatmap
        ├── ParetoCurve.tsx              # Accuracy vs Latency vs FLOPs SVG graph
        └── AugmentationStorms.tsx       # Retro toggle buttons for environmental hazards
```

---

## 3. CORE TECHNICAL IMPLEMENTATION DETAILS

### A. The Cursor Eye-Tracking Hook (`useEyeTracker.ts`)
```typescript
import { useState, useEffect, RefObject } from "react";

export function useEyeTracker(ref: RefObject<HTMLElement>, maxRadius = 4) {
  const [pupil, setPupil] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const angle = Math.atan2(e.clientY - cy, e.clientX - cx);
      setPupil({
        x: Math.round(Math.cos(angle) * maxRadius),
        y: Math.round(Math.sin(angle) * maxRadius),
      });
    };
    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, [ref, maxRadius]);

  return pupil;
}
```

### B. The Standalone Simulation State Machine
In `Cockpit.tsx`, implement a built-in tick loop (`requestAnimationFrame` or `setInterval` at 40ms = 25 FPS):
* Generate arriving server components: 70% Optimal, 10% Thermal, 8% Bent Cable, 7% LED Error, 5% Dust Clog.
* Queue Dynamics:
  * When `SQUEEZE-BOT` is active: Service latency = 4.2ms $\implies$ Queue drains to 0.
  * When `MOBI-NET V2` is active: Service latency = 11.5ms $\implies$ Queue stays stable (1-3 items).
  * When `RES-TITAN 18` is active: Service latency = 44.0ms $\implies$ Queue fills up (+1 every 3s) until it hits 10, triggering dropped frames and Courage's panic shake!
* Confusion Matrix: Update rolling 5x5 matrix values in real-time as frames are processed.

### C. Retro Pixel Styling Tokens (Tailwind)
```javascript
// tailwind.config.mjs additions
theme: {
  extend: {
    fontFamily: {
      pixel: ['"Press Start 2P"', 'monospace'],
      display: ['"Silkscreen"', 'sans-serif'],
    },
    colors: {
      retro: {
        bg: '#0a101d',
        crt: '#0f380f',
        phosphor: '#8bac0f',
        amber: '#e69138',
        crimson: '#cc0000',
        cyan: '#00ffff',
      }
    }
  }
}
```

---

## 4. DELIVERABLES REQUIRED
1. Complete, self-contained Astro and React 19 source code.
2. Fully styled HTML5 Canvas or SVG parallax landscape code matching the video reference.
3. Cartoon NPC components with working walk cycles and dynamic cursor eye-tracking.
4. Interactive arcade controls with live feedback, audio-visual alert states, and a CRT confusion matrix.
