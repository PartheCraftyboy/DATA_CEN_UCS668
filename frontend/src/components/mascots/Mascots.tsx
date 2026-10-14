/**
 * THE INSPECTION CREW — original mascots, hand-authored SVG.
 *
 * Six characters drawn in the flat-vector cartoon idiom of the late 2000s /
 * early 2010s: heavy uniform outlines, flat fills with a single highlight,
 * bean-and-pear body shapes, noodle limbs ending in mitten hands, oversized
 * eyes doing all the acting. No gradients, no soft shadows, no filters —
 * everything is a path with a stroke, the way it would have been drawn in
 * Flash or Illustrator and exported for the web.
 *
 * These are original designs made for this project. Each of the five class
 * mascots maps to one inspection target and reacts live to what the classifier
 * is doing; PIP hosts the cockpit.
 *
 *   PIP      host     — treaded rover with a CRT head, the player's operator
 *   BLIP     class 0  — Optimal_Rack. Perpetually pleased. Thumbs up.
 *   SCORCH   class 1  — Thermal_Hotspot. On fire, about it.
 *   TANGLES  class 2  — Cable_Bend_Violation. Tied in a knot, patient about it.
 *   FLICKER  class 3  — Port_LED_Error. One eye, blinks when something's wrong.
 *   DUSTY    class 4  — Intake_Dust_Clogged. Asleep under a layer of lint.
 *
 * Every mascot takes the same props so they are interchangeable in a grid.
 */

import type { CSSProperties, ReactNode } from "react";

// ─────────────────────────────────────────────────────────────────────────────
//  Shared ink
// ─────────────────────────────────────────────────────────────────────────────

/** One ink colour for every outline — what makes the crew read as one set. */
const INK = "#0c1018";
const STROKE = 7;

export type MascotMood = "idle" | "alert" | "alarm" | "happy" | "sleep";

export interface MascotProps {
  /** Rendered pixel size (square-ish; the viewBox is 200×230). */
  size?: number;
  /** Drives expression and animation. */
  mood?: MascotMood;
  className?: string;
  style?: CSSProperties;
  title?: string;
}

interface ShellProps extends MascotProps {
  children: ReactNode;
}

/**
 * Common SVG shell. The mood lands as a data attribute so all motion lives in
 * CSS (global.css) rather than in JS timers — six mascots each running their
 * own requestAnimationFrame would be a waste of a frame budget we are
 * explicitly simulating the scarcity of.
 */
function Shell({
  size = 120,
  mood = "idle",
  className = "",
  style,
  title,
  children,
}: ShellProps) {
  return (
    <svg
      viewBox="0 0 200 230"
      width={size}
      height={(size * 230) / 200}
      className={`mascot mascot--${mood} ${className}`}
      style={style}
      role="img"
      aria-label={title}
      shapeRendering="geometricPrecision"
    >
      {title ? <title>{title}</title> : null}
      {/* Contact shadow — one flat ellipse, no blur, as it would have been. */}
      <ellipse cx="100" cy="214" rx="54" ry="9" fill="#05070c" opacity="0.45" />
      <g className="mascot__body">{children}</g>
    </svg>
  );
}

/** Mitten hand: a circle with a thumb notch. The crew's signature detail. */
function Mitten({
  x,
  y,
  r = 13,
  fill = "#f5f1e6",
  flip = false,
}: {
  x: number;
  y: number;
  r?: number;
  fill?: string;
  flip?: boolean;
}) {
  const t = flip ? -1 : 1;
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={fill} stroke={INK} strokeWidth={STROKE} />
      <path
        d={`M ${x + t * r * 0.55} ${y - r * 0.5} q ${t * r * 0.7} ${-r * 0.1} ${t * r * 0.45} ${r * 0.55}`}
        fill="none"
        stroke={INK}
        strokeWidth={STROKE - 2}
        strokeLinecap="round"
      />
    </g>
  );
}

/** Noodle limb: a stroked quadratic arc with round caps. */
function Noodle({
  d,
  stroke,
  width = 11,
}: {
  d: string;
  stroke: string;
  width?: number;
}) {
  return (
    <>
      <path d={d} fill="none" stroke={INK} strokeWidth={width + 6} strokeLinecap="round" />
      <path d={d} fill="none" stroke={stroke} strokeWidth={width} strokeLinecap="round" />
    </>
  );
}

/** Standard eye: white sclera, black pupil, optional catchlight. */
function Eye({
  cx,
  cy,
  rx = 15,
  ry = 17,
  px = 0,
  py = 0,
  pr = 6.5,
  lid = 0,
}: {
  cx: number;
  cy: number;
  rx?: number;
  ry?: number;
  px?: number;
  py?: number;
  pr?: number;
  /** 0 = wide open, 1 = fully closed. Drawn as a lid wedge, not a scale. */
  lid?: number;
}) {
  return (
    <g>
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="#fdfcf7" stroke={INK} strokeWidth={STROKE - 1} />
      <circle cx={cx + px} cy={cy + py} r={pr} fill={INK} />
      <circle cx={cx + px - pr * 0.4} cy={cy + py - pr * 0.45} r={pr * 0.33} fill="#fdfcf7" />
      {lid > 0 ? (
        <path
          d={`M ${cx - rx - 1} ${cy} a ${rx + 1} ${ry + 1} 0 0 1 ${2 * (rx + 1)} 0 Z`}
          transform={`translate(0 ${-ry + lid * 2 * ry})`}
          fill="currentColor"
          stroke={INK}
          strokeWidth={STROKE - 2}
        />
      ) : null}
    </g>
  );
}

/** A flat highlight patch — the one piece of shading the style allows. */
function Highlight({ d, opacity = 0.22 }: { d: string; opacity?: number }) {
  return <path d={d} fill="#ffffff" opacity={opacity} />;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PIP — the host rover
// ─────────────────────────────────────────────────────────────────────────────

export function Pip(props: MascotProps) {
  const mood = props.mood ?? "idle";
  const BODY = "#2bb3c9";
  const DARK = "#1b8296";

  return (
    <Shell {...props} title={props.title ?? "PIP, inspection rover"}>
      {/* Treads */}
      <g>
        <rect x="32" y="172" width="136" height="34" rx="17" fill={DARK} stroke={INK} strokeWidth={STROKE} />
        <circle cx="58" cy="189" r="9" fill="#0f1620" stroke={INK} strokeWidth={4} />
        <circle cx="100" cy="189" r="9" fill="#0f1620" stroke={INK} strokeWidth={4} />
        <circle cx="142" cy="189" r="9" fill="#0f1620" stroke={INK} strokeWidth={4} />
      </g>

      {/* Arms, drawn behind the chassis */}
      <Noodle d="M 44 128 q -26 10 -24 36" stroke={BODY} />
      <Noodle d="M 156 128 q 28 6 28 -22" stroke={BODY} />
      <Mitten x={20} y={168} r={14} />
      <Mitten x={184} y={100} r={14} flip />

      {/* Chassis */}
      <rect x="44" y="104" width="112" height="76" rx="16" fill={BODY} stroke={INK} strokeWidth={STROKE} />
      <Highlight d="M 56 114 h 34 v 56 h -34 a 8 8 0 0 1 -8 -8 v -40 a 8 8 0 0 1 8 -8 Z" />
      {/* Status lamps on the chassis */}
      <circle cx="104" cy="156" r="6" fill="#10b981" stroke={INK} strokeWidth={4} />
      <circle cx="124" cy="156" r="6" fill="#f59e0b" stroke={INK} strokeWidth={4} />
      <circle cx="144" cy="156" r="6" fill="#ef4444" stroke={INK} strokeWidth={4} />

      {/* Antenna */}
      <path d="M 100 68 V 44" stroke={INK} strokeWidth={8} strokeLinecap="round" />
      <circle className="mascot__beacon" cx="100" cy="36" r="10" fill="#f59e0b" stroke={INK} strokeWidth={STROKE - 1} />

      {/* CRT head */}
      <rect x="40" y="58" width="120" height="58" rx="14" fill="#e9e4d6" stroke={INK} strokeWidth={STROKE} />
      <rect x="52" y="68" width="96" height="38" rx="8" fill="#0d1a22" stroke={INK} strokeWidth={5} />
      {/* Scanline on the screen */}
      <path className="mascot__scanline" d="M 54 76 H 146" stroke="#2bb3c9" strokeWidth="3" opacity="0.5" />

      {/* Eyes drawn as screen phosphor, not physical eyeballs */}
      {mood === "sleep" ? (
        <>
          <path d="M 70 88 q 10 -8 20 0" fill="none" stroke="#45e0c0" strokeWidth={6} strokeLinecap="round" />
          <path d="M 110 88 q 10 -8 20 0" fill="none" stroke="#45e0c0" strokeWidth={6} strokeLinecap="round" />
        </>
      ) : (
        <g className="mascot__blink">
          <rect x="70" y="78" width="20" height={mood === "alarm" ? 22 : 18} rx="5" fill="#45e0c0" />
          <rect x="110" y="78" width="20" height={mood === "alarm" ? 22 : 18} rx="5" fill="#45e0c0" />
        </g>
      )}
      {mood === "alarm" ? (
        <path d="M 86 103 q 14 10 28 0" fill="none" stroke="#45e0c0" strokeWidth={5} strokeLinecap="round" />
      ) : (
        <path d="M 86 99 q 14 8 28 0" fill="none" stroke="#45e0c0" strokeWidth={5} strokeLinecap="round" />
      )}
    </Shell>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  BLIP — class 0, Optimal_Rack
// ─────────────────────────────────────────────────────────────────────────────

export function Blip(props: MascotProps) {
  const mood = props.mood ?? "idle";
  const BODY = "#3ddc97";
  const DARK = "#1fa873";

  return (
    <Shell {...props} title={props.title ?? "BLIP, all-clear"}>
      {/* Feet */}
      <ellipse cx="76" cy="196" rx="20" ry="11" fill="#f5f1e6" stroke={INK} strokeWidth={STROKE} />
      <ellipse cx="124" cy="196" rx="20" ry="11" fill="#f5f1e6" stroke={INK} strokeWidth={STROKE} />

      {/* Arms: one on the hip, one giving a thumbs up */}
      <Noodle d="M 54 132 q -24 16 -18 36" stroke={BODY} />
      <Noodle d="M 150 118 q 28 -12 22 -34" stroke={BODY} />
      <Mitten x={34} y={176} r={13} />
      {/* Thumbs up: a closed fist with the thumb cocked up and out, so it
          reads as a gesture rather than a pointing finger. */}
      <g transform="rotate(-10 174 88)">
        {/* Fist first, thumb on top of it, so the thumb actually reads. */}
        <circle cx="176" cy="96" r="15" fill="#f5f1e6" stroke={INK} strokeWidth={STROKE} />
        {/* Thumb sits high enough that the fist still reads underneath it. */}
        <path
          d="M 161 70 a 9 9 0 0 1 18 0 v 12 a 9 9 0 0 1 -18 0 Z"
          fill="#f5f1e6"
          stroke={INK}
          strokeWidth={STROKE - 1}
          strokeLinejoin="round"
        />
        {/* One crease where the curled fingers meet — not a stripe pattern. */}
        <path d="M 168 101 q 9 5 17 0" fill="none" stroke={INK} strokeWidth={4} strokeLinecap="round" opacity="0.75" />
      </g>

      {/* Body — a confident bean */}
      <path
        d="M 100 46 c 42 0 58 32 58 70 c 0 36 -22 60 -58 60 c -36 0 -58 -24 -58 -60 c 0 -38 16 -70 58 -70 Z"
        fill={BODY}
        stroke={INK}
        strokeWidth={STROKE}
      />
      <Highlight d="M 62 82 c 10 -22 26 -30 40 -30 c -22 6 -32 24 -36 48 Z" opacity={0.3} />
      {/* Belly plate, low enough to stay clear of the mouth */}
      <ellipse cx="100" cy="155" rx="27" ry="17" fill={DARK} stroke={INK} strokeWidth={5} opacity="0.75" />

      {/* Indicator bulb on top */}
      <path d="M 100 46 v -14" stroke={INK} strokeWidth={8} strokeLinecap="round" />
      <circle className="mascot__beacon" cx="100" cy="24" r="11" fill="#9bffd4" stroke={INK} strokeWidth={STROKE - 1} />

      {/* Face */}
      {mood === "sleep" ? (
        <>
          <path d="M 68 94 q 12 -10 24 0" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
          <path d="M 108 94 q 12 -10 24 0" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
        </>
      ) : (
        <g className="mascot__blink" color={BODY}>
          <Eye cx={76} cy={92} rx={14} ry={16} px={1} py={1} />
          <Eye cx={124} cy={92} rx={14} ry={16} px={1} py={1} />
        </g>
      )}
      {/* Big easy grin */}
      <path
        d={mood === "happy" ? "M 80 120 q 20 24 40 0 q -20 7 -40 0 Z" : "M 82 120 q 18 16 36 0"}
        fill={mood === "happy" ? INK : "none"}
        stroke={INK}
        strokeWidth={6}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Shell>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  SCORCH — class 1, Thermal_Hotspot
// ─────────────────────────────────────────────────────────────────────────────

export function Scorch(props: MascotProps) {
  const mood = props.mood ?? "alarm";
  const BODY = "#ff7a3d";
  const DARK = "#d9511c";

  return (
    <Shell {...props} title={props.title ?? "SCORCH, thermal hotspot"}>
      {/* Feet */}
      <ellipse cx="74" cy="198" rx="19" ry="10" fill="#f5f1e6" stroke={INK} strokeWidth={STROKE} />
      <ellipse cx="126" cy="198" rx="19" ry="10" fill="#f5f1e6" stroke={INK} strokeWidth={STROKE} />

      {/* Arms flung up in alarm */}
      <Noodle d="M 56 132 q -28 -16 -26 -48" stroke={BODY} />
      <Noodle d="M 144 132 q 28 -16 26 -48" stroke={BODY} />
      <Mitten x={28} y={76} r={13} />
      <Mitten x={172} y={76} r={13} flip />

      {/* Pear body */}
      <path
        d="M 100 56 c 34 0 46 26 50 60 c 4 34 -14 62 -50 62 c -36 0 -54 -28 -50 -62 c 4 -34 16 -60 50 -60 Z"
        fill={BODY}
        stroke={INK}
        strokeWidth={STROKE}
      />
      <Highlight d="M 66 94 c 8 -22 22 -30 34 -31 c -20 8 -28 24 -30 48 Z" opacity={0.26} />
      {/* Heat bands across the belly */}
      <path d="M 66 148 q 34 14 68 0" fill="none" stroke={DARK} strokeWidth={7} strokeLinecap="round" />
      <path d="M 70 164 q 30 12 60 0" fill="none" stroke={DARK} strokeWidth={6} strokeLinecap="round" opacity="0.8" />

      {/* Flame crest */}
      <g className="mascot__flame">
        <path
          d="M 100 20 c 12 16 20 22 20 34 c 0 12 -9 20 -20 20 c -11 0 -20 -8 -20 -20 c 0 -12 8 -18 20 -34 Z"
          fill="#ffd23f"
          stroke={INK}
          strokeWidth={STROKE - 1}
        />
        <path
          d="M 76 38 c 7 9 11 13 11 20 c 0 7 -5 11 -11 11 c -7 0 -11 -4 -11 -11 c 0 -7 4 -11 11 -20 Z"
          fill="#ff9d2e"
          stroke={INK}
          strokeWidth={5}
        />
        <path
          d="M 124 38 c 7 9 11 13 11 20 c 0 7 -5 11 -11 11 c -7 0 -11 -4 -11 -11 c 0 -7 4 -11 11 -20 Z"
          fill="#ff9d2e"
          stroke={INK}
          strokeWidth={5}
        />
      </g>

      {/* Panicked face */}
      <Eye cx={80} cy={102} rx={17} ry={19} px={1} py={3} pr={7} />
      <Eye cx={122} cy={102} rx={17} ry={19} px={-1} py={3} pr={7} />
      {/* Worried brows */}
      <path d="M 64 80 q 14 -8 28 -2" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
      <path d="M 136 80 q -14 -8 -28 -2" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
      {/* Open mouth */}
      <ellipse cx="101" cy="140" rx="15" ry="13" fill={INK} />
      <ellipse cx="101" cy="146" rx="9" ry="6" fill="#ff5d6c" />

      {/* Sweat bead */}
      <path
        className="mascot__sweat"
        d="M 150 92 c 6 9 9 12 9 17 a 9 9 0 0 1 -18 0 c 0 -5 3 -8 9 -17 Z"
        fill="#7dd3fc"
        stroke={INK}
        strokeWidth={5}
      />
    </Shell>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  TANGLES — class 2, Cable_Bend_Violation
// ─────────────────────────────────────────────────────────────────────────────

export function Tangles(props: MascotProps) {
  const mood = props.mood ?? "idle";
  const BODY = "#b06cf5";
  const CORD = "#8b3ddb";

  return (
    <Shell {...props} title={props.title ?? "TANGLES, cable bend violation"}>
      {/*
        The body IS the patch cord, and the defect is drawn literally: it runs
        down from the head into a loop far tighter than its minimum bend radius,
        then out to a connector. Over/under is drawn explicitly — the back half
        of the loop is stroked first, the crossing strand on top of it — because
        a knot that doesn't read as three-dimensional just looks like a blob.
      */}
      <g>
        {/* Back half of the loop */}
        <path
          d="M 96 124 C 52 132 44 182 84 184 C 112 186 124 164 112 152"
          fill="none"
          stroke={INK}
          strokeWidth={30}
          strokeLinecap="round"
        />
        <path
          d="M 96 124 C 52 132 44 182 84 184 C 112 186 124 164 112 152"
          fill="none"
          stroke={CORD}
          strokeWidth={20}
          strokeLinecap="round"
        />
        {/* Front strand crossing over, out to the connector */}
        <path
          d="M 96 158 C 118 176 150 174 164 162"
          fill="none"
          stroke={INK}
          strokeWidth={30}
          strokeLinecap="round"
        />
        <path
          d="M 96 158 C 118 176 150 174 164 162"
          fill="none"
          stroke={BODY}
          strokeWidth={20}
          strokeLinecap="round"
        />
        {/* LC connector boot at the loose end */}
        <g transform="rotate(-18 172 158)">
          <rect x="158" y="148" width="30" height="20" rx="5" fill="#f5f1e6" stroke={INK} strokeWidth={5} />
          <path d="M 188 153 h 8 v 10 h -8" fill="#f5f1e6" stroke={INK} strokeWidth={5} strokeLinejoin="round" />
        </g>
      </g>

      {/* Head — a rounded cube, like a strain-relief boot */}
      <path
        d="M 100 34 c 36 0 54 20 54 46 c 0 26 -18 44 -54 44 c -36 0 -54 -18 -54 -44 c 0 -26 18 -46 54 -46 Z"
        fill={BODY}
        stroke={INK}
        strokeWidth={STROKE}
      />
      <Highlight d="M 62 68 c 8 -18 22 -26 34 -27 c -20 8 -28 20 -30 42 Z" opacity={0.26} />

      {/* Antennae — two stray fiber strands */}
      <path d="M 76 38 q -10 -22 -26 -26" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
      <circle cx="48" cy="10" r="7" fill="#fcd34d" stroke={INK} strokeWidth={5} />
      <path d="M 124 38 q 10 -22 26 -26" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
      <circle cx="152" cy="10" r="7" fill="#22d3ee" stroke={INK} strokeWidth={5} />

      {/* Face — resigned */}
      {mood === "sleep" ? (
        <>
          <path d="M 70 76 q 12 -9 24 0" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
          <path d="M 106 76 q 12 -9 24 0" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
        </>
      ) : (
        <g className="mascot__blink" color={BODY}>
          <Eye cx={81} cy={74} rx={15} ry={17} px={-2} py={2} />
          <Eye cx={121} cy={74} rx={15} ry={17} px={-2} py={2} />
        </g>
      )}
      {/* Flat, long-suffering mouth */}
      <path
        d={mood === "alarm" ? "M 82 102 q 18 -14 38 0" : "M 82 100 h 36"}
        fill="none"
        stroke={INK}
        strokeWidth={6}
        strokeLinecap="round"
      />
    </Shell>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  FLICKER — class 3, Port_LED_Error
// ─────────────────────────────────────────────────────────────────────────────

export function Flicker(props: MascotProps) {
  const mood = props.mood ?? "alert";
  const BODY = "#fbbf24";
  const DARK = "#d97706";

  return (
    <Shell {...props} title={props.title ?? "FLICKER, port link fault"}>
      {/* Six little legs, kicking out past the shell so they stay visible */}
      {[0, 1, 2].map((i) => {
        // Start below the eye line, so they read as legs rather than as a
        // second set of signal arcs.
        const y0 = 124 + i * 20;
        const reach = 26 + i * 5;
        const drop = 30 - i * 6;
        return (
          <g key={i}>
            <path
              d={`M 58 ${y0} q ${-reach} 10 ${-reach - 4} ${drop}`}
              fill="none"
              stroke={INK}
              strokeWidth={9}
              strokeLinecap="round"
            />
            <path
              d={`M 142 ${y0} q ${reach} 10 ${reach + 4} ${drop}`}
              fill="none"
              stroke={INK}
              strokeWidth={9}
              strokeLinecap="round"
            />
          </g>
        );
      })}

      {/* Signal arcs — side lobes up by the eye, not closed rings, so it reads
          as a transmission rather than a cage around the character. */}
      <g className="mascot__pulse">
        <path d="M 22 66 a 58 58 0 0 0 -4 44" fill="none" stroke={DARK} strokeWidth={6} strokeLinecap="round" opacity="0.6" />
        <path d="M 178 66 a 58 58 0 0 1 4 44" fill="none" stroke={DARK} strokeWidth={6} strokeLinecap="round" opacity="0.6" />
      </g>

      {/* Chassis body — an oval bug shell */}
      <ellipse cx="100" cy="112" rx="58" ry="62" fill={BODY} stroke={INK} strokeWidth={STROKE} />
      <Highlight d="M 62 86 c 10 -22 26 -30 38 -31 c -22 8 -32 24 -34 50 Z" opacity={0.28} />
      {/* Shell seam */}
      <path d="M 100 52 v 120" stroke={INK} strokeWidth={5} opacity="0.55" />
      {/* Port strip across the lower shell */}
      <rect x="66" y="142" width="68" height="18" rx="6" fill="#1a1f2b" stroke={INK} strokeWidth={5} />
      {[0, 1, 2, 3].map((i) => (
        <rect
          key={i}
          x={73 + i * 15}
          y={147}
          width="9"
          height="8"
          rx="2"
          fill={i === 2 ? "#ef4444" : "#10b981"}
        />
      ))}

      {/* Antennae */}
      <path d="M 76 56 q -12 -26 -28 -32" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
      <circle cx="46" cy="20" r="8" fill="#ef4444" stroke={INK} strokeWidth={5} />
      <path d="M 124 56 q 12 -26 28 -32" fill="none" stroke={INK} strokeWidth={6} strokeLinecap="round" />
      <circle cx="154" cy="20" r="8" fill="#10b981" stroke={INK} strokeWidth={5} />

      {/* One enormous eye */}
      {mood === "sleep" ? (
        <path d="M 72 98 q 28 -20 56 0" fill="none" stroke={INK} strokeWidth={7} strokeLinecap="round" />
      ) : (
        <g className="mascot__blink" color={BODY}>
          <circle cx="100" cy="96" r="30" fill="#fdfcf7" stroke={INK} strokeWidth={STROKE} />
          <circle cx="100" cy="96" r={mood === "alarm" ? 10 : 14} fill={INK} />
          <circle cx="92" cy="88" r="6" fill="#fdfcf7" />
        </g>
      )}
    </Shell>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  DUSTY — class 4, Intake_Dust_Clogged
// ─────────────────────────────────────────────────────────────────────────────

export function Dusty(props: MascotProps) {
  const mood = props.mood ?? "sleep";
  const BODY = "#a8adba";
  const DARK = "#7d8392";

  /** Scalloped fur silhouette, built from arcs around a circle. */
  const tufts = 16;
  const R = 62;
  let fur = "";
  for (let i = 0; i <= tufts; i++) {
    const a0 = (i / tufts) * Math.PI * 2 - Math.PI / 2;
    const x = 100 + Math.cos(a0) * R;
    const y = 118 + Math.sin(a0) * R;
    fur += i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)} ` : `A 11 11 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)} `;
  }
  fur += "Z";

  return (
    <Shell {...props} title={props.title ?? "DUSTY, clogged intake"}>
      {/* Stubby feet */}
      <ellipse cx="80" cy="190" rx="17" ry="10" fill="#e4e0d4" stroke={INK} strokeWidth={STROKE} />
      <ellipse cx="120" cy="190" rx="17" ry="10" fill="#e4e0d4" stroke={INK} strokeWidth={STROKE} />

      {/* Arms, drawn behind the fur so they emerge from the silhouette */}
      <Noodle d="M 54 150 q -22 18 -30 28" stroke={DARK} width={10} />
      <Noodle d="M 146 150 q 22 18 30 28" stroke={DARK} width={10} />
      <Mitten x={18} y={184} r={12} fill="#e4e0d4" />
      <Mitten x={182} y={184} r={12} fill="#e4e0d4" flip />

      {/* Fur ball */}
      <path d={fur} fill={BODY} stroke={INK} strokeWidth={STROKE} strokeLinejoin="round" />
      <Highlight d="M 62 94 c 12 -22 28 -28 40 -29 c -22 10 -32 26 -34 50 Z" opacity={0.3} />
      {/* Lint strands */}
      <path d="M 70 136 q 16 8 34 2" fill="none" stroke={DARK} strokeWidth={5} strokeLinecap="round" />
      <path d="M 104 152 q 18 6 32 -4" fill="none" stroke={DARK} strokeWidth={5} strokeLinecap="round" />

      {/* Sleepy face */}
      {mood === "sleep" || mood === "idle" ? (
        <>
          <path d="M 68 112 q 14 12 28 0" fill="none" stroke={INK} strokeWidth={7} strokeLinecap="round" />
          <path d="M 104 112 q 14 12 28 0" fill="none" stroke={INK} strokeWidth={7} strokeLinecap="round" />
        </>
      ) : (
        <g className="mascot__blink" color={BODY}>
          <Eye cx={80} cy={110} rx={14} ry={15} py={3} lid={0.35} />
          <Eye cx={120} cy={110} rx={14} ry={15} py={3} lid={0.35} />
        </g>
      )}
      {/* Small open snore mouth */}
      <ellipse cx="100" cy="142" rx="10" ry={mood === "alarm" ? 12 : 8} fill={INK} />

      {/* Drifting dust motes */}
      <g className="mascot__motes">
        <circle cx="166" cy="70" r="5" fill={DARK} stroke={INK} strokeWidth={3} />
        <circle cx="184" cy="46" r="3.5" fill={DARK} stroke={INK} strokeWidth={3} />
        <circle cx="38" cy="58" r="4" fill={DARK} stroke={INK} strokeWidth={3} />
      </g>

      {/* Sleep bubble */}
      {mood === "sleep" ? (
        <g className="mascot__snore">
          <text x="150" y="40" fontFamily="'Fira Code', monospace" fontSize="26" fontWeight="700" fill={INK}>
            z
          </text>
          <text x="172" y="22" fontFamily="'Fira Code', monospace" fontSize="18" fontWeight="700" fill={INK}>
            z
          </text>
        </g>
      ) : null}
    </Shell>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Registry
// ─────────────────────────────────────────────────────────────────────────────

export interface CrewMember {
  classId: number;
  codename: string;
  role: string;
  accent: string;
  Component: (props: MascotProps) => JSX.Element;
  /** One line, in character, shown when this class is detected. */
  callout: string;
}

export const CREW: CrewMember[] = [
  {
    classId: 0,
    codename: "BLIP",
    role: "All clear",
    accent: "#3ddc97",
    Component: Blip,
    callout: "Rack's happy. Nothing to see here.",
  },
  {
    classId: 1,
    codename: "SCORCH",
    role: "Thermal hotspot",
    accent: "#ff7a3d",
    Component: Scorch,
    callout: "It's VERY warm over here!",
  },
  {
    classId: 2,
    codename: "TANGLES",
    role: "Cable bend violation",
    accent: "#b06cf5",
    Component: Tangles,
    callout: "Someone kinked the fiber. Again.",
  },
  {
    classId: 3,
    codename: "FLICKER",
    role: "Port LED error",
    accent: "#fbbf24",
    Component: Flicker,
    callout: "Port three is blinking red at me.",
  },
  {
    classId: 4,
    codename: "DUSTY",
    role: "Intake dust clog",
    accent: "#a8adba",
    Component: Dusty,
    callout: "...zzz... can't breathe... zzz...",
  },
];

export const CREW_BY_CLASS: Record<number, CrewMember> = Object.fromEntries(
  CREW.map((c) => [c.classId, c]),
) as Record<number, CrewMember>;

export { Pip as Host };
