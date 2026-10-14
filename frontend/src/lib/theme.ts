import { useSyncExternalStore } from "react";

/**
 * Three terminal palettes. The CSS custom properties in global.css carry the
 * same values for the DOM; this copy exists for the two places that cannot read
 * CSS variables: the conveyor canvas and the Recharts SVG attributes.
 */
export type ThemeName = "terminal" | "dark" | "paper";

export interface Palette {
  bg: string;
  sunk: string;
  ink: string;
  muted: string;
  rule: string;
  accent: string;
  accentInk: string;
  warn: string;
  alarm: string;
  /** The swatch shown for this theme in the switcher. */
  swatch: string;
}

export const PALETTES: Record<ThemeName, Palette> = {
  terminal: {
    bg: "#e4e6dd",
    sunk: "#d7dacf",
    ink: "#1e201b",
    muted: "#5e6258",
    rule: "#c3c6b9",
    accent: "#76a01c",
    accentInk: "#4c6a0d",
    warn: "#946200",
    alarm: "#b3361e",
    swatch: "#7fa521",
  },
  dark: {
    bg: "#1c1e1a",
    sunk: "#272a24",
    ink: "#e4e6dd",
    muted: "#8d9186",
    rule: "#383c34",
    accent: "#8fbf2a",
    accentInk: "#a9d24a",
    warn: "#d9a33a",
    alarm: "#e2654a",
    swatch: "#4a4c48",
  },
  paper: {
    bg: "#fafaf7",
    sunk: "#eeeee9",
    ink: "#1a1a18",
    muted: "#6b6b65",
    rule: "#dadad3",
    accent: "#6f9a14",
    accentInk: "#4a6a0b",
    warn: "#946200",
    alarm: "#b3361e",
    swatch: "#f4f4ef",
  },
};

export const THEME_ORDER: ThemeName[] = ["terminal", "dark", "paper"];
const STORAGE_KEY = "dcv-theme";

function readInitial(): ThemeName {
  if (typeof document === "undefined") return "terminal";
  const attr = document.documentElement.dataset.theme as ThemeName | undefined;
  return attr && attr in PALETTES ? attr : "terminal";
}

let current: ThemeName = readInitial();
const listeners = new Set<() => void>();

export function setTheme(name: ThemeName): void {
  current = name;
  document.documentElement.dataset.theme = name;
  try {
    localStorage.setItem(STORAGE_KEY, name);
  } catch {
    /* storage unavailable: the choice lasts for this page view */
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTheme(): { name: ThemeName; palette: Palette } {
  const name = useSyncExternalStore(subscribe, () => current, () => "terminal" as ThemeName);
  return { name, palette: PALETTES[name] };
}
