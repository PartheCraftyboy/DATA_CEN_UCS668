/** @type {import('tailwindcss').Config} */

/**
 * Terminal design tokens.
 *
 * Every colour resolves to a CSS custom property defined per theme in
 * src/styles/global.css, so a theme switch is one attribute change on <html>.
 * One typeface, one accent. Green means "healthy / selected"; warn and alarm
 * are reserved for the two kinds of trouble the simulator is about.
 */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,svelte,ts,tsx,vue}"],
  theme: {
    extend: {
      colors: {
        bg: v("bg"),
        sunk: v("sunk"),
        ink: v("ink"),
        muted: v("muted"),
        rule: v("rule"),
        accent: v("accent"),
        "accent-ink": v("accent-ink"),
        warn: v("warn"),
        alarm: v("alarm"),
      },
      fontFamily: {
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        sans: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      fontSize: {
        xs: ["0.6875rem", { lineHeight: "1rem" }],
        sm: ["0.8125rem", { lineHeight: "1.25rem" }],
        base: ["0.875rem", { lineHeight: "1.375rem" }],
      },
    },
  },
  plugins: [],
};
