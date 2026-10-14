// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";

/**
 * Astro 5 — island architecture.
 *
 * The page shell (layout, meta, fonts, the static chrome around the cockpit)
 * ships as zero-JS HTML. Only the interactive panels hydrate, and each one is
 * its own island, so a heavy panel re-rendering at 25 FPS cannot stall the
 * rest of the page.
 *
 * `output: "static"` because the cockpit is driven entirely by the WebSocket —
 * there is no server-side data to fetch per request, so there is nothing for
 * SSR to do. The whole frontend is a CDN-cacheable bundle that happens to open
 * a socket on load.
 */
export default defineConfig({
  integrations: [react(), tailwind({ applyBaseStyles: false })],
  output: "static",
  server: { port: 4321, host: true },
  vite: {
    build: {
      // Recharts is the single biggest dependency; splitting it out keeps the
      // cockpit's own code in a chunk that invalidates independently.
      rollupOptions: {
        output: {
          manualChunks: {
            recharts: ["recharts"],
          },
        },
      },
    },
  },
});
