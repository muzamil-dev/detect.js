// @ts-check
import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import svelte from "@astrojs/svelte";
import preact from "@astrojs/preact";
import vercel from "@astrojs/vercel";

// https://astro.build/config
export default defineConfig({
  output: "static",
  adapter: vercel({
    webAnalytics: {
      enabled: true,
    },
  }),
  integrations: [tailwind(), svelte(), preact()],
});
