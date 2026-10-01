import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";
import { loadCatalog } from "./scripts/catalog.mjs";

export default defineConfig(async ({ mode }) => {
  const publicBuild = mode === "public";
  const catalog = await loadCatalog(
    fileURLToPath(new URL(".", import.meta.url)),
    { publicBuild, catalogPath: process.env.FORMATION_CATALOG },
  );
  const registry = { seasons: {} };
  return {
    plugins: [
      react(),
      {
        name: "reviewed-forecast-catalog",
        resolveId(id) {
          if (
            ["virtual:forecast-catalog", "virtual:identity-registry"].includes(
              id,
            )
          )
            return "\0" + id;
        },
        load(id) {
          if (id === "\0virtual:forecast-catalog")
            return `export default ${JSON.stringify(catalog)}`;
          if (id === "\0virtual:identity-registry")
            return `export default ${JSON.stringify(registry)}`;
        },
        async generateBundle() {
          if (!publicBuild) return;
          for (const name of [
            "formation-car.glb",
            "car-fallback.png",
            "car-fallback-mobile.png",
            "car-fallback-provenance.json",
            "car-provenance.json",
            "barlow-condensed.ttf",
            "plex-mono.ttf",
            "barlow-OFL.txt",
            "plex-OFL.txt",
            "react-LICENSE.txt",
            "react-dom-LICENSE.txt",
            "three-LICENSE.txt",
            "gsap-NOTICE.txt",
            "lucide-license.txt",
          ])
            this.emitFile({
              type: "asset",
              fileName: `assets/${name}`,
              source: await readFile(
                new URL(`public/assets/${name}`, import.meta.url),
              ),
            });
        },
      },
    ],
    publicDir: publicBuild ? false : "public",
    server: {
      host: "127.0.0.1",
      watch: { ignored: ["**/.cache/**", "**/test-results/**", "**/qa/**"] },
      fs: {
        strict: true,
        allow: [fileURLToPath(new URL(".", import.meta.url))],
        deny: [".env", ".env.*", "*.{crt,pem}", "**/.git/**", "**/.cache/**"],
      },
    },
    build: {
      outDir: publicBuild ? "dist-public" : "dist",
      rollupOptions: {
        output: {
          manualChunks: {
            three: [
              "three",
              "three/addons/loaders/GLTFLoader.js",
              "three/addons/controls/OrbitControls.js",
              "three/addons/environments/RoomEnvironment.js",
            ],
            motion: ["gsap"],
          },
        },
      },
    },
  };
});
