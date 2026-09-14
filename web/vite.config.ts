import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    sourcemap: false,
  },
});
