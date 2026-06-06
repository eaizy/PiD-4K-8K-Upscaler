import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The backend runs on :17820. We proxy API + WS calls so the frontend can use
// same-origin relative paths (no CORS juggling in dev).
const BACKEND = process.env.PID_BACKEND ?? "http://127.0.0.1:17820";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        ws: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
