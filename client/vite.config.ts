import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5174,
    open: true,
    proxy: {
      "/stt": {
        target: "http://127.0.0.1:8010",
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/stt/, ""),
      },
      "/sam": {
        target: "http://127.0.0.1:7860",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/sam/, ""),
      },
      "/track": {
        target: "http://127.0.0.1:7861",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/track/, ""),
      },
      "/viewer": {
        target: "http://127.0.0.1:5173",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/viewer/, ""),
      },
    },
  },
});
