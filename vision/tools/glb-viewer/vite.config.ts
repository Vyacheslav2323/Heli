import { defineConfig } from "vite";

function healthPlugin() {
  const respond = (_req: unknown, res: { setHeader: (k: string, v: string) => void; end: (body: string) => void }) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ ok: true }));
  };

  return {
    name: "health-endpoint",
    configureServer(server: { middlewares: { use: (path: string, handler: (req: unknown, res: { setHeader: (k: string, v: string) => void; end: (body: string) => void }) => void) => void } }) {
      server.middlewares.use("/health", (_req, res) => respond(_req, res));
    },
    configurePreviewServer(server: { middlewares: { use: (path: string, handler: (req: unknown, res: { setHeader: (k: string, v: string) => void; end: (body: string) => void }) => void) => void } }) {
      server.middlewares.use("/health", (_req, res) => respond(_req, res));
    },
  };
}

export default defineConfig({
  server: {
    port: 5173,
    open: true,
  },
  plugins: [healthPlugin()],
});
