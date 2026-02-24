import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const isDev = mode === "development";

  return {
    plugins: [react()],

    resolve: {
      alias: {
        "@": path.resolve(__dirname, "."),
      },
    },

    server: {
      port: 3000,
      host: "0.0.0.0",

      // Chỉ dùng proxy khi dev. Production: dùng VITE_API_URL trong .env.production
      proxy: isDev
        ? { "/api": "http://localhost:8000" }
        : undefined,
    },
  };
});