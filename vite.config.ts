import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import os from 'os';

function getLocalIp() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name] || []) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return '127.0.0.1';
}

const LOCAL_IP = getLocalIp();
console.log(`🖥️ External IP: ${LOCAL_IP} | Proxy Target: http://127.0.0.1:8000`);

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    host: '0.0.0.0',
    port: 1420,
    strictPort: true,
    proxy: {
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/analyze": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/state": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/stop": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/preview": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/start": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ip": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/windows": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/windows/select": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});