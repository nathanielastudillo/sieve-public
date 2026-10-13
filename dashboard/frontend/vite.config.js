import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8096',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8096',
        ws: true,
      },
    },
  },
});
