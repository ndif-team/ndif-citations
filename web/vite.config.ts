import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8723',
        changeOrigin: true,
      },
      '/images': {
        target: 'http://127.0.0.1:8723',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        // Split heavy/shared vendor libs out of the entry chunk. recharts (charts-only,
        // pulled by the lazy Dashboard) and radix stay loadable on demand.
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (id.includes('recharts') || id.includes('d3-') || id.includes('victory-vendor')) return 'charts'
          if (id.includes('react-router') || id.includes('react-dom') || id.includes('/react/') || id.includes('scheduler')) return 'react-vendor'
          if (id.includes('@radix-ui')) return 'radix'
          if (id.includes('@tanstack')) return 'query'
        },
      },
    },
  },
})
