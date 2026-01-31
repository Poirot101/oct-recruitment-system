import { defineConfig } from 'vite';

export default defineConfig({
  // Multi-page app configuration
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        student: 'student.html',
        recruiter: 'recruiter.html',
        admin: 'admin.html'
      }
    }
  },
  // For local development server
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true
      }
    }
  }
});
