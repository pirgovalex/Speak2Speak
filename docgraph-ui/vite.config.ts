import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

// vitest ships its own vite build - vite-plugin-solid cjs crashes in that context
// so only load the solid plugin when not running tests
const isTest = process.env.VITEST === 'true' || process.env.NODE_ENV === 'test';

export default defineConfig({
  plugins: isTest ? [] : [solidPlugin()],
  resolve: {
    // .jsx intentionally excluded - this is a typescript-only project
    extensions: ['.tsx', '.ts', '.js'],
  },
  server: {
    port: 3000,
  },
  build: {
    target: 'esnext',
  },
  test: {
    environment: 'happy-dom',
    globals: true,
  },
});
