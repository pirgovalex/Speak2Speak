// centralises all env variable access - one place to audit config

// falls back to localhost so local dev works without a .env file
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export const AUTH_LOGIN_PATH = '/auth/login';
export const AUTH_REFRESH_PATH = '/auth/refresh';
export const QUERY_PATH = '/query';

console.log('[docgraph config] api base url:', API_BASE_URL);
