// manages jwt tokens in sessionstorage - cleared when tab closes, safer than localstorage

const ACCESS_TOKEN_KEY = 'docgraph_access_token';
const REFRESH_TOKEN_KEY = 'docgraph_refresh_token';

// log presence only - never log the token value itself
export const getAccessToken = (): string | null => {
  const token = sessionStorage.getItem(ACCESS_TOKEN_KEY);
  console.log('[docgraph auth] access token present:', token !== null);
  return token;
};

export const getRefreshToken = (): string | null =>
  sessionStorage.getItem(REFRESH_TOKEN_KEY);

// stores both tokens after login or a successful refresh
export const setTokens = (accessToken: string, refreshToken: string): void => {
  sessionStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  sessionStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  console.log('[docgraph auth] tokens stored');
};

// called on logout or when refresh fails - forces re-login
export const clearTokens = (): void => {
  sessionStorage.removeItem(ACCESS_TOKEN_KEY);
  sessionStorage.removeItem(REFRESH_TOKEN_KEY);
  console.warn('[docgraph auth] tokens cleared');
};

// does not validate expiry - server handles that via 401
export const isAuthenticated = (): boolean =>
  sessionStorage.getItem(ACCESS_TOKEN_KEY) !== null;
