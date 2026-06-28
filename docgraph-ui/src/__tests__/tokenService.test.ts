// tests for sessionstorage-backed jwt token helpers
import { describe, it, expect, beforeEach } from 'vitest';
import {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
  isAuthenticated,
} from '../services/tokenService';

describe('tokenService', () => {
  // start every test with an empty sessionstorage
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('returns null when no token is stored', () => {
    expect(getAccessToken()).toBeNull();
  });

  it('stores and retrieves access token', () => {
    setTokens('access-123', 'refresh-456');
    expect(getAccessToken()).toBe('access-123');
  });

  it('stores and retrieves refresh token', () => {
    setTokens('access-123', 'refresh-456');
    expect(getRefreshToken()).toBe('refresh-456');
  });

  it('clears both tokens on clearTokens()', () => {
    setTokens('access-123', 'refresh-456');
    clearTokens();
    expect(getAccessToken()).toBeNull();
    expect(getRefreshToken()).toBeNull();
  });

  it('isAuthenticated returns false when no token', () => {
    expect(isAuthenticated()).toBe(false);
  });

  it('isAuthenticated returns true after setTokens()', () => {
    setTokens('access-123', 'refresh-456');
    expect(isAuthenticated()).toBe(true);
  });

  it('isAuthenticated returns false after clearTokens()', () => {
    setTokens('access-123', 'refresh-456');
    clearTokens();
    expect(isAuthenticated()).toBe(false);
  });
});
