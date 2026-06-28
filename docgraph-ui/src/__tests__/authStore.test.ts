// tests for the solidjs reactive auth store - resets module state between tests via vi.resetModules
import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('authStore', () => {
  // clear storage and module registry so each test gets fresh signals
  beforeEach(() => {
    sessionStorage.clear();
    vi.resetModules();
  });

  it('loginSuccess sets isLoggedIn to true', async () => {
    const { loginSuccess, isLoggedIn } = await import('../stores/authStore');
    loginSuccess('testuser');
    expect(isLoggedIn()).toBe(true);
  });

  it('logoutUser sets isLoggedIn to false', async () => {
    const { loginSuccess, logoutUser, isLoggedIn } = await import('../stores/authStore');
    loginSuccess('testuser');
    logoutUser();
    expect(isLoggedIn()).toBe(false);
  });

  it('loginSuccess sets currentUser', async () => {
    const { loginSuccess, currentUser } = await import('../stores/authStore');
    loginSuccess('alice');
    expect(currentUser()).toBe('alice');
  });

  it('logoutUser clears currentUser', async () => {
    const { loginSuccess, logoutUser, currentUser } = await import('../stores/authStore');
    loginSuccess('alice');
    logoutUser();
    expect(currentUser()).toBeNull();
  });
});
