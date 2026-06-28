// solidjs reactive store for auth state - components subscribe to these signals
import { createSignal } from 'solid-js';
import { isAuthenticated } from '../services/tokenService';

// seeded from sessionstorage so a page refresh doesn't log the user out
const [isLoggedIn, setIsLoggedIn] = createSignal<boolean>(isAuthenticated());
const [currentUser, setCurrentUser] = createSignal<string | null>(null);

console.log('[docgraph auth] store init. logged in:', isLoggedIn());

// called after a successful login response
export const loginSuccess = (username?: string): void => {
  setIsLoggedIn(true);
  if (username) setCurrentUser(username);
  console.log('[docgraph auth] login success:', username ?? 'unknown');
};

// called on logout or when token refresh fails
export const logoutUser = (): void => {
  setIsLoggedIn(false);
  setCurrentUser(null);
  console.warn('[docgraph auth] user logged out');
};

// exported as read-only - use action functions above to mutate
export { isLoggedIn, currentUser };
