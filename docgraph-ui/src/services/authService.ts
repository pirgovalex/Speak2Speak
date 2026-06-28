// high-level auth calls - components never touch axios or tokens directly
import axios from 'axios';
import { setTokens, clearTokens } from './tokenService';
import { loginSuccess, logoutUser } from '../stores/authStore';
import { API_BASE_URL, AUTH_LOGIN_PATH } from '../config/env';

export interface LoginCredentials {
  username: string;
  password: string;
}

interface LoginResponse {
  access_token: string;
  refresh_token: string;
  username: string;
}

// raw axios - login has no bearer token yet, no need for apiClient interceptors
export const login = async (credentials: LoginCredentials): Promise<void> => {
  console.log('[docgraph auth] login attempt:', credentials.username);
  try {
    const response = await axios.post<LoginResponse>(
      `${API_BASE_URL}${AUTH_LOGIN_PATH}`,
      credentials
    );
    const { access_token, refresh_token, username } = response.data;
    setTokens(access_token, refresh_token);
    loginSuccess(username);
    console.log('[docgraph auth] login ok:', username);
  } catch (loginError) {
    console.error('[docgraph auth] login failed:', loginError);
    throw loginError; // re-throw so the ui component can show feedback
  }
};

export const logout = (): void => {
  console.warn('[docgraph auth] logging out');
  clearTokens();
  logoutUser();
};
