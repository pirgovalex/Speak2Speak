// single shared axios instance - all api calls go through this, never raw axios
import axios, {
  type AxiosInstance,
  type AxiosRequestConfig,
  type InternalAxiosRequestConfig,
  type AxiosResponse,
  type AxiosError,
} from 'axios';
import {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
} from './tokenService';
import { API_BASE_URL, AUTH_REFRESH_PATH } from '../config/env';
import { logoutUser } from '../stores/authStore';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15_000, // generous for llm inference latency
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

console.log('[docgraph api] axios instance ready. base url:', API_BASE_URL);

// request interceptor - injects bearer token if one exists
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    const accessToken = getAccessToken();
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
      console.log('[docgraph api] bearer token attached');
    } else {
      console.log('[docgraph api] no token - unauthenticated request');
    }
    return config;
  },
  (requestError: AxiosError) => {
    console.error('[docgraph api] request setup error:', requestError.message);
    return Promise.reject(requestError);
  }
);

// prevents duplicate refresh calls when multiple requests get a 401 simultaneously
let isRefreshing = false;

// queued requests waiting for a refresh to complete
let refreshQueue: Array<{
  resolve: (newToken: string) => void;
  reject: (error: unknown) => void;
}> = [];

// resolves or rejects all queued requests then empties the queue
const drainRefreshQueue = (drainError: unknown, newToken: string | null): void => {
  refreshQueue.forEach(({ resolve, reject }) => {
    if (newToken) resolve(newToken);
    else reject(drainError);
  });
  refreshQueue = [];
};

// response interceptor - handles 401 by refreshing the token and retrying once
apiClient.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => response,

  async (responseError: AxiosError): Promise<AxiosResponse> => {
    const originalRequest = responseError.config as AxiosRequestConfig & {
      _retried?: boolean;
    };

    const isUnauthorized = responseError.response?.status === 401;
    const alreadyRetried = originalRequest._retried === true;

    if (isUnauthorized && !alreadyRetried) {
      originalRequest._retried = true; // prevents infinite retry loop

      if (isRefreshing) {
        // another refresh already in flight - queue this request
        console.log('[docgraph auth] refresh in progress, queuing request');
        return new Promise<AxiosResponse>((resolve, reject) => {
          refreshQueue.push({
            resolve: (newToken: string) => {
              if (originalRequest.headers) {
                originalRequest.headers.Authorization = `Bearer ${newToken}`;
              }
              resolve(apiClient(originalRequest));
            },
            reject,
          });
        });
      }

      isRefreshing = true;
      const refreshToken = getRefreshToken();

      if (!refreshToken) {
        console.warn('[docgraph auth] no refresh token - forcing logout');
        clearTokens();
        logoutUser();
        drainRefreshQueue(responseError, null);
        isRefreshing = false;
        return Promise.reject(responseError);
      }

      try {
        console.log('[docgraph auth] attempting token refresh');

        // raw axios here - avoids triggering this interceptor recursively
        const refreshResponse = await axios.post<{
          access_token: string;
          refresh_token: string;
        }>(`${API_BASE_URL}${AUTH_REFRESH_PATH}`, { refresh_token: refreshToken });

        const { access_token: newAccessToken, refresh_token: newRefreshToken } =
          refreshResponse.data;

        setTokens(newAccessToken, newRefreshToken);
        console.log('[docgraph auth] refresh ok - replaying queued requests');

        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        }

        drainRefreshQueue(null, newAccessToken);
        isRefreshing = false;
        return apiClient(originalRequest);
      } catch (refreshError) {
        console.error('[docgraph auth] refresh failed:', refreshError);
        clearTokens();
        logoutUser();
        drainRefreshQueue(refreshError, null);
        isRefreshing = false;
        return Promise.reject(refreshError);
      }
    }

    console.error(
      '[docgraph api] response error:',
      responseError.response?.status ?? 'no_status',
      responseError.message
    );
    return Promise.reject(responseError);
  }
);

export default apiClient;
