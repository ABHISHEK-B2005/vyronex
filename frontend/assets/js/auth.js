/**
 * Vyronex Auth & API Utility Module
 */

export const API_BASE_URL = "http://localhost:8000";
export const TOKEN_KEY = "vyronex_session_token";

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) {
    sessionStorage.setItem(TOKEN_KEY, token);
  }
}

export function logout() {
  sessionStorage.removeItem(TOKEN_KEY);
  window.location.assign("login.html");
}

export function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export async function checkBackendConnection() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function apiFetch(endpoint, options = {}) {
  const url = endpoint.startsWith("http") ? endpoint : `${API_BASE_URL}${endpoint}`;
  const headers = new Headers(options.headers || {});
  
  const token = getToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
      sessionStorage.removeItem(TOKEN_KEY);
      window.location.assign("login.html");
      throw new Error("Session expired. Please log in again.");
    }

    if (!response.ok) {
      let errorMsg = `Request failed (${response.status})`;
      try {
        const data = await response.json();
        if (typeof data.detail === "string") errorMsg = data.detail;
        else if (Array.isArray(data.detail)) errorMsg = data.detail.map(d => d.msg).join(". ");
      } catch {
        // No JSON body in response
      }
      const error = new Error(errorMsg);
      error.status = response.status;
      throw error;
    }

    return response;
  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      throw new Error("Unable to connect to Vyronex backend server (http://localhost:8000). Please ensure the backend is running.");
    }
    throw err;
  }
}
