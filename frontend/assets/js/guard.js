/**
 * Vyronex Navigation Guard & Header Component
 */

import { apiFetch, checkBackendConnection, escapeHtml, getToken, logout } from "./auth.js";

export async function requireAuth(expectedRole = null) {
  const token = getToken();
  if (!token) {
    window.location.replace("login.html");
    return null;
  }

  try {
    const res = await apiFetch("/api/auth/me");
    const user = await res.json();

    if (expectedRole && user.role !== expectedRole) {
      const redirectPage = user.role === "gym" ? "gym.html" : (user.role === "athlete" ? "athlete.html" : "index.html");
      window.location.replace(redirectPage);
      return null;
    }

    renderHeaderUser(user);
    return user;
  } catch (err) {
    console.error("Auth guard error:", err);
    if (!getToken()) {
      window.location.replace("login.html");
    } else {
      showBackendDownBanner();
    }
    return null;
  }
}

export function showBackendDownBanner() {
  if (document.getElementById("backend-warning-banner")) return;
  const banner = document.createElement("div");
  banner.id = "backend-warning-banner";
  banner.className = "backend-banner";
  banner.innerHTML = `⚠️ Vyronex backend is currently offline at http://localhost:8000. Start it with: <code>uvicorn app:app --reload --port 8000</code>`;
  document.body.prepend(banner);
}

export function renderHeaderUser(user) {
  const nameEl = document.getElementById("header-user-name");
  if (nameEl) {
    nameEl.textContent = user.name || (user.role === "gym" ? "Gym User" : "Athlete");
  }

  const logoutBtn = document.getElementById("header-logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", logout);
  }
}

// Check backend status automatically
document.addEventListener("DOMContentLoaded", async () => {
  const healthy = await checkBackendConnection();
  if (!healthy && !document.getElementById("backend-warning-banner")) {
    showBackendDownBanner();
  }
});
