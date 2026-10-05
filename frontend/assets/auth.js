const API_BASE_URL = "http://localhost:8000";
const TOKEN_KEY = "vyronex_access_token";

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function logout() {
  sessionStorage.removeItem(TOKEN_KEY);
  window.location.assign("login.html");
}

function rolePage(role) {
  return role === "gym" ? "gym.html" : "athlete.html";
}

async function responseError(response) {
  let message = `Request failed (${response.status})`;
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      message = payload.detail;
    } else if (Array.isArray(payload.detail)) {
      message = payload.detail.map((item) => item.msg).join(". ");
    }
  } catch {
    // Keep the HTTP status message when the response has no JSON body.
  }
  return new Error(message);
}

export async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(
    url.startsWith("http") ? url : `${API_BASE_URL}${url}`,
    { ...options, headers },
  );

  if (response.status === 401) {
    sessionStorage.removeItem(TOKEN_KEY);
    window.location.assign("login.html");
    throw new Error("Your session has expired. Please log in again.");
  }
  if (!response.ok) {
    throw await responseError(response);
  }
  return response;
}

export async function signup(formData) {
  const response = await apiFetch("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify(formData),
  });
  const result = await response.json();
  setToken(result.access_token);
  window.location.assign(rolePage(result.role || result.user.role));
  return result;
}

export async function login(formData) {
  const response = await apiFetch("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(formData),
  });
  const result = await response.json();
  setToken(result.access_token);
  window.location.assign(rolePage(result.role || result.user.role));
  return result;
}

export async function requireAuth(expectedRole, page = null) {
  if (!getToken()) {
    window.location.replace("login.html");
    return null;
  }

  try {
    const response = await apiFetch("/api/auth/me");
    const user = await response.json();
    if (expectedRole && user.role !== expectedRole) {
      window.location.replace(rolePage(user.role));
      return null;
    }
    return user;
  } catch (error) {
    if (getToken()) {
      const message = document.querySelector("[data-auth-error]");
      if (message) {
        message.textContent = error.message;
        message.hidden = false;
      }
      if (page) page.hidden = false;
    }
    return null;
  }
}

function showFormError(form, message) {
  const error = form.querySelector("[data-form-error]");
  error.textContent = message;
  error.hidden = false;
}

function clearFormError(form) {
  const error = form.querySelector("[data-form-error]");
  error.textContent = "";
  error.hidden = true;
}

const signupForm = document.querySelector("#signup-form");
if (signupForm) {
  const submitButton = signupForm.querySelector('button[type="submit"]');
  signupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearFormError(signupForm);
    if (!signupForm.reportValidity()) return;

    const form = new FormData(signupForm);
    const password = String(form.get("password"));
    if (password.length < 8) {
      showFormError(signupForm, "Password must be at least 8 characters.");
      return;
    }
    if (password !== form.get("confirm_password")) {
      showFormError(signupForm, "Passwords do not match.");
      signupForm.elements.confirm_password.focus();
      return;
    }

    const payload = {
      name: String(form.get("name")).trim(),
      email: String(form.get("email")).trim(),
      password,
      role: String(form.get("role")),
    };
    for (const field of ["age", "height_cm", "weight_kg"]) {
      const value = form.get(field);
      if (value) payload[field] = Number(value);
    }

    submitButton.disabled = true;
    submitButton.textContent = "Creating account...";
    try {
      await signup(payload);
    } catch (error) {
      showFormError(signupForm, error.message);
      submitButton.disabled = false;
      submitButton.textContent = "Create account";
    }
  });
}

const loginForm = document.querySelector("#login-form");
if (loginForm) {
  const submitButton = loginForm.querySelector('button[type="submit"]');
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearFormError(loginForm);
    if (!loginForm.reportValidity()) return;

    const form = new FormData(loginForm);
    submitButton.disabled = true;
    submitButton.textContent = "Signing in...";
    try {
      await login({
        email: String(form.get("email")).trim(),
        password: String(form.get("password")),
      });
    } catch (error) {
      showFormError(loginForm, error.message);
      submitButton.disabled = false;
      submitButton.textContent = "Log in";
    }
  });
}

for (const button of document.querySelectorAll("[data-logout]")) {
  button.addEventListener("click", logout);
}
