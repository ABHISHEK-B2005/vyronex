/**
 * Vyronex AI Movement Coach Chat Client
 */

import { apiFetch, escapeHtml } from "./auth.js";

export function initCoachChat() {
  const form = document.getElementById("coach-chat-form");
  const input = document.getElementById("coach-user-input");
  const messagesBox = document.getElementById("chat-messages");

  if (!form || !input || !messagesBox) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    // 1. Append user message
    appendMessage(messagesBox, text, "user");
    input.value = "";

    // 2. Show typing indicator
    const typingEl = showTypingIndicator(messagesBox);

    try {
      const res = await apiFetch("/api/coach/chat", {
        method: "POST",
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      
      typingEl.remove();
      appendMessage(messagesBox, data.reply, "assistant", data.disclaimer);
    } catch (err) {
      typingEl.remove();
      appendMessage(messagesBox, `Error: ${err.message}`, "error");
    }
  });
}

function appendMessage(container, text, role, disclaimer = null) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg ${role === 'user' ? 'msg-user' : 'msg-assistant'}`;
  
  if (role === "user") {
    msgDiv.style.cssText = "align-self: flex-end; background: var(--red); color: #fff; padding: 12px 18px; border-radius: 14px 14px 2px 14px; max-width: 80%; margin-bottom: 12px; font-weight: 500;";
    msgDiv.textContent = text;
  } else if (role === "assistant") {
    msgDiv.style.cssText = "align-self: flex-start; background: #FFFFFF; border: 1px solid var(--line); color: var(--ink); padding: 14px 18px; border-radius: 14px 14px 14px 2px; max-width: 85%; margin-bottom: 12px; box-shadow: var(--shadow-sm);";
    msgDiv.innerHTML = `${escapeHtml(text).replace(/\n/g, '<br>')}`;
    if (disclaimer) {
      msgDiv.innerHTML += `<div style="font-size: 0.78rem; color: var(--muted); margin-top: 8px; border-top: 1px solid var(--line); padding-top: 6px;">* ${escapeHtml(disclaimer)}</div>`;
    }
  } else {
    msgDiv.style.cssText = "align-self: flex-start; background: #fff1f2; border: 1px solid #fecdd3; color: var(--red-deep); padding: 10px 14px; border-radius: 8px; max-width: 85%; margin-bottom: 12px;";
    msgDiv.textContent = text;
  }

  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function showTypingIndicator(container) {
  const typingDiv = document.createElement("div");
  typingDiv.className = "chat-msg msg-assistant";
  typingDiv.style.cssText = "align-self: flex-start; background: #f0f7ff; border: 1px solid var(--line); color: var(--blue); padding: 8px 14px; border-radius: 8px; margin-bottom: 12px;";
  typingDiv.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px; vertical-align: middle; margin-right: 6px;"></span> Coach is analyzing...`;
  container.appendChild(typingDiv);
  container.scrollTop = container.scrollHeight;
  return typingDiv;
}
