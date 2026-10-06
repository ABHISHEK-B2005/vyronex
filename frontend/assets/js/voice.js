/**
 * Vyronex Voice Coaching Assistant (Web Speech API)
 */

let lastSpokenCue = "";
let lastSpokenTime = 0;
const SPEECH_COOLDOWN_MS = 2500;

export function speakCue(text, force = false) {
  if (!('speechSynthesis' in window) || !text) return;

  const now = Date.now();
  const cleanText = text.trim();

  // Only speak if cue changed or cooldown expired
  if (!force && cleanText === lastSpokenCue && (now - lastSpokenTime < SPEECH_COOLDOWN_MS)) {
    return;
  }

  // Cancel any ongoing utterance so voice is immediate
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = 1.05;
  utterance.pitch = 1.0;
  utterance.volume = 0.9;

  lastSpokenCue = cleanText;
  lastSpokenTime = now;

  window.speechSynthesis.speak(utterance);
}

export function cancelVoice() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
}
