export const POSE_VERSION = "0.5.1675469404";
const POSE_SCRIPT = `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/pose.js`;

export function loadPoseScript() {
  if (typeof window.Pose === "function") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.querySelector(`script[src="${POSE_SCRIPT}"]`) || document.createElement("script");
    const timer = window.setTimeout(() => reject(new Error("The MediaPipe script load timed out.")), 15000);
    script.addEventListener("load", () => {
      window.clearTimeout(timer);
      typeof window.Pose === "function" ? resolve() : reject(new Error("MediaPipe loaded without Pose."));
    }, { once: true });
    script.addEventListener("error", () => {
      window.clearTimeout(timer);
      reject(new Error("The MediaPipe script could not be downloaded."));
    }, { once: true });
    if (!script.src) {
      script.src = POSE_SCRIPT;
      script.crossOrigin = "anonymous";
      document.head.appendChild(script);
    }
  });
}
