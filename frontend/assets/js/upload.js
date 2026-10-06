/**
 * Vyronex Video Upload & Background Analysis Handler
 */

import { apiFetch, API_BASE_URL, getToken } from "./auth.js";

const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB
const ALLOWED_TYPES = ["video/mp4", "video/quicktime", "video/webm"];

export async function uploadAndAnalyzeVideo(file, activityId, onProgress) {
  if (!file) throw new Error("Please select a video file first.");

  if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(/\.(mp4|mov|webm)$/i)) {
    throw new Error("Invalid format. Please upload an MP4, MOV, or WebM video.");
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    throw new Error("File too large. Maximum supported video size is 100 MB.");
  }

  onProgress({ percent: 5, message: "Uploading video to analysis engine..." });

  const formData = new FormData();
  formData.append("file", file);
  formData.append("activity", activityId);

  const token = getToken();
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}/api/upload/analyze`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (response.status === 415) {
    throw new Error("Unsupported media format. Please upload a standard MP4, MOV, or WebM video.");
  }

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `Upload failed with status ${response.status}`);
  }

  const result = await response.json();
  const jobId = result.job_id;

  if (result.session_id && result.status === "completed") {
    onProgress({ percent: 100, message: "Analysis complete! Loading report..." });
    return result.session_id;
  }

  // Poll job status
  return await pollJobStatus(jobId, onProgress);
}

async function pollJobStatus(jobId, onProgress) {
  const maxAttempts = 120; // 2 minutes max
  let attempts = 0;

  while (attempts < maxAttempts) {
    attempts++;
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      const res = await apiFetch(`/api/upload/status/${jobId}`);
      const data = await res.json();

      const progress = Math.max(10, Math.min(98, data.progress || (attempts * 4)));
      onProgress({
        percent: progress,
        message: data.message || `Processing kinematic frames (${progress}%)...`,
      });

      if (data.status === "completed" && data.session_id) {
        onProgress({ percent: 100, message: "Analysis complete! Generating report..." });
        return data.session_id;
      }

      if (data.status === "failed") {
        throw new Error(data.error || "Video analysis encountered an error.");
      }
    } catch (err) {
      if (err.message.includes("Analysis encountered an error") || err.message.includes("404")) {
        throw err;
      }
    }
  }

  throw new Error("Analysis timed out. Please try again with a shorter clip.");
}
