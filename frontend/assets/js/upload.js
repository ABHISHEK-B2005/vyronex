/**
 * Vyronex Video Upload & Background Analysis Handler
 */

import { apiFetch, API_BASE_URL, getToken } from "./auth.js";

const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024;
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

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/upload/analyze`, {
      method: "POST",
      headers,
      body: formData,
    });
  } catch {
    throw new Error("Unable to connect to Vyronex backend server (http://localhost:8000). Please ensure the backend is running.");
  }

  if (response.status === 415) {
    throw new Error("Unsupported media format. Please upload a standard MP4, MOV, or WebM video.");
  }

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `Upload failed with status ${response.status}`);
  }

  const result = await response.json();
  const jobId = result.job_id;
  if (!jobId) throw new Error("The backend accepted the upload but did not return an analysis job ID.");

  if (result.session_id && result.status === "completed") {
    onProgress({ percent: 100, message: "Analysis complete! Loading report..." });
    return result.session_id;
  }

  // Poll job status
  return await pollJobStatus(jobId, onProgress);
}

async function pollJobStatus(jobId, onProgress) {
  const maxAttempts = 600;
  let attempts = 0;
  let consecutiveErrors = 0;

  while (attempts < maxAttempts) {
    attempts++;
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      const res = await apiFetch(`/api/upload/status/${jobId}`);
      const data = await res.json();
      consecutiveErrors = 0;

      const progress = Math.max(5, Math.min(99, Number(data.progress ?? 0)));
      onProgress({
        percent: progress,
        message: data.message || `Analyzing video frames (${Math.round(progress)}%)...`,
      });

      if (data.status === "completed" && data.session_id) {
        onProgress({ percent: 100, message: "Analysis complete! Generating report..." });
        return data.session_id;
      }

      if (data.status === "failed") {
        throw new Error(data.error || "Video analysis encountered an error.");
      }
      if (data.status === "completed" && !data.session_id) {
        throw new Error("Analysis completed, but the backend did not return a report session.");
      }
    } catch (err) {
      if (err.status || !err.message.includes("Unable to connect")) throw err;
      consecutiveErrors += 1;
      if (consecutiveErrors >= 3) {
        throw new Error("The connection to the backend was lost. Please check that it is running and retry.");
      }
    }
  }

  throw new Error("Video analysis is taking longer than expected. Check the backend and try a shorter clip.");
}
