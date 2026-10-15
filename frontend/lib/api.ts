import type { HealthResponse, PredictionResponse } from "./types";

// Same-origin path; proxied to FastAPI by next.config.mjs.
const BASE = "/api";

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse | null> {
  try {
    const r = await fetch(`${BASE}/health`, { cache: "no-store", signal });
    if (!r.ok) return null;
    return (await r.json()) as HealthResponse;
  } catch {
    return null;
  }
}

export async function predictImage(file: File, signal?: AbortSignal): Promise<PredictionResponse> {
  const body = new FormData();
  body.append("file", file);
  try {
    const r = await fetch(`${BASE}/predict`, { method: "POST", body, signal });
    const data = (await r.json().catch(() => null)) as PredictionResponse | null;
    if (data && typeof data.status === "string") return data;
    return {
      status: "inference_error",
      predicted_denomination: null,
      confidence: null,
      inference_ms: null,
      message: "Unexpected response from the server.",
    };
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    return {
      status: "network_error",
      predicted_denomination: null,
      confidence: null,
      inference_ms: null,
      message: "Could not reach the prediction service. Is the backend running?",
    };
  }
}
