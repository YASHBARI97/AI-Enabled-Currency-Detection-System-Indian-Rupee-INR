export type PredictionStatus =
  | "prediction_available"
  | "low_confidence"
  | "empty_input"
  | "unsupported_format"
  | "unreadable_image"
  | "file_too_large"
  | "inference_error"
  | "model_unavailable"
  | "network_error"
  | "client_validation_error";

export interface TopK {
  label: string;
  confidence: number;
}

export interface PredictionResponse {
  status: PredictionStatus;
  predicted_denomination: string | null;
  confidence: number | null;
  inference_ms: number | null;
  model_version?: string | null;
  message?: string | null;
  top_k?: TopK[] | null;
  disclaimer?: string;
}

export interface HealthResponse {
  status: "ok" | "model_unavailable";
  model_version: string | null;
  classes: string[];
  threshold: number | null;
  max_upload_mb: number;
}

export interface HistoryItem {
  id: number;
  time: string;
  fileName: string;
  result: PredictionResponse;
}
