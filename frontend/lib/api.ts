import type {
  FeaturesResponse,
  HistoryDeleteResponse,
  HistoryDetail,
  HistoryList,
  Mode,
  ModelMetricsResponse,
  PredictRequest,
  PredictResponse,
} from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  details: string[];

  constructor(status: number, message: string, details: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

type ValidationItem = { loc?: (string | number)[]; msg?: string };

function describeValidation(items: ValidationItem[]): string[] {
  return items.map((item) => {
    // loc looks like ["body", "A", "inputs", "age"]; keep what follows "inputs".
    const loc = item.loc ?? [];
    const at = loc.indexOf("inputs");
    const field = at >= 0 ? loc.slice(at + 1).join(".") : loc.filter((p) => p !== "body").join(".");
    return field ? `${field}: ${item.msg ?? "invalid value"}` : item.msg ?? "invalid value";
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "Cannot reach the assessment service. Check that the backend is running.");
  }
  if (response.ok) {
    return (await response.json()) as T;
  }
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    /* non-JSON error body */
  }
  const obj = (body ?? {}) as Record<string, unknown>;
  if (response.status === 422 && Array.isArray(obj.detail)) {
    throw new ApiError(422, "Some values were not accepted.", describeValidation(obj.detail as ValidationItem[]));
  }
  if (response.status === 503) {
    throw new ApiError(
      503,
      "The assessment models are currently unavailable.",
      Array.isArray(obj.details) ? (obj.details as string[]) : [],
    );
  }
  if (typeof obj.detail === "string") {
    throw new ApiError(response.status, obj.detail);
  }
  const requestId = typeof obj.request_id === "string" ? ` (reference ${obj.request_id})` : "";
  throw new ApiError(response.status, `The service could not complete the request${requestId}.`);
}

export function getFeatures(mode: Mode): Promise<FeaturesResponse> {
  return request<FeaturesResponse>(`/v2/features/${mode}`);
}

export function predict(body: PredictRequest): Promise<PredictResponse> {
  return request<PredictResponse>("/v2/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getHistory(limit = 10, beforeId?: number | null): Promise<HistoryList> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (beforeId != null) q.set("before_id", String(beforeId));
  return request<HistoryList>(`/v2/history?${q.toString()}`);
}

export function getHistoryDetail(submissionId: string): Promise<HistoryDetail> {
  return request<HistoryDetail>(`/v2/history/${encodeURIComponent(submissionId)}`);
}

export function deleteHistoryItem(submissionId: string): Promise<HistoryDeleteResponse> {
  return request<HistoryDeleteResponse>(`/v2/history/${encodeURIComponent(submissionId)}`, { method: "DELETE" });
}

export function clearHistory(): Promise<HistoryDeleteResponse> {
  return request<HistoryDeleteResponse>("/v2/history?confirm=true", { method: "DELETE" });
}

export function getModelMetrics(): Promise<ModelMetricsResponse> {
  return request<ModelMetricsResponse>("/v2/model-metrics");
}