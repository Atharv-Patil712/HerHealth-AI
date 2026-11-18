/**
 * Thin API service layer. All HTTP requests go through here so endpoints are
 * not scattered across components. Requests are relative; the Vite dev server
 * proxies /api to the local FastAPI backend.
 */
import { API_BASE } from "./config";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body: unknown = undefined
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return await response.text();
    }
  }
  if (contentType.includes("text/html")) {
    throw new ApiError(response.status, "Backend API is not connected.");
  }
  return await response.text();
}

async function request<T>(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (!(init.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const rawUrl = input.toString();
  const url = rawUrl.startsWith("/") ? `${API_BASE}${rawUrl}` : rawUrl;
  const response = await fetch(url, {
    ...init,
    headers,
  });

  const body = await parseBody(response);

  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail && typeof detail === "object" && "message" in detail
          ? String(detail.message)
          : response.statusText || "Request failed";
    throw new ApiError(response.status, message, body);
  }

  return body as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path, { method: "GET" }),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PUT", body: body ? JSON.stringify(body) : undefined }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  postForm: <T>(path: string, formData: FormData) =>
    request<T>(path, { method: "POST", body: formData }),
};