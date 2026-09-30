import type { Scenario, SystemInfo } from "../types/events";

const configured = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";
export const apiBase = configured.replace(/\/$/, "");
export const wsUrl = (sessionId: string) => `${apiBase.replace(/^http/, "ws")}/ws/session/${sessionId}`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiBase + path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* body was not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export interface DocumentOut {
  document_id: string;
  title: string;
  category: string;
  subcategory: string;
  version: string;
  section: string;
  chunks: { chunk_id: string; section: string; text: string }[];
}

export const api = {
  health: () => request<{ status: string; system: SystemInfo }>("/health"),
  createSession: () => request<{ session_id: string }>("/api/session", { method: "POST" }),
  scenarios: () => request<{ scenarios: Scenario[] }>("/api/scenarios"),
  document: (id: string) => request<DocumentOut>(`/api/documents/${encodeURIComponent(id)}`),
  benchmarkResults: () => request<any>("/api/benchmark/results"),
  runBenchmark: () => request<any>("/api/benchmark/run", { method: "POST" }),
};
