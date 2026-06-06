/**
 * PiD Studio backend client (HTTP + WebSocket).
 *
 * This replaces the original Tauri `invoke` layer (sidecar-client.ts +
 * project-fs.ts). Everything goes over HTTP to the FastAPI backend, proxied by
 * Vite under `/api` in dev (see vite.config.ts).
 */
import type {
  PidHealthResponse,
  PidProjectEntry,
  PidProjectManifest,
  PidSidecarEvent,
  PidUpscaleRequest,
  SetupStatus,
} from "./types";

const API = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${txt ? `: ${txt}` : ""}`);
  }
  return res.json() as Promise<T>;
}

// ── Health / setup ────────────────────────────────────────────────────────
export async function getHealth(): Promise<PidHealthResponse | null> {
  try {
    return await json<PidHealthResponse>(await fetch(`${API}/health`, { cache: "no-store" }));
  } catch {
    return null;
  }
}

export async function getSetupStatus(): Promise<SetupStatus> {
  return json<SetupStatus>(await fetch(`${API}/setup/status`, { cache: "no-store" }));
}

export async function runSetup(steps?: string[]): Promise<{ started: boolean }> {
  return json(
    await fetch(`${API}/setup/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(steps ?? null),
    })
  );
}

// ── Projects ────────────────────────────────────────────────────────────────
export async function listProjects(): Promise<PidProjectEntry[]> {
  return json<PidProjectEntry[]>(await fetch(`${API}/projects`, { cache: "no-store" }));
}

export async function createProject(title: string): Promise<PidProjectManifest> {
  return json<PidProjectManifest>(
    await fetch(`${API}/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    })
  );
}

export async function getProject(slug: string): Promise<PidProjectManifest | null> {
  const res = await fetch(`${API}/projects/${slug}`, { cache: "no-store" });
  if (res.status === 404) return null;
  return json<PidProjectManifest>(res);
}

export async function saveProject(
  slug: string,
  manifest: PidProjectManifest
): Promise<PidProjectManifest> {
  return json<PidProjectManifest>(
    await fetch(`${API}/projects/${slug}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(manifest),
    })
  );
}

export async function deleteProject(slug: string): Promise<void> {
  await fetch(`${API}/projects/${slug}`, { method: "DELETE" });
}

// ── Images ────────────────────────────────────────────────────────────────
export async function uploadImage(
  slug: string,
  key: string,
  file: File
): Promise<{ key: string; path: string; resolution: [number, number] | null }> {
  const fd = new FormData();
  fd.append("key", key);
  fd.append("file", file);
  return json(await fetch(`${API}/projects/${slug}/images`, { method: "POST", body: fd }));
}

/** Stable URL for an image; append a cache-buster when the image is replaced. */
export function imageUrl(slug: string, key: string, bust?: number): string {
  return `${API}/projects/${slug}/images/${key}${bust ? `?t=${bust}` : ""}`;
}

export async function deleteImage(slug: string, key: string): Promise<void> {
  await fetch(`${API}/projects/${slug}/images/${key}`, { method: "DELETE" });
}

// ── Upscale ────────────────────────────────────────────────────────────────
export async function requestUpscale(
  req: PidUpscaleRequest
): Promise<{ jobId: string; queuePosition: number }> {
  return json(
    await fetch(`${API}/upscale`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    })
  );
}

// ── WebSocket preview/progress/setup stream ──────────────────────────────────
export function openPreviewStream(
  onEvent: (event: PidSidecarEvent) => void,
  onStatusChange?: (status: "connecting" | "open" | "closed" | "error") => void
): () => void {
  let socket: WebSocket | null = null;
  let cancelled = false;
  let retryTimer: number | null = null;

  const wsUrl = () => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}${API}/preview`;
  };

  const connect = () => {
    if (cancelled) return;
    onStatusChange?.("connecting");
    socket = new WebSocket(wsUrl());
    socket.onopen = () => onStatusChange?.("open");
    socket.onmessage = (msg) => {
      try {
        onEvent(JSON.parse(msg.data) as PidSidecarEvent);
      } catch (err) {
        console.warn("[pid] WS parse error", err);
      }
    };
    socket.onerror = () => onStatusChange?.("error");
    socket.onclose = () => {
      onStatusChange?.("closed");
      if (!cancelled) retryTimer = window.setTimeout(connect, 2000);
    };
  };

  connect();

  return () => {
    cancelled = true;
    if (retryTimer != null) window.clearTimeout(retryTimer);
    if (socket && socket.readyState <= WebSocket.OPEN) socket.close();
  };
}
