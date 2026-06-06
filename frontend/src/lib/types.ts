/**
 * PiD Studio — shared types.
 *
 * Licensing note: PiD model weights are under NVIDIA NSCLv1 (non-commercial);
 * the application code is Apache-2.0.
 */

export type PidProjectStatus = "idle" | "queued" | "running" | "done" | "error";
export type Backbone = "flux" | "flux2" | "scale_rae";
export type TargetQuality = "2k" | "4k" | "8k";

export interface PidOutputParams {
  backbone: Backbone;
  targetQuality: TargetQuality;
  /** 1 = fast (single-shot) | 4 = quality. */
  inferenceSteps: number;
  degradeSigma: number;
  seed: number;
  prompt: string | null;
  sourceKey?: string;
}

export interface PidOutputEntry {
  key: string;
  kind: "input" | "xt-step" | "output";
  step?: number;
  resolution?: [number, number] | null;
  createdAt: number;
  absolutePath?: string;
  params?: PidOutputParams;
}

export interface PidProjectManifest {
  slug: string;
  title: string;
  status: PidProjectStatus;
  inputKey: string | null;
  outputs: PidOutputEntry[];
  params: {
    targetQuality: TargetQuality;
    inferenceSteps: number;
    degradeSigma: number;
    saveIntermediate: boolean;
  };
  createdAt: number;
  updatedAt: number;
}

export interface PidProjectEntry {
  slug: string;
  title: string;
  updatedAt: number;
  outputCount: number;
}

export interface PidUpscaleRequest {
  slug: string;
  inputKey: string;
  backbone: Backbone;
  targetQuality: TargetQuality;
  inferenceSteps: number;
  degradeSigma: number;
  saveIntermediate: boolean;
  seed: number;
  prompt: string | null;
}

export interface PidHealthResponse {
  status: string;
  models2kReady: boolean;
  models4kReady: boolean;
  fluxVaeReady: boolean;
  flux2Ready: boolean;
  scaleRaeReady: boolean;
  pidRepoReady: boolean;
  comfyReady?: boolean;
  comfyFlux2kReady?: boolean;
  comfyFlux4kReady?: boolean;
  comfyFlux22kReady?: boolean;
  comfyFlux24kReady?: boolean;
  jobsInQueue: number;
  currentJobId: string | null;
}

export interface SetupModelStatus {
  name: string;
  subdir: string;
  present: boolean;
  verified: boolean;
}

export interface SetupStatus {
  hfTokenPresent: boolean;
  comfyInstalled: boolean;
  comfyRunning: boolean;
  pidUpstreamCloned: boolean;
  models: SetupModelStatus[];
  modelsMissing: string[];
  scaleRae?: { checkpoint: boolean; venv: boolean; ready: boolean };
  ready: boolean;
  autoInstall: boolean;
  config: Record<string, unknown>;
}

// ── WebSocket events ─────────────────────────────────────────────────────────
export interface PidXtStepEvent {
  type: "xt-step";
  slug: string;
  jobId?: string;
  step: number;
  totalSteps: number;
  imageBase64: string;
}

export interface PidProgressEvent {
  type: "progress";
  slug: string;
  jobId?: string;
  phase: "queued" | "encoding" | "denoising" | "decoding" | "saving";
  progress01: number;
}

export interface PidDoneEvent {
  type: "done";
  slug: string;
  jobId: string;
  outputKey: string;
  outputPath?: string;
  durationMs: number;
}

export interface PidErrorEvent {
  type: "error";
  slug: string;
  jobId?: string;
  message: string;
}

export interface PidSetupEvent {
  type: "setup";
  step: string;
  status: string; // running | log | ok | warn | skip | error | done
  message: string;
  progress01: number;
}

export type PidSidecarEvent =
  | PidXtStepEvent
  | PidProgressEvent
  | PidDoneEvent
  | PidErrorEvent
  | PidSetupEvent;
