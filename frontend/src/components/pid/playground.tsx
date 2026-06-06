import * as React from "react";
import {
  Dice5,
  ImageUp,
  Loader2,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ImageCompareSlider } from "./image-compare-slider";
import { SetupWizard } from "./setup-wizard";
import * as api from "@/lib/api";
import type {
  Backbone,
  PidHealthResponse,
  PidProjectEntry,
  PidProjectManifest,
  PidSidecarEvent,
  SetupStatus,
  TargetQuality,
} from "@/lib/types";

export function Playground() {
  // ── connection / setup ──
  const [health, setHealth] = React.useState<PidHealthResponse | null>(null);
  const [setupStatus, setSetupStatus] = React.useState<SetupStatus | null>(null);
  const [showSetup, setShowSetup] = React.useState(false);
  const [setupLog, setSetupLog] = React.useState<string[]>([]);
  const [setupBusy, setSetupBusy] = React.useState(false);

  // ── projects ──
  const [projects, setProjects] = React.useState<PidProjectEntry[]>([]);
  const [slug, setSlug] = React.useState<string | null>(null);
  const [manifest, setManifest] = React.useState<PidProjectManifest | null>(null);
  const [inputBust, setInputBust] = React.useState(0);

  // ── parameters ──
  const [backbone, setBackbone] = React.useState<Backbone>("flux");
  const [quality, setQuality] = React.useState<TargetQuality>("4k");
  const [mode, setMode] = React.useState<"fast" | "quality">("quality");
  const [degradeSigma, setDegradeSigma] = React.useState(0);
  const [seed, setSeed] = React.useState(5);
  const [prompt, setPrompt] = React.useState("");

  // ── job ──
  const [running, setRunning] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [phase, setPhase] = React.useState<string>("");
  const [preview, setPreview] = React.useState<string | null>(null);
  const [outputKey, setOutputKey] = React.useState<string | null>(null);
  const [outputBust, setOutputBust] = React.useState(0);
  const [error, setError] = React.useState<string | null>(null);

  const refreshStatus = React.useCallback(async () => {
    const [h, s] = await Promise.all([api.getHealth(), api.getSetupStatus().catch(() => null)]);
    setHealth(h);
    setSetupStatus(s);
    if (s && !s.ready) setShowSetup(true);
  }, []);

  const reloadProjects = React.useCallback(async () => {
    const list = await api.listProjects().catch(() => []);
    setProjects(list);
    return list;
  }, []);

  const selectProject = React.useCallback(async (s: string) => {
    setSlug(s);
    const m = await api.getProject(s);
    setManifest(m);
    setOutputKey(m?.outputs?.length ? m.outputs[m.outputs.length - 1].key : null);
    setInputBust(Date.now());
    setOutputBust(Date.now());
    setPreview(null);
    setError(null);
  }, []);

  // initial load
  React.useEffect(() => {
    void refreshStatus();
    void reloadProjects().then((list) => {
      if (list.length) void selectProject(list[0].slug);
    });
  }, [refreshStatus, reloadProjects, selectProject]);

  // WebSocket event routing
  React.useEffect(() => {
    const close = api.openPreviewStream((ev: PidSidecarEvent) => {
      switch (ev.type) {
        case "setup":
          setSetupLog((l) => [...l, `[${ev.step}] ${ev.status}${ev.message ? ` — ${ev.message}` : ""}`]);
          if (ev.step === "all" && (ev.status === "done" || ev.status === "error")) {
            setSetupBusy(false);
            void refreshStatus();
          }
          break;
        case "progress":
          setProgress(Math.round(ev.progress01 * 100));
          setPhase(ev.phase);
          break;
        case "xt-step":
          setPreview(`data:image/png;base64,${ev.imageBase64}`);
          break;
        case "done":
          setRunning(false);
          setProgress(100);
          setPhase("done");
          setPreview(null);
          if (ev.slug === slug) {
            setOutputKey(ev.outputKey);
            setOutputBust(Date.now());
            void api.getProject(ev.slug).then((m) => m && setManifest(m));
            void reloadProjects();
          }
          break;
        case "error":
          setRunning(false);
          setError(ev.message);
          break;
      }
    });
    return close;
  }, [slug, refreshStatus, reloadProjects]);

  // periodic health refresh (ComfyUI may start after the app)
  React.useEffect(() => {
    const t = window.setInterval(() => void api.getHealth().then(setHealth), 5000);
    return () => window.clearInterval(t);
  }, []);

  // ── handlers ──
  async function handleNewProject() {
    const m = await api.createProject("Untitled");
    await reloadProjects();
    await selectProject(m.slug);
  }

  async function handleUpload(file: File) {
    let s = slug;
    if (!s) {
      const m = await api.createProject(file.name.replace(/\.[^.]+$/, "") || "Untitled");
      s = m.slug;
      await reloadProjects();
      await selectProject(s);
    }
    await api.uploadImage(s, "input", file);
    // persist inputKey on the manifest
    const m = await api.getProject(s);
    if (m) {
      m.inputKey = "input";
      await api.saveProject(s, m);
      setManifest(m);
    }
    setOutputKey(null);
    setInputBust(Date.now());
  }

  async function handleRun() {
    if (!slug || !manifest?.inputKey) return;
    setError(null);
    setRunning(true);
    setProgress(0);
    setPhase("queued");
    setPreview(null);
    try {
      await api.requestUpscale({
        slug,
        inputKey: manifest.inputKey,
        backbone,
        targetQuality: quality,
        inferenceSteps: mode === "fast" ? 1 : 4,
        degradeSigma,
        saveIntermediate: true,
        seed,
        prompt: prompt.trim() || null,
      });
    } catch (e) {
      setRunning(false);
      setError((e as Error).message);
    }
  }

  async function handleDeleteProject() {
    if (!slug) return;
    await api.deleteProject(slug);
    const list = await reloadProjects();
    setManifest(null);
    setSlug(null);
    setOutputKey(null);
    if (list.length) await selectProject(list[0].slug);
  }

  function runSetup(includeScaleRae: boolean) {
    setSetupBusy(true);
    setSetupLog([]);
    const steps = ["comfyui", "models", "pid-upstream", ...(includeScaleRae ? ["scale-rae"] : [])];
    void api.runSetup(steps);
  }

  const hasInput = !!manifest?.inputKey;
  const inputSrc = slug && hasInput ? api.imageUrl(slug, manifest!.inputKey!, inputBust) : null;
  const outputSrc = slug && outputKey ? api.imageUrl(slug, outputKey, outputBust) : null;
  const comfyOk = !!health?.comfyReady;
  const modelOk =
    backbone === "scale_rae"
      ? !!health?.scaleRaeReady
      : quality === "2k"
      ? backbone === "flux2"
        ? health?.comfyFlux22kReady
        : health?.comfyFlux2kReady
      : backbone === "flux2"
      ? health?.comfyFlux24kReady
      : health?.comfyFlux4kReady;
  const canRun = !!slug && hasInput && !running && (backbone === "scale_rae" || (comfyOk && modelOk));

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      {/* top bar */}
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-semibold">
            <Sparkles className="size-5 text-primary" /> PiD Studio
          </div>
          <Separator orientation="vertical" className="h-6" />
          <Select value={slug ?? ""} onValueChange={(v) => void selectProject(v)}>
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="No project — create one" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((p) => (
                <SelectItem key={p.slug} value={p.slug}>
                  {p.title}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={() => void handleNewProject()} title="New project">
            <Plus className="size-4" />
          </Button>
          {slug && (
            <Button variant="ghost" size="icon" onClick={() => void handleDeleteProject()} title="Delete project">
              <Trash2 className="size-4" />
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={comfyOk ? "success" : "warning"}>
            ComfyUI {comfyOk ? "online" : "offline"}
          </Badge>
          <Button variant="outline" size="sm" onClick={() => setShowSetup(true)}>
            Setup
          </Button>
          <Button onClick={() => void handleRun()} disabled={!canRun}>
            {running ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
            {running ? "Upscaling…" : "Run upscale"}
          </Button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* center canvas */}
        <main className="flex min-w-0 flex-1 flex-col gap-3 p-4">
          <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-lg border bg-muted/20">
            {preview ? (
              <img src={preview} alt="live preview" className="max-h-full max-w-full object-contain opacity-90" />
            ) : outputSrc && inputSrc ? (
              <ImageCompareSlider before={inputSrc} after={outputSrc} className="h-full w-full" />
            ) : inputSrc ? (
              <img src={inputSrc} alt="input" className="max-h-full max-w-full object-contain" />
            ) : (
              <Dropzone onFile={(f) => void handleUpload(f)} />
            )}
          </div>

          {(running || progress > 0) && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="capitalize">{phase || "idle"}</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          )}

          {error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive">
              {error}
            </div>
          )}

          {inputSrc && (
            <div className="flex items-center gap-2">
              <label className="cursor-pointer">
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && void handleUpload(e.target.files[0])}
                />
                <span className="inline-flex h-8 items-center gap-1.5 rounded-md border px-3 text-xs hover:bg-accent">
                  <ImageUp className="size-3.5" /> Replace input
                </span>
              </label>
              {manifest?.outputs?.length ? (
                <span className="text-xs text-muted-foreground">
                  {manifest.outputs.length} output{manifest.outputs.length > 1 ? "s" : ""}
                </span>
              ) : null}
            </div>
          )}
        </main>

        {/* right settings panel */}
        <aside className="flex w-80 shrink-0 flex-col gap-5 overflow-y-auto border-l p-4">
          <Field label="Backbone" hint="flux/flux2 via ComfyUI · scale_rae 8× via CLI">
            <Select value={backbone} onValueChange={(v) => setBackbone(v as Backbone)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="flux">Flux (4×)</SelectItem>
                <SelectItem value="flux2">Flux.2 (4×)</SelectItem>
                <SelectItem value="scale_rae">Scale-RAE (8×, single-pass)</SelectItem>
              </SelectContent>
            </Select>
          </Field>

          <Field label="Target quality">
            <Select
              value={quality}
              onValueChange={(v) => setQuality(v as TargetQuality)}
              disabled={backbone === "scale_rae"}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="2k">2K (2048 px)</SelectItem>
                <SelectItem value="4k">4K (4096 px)</SelectItem>
                <SelectItem value="8k">8K (4K + Lanczos 2×)</SelectItem>
              </SelectContent>
            </Select>
          </Field>

          <Field label="Mode" hint="Distilled steps: fast = 1, quality = 4">
            <Tabs value={mode} onValueChange={(v) => setMode(v as "fast" | "quality")}>
              <TabsList className="w-full">
                <TabsTrigger value="fast" className="flex-1">Fast</TabsTrigger>
                <TabsTrigger value="quality" className="flex-1">Quality</TabsTrigger>
              </TabsList>
            </Tabs>
          </Field>

          <Field label={`Detail synthesis (degrade σ) — ${degradeSigma.toFixed(2)}`} hint="Higher = more invented detail">
            <Slider
              value={[degradeSigma]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={([v]) => setDegradeSigma(v)}
            />
          </Field>

          <Field label="Seed">
            <div className="flex gap-2">
              <Input
                type="number"
                value={seed}
                min={0}
                onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
              />
              <Button variant="outline" size="icon" onClick={() => setSeed(Math.floor(Math.random() * 1e6))} title="Randomize">
                <Dice5 className="size-4" />
              </Button>
            </div>
          </Field>

          <Field label="Prompt (optional)" hint="Texture/style hint; SR is mostly image-driven">
            <Textarea
              value={prompt}
              placeholder="a high quality photo, sharp focus, detailed"
              rows={3}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </Field>

          <Separator />
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            PiD weights are licensed NSCLv1 — research / evaluation only. Code is Apache-2.0.
          </p>
        </aside>
      </div>

      {showSetup && (
        <SetupWizard
          status={setupStatus}
          log={setupLog}
          busy={setupBusy}
          onRun={runSetup}
          onDismiss={() => setShowSetup(false)}
        />
      )}
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      {children}
      {hint ? <p className="text-[11px] text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

function Dropzone({ onFile }: { onFile: (f: File) => void }) {
  const [over, setOver] = React.useState(false);
  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      className={`flex h-full w-full cursor-pointer flex-col items-center justify-center gap-2 text-muted-foreground transition-colors ${
        over ? "bg-accent/40" : ""
      }`}
    >
      <input
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      <ImageUp className="size-10" />
      <p className="text-sm">Drop an image here, or click to choose</p>
      <p className="text-xs">PNG · JPG · WebP</p>
    </label>
  );
}
