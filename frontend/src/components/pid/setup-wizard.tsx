import * as React from "react";
import { Check, CircleAlert, Download, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { SetupStatus } from "@/lib/types";

function Row({ ok, label, detail }: { ok: boolean; label: string; detail?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1.5 text-sm">
      <div className="flex items-center gap-2">
        {ok ? (
          <Check className="size-4 text-emerald-500" />
        ) : (
          <X className="size-4 text-muted-foreground" />
        )}
        <span>{label}</span>
      </div>
      {detail ? <span className="text-xs text-muted-foreground">{detail}</span> : null}
    </div>
  );
}

export function SetupWizard({
  status,
  log,
  busy,
  onRun,
  onDismiss,
}: {
  status: SetupStatus | null;
  log: string[];
  busy: boolean;
  onRun: (includeScaleRae: boolean) => void;
  onDismiss: () => void;
}) {
  const logRef = React.useRef<HTMLDivElement>(null);
  const [scaleRae, setScaleRae] = React.useState(false);
  React.useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [log]);

  const missing = status?.modelsMissing ?? [];
  const canInstall = !!status?.hfTokenPresent && !busy;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Set up PiD Studio</CardTitle>
            {status?.ready ? (
              <Button variant="ghost" size="icon" onClick={onDismiss}>
                <X className="size-4" />
              </Button>
            ) : null}
          </div>
          <CardDescription>
            One token, the rest is automatic. PiD Studio downloads ComfyUI and the
            model weights into the right places for you.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!status ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Connecting to backend…
            </div>
          ) : (
            <>
              {!status.hfTokenPresent && (
                <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
                  <CircleAlert className="mt-0.5 size-4 shrink-0 text-amber-500" />
                  <div>
                    <p className="font-medium">HF_TOKEN is not set.</p>
                    <p className="text-muted-foreground">
                      Add it to <code>.env</code>, accept the license at{" "}
                      <a className="underline" href="https://huggingface.co/nvidia/PiD" target="_blank" rel="noreferrer">
                        huggingface.co/nvidia/PiD
                      </a>{" "}
                      (gated, NSCLv1 non-commercial), then restart.
                    </p>
                  </div>
                </div>
              )}

              <div className="rounded-md border p-3">
                <Row ok={status.hfTokenPresent} label="Hugging Face token" />
                <Row ok={status.comfyInstalled} label="ComfyUI installed" />
                <Row ok={status.comfyRunning} label="ComfyUI running" detail={status.comfyRunning ? undefined : "start ComfyUI to run jobs"} />
                <Row
                  ok={missing.length === 0}
                  label="PiD model weights (flux / flux2)"
                  detail={missing.length ? `${missing.length} missing` : "all present"}
                />
                <Row
                  ok={!!status.scaleRae?.ready}
                  label="Scale-RAE 8× (optional)"
                  detail={status.scaleRae?.ready ? "ready" : "not installed"}
                />
              </div>

              <label className="flex cursor-pointer items-start gap-2 rounded-md border p-3 text-sm">
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={scaleRae}
                  disabled={busy}
                  onChange={(e) => setScaleRae(e.target.checked)}
                />
                <span>
                  <span className="font-medium">Also set up Scale-RAE (8×)</span>
                  <span className="block text-xs text-muted-foreground">
                    Single-pass 8× upscaling. Downloads PyTorch + a gated nvidia/PiD
                    checkpoint into a dedicated venv (several GB, slower install).
                  </span>
                </span>
              </label>

              {missing.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {missing.map((m) => (
                    <Badge key={m} variant="outline" className="font-mono text-[10px]">
                      {m}
                    </Badge>
                  ))}
                </div>
              )}

              {log.length > 0 && (
                <>
                  <Separator />
                  <div
                    ref={logRef}
                    className="h-40 overflow-auto rounded-md bg-muted/50 p-2 font-mono text-[11px] leading-relaxed text-muted-foreground"
                  >
                    {log.map((l, i) => (
                      <div key={i} className="whitespace-pre-wrap">{l}</div>
                    ))}
                  </div>
                </>
              )}

              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-muted-foreground">
                  Weights are ~11&nbsp;GB and download once. NSCLv1 — research/evaluation only.
                </p>
                <Button onClick={() => onRun(scaleRae)} disabled={!canInstall}>
                  {busy ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
                  {busy ? "Installing…" : "Install everything"}
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
