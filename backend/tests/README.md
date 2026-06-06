# Tests

These run against a **live backend** — the one you already run locally, or the
ported `server.py` (`python scripts/run.py`). They mirror the original sidecar
e2e tests and the frontend's real call flow.

```bash
# from the repo root, with the backend running on :17820
.venv/bin/python -m pip install -r backend/tests/requirements-dev.txt
.venv/bin/python -m pytest backend          # (use .venv\Scripts\python.exe on Windows)

# point at a different backend
PID_BASE=http://127.0.0.1:17820 pytest backend

# pick the backbone/quality for the heavy e2e
PID_E2E_BACKBONE=flux2 PID_E2E_QUALITY=4k pytest backend/tests/test_http_e2e.py
```

| File | Needs | Skips when |
|---|---|---|
| `test_api.py` | backend only | backend down |
| `test_http_e2e.py` | backend + ComfyUI + weights | model for backbone/quality not ready |
| `test_comfy_e2e.py` | ComfyUI reachable | ComfyUI offline / workflow rejected |

Everything auto-skips gracefully, so the suite is green on a machine without a
GPU as long as the backend process is up.
