# ShutterSift Server (Future)

FastAPI-based local HTTP server that bridges the Engine to GUI clients.

## Why a server layer?

The Engine is a Python library. GUI clients (Tauri desktop app, web browser)
cannot call Python directly — they need an HTTP interface.

## Architecture

```
GUI client
  │  HTTP / SSE
  ▼
FastAPI server  (this module, port 7788)
  │  Python function call
  ▼
Engine.analyze(on_progress=sse_emit)
```

The `on_progress` callback in `Engine.analyze()` was designed with this in
mind: it sends per-photo events as they complete, which maps directly to
SSE (Server-Sent Events) for real-time progress in any GUI.

## Start (once implemented)

```bash
shuttersift serve --port 7788
```
