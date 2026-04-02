# ShutterSift Desktop Client (Future)

This directory will contain a Tauri or Electron desktop application.

## Integration

The desktop client communicates with the ShutterSift Python engine via a local
HTTP server defined in `src/shuttersift/server/app.py`.

Start the server: `shuttersift serve --port 7788`
API base URL: `http://localhost:7788`

## Planned Endpoints

- `POST /analyze` — start analysis, returns job ID
- `GET /jobs/{id}/progress` — SSE stream of per-photo results
- `GET /jobs/{id}/results` — final AnalysisResult JSON
- `GET /capabilities` — detected GPU/VLM/RAW capabilities
