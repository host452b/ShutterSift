"""
ShutterSift Local HTTP Server — GUI Bridge Layer
================================================
This module is a placeholder for a future FastAPI server that GUI clients
(desktop via Tauri/Electron, or web via Gradio/React) can call.

The GUI does NOT need to know about the Engine internals — it communicates
only via the REST API and SSE stream defined here.

Planned endpoints:
  POST  /analyze          — Start analysis job, returns {"job_id": "..."}
  GET   /jobs/{id}/stream — SSE stream of PhotoResult events (one per photo)
  GET   /jobs/{id}/result — Final AnalysisResult JSON when complete
  GET   /capabilities     — Detected GPU/VLM/RAW capabilities
  GET   /health           — Liveness probe

To start (future):
  shuttersift serve --port 7788

The CLI `on_progress` callback maps cleanly to SSE events — the Engine
interface does not need to change.
"""

# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from shuttersift.engine import Engine
# from shuttersift.config import Config
#
# app = FastAPI(title="ShutterSift", version="0.1.0")
#
# @app.get("/capabilities")
# def get_capabilities():
#     return Engine(Config()).capabilities()
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
