"""
The web API and app page (2026-10-14, build step 5): a CPU Cloud Run service.

Private until step 6 adds login, consent and spending limits: Cloud Run only
lets through requests carrying an identity token, so Dan opens it with
`gcloud run services proxy lomb-api --region us-east4`, which adds his.

  GET  /                            the app page (api/static/index.html)
  POST /api/uploads                 a Cloud Storage upload URL for the browser
  POST /api/sessions                register the uploaded audio, queue transcription
  GET  /api/sessions/{id}           status (+ the speaker choices at the picker)
  POST /api/sessions/{id}/speaker   the user's pick, queue the analysis
  GET  /api/sessions/{id}/report    Report 1 (HTML)
  GET  /api/sessions/{id}/report2   Report 2 (.xlsx)

The audio goes from the browser straight to Cloud Storage (a 30-minute
lesson is too big for a Cloud Run request). GPU work goes to the worker
through a Cloud Tasks queue, which retries if an attempt fails.

Settings: service/cloud.py's, plus
  LOMB_WORKER_URL   the worker's https URL
  LOMB_TASKS_QUEUE  projects/<p>/locations/<r>/queues/<q>
  LOMB_TASKS_SA     service account the queue calls the worker as
  LOMB_MAX_UPLOAD_MB                                   default 300
"""

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from reporting.report import render_html  # noqa: E402
from service.cloud import cloud_deps  # noqa: E402
from service.lesson_job import register_upload, speaker_choices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("lomb.api")

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".webm", ".ogg", ".flac", ".aac", ".mp4"}
MAX_UPLOAD_BYTES = int(os.environ.get("LOMB_MAX_UPLOAD_MB", "300")) * 1024 * 1024
STATIC = Path(__file__).parent / "static"
state = {}


@asynccontextmanager
async def lifespan(_app):
    state["deps"] = cloud_deps(with_transcriber=False, with_llm=False)
    from google.cloud import tasks_v2
    state["tasks"] = tasks_v2.CloudTasksClient()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


def enqueue(path: str, body: dict) -> None:
    """Hand a job to the GPU worker through the queue (retried on failure)."""
    worker = os.environ["LOMB_WORKER_URL"].rstrip("/")
    state["tasks"].create_task(parent=os.environ["LOMB_TASKS_QUEUE"], task={
        "http_request": {
            "http_method": "POST",
            "url": worker + path,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body).encode(),
            "oidc_token": {"service_account_email": os.environ["LOMB_TASKS_SA"], "audience": worker},
        },
        "dispatch_deadline": {"seconds": 1800},  # Cloud Tasks' maximum; a 60-min lesson needs ~10
    })


def _session_or_404(session_id: str) -> dict:
    session = state["deps"].repo.get_session(session_id)
    if not session:
        raise HTTPException(404, "no such lesson")
    return session


class UploadRequest(BaseModel):
    filename: str
    size_bytes: int
    content_type: str | None = None


class SessionRequest(BaseModel):
    storage_uri: str
    num_speakers: int | None = None


class SpeakerRequest(BaseModel):
    speaker_label: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/uploads")
def create_upload(req: UploadRequest, request: Request):
    suffix = Path(req.filename).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(400, f"unsupported file type {suffix or '(none)'}; use one of "
                                 f"{', '.join(sorted(AUDIO_SUFFIXES))}")
    if not 0 < req.size_bytes <= MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file must be under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    key = f"uploads/{uuid.uuid4()}{suffix}"
    blobs = state["deps"].blobs
    origin = request.headers.get("origin") or str(request.base_url).rstrip("/")
    url = blobs.browser_upload_url(key, content_type=req.content_type or "application/octet-stream",
                                   size=req.size_bytes, origin=origin)
    return {"upload_url": url, "storage_uri": f"gs://{blobs.bucket_name}/{key}"}


@app.post("/api/sessions")
def create_session(req: SessionRequest):
    deps = state["deps"]
    if not req.storage_uri.startswith(f"gs://{deps.blobs.bucket_name}/uploads/"):
        raise HTTPException(400, "storage_uri must come from /api/uploads")
    if req.num_speakers is not None and not 1 <= req.num_speakers <= 6:
        raise HTTPException(400, "num_speakers must be 1-6")
    if not deps.blobs.exists(req.storage_uri):
        raise HTTPException(400, "upload not finished")
    try:
        session_id = register_upload(deps, storage_uri=req.storage_uri, user_id=None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    enqueue("/process", {"session_id": session_id, "num_speakers": req.num_speakers})
    log.info("queued %s", session_id)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    deps = state["deps"]
    session = _session_or_404(session_id)
    status, transcript = session["status"], deps.repo.latest_transcript(session_id)
    stage = status
    if status == "processing":
        stage = "analyzing" if transcript else "transcribing"
    audio = deps.repo.get_audio_asset(session_id)
    out = {"session_id": session_id, "status": status, "stage": stage, "created_at": session["created_at"],
           "duration_seconds": audio["duration_seconds"] if audio else None}
    if status == "awaiting_speaker_selection":
        out["speakers"] = speaker_choices(deps, session_id)
    return out


@app.post("/api/sessions/{session_id}/speaker")
def choose_speaker(session_id: str, req: SpeakerRequest):
    deps = state["deps"]
    if _session_or_404(session_id)["status"] != "awaiting_speaker_selection":
        raise HTTPException(409, "this lesson isn't waiting for a speaker choice")
    if req.speaker_label not in {c["speaker_label"] for c in speaker_choices(deps, session_id)}:
        raise HTTPException(400, "unknown speaker")
    deps.repo.update_session(session_id, status="processing")
    enqueue("/resume", {"session_id": session_id, "speaker_label": req.speaker_label})
    return {"session_id": session_id, "status": "processing"}


@app.get("/api/sessions/{session_id}/report", response_class=HTMLResponse)
def report1(session_id: str):
    _session_or_404(session_id)
    snapshot = state["deps"].repo.latest_report_snapshot(session_id)
    if not snapshot:
        raise HTTPException(404, "no report yet")
    return render_html(json.loads(snapshot["payload_json"]))


@app.get("/api/sessions/{session_id}/report2")
def report2(session_id: str):
    _session_or_404(session_id)
    export = state["deps"].repo.latest_report2_export(session_id)
    if not export:
        raise HTTPException(404, "no report yet")
    return Response(state["deps"].blobs.read_bytes(export["storage_uri"]),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="lomb_report2_{session_id[5:13]}.xlsx"'})


@app.get("/health")  # not /healthz: Cloud Run reserves paths ending in "z"
def health():
    return {"ok": "deps" in state}
