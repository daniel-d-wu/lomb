"""
The web API and app page (2026-10-14 build step 5; accounts, consent and
limits added 2026-10-19, step 6): a CPU Cloud Run service.

Public URL, but every /api call except /api/config needs a Firebase sign-in
token (Google sign-in) for an invited email, and lesson endpoints also need
the current terms + recording-permission consent. Lessons are only visible
to the user who uploaded them.

  GET    /                              the app page (api/static/index.html)
  GET    /privacy                       privacy policy + voiceprint retention schedule
  GET    /api/config                    Firebase web config, policy version
  GET    /api/me                        account, consents, voiceprint, limits
  DELETE /api/me                        delete the account and everything in it
  POST   /api/consents                  agree to terms / recording permission / biometric
  DELETE /api/consents/biometric_enrollment   withdraw: the voiceprint is deleted
  POST   /api/uploads                   a Cloud Storage upload URL for the browser
  POST   /api/sessions                  register the uploaded audio, queue transcription
  GET    /api/sessions                  the user's lessons
  GET    /api/sessions/{id}             status (+ the speaker choices at the picker)
  POST   /api/sessions/{id}/speaker     the user's pick, queue the analysis
  GET    /api/sessions/{id}/report      Report 1 (HTML)
  GET    /api/sessions/{id}/report2     Report 2 (.xlsx)
  DELETE /api/sessions/{id}             delete the lesson: rows, audio, transcript, reports
  POST   /internal/cleanup              daily (Cloud Scheduler): destroy voiceprints unused a year

The audio goes from the browser straight to Cloud Storage (a long lesson is
too big for a Cloud Run request). GPU work goes to the worker through a
Cloud Tasks queue, which retries if an attempt fails.

Settings: service/cloud.py's, plus
  LOMB_WORKER_URL       the worker's https URL
  LOMB_TASKS_QUEUE      projects/<p>/locations/<r>/queues/<q>
  LOMB_TASKS_SA         service account the queue and scheduler call as
  LOMB_FIREBASE_CONFIG  the Firebase web app config (JSON; public by design)
  LOMB_FIREBASE_PROJECT                                    default lomb-prod
  LOMB_API_URL          this service's URL (audience of the scheduler's token)
  LOMB_CONTACT_EMAIL    shown on the privacy page
  LOMB_DAILY_LESSONS    per user, rolling 24 h             default 3
  LOMB_MAX_MINUTES      per lesson                         default 60
  LOMB_MONTHLY_CAP_USD  uploads pause above this estimate  default 50
  LOMB_COST_PER_LESSON / LOMB_COST_PER_AUDIO_MIN   the estimate (2026-10-19:
                        0.25 GPU idle + cold start, 0.0045 per audio minute)
  LOMB_MAX_UPLOAD_MB                                       default 300
"""

import json
import logging
import os
import sys
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from reporting.report import render_html  # noqa: E402
from service.lesson_job import (  # noqa: E402
    BIOMETRIC_CONSENT, delete_account, delete_lesson, register_upload, speaker_choices,
)
from voice_enrollment.sql_voiceprints import SQLVoiceprintRepository  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("lomb.api")

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".webm", ".ogg", ".flac", ".aac", ".mp4"}
MAX_UPLOAD_BYTES = int(os.environ.get("LOMB_MAX_UPLOAD_MB", "300")) * 1024 * 1024
DAILY_LESSONS = int(os.environ.get("LOMB_DAILY_LESSONS", "3"))
MAX_MINUTES = float(os.environ.get("LOMB_MAX_MINUTES", "60"))
MONTHLY_CAP_USD = float(os.environ.get("LOMB_MONTHLY_CAP_USD", "50"))
COST_PER_LESSON = float(os.environ.get("LOMB_COST_PER_LESSON", "0.25"))
COST_PER_AUDIO_MIN = float(os.environ.get("LOMB_COST_PER_AUDIO_MIN", "0.0045"))
FIREBASE_PROJECT = os.environ.get("LOMB_FIREBASE_PROJECT", "lomb-prod")
VOICEPRINT_UNUSED_DAYS = 365

# Bump when the text in api/static changes; users are asked to agree again.
POLICY_VERSION = "2026-10-19"
REQUIRED_CONSENTS = ("terms", "recording_permission")
CONSENT_TYPES = REQUIRED_CONSENTS + (BIOMETRIC_CONSENT,)

STATIC = Path(__file__).parent / "static"
state = {}


@asynccontextmanager
async def lifespan(_app):
    if "deps" not in state:  # tests pre-fill state with local stand-ins
        from google.cloud import tasks_v2

        from service.cloud import cloud_deps
        state["deps"] = cloud_deps(with_transcriber=False, with_llm=False)
        state["tasks"] = tasks_v2.CloudTasksClient()
        state["verify"] = verify_firebase_token
        state["enqueue"] = enqueue
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


# --- sign-in ---------------------------------------------------------------------

_certs = {"at": 0.0, "certs": None}


def verify_firebase_token(token: str) -> dict:
    """A Firebase ID token's claims, after checking Google's signature, the
    project, expiry and a verified email."""
    from google.auth import jwt
    if time.time() - _certs["at"] > 3600:
        url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        with urllib.request.urlopen(url, timeout=10) as r:
            _certs["certs"], _certs["at"] = json.load(r), time.time()
    claims = jwt.decode(token, certs=_certs["certs"], audience=FIREBASE_PROJECT)
    if claims.get("iss") != f"https://securetoken.google.com/{FIREBASE_PROJECT}":
        raise ValueError("wrong issuer")
    if not claims.get("email") or not claims.get("email_verified"):
        raise ValueError("no verified email")
    return claims


def current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "sign in first")
    try:
        claims = state["verify"](authorization[len("Bearer "):])
    except Exception:
        raise HTTPException(401, "your sign-in expired; sign in again")
    repo = state["deps"].repo
    email = claims["email"].lower()
    if not repo.is_invited(email):
        raise HTTPException(403, {"code": "not_invited", "email": email})
    user = repo.get_user(claims["sub"])
    last_seen = user and user.get("last_seen_at")
    if user is None or not last_seen or last_seen < (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat():
        user = repo.upsert_user(claims["sub"], email, claims.get("name"))
    user["consents"] = repo.active_consents(user["user_id"])
    return user


def consented_user(user: dict = Depends(current_user)) -> dict:
    for kind in REQUIRED_CONSENTS:
        c = user["consents"].get(kind)
        if not c or c["policy_version"] != POLICY_VERSION:
            raise HTTPException(403, {"code": "consent_required"})
    return user


def _owned_session(session_id: str, user: dict) -> dict:
    session = state["deps"].repo.get_session(session_id)
    if not session or session["user_id"] != user["user_id"]:
        raise HTTPException(404, "no such lesson")  # someone else's lesson looks the same as none
    return session


# --- limits ----------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def usage(user: dict) -> dict:
    repo = state["deps"].repo
    lessons_today = repo.count_user_sessions_since(user["user_id"], (_now() - timedelta(hours=24)).isoformat())
    month_start = _now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    lessons, minutes = repo.lesson_minutes_since(month_start)
    spend = lessons * COST_PER_LESSON + minutes * COST_PER_AUDIO_MIN
    return {"lessons_last_24h": lessons_today, "daily_lessons": DAILY_LESSONS, "max_minutes": MAX_MINUTES,
            "paused": spend >= MONTHLY_CAP_USD}


def check_limits(user: dict) -> None:
    u = usage(user)
    if u["paused"]:
        raise HTTPException(503, "Lomb has reached its budget for this month; uploads reopen on the 1st")
    if u["lessons_last_24h"] >= DAILY_LESSONS:
        raise HTTPException(429, f"you can analyse {DAILY_LESSONS} lessons per 24 hours; try again later")


# --- the queue -------------------------------------------------------------------

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


# --- requests --------------------------------------------------------------------

class ConsentRequest(BaseModel):
    consent_type: str
    signature: str | None = None    # typed full name, required for biometric consent
    session_id: str | None = None   # biometric: enroll the voice picked in this lesson


class UploadRequest(BaseModel):
    filename: str
    size_bytes: int
    content_type: str | None = None


class SessionRequest(BaseModel):
    storage_uri: str
    num_speakers: int | None = None


class SpeakerRequest(BaseModel):
    speaker_label: str


# --- pages -----------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    page = (STATIC / "privacy.html").read_text(encoding="utf-8")
    contact = os.environ.get("LOMB_CONTACT_EMAIL", "the person who invited you")
    return page.replace("{{POLICY_VERSION}}", POLICY_VERSION).replace("{{CONTACT}}", contact)


@app.get("/api/config")
def config():
    return {"firebase": json.loads(os.environ.get("LOMB_FIREBASE_CONFIG", "{}")), "policy_version": POLICY_VERSION}


# --- account and consent -----------------------------------------------------------

@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    current = {k: bool(c) and c["policy_version"] == POLICY_VERSION for k, c in
               ((k, user["consents"].get(k)) for k in CONSENT_TYPES)}
    return {"email": user["email"], "name": user["display_name"], "consents": current,
            "has_voiceprint": SQLVoiceprintRepository(state["deps"].repo).exists(user["user_id"]),
            "usage": usage(user)}


@app.delete("/api/me")
def delete_me(user: dict = Depends(current_user)):
    delete_account(state["deps"], user["user_id"])
    log.info("account deleted: %s", user["user_id"])
    return {"deleted": True}


@app.post("/api/consents")
def give_consent(req: ConsentRequest, user: dict = Depends(current_user)):
    if req.consent_type not in CONSENT_TYPES:
        raise HTTPException(400, "unknown consent")
    method, signature = "web_checkbox", None
    if req.consent_type == BIOMETRIC_CONSENT:
        if "terms" not in user["consents"]:
            raise HTTPException(403, {"code": "consent_required"})
        signature = (req.signature or "").strip()
        if len(signature) < 3:
            raise HTTPException(400, "type your full name to sign")
        method = "web_esignature"
    state["deps"].repo.record_consent(f"consent-{uuid.uuid4()}", user["user_id"], req.consent_type,
                                      POLICY_VERSION, method=method, signature=signature)
    if req.consent_type == BIOMETRIC_CONSENT and req.session_id:
        _owned_session(req.session_id, user)
        state["enqueue"]("/enroll", {"session_id": req.session_id, "user_id": user["user_id"]})
    return {"recorded": req.consent_type}


@app.delete("/api/consents/biometric_enrollment")
def withdraw_biometric(user: dict = Depends(current_user)):
    repo = state["deps"].repo
    with repo.transaction():
        SQLVoiceprintRepository(repo).delete(user["user_id"])
        repo.revoke_consent(user["user_id"], BIOMETRIC_CONSENT)
    return {"voiceprint_deleted": True}


# --- lessons -----------------------------------------------------------------------

@app.post("/api/uploads")
def create_upload(req: UploadRequest, request: Request, user: dict = Depends(consented_user)):
    check_limits(user)
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
def create_session(req: SessionRequest, user: dict = Depends(consented_user)):
    deps = state["deps"]
    check_limits(user)
    if not req.storage_uri.startswith(f"gs://{deps.blobs.bucket_name}/uploads/"):
        raise HTTPException(400, "storage_uri must come from /api/uploads")
    if req.num_speakers is not None and not 1 <= req.num_speakers <= 6:
        raise HTTPException(400, "num_speakers must be 1-6")
    if not deps.blobs.exists(req.storage_uri):
        raise HTTPException(400, "upload not finished")
    try:
        session_id = register_upload(deps, storage_uri=req.storage_uri, user_id=user["user_id"],
                                     max_seconds=MAX_MINUTES * 60)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state["enqueue"]("/process", {"session_id": session_id, "user_id": user["user_id"],
                                  "num_speakers": req.num_speakers})
    log.info("queued %s for %s", session_id, user["user_id"])
    return {"session_id": session_id}


@app.get("/api/sessions")
def list_sessions(user: dict = Depends(current_user)):
    return {"sessions": state["deps"].repo.list_user_sessions(user["user_id"])}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, user: dict = Depends(current_user)):
    deps = state["deps"]
    session = _owned_session(session_id, user)
    status, transcript = session["status"], deps.repo.latest_transcript(session_id)
    stage = status
    if status == "processing":
        stage = "analyzing" if transcript else "transcribing"
    audio = deps.repo.get_audio_asset(session_id)
    out = {"session_id": session_id, "status": status, "stage": stage, "created_at": session["created_at"],
           "duration_seconds": audio["duration_seconds"] if audio else None,
           "speaker_method": transcript["speaker_resolution_method"] if transcript else None}
    if status == "awaiting_speaker_selection":
        out["speakers"] = speaker_choices(deps, session_id)
    return out


@app.post("/api/sessions/{session_id}/speaker")
def choose_speaker(session_id: str, req: SpeakerRequest, user: dict = Depends(current_user)):
    deps = state["deps"]
    if _owned_session(session_id, user)["status"] != "awaiting_speaker_selection":
        raise HTTPException(409, "this lesson isn't waiting for a speaker choice")
    if req.speaker_label not in {c["speaker_label"] for c in speaker_choices(deps, session_id)}:
        raise HTTPException(400, "unknown speaker")
    deps.repo.update_session(session_id, status="processing")
    state["enqueue"]("/resume", {"session_id": session_id, "speaker_label": req.speaker_label})
    return {"session_id": session_id, "status": "processing"}


@app.get("/api/sessions/{session_id}/report", response_class=HTMLResponse)
def report1(session_id: str, user: dict = Depends(current_user)):
    _owned_session(session_id, user)
    snapshot = state["deps"].repo.latest_report_snapshot(session_id)
    if not snapshot:
        raise HTTPException(404, "no report yet")
    return render_html(json.loads(snapshot["payload_json"]))


@app.get("/api/sessions/{session_id}/report2")
def report2(session_id: str, user: dict = Depends(current_user)):
    _owned_session(session_id, user)
    export = state["deps"].repo.latest_report2_export(session_id)
    if not export:
        raise HTTPException(404, "no report yet")
    return Response(state["deps"].blobs.read_bytes(export["storage_uri"]),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="lomb_report2_{session_id[5:13]}.xlsx"'})


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(current_user)):
    if _owned_session(session_id, user)["status"] == "processing":
        raise HTTPException(409, "wait until this lesson finishes processing, then delete it")
    delete_lesson(state["deps"], session_id)
    return {"deleted": session_id}


# --- scheduled cleanup -------------------------------------------------------------

@app.post("/internal/cleanup")
def cleanup(authorization: str | None = Header(None)):
    """Destroys voiceprints unused for a year (the privacy policy's retention
    schedule). Only Cloud Scheduler, signed in as LOMB_TASKS_SA, may call it."""
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2 import id_token
    try:
        claims = id_token.verify_oauth2_token((authorization or "")[len("Bearer "):], GoogleRequest(),
                                              audience=os.environ["LOMB_API_URL"])
        assert claims.get("email") == os.environ["LOMB_TASKS_SA"] and claims.get("email_verified")
    except Exception:
        raise HTTPException(403, "forbidden")
    repo = state["deps"].repo
    voiceprints = SQLVoiceprintRepository(repo)
    cutoff = (_now() - timedelta(days=VOICEPRINT_UNUSED_DAYS)).isoformat()
    expired = voiceprints.unused_since(cutoff)
    for user_id in expired:
        with repo.transaction():
            voiceprints.delete(user_id)
            repo.revoke_consent(user_id, BIOMETRIC_CONSENT)
    log.info("cleanup: %d voiceprints unused since %s destroyed", len(expired), cutoff)
    return {"voiceprints_destroyed": len(expired)}


@app.get("/health")  # not /healthz: Cloud Run reserves paths ending in "z"
def health():
    return {"ok": "deps" in state}
