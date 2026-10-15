"""
The GPU worker (2026-10-14, build step 3): a Cloud Run service on an NVIDIA
L4 that runs lesson jobs. One lesson at a time per instance (concurrency 1);
models load once when the instance starts, before it accepts requests, and
stay loaded while it's warm.

Not public: Cloud Run requires an identity token from a caller allowed to
invoke it (Cloud Tasks from the web API, step 5; Dan's account for tests).

  POST /process  {session_id, user_id?, num_speakers?}
       transcribe -> speaker picker pause (no voiceprints in the cloud yet)
  POST /resume   {session_id, speaker_label}
       record the pick -> analyse -> reports
  GET  /healthz

Both POSTs answer when the work is done. A failure marks the session
'failed' (lesson_job does that) and returns 500, so a queue can retry.

Run locally: uvicorn worker.main:app --port 8080   (needs a GPU + the env
vars listed in service/cloud.py)
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from service.cloud import cloud_deps  # noqa: E402
from service.lesson_job import LessonResult, process_lesson, resume_with_speaker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("lomb.worker")
state = {}


@asynccontextmanager
async def lifespan(_app):
    log.info("loading models and connecting")
    state["deps"] = cloud_deps(with_transcriber=True)
    log.info("ready on %s", state["deps"].transcriber.device)
    yield


app = FastAPI(lifespan=lifespan)


class ProcessRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    num_speakers: int | None = None


class ResumeRequest(BaseModel):
    session_id: str
    speaker_label: str


def _summary(out: LessonResult) -> dict:
    return {"session_id": out.session_id, "status": out.status, "transcript_id": out.transcript_id,
            "run_id": out.run_id, "candidates": out.candidates,
            "guardrail_dropped": len(out.guardrail_dropped)}


# Plain `def`, not `async def`: FastAPI runs these in a thread, so a
# minutes-long lesson doesn't block /healthz.
@app.post("/process")
def process(req: ProcessRequest):
    try:
        out = process_lesson(state["deps"], session_id=req.session_id, user_id=req.user_id,
                             num_speakers=req.num_speakers)
    except Exception as e:
        raise HTTPException(500, f"lesson {req.session_id} failed: {type(e).__name__}: {e}")
    return _summary(out)


@app.post("/resume")
def resume(req: ResumeRequest):
    try:
        out = resume_with_speaker(state["deps"], session_id=req.session_id, speaker_label=req.speaker_label)
    except Exception as e:
        raise HTTPException(500, f"lesson {req.session_id} failed: {type(e).__name__}: {e}")
    return _summary(out)


@app.get("/healthz")
def healthz():
    return {"ok": "deps" in state}
