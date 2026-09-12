"""
enroll_api.py -- the actual HTTP endpoint for active voice enrollment.
This is the first piece of lomb_backend_prd_v1.md Section 4's pipeline
that has ever been wired up to a running server; everything before this
(voice_enrollment.py, audio_decode.py) was pure logic with no HTTP layer.

SCOPE, DELIBERATELY NARROW: this is the smallest end-to-end slice that
proves active enrollment works -- upload a recording, get a voiceprint on
file. It does NOT include: /analyze, the diarization-glue step, the
picker's /analyze/{job_id}/speaker endpoint, or any of the rest of
Section 4's pipeline. Those need the same audio_decode.py + resolve()
pieces but aren't built here.

user_id IS HARDCODED TO "1". lomb_prd_v1.md Section 8 (persistent
identity/accounts) is still unscoped -- rather than invent an auth scheme
this app has no business deciding, every request here is treated as the
same single user, "1", so the actual enrollment/resolution logic can be
exercised end-to-end today. Swapping in real identity later means
replacing USER_ID_FOR_NOW's one call site with whatever the auth layer
resolves (a cookie, a token, a real account row) -- nothing about
voice_enrollment.py or audio_decode.py needs to change for that.

App factory pattern (create_app), not a bare module-level `app`, so
test_enroll_api.py can inject a FakeEmbedder and a temp-file repository
instead of loading the real ~80MB speechbrain model on every test run.
Production wiring (main.py) constructs the real ones.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from audio_decode import AudioDecodeError, decode_audio_bytes
from voice_enrollment import (
    MIN_ACTIVE_ENROLLMENT_SECONDS,
    SpeakerEmbedder,
    SpeakerResolutionService,
    VoiceprintRepository,
)

USER_ID_FOR_NOW = "1"  # see module docstring -- the one line that changes when real identity exists

WIDGET_HTML_PATH = Path(__file__).resolve().parent / "enroll_widget.html"


def create_app(embedder: SpeakerEmbedder, repository: VoiceprintRepository) -> FastAPI:
    service = SpeakerResolutionService(embedder, repository)
    app = FastAPI(title="Lomb voice enrollment (dev slice)")

    @app.get("/")
    def widget():
        """Serves the mic-capture widget itself, same-origin, so the
        widget's fetch('/enroll') needs no CORS configuration. Fine for
        this dev slice; a real deployment serves the widget from wherever
        the rest of the frontend lives instead."""
        return FileResponse(WIDGET_HTML_PATH)

    @app.post("/enroll")
    async def enroll(recording: UploadFile, force: bool = False):
        raw_bytes = await recording.read()

        try:
            waveform = decode_audio_bytes(raw_bytes)
        except AudioDecodeError as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

        try:
            service.enroll_active(USER_ID_FOR_NOW, waveform, force=force)
        except ValueError as e:
            # Both of enroll_active's own failure modes (too short;
            # already enrolled without force=True) are ValueError, and
            # both already carry a message written for a human to read --
            # see voice_enrollment.py's enroll_active() docstring.
            return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

        duration_seconds = round(len(waveform) / 16000, 1)
        return {"ok": True, "message": f"Voice enrolled from a {duration_seconds}s recording."}

    @app.get("/status")
    def status():
        """Not part of the real product surface -- lets the widget (and
        this file's own manual testing) show whether user_id=1 already
        has a voiceprint on file, since the widget has no other way to
        know before it tries force=True."""
        stored = repository.get(USER_ID_FOR_NOW)
        return {"enrolled": stored is not None, "sample_count": stored[1] if stored else 0}

    return app
