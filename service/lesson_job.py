"""
One lesson, end to end -- the functions the production web API and GPU
worker call (2026-10-11, Google Cloud build, step 1).

    submit_lesson()        upload: sessions row ('processing') + audio_assets
                           row + the audio file in the BlobStore
    process_lesson()       GPU worker: transcribe -> transcripts row ->
                           identify the learner by voiceprint ->
                           speaker_resolutions rows -> analyse (or pause at
                           'awaiting_speaker_selection' with the candidates)
    resume_with_speaker()  after the user picks "which one is you?"
    analyze_lesson()       pipeline -> analysis_runs -> information_items,
                           correction_checks, metric_values -> Report 1
                           snapshot -> Report 2 export -> 'complete'.
                           Also used to re-analyse an existing transcript:
                           a new analysis_run on the SAME session (closes the
                           test script's "re-run = new session" gap, D1).

Order matches the product, not the old test script: the session exists the
moment the audio is uploaded, so the UI can show "processing" while the GPU
works. Any exception marks the session 'failed' and is re-raised.

An analysis's rows are written in one transaction (D4), on SQLite locally
or Cloud SQL Postgres in production. Not here yet (later steps): Cloud
Storage, users/consent, rate limits.
"""

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pipeline import run_pipeline_from_turns
from reporting.report import build_report
from reporting.report2 import build_report2_xlsx
from storage.blob_store import BlobStore
from storage.storage import StorageRepository, persist_pipeline_result
from transcript_processing.whisperx_adapter import (
    REPETITION_RUN_THRESHOLD, filter_repetition_loops, from_whisperx_transcript,
)

log = logging.getLogger("lomb.lesson_job")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


@dataclass
class LessonDeps:
    """Everything a lesson job needs, injected -- local and cloud differ only here."""
    repo: StorageRepository
    blobs: BlobStore
    transcriber: Any                     # .transcribe(audio_path, num_speakers=None) -> WhisperX dict
    provider: Any                        # LLMProvider (OpenAIProvider in production)
    speaker_service: Any = None          # voice_enrollment SpeakerResolutionService; None = always ask the user
    llm_model: str = "unknown"
    llm_settings: dict = field(default_factory=dict)
    ids: Callable[[str], str] = new_id


@dataclass
class LessonResult:
    session_id: str
    status: str                          # 'complete' | 'awaiting_speaker_selection' | 'failed'
    transcript_id: str | None = None
    resolution_id: str | None = None
    candidates: list[dict] = field(default_factory=list)   # for the "which one is you?" picker
    run_id: str | None = None
    report: dict | None = None
    result: dict | None = None           # the full pipeline result
    guardrail_dropped: list[dict] = field(default_factory=list)


# --- helpers -----------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_duration_seconds(path: Path) -> float:
    """Audio length via ffprobe (ffmpeg is required by WhisperX anyway); 0.0 if it can't tell."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def asr_provenance(whisperx_json: dict) -> tuple[str | None, str | None, dict | None]:
    """(asr_model, diarization_model, asr_settings) from the transcript's
    _meta block. Fields an older transcript lacks are left out -- recorded
    as unknown, never guessed."""
    meta = whisperx_json.get("_meta")
    if not meta:
        return None, None, None
    settings = {k: meta[k] for k in ("compute_type", "batch_size", "language", "num_speakers_hint",
                                     "engine_version", "hardened_script_version") if k in meta}
    if meta.get("initial_prompt"):
        settings["initial_prompt_version"] = hashlib.sha256(meta["initial_prompt"].encode("utf-8")).hexdigest()[:12]
    return meta.get("model"), meta.get("diarization_model"), settings


def _turns(whisperx_json: dict) -> tuple[list, list[dict]]:
    _, dropped = filter_repetition_loops(whisperx_json.get("segments", []))
    for s in dropped:
        log.warning("guardrail: dropped likely Whisper repetition loop %.1f-%.1fs", s["start"], s["end"])
    return from_whisperx_transcript(whisperx_json), dropped


def _fetch_audio(deps: LessonDeps, session_id: str, workdir: Path) -> Path:
    asset = deps.repo.get_audio_asset(session_id)
    return deps.blobs.fetch(asset["storage_uri"], workdir / f"audio{Path(asset['storage_uri']).suffix}")


# --- the job steps ------------------------------------------------------------

def submit_lesson(deps: LessonDeps, *, audio_path: str | Path, user_id: str | None,
                  source: str = "public_demo") -> str:
    """Upload step: create the session ('processing') and store the audio."""
    audio_path = Path(audio_path)
    session_id, audio_id = deps.ids("sess"), deps.ids("audio")
    deps.repo.create_session(session_id, source=source, user_id=user_id, status="processing")
    uri = deps.blobs.put_file(audio_path, f"audio/{session_id}/{audio_id}{audio_path.suffix.lower()}")
    deps.repo.create_audio_asset(audio_id, session_id, storage_uri=uri, format=audio_path.suffix.lstrip(".").lower(),
                                 duration_seconds=probe_duration_seconds(audio_path),
                                 size_bytes=audio_path.stat().st_size, sha256=sha256_of(audio_path))
    return session_id


def register_upload(deps: LessonDeps, *, storage_uri: str, user_id: str | None,
                    source: str = "public_demo") -> str:
    """Upload step when the audio is already in the BlobStore -- production:
    the browser sends it straight to Cloud Storage, since a lesson is too big
    to pass through the web API. Downloads it once to measure it, then creates
    the session ('processing') and audio_assets rows together."""
    session_id, audio_id = deps.ids("sess"), deps.ids("audio")
    suffix = Path(storage_uri).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        local = deps.blobs.fetch(storage_uri, Path(tmp) / f"audio{suffix}")
        duration, size, digest = probe_duration_seconds(local), local.stat().st_size, sha256_of(local)
    if duration <= 0:
        raise ValueError("not a readable audio file")
    with deps.repo.transaction():
        deps.repo.create_session(session_id, source=source, user_id=user_id, status="processing")
        deps.repo.create_audio_asset(audio_id, session_id, storage_uri=storage_uri, format=suffix.lstrip("."),
                                     duration_seconds=duration, size_bytes=size, sha256=digest)
    return session_id


def speaker_choices(deps: LessonDeps, session_id: str) -> list[dict]:
    """The "which one is you?" options for a session at the picker, rebuilt
    from what's stored (latest transcript + speaker_resolutions) so the page
    can be reloaded at any point. Most talkative speaker first."""
    transcript = deps.repo.latest_transcript(session_id)
    with tempfile.TemporaryDirectory() as tmp:
        raw = deps.blobs.fetch(transcript["raw_json_uri"], Path(tmp) / "transcript.json")
        turns, _ = _turns(json.loads(raw.read_text(encoding="utf-8")))
    resolutions = deps.repo.list_speaker_resolutions(session_id)
    similarity = {c["speaker_label"]: c["similarity"] for c in resolutions[-1]["candidates"]} if resolutions else {}
    choices = []
    for label in sorted({t.speaker_id for t in turns}):
        own = [t for t in turns if t.speaker_id == label]
        choices.append({"speaker_label": label, "similarity": similarity.get(label),
                        "word_count": sum(len(t.text.split()) for t in own),
                        "talk_seconds": round(sum(t.end - t.start for t in own), 1),
                        "preview_text": " ".join(t.text for t in own)[:300]})
    return sorted(choices, key=lambda c: -c["word_count"])


def process_lesson(deps: LessonDeps, *, session_id: str, user_id: str | None,
                   num_speakers: int | None = None, speaker_override: str | None = None) -> LessonResult:
    """GPU worker step. Transcribes, identifies the learner, then analyses --
    or stops at 'awaiting_speaker_selection' when the voiceprint match isn't
    confident (the UI then shows the candidates)."""
    try:
        deps.repo.update_session(session_id, status="processing")  # also when a queue retries a failed attempt
        with tempfile.TemporaryDirectory() as tmp:
            audio = _fetch_audio(deps, session_id, Path(tmp))
            whisperx_json = deps.transcriber.transcribe(str(audio), num_speakers=num_speakers)
            transcript_id = deps.ids("transcript")
            raw_uri = deps.blobs.put_bytes(json.dumps(whisperx_json, ensure_ascii=False).encode("utf-8"),
                                           f"transcripts/{session_id}/{transcript_id}.json")
            asr_model, diarization_model, asr_settings = asr_provenance(whisperx_json)
            deps.repo.create_transcript(transcript_id, session_id, engine="whisperx", raw_json_uri=raw_uri,
                                        audio_id=deps.repo.get_audio_asset(session_id)["audio_id"],
                                        asr_model=asr_model, diarization_model=diarization_model,
                                        asr_settings=asr_settings)
            turns, dropped = _turns(whisperx_json)
            detected = sorted({t.speaker_id for t in turns})

            if speaker_override:
                if speaker_override not in detected:
                    raise ValueError(f"speaker {speaker_override!r} not in detected speakers {detected}")
                target, method, resolution = speaker_override, "manual_picker", None
            else:
                resolution = _resolve_speaker(deps, session_id, user_id, audio, turns)
                if resolution["needs_manual"]:
                    deps.repo.update_session(session_id, status="awaiting_speaker_selection")
                    return LessonResult(session_id, "awaiting_speaker_selection", transcript_id=transcript_id,
                                        resolution_id=resolution["resolution_id"],
                                        candidates=resolution["candidates"], guardrail_dropped=dropped)
                target, method = resolution["label"], "voiceprint_auto"

            deps.repo.set_transcript_speaker(transcript_id, target, method)
            out = analyze_lesson(deps, session_id=session_id, transcript_id=transcript_id, turns=turns,
                                 target_speaker=target)
            out.resolution_id = resolution["resolution_id"] if resolution else None
            out.guardrail_dropped = dropped
            return out
    except Exception:
        deps.repo.update_session(session_id, status="failed")
        log.exception("lesson %s failed", session_id)
        raise


def _resolve_speaker(deps: LessonDeps, session_id: str, user_id: str | None, audio: Path, turns: list) -> dict:
    """Voiceprint match, persisted to speaker_resolutions(+_candidates)
    whatever the outcome. No speaker_service or no user -> always ask."""
    from voice_enrollment.audio_decode import extract_speaker_audio
    from voice_enrollment.voice_enrollment import SpeakerAudio

    resolution_id = deps.ids("resolution")
    labels = sorted({t.speaker_id for t in turns})
    if deps.speaker_service is None or user_id is None:
        ranked, reason, label, match_t, margin_t = [(lbl, 0.0) for lbl in labels], "no_enrolled_voiceprint", None, None, None
    else:
        waveforms = extract_speaker_audio(str(audio), turns)
        res = deps.speaker_service.resolve(user_id, [SpeakerAudio(lbl, wf) for lbl, wf in waveforms.items()])
        ranked = [(c.speaker_label, c.similarity) for c in res.candidates]
        reason = res.reason
        label = None if res.needs_manual_confirmation else res.resolved_speaker_label
        match_t, margin_t = deps.speaker_service.match_threshold, deps.speaker_service.margin_threshold
    deps.repo.create_speaker_resolution(
        resolution_id, session_id, user_id=user_id, resolved_speaker_label=label, reason=reason,
        outcome="auto_resolved" if label else "flagged", match_threshold=match_t, margin_threshold=margin_t,
        candidates=[(deps.ids("cand"), lbl, sim) for lbl, sim in ranked])
    preview = {lbl: " ".join(t.text for t in turns if t.speaker_id == lbl)[:200] for lbl in labels}
    return {"resolution_id": resolution_id, "needs_manual": label is None, "label": label,
            "candidates": [{"speaker_label": lbl, "similarity": round(sim, 3), "preview_text": preview.get(lbl, "")}
                           for lbl, sim in ranked]}


def resume_with_speaker(deps: LessonDeps, *, session_id: str, speaker_label: str) -> LessonResult:
    """After the user answers "which one is you?": record the pick, then analyse."""
    try:
        transcript = deps.repo.latest_transcript(session_id)
        with tempfile.TemporaryDirectory() as tmp:
            raw = deps.blobs.fetch(transcript["raw_json_uri"], Path(tmp) / "transcript.json")
            turns, dropped = _turns(json.loads(raw.read_text(encoding="utf-8")))
        detected = sorted({t.speaker_id for t in turns})
        if speaker_label not in detected:
            raise ValueError(f"speaker {speaker_label!r} not in detected speakers {detected}")
        resolutions = deps.repo.list_speaker_resolutions(session_id)
        if resolutions:
            deps.repo.record_manual_speaker_pick(resolutions[-1]["resolution_id"], speaker_label)
        deps.repo.set_transcript_speaker(transcript["transcript_id"], speaker_label, "manual_picker")
        deps.repo.update_session(session_id, status="processing")
        out = analyze_lesson(deps, session_id=session_id, transcript_id=transcript["transcript_id"], turns=turns,
                             target_speaker=speaker_label)
        out.resolution_id = resolutions[-1]["resolution_id"] if resolutions else None
        out.guardrail_dropped = dropped
        return out
    except Exception:
        deps.repo.update_session(session_id, status="failed")
        log.exception("lesson %s failed after speaker pick", session_id)
        raise


def analyze_lesson(deps: LessonDeps, *, session_id: str, transcript_id: str, turns: list,
                   target_speaker: str) -> LessonResult:
    """Analysis + every downstream table for one analysis run. Calling it
    again on the same session/transcript is a re-analysis: a new run_id,
    nothing overwritten."""
    result = run_pipeline_from_turns(deps.provider, turns, target_speaker_id=target_speaker)
    run_id = deps.ids("run")
    settings = {"error_method": result["error_method"], "count_unit": result["count_unit"],
                "chunk_seconds": result["chunk_seconds"], "repetition_run_threshold": REPETITION_RUN_THRESHOLD,
                "llm_generation_config": deps.llm_settings}
    # One transaction (D4): a failure anywhere below leaves no half-written run.
    # The LLM calls above stay outside it, so no connection is held open during them.
    with deps.repo.transaction():
        deps.repo.create_analysis_run(run_id, session_id, transcript_id, llm_model=deps.llm_model,
                                      prompt_versions=result["fluenceme_versions"], settings=settings)
        persist_pipeline_result(deps.repo, session_id, transcript_id, result, id_factory=lambda: deps.ids("id"),
                                run_id=run_id)
        report = build_report(result)
        deps.repo.create_report_snapshot(deps.ids("report"), session_id, json.dumps(report, ensure_ascii=False),
                                         run_id=run_id, transcript_id=transcript_id)
        items = deps.repo.list_information_items(session_id, run_id=run_id)
        export_id = deps.ids("export")
        uri = deps.blobs.put_bytes(build_report2_xlsx(items), f"exports/{session_id}/{export_id}.xlsx")
        deps.repo.create_report2_export(export_id, session_id, uri, run_id=run_id, row_count=len(items))
        deps.repo.update_session(session_id, status="complete", word_count=result["word_count"],
                                 duration_seconds=result["duration_seconds"])
    return LessonResult(session_id, "complete", transcript_id=transcript_id, run_id=run_id, report=report,
                        result=result)
