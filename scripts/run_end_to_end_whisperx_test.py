"""
Phase 1 end-to-end test: a real whisperX+pyannote transcript in, real rows
in the 6 essential data tables out. This is the actual thing Dan asked
for -- "we get back the data tables we want" -- run against one of the 33
already-produced real transcripts (whisperX_batch_job.ipynb's own output),
not a synthetic fixture.

Chain this script proves, end to end:
  whisperX+pyannote JSON file
    -> transcript_processing.whisperx_adapter.from_whisperx_transcript()   Turn objects
    -> pipeline.pipeline.run_pipeline_from_turns()                        scored metrics
    -> storage.storage.persist_pipeline_result()                         information_items + metric_values rows
    -> a minimal Report 2 .xlsx export                                    report2_exports row

Every step after the adapter is unchanged, off-the-shelf code this
project already has (or that this same effort just added as its own
reusable module, not a one-off script-local copy) -- this script is
orchestration glue, not new logic. That's the actual "easily migrate to
a cloud/production env" property Dan asked for: a real backend service
would call the exact same run_pipeline_from_turns()/persist_pipeline_
result() functions this script calls, just triggered by a job queue
instead of argv.

WHAT THIS DELIBERATELY DOES NOT DO (placeholders, stated plainly, not
glossed over):

  - Speaker resolution: real voiceprint matching (2026-09-24) --
    SpeakerResolutionService against the user's enrolled voiceprint, using
    per-speaker audio sliced by voice_enrollment.audio_decode.
    extract_speaker_audio(). The old "most words wins" guess is gone: it
    picked the fluent tutor instead of the learner on a real 5-min
    session. If the match isn't confident, this script stops and prints
    the candidates (the production picker-UI case) -- pass --speaker to
    confirm one by hand. No silent fallback. speaker_resolutions /
    speaker_resolution_candidates tables are not written yet (still out
    of storage.py's scope); the candidates are printed instead.

  - LLM provider: defaults to a FakeProvider (see pipeline.pipeline's own
    __main__ block for the original of this pattern) -- this sandbox has
    no route to api.openai.com (confirmed repeatedly elsewhere in this
    project), and Dan's own stated tolerance for this test is "accuracy
    doesn't matter too much yet." Pass --use-openai (with OPENAI_API_KEY
    set) to run the real LLM-assisted fluencemes for real, e.g. from Dan's
    own machine.

  - Audio: passed explicitly via --audio (required -- voiceprint matching
    needs it). audio_assets gets the real local path, size_bytes, format
    and sha256. duration_seconds still comes from the transcript's last
    word timestamp, not from decoding the audio.

  - Report 2 export: a minimal, real .xlsx (openpyxl) dump of this
    session's information_items -- proves report2_exports gets a real
    row pointing at a real file, not a claim that this is the final
    Report 2 design (no such design exists yet -- see
    lomb_reporting_requirements_v1.md for what's actually specified for
    Report 1; Report 2's own layout was never separately designed).

Usage:
  python run_end_to_end_whisperx_test.py --transcript PATH --audio PATH
      [--user-id ID] [--voiceprint-db PATH] [--speaker ID]
      [--db PATH] [--use-openai] [--known-prompt-leak TEXT]
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pipeline import run_pipeline_from_turns
from reporting.report import build_report
from storage.storage import SQLiteStorageRepository, persist_pipeline_result
from transcript_processing.whisperx_adapter import (
    REPETITION_RUN_THRESHOLD, filter_repetition_loops, from_whisperx_transcript,
)
from voice_enrollment.audio_decode import extract_speaker_audio
from voice_enrollment.voice_enrollment import (
    SpeakerAudio, SpeakerResolutionService, SQLiteVoiceprintRepository,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPT = REPO_ROOT / "data" / "whisperx_sample.json"
DEFAULT_DB = REPO_ROOT / "data" / "lomb_phase1_test.sqlite"
DEFAULT_VOICEPRINT_DB = REPO_ROOT / "scripts" / "voiceprints_dev.sqlite"
DEFAULT_USER_ID = "1"  # the one enrolled voiceprint in voiceprints_dev.sqlite (Dan)

# The 7 essential tables this whole test exists to put real rows into --
# printed at the end so "did this actually work" has a real number
# attached to it, not just "no exception was raised."
ESSENTIAL_TABLES = [
    "sessions", "audio_assets", "transcripts",
    "analysis_runs", "information_items", "metric_values", "report2_exports", "report_snapshots",
]


class FakeProvider:
    """Same pattern as pipeline.pipeline's own __main__ FakeProvider --
    duplicated here rather than imported because the original is defined
    inside that module's `if __name__ == "__main__":` block (not
    importable). Makes no linguistic judgment; only proves the wiring
    reaches classify() correctly. See pipeline.pipeline's own FakeProvider
    docstring for the fuller reasoning -- unchanged here."""

    def __init__(self):
        self.call_count = 0

    def classify(self, config, input_data):
        self.call_count += 1
        note = "FakeProvider -- orchestration test only, not a real judgment."
        if config.key == "STRUCTURE_BREADTH":
            answer = {"structures": ["none"], "confidence": "high", "reasoning": note}
        else:
            answer = {"error": False, "confidence": "high", "reasoning": note}
        return {"results": [{"index": it["index"], **answer} for it in json.loads(input_data)]}


def resolve_target_speaker(turns, audio_path: Path, user_id: str, voiceprint_db: Path):
    """Same call the production /analyze handler will make after
    diarization: slice each speaker's audio, match against the user's
    enrolled voiceprint. Returns the SpeakerResolution unchanged."""
    from voice_enrollment.speechbrain_embedder import EcapaSpeakerEmbedder  # heavy; import only when used

    waveforms = extract_speaker_audio(str(audio_path), turns)
    skipped = sorted({t.speaker_id for t in turns} - set(waveforms))
    if skipped:
        print(f"  (skipped for matching, <3s of speech: {skipped})")
    service = SpeakerResolutionService(
        embedder=EcapaSpeakerEmbedder(),
        repository=SQLiteVoiceprintRepository(voiceprint_db),
    )
    return service.resolve(user_id, [SpeakerAudio(label, wf) for label, wf in waveforms.items()])


def asr_provenance(whisperx_json: dict) -> tuple[str | None, str | None, dict | None]:
    """(asr_model, diarization_model, asr_settings) from the transcript's
    _meta block, written by hardened_whisperx_batch_job.py. Fields a given
    transcript's _meta doesn't have (older runs) come back None/absent --
    recorded as unknown, never guessed."""
    meta = whisperx_json.get("_meta")
    if not meta:
        return None, None, None
    settings = {k: meta[k] for k in ("compute_type", "batch_size", "language", "num_speakers_hint",
                                     "hardened_script_version") if k in meta}
    if meta.get("initial_prompt"):
        settings["initial_prompt_version"] = hashlib.sha256(meta["initial_prompt"].encode("utf-8")).hexdigest()[:12]
    return meta.get("model"), meta.get("diarization_model"), settings


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transcript_duration_seconds(turns) -> float:
    return max((t.end for t in turns), default=0.0)


def build_report2_export(items: list[dict], out_path: Path) -> int:
    """Minimal, real Report 2 export -- every information_items row for
    this session, one per line, plus a Summary sheet. See module
    docstring's placeholder note: this proves report2_exports gets a
    real file with real rows, not a finished Report 2 design."""
    import openpyxl
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "All Items"
    headers = ["feature_key", "utterance_ref", "input_ref", "review_status", "created_at", "output_json"]
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(name="Arial", bold=True)
    for row_i, item in enumerate(items, start=2):
        ws.cell(row=row_i, column=1, value=item["feature_key"]).font = Font(name="Arial")
        ws.cell(row=row_i, column=2, value=item["utterance_ref"]).font = Font(name="Arial")
        ws.cell(row=row_i, column=3, value=item["input_ref"]).font = Font(name="Arial")
        ws.cell(row=row_i, column=4, value=item["review_status"]).font = Font(name="Arial")
        ws.cell(row=row_i, column=5, value=item["created_at"]).font = Font(name="Arial")
        ws.cell(row=row_i, column=6, value=item["output_json"]).font = Font(name="Arial")
    for col_letter, width in zip("ABCDEF", (20, 14, 50, 14, 26, 60)):
        ws.column_dimensions[col_letter].width = width

    summary = wb.create_sheet("Summary")
    summary.cell(row=1, column=1, value="feature_key").font = Font(name="Arial", bold=True)
    summary.cell(row=1, column=2, value="flagged_count").font = Font(name="Arial", bold=True)
    feature_keys = sorted({item["feature_key"] for item in items})
    last_row = len(items) + 1
    for row_i, feature_key in enumerate(feature_keys, start=2):
        summary.cell(row=row_i, column=1, value=feature_key).font = Font(name="Arial")
        formula = f'=COUNTIF(\'All Items\'!A2:A{last_row},A{row_i})'
        summary.cell(row=row_i, column=2, value=formula).font = Font(name="Arial")
    summary.column_dimensions["A"].width = 20
    summary.column_dimensions["B"].width = 14

    wb.save(out_path)
    return len(items)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default=str(DEFAULT_TRANSCRIPT),
                         help="Path to a whisperX+pyannote output JSON file.")
    parser.add_argument("--audio", required=True,
                         help="Path to the source audio the transcript was produced from (needed for voiceprint matching).")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID,
                         help=f"Enrolled user whose voiceprint to match. Default: {DEFAULT_USER_ID}")
    parser.add_argument("--voiceprint-db", default=str(DEFAULT_VOICEPRINT_DB),
                         help="SQLite voiceprint store (SQLiteVoiceprintRepository).")
    parser.add_argument("--speaker", default=None,
                         help="Manually confirm the target speaker id (the picker-UI case) -- skips voiceprint matching.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite file to write into.")
    parser.add_argument("--use-openai", action="store_true",
                         help="Use the real OpenAIProvider (requires OPENAI_API_KEY) instead of FakeProvider.")
    parser.add_argument("--known-prompt-leak", default=None,
                         help="The exact initial_prompt text used for this job, for reliable prompt-leak filtering.")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    with open(transcript_path, "r", encoding="utf-8") as f:
        whisperx_json = json.load(f)

    print(f"Transcript: {transcript_path}")
    _, loop_segments = filter_repetition_loops(whisperx_json.get("segments", []))
    for s in loop_segments:
        print(f"GUARDRAIL: dropped likely Whisper repetition loop {s['start']:.1f}-{s['end']:.1f}s: "
              f"{s.get('text', '').strip()[:60]!r}...")
    turns = from_whisperx_transcript(whisperx_json, known_prompt_leak=args.known_prompt_leak)
    print(f"Adapted {len(turns)} turns, speakers detected: "
          f"{sorted({t.speaker_id for t in turns})}")

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"--audio not found: {audio_path}")
        return 1

    detected = sorted({t.speaker_id for t in turns})
    if args.speaker:
        if args.speaker not in detected:
            print(f"--speaker {args.speaker} is not one of the detected speakers {detected}")
            return 1
        target_speaker_id = args.speaker
        resolution_method = "manual_override"
        print(f"Target speaker: {target_speaker_id} (manually confirmed via --speaker)")
    else:
        print(f"Resolving target speaker by voiceprint (user_id={args.user_id})...")
        resolution = resolve_target_speaker(turns, audio_path, args.user_id, Path(args.voiceprint_db))
        for c in resolution.candidates:
            print(f"  {c.speaker_label}: similarity {c.similarity:.3f}")
        if resolution.needs_manual_confirmation:
            print(f"No confident match ({resolution.reason}). Stopping before any pipeline/LLM work -- "
                  f"rerun with --speaker <one of {detected}> to confirm by hand.")
            return 2
        target_speaker_id = resolution.resolved_speaker_label
        resolution_method = "voiceprint_auto"
        print(f"Target speaker: {target_speaker_id} (voiceprint match, {resolution.reason})")

    if args.use_openai:
        from providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
        print(f"Provider: OpenAIProvider (model={provider.model})")
    else:
        provider = FakeProvider()
        print("Provider: FakeProvider (no network call -- see module docstring)")

    result = run_pipeline_from_turns(provider, turns, target_speaker_id=target_speaker_id)
    print(f"\nPipeline result: sentence_count={result['sentence_count']}  "
          f"word_count={result['word_count']}  duration_seconds={result['duration_seconds']:.1f}  "
          f"wpm={result['wpm']}  chunks={result['chunk_count']}  llm_calls={result['llm_call_count']}")
    if result["llm_missing"]:
        print(f"WARNING: model omitted sentences from its batch answer: {result['llm_missing']}")

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    repo = SQLiteStorageRepository(db_path)
    print(f"\nDatabase: {db_path}")

    session_id = f"sess-{uuid.uuid4()}"
    repo.create_session(session_id, source="personal_pipeline")

    audio_id = f"audio-{uuid.uuid4()}"
    repo.create_audio_asset(
        audio_id, session_id,
        storage_uri=str(audio_path.resolve()), format=audio_path.suffix.lstrip("."),
        duration_seconds=transcript_duration_seconds(turns),
        size_bytes=audio_path.stat().st_size,
        sha256=sha256_of(audio_path),
    )
    print(f"audio_assets: {audio_path.resolve()} ({audio_path.stat().st_size} bytes)")

    # Transcription provenance lives on the transcript (what produced the text).
    asr_model, diarization_model, asr_settings = asr_provenance(whisperx_json)
    transcript_id = f"transcript-{uuid.uuid4()}"
    repo.create_transcript(
        transcript_id, session_id, engine="whisperx", raw_json_uri=str(transcript_path.resolve()),
        audio_id=audio_id, target_speaker_label=target_speaker_id,
        speaker_resolution_method=resolution_method,
        asr_model=asr_model, diarization_model=diarization_model, asr_settings=asr_settings,
    )
    print(f"transcripts: asr_model={asr_model}  diarization_model={diarization_model}  settings={asr_settings}")

    # Analysis provenance lives on the run (what produced the verdicts).
    run_id = f"run-{uuid.uuid4()}"
    llm_model = provider.model if args.use_openai else "FakeProvider"
    run_settings = {
        "chunk_seconds": result["chunk_seconds"],
        "repetition_run_threshold": REPETITION_RUN_THRESHOLD,
        "known_prompt_leak_given": args.known_prompt_leak is not None,
    }
    if args.use_openai:
        from providers.openai_provider import DEFAULT_GENERATION_CONFIG
        run_settings["llm_generation_config"] = DEFAULT_GENERATION_CONFIG
    repo.create_analysis_run(run_id, session_id, transcript_id, llm_model=llm_model,
                             prompt_versions=result["fluenceme_versions"], settings=run_settings)
    print(f"analysis_runs: {run_id}  llm_model={llm_model}  "
          f"{len(result['fluenceme_versions'])} fluenceme versions  settings={run_settings}")

    write_summary = persist_pipeline_result(
        repo, session_id, transcript_id, result, id_factory=lambda: str(uuid.uuid4()), run_id=run_id
    )
    print(f"\npersist_pipeline_result() -> {write_summary}")

    report1_payload = build_report(result)
    report_id = f"report-{uuid.uuid4()}"
    repo.create_report_snapshot(
        report_id, session_id, json.dumps(report1_payload, ensure_ascii=False),
        run_id=run_id, transcript_id=transcript_id,
    )
    print(f"report_snapshots: 1 row written (report_id={report_id})")

    repo.update_session(
        session_id, status="complete",
        word_count=result["word_count"], duration_seconds=result["duration_seconds"],
    )

    items = repo.list_information_items(session_id, run_id=run_id)
    export_dir = db_path.parent / "report2_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{session_id}_report2.xlsx"
    row_count = build_report2_export(items, export_path)

    import subprocess
    recalc_script = Path(
        "/root/.claude/skills/synced/e39eca76-c9e9-4f1b-b9b3-1a02574a6a98_ec68c5c2-312c-4631-a3e7-869990cd593d"
        "/xlsx/scripts/recalc.py"
    )
    if recalc_script.exists():
        recalc_result = subprocess.run(
            [sys.executable, str(recalc_script), str(export_path)],
            capture_output=True, text=True,
        )
        print(f"recalc.py: {recalc_result.stdout.strip() or recalc_result.stderr.strip()}")

    repo.create_report2_export(f"export-{uuid.uuid4()}", session_id, str(export_path),
                               run_id=run_id, row_count=row_count)
    print(f"report2_exports: {export_path} ({row_count} rows)")

    print(f"\n=== Row counts, the {len(ESSENTIAL_TABLES)} essential tables (session {session_id}) ===")
    with sqlite3.connect(db_path) as conn:
        for table in ESSENTIAL_TABLES:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            print(f"  {table:<20} {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
