"""
End-to-end test of one lesson on a local machine: an existing
WhisperX+pyannote transcript and its audio in, every data table filled out.

Since 2026-10-11 this is a thin command-line wrapper around
service/lesson_job.py -- the exact functions the production web API and GPU
worker call -- run against local stand-ins:

  database    SQLite file (--db)                 production: Cloud SQL Postgres
  files       a folder next to the database      production: Cloud Storage
  transcriber the given transcript JSON          production: WhisperXTranscriber on an L4 GPU
  speaker     voiceprint match against
              --voiceprint-db, or --speaker      production: same, plus the picker UI
  LLM         FakeProvider, or --use-openai      production: OpenAIProvider

Order follows the product: session + audio stored first, then transcript,
speaker decision (speaker_resolutions), analysis, reports. If the voiceprint
match isn't confident the run stops at 'awaiting_speaker_selection' and
prints the candidates -- rerun with --speaker to confirm by hand (exit 2).

Writes pipeline_result.json and report1.html next to the database for review.

Usage:
  python scripts/run_end_to_end_whisperx_test.py --transcript PATH --audio PATH
      [--user-id ID] [--voiceprint-db PATH] [--speaker ID] [--db PATH]
      [--use-openai [--llm-model M] [--reasoning-effort E]]
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fake_provider import FakeProvider  # noqa: E402
from reporting.report import render_html  # noqa: E402
from service.lesson_job import LessonDeps, process_lesson, submit_lesson  # noqa: E402
from storage.blob_store import LocalBlobStore  # noqa: E402
from storage.storage import SQLiteStorageRepository  # noqa: E402
from transcript_processing.whisperx_engine import PrecomputedTranscriber  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "lomb_phase1_test.sqlite"
DEFAULT_VOICEPRINT_DB = REPO_ROOT / "scripts" / "voiceprints_dev.sqlite"
DEFAULT_USER_ID = "1"  # the one enrolled voiceprint in voiceprints_dev.sqlite (Dan)

# Every table a lesson writes, printed at the end so "did it work" has numbers.
SESSION_TABLES = [
    "sessions", "audio_assets", "transcripts", "speaker_resolutions", "analysis_runs", "information_items",
    "correction_checks", "metric_values", "report2_exports", "report_snapshots",
]


def print_error_summary(result: dict) -> None:
    """What the finder found and where the sorter put it."""
    from pipeline.flag_quality import rejection_reason
    from pipeline.registry import ERROR_TAGS

    print(f"Errors found: {result['found_error_count']} (one entry per error)")
    for key in ERROR_TAGS:
        flagged = [e for e in result["results"].get(key, []) if e["output"] and e["output"].get("error") is True]
        if not flagged:
            continue
        rejected = [r for r in (rejection_reason(e["output"], e["input"]) for e in flagged) if r]
        line = f"  {key:<12} {len(flagged) - len(rejected)} counted"
        if rejected:
            line += f", {len(rejected)} auto-rejected ({', '.join(sorted(set(rejected)))})"
        print(line)


def speaker_service(voiceprint_db: Path):
    from voice_enrollment.speechbrain_embedder import EcapaSpeakerEmbedder  # heavy; only when matching
    from voice_enrollment.voice_enrollment import SpeakerResolutionService, SQLiteVoiceprintRepository
    return SpeakerResolutionService(embedder=EcapaSpeakerEmbedder(),
                                    repository=SQLiteVoiceprintRepository(voiceprint_db))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", required=True, help="An existing WhisperX+pyannote output JSON.")
    parser.add_argument("--audio", required=True, help="The audio that transcript was made from.")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help=f"Enrolled user to match. Default: {DEFAULT_USER_ID}")
    parser.add_argument("--voiceprint-db", default=str(DEFAULT_VOICEPRINT_DB))
    parser.add_argument("--speaker", default=None, help="Confirm the learner's speaker label by hand (skips matching).")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite file to write into.")
    parser.add_argument("--use-openai", action="store_true", help="Real OpenAIProvider (needs OPENAI_API_KEY).")
    parser.add_argument("--llm-model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default=None, help="none | low | medium | high (with --use-openai)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.use_openai:
        from providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(model=args.llm_model, reasoning_effort=args.reasoning_effort)
        llm_model, llm_settings = provider.model, provider.generation_config
    else:
        provider, llm_model, llm_settings = FakeProvider(), "FakeProvider", {}
    print(f"Provider: {llm_model}")

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    deps = LessonDeps(
        repo=SQLiteStorageRepository(db_path),
        blobs=LocalBlobStore(db_path.parent / "blobs"),
        transcriber=PrecomputedTranscriber(args.transcript),
        provider=provider,
        speaker_service=None if args.speaker else speaker_service(Path(args.voiceprint_db)),
        llm_model=llm_model,
        llm_settings=llm_settings,
    )

    session_id = submit_lesson(deps, audio_path=args.audio, user_id=args.user_id, source="personal_pipeline")
    print(f"Session {session_id}: uploaded")
    out = process_lesson(deps, session_id=session_id, user_id=args.user_id, speaker_override=args.speaker)

    for r in deps.repo.list_speaker_resolutions(session_id):
        print(f"Speaker decision: {r['reason']} -> {r['outcome']} "
              f"({', '.join(f'{c['speaker_label']} {c['similarity']:.3f}' for c in r['candidates'])})")
    for s in out.guardrail_dropped:
        print(f"GUARDRAIL: dropped likely Whisper repetition loop {s['start']:.1f}-{s['end']:.1f}s")

    if out.status == "awaiting_speaker_selection":
        print("\nNo confident voiceprint match -- which one is you?")
        for c in out.candidates:
            print(f"  {c['speaker_label']} ({c['similarity']}): {c['preview_text'][:100]}")
        print("Rerun with --speaker <label> to confirm by hand.")
        return 2

    result = out.result
    print(f"\nPipeline result: sentence_count={result['sentence_count']}  word_count={result['word_count']}  "
          f"duration_seconds={result['duration_seconds']:.1f}  wpm={result['wpm']}  "
          f"chunks={result['chunk_count']}  llm_calls={result['llm_call_count']}")
    if result["llm_missing"]:
        print(f"WARNING: model left items out of its answer: {result['llm_missing']}")
    print_error_summary(result)

    with open(db_path.parent / "pipeline_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(db_path.parent / "report1.html", "w", encoding="utf-8") as f:
        f.write(render_html(out.report))
    print(f"saved: {db_path.parent / 'pipeline_result.json'} and report1.html")

    print(f"\n=== Rows written for session {session_id} (run {out.run_id}) ===")
    with sqlite3.connect(db_path) as conn:
        for table in SESSION_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
            print(f"  {table:<22} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
