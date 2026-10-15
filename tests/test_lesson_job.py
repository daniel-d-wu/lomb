"""
Offline test of service/lesson_job.py -- every path a production lesson can
take, with no GPU, no network and no API key:

  A. voiceprint confident  -> analysed straight through, every table filled
  B. voiceprint unsure     -> 'awaiting_speaker_selection' with candidates,
                              then resume_with_speaker() -> complete
  C. re-analysis           -> a second analysis_run on the SAME session
  D. a failure mid-job     -> session marked 'failed'
  E. a failure while saving -> the whole analysis rolls back (D4)

Uses a tiny synthetic WhisperX transcript, the pipeline's FakeProvider, a
fake speaker-resolution service, and a local folder for files. The database
is a temporary SQLite file, or with --cloudsql the real Cloud SQL Postgres
(its lomb_test database), logged in as the given Google account.

Run from lomb_prompts/:  python tests/test_lesson_job.py [--cloudsql EMAIL]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import voice_enrollment.audio_decode as audio_decode  # noqa: E402
from pipeline.fake_provider import FakeProvider  # noqa: E402
from service.lesson_job import (  # noqa: E402
    LessonDeps, analyze_lesson, process_lesson, resume_with_speaker, submit_lesson,
)
from storage.blob_store import LocalBlobStore  # noqa: E402
from storage.storage import PHASE1_TAXONOMY, SQLiteStorageRepository  # noqa: E402
from voice_enrollment.voice_enrollment import SpeakerCandidate, SpeakerResolution  # noqa: E402

LEARNER = ["Ich wohne in die Stadt.", "Gestern ich gehe nach Hause.", "Das ist ein gut Idee, äh, glaube ich."]
TUTOR = ["Wo wohnst du jetzt?", "Und was hast du gestern gemacht?", "Das klingt gut."]


def synthetic_whisperx() -> dict:
    segments, t = [], 0.0
    for tutor_line, learner_line in zip(TUTOR, LEARNER):
        for spk, line in (("SPEAKER_01", tutor_line), ("SPEAKER_00", learner_line)):
            words = []
            for w in line.split():
                words.append({"word": w, "start": round(t, 2), "end": round(t + 0.4, 2), "score": 0.9, "speaker": spk})
                t += 0.5
            segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": " " + line,
                             "speaker": spk, "words": words})
            t += 0.8
    return {"segments": segments, "_meta": {"engine": "whisperx", "model": "large-v3",
                                             "diarization_model": "pyannote/speaker-diarization-community-1",
                                             "language": "de", "compute_type": "float16", "batch_size": 1,
                                             "engine_version": "test"}}


class StubTranscriber:
    def transcribe(self, audio_path, *, num_speakers=None):
        return synthetic_whisperx()


class FakeSpeakerService:
    match_threshold, margin_threshold = 0.55, 0.25

    def __init__(self, confident: bool):
        self.confident = confident

    def resolve(self, user_id, speakers):
        ranked = [SpeakerCandidate("SPEAKER_00", 0.71 if self.confident else 0.52), SpeakerCandidate("SPEAKER_01", 0.04)]
        if self.confident:
            return SpeakerResolution(False, "SPEAKER_00", ranked, "confident_match")
        return SpeakerResolution(True, None, ranked, "below_match_threshold")


class BrokenProvider(FakeProvider):
    def classify(self, config, input_data):
        raise RuntimeError("simulated OpenAI outage")


class ExportFailsBlobStore(LocalBlobStore):
    def put_bytes(self, data, key):
        if key.startswith("exports/"):
            raise OSError("simulated storage outage")
        return super().put_bytes(data, key)


def counts(repo, session_id: str) -> dict:
    q = {
        "sessions": "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
        "audio_assets": "SELECT COUNT(*) FROM audio_assets WHERE session_id = ?",
        "transcripts": "SELECT COUNT(*) FROM transcripts WHERE session_id = ?",
        "speaker_resolutions": "SELECT COUNT(*) FROM speaker_resolutions WHERE session_id = ?",
        "speaker_resolution_candidates": "SELECT COUNT(*) FROM speaker_resolution_candidates c JOIN "
                                         "speaker_resolutions r ON r.resolution_id = c.resolution_id WHERE r.session_id = ?",
        "analysis_runs": "SELECT COUNT(*) FROM analysis_runs WHERE session_id = ?",
        "information_items": "SELECT COUNT(*) FROM information_items WHERE session_id = ?",
        "correction_checks": "SELECT COUNT(*) FROM correction_checks WHERE session_id = ?",
        "metric_values": "SELECT COUNT(*) FROM metric_values WHERE session_id = ?",
        "report_snapshots": "SELECT COUNT(*) FROM report_snapshots WHERE session_id = ?",
        "report2_exports": "SELECT COUNT(*) FROM report2_exports WHERE session_id = ?",
    }
    return {k: next(iter(repo._one(sql, (session_id,)).values())) for k, sql in q.items()}


def make_repo(args, tmp: Path):
    if not args.cloudsql:
        return SQLiteStorageRepository(tmp / "lomb.sqlite")
    from storage.postgres_repo import PostgresStorageRepository, cloud_sql_connect
    credentials = None
    if args.token_env:  # a short-lived `gcloud auth print-access-token`, when ADC isn't set up
        import datetime
        import os
        from google.oauth2.credentials import Credentials

        class TokenOnly(Credentials):  # the connector refreshes before login; a bare token can't, and needn't
            def refresh(self, request):
                pass

        expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(minutes=50)
        credentials = TokenOnly(os.environ[args.token_env], expiry=expiry)
    return PostgresStorageRepository(cloud_sql_connect(args.cloudsql, "lomb_test", credentials=credentials))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudsql", metavar="EMAIL", help="Run against Cloud SQL (lomb_test) as this IAM user.")
    parser.add_argument("--token-env", help="Env var holding an access token to log in with (else ADC).")
    args = parser.parse_args()
    # Speaker-audio slicing needs real audio + ffmpeg; the fake service ignores it anyway.
    audio_decode.extract_speaker_audio = lambda path, turns, **kw: {
        t.speaker_id: np.zeros(16000 * 5, dtype=np.float32) for t in turns}
    tmp = Path(tempfile.mkdtemp())
    try:
        repo = make_repo(args, tmp)
        print(f"database: {type(repo).__name__}")
        blobs = LocalBlobStore(tmp / "blobs")
        audio = tmp / "lesson.m4a"
        audio.write_bytes(b"not really audio")
        n_metrics = len(PHASE1_TAXONOMY)

        def deps(confident=True, provider=None):
            return LessonDeps(repo=repo, blobs=blobs, transcriber=StubTranscriber(),
                              provider=provider or FakeProvider(error_every=1),
                              speaker_service=FakeSpeakerService(confident),
                              llm_model="FakeProvider")

        print("--- A. confident voiceprint match -> straight through ---")
        d = deps(confident=True)
        sid = submit_lesson(d, audio_path=audio, user_id="1")
        assert repo.get_session(sid)["status"] == "processing"
        out = process_lesson(d, session_id=sid, user_id="1")
        c = counts(repo, sid)
        print(f"  status={out.status}  {c}")
        assert out.status == "complete" and repo.get_session(sid)["status"] == "complete"
        assert c["speaker_resolutions"] == 1 and c["speaker_resolution_candidates"] == 2
        assert c["analysis_runs"] == 1 and c["report_snapshots"] == 1 and c["report2_exports"] == 1
        assert c["information_items"] > 0 and c["correction_checks"] > 0 and c["metric_values"] == n_metrics
        t = repo.latest_transcript(sid)
        assert (t["target_speaker_label"], t["speaker_resolution_method"]) == ("SPEAKER_00", "voiceprint_auto")
        assert repo.list_speaker_resolutions(sid)[0]["outcome"] == "auto_resolved"
        for uri in (repo.get_audio_asset(sid)["storage_uri"], t["raw_json_uri"]):
            assert blobs.exists(uri), f"stored file missing: {uri}"

        print("--- B. unsure match -> picker -> resume ---")
        d = deps(confident=False)
        sid_b = submit_lesson(d, audio_path=audio, user_id="1")
        out = process_lesson(d, session_id=sid_b, user_id="1")
        print(f"  status={out.status}  candidates={[(x['speaker_label'], x['similarity']) for x in out.candidates]}")
        assert out.status == "awaiting_speaker_selection" == repo.get_session(sid_b)["status"]
        assert out.candidates[0]["preview_text"].startswith("Ich wohne"), out.candidates[0]
        assert counts(repo, sid_b)["analysis_runs"] == 0, "nothing analysed before the user picks"
        out = resume_with_speaker(d, session_id=sid_b, speaker_label="SPEAKER_00")
        res = repo.list_speaker_resolutions(sid_b)[0]
        print(f"  after pick: status={out.status}  resolution={res['outcome']} -> {res['resolved_speaker_label']}")
        assert out.status == "complete" and res["outcome"] == "manual_override"
        assert repo.latest_transcript(sid_b)["speaker_resolution_method"] == "manual_picker"

        print("--- C. re-analysis -> second run on the same session ---")
        t = repo.latest_transcript(sid)
        from service.lesson_job import _turns
        import json as _json
        turns, _ = _turns(_json.loads(Path(t["raw_json_uri"]).read_text(encoding="utf-8")))
        analyze_lesson(deps(), session_id=sid, transcript_id=t["transcript_id"], turns=turns,
                       target_speaker="SPEAKER_00")
        c = counts(repo, sid)
        print(f"  {c['analysis_runs']} runs, {c['metric_values']} metric values, {c['report_snapshots']} reports")
        assert c["analysis_runs"] == 2 and c["metric_values"] == 2 * n_metrics and c["sessions"] == 1

        print("--- D. failure mid-job -> session 'failed' ---")
        d = deps(provider=BrokenProvider())
        sid_d = submit_lesson(d, audio_path=audio, user_id="1")
        try:
            process_lesson(d, session_id=sid_d, user_id="1")
            raise AssertionError("expected the simulated outage to raise")
        except RuntimeError as e:
            assert "simulated OpenAI outage" in str(e)
        print(f"  status={repo.get_session(sid_d)['status']}")
        assert repo.get_session(sid_d)["status"] == "failed"

        print("--- E. failure while saving -> nothing half-written ---")
        d = deps()
        d.blobs = ExportFailsBlobStore(tmp / "blobs")
        sid_e = submit_lesson(d, audio_path=audio, user_id="1")
        try:
            process_lesson(d, session_id=sid_e, user_id="1")
            raise AssertionError("expected the simulated storage outage to raise")
        except OSError as e:
            assert "simulated storage outage" in str(e)
        c = counts(repo, sid_e)
        print(f"  status={repo.get_session(sid_e)['status']}  {c}")
        assert repo.get_session(sid_e)["status"] == "failed"
        assert c["transcripts"] == 1 and c["speaker_resolutions"] == 1, "work before the analysis is kept"
        for table in ("analysis_runs", "information_items", "correction_checks", "metric_values", "report_snapshots"):
            assert c[table] == 0, f"{table} should have rolled back"

        print("\nAll lesson-job checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # SQLite handles may linger on Windows (D5)


if __name__ == "__main__":
    raise SystemExit(main())
