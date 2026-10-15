"""
One real lesson through the deployed Google Cloud stack, driven from this
machine (2026-10-14, build steps 3-4) -- until the web API exists (step 5),
this script plays its part:

  1. upload: the audio to Cloud Storage + sessions/audio_assets rows in
     Cloud SQL (the same submit_lesson() the API will call)
  2. POST /process to the GPU worker -> transcript + "which one is you?"
  3. pick the learner (--speaker, else ask on the terminal)
  4. POST /resume -> analysis, reports, 'complete'
  5. print the rows written and the timings

Logs in as the gcloud account (`gcloud auth print-access-token` /
`print-identity-token`), so no application-default credentials are needed.

Usage (from lomb_prompts/):
  python scripts/cloud_smoke_test.py --audio PATH --worker-url URL
      [--speaker SPEAKER_00] [--num-speakers 2] [--db lomb]
  python scripts/cloud_smoke_test.py --session ID --speaker LABEL --worker-url URL
      (finish a session left at the picker -- e.g. when run non-interactively)
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.oauth2.credentials import Credentials  # noqa: E402

from service.lesson_job import LessonDeps, submit_lesson  # noqa: E402
from storage.blob_store import GCSBlobStore  # noqa: E402
from storage.postgres_repo import PostgresStorageRepository, cloud_sql_connect  # noqa: E402

PROJECT, BUCKET = "lomb-prod", "lomb-prod-data"
TABLES = ["sessions", "audio_assets", "transcripts", "speaker_resolutions", "analysis_runs", "information_items",
          "correction_checks", "metric_values", "report_snapshots", "report2_exports"]


def gcloud(*args: str) -> str:
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=True, shell=sys.platform == "win32"
                          ).stdout.strip()


class TokenOnly(Credentials):  # gcloud's token is short-lived and this run is short; nothing to refresh
    def refresh(self, request):
        pass


def call_worker(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {gcloud('auth', 'print-identity-token')}"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"worker {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio")
    ap.add_argument("--session", help="Resume a session waiting at the picker (with --speaker).")
    ap.add_argument("--worker-url", required=True)
    ap.add_argument("--speaker", help="The learner's label; asked on the terminal if omitted.")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--db", default="lomb")
    args = ap.parse_args()

    expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(minutes=50)
    creds = TokenOnly(gcloud("auth", "print-access-token"), expiry=expiry)
    account = gcloud("config", "get", "account")
    from google.cloud import storage
    deps = LessonDeps(repo=PostgresStorageRepository(cloud_sql_connect(account, args.db, credentials=creds)),
                      blobs=GCSBlobStore(BUCKET, client=storage.Client(project=PROJECT, credentials=creds)),
                      transcriber=None, provider=None)

    t0 = time.time()
    if args.session:
        session_id, speaker = args.session, args.speaker
    else:
        session_id = submit_lesson(deps, audio_path=args.audio, user_id=None, source="cloud_smoke_test")
        print(f"1. uploaded  session={session_id}  ({time.time() - t0:.0f}s)")

        t1 = time.time()
        out = call_worker(args.worker_url, "/process", {"session_id": session_id, "num_speakers": args.num_speakers})
        print(f"2. /process  status={out['status']}  ({time.time() - t1:.0f}s, includes any cold start)")
        if out["status"] != "awaiting_speaker_selection":
            sys.exit(f"unexpected: {out}")
        for c in out["candidates"]:
            print(f"     {c['speaker_label']}: {c['preview_text'][:160]}")
        if not args.speaker and not sys.stdin.isatty():
            print(f"3. no --speaker: rerun with --session {session_id} --speaker <label>")
            return 2
        speaker = args.speaker or input("3. which one is the learner? ").strip()

    t2 = time.time()
    out = call_worker(args.worker_url, "/resume", {"session_id": session_id, "speaker_label": speaker})
    print(f"4. /resume   status={out['status']}  run={out['run_id']}  ({time.time() - t2:.0f}s)")

    print(f"5. rows for {session_id}:")
    for table in TABLES:
        n = deps.repo._one(f"SELECT COUNT(*) AS n FROM {table} WHERE session_id = ?", (session_id,))["n"]
        print(f"     {table:<20} {n}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
