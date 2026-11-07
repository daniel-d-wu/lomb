"""
Offline test of the web API's rules (api/main.py, step 6): sign-in, the
invite list, consent, who can see which lesson, limits and deletion. No
network: SQLite stands in for Cloud SQL, a folder for Cloud Storage, a
dict of fake tokens for Firebase, and a list for the Cloud Tasks queue.

Run from lomb_prompts/:  python tests/test_api.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.main as api  # noqa: E402
from service.lesson_job import LessonDeps  # noqa: E402
from storage.blob_store import LocalBlobStore  # noqa: E402
from storage.storage import SQLiteStorageRepository  # noqa: E402
from voice_enrollment.sql_voiceprints import SQLVoiceprintRepository  # noqa: E402

TOKENS = {  # fake Firebase tokens -> claims
    "tok-dan": {"sub": "uid-dan", "email": "Dan@Example.com", "email_verified": True, "name": "Dan"},
    "tok-ana": {"sub": "uid-ana", "email": "ana@example.com", "email_verified": True, "name": "Ana"},
    "tok-eve": {"sub": "uid-eve", "email": "eve@example.com", "email_verified": True, "name": "Eve"},
}


class BucketStore(LocalBlobStore):
    """A folder that hands out gs://test-bucket/<key> URIs like GCSBlobStore."""
    bucket_name = "test-bucket"

    def _local(self, uri):
        return str(self.root / uri.removeprefix("gs://test-bucket/"))

    def put_file(self, local_path, key):
        super().put_file(local_path, key)
        return f"gs://test-bucket/{key}"

    def put_bytes(self, data, key):
        super().put_bytes(data, key)
        return f"gs://test-bucket/{key}"

    def fetch(self, uri, local_path):
        return super().fetch(self._local(uri), local_path)

    def exists(self, uri):
        return super().exists(self._local(uri))

    def read_bytes(self, uri):
        return super().read_bytes(self._local(uri))

    def move(self, uri, key):
        super().move(self._local(uri), key)
        return f"gs://test-bucket/{key}"

    def delete(self, uri):
        super().delete(self._local(uri))

    def browser_upload_url(self, key, *, content_type, size, origin):
        return f"fake-upload://{key}"


def verify(token):
    if token not in TOKENS:
        raise ValueError("bad token")
    return TOKENS[token]


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        repo = SQLiteStorageRepository(tmp / "lomb.sqlite")
        blobs = BucketStore(tmp / "blobs")
        queued = []
        api.state.update(deps=LessonDeps(repo=repo, blobs=blobs, transcriber=None, provider=None),
                         verify=verify, enqueue=lambda path, body: queued.append((path, body)))
        repo.add_invite("dan@example.com")
        repo.add_invite("ana@example.com")
        tone = tmp / "tone.wav"
        sf.write(tone, (0.1 * np.sin(np.arange(16000 * 3) / 10)).astype(np.float32), 16000)

        with TestClient(api.app) as c:
            def h(tok):
                return {"Authorization": f"Bearer {tok}"}

            def upload(tok):
                r = c.post("/api/uploads", headers=h(tok), json={"filename": "l.wav", "size_bytes": 1000})
                if r.status_code != 200:
                    return r
                key = r.json()["upload_url"].removeprefix("fake-upload://")
                blobs.put_file(tone, key)  # what the browser's PUT does
                return c.post("/api/sessions", headers=h(tok), json={"storage_uri": r.json()["storage_uri"]})

            print("--- sign-in and invites ---")
            assert c.get("/api/me").status_code == 401
            assert c.get("/api/me", headers=h("forged")).status_code == 401
            r = c.get("/api/me", headers=h("tok-eve"))
            assert r.status_code == 403 and r.json()["detail"]["code"] == "not_invited"
            me = c.get("/api/me", headers=h("tok-dan")).json()
            assert me["email"] == "dan@example.com" and not any(me["consents"].values())
            print("  no token / forged token -> 401, not invited -> 403, invited -> account created")

            print("--- consent ---")
            r = upload("tok-dan")
            assert r.status_code == 403 and r.json()["detail"]["code"] == "consent_required"
            for kind in ("terms", "recording_permission"):
                assert c.post("/api/consents", headers=h("tok-dan"), json={"consent_type": kind}).status_code == 200
            r = upload("tok-dan")
            assert r.status_code == 200, r.text
            sid = r.json()["session_id"]
            assert queued[-1] == ("/process", {"session_id": sid, "user_id": "uid-dan", "num_speakers": None})
            assert repo.get_audio_asset(sid)["storage_uri"].startswith(f"gs://test-bucket/audio/{sid}/")
            print("  uploads need terms + recording permission; then the lesson is queued for the worker")

            print("--- each user sees only their own lessons ---")
            for kind in ("terms", "recording_permission"):
                c.post("/api/consents", headers=h("tok-ana"), json={"consent_type": kind})
            assert c.get(f"/api/sessions/{sid}", headers=h("tok-ana")).status_code == 404
            assert c.delete(f"/api/sessions/{sid}", headers=h("tok-ana")).status_code == 404
            assert c.get(f"/api/sessions/{sid}/report", headers=h("tok-ana")).status_code == 404
            assert c.get("/api/sessions", headers=h("tok-ana")).json()["sessions"] == []
            assert c.get(f"/api/sessions/{sid}", headers=h("tok-dan")).json()["stage"] == "transcribing"
            print("  another user gets 404 for Dan's lesson, its report, and deleting it")

            print("--- limits ---")
            assert upload("tok-dan").status_code == 200 and upload("tok-dan").status_code == 200
            r = upload("tok-dan")
            assert r.status_code == 429, r.text
            cap = api.MONTHLY_CAP_USD
            api.MONTHLY_CAP_USD = 0.5
            r = upload("tok-ana")
            assert r.status_code == 503, r.text
            api.MONTHLY_CAP_USD = cap
            print(f"  4th lesson in 24 h -> 429; monthly budget reached -> 503 (3 lessons ~ $0.75 estimated)")

            print("--- biometric consent and voiceprint ---")
            r = c.post("/api/consents", headers=h("tok-dan"), json={"consent_type": "biometric_enrollment"})
            assert r.status_code == 400, "an e-signature is required"
            r = c.post("/api/consents", headers=h("tok-dan"),
                       json={"consent_type": "biometric_enrollment", "signature": "Dan Wu", "session_id": sid})
            assert r.status_code == 200 and queued[-1] == ("/enroll", {"session_id": sid, "user_id": "uid-dan"})
            SQLVoiceprintRepository(repo).upsert("uid-dan", np.ones(4, np.float32) / 2, 1, "test", "session")
            assert c.get("/api/me", headers=h("tok-dan")).json()["has_voiceprint"]
            assert c.delete("/api/consents/biometric_enrollment", headers=h("tok-dan")).status_code == 200
            me = c.get("/api/me", headers=h("tok-dan")).json()
            assert not me["has_voiceprint"] and not me["consents"]["biometric_enrollment"]
            print("  signature required; enrollment queued; withdrawing deletes the voiceprint")

            print("--- deletion ---")
            assert c.delete(f"/api/sessions/{sid}", headers=h("tok-dan")).status_code == 409, "still processing"
            repo.update_session(sid, status="complete")
            audio_uri = repo.get_audio_asset(sid)["storage_uri"]
            assert c.delete(f"/api/sessions/{sid}", headers=h("tok-dan")).status_code == 200
            assert repo.get_session(sid) is None and not blobs.exists(audio_uri)
            assert c.delete("/api/me", headers=h("tok-dan")).status_code == 200
            assert repo.get_user("uid-dan") is None and repo.list_user_sessions("uid-dan") == []
            print("  lesson deleted with its audio; account deleted with everything in it")

            print("--- pages and the scheduler endpoint ---")
            assert "2026-10-19" in c.get("/privacy").text and "{{" not in c.get("/privacy").text
            assert c.get("/api/config").json()["policy_version"] == api.POLICY_VERSION
            assert c.post("/internal/cleanup").status_code == 403
            assert c.post("/internal/cleanup", headers=h("tok-dan")).status_code == 403
            print("  privacy page filled in; cleanup refuses anyone but the scheduler")

        print("\nAll API checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
