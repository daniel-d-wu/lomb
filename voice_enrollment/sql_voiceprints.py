"""
Voiceprints in the main database (2026-10-19, build step 6) -- the
voiceprints table that storage.py creates (schema v1.6), through the same
SQLStorageRepository connection, so it works on SQLite locally and Cloud SQL
in production. Replaces the separate voiceprints.db file for the cloud.

Beyond VoiceprintRepository's get/upsert: delete (consent withdrawn or
account deleted), touch (each time it's used to match a lesson), and
unused_since (the privacy policy's destroy-after-a-year-unused rule).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_enrollment.voice_enrollment import VoiceprintRepository  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLVoiceprintRepository(VoiceprintRepository):
    def __init__(self, repo):
        self.repo = repo  # a storage.storage.SQLStorageRepository

    def get(self, user_id: str) -> tuple[np.ndarray, int] | None:
        row = self.repo._one("SELECT embedding, sample_count FROM voiceprints WHERE user_id = ?", (user_id,))
        if row is None:
            return None
        return np.frombuffer(bytes(row["embedding"]), dtype=np.float32).copy(), int(row["sample_count"])

    def upsert(self, user_id, unit_direction_vector, sample_count, embedder_id, enrollment_source,
               reset=False) -> None:
        vec = np.asarray(unit_direction_vector, dtype=np.float32)
        now = _now()
        origin = ", created_at = excluded.created_at, enrollment_source = excluded.enrollment_source" if reset else ""
        self.repo._execute(
            f"""
            INSERT INTO voiceprints (user_id, embedding, embedding_dim, sample_count, embedder_id,
                                     enrollment_source, created_at, updated_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                embedding = excluded.embedding, embedding_dim = excluded.embedding_dim,
                sample_count = excluded.sample_count, embedder_id = excluded.embedder_id,
                updated_at = excluded.updated_at, last_used_at = excluded.last_used_at{origin}
            """,
            (user_id, vec.tobytes(), int(vec.shape[0]), int(sample_count), embedder_id, enrollment_source,
             now, now, now),
        )

    def exists(self, user_id: str) -> bool:
        return self.repo._one("SELECT 1 AS ok FROM voiceprints WHERE user_id = ?", (user_id,)) is not None

    def touch(self, user_id: str) -> None:
        self.repo._execute("UPDATE voiceprints SET last_used_at = ? WHERE user_id = ?", (_now(), user_id))

    def delete(self, user_id: str) -> None:
        self.repo._execute("DELETE FROM voiceprints WHERE user_id = ?", (user_id,))

    def unused_since(self, cutoff_iso: str) -> list[str]:
        return [r["user_id"] for r in self.repo._all("SELECT user_id FROM voiceprints WHERE last_used_at < ?",
                                                      (cutoff_iso,))]
