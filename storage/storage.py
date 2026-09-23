"""
Storage layer for the 6 tables a Phase 1 end-to-end run actually needs to
write real rows into: sessions, audio_assets, transcripts,
information_items, metric_values, report2_exports. Plus the two lookup
tables information_items/metric_values are FK-bound to (linguistic_features,
metric_definitions) -- seeded here from a placeholder taxonomy, not hand-
entered per session; see PHASE1_TAXONOMY below for exactly what "placeholder"
means and why.

STORAGE SEAM, same pattern voice_enrollment/voice_enrollment.py already
uses for VoiceprintRepository -- this is deliberately not a new pattern:
StorageRepository (ABC) is what pipeline-orchestration code is written
against; SQLiteStorageRepository is the shipped default (zero extra
infra, fine at this scale). Swapping in a Postgres implementation later
means writing one new class against the same abstract interface -- no
caller of this module needs to change. That's the actual "can easily
migrate to a cloud/production env" this module was asked to satisfy.

SCOPE, stated explicitly (per Dan's own framing -- "code to produce the
crucial data entries within the data tables... 6 or 7?"): this covers
the 6 tables a single analysis run touches to prove the whole pipeline
end-to-end. It deliberately does NOT cover identity (users,
consent_events), voice-enrollment matching (voiceprints,
speaker_resolutions, speaker_resolution_candidates, diarization_candidates),
or Report 1 (report_snapshots) -- all explicitly out of scope for this
particular test. Full DDL for those lives in claude/lomb_data_schema_v1.md
already; this file only creates what it actually writes to, so an empty
sessions.db doesn't silently imply tables nothing here ever populates.

WHAT'S A REAL DECISION HERE VS. A PLACEHOLDER, stated plainly rather than
glossed over (this project's established documentation style -- see
direct_computation.py, speaker_filter.py for the same practice):

  REAL, taken directly from claude/lomb_data_schema_v1.md's already-drafted
  DDL: every column definition and FK relationship below.

  PLACEHOLDER, because the real design doesn't exist yet (confirmed by
  reading claude/lomb_metric_architecture_v1.md Section 5, point 2: "Per-
  feature metric sets above are my judgment calls, not yet yours" --
  i.e. the actual metric_aggregator.py this project describes, with real
  per-metric formulas, is explicitly undesigned):
    - PHASE1_TAXONOMY's metric_definitions rows (unit/formula columns) --
      a minimal stand-in (raw session-level count, or a pass-through of
      whatever pipeline.py already computes as a convenience value: wpm,
      structure_breadth_score), NOT the real Metric Aggregator's output.
      Good enough to prove metric_values gets real, correctly-shaped
      rows; not a claim that these are the RIGHT numbers to report on.
    - information_items population rule: only sentences/windows/
      occurrences that were actually FLAGGED get a row (an LLM sentence
      metric's output.error is True; STRUCTURE_BREADTH's output.structures
      has a non-"none" label; every direct-computation metric's entries,
      since those are already occurrence-only by construction -- see
      direct_computation.py's own docstring). This reading comes from
      lomb_metric_architecture_v1.md's own description of what this
      table is FOR ("errors flagged in this session, shown with
      transcript context") -- not from an explicit population-rule
      spec, because none exists yet either.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    user_id          TEXT,
    source           TEXT NOT NULL,
    language_pair    TEXT NOT NULL DEFAULT 'de-en',
    duration_seconds FLOAT,
    word_count       INTEGER,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_assets (
    audio_id             TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL REFERENCES sessions(session_id),
    storage_uri          TEXT NOT NULL,
    format               TEXT NOT NULL,
    size_bytes           INTEGER,
    duration_seconds     FLOAT NOT NULL,
    sha256               TEXT,
    uploaded_at          TEXT NOT NULL,
    speaker_count_hint   INTEGER,
    retention_class      TEXT NOT NULL DEFAULT 'transient',
    consent_for_training BOOLEAN NOT NULL DEFAULT 0,
    expires_at           TEXT,
    deleted_at           TEXT
);

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id             TEXT PRIMARY KEY,
    session_id                TEXT NOT NULL REFERENCES sessions(session_id),
    audio_id                  TEXT REFERENCES audio_assets(audio_id),
    engine                    TEXT NOT NULL,
    raw_json_uri              TEXT NOT NULL,
    target_speaker_label      TEXT,
    speaker_resolution_method TEXT,
    created_at                TEXT NOT NULL,
    retention_class           TEXT NOT NULL DEFAULT 'transient',
    consent_for_training      BOOLEAN NOT NULL DEFAULT 0,
    expires_at                TEXT,
    deleted_at                TEXT
);

CREATE TABLE IF NOT EXISTS constructs (
    construct_key    TEXT PRIMARY KEY,
    display_name     TEXT
);

CREATE TABLE IF NOT EXISTS linguistic_features (
    feature_key       TEXT PRIMARY KEY,
    construct_key     TEXT REFERENCES constructs(construct_key),
    display_name      TEXT,
    computation_path  TEXT,
    description       TEXT
);

CREATE TABLE IF NOT EXISTS metric_definitions (
    metric_key    TEXT PRIMARY KEY,
    feature_key   TEXT REFERENCES linguistic_features(feature_key),
    unit          TEXT,
    formula       TEXT
);

CREATE TABLE IF NOT EXISTS information_items (
    item_id        TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id),
    transcript_id  TEXT REFERENCES transcripts(transcript_id),
    feature_key    TEXT NOT NULL REFERENCES linguistic_features(feature_key),
    utterance_ref  TEXT,
    input_ref      TEXT NOT NULL,
    output_json    TEXT NOT NULL,
    review_status  TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_information_items_session_feature ON information_items(session_id, feature_key);

CREATE TABLE IF NOT EXISTS metric_values (
    value_id      TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    transcript_id TEXT REFERENCES transcripts(transcript_id),
    metric_key    TEXT NOT NULL REFERENCES metric_definitions(metric_key),
    value         FLOAT NOT NULL,
    computed_at   TEXT NOT NULL,
    UNIQUE(session_id, metric_key)
);

CREATE TABLE IF NOT EXISTS report2_exports (
    export_id      TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id),
    storage_uri    TEXT NOT NULL,
    format         TEXT NOT NULL DEFAULT 'xlsx',
    row_count      INTEGER,
    generated_at   TEXT NOT NULL
);
"""

# Placeholder taxonomy -- see module docstring's PLACEHOLDER section.
# feature_key values match pipeline.py's own metric_key strings exactly
# (SENTENCE_METRICS + WINDOWED_SENTENCE_METRICS + DIRECT_METRICS), so a
# pipeline.py result's "results" dict keys map onto linguistic_features
# rows with no translation table needed. metric_key values are a plain,
# undesigned stand-in (mostly "<feature>_count", lowercased) -- NOT
# claude/lomb_metric_architecture_v1.md's eventual real metric_key set
# ('gvt1_error_rate', etc.), which needs a real Metric Aggregator this
# project hasn't built yet (see module docstring).
PHASE1_TAXONOMY = [
    # (feature_key, construct_key, computation_path, metric_key, unit, formula)
    ("GDD-1", "ACCURACY", "llm", "gdd1_count", "count", "count of flagged sentences this session (placeholder -- not an error rate)"),
    ("GDD-2", "ACCURACY", "llm", "gdd2_count", "count", "count of flagged sentences this session (placeholder -- not an error rate)"),
    ("GVT-1", "ACCURACY", "llm", "gvt1_count", "count", "count of flagged sentence-windows this session (placeholder -- not an error rate)"),
    ("GVT-2", "ACCURACY", "llm", "gvt2_count", "count", "count of flagged sentences this session (placeholder -- not an error rate)"),
    ("LPF", "ACCURACY", "llm", "lpf_count", "count", "count of flagged sentences this session (placeholder -- not an error rate)"),
    ("LP", "ACCURACY", "llm", "lp_count", "count", "count of flagged sentences this session (placeholder -- not an error rate)"),
    ("STRUCTURE_BREADTH", "COMPLEXITY", "llm", "structure_breadth_score", "score", "distinct non-'none' structure labels seen this session -- pipeline.py's own convenience value, pass-through"),
    ("FORMULAIC", "COMPLEXITY", "direct", "formulaic_count", "count", "count of BUNDLES regex matches this session"),
    ("FILLED_PAUSE", "FLUENCY", "direct", "filled_pause_count", "count", "count of filler-token occurrences this session"),
    ("UNFILLED_PAUSE", "FLUENCY", "direct", "unfilled_pause_count", "count", "count of over-threshold gap occurrences this session"),
    ("WPM", "FLUENCY", "direct", "wpm", "wpm", "word_count / (duration_seconds / 60) -- pipeline.py's own convenience value, pass-through"),
]

_CONSTRUCTS = [
    ("FLUENCY", "Fluency"),
    ("ACCURACY", "Accuracy"),
    ("COMPLEXITY", "Complexity"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StorageRepository(ABC):
    """Storage seam. Swap SQLiteStorageRepository for a Postgres
    implementation without touching any orchestration code written
    against this interface -- see module docstring."""

    @abstractmethod
    def create_session(self, session_id: str, source: str, *, user_id: str | None = None,
                        language_pair: str = "de-en", status: str = "processing") -> None: ...

    @abstractmethod
    def update_session(self, session_id: str, **fields) -> None: ...

    @abstractmethod
    def get_session(self, session_id: str) -> dict | None: ...

    @abstractmethod
    def create_audio_asset(self, audio_id: str, session_id: str, storage_uri: str, format: str,
                            duration_seconds: float, *, size_bytes: int | None = None,
                            sha256: str | None = None, speaker_count_hint: int | None = None) -> None: ...

    @abstractmethod
    def create_transcript(self, transcript_id: str, session_id: str, engine: str, raw_json_uri: str,
                           *, audio_id: str | None = None, target_speaker_label: str | None = None,
                           speaker_resolution_method: str | None = None) -> None: ...

    @abstractmethod
    def ensure_taxonomy_seeded(self) -> None: ...

    @abstractmethod
    def write_information_item(self, item_id: str, session_id: str, feature_key: str, input_ref: str,
                                output: dict, *, transcript_id: str | None = None,
                                utterance_ref: str | None = None, review_status: str = "unreviewed") -> None: ...

    @abstractmethod
    def write_metric_value(self, value_id: str, session_id: str, metric_key: str, value: float,
                            *, transcript_id: str | None = None) -> None: ...

    @abstractmethod
    def create_report2_export(self, export_id: str, session_id: str, storage_uri: str,
                               *, format: str = "xlsx", row_count: int | None = None) -> None: ...

    @abstractmethod
    def list_information_items(self, session_id: str) -> list[dict]: ...

    @abstractmethod
    def list_metric_values(self, session_id: str) -> list[dict]: ...


class SQLiteStorageRepository(StorageRepository):
    """Default, zero-extra-infra implementation -- one file, created (and
    migrated forward via CREATE TABLE IF NOT EXISTS) on first connect,
    same as SQLiteVoiceprintRepository does for voiceprints.db. Foreign
    keys are enforced (PRAGMA foreign_keys = ON on every connection) --
    SQLite doesn't do this by default, and a repo meant to prove real
    referential shape shouldn't silently allow rows that violate it."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # --- sessions -----------------------------------------------------

    def create_session(self, session_id, source, *, user_id=None, language_pair="de-en", status="processing") -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, source, language_pair, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    source = excluded.source, language_pair = excluded.language_pair,
                    status = excluded.status, updated_at = excluded.updated_at
                """,
                (session_id, user_id, source, language_pair, status, now, now),
            )

    def update_session(self, session_id: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
                (*fields.values(), session_id),
            )

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    # --- audio_assets ---------------------------------------------------

    def create_audio_asset(self, audio_id, session_id, storage_uri, format, duration_seconds, *,
                            size_bytes=None, sha256=None, speaker_count_hint=None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audio_assets
                    (audio_id, session_id, storage_uri, format, size_bytes, duration_seconds,
                     sha256, uploaded_at, speaker_count_hint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audio_id) DO UPDATE SET
                    storage_uri = excluded.storage_uri, duration_seconds = excluded.duration_seconds
                """,
                (audio_id, session_id, storage_uri, format, size_bytes, duration_seconds,
                 sha256, _now(), speaker_count_hint),
            )

    # --- transcripts ------------------------------------------------------

    def create_transcript(self, transcript_id, session_id, engine, raw_json_uri, *,
                           audio_id=None, target_speaker_label=None, speaker_resolution_method=None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcripts
                    (transcript_id, session_id, audio_id, engine, raw_json_uri,
                     target_speaker_label, speaker_resolution_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transcript_id) DO UPDATE SET
                    target_speaker_label = excluded.target_speaker_label
                """,
                (transcript_id, session_id, audio_id, engine, raw_json_uri,
                 target_speaker_label, speaker_resolution_method, _now()),
            )

    # --- taxonomy (lookup tables) ------------------------------------------

    def ensure_taxonomy_seeded(self) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO constructs (construct_key, display_name) VALUES (?, ?)",
                _CONSTRUCTS,
            )
            for feature_key, construct_key, computation_path, metric_key, unit, formula in PHASE1_TAXONOMY:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO linguistic_features
                        (feature_key, construct_key, display_name, computation_path, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (feature_key, construct_key, feature_key, computation_path,
                     "Phase 1 placeholder taxonomy -- see storage.py module docstring."),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO metric_definitions (metric_key, feature_key, unit, formula)
                    VALUES (?, ?, ?, ?)
                    """,
                    (metric_key, feature_key, unit, formula),
                )

    # --- information_items --------------------------------------------------

    def write_information_item(self, item_id, session_id, feature_key, input_ref, output, *,
                                transcript_id=None, utterance_ref=None, review_status="unreviewed") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO information_items
                    (item_id, session_id, transcript_id, feature_key, utterance_ref,
                     input_ref, output_json, review_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO NOTHING
                """,
                (item_id, session_id, transcript_id, feature_key, utterance_ref,
                 input_ref, json.dumps(output, ensure_ascii=False), review_status, _now()),
            )

    def list_information_items(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM information_items WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- metric_values -----------------------------------------------------

    def write_metric_value(self, value_id, session_id, metric_key, value, *, transcript_id=None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metric_values (value_id, session_id, transcript_id, metric_key, value, computed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, metric_key) DO UPDATE SET
                    value = excluded.value, computed_at = excluded.computed_at, transcript_id = excluded.transcript_id
                """,
                (value_id, session_id, transcript_id, metric_key, value, _now()),
            )

    def list_metric_values(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM metric_values WHERE session_id = ? ORDER BY metric_key", (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- report2_exports -----------------------------------------------------

    def create_report2_export(self, export_id, session_id, storage_uri, *, format="xlsx", row_count=None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO report2_exports (export_id, session_id, storage_uri, format, row_count, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(export_id) DO UPDATE SET
                    storage_uri = excluded.storage_uri, row_count = excluded.row_count, generated_at = excluded.generated_at
                """,
                (export_id, session_id, storage_uri, format, row_count, _now()),
            )


def is_flagged(feature_key: str, entry: dict) -> bool:
    """The information_items population rule -- see module docstring's
    PLACEHOLDER section for why this is a judgment call, not a spec.
    entry is one item from a pipeline.py result["results"][feature_key]
    list (the {"input","skipped","output"} shape every metric shares).
    """
    if entry.get("skipped"):
        return False
    output = entry.get("output") or {}
    if feature_key == "STRUCTURE_BREADTH":
        return any(label != "none" for label in output.get("structures", []))
    if feature_key in ("FORMULAIC", "FILLED_PAUSE", "UNFILLED_PAUSE", "WPM"):
        # Direct-computation metrics only ever produce occurrence rows in
        # the first place (see direct_computation.py's own docstring) --
        # every entry IS a flagged instance, WPM's one session-level
        # Information entry included.
        return True
    return bool(output.get("error"))


def persist_pipeline_result(
    repo: StorageRepository,
    session_id: str,
    transcript_id: str,
    result: dict,
    id_factory,
) -> dict:
    """Writes one run_pipeline_from_turns()/run_pipeline() result into
    information_items + metric_values for one session -- the actual
    'take a pipeline result and produce real rows in the data tables'
    step, kept in storage.py (not the orchestration script) so this
    mapping is the same regardless of which engine's adapter produced
    the turns that fed the pipeline, or which script calls this.

    id_factory: a zero-arg callable returning a fresh unique id string
    (e.g. lambda: str(uuid.uuid4())) -- not hardcoded to uuid here so a
    caller with its own id scheme (or a deterministic one for tests) can
    swap it in.

    Returns {"information_items_written": int, "metric_values_written": int}
    for the caller to print/verify, not swallowed silently.
    """
    repo.ensure_taxonomy_seeded()

    items_written = 0
    for feature_key, entries in result["results"].items():
        for entry in entries:
            if not is_flagged(feature_key, entry):
                continue
            output = entry.get("output") or {}
            sentence_indices = output.get("sentence_indices")
            utterance_ref = (
                str(sentence_indices[0]) if sentence_indices else None
            )
            repo.write_information_item(
                item_id=id_factory(),
                session_id=session_id,
                feature_key=feature_key,
                input_ref=entry["input"],
                output=output,
                transcript_id=transcript_id,
                utterance_ref=utterance_ref,
            )
            items_written += 1

    metric_values_written = 0
    metric_key_by_feature = {row[0]: row[3] for row in PHASE1_TAXONOMY}
    for feature_key, entries in result["results"].items():
        metric_key = metric_key_by_feature[feature_key]
        if feature_key == "WPM":
            value = result["wpm"]
            if value is None:
                continue
        elif feature_key == "STRUCTURE_BREADTH":
            value = result["structure_breadth_score"]
        else:
            value = float(sum(1 for e in entries if is_flagged(feature_key, e)))
        repo.write_metric_value(
            value_id=id_factory(),
            session_id=session_id,
            metric_key=metric_key,
            value=value,
            transcript_id=transcript_id,
        )
        metric_values_written += 1

    return {"information_items_written": items_written, "metric_values_written": metric_values_written}


if __name__ == "__main__":
    import tempfile
    import uuid

    # Self-test against a small synthetic pipeline-result-shaped dict --
    # proves the write path and the flagged/not-flagged split work, and
    # that FK integrity actually holds (PRAGMA foreign_keys = ON), without
    # needing a real transcript or a live/fake LLM provider.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_storage.sqlite"
        repo = SQLiteStorageRepository(db_path)

        session_id = "sess-test-1"
        repo.create_session(session_id, source="personal_pipeline")
        repo.create_audio_asset("audio-test-1", session_id, "file:///tmp/fake.wav", "wav", 12.3)
        repo.create_transcript("transcript-test-1", session_id, "whisperx", "file:///tmp/fake.json",
                                audio_id="audio-test-1", target_speaker_label="SPEAKER_00")

        fake_result = {
            "wpm": 142.5,
            "structure_breadth_score": 2,
            "results": {
                "GDD-1": [
                    {"input": "s1", "skipped": False, "output": {"error": True, "sentence_indices": [0]}},
                    {"input": "s2", "skipped": False, "output": {"error": False, "sentence_indices": [1]}},
                    {"input": "s3", "skipped": True, "output": None},
                ],
                "STRUCTURE_BREADTH": [
                    {"input": "s1", "skipped": False, "output": {"structures": ["dass_clause"], "sentence_indices": [0]}},
                    {"input": "s2", "skipped": False, "output": {"structures": ["none"], "sentence_indices": [1]}},
                ],
                "WPM": [
                    {"input": "session", "skipped": False, "output": {"word_count": 100, "duration_seconds": 42.1, "sentence_indices": []}},
                ],
                "FORMULAIC": [],
                "FILLED_PAUSE": [],
                "UNFILLED_PAUSE": [],
                "GDD-2": [{"input": "s1", "skipped": False, "output": {"error": False, "sentence_indices": [0]}}],
                "GVT-1": [{"input": "w1", "skipped": False, "output": {"error": False, "sentence_indices": [0, 1]}}],
                "GVT-2": [{"input": "s1", "skipped": False, "output": {"error": False, "sentence_indices": [0]}}],
                "LPF": [{"input": "s1", "skipped": False, "output": {"error": False, "sentence_indices": [0]}}],
                "LP": [{"input": "s1", "skipped": False, "output": {"error": False, "sentence_indices": [0]}}],
            },
        }

        summary = persist_pipeline_result(repo, session_id, "transcript-test-1", fake_result,
                                           id_factory=lambda: str(uuid.uuid4()))
        print(f"persist_pipeline_result() -> {summary}")

        items = repo.list_information_items(session_id)
        values = repo.list_metric_values(session_id)
        print(f"\ninformation_items ({len(items)}):")
        for it in items:
            print(f"  {it['feature_key']:<20} {it['input_ref']!r}")
        print(f"\nmetric_values ({len(values)}):")
        for v in values:
            print(f"  {v['metric_key']:<25} {v['value']}")

        # Only the actually-flagged GDD-1 sentence and the actually-non-
        # "none" STRUCTURE_BREADTH sentence should have produced an
        # information_items row -- proves is_flagged() actually filtered,
        # not just that every entry got written.
        assert summary["information_items_written"] == 3, (
            f"expected exactly 3 flagged items (1 GDD-1 error, 1 non-none STRUCTURE_BREADTH, "
            f"1 WPM occurrence), got {summary['information_items_written']}"
        )
        assert summary["metric_values_written"] == len(PHASE1_TAXONOMY), (
            "expected exactly one metric_values row per taxonomy feature"
        )
        gdd1_items = [it for it in items if it["feature_key"] == "GDD-1"]
        assert len(gdd1_items) == 1 and gdd1_items[0]["input_ref"] == "s1", (
            "only the flagged GDD-1 sentence ('s1', error=True) should have a row -- "
            "'s2' (error=False) and 's3' (skipped) must not"
        )

        repo.create_report2_export("export-test-1", session_id, "file:///tmp/fake.xlsx", row_count=len(items))

        session_row = repo.get_session(session_id)
        print(f"\nsessions row: {dict(session_row)}")

        # FK integrity actually enforced -- a bogus session_id must fail,
        # not silently insert an orphaned row.
        try:
            repo.write_metric_value("bad-1", "no-such-session", "wpm", 1.0)
            raise AssertionError("expected a foreign key violation, insert succeeded instead")
        except sqlite3.IntegrityError as e:
            print(f"\nForeign key enforcement confirmed: {e}")

        print("\nAll storage.py self-checks passed.")
