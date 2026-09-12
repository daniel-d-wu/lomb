"""
voice_enrollment.py -- production speaker-enrollment/resolution service for
Lomb's backend.

WHERE THIS FITS
  lomb_backend_prd_v1.md Section 6.2 ("Speaker resolution") already commits
  to a manual flow for the public, stateless demo: diarize, show the
  visitor each detected speaker's clip, let them click which one is them,
  every single upload. That's correct for a first-time or anonymous
  visitor -- there's nothing to match against yet.

  lomb_prd_v1.md Section 9's first open question is the one this module
  answers: "Production ingestion... Speaker enrollment via voiceprint
  (pyannote-audio flagged as viable) is the likely path for scalable
  speaker ID." This module is that path, built so it SLOTS UNDERNEATH the
  existing manual picker rather than replacing it:

    - First upload ever for a user_id, with NO dedicated enrollment step:
      no voiceprint on file -> always falls through to the existing
      manual picker (Section 6.2, unchanged). Whatever the visitor picks
      gets enrolled via enroll().
    - OR, a dedicated active-enrollment step at onboarding, before any
      real upload: the visitor records a short clean sample directly
      (read a passage, or just talk for a few seconds) and enroll_active()
      seeds the voiceprint from that instead. This can make even the
      visitor's FIRST real upload skip the picker, if the enrollment
      sample matches confidently enough -- session-derived enrollment
      can't do that, since there's nothing to compare against until
      after the first picker click. Both paths write into the exact
      same store and the exact same resolve() logic below doesn't know
      or care which one produced the voiceprint it's matching against.
    - Every later upload, regardless of which enrollment path started it:
      diarize as normal, embed each detected speaker,
      compare against the user's stored voiceprint. High-confidence match
      -> skip the picker entirely (this is the actual product payoff --
      a returning learner never re-identifies themselves). Low-confidence
      match -> fall through to the SAME manual picker, never a silent
      guess, and use whatever the visitor confirms to reinforce the
      voiceprint. A wrong auto-match silently swaps "the learner's errors"
      for "the tutor's errors" for a whole session with nothing downstream
      able to catch it -- confirmed the hard way on Dan's own personal
      corpus (natasja_italki6_4_14_2026, see identify_target_speaker.py's
      docstring) -- so the threshold policy below is deliberately
      conservative and every auto-resolution is logged with the evidence
      behind it.

  HONEST DEPENDENCY THIS MODULE INTRODUCES: a stored voiceprint has to be
  keyed to *something* that persists across uploads. lomb_prd_v1.md
  Section 8 marks persistent multi-tenant accounts as unscoped ("Architecture
  (Unscoped)" -- backend, data store, and frontend are all open questions).
  This module doesn't decide that for you; VoiceprintRepository is an
  interface so user_id can be a real account id, an email-gate token
  (lomb_backend_prd_v1.md Section 10's beta email gate already assigns
  something identity-shaped), or a browser-local anonymous id -- whatever
  identity concept the rest of the backend lands on, this plugs into it.
  The one thing that's NOT optional: some notion of "the same visitor
  came back" has to exist before enrollment can pay off at all.

STORAGE
  SQLiteVoiceprintRepository below is the shipped default -- zero
  additional infra, works on a single Fly.io/Cloud Run instance
  (lomb_build_roadmap.md 2.7's already-chosen deploy targets), good enough
  for the beta learner counts lomb_build_roadmap.md Stage 3.3 describes
  (3-5 beta learners). VoiceprintRepository is the seam to swap in
  Postgres+pgvector (or any vector store) later without touching
  SpeakerResolutionService -- do that when/if learner count or
  multi-instance deployment makes SQLite's single-file-lock model a real
  bottleneck, not before.

  Embeddings are stored as an incremental spherical mean (running sum of
  L2-normalized vectors + a sample count), not just "the latest
  embedding" or a plain arithmetic mean of raw vectors -- this means (a)
  one noisy sample early on doesn't permanently define the voiceprint,
  (b) the representation is stable to accumulate over many sessions, and
  (c) it costs O(1) storage and O(1) update per enrollment, not "keep
  every embedding and average on read."

THRESHOLD POLICY
  Two independent gates must both pass before an auto-resolution is
  trusted:
    - MATCH_THRESHOLD: absolute cosine similarity to the stored
      voiceprint must be high enough that this really sounds like the
      same person, not just "the closer of two options."
    - MARGIN_THRESHOLD: the gap between the best and second-best
      candidate must be large enough that the call isn't a coin flip
      (this is what would have caught natasja_italki6-style errors --
      an absolute-similarity-only check can still be confidently wrong
      when two candidates are both close).
  Miss either gate -> needs_manual_confirmation=True. Never silently pick
  the best of two bad options.

  WHERE THE NUMBERS COME FROM (read this before changing them): they are
  a deliberately conservative placeholder, not a calibrated result --
  checked against the actual literature and it does not converge on one
  number. SpeechBrain's own SpeakerRecognition.verify_batch() ships a
  default decision threshold of 0.25 on this exact raw-cosine score
  (per a maintainer's comment in speechbrain/speechbrain#2148), which is
  the balanced operating point (~EER) on VoxCeleb1-test-cleaned -- curated,
  largely-English, single-utterance studio recordings, not German-language
  tutoring calls over videocall compression with real conversational
  turn-taking. A separate community thread on the model's own HuggingFace
  page (discussion #7) cites 0.7-0.75 as typical, but "scaled to [0,1]",
  which may or may not be the same convention this file's raw cosine
  score uses -- the two sources don't reconcile cleanly, and neither is a
  clean stand-in for this domain.

  Rather than block shipping on a proper calibration (that needs a labeled
  batch of real uploads, which doesn't exist yet), MATCH_THRESHOLD and
  MARGIN_THRESHOLD below are set well above SpeechBrain's own balanced
  default -- deliberately trading recall for precision, because the two
  failure directions are not symmetric: setting them too high just means
  more visitors see the manual picker than strictly necessary (mildly
  annoying, self-correcting the moment they click); setting them too low
  risks a confident, silent, wrong auto-resolution (natasja_italki6's
  failure mode, now automated). When in doubt, fail toward the picker.

  Getting a REAL number later is cheap, not a re-architecture: every
  SpeakerResolution already carries the full ranked candidate list with
  raw similarity scores (see the `candidates` field), whichever way a
  resolution went. The only thing production needs to do is persist that
  struct somewhere it can be queried (a log line is enough to start) --
  once a few hundred real resolutions have accumulated, plot similarity
  for confirmed-correct vs. confirmed-wrong auto-matches (from "Switch
  speaker" corrections) and pick the threshold that actually separates
  them on this data, the same way identify_target_speaker.py already
  does offline for Dan's personal corpus. Nothing here needs to change to
  make that possible later -- only to make it happen.

EMBEDDING MODEL
  SpeechBrain's ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), the same
  embedding family pyannote's own open-source clustering step uses
  internally, and the one identify_target_speaker.py already uses and
  documents for Dan's personal corpus -- kept identical on purpose so
  findings from that offline diagnostic tool transfer directly to this
  production path instead of being a second, divergently-tuned system.
  SpeakerEmbedder wraps it behind an ABC so the model can be swapped
  (e.g. for pyannoteAI's paid /voiceprint API, if the cost/accuracy
  trade-off ever favors that) without touching resolution logic.

THIS FILE HAS NO FASTAPI/FLASK CODE IN IT ON PURPOSE. It's the service
layer lomb_backend_prd_v1.md Section 4's `POST /analyze` handler should
import and call, kept framework-free so it's unit-testable without spinning
up the web app (see test_voice_enrollment.py) and portable if the framework
choice ever changes.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# Tunable policy -- see "THRESHOLD POLICY" above for the full reasoning.
# Deliberately conservative, deliberately not calibrated yet: set above
# SpeechBrain's own shipped balanced-operating-point default (0.25 raw
# cosine, speechbrain/speechbrain#2148) because a false accept here is
# far more costly than an unnecessary manual pick. Revisit once real
# resolution logs exist -- see "Getting a REAL number later" above.
# ---------------------------------------------------------------------
MATCH_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.10

SAMPLE_RATE = 16000  # 16kHz mono float32 -- matches identify_target_speaker.py and what the ECAPA model expects
MIN_ACTIVE_ENROLLMENT_SECONDS = 8.0  # below this, a dedicated enrollment clip is rejected outright -- see enroll_active()


@dataclass(frozen=True)
class SpeakerAudio:
    """One detected speaker's audio, already extracted from the diarized
    turns for a single upload -- e.g. via the same ffmpeg atrim+concat
    approach identify_target_speaker.py uses. speaker_label is whatever
    the diarizer called this cluster for THIS upload only (SPEAKER_00,
    SPEAKER_01, ...) -- it carries no identity across uploads, that's the
    entire problem this module exists to solve."""
    speaker_label: str
    waveform: np.ndarray  # 16kHz mono float32


@dataclass(frozen=True)
class SpeakerCandidate:
    speaker_label: str
    similarity: float


@dataclass(frozen=True)
class SpeakerResolution:
    """Result of trying to auto-resolve which detected speaker is
    user_id. The caller (the /analyze handler) branches on
    needs_manual_confirmation: True means "show the existing Section 6.2
    picker UI," exactly as it already does for a brand-new user."""
    needs_manual_confirmation: bool
    resolved_speaker_label: str | None   # set only when not needs_manual_confirmation
    candidates: list[SpeakerCandidate]   # every candidate, best first -- always populated, for logging/debugging even on a manual-confirmation result
    reason: str                          # "no_enrolled_voiceprint" | "confident_match" | "below_match_threshold" | "ambiguous_margin"


class VoiceprintRepository(ABC):
    """Storage seam. Swap SQLiteVoiceprintRepository for a Postgres/pgvector
    (or any vector store) implementation without touching
    SpeakerResolutionService -- see module docstring's STORAGE section for
    when that swap is actually warranted."""

    @abstractmethod
    def get(self, user_id: str) -> tuple[np.ndarray, int] | None:
        """Returns (unit_direction_vector, sample_count), or None if this
        user has never been enrolled."""

    @abstractmethod
    def upsert(self, user_id: str, unit_direction_vector: np.ndarray, sample_count: int) -> None:
        """Overwrite the stored voiceprint with the caller's already-updated
        running mean. SpeakerResolutionService owns the running-mean math
        (see _update_running_mean); the repository just persists whatever
        it's handed."""


class SQLiteVoiceprintRepository(VoiceprintRepository):
    """Default, zero-extra-infra implementation. One row per user_id.
    Embedding stored as raw float32 bytes -- fine at this scale (a few
    hundred bytes per user); move to a real vector column type only if/
    when this repository is swapped for Postgres+pgvector."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voiceprints (
                    user_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, user_id: str) -> tuple[np.ndarray, int] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT embedding, embedding_dim, sample_count FROM voiceprints WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        blob, dim, count = row
        vec = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
        return vec, count

    def upsert(self, user_id: str, unit_direction_vector: np.ndarray, sample_count: int) -> None:
        vec = unit_direction_vector.astype(np.float32)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO voiceprints (user_id, embedding, embedding_dim, sample_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    embedding_dim = excluded.embedding_dim,
                    sample_count = excluded.sample_count
                """,
                (user_id, vec.tobytes(), vec.shape[0], sample_count),
            )
            conn.commit()


class SpeakerEmbedder(ABC):
    """Wraps whatever model turns a waveform into a fixed-length speaker
    embedding. Kept abstract so the ECAPA implementation (below, in
    speechbrain_embedder.py -- not imported by this module directly, so
    unit tests here never need speechbrain/torch installed) can be
    swapped without touching SpeakerResolutionService."""

    @abstractmethod
    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """waveform: 16kHz mono float32. Returns a 1-D embedding vector;
        does not need to be pre-normalized -- SpeakerResolutionService
        normalizes on the way in."""


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector -- embedder returned all-zeros, which means it was fed silence or failed silently.")
    return v / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


class SpeakerResolutionService:
    """The thing lomb_backend_prd_v1.md Section 4's POST /analyze handler
    calls after diarization completes and before deciding whether to show
    the Section 6.2 picker UI."""

    def __init__(
        self,
        embedder: SpeakerEmbedder,
        repository: VoiceprintRepository,
        match_threshold: float = MATCH_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
    ):
        self.embedder = embedder
        self.repository = repository
        self.match_threshold = match_threshold
        self.margin_threshold = margin_threshold

    def resolve(self, user_id: str, speakers: list[SpeakerAudio]) -> SpeakerResolution:
        if not speakers:
            raise ValueError("resolve() called with no detected speakers -- caller should have failed the upload before this point")

        stored = self.repository.get(user_id)
        candidate_embeddings = {s.speaker_label: self.embedder.embed(s.waveform) for s in speakers}

        if stored is None:
            # No voiceprint on file yet -- always the existing manual picker.
            # candidates has no meaningful similarity to report yet; still
            # populate labels so the picker UI has something to key off.
            return SpeakerResolution(
                needs_manual_confirmation=True,
                resolved_speaker_label=None,
                candidates=[SpeakerCandidate(label, similarity=0.0) for label in candidate_embeddings],
                reason="no_enrolled_voiceprint",
            )

        reference_vec, _sample_count = stored
        ranked = sorted(
            (SpeakerCandidate(label, _cosine(emb, reference_vec)) for label, emb in candidate_embeddings.items()),
            key=lambda c: c.similarity,
            reverse=True,
        )
        best = ranked[0]
        margin = best.similarity - ranked[1].similarity if len(ranked) > 1 else best.similarity

        if best.similarity < self.match_threshold:
            return SpeakerResolution(True, None, ranked, "below_match_threshold")
        if margin < self.margin_threshold:
            return SpeakerResolution(True, None, ranked, "ambiguous_margin")

        return SpeakerResolution(False, best.speaker_label, ranked, "confident_match")

    def enroll(self, user_id: str, confirmed_speaker: SpeakerAudio) -> None:
        """Call this whenever a speaker identity is confirmed for a
        user_id -- whether via the manual picker (first-ever upload, or a
        needs_manual_confirmation fallback) or implicitly after a
        confident_match the visitor didn't have to correct. Every
        confirmation reinforces the voiceprint, so it drifts gracefully
        with the learner's own voice (different mic, a cold, background
        noise) instead of being frozen at whatever the first sample
        happened to sound like."""
        new_embedding = self.embedder.embed(confirmed_speaker.waveform)
        stored = self.repository.get(user_id)
        if stored is None:
            self.repository.upsert(user_id, _unit(new_embedding), sample_count=1)
            return
        reference_vec, sample_count = stored
        updated_vec, updated_count = _update_running_mean(reference_vec, sample_count, new_embedding)
        self.repository.upsert(user_id, updated_vec, updated_count)

    def enroll_active(
        self,
        user_id: str,
        recording: np.ndarray,
        initial_weight: int = 4,
        force: bool = False,
    ) -> None:
        """Dedicated, first-time enrollment -- the counterpart to enroll()
        for a visitor who records a short clean sample (read a passage
        aloud, or just talk for a few seconds) BEFORE ever uploading a
        real tutoring session, rather than the voiceprint being created
        as a side effect of the first picker click.

        Two real differences from a session-derived enrollment, not just
        a different call site:

          1. No diarization, no speaker_label -- the clip is single-
             speaker by construction (nothing else was recorded), so it
             embeds directly. Reusing SpeakerAudio here would carry a
             speaker_label field that means nothing; a bare waveform is
             the honest shape for this input.

          2. Seeded with initial_weight (default 4) instead of the usual
             1 an ordinary enroll() confirmation gets. A purpose-recorded
             clip is typically cleaner (closer mic, no cross-talk, the
             visitor was asked to just talk) than a slice pulled out of a
             real tutoring call, so it earns more say in the running mean
             before ordinary sessions start diluting it. This is a head
             start, not a lock -- enough real enroll() confirmations
             still move the voiceprint over time, same as always; a
             higher initial_weight only slows how fast that happens
             early on. Tune this only once real accuracy data says
             whether 4 is too timid or too stubborn (see
             voice_enrollment.py's THRESHOLD POLICY docstring for the
             same "don't hand-tune blind" caveat -- it applies here too).

        Called once, at onboarding -- raises if user_id is already
        enrolled unless force=True (re-recording deliberately, e.g. after
        a run of wrong auto-matches poisoned the running mean; this is
        also the "reset voice profile" affordance flagged as missing in
        the design doc -- force=True from a settings-page action is
        exactly that reset, implemented as a fresh enroll_active() call
        rather than a separate delete endpoint).
        """
        duration_seconds = len(recording) / SAMPLE_RATE
        if duration_seconds < MIN_ACTIVE_ENROLLMENT_SECONDS:
            raise ValueError(
                f"recording is {duration_seconds:.1f}s, need at least "
                f"{MIN_ACTIVE_ENROLLMENT_SECONDS:.0f}s for a reliable enrollment -- "
                "ask the visitor to keep talking a little longer"
            )
        if self.repository.get(user_id) is not None and not force:
            raise ValueError(
                f"user_id {user_id!r} already has a voiceprint on file -- "
                "pass force=True to deliberately replace it (e.g. a "
                "visitor-initiated 'reset my voice profile')"
            )
        embedding = self.embedder.embed(recording)
        self.repository.upsert(user_id, _unit(embedding), sample_count=initial_weight)


def _update_running_mean(reference_unit_vec: np.ndarray, sample_count: int, new_raw_embedding: np.ndarray) -> tuple[np.ndarray, int]:
    """Incremental spherical mean: accumulate in the current mean's
    direction, weighted by how many samples it already represents, add
    the new unit-normalized sample, renormalize. Closed-form running sum
    would need sample_count to be stored as an actual running SUM
    (magnitude carries weight information); this reconstructs that from
    the stored unit vector by scaling it back up by sample_count before
    adding, which is exactly equivalent and avoids a second stored field."""
    weighted_sum = reference_unit_vec * sample_count + _unit(new_raw_embedding)
    updated_count = sample_count + 1
    return _unit(weighted_sum), updated_count
