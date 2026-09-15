"""
Unit tests for voice_enrollment.py's decision logic -- the part that
actually matters for safety (never silently misattribute a session).
Uses a FakeEmbedder so these run anywhere, no speechbrain/torch/audio
required -- this is exactly the part identify_target_speaker.py's own
verification couldn't reach in the cloud sandbox it was written in.

Run: python -m pytest test_voice_enrollment.py -v
  (or just: python test_voice_enrollment.py)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# repo root on sys.path -- voice_enrollment.py moved into voice_enrollment/
# post-reorg, a sibling of this file's own new directory (tests/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_enrollment.voice_enrollment import (
    MIN_ACTIVE_ENROLLMENT_SECONDS,
    SAMPLE_RATE,
    SpeakerAudio,
    SpeakerEmbedder,
    SpeakerResolutionService,
    SQLiteVoiceprintRepository,
)


class TableEmbedder(SpeakerEmbedder):
    """A speaker's waveform IS its label, wrapped -- SpeakerAudio.waveform
    is opaque to SpeakerResolutionService, so tests can pass a 1-element
    array that encodes an index into a lookup table instead of decoding
    real audio."""

    def __init__(self):
        self._table: dict[int, np.ndarray] = {}
        self._next_id = 0

    def register(self, embedding: np.ndarray) -> np.ndarray:
        """Returns a fake 'waveform' that this embedder will map back to
        `embedding` when embed() is called on it."""
        idx = self._next_id
        self._next_id += 1
        self._table[idx] = embedding
        return np.array([idx], dtype=np.float32)

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        return self._table[int(waveform[0])].copy()

    def register_recording(self, embedding: np.ndarray, seconds: float) -> np.ndarray:
        """Same idea as register(), but padded out to a real sample count
        so enroll_active()'s duration gate (which reads len(recording),
        not just the lookup index) sees a plausible-length clip."""
        idx = self._next_id
        self._next_id += 1
        self._table[idx] = embedding
        n_samples = int(seconds * SAMPLE_RATE)
        padded = np.zeros(n_samples, dtype=np.float32)
        padded[0] = idx
        return padded


def _tmp_db() -> str:
    return tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name


def test_first_upload_always_needs_manual_confirmation():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    dan_wave = embedder.register(np.array([1.0, 0.0, 0.0]))
    tutor_wave = embedder.register(np.array([0.0, 1.0, 0.0]))
    speakers = [SpeakerAudio("SPEAKER_00", dan_wave), SpeakerAudio("SPEAKER_01", tutor_wave)]

    result = service.resolve(user_id="dan", speakers=speakers)

    assert result.needs_manual_confirmation is True
    assert result.reason == "no_enrolled_voiceprint"
    assert result.resolved_speaker_label is None
    print("PASS: first-ever upload always falls through to manual picker")


def test_confident_match_after_enrollment_skips_manual_pick():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    dan_ref = np.array([1.0, 0.0, 0.0])
    service.enroll("dan", SpeakerAudio("SPEAKER_00", embedder.register(dan_ref)))

    # New upload: same speaker (near-identical embedding, tiny noise) vs.
    # a clearly different speaker.
    dan_wave = embedder.register(dan_ref + np.array([0.01, 0.0, 0.0]))
    tutor_wave = embedder.register(np.array([0.0, 1.0, 0.0]))
    result = service.resolve("dan", [SpeakerAudio("SPEAKER_00", dan_wave), SpeakerAudio("SPEAKER_01", tutor_wave)])

    assert result.needs_manual_confirmation is False
    assert result.resolved_speaker_label == "SPEAKER_00"
    assert result.reason == "confident_match"
    print(f"PASS: confident match auto-resolves (top similarity={result.candidates[0].similarity:.3f})")


def test_ambiguous_margin_falls_back_to_manual_even_with_high_similarity():
    """The exact failure mode natasja_italki6 exposed: two candidates both
    plausible. High absolute similarity to ONE of them isn't enough if the
    other is nearly as close."""
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    dan_ref = np.array([1.0, 0.0, 0.0])
    service.enroll("dan", SpeakerAudio("SPEAKER_00", embedder.register(dan_ref)))

    # Two candidates nearly equidistant from the reference -- a coin flip.
    a = embedder.register(np.array([1.0, 0.05, 0.0]))
    b = embedder.register(np.array([1.0, 0.06, 0.0]))
    result = service.resolve("dan", [SpeakerAudio("A", a), SpeakerAudio("B", b)])

    assert result.needs_manual_confirmation is True
    assert result.reason == "ambiguous_margin"
    print(f"PASS: near-tie correctly deferred to manual pick (margin={result.candidates[0].similarity - result.candidates[1].similarity:.4f})")


def test_low_similarity_falls_back_even_if_it_is_the_only_candidate_shape():
    """Voice changed enough (new mic, sick, years later) that nothing
    matches well -- must not force a match just because something scored
    highest."""
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()), match_threshold=0.9)

    dan_ref = np.array([1.0, 0.0, 0.0])
    service.enroll("dan", SpeakerAudio("SPEAKER_00", embedder.register(dan_ref)))

    far_a = embedder.register(np.array([0.2, 0.9, 0.1]))
    far_b = embedder.register(np.array([-0.5, 0.3, 0.8]))
    result = service.resolve("dan", [SpeakerAudio("A", far_a), SpeakerAudio("B", far_b)])

    assert result.needs_manual_confirmation is True
    assert result.reason == "below_match_threshold"
    print("PASS: no candidate close enough -> manual pick, not a forced best-guess")


def test_enrollment_running_mean_moves_toward_repeated_samples():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    service.enroll("dan", SpeakerAudio("x", embedder.register(np.array([1.0, 0.0, 0.0]))))
    for _ in range(5):
        service.enroll("dan", SpeakerAudio("x", embedder.register(np.array([0.0, 1.0, 0.0]))))

    stored_vec, count = service.repository.get("dan")
    assert count == 6
    # After 5 reinforcements toward [0,1,0], the running mean should now
    # point much closer to [0,1,0] than to the original [1,0,0].
    assert stored_vec[1] > stored_vec[0]
    print(f"PASS: running mean after 1x[1,0,0] + 5x[0,1,0] = {stored_vec.round(3)} (count={count})")


def test_different_users_have_independent_voiceprints():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    service.enroll("dan", SpeakerAudio("x", embedder.register(np.array([1.0, 0.0, 0.0]))))
    service.enroll("alicia", SpeakerAudio("x", embedder.register(np.array([0.0, 1.0, 0.0]))))

    dan_stored, _ = service.repository.get("dan")
    alicia_stored, _ = service.repository.get("alicia")
    assert dan_stored[0] > 0.9
    assert alicia_stored[1] > 0.9
    print("PASS: per-user voiceprints stored independently, no cross-contamination")


def test_active_enrollment_seeds_voiceprint_with_higher_weight_than_passive():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    clip = embedder.register_recording(np.array([1.0, 0.0, 0.0]), seconds=12.0)
    service.enroll_active("dan", clip, initial_weight=4)

    stored_vec, count = service.repository.get("dan")
    assert count == 4  # not 1, like a passive enroll() would give it
    assert stored_vec[0] > 0.9
    print(f"PASS: enroll_active seeds sample_count={count} (vs. 1 for an ordinary passive confirmation)")


def test_active_enrollment_rejects_too_short_a_recording():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    too_short = embedder.register_recording(np.array([1.0, 0.0, 0.0]), seconds=MIN_ACTIVE_ENROLLMENT_SECONDS - 1)
    try:
        service.enroll_active("dan", too_short)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert service.repository.get("dan") is None  # rejected clip must not partially enroll
    print("PASS: a too-short active-enrollment recording is rejected outright, nothing stored")


def test_active_enrollment_refuses_to_silently_overwrite_without_force():
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    first = embedder.register_recording(np.array([1.0, 0.0, 0.0]), seconds=10.0)
    service.enroll_active("dan", first)

    second = embedder.register_recording(np.array([0.0, 1.0, 0.0]), seconds=10.0)
    try:
        service.enroll_active("dan", second)
        raised = False
    except ValueError:
        raised = True
    assert raised
    unchanged_vec, _ = service.repository.get("dan")
    assert unchanged_vec[0] > 0.9  # still the FIRST enrollment, untouched

    service.enroll_active("dan", second, force=True)
    replaced_vec, replaced_count = service.repository.get("dan")
    assert replaced_vec[1] > 0.9  # now the second recording
    assert replaced_count == 4  # default initial_weight, a fresh start -- not additive with the old one
    print("PASS: re-enrolling without force is refused; force=True cleanly replaces (the 'reset voice profile' path)")


def test_active_enrollment_lets_the_very_first_real_upload_skip_the_picker():
    """The actual payoff of active enrollment over passive: a session-
    derived voiceprint can't exist until AFTER the first picker click, so
    session 1 always shows the picker. An actively-enrolled voiceprint
    already exists before session 1 arrives."""
    embedder = TableEmbedder()
    service = SpeakerResolutionService(embedder, SQLiteVoiceprintRepository(_tmp_db()))

    onboarding_clip = embedder.register_recording(np.array([1.0, 0.0, 0.0]), seconds=15.0)
    service.enroll_active("dan", onboarding_clip)

    # First real tutoring-session upload, ever -- no prior enroll() call.
    dan_in_session = embedder.register(np.array([0.97, 0.02, 0.0]))
    tutor_in_session = embedder.register(np.array([0.0, 1.0, 0.0]))
    result = service.resolve("dan", [SpeakerAudio("SPEAKER_00", dan_in_session), SpeakerAudio("SPEAKER_01", tutor_in_session)])

    assert result.needs_manual_confirmation is False
    assert result.resolved_speaker_label == "SPEAKER_00"
    print("PASS: session 1 skips the picker when active enrollment already happened at onboarding")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
