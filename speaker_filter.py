"""
Speaker filtering -- the missing piece between diarization output and the
metric prompt registry.

Nothing in registry.py or prompts/*.py filters by speaker. Each of the 10
prompts just receives whatever sentence / word-timestamp window / audio
turn it's handed and trusts it's already single-speaker (verified below).
Before today, nothing in this codebase actually produced that guarantee --
PRD Section 6.2 requires it ("the visitor tells the system which diarized
speaker is them... before running any metrics"), Section 6.3 assumes it
("input a diarized transcript JSON" implicitly means already resolved to
one speaker), but the code to do the resolving didn't exist. This module
is that piece.

Deliberately speaker-COUNT-agnostic: every function here takes a
target_speaker_id and a set of turns, and works identically whether the
original recording had 2 speakers, 5, or (degenerate but valid) 1 -- none
of this counts or assumes how many OTHER speakers exist. See the
speaker-count audit below the class definitions for how that claim was
checked against all 10 registered prompts, not just asserted.
"""

import re
from dataclasses import dataclass


class UnknownSpeakerError(ValueError):
    """Raised when target_speaker_id doesn't match any speaker this
    transcript's diarization actually produced. This is the same check
    lomb_backend_prd_v1.md Section 5 requires from
    POST /analyze/{job_id}/speaker -- reused here so the metrics layer
    can't silently run on an empty or wrong-speaker slice if that
    endpoint's own validation is ever bypassed or buggy."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds -- exact reference point (session start vs.
    end: float     # turn start) TBD at Milestone 3 against real WhisperX output


@dataclass(frozen=True)
class Turn:
    """One continuous stretch of speech from one speaker, as diarization
    would produce it. First-cut approximation of WhisperX + pyannote
    output shape -- reconcile field-for-field against the real library
    output during PRD Milestone 3 (Section 6.1); not yet validated
    against a live diarization run."""
    speaker_id: str
    text: str
    start: float
    end: float
    words: list[Word]


def speakers_present(transcript: list[Turn]) -> set[str]:
    """All speaker_ids diarization actually produced for this file -- the
    source of truth POST /analyze/{job_id}/speaker validates a submitted
    speaker_id against (lomb_backend_prd_v1.md Section 5)."""
    return {turn.speaker_id for turn in transcript}


def filter_to_target_speaker(transcript: list[Turn], target_speaker_id: str) -> list[Turn]:
    """Every metric input-builder below is built on top of this. Drops
    every turn not spoken by target_speaker_id, preserves the original
    chronological order of the ones that remain (GVT-1 needs consecutive
    clauses "in the order they were spoken" -- reordering would corrupt
    its tense-frame tracking), and refuses to silently proceed if
    target_speaker_id was never actually detected in this transcript.

    Works identically for a 2-speaker file, a 5-speaker group call, or a
    solo recording -- nothing here counts or assumes how many OTHER
    speakers exist. That's the actual mechanism behind "metrics are
    computed with the confirmed target speaker, independent of how many
    other voices were in the room."
    """
    present = speakers_present(transcript)
    if target_speaker_id not in present:
        raise UnknownSpeakerError(
            f"target_speaker_id {target_speaker_id!r} was not produced by "
            f"diarization for this transcript (detected speakers: {sorted(present)}). "
            "POST /analyze/{job_id}/speaker should have already rejected this "
            "value before it ever reached the metrics layer -- treat this as a "
            "bug, not an expected user-facing error, if it fires here."
        )
    return [turn for turn in transcript if turn.speaker_id == target_speaker_id]


# ---------------------------------------------------------------------------
# Per-metric input builders. Each one only ever sees the pre-filtered,
# single-speaker turn list filter_to_target_speaker() produced above -- none
# of them re-check speaker_id, because by this point there's only one
# speaker left to check against. That's deliberate: it keeps every prompt in
# prompts/*.py completely unaware that other speakers ever existed in the
# source recording, and keeps this module the ONLY place speaker-count logic
# lives at all.
# ---------------------------------------------------------------------------

def to_sentences(target_turns: list[Turn]) -> list[str]:
    """Feeds the 6 metrics that run one call per raw sentence: GDD-1, GDD-2,
    GVT-2, LPF, LP, STRUCTURE_BREADTH. Also the raw material two OTHER
    functions below build on top of, rather than feeding directly: GVT-1
    actually runs on windows of these (see to_sentence_windows()), and
    FORMULAIC actually runs on (candidate, sentence) pairs built from
    these (see to_formulaic_candidates()) -- both are "input_kind ==
    sentence" on paper (metric_types.py has no richer category for either
    shape) but neither is fed one bare entry from this list at a time.

    Placeholder sentence-splitting (naive split on .?!) -- German
    sentence-boundary detection on disfluent ASR output is a real
    problem on its own (Whisper's punctuation is not reliable ground
    truth) and deserves a proper tool (e.g. spaCy's German sentencizer)
    before this is trusted in production. Flagging this as a stub rather
    than quietly shipping it as if the segmentation problem were solved.
    """
    sentences = []
    for turn in target_turns:
        for raw in re.split(r"(?<=[.?!])\s+", turn.text.strip()):
            raw = raw.strip()
            if raw:
                sentences.append(raw)
    return sentences


def to_sentence_windows(target_turns: list[Turn], window_size: int = 3) -> list[str]:
    """Feeds GVT-1 specifically (input_kind == "sentence" on paper, but per
    its own docstring actually "a short sequence of consecutive German
    clauses, in the order they were spoken").

    2026-09-03: added because pipeline.py was feeding GVT-1 one entry from
    to_sentences() at a time, same as every other sentence metric -- which
    meant a past-tense frame set up in one sentence and a later sentence
    reverting to present tense were NEVER in the same request, so GVT-1
    could structurally never catch its own target error. to_sentences()
    already produces the right raw material (chronologically ordered,
    target-speaker-only, turn boundaries dropped) -- confirmed against a
    real corpus example, dan_error_analysis_master_v3.md's leonie_tandem1
    GVT example 1, which spans TWO separate sentences ("und dann kommt
    letztes Jahr." / "Ich mache ein Urlaub in Hamburg und nach Koeln.")
    with another speaker's turn in between them in the original audio --
    exactly the case to_sentences()' flattening is supposed to handle, and
    exactly the case single-sentence feeding could never catch. This
    function is the missing piece: group to_sentences()' flat list into
    windows before handing them to the LLM, the same pattern
    to_word_timestamp_windows() below already uses for UNFILLED_PAUSE,
    applied to sentences instead of words.

    Sliding window (stride 1, not stride == window_size): a stride equal
    to the window size would let a drift spanning a window BOUNDARY slip
    through uncaught (the past frame lands in the last sentence of window
    N, the reverted-to-present verb lands in the first sentence of window
    N+1 -- disjoint windows would never put both in the same request).
    Sliding by 1 guarantees every adjacent pair of sentences shares at
    least one window, at the cost of the same sentence appearing in
    multiple windows and (real, not yet solved) the possibility of the
    same underlying error getting flagged more than once across
    overlapping windows -- report-layer dedup already has to handle
    cross-metric duplicates for the same reason; this adds a
    within-metric version of that same problem, not a new kind of
    problem.
    """
    sentences = to_sentences(target_turns)
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
    return [
        " ".join(sentences[i:i + window_size])
        for i in range(len(sentences) - window_size + 1)
    ]


def to_formulaic_candidates(target_turns: list[Turn], bundles: list[str]) -> list[str]:
    """Feeds FORMULAIC. 2026-09-03: added alongside prompts/formulaic.py's
    new BUNDLES list, for the same reason to_sentence_windows() got added
    for GVT-1 today -- FORMULAIC's own CONFIG.input_kind comment already
    said "actually candidate + sentence -- see
    speaker_filter.to_formulaic_candidates()" as a forward reference before
    this function existed; this is that function.

    Unlike every other builder in this module, FORMULAIC's real unit of
    work isn't one sentence -- it's one (candidate, sentence) PAIR, because
    the whole point of the metric is deciding whether one specific
    candidate word/phrase is being used formulaically IN that sentence, not
    classifying the sentence as a whole. So this scans each sentence from
    to_sentences() against `bundles` (the caller passes prompts.formulaic's
    own BUNDLES constant -- not imported directly here, so this module
    never has to know FORMULAIC-specific content, same separation
    to_word_timestamp_windows() keeps from FILLED_PAUSE/UNFILLED_PAUSE's
    own trigger tokens) and emits one formatted
    'Candidate: "X" | Sentence: "Y"' string -- the exact shape
    prompts/formulaic.py's own few-shot examples use -- per match.

    Matching reuses prefilter.py's own approach (\\b...\\b, case-insensitive,
    longest-candidate-first so a short candidate can't shadow a longer one
    that contains it) rather than inventing a second way to do the same
    thing. A sentence containing none of `bundles` contributes zero
    entries -- the same "skip what can't possibly trigger" property
    prefilter.py's should_run() gives GDD-1/GDD-2, achieved here by
    construction instead of a separate filter step, since for FORMULAIC the
    candidate-scan and the "is it worth a call" decision are the same
    question.

    Deduplicated per sentence: if a candidate appears more than once in the
    same sentence (repetition, false starts), it's still only one pair --
    repeating it wouldn't give the LLM call any new context, only inflate
    the call count.
    """
    sentences = to_sentences(target_turns)
    if not sentences or not bundles:
        return []
    ordered = sorted(set(bundles), key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(b) for b in ordered) + r")\b", re.IGNORECASE)
    # Case-insensitive matching means the text actually found in the
    # sentence (e.g. capitalized "Mal" at a sentence start) may not be
    # spelled exactly like its BUNDLES entry -- report the canonical
    # BUNDLES spelling in the candidate pair (what prompts/formulaic.py's
    # few-shot examples key off of), not whatever casing happened to
    # appear in this particular sentence.
    canonical_by_lower = {b.lower(): b for b in bundles}
    candidates = []
    for sentence in sentences:
        seen = set()
        for match in pattern.finditer(sentence):
            canonical = canonical_by_lower.get(match.group(0).lower(), match.group(0))
            if canonical in seen:
                continue
            seen.add(canonical)
            candidates.append(f'Candidate: "{canonical}" | Sentence: "{sentence}"')
    return candidates


def to_word_timestamp_windows(target_turns: list[Turn], window_size: int = 12) -> list[list[dict]]:
    """Feeds UNFILLED_PAUSE and, as of 2026-09-03, FILLED_PAUSE too (both
    input_kind == "word_timestamps" -- FILLED_PAUSE was redefined off
    audio onto transcript text per Dan's direct instruction; see
    prompts/filled_pause.py's own docstring for the full reasoning and the
    accuracy-ceiling tradeoff that redefinition accepts). Both metrics
    share these exact windows -- no separate builder needed for the
    second one, since scanning a window of words for hesitation tokens and
    scanning the same window for large silences are just two different
    questions asked of the same input shape.

    The thing this has to get right: windows are built PER TURN, never
    spanning a turn boundary. A naive "keep every word where speaker ==
    target, then chunk the flat list" would splice together the target's
    last word before another speaker's turn and their first word after
    it -- making the silence WHILE THE OTHER SPEAKER WAS TALKING look
    like one giant unfilled pause from the target. That failure mode
    isn't tied to a specific speaker count (it happens with 2 speakers
    exactly as easily as 5); it's a turn-boundary bug, and chunking
    strictly within each turn (never across turns) is what avoids it.
    """
    windows = []
    for turn in target_turns:
        words = [{"word": w.text, "start": w.start, "end": w.end} for w in turn.words]
        for i in range(0, len(words), window_size):
            chunk = words[i:i + window_size]
            if chunk:
                windows.append(chunk)
    return windows


def to_audio_turns(target_turns: list[Turn]) -> list[dict]:
    """ORPHANED as of 2026-09-03 -- fed FILLED_PAUSE back when that metric's
    input_kind was "audio_turn"; FILLED_PAUSE was redefined to run off
    transcript word-timestamps instead (prompts/filled_pause.py's docstring
    has the full reasoning), so nothing in this codebase calls this
    function anymore. Left in place, not deleted: it's still a correct,
    independently useful building block (per-turn [start, end] + ASR text,
    ready for audio slicing) if a future metric or a revisit of
    FILLED_PAUSE's accuracy ceiling ever needs real audio input again --
    deleting working code because its one caller went away would just mean
    re-deriving the same turn-boundary logic later. The self-test below
    still exercises it directly so it can't silently rot un-runnable.

    Returns each turn's own [start, end] plus its ASR text -- the actual
    audio bytes still need to be sliced from the original upload using
    these timestamps (ffmpeg/pydub, wherever the real audio file lives;
    this module only has the transcript, not the audio, so it can't do
    that slicing itself). Because these windows come straight from
    filter_to_target_speaker()'s per-turn boundaries, each clip is
    guaranteed to contain only the target speaker's own voice for that
    stretch -- modulo whatever diarization boundary error already exists
    upstream (cross-talk right at a turn edge is a diarization-accuracy
    problem, not something this filtering step can fix).
    """
    return [
        {"speaker_id": t.speaker_id, "start": t.start, "end": t.end, "asr_text": t.text}
        for t in target_turns
    ]


# ---------------------------------------------------------------------------
# Speaker-count audit (2026-09-01, updated 2026-09-03 for the GVT-1
# windowing fix and the FORMULAIC/FILLED_PAUSE redefinitions -- the
# conclusion didn't change, only which builder function each metric's
# input now comes from): read all 10 files in prompts/ looking for any
# dependency on the number of speakers in the source recording -- "2
# speakers," "interlocutor," "the other speaker," a fixed dialogue/
# turn-taking structure, anything that would break if a transcript had 3+
# people instead of 2. None found. Every prompt's SYSTEM_INSTRUCTION talks
# about "the speaker" (singular) or "one person" -- generic language about
# whichever single speaker's sentence/window/candidate/turn it was handed,
# not a fixed conversation shape:
#   GDD-1, GDD-2, LPF, LP, STRUCTURE_BREADTH -- input_kind "sentence", fed
#     one bare entry from to_sentences() at a time. No reference to any
#     other speaker at all.
#   GVT-1 -- input_kind "sentence" but actually "a short sequence of German
#     clauses (in the order they were spoken)" per its own SYSTEM_INSTRUCTION:
#     needs consecutive clauses from ONE person's own narrative (to track a
#     past-tense frame across them), fed via to_sentence_windows() (built on
#     top of to_sentences(), same single-speaker source). No count assumption.
#   GVT-2 -- input_kind "sentence", evaluates one speaker's self-corrections
#     within a single clause. No count assumption.
#   FORMULAIC -- input_kind "sentence" but actually "one candidate word/
#     phrase + the sentence it appeared in," fed via
#     to_formulaic_candidates() (also built on top of to_sentences()). Still
#     just one speaker's own sentence per pair -- the candidate scan doesn't
#     add a second speaker into the picture.
#   UNFILLED_PAUSE, FILLED_PAUSE -- both input_kind "word_timestamps" as of
#     2026-09-03, both fed by to_word_timestamp_windows(): "a short window of
#     consecutive ASR words... spoken by one person." Depends on turn-aware
#     filtering (windows never cross a turn boundary), not on speaker count.
# Conclusion: none of the 10 LLM-assisted metrics have a 2-speaker
# dependency. The thing that DOES matter -- and that this module exists to
# guarantee -- is that every input handed to registry.classify() is already
# filtered to the confirmed target speaker before it gets there.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Self-proof, same spirit as registry.py's own __main__ block: a
    # synthetic transcript with 3 speakers (deliberately not 2, to
    # actually exercise the "independent of speaker count" claim rather
    # than just asserting it) run through the full filter -> per-metric
    # input pipeline, then plugged into a real prompt's build_request()
    # to confirm the output shape is exactly what registry.py expects.
    import sys
    sys.path.insert(0, "providers")
    from providers.gemini_provider import GeminiProvider
    from registry import METRIC_PROMPTS

    transcript = [
        Turn("SPEAKER_00", "Guten Tag, wie geht es Ihnen heute?", 0.0, 2.1, [
            Word("Guten", 0.0, 0.3), Word("Tag,", 0.3, 0.6), Word("wie", 0.7, 0.9),
            Word("geht", 0.9, 1.1), Word("es", 1.1, 1.2), Word("Ihnen", 1.3, 1.6),
            Word("heute?", 1.7, 2.1),
        ]),
        Turn("SPEAKER_01", "Frueher, als ich Kind war, spiele ich gern Basketball.", 2.2, 5.0, [
            Word("Frueher,", 2.2, 2.6), Word("als", 2.7, 2.8), Word("ich", 2.8, 2.9),
            Word("Kind", 3.0, 3.2), Word("war,", 3.2, 3.5), Word("spiele", 3.6, 3.9),
            Word("ich", 3.9, 4.0), Word("gern", 4.1, 4.3), Word("Basketball.", 4.4, 5.0),
        ]),
        Turn("SPEAKER_02", "Ah interessant, erzaehlen Sie mehr.", 5.1, 6.4, [
            Word("Ah", 5.1, 5.3), Word("interessant,", 5.3, 5.8),
            Word("erzaehlen", 5.9, 6.1), Word("Sie", 6.1, 6.2), Word("mehr.", 6.2, 6.4),
        ]),
        Turn("SPEAKER_01", "Ich habe das Bild an die Wand gehaengt.", 6.5, 8.2, [
            Word("Ich", 6.5, 6.6), Word("habe", 6.6, 6.8), Word("das", 6.8, 6.9),
            Word("Bild", 6.9, 7.1), Word("an", 7.1, 7.2), Word("die", 7.2, 7.3),
            Word("Wand", 7.4, 7.7), Word("gehaengt.", 7.8, 8.2),
        ]),
    ]

    target = "SPEAKER_01"
    print(f"Speakers detected in this (3-speaker) transcript: {sorted(speakers_present(transcript))}")

    filtered = filter_to_target_speaker(transcript, target)
    print(f"Turns belonging to {target}: {len(filtered)} of {len(transcript)} total\n")

    sentences = to_sentences(filtered)
    print(f"to_sentences() -> {sentences}\n")

    windows = to_word_timestamp_windows(filtered, window_size=5)
    print(f"to_word_timestamp_windows() -> {len(windows)} window(s), "
          f"sizes {[len(w) for w in windows]}\n")

    audio_turns = to_audio_turns(filtered)
    print(f"to_audio_turns() -> {audio_turns}\n")

    # to_formulaic_candidates() self-test. Imported here, not at module
    # level, so this module never has a hard dependency on FORMULAIC's own
    # content -- only this __main__ self-proof needs a concrete BUNDLES list
    # to test against.
    from prompts.formulaic import BUNDLES

    formulaic_turns = filtered + [
        # Deliberately two candidates in one sentence (tests the "more than
        # one pair per sentence" path) plus a sentence-initial capitalized
        # "Aber" (tests that matching is case-insensitive AND that the
        # canonical lowercase BUNDLES spelling is what gets reported, not
        # whatever casing happened to appear here) plus a repeated "ja ja"
        # (tests per-sentence dedup -- one "ja" pair, not two).
        Turn("SPEAKER_01", "Ich koche gern verschiedene Sachen, aber heute nicht.", 8.3, 10.5, [
            Word("Ich", 8.3, 8.4), Word("koche", 8.4, 8.6), Word("gern", 8.6, 8.8),
            Word("verschiedene", 8.9, 9.3), Word("Sachen,", 9.3, 9.7),
            Word("aber", 9.8, 10.0), Word("heute", 10.0, 10.2), Word("nicht.", 10.2, 10.5),
        ]),
        Turn("SPEAKER_01", "Aber ja ja, das stimmt schon.", 10.6, 12.0, [
            Word("Aber", 10.6, 10.8), Word("ja", 10.8, 10.9), Word("ja,", 10.9, 11.0),
            Word("das", 11.1, 11.2), Word("stimmt", 11.2, 11.5), Word("schon.", 11.5, 11.8),
        ]),
    ]
    formulaic_candidates = to_formulaic_candidates(formulaic_turns, BUNDLES)
    print(f"to_formulaic_candidates() -> {len(formulaic_candidates)} pair(s):")
    for c in formulaic_candidates:
        print(f"  {c}")
    assert any(c.startswith('Candidate: "verschiedene Sachen"') for c in formulaic_candidates), (
        "expected the corpus-confirmed candidate to be found"
    )
    assert any(c.startswith('Candidate: "aber"') and "Ich koche" in c for c in formulaic_candidates), (
        "expected the second candidate in the same sentence to also be found"
    )
    aber_second_sentence = [c for c in formulaic_candidates if c.startswith('Candidate: "aber"') and "das stimmt" in c]
    assert len(aber_second_sentence) == 1, (
        f"expected sentence-initial capitalized 'Aber' to match and normalize to lowercase 'aber', got {aber_second_sentence}"
    )
    ja_matches = [c for c in formulaic_candidates if c.startswith('Candidate: "ja"') and "das stimmt" in c]
    assert len(ja_matches) == 1, (
        f"expected repeated 'ja ja' in one sentence to dedup to a single pair, got {ja_matches}"
    )
    print("\nto_formulaic_candidates() self-checks passed.\n")

    # Prove a filtered sentence plugs cleanly into an actual registered
    # prompt's request-building -- not just that this module runs, but
    # that its output is consumable by registry.py without modification.
    gemini = GeminiProvider()
    req = gemini.build_request(METRIC_PROMPTS["GDD-2"], sentences[-1])
    print(f"GDD-2 request built from filtered sentence {sentences[-1]!r}: "
          f"{list(req.keys())}")

    # Prove the unknown-speaker guard actually fires.
    try:
        filter_to_target_speaker(transcript, "SPEAKER_99")
        raise AssertionError("expected UnknownSpeakerError")
    except UnknownSpeakerError as e:
        print(f"\nUnknownSpeakerError correctly raised for a bogus id: {e}")
