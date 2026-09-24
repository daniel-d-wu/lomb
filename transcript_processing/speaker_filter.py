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

2026-09-06: added map_words_to_sentences(), which links to_sentences()'
sentence-level view and to_word_timestamp_windows()' word-level view
together for the first time -- see that function's own docstring. This
closes the gap that was blocking FILLED_PAUSE/UNFILLED_PAUSE (moving to
direct/regex computation) from reporting which sentence(s) a pause falls
in using the same sentence_indices scheme as every other feature.
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
    """Feeds every LLM fluenceme (as of 2026-09-24 the whole ordered list
    goes to each one in a single batched call -- see pipeline/batching.py).
    Also the raw material for FORMULAIC's regex matches (see
    find_formulaic_matches()); FORMULAIC isn't an LLM metric at all
    anymore as of 2026-09-05, so it has no input_kind to speak of.

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


def find_formulaic_matches(target_turns: list[Turn], bundles: list[str]) -> list[dict]:
    """Computes FORMULAIC. Renamed 2026-09-05 from to_formulaic_candidates()
    -- per Dan's explicit instruction, FORMULAIC is now a deterministic
    regex metric, not an LLM-assisted one (see prompts/formulaic.py's
    module docstring for the accuracy tradeoff that acceptance makes), so
    what this function returns changed along with the name: it used to
    format an LLM-prompt-ready 'Candidate: "X" | Sentence: "Y"' string per
    hit (the exact shape prompts/formulaic.py's now-deleted few-shot
    examples used) for pipeline.py to hand to registry.classify(). There's
    no LLM prompt to format for anymore, so this just returns plain
    {"candidate": str, "sentence": str} records -- pipeline.py builds
    whatever display string it wants directly from those two fields, and a
    match against `bundles` now IS the metric's answer, not merely a
    screen for whether an LLM call is worth making.

    Matching reuses prefilter.py's own approach (\\b...\\b, case-insensitive,
    longest-candidate-first so a short candidate can't shadow a longer one
    that contains it) rather than inventing a second way to do the same
    thing. A sentence containing none of `bundles` contributes zero
    entries.

    Deduplicated per sentence: if a candidate appears more than once in the
    same sentence (repetition, false starts), it's still only one match
    record -- repeating it wouldn't add information, only inflate the
    count.
    """
    sentences = to_sentences(target_turns)
    if not sentences or not bundles:
        return []
    ordered = sorted(set(bundles), key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(b) for b in ordered) + r")\b", re.IGNORECASE)
    # Case-insensitive matching means the text actually found in the
    # sentence (e.g. capitalized "Mal" at a sentence start) may not be
    # spelled exactly like its BUNDLES entry -- report the canonical
    # BUNDLES spelling as the candidate, not whatever casing happened to
    # appear in this particular sentence.
    canonical_by_lower = {b.lower(): b for b in bundles}
    matches = []
    for sentence in sentences:
        seen = set()
        for match in pattern.finditer(sentence):
            canonical = canonical_by_lower.get(match.group(0).lower(), match.group(0))
            if canonical in seen:
                continue
            seen.add(canonical)
            matches.append({"candidate": canonical, "sentence": sentence})
    return matches


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


CLAUSE_CONJUNCTIONS = {
    # Common German subordinating conjunctions -- a word immediately
    # followed by one of these is treated as a likely clause boundary.
    "weil", "dass", "als", "wenn", "obwohl", "waehrend", "während",
    "bevor", "nachdem", "ob", "damit", "sodass", "indem", "sobald",
    "seit", "seitdem",
    # Common coordinating conjunctions -- same treatment.
    "und", "aber", "oder", "sondern", "denn",
}


def map_words_to_sentences(target_turns: list[Turn]) -> list[dict]:
    """The missing link between to_sentences() (sentence-level text, no
    word linkage) and to_word_timestamp_windows() (word-level timestamps,
    no sentence linkage) -- added 2026-09-06 because FILLED_PAUSE and
    UNFILLED_PAUSE (moving to direct/regex computation, per Dan's
    instruction) need to report which sentence(s) a pause falls in using
    the SAME sentence_indices scheme every other feature's occurrences use
    (information_items.sentence_indices, an array -- see
    lomb_metric_architecture_v1.md), not a separate coordinate system.
    Before this function, there was no way to answer "which sentence does
    this word belong to" at all.

    Returns a flat list, one entry per word, in the exact same per-turn,
    in-order sequence to_word_timestamp_windows() chunks from (so the
    two can be lined up position-for-position when both are built from
    the same target_turns list):
        {
            "word": "spreche", "start": 1.60, "end": 1.90,
            "sentence_index": 3,     # position in to_sentences()' own list
            "is_sentence_end": False,
            "is_clause_end": False,
        }

    sentence_index alignment: matches to_sentences() position-for-position
    IF both are computed from the same target_turns. A word is treated as
    ending a sentence if it ends in . ? or ! (the same trigger
    to_sentences()' regex split uses) OR if it's the last word of its turn
    -- to_sentences() always closes out whatever's left in a turn as its
    own sentence entry even without terminal punctuation (its per-turn
    split never lets a "sentence" span two turns), so this function forces
    the same boundary at every turn edge to stay in sync. If to_sentences()'
    splitting approach ever changes (its own docstring already flags the
    naive .?! split as a stub -- e.g. swapped for a real sentencizer), this
    function's boundary rule must change with it or the two will silently
    drift out of alignment.

    is_clause_end is a heuristic, not a parser: a word is flagged if it
    ends in a comma, OR if the very next word is one of CLAUSE_CONJUNCTIONS
    above. This will both over- and under-flag real clause boundaries --
    German clause structure isn't reducible to a fixed word list -- but
    it's enough to distinguish "this pause landed at an obvious clause
    break" from "this pause landed mid-clause" for reporting purposes, not
    a claim of syntactic correctness. Never true at the same time as
    is_sentence_end (a sentence boundary is reported as that, not also as
    a clause boundary).
    """
    mapped: list[dict] = []
    sentence_index = 0
    for turn in target_turns:
        words = turn.words
        n = len(words)
        for i, w in enumerate(words):
            text = w.text.strip()
            is_sentence_end = bool(re.search(r"[.?!]$", text)) or (i == n - 1)
            is_clause_end = False
            if not is_sentence_end:
                next_text = words[i + 1].text.strip().rstrip(".,!?").lower()
                is_clause_end = text.endswith(",") or next_text in CLAUSE_CONJUNCTIONS
            mapped.append({
                "word": w.text,
                "start": w.start,
                "end": w.end,
                "sentence_index": sentence_index,
                "is_sentence_end": is_sentence_end,
                "is_clause_end": is_clause_end,
            })
            if is_sentence_end:
                sentence_index += 1
    return mapped


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
#   FORMULAIC -- as of 2026-09-05 a deterministic regex computation, not an
#     LLM-assisted metric at all (see prompts/formulaic.py and this
#     module's find_formulaic_matches(), renamed from
#     to_formulaic_candidates() the same day) -- built on top of
#     to_sentences(), same single-speaker source. Still just one speaker's
#     own sentence per match -- the regex scan doesn't add a second speaker
#     into the picture.
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
    from pathlib import Path

    # repo root on sys.path -- needed post-reorg since providers/ and
    # pipeline/ are now siblings of this file's own directory
    # (transcript_processing/), not the directory Python auto-adds for us.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from providers.gemini_provider import GeminiProvider
    from pipeline.registry import METRIC_PROMPTS

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

    # map_words_to_sentences() self-test: sentence_index must line up
    # exactly with to_sentences()' own list (same count, same order), and
    # the known comma-clause structure in "Frueher, als ich Kind war,
    # spiele ich gern Basketball." must actually get flagged.
    word_sentence_map = map_words_to_sentences(filtered)
    print(f"map_words_to_sentences() -> {len(word_sentence_map)} word(s):")
    for m in word_sentence_map:
        flags = []
        if m["is_sentence_end"]:
            flags.append("SENTENCE-END")
        if m["is_clause_end"]:
            flags.append("clause-end")
        print(f"  [{m['sentence_index']}] {m['word']:<12} {' '.join(flags)}")

    max_sentence_index = max(m["sentence_index"] for m in word_sentence_map)
    assert max_sentence_index == len(sentences) - 1, (
        f"map_words_to_sentences() sentence_index range (0..{max_sentence_index}) "
        f"must match to_sentences()' {len(sentences)} sentences exactly"
    )
    frueher_comma = next(m for m in word_sentence_map if m["word"] == "Frueher,")
    assert frueher_comma["is_clause_end"] and not frueher_comma["is_sentence_end"], (
        "'Frueher,' should be flagged as a clause boundary (trailing comma "
        "AND followed by the subordinating conjunction 'als'), not a sentence end"
    )
    war_comma = next(m for m in word_sentence_map if m["word"] == "war,")
    assert war_comma["is_clause_end"] and not war_comma["is_sentence_end"], (
        "'war,' should be flagged as a clause boundary (trailing comma), not a sentence end"
    )
    basketball_period = next(m for m in word_sentence_map if m["word"] == "Basketball.")
    assert basketball_period["is_sentence_end"] and not basketball_period["is_clause_end"], (
        "'Basketball.' ends the sentence, not just a clause"
    )
    print("\nmap_words_to_sentences() self-checks passed "
          "(sentence_index alignment + real comma-clause detection).\n")

    # find_formulaic_matches() self-test. Imported here, not at module
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
    formulaic_matches = find_formulaic_matches(formulaic_turns, BUNDLES)
    print(f"find_formulaic_matches() -> {len(formulaic_matches)} match(es):")
    for m in formulaic_matches:
        print(f"  {m}")
    assert any(m["candidate"] == "verschiedene Sachen" for m in formulaic_matches), (
        "expected the corpus-confirmed candidate to be found"
    )
    assert any(m["candidate"] == "aber" and "Ich koche" in m["sentence"] for m in formulaic_matches), (
        "expected the second candidate in the same sentence to also be found"
    )
    aber_second_sentence = [m for m in formulaic_matches if m["candidate"] == "aber" and "das stimmt" in m["sentence"]]
    assert len(aber_second_sentence) == 1, (
        f"expected sentence-initial capitalized 'Aber' to match and normalize to lowercase 'aber', got {aber_second_sentence}"
    )
    ja_matches = [m for m in formulaic_matches if m["candidate"] == "ja" and "das stimmt" in m["sentence"]]
    assert len(ja_matches) == 1, (
        f"expected repeated 'ja ja' in one sentence to dedup to a single match, got {ja_matches}"
    )
    print("\nfind_formulaic_matches() self-checks passed.\n")

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
