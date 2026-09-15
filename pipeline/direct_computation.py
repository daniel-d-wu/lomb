"""
Direct-path Fluencemes: computed straight from transcript data, zero LLM
calls. FILLED_PAUSE and UNFILLED_PAUSE were moved off the LLM here on
2026-09-06, per Dan's explicit instruction -- same treatment FORMULAIC got
on 2026-09-05, but implemented as their own functions here rather than
inline in speaker_filter.py, since both need real per-word logic (a token
lookup, a gap-threshold check) beyond a single regex scan. WPM (Speed) was
added here 2026-09-07 -- the last of the 11 Fluencemes in Phase 1's scope,
per Dan's decision (9 LLM-assisted + FORMULAIC + WPM).

WPM is structurally different from the other three: its Information is
SESSION-LEVEL (word_count, duration_seconds), not one entry per occurrence
found -- see compute_wpm()'s own docstring for how that's still made to
fit this module's uniform {"input","skipped","output"} list shape.

WHY THIS FILE, AND WHY THIS SHAPE (matters for downstream storage, not
just code organization): every function here returns the exact same
{"input", "skipped", "output"} entry shape pipeline.py's LLM-backed
metrics and FORMULAIC's regex path already use (see pipeline.py's own
module docstring). That's deliberate, not incidental -- report.py,
pipeline.py's own self-tests, and eventually storage.py's write path all
consume "one metric_key -> list of these entries" without caring whether
an entry came from an LLM call or a direct computation. A Fluenceme
switching computation paths (as both of these just did) should never
require a shape change on the consuming side, only a different producer.

GRANULARITY, matching FORMULAIC's precedent: one entry per ACTUAL
OCCURRENCE FOUND (one flagged filler word, one flagged pause), not one
entry per word/boundary CHECKED. This is a deliberate change from the old
LLM versions, which returned one entry per word-timestamp WINDOW checked
(covering every boundary in that window, flagged or not). The old
window-level shape doesn't generalize past this rewrite -- there's no
window concept left once the computation is direct arithmetic over the
full word list, and a "no occurrence found" row would need to be one per
word-timestamp pair, which is a Fluency check, not a Fluenceme aggregate --
so switching to positive-occurrences-only keeps this module's shape
consistent with FORMULAIC's and with what information_items is actually
meant to hold (see lomb_metric_architecture_v1.md): a row per occurrence,
not a row per unit-of-input-considered.

sentence_indices / boundary_type on every occurrence here: both pull from
speaker_filter.map_words_to_sentences(), the 2026-09-06 addition that links
word-level timestamps to sentence-level position for the first time. Every
Fluenceme's occurrences -- LLM-judged or direct -- use this same
sentence_indices array field (see that function's own docstring and
lomb_metric_architecture_v1.md's schema section): a single-sentence LLM
check gets a 1-element array, GVT-1's window gets a 3-element array, and
these two get 1 element (a filler sitting in one sentence) or up to 2 (a
pause gap that crosses a sentence boundary). One field, one join pattern,
regardless of computation path -- that's the point.
"""

import sys
from pathlib import Path

# repo root on sys.path -- prompts/ and transcript_processing/ are both
# siblings of this file's own new directory (pipeline/) post-reorg.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.filled_pause import FILLER_TOKENS
from prompts.unfilled_pause import PAUSE_THRESHOLD_SECONDS
from transcript_processing.speaker_filter import Turn, map_words_to_sentences


def _boundary_type(prev_entry: dict | None, same_sentence: bool) -> str:
    """Shared classification logic for both functions below: given the
    map_words_to_sentences() entry for the word immediately BEFORE an
    occurrence (None if the occurrence is at the very start of the
    transcript), decide whether it sits at a sentence boundary, a clause
    boundary, or neither. `same_sentence` is only relevant to
    compute_unfilled_pause() (a gap can itself cross a sentence boundary
    even if prev_entry's own flags don't say so); compute_filled_pause()
    always passes True since a single word has no second side to check.
    """
    if prev_entry is None or prev_entry["is_sentence_end"] or not same_sentence:
        return "sentence"
    if prev_entry["is_clause_end"]:
        return "clause"
    return "none"


def compute_filled_pause(target_turns: list[Turn]) -> list[dict]:
    """Computes FILLED_PAUSE. Scans every word the target speaker said (in
    order, per turn) for an exact (case-insensitive, punctuation-stripped)
    match against FILLER_TOKENS. One occurrence per match -- see this
    module's docstring for why non-matches aren't reported at all.

    preceding_word/following_word are taken strictly within the same turn
    (None at a turn's own first/last word) -- never reaching into a
    different turn, for the same reason to_word_timestamp_windows() never
    chunks across a turn boundary: the words on the other side of a turn
    edge may have another speaker's entire turn sitting between them in
    real time, so treating them as textually "adjacent" would be wrong.
    """
    word_map = map_words_to_sentences(target_turns)
    occurrences = []
    flat_i = 0
    for turn in target_turns:
        words = turn.words
        n = len(words)
        for i, w in enumerate(words):
            cleaned = w.text.strip().rstrip(".,!?").lower()
            if cleaned in FILLER_TOKENS:
                entry = word_map[flat_i]
                preceding = words[i - 1] if i > 0 else None
                following = words[i + 1] if i + 1 < n else None
                prev_entry = word_map[flat_i - 1] if flat_i > 0 else None
                occurrences.append({
                    "input": f'Word: "{w.text}" (sentence {entry["sentence_index"]})',
                    "skipped": False,
                    "output": {
                        "filler": True,
                        "word": w.text,
                        "start": w.start,
                        "end": w.end,
                        "confidence": "high",
                        "reasoning": (
                            "Deterministic match against the fixed hesitation-token "
                            "list (prompts/filled_pause.FILLER_TOKENS) -- no LLM "
                            "judgment involved as of 2026-09-06."
                        ),
                        "sentence_indices": [entry["sentence_index"]],
                        "boundary_type": _boundary_type(prev_entry, same_sentence=True),
                        "preceding_word": (
                            {"word": preceding.text, "start": preceding.start, "end": preceding.end}
                            if preceding else None
                        ),
                        "following_word": (
                            {"word": following.text, "start": following.start, "end": following.end}
                            if following else None
                        ),
                    },
                })
            flat_i += 1
    return occurrences


def compute_unfilled_pause(target_turns: list[Turn]) -> list[dict]:
    """Computes UNFILLED_PAUSE. Walks every pair of CONSECUTIVE words within
    the same turn (never across a turn boundary -- see
    to_word_timestamp_windows()'s docstring for exactly why that matters:
    the gap between two turns can include another speaker's entire turn).
    A gap strictly greater than PAUSE_THRESHOLD_SECONDS is one occurrence.

    See prompts/unfilled_pause.py's module docstring for the accuracy
    tradeoff this direct version accepts (no more ASR-boundary-
    trustworthiness verification, unlike the LLM version this replaces).
    """
    word_map = map_words_to_sentences(target_turns)
    occurrences = []
    flat_i = 0
    for turn in target_turns:
        words = turn.words
        n = len(words)
        for i in range(n):
            if i < n - 1:
                gap = words[i + 1].start - words[i].end
                if gap > PAUSE_THRESHOLD_SECONDS:
                    entry_a = word_map[flat_i]
                    entry_b = word_map[flat_i + 1]
                    same_sentence = entry_a["sentence_index"] == entry_b["sentence_index"]
                    sentence_indices = sorted({entry_a["sentence_index"], entry_b["sentence_index"]})
                    occurrences.append({
                        "input": f'Gap: "{words[i].text}" -> "{words[i + 1].text}" ({gap:.2f}s)',
                        "skipped": False,
                        "output": {
                            "pause": True,
                            "gap_seconds": round(gap, 3),
                            "confidence": "high",
                            "reasoning": (
                                f"Deterministic: gap exceeds the fixed "
                                f"{PAUSE_THRESHOLD_SECONDS}s threshold "
                                "(prompts/unfilled_pause.PAUSE_THRESHOLD_SECONDS) -- "
                                "no ASR-boundary trustworthiness check performed "
                                "as of 2026-09-06 (see that module's docstring)."
                            ),
                            "sentence_indices": sentence_indices,
                            "boundary_type": _boundary_type(entry_a, same_sentence),
                            "preceding_word": {"word": words[i].text, "start": words[i].start, "end": words[i].end},
                            "following_word": {
                                "word": words[i + 1].text, "start": words[i + 1].start, "end": words[i + 1].end,
                            },
                        },
                    })
            flat_i += 1
    return occurrences


def compute_wpm(target_turns: list[Turn]) -> list[dict]:
    """Computes WPM (Speed). Unlike FORMULAIC/FILLED_PAUSE/UNFILLED_PAUSE
    above, WPM's Information is SESSION-LEVEL, not per-occurrence -- there
    is nothing to scan for, just two numbers to tally across the whole
    target-speaker turn set. Still returned as a one-entry list, in the
    same {"input","skipped","output"} shape every other metric in this
    module uses, so pipeline.py/report.py/storage.py-to-be can treat every
    metric_key uniformly (iterate results[key], one row per occurrence)
    without a special case for "this one is session-level" -- the only
    real differences are that this list always has exactly length 1, and
    its sentence_indices is always empty (there's no single sentence this
    Information belongs to -- it describes the whole session).

    word_count: total words the target speaker actually said this session
    (every turn's word list, summed).

    duration_seconds: SPEAKING time, not session span -- the sum of each
    of the target speaker's own turns' (end - start), not the gap between
    the first and last thing they ever said. filter_to_target_speaker()
    has already dropped every other speaker's turns before this function
    ever sees them, so summing turn durations naturally excludes time
    spent listening to someone else and excludes inter-turn silence -- but
    DOES include any pause WITHIN one of the target speaker's own turns.
    That inclusion is deliberate: a speech-tempo metric should reflect
    real speaking pace including the speaker's own hesitations, not an
    idealized pause-free rate.

    Per Dan's Information/Metric distinction (2026-09-05 terminology
    clarification): this function returns Information only -- word_count
    and duration_seconds, not a computed `wpm` number. The actual `wpm`
    Metric (word_count divided by duration in minutes) is a Metric
    Aggregator's job, which doesn't exist yet (see
    lomb_metric_architecture_v1.md). run_pipeline() in pipeline.py also
    computes a convenience `wpm` value at its return dict's top level, the
    same way it already does for structure_breadth_score -- useful for
    reading a result by eye before a real aggregator exists, not itself
    the stored Metric.
    """
    word_count = sum(len(turn.words) for turn in target_turns)
    duration_seconds = sum(turn.end - turn.start for turn in target_turns)
    return [{
        "input": "session",
        "skipped": False,
        "output": {
            "word_count": word_count,
            "duration_seconds": round(duration_seconds, 3),
            "confidence": "high",
            "reasoning": (
                "Deterministic: word_count is the total words the target speaker "
                "said this session; duration_seconds is the sum of their own "
                "turns' durations (speaking time, not session span or listening time)."
            ),
            "sentence_indices": [],
        },
    }]


if __name__ == "__main__":
    # Self-proof against a small synthetic turn with a planted filler, a
    # planted large gap (unfilled pause), and a planted near-zero gap
    # (should NOT be flagged) -- proves both functions fire correctly and
    # that sentence_indices/boundary_type come out right, not just that
    # the module imports cleanly.
    from speaker_filter import Word

    turns = [
        Turn("A", "Ich moechte, aehm, mehr ueben. Aber ich habe keine Zeit.", 0.0, 5.0, [
            Word("Ich", 0.00, 0.20), Word("moechte,", 0.25, 0.60),
            Word("aehm,", 0.65, 0.90),          # filler, right after a comma (clause boundary)
            Word("mehr", 2.70, 2.90),            # 1.80s gap before this -- unfilled pause, same sentence
            Word("ueben.", 2.95, 3.30),
            Word("Aber", 3.35, 3.55),            # new sentence starts here
            Word("ich", 3.58, 3.70), Word("habe", 3.72, 3.95),
            Word("keine", 3.98, 4.20), Word("Zeit.", 4.22, 4.50),
        ]),
    ]

    filled = compute_filled_pause(turns)
    print(f"compute_filled_pause() -> {len(filled)} occurrence(s):")
    for f in filled:
        print(f"  {f['input']}  boundary_type={f['output']['boundary_type']}  "
              f"sentence_indices={f['output']['sentence_indices']}")
    assert len(filled) == 1, f"expected exactly 1 filler ('aehm,'), got {len(filled)}"
    assert filled[0]["output"]["boundary_type"] == "clause", (
        "'aehm,' follows 'moechte,' which ends in a comma -- should be a clause boundary"
    )
    assert filled[0]["output"]["sentence_indices"] == [0]

    unfilled = compute_unfilled_pause(turns)
    print(f"\ncompute_unfilled_pause() -> {len(unfilled)} occurrence(s):")
    for u in unfilled:
        print(f"  {u['input']}  boundary_type={u['output']['boundary_type']}  "
              f"sentence_indices={u['output']['sentence_indices']}")
    assert len(unfilled) == 1, (
        f"expected exactly 1 unfilled pause (the 1.80s gap before 'mehr'), got {len(unfilled)}"
    )
    assert unfilled[0]["output"]["gap_seconds"] == 1.8
    assert unfilled[0]["output"]["sentence_indices"] == [0], (
        "the flagged gap sits between 'aehm,' and 'mehr', both in sentence 0"
    )
    # The gap between "ueben." (sentence end) and "Aber" (0.05s) is well
    # under threshold, so it must NOT appear -- confirms the threshold
    # check, not just that SOME gap gets reported.
    assert not any(u["output"]["preceding_word"]["word"] == "ueben." for u in unfilled), (
        "the tiny 0.05s gap after 'ueben.' must not be flagged -- it's under threshold"
    )

    wpm_result = compute_wpm(turns)
    print(f"\ncompute_wpm() -> {wpm_result}")
    assert len(wpm_result) == 1, "WPM is session-level -- always exactly one Information entry"
    wpm_out = wpm_result[0]["output"]
    assert wpm_out["sentence_indices"] == [], "WPM has no sentence position -- sentence_indices must be empty"
    assert wpm_out["word_count"] == 10, f"expected 10 words in the planted turn, got {wpm_out['word_count']}"
    assert wpm_out["duration_seconds"] == 5.0, (
        f"expected turn.end(5.0) - turn.start(0.0) = 5.0s, got {wpm_out['duration_seconds']}"
    )
    wpm_value = wpm_out["word_count"] / (wpm_out["duration_seconds"] / 60)
    print(f"  word_count={wpm_out['word_count']}  duration_seconds={wpm_out['duration_seconds']}  "
          f"-> convenience wpm={wpm_value:.1f}")

    print("\nAll direct_computation.py self-checks passed.")
