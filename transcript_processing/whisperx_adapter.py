"""
WhisperX + pyannote transcript JSON -> speaker_filter.Turn/Word.

The whisperX-side counterpart to assemblyai_adapter.py -- same job
(produce the list[Turn] everything downstream of speaker_filter.py
expects), structurally different problem. AssemblyAI's diarized response
already segments speech into single-speaker "utterances" (Turn objects
fall out directly, see that adapter's own docstring); whisperx's
assign_word_speakers() step instead tags EACH WORD with a speaker guess
and leaves them inside ASR "segments" that are NOT guaranteed to be
single-speaker. Confirmed against a real 44-segment whisperX+pyannote
output (whisperX_batch_job.ipynb's own product, lilli_tandem1's 5-minute
clip): segment 9's own top-level "speaker" field (a majority-vote label
whisperx.assign_word_speakers() writes per segment) says SPEAKER_00,
while every one of that segment's words is individually tagged
SPEAKER_01. Trusting the segment-level field -- the naive reading of this
data -- would have mis-attributed that entire segment's words to the
wrong speaker. Turns here are therefore built by regrouping consecutive
SAME-WORD-SPEAKER runs across the flattened word stream, never from
whisperX's own segment boundaries or segment-level speaker field.

Two more real, confirmed (not guessed) problems this closes:

1. Missing per-word speaker tags. pyannote's diarization turns don't
   always cover every ASR word -- confirmed on the same sample file
   (1 of 401 words, ~0.25%, had no "speaker" key at all). A word with no
   speaker tag is assigned to whichever speaker the immediately preceding
   word belongs to (forward-fill): pyannote's diarization turns are
   contiguous stretches of time, so a short assignment gap almost always
   falls inside the surrounding speaker's own turn. A leading run of
   words with no prior speaker at all (nothing to forward-fill from, only
   possible right at the very start of a file) is dropped, not guessed --
   see _flatten_and_fill()'s dropped_unattributed count.

2. Prompt-leak hallucination. A whisper `initial_prompt` used to steer
   transcription (whisperX_batch_job.ipynb sets one, instructing
   verbatim/error-preserving German transcription) can itself get
   transcribed back out as fake leading segment(s) -- confirmed by exact
   textual overlap between that notebook's own initial_prompt string and
   segments 0-1 of the same sample file, both landing before real speech
   actually starts (~23s in). filter_prompt_leak() below drops any
   segment within leak_cutoff_seconds of file start whose text closely
   matches a caller-supplied known_prompt_leak string (pass the exact
   initial_prompt text a given job actually used -- this file never
   hardcodes it, since a future job may use a different prompt, or none).
   Without known_prompt_leak, a weaker fallback heuristic (near-duplicate
   consecutive early segments -- the actual shape the leak took in the
   one real file this was built against, split across two segments that
   closely resemble each other) is used instead. Flagged plainly as
   best-effort, not a guarantee, same as every other heuristic already in
   this codebase (map_words_to_sentences()'s is_clause_end, to_sentences()'s
   naive .?! split) gets flagged rather than glossed over.

Timestamps: whisperX's are already in SECONDS (unlike AssemblyAI's
milliseconds, which assemblyai_adapter.py has to divide by 1000) -- no
unit conversion needed here. Confirmed against real output: word start/
end values for a 5-minute clip sit in the tens of seconds, not the tens
of thousands a millisecond reading would produce.

Field names: whisperX's word objects use "word" for the token string
(assemblyai_adapter.py's own docstring flags the AssemblyAI equivalent as
"text" -- a real difference, not a typo in either file).

Not yet verified against a live whisperX+pyannote run from this adapter's
own build environment (this was built and tested entirely against
already-produced output files staged from Dan's machine -- see this
module's __main__ block for exactly what that self-test does and
doesn't prove). A fresh live run, to confirm the adapter also handles
whatever a NEW run of whisperX_batch_job.ipynb's hardened successor
produces, is Dan's own machine's job (GPU, real audio) -- not this
sandbox's.
"""

import difflib
import re
import sys
from pathlib import Path as _Path

# repo root on sys.path -- makes the dotted import below resolve whether
# this file is run directly or imported as
# transcript_processing.whisperx_adapter by another script (e.g.
# pipeline/pipeline.py), same bootstrap assemblyai_adapter.py uses.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from transcript_processing.speaker_filter import Turn, Word

# Segments starting after this many seconds are never treated as a
# prompt-leak candidate, regardless of text similarity -- a real repeated
# phrase spoken deep into a conversation (an actual formulaic expression,
# or two speakers genuinely echoing each other) must never be filtered
# just because it happens to resemble the prompt. Every leaked-prompt
# segment observed so far has landed before real speech starts (~23s in
# the one sample file this was built against), well inside this default.
DEFAULT_LEAK_CUTOFF_SECONDS = 15.0

# Below this similarity ratio, a segment is NOT considered a prompt-leak
# match against an ADJACENT SEGMENT (fallback path only -- see
# CONTAINMENT_THRESHOLD for the known-prompt path's own, differently-
# computed threshold). Chosen loosely rather than tightly: a real leaked
# segment can be a truncated or garbled reading of the prompt, not a
# verbatim copy -- confirmed on the sample file, where segment 0's text
# is a strict prefix of segment 1's, itself only a partial reading of the
# actual (longer) prompt text.
SIMILARITY_THRESHOLD = 0.5

# Below this containment ratio (see _contains_ratio()), a segment is NOT
# considered a prompt-leak match against a caller-supplied known_prompt_
# leak string. This is a DIFFERENT metric from SIMILARITY_THRESHOLD above
# (fraction of the SEGMENT's own text found as one contiguous run inside
# the prompt, not a whole-string ratio) -- whole-string
# difflib.SequenceMatcher.ratio() badly under-scores a short segment
# against a much longer prompt even when the segment is an exact verbatim
# substring of it (ratio is diluted by all the prompt content the segment
# doesn't need to match), which is exactly the known-prompt-leak shape.
# Confirmed on the sample file: the two real leaked segments score a
# perfect 1.0 here (exact substrings of the real initial_prompt); the
# closest a genuine short utterance ("Mm-hm.") came was 0.33.
CONTAINMENT_THRESHOLD = 0.6


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # autojunk=False: difflib's default autojunk heuristic treats any
    # character that recurs "too often" in a sequence longer than 200
    # items as junk and excludes it from matches -- for ordinary text
    # (spaces alone trigger this on anything long enough) that silently
    # breaks real substring matches. Confirmed directly on this file's
    # own known_prompt_leak case: with autojunk left at its default,
    # find_longest_match() on an exact, verbatim, real substring returned
    # a near-zero match; autojunk=False fixed it. Always disabled here,
    # for both this function and _contains_ratio() below.
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _contains_ratio(needle: str, haystack: str) -> float:
    """Fraction of `needle`'s own length covered by the single longest
    contiguous run of `needle` found anywhere inside `haystack`. Used for
    known_prompt_leak matching specifically -- see CONTAINMENT_THRESHOLD
    for why this, not _similar(), is the right metric when needle (a
    segment) is much shorter than haystack (the whole prompt)."""
    if not needle or not haystack:
        return 0.0
    match = difflib.SequenceMatcher(None, needle, haystack, autojunk=False).find_longest_match(
        0, len(needle), 0, len(haystack)
    )
    return match.size / len(needle)


def filter_prompt_leak(
    segments: list[dict],
    known_prompt_leak: str | None = None,
    leak_cutoff_seconds: float = DEFAULT_LEAK_CUTOFF_SECONDS,
) -> tuple[list[dict], list[dict]]:
    """Returns (kept_segments, dropped_segments), same order as input.
    Only segments starting before leak_cutoff_seconds are ever candidates
    for dropping -- see DEFAULT_LEAK_CUTOFF_SECONDS.

    known_prompt_leak given: drop a candidate segment if its normalized
    text scores >= CONTAINMENT_THRESHOLD against the (also normalized)
    known_prompt_leak string, via _contains_ratio() -- see that
    function's docstring for why this is a containment check, not a
    whole-string similarity ratio. This is the reliable path -- use it
    whenever the actual initial_prompt text is available.

    known_prompt_leak absent (fallback, best-effort): drop a candidate
    segment if it closely resembles an ADJACENT candidate segment (either
    side) -- the observed real-world pattern (see module docstring) is a
    leaked prompt split across 2+ consecutive early segments that
    strongly resemble each other, distinct from real early speech, which
    doesn't repeat itself near-verbatim segment-to-segment. Weaker than
    the known-prompt path, can both over- and under-match; documented as
    a fallback, not a substitute for passing the real prompt text.
    """
    candidates = [s for s in segments if s.get("start", 0.0) < leak_cutoff_seconds]
    rest = [s for s in segments if s.get("start", 0.0) >= leak_cutoff_seconds]

    if known_prompt_leak:
        norm_prompt = _norm(known_prompt_leak)
        kept, dropped = [], []
        for seg in candidates:
            if _contains_ratio(_norm(seg.get("text", "")), norm_prompt) >= CONTAINMENT_THRESHOLD:
                dropped.append(seg)
            else:
                kept.append(seg)
        return kept + rest, dropped

    texts = [_norm(s.get("text", "")) for s in candidates]
    drop_flags = [False] * len(candidates)
    for i in range(len(candidates)):
        neighbors = []
        if i > 0:
            neighbors.append(texts[i - 1])
        if i + 1 < len(candidates):
            neighbors.append(texts[i + 1])
        if any(_similar(texts[i], n) >= SIMILARITY_THRESHOLD for n in neighbors):
            drop_flags[i] = True

    kept = [seg for seg, flag in zip(candidates, drop_flags) if not flag]
    dropped = [seg for seg, flag in zip(candidates, drop_flags) if flag]
    return kept + rest, dropped


# Whisper hallucination guardrail (2026-09-24). A real session produced a
# 69x "äh" loop that inflated FILLED_PAUSE ~2.7x. Across 83 real transcripts
# (5,178 segments), real large-v3 speech never repeated one token more than
# 9x in a row, so 15 leaves a wide margin: only unmistakable loops are
# dropped. Whisper's own confidence can't be used instead -- it scored the
# loop MORE confident (avg_logprob -0.119) than normal speech (-0.203).
REPETITION_RUN_THRESHOLD = 15


def _longest_repeat_run(segment: dict) -> int:
    best = run = 0
    prev = None
    for w in segment.get("words", []):
        token = re.sub(r"[^\w]", "", w.get("word", "").lower())
        if not token:
            continue
        run = run + 1 if token == prev else 1
        prev = token
        best = max(best, run)
    return best


def filter_repetition_loops(
    segments: list[dict], threshold: int = REPETITION_RUN_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Drop whole segments where one token repeats >= threshold times in a
    row (a Whisper decoding loop, not speech). Returns (kept, dropped);
    the caller decides how to report dropped. Conservative by design: a
    real learner's "äh, äh, äh, äh" (4x) is kept."""
    kept, dropped = [], []
    for s in segments:
        (dropped if _longest_repeat_run(s) >= threshold else kept).append(s)
    return kept, dropped


def _flatten_and_fill(segments: list[dict]) -> tuple[list[tuple[str, float, float, str]], int]:
    """Flattens every kept segment's words into one chronological list of
    (text, start, end, speaker) tuples, forward-filling any word missing
    its own "speaker" key from the immediately preceding word. Returns
    (filled_words, dropped_unattributed_count) -- the count is words at
    the very start of the transcript with no prior speaker to inherit
    from, which are dropped rather than guessed at.

    Words missing "start"/"end" entirely (whisperX can leave these off a
    very-low-confidence alignment) are skipped outright -- useless to any
    downstream timing-based metric (UNFILLED_PAUSE, WPM) even if kept
    with a fabricated time, and speaker-continuity for forward-fill
    purposes doesn't depend on keeping them.
    """
    flat_words = []
    for seg in segments:
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            flat_words.append((w.get("word", ""), w["start"], w["end"], w.get("speaker")))

    filled = []
    last_speaker = None
    dropped_unattributed = 0
    for text, start, end, speaker in flat_words:
        if speaker is None:
            speaker = last_speaker
        if speaker is None:
            dropped_unattributed += 1
            continue
        last_speaker = speaker
        filled.append((text, start, end, speaker))
    return filled, dropped_unattributed


def _turn_from_group(speaker_id: str, words: list[Word]) -> Turn:
    return Turn(
        speaker_id=speaker_id,
        text=" ".join(w.text for w in words),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )


def from_whisperx_transcript(
    data: dict,
    known_prompt_leak: str | None = None,
    leak_cutoff_seconds: float = DEFAULT_LEAK_CUTOFF_SECONDS,
) -> list[Turn]:
    """Parse one whisperx.align() + assign_word_speakers() result (already
    JSON-decoded -- the exact {"segments": [...], "word_segments": [...]}
    shape whisperX_batch_job.ipynb writes to disk) into the list[Turn]
    the rest of this codebase expects. Output shape is identical to
    assemblyai_adapter.from_assemblyai_transcript()'s -- pipeline.
    run_pipeline_from_turns() (or anything else downstream of
    speaker_filter.py) doesn't need to know or care which engine produced
    a given transcript. See this module's own docstring for the three
    real problems (segment-vs-word speaker disagreement, missing per-word
    speaker tags, prompt-leak hallucination) this function's job is to
    absorb before anything reaches Turn objects.

    known_prompt_leak / leak_cutoff_seconds: passed straight through to
    filter_prompt_leak() -- see that function's docstring.

    Raises ValueError if `data` has no "segments" key (doesn't look like
    a whisperX result), or if every word ends up unattributable to any
    speaker (e.g. diarization never actually ran) -- same "don't silently
    return an empty list" reasoning assemblyai_adapter.py's own missing-
    utterances case uses; a caller needs to be able to tell "zero turns,
    genuinely" apart from "this input was unusable."
    """
    segments = data.get("segments")
    if not segments:
        raise ValueError(
            "No 'segments' in this transcript -- this doesn't look like a "
            "whisperX result (or the job produced zero segments)."
        )

    kept_segments, _dropped_leak_segments = filter_prompt_leak(
        segments, known_prompt_leak=known_prompt_leak, leak_cutoff_seconds=leak_cutoff_seconds
    )
    kept_segments, _dropped_loop_segments = filter_repetition_loops(kept_segments)
    # A dropped segment is a hole, not silence: end the turn there, so the
    # hole never counts as speaking time or as one giant unfilled pause.
    holes = [(s.get("start", 0.0), s.get("end", 0.0)) for s in _dropped_leak_segments + _dropped_loop_segments]

    filled, _dropped_unattributed = _flatten_and_fill(kept_segments)
    if not filled:
        raise ValueError(
            "No word in this transcript could be attributed to any speaker "
            "-- diarization may not have run, or every word landed outside "
            "every diarization turn. Nothing to build Turns from."
        )

    turns: list[Turn] = []
    group_words: list[Word] = []
    group_speaker = filled[0][3]
    for text, start, end, speaker in filled:
        crosses_hole = bool(group_words) and any(
            h_start < start and h_end > group_words[-1].end for h_start, h_end in holes
        )
        if group_words and (speaker != group_speaker or crosses_hole):
            turns.append(_turn_from_group(group_speaker, group_words))
            group_words = []
        group_speaker = speaker
        group_words.append(Word(text=text, start=start, end=end))
    if group_words:
        turns.append(_turn_from_group(group_speaker, group_words))

    return turns


if __name__ == "__main__":
    import json

    from transcript_processing.speaker_filter import speakers_present, filter_to_target_speaker, to_sentences

    # Real whisperX+pyannote output, staged from Dan's own already-run
    # batch job (whisperX_batch_job.ipynb) -- not a synthetic fixture.
    # Path is relative to this file's own new home in a real checkout
    # (transcript_processing/); adjust if run from elsewhere.
    default_path = _Path(__file__).resolve().parent.parent / "data" / "whisperx_sample.json"
    transcript_path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_segment_count = len(data["segments"])
    kept, dropped = filter_prompt_leak(data["segments"])
    print(f"Raw segments: {raw_segment_count}")
    print(f"Prompt-leak filter (fallback heuristic, no known_prompt_leak given): "
          f"dropped {len(dropped)}, kept {len(kept)}")
    for seg in dropped:
        print(f"  DROPPED  start={seg['start']:.2f}  text={seg['text']!r}")

    turns = from_whisperx_transcript(data)
    print(f"\nParsed {len(turns)} turns from {len(kept)} kept segments.")
    print(f"Speakers detected: {sorted(speakers_present(turns))}")

    for speaker in sorted(speakers_present(turns)):
        filtered = filter_to_target_speaker(turns, speaker)
        sentences = to_sentences(filtered)
        total_words = sum(len(t.words) for t in filtered)
        print(f"\nSpeaker {speaker}: {len(filtered)} turn(s), {total_words} word(s), "
              f"{len(sentences)} sentence(s)")
        for s in sentences[:3]:
            print(f"  {s!r}")
        if len(sentences) > 3:
            print(f"  ... ({len(sentences) - 3} more)")

    # Sanity checks against what was actually confirmed by hand-inspecting
    # this real file while building this adapter -- not just "did it run."
    assert len(dropped) >= 2, (
        f"expected the fallback heuristic to catch the known 2-segment prompt "
        f"leak at the start of this file, dropped only {len(dropped)}"
    )
    all_words_have_positive_duration = all(
        w.end >= w.start for t in turns for w in t.words
    )
    assert all_words_have_positive_duration, "every Word must have end >= start"
    # Chronological ordering within each turn, and across turns overall --
    # a regrouping bug could silently shuffle words out of order.
    for t in turns:
        assert all(a.start <= b.start for a, b in zip(t.words, t.words[1:])), (
            f"words within one Turn must stay in chronological order, got {t}"
        )
    assert all(a.start <= b.start for a, b in zip(turns, turns[1:])), (
        "Turns must stay in chronological order"
    )
    print("\nAll whisperx_adapter.py self-checks passed.")
