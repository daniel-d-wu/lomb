"""
AssemblyAI transcript JSON -> speaker_filter.Turn/Word.

The missing piece speaker_filter.py's own docstrings flagged but didn't
have: something that actually produces Turn/Word objects from a real
diarization provider's output, rather than assuming the shape. Two
concrete mismatches this closes, both confirmed against AssemblyAI's real,
current API reference (assemblyai.com/docs/api-reference/transcripts/get),
not guessed:

1. Units: speaker_filter.py's Word.start/end are documented as "seconds",
   with an explicit "TBD at Milestone 3 against real WhisperX output" note
   -- this never got resolved before now. AssemblyAI's real timestamps are
   MILLISECONDS. Every conversion in this file divides by 1000 at the
   boundary, once, here -- nothing downstream of this adapter needs to
   know AssemblyAI ever used milliseconds.

2. Field names: AssemblyAI's word objects use "text" for the word string,
   not "word" -- prompts/unfilled_pause.py's own few-shot examples
   (predating this adapter) use {"word": ..., "start": ..., "end": ...},
   which is this codebase's own internal convention for what gets sent to
   the LLM (built by speaker_filter.to_word_timestamp_windows(), not by
   this file) -- NOT a claim about AssemblyAI's raw field name. This
   adapter reads AssemblyAI's real "text" key when building Word objects;
   the "word" key downstream is a separate, internal naming choice that
   stays exactly as it already was.

AssemblyAI's diarized response already segments speech into "utterances"
-- {speaker, text, start, end, confidence, words: [...]} -- which maps
directly onto Turn's own definition ("one continuous stretch of speech
from one speaker"), so this adapter builds Turn objects straight from
utterances rather than needing to reconstruct turns from the flat
top-level "words" array itself. That flat array exists in a real response
too (all words across all speakers, unsegmented) but isn't used here --
utterances already provide everything Turn needs, more directly.

Deliberately NOT verified against a live AssemblyAI job: no real
audio file or live API call was available while building this (same
limitation prompts/unfilled_pause.py already flagged for its own few-shot
examples). sample_transcript_assemblyai.json's own header explains exactly
which parts of that sample are shape-verified against real docs versus
constructed for testing -- read that before assuming this adapter has been
proven against a genuine AssemblyAI job, not just against the documented
schema.
"""

from speaker_filter import Turn, Word


def from_assemblyai_transcript(data: dict) -> list[Turn]:
    """Parse one AssemblyAI GET /v2/transcript/{id} response (as a dict,
    already JSON-decoded) into the list[Turn] the rest of this codebase
    (speaker_filter.py, and everything downstream of it) expects.

    Raises ValueError if the response isn't actually diarized (no
    "utterances" key, or speaker_labels wasn't requested) -- this
    function has nothing useful to build without per-speaker segments,
    and a silent empty-list return would be indistinguishable from "zero
    turns detected" (which the caller needs to treat very differently
    from "this transcript can't be turn-segmented at all").
    """
    utterances = data.get("utterances")
    if not utterances:
        raise ValueError(
            "No 'utterances' in this transcript -- either speaker_labels "
            "wasn't set to true when this job was submitted (see "
            "scanner_main.py's submit_transcription()), or the job hasn't "
            "finished diarizing yet. This adapter has nothing to build "
            "turns from without it."
        )

    turns = []
    for utt in utterances:
        words = [
            Word(text=w["text"], start=w["start"] / 1000.0, end=w["end"] / 1000.0)
            for w in utt.get("words", [])
        ]
        turns.append(
            Turn(
                speaker_id=utt["speaker"],
                text=utt["text"],
                start=utt["start"] / 1000.0,
                end=utt["end"] / 1000.0,
                words=words,
            )
        )
    return turns


if __name__ == "__main__":
    import json

    from speaker_filter import speakers_present, filter_to_target_speaker, to_sentences

    with open("sample_transcript_assemblyai.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    turns = from_assemblyai_transcript(data)
    print(f"Parsed {len(turns)} turns from the sample transcript.")
    print(f"Speakers detected: {sorted(speakers_present(turns))}")

    for speaker in sorted(speakers_present(turns)):
        filtered = filter_to_target_speaker(turns, speaker)
        sentences = to_sentences(filtered)
        print(f"\nSpeaker {speaker}: {len(filtered)} turn(s), {len(sentences)} sentence(s)")
        for s in sentences:
            print(f"  {s!r}")

    # Spot-check the units conversion actually happened -- the deliberate
    # large gap in the sample (~1.8s, inserted in ms as 1800) should come
    # out as ~1.8 in seconds after this adapter runs, not ~1800.
    #
    # IMPORTANT: gaps must be measured WITHIN each turn's own words, not
    # across turns concatenated together. Speaker A's turns are
    # interleaved with speaker B's (real conversation structure), so the
    # gap between the end of one A-turn and the start of the next A-turn
    # spans however long B was talking in between -- a large, meaningless
    # number that has nothing to do with the deliberate intra-utterance
    # pause this check is trying to isolate. First cut of this script got
    # this wrong (flattened all of A's words across turns before diffing,
    # picked up a 3.84s inter-turn span instead of the real 1.8s
    # intra-turn gap) -- fixed by computing gaps per turn.
    target = filter_to_target_speaker(turns, "A")
    max_gap = max(
        w2.start - w1.end
        for t in target
        for w1, w2 in zip(t.words, t.words[1:])
    )
    print(f"\nLargest intra-turn inter-word gap for speaker A: {max_gap:.3f}s "
          f"(expect ~1.8s, not ~1800 -- confirms ms->s conversion happened)")
    assert 1.5 < max_gap < 2.1, f"expected the deliberate ~1.8s gap, got {max_gap}"
    print("Units conversion confirmed correct.")
