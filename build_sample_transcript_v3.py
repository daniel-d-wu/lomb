"""
Builds sample_transcript_assemblyai_v3.json -- v2 plus two additions
specifically so FORMULAIC and FILLED_PAUSE (both newly wired into
pipeline.py on 2026-09-03) have something in a transcript actually worth
scanning, not just empty windows/zero candidates. Everything else (all of
v2's turns, timing rules, the two UNFILLED_PAUSE anomalies) is unchanged --
see build_sample_transcript_v2.py's own docstring for that content.

The two additions, and exactly how honest each one is:

1. FORMULAIC: "Ich koche gern verschiedene Sachen." inserted as a new A
   turn. NOT invented -- this is dan_error_analysis_master_v3.md's own
   Idio-008 "Unidiomatic" example verbatim (confirmed by reading that doc
   directly, not assumed from memory), the same real corpus quote
   prompts/formulaic.py's own few-shot examples already use. Gives the
   candidate scan a second hit beyond v2's existing "schon" occurrence in
   "Ich suche schon seit ein paar Monaten...".

2. FILLED_PAUSE: "Ich, aeh, ich weiss nicht genau, aehm, wie ich das sagen
   soll." inserted as a new A turn, with "aeh" and "aehm," as their own
   literal word tokens. THIS PART IS INVENTED, not corpus-sourced --
   flagging that plainly rather than letting it pass as real: real ASR
   transcripts in this project's corpus do not preserve filler-sound
   tokens as their own words (that's the whole reason FILLED_PAUSE's
   redefinition accepts an accuracy ceiling -- see
   prompts/filled_pause.py's docstring), so there is no real corpus
   sentence to pull this from the way there was for FORMULAIC. This is a
   synthetic stand-in built to exercise the plumbing (does a planted
   hesitation token survive from transcript JSON -> Word objects ->
   word-timestamp windows -> the LLM request unchanged) and, if/when this
   runs against a real API key outside this sandbox, to see whether the
   redefined prompt actually flags what it's designed to flag. It is not
   evidence about how often real learners produce filler sounds, and
   should never be read as such.
"""

import json

from build_sample_transcript_v2 import TURNS as V2_TURNS

WORD_MS = 380
GAP_WITHIN_MS = 0
GAP_BETWEEN_MS = 400
BIG_GAP_MS = 1800
OVERLAP_MS = -100

# Insert the two new A turns after the existing "Trotzdem kann ich nicht
# mit die Leute sprechen." / tutor-reply pair (index 4 in V2_TURNS), so
# they sit early in the transcript rather than only at the very end.
_INSERT_AFTER_TEXT = "Wieso? Was ist das Problem?"

NEW_TURNS = [
    ("A", "Ich koche gern verschiedene Sachen.", None),
    ("B", "Oh, was kochst du am liebsten?", None),
    ("A", "Ich, aeh, ich weiss nicht genau, aehm, wie ich das sagen soll.", None),
]

TURNS = []
for turn in V2_TURNS:
    TURNS.append(turn)
    if turn[1] == _INSERT_AFTER_TEXT:
        TURNS.extend(NEW_TURNS)


def build():
    t = 400
    words_flat = []
    utterances = []

    for speaker, text, anomaly in TURNS:
        tokens = text.split(" ")
        utt_words = []
        utt_start = t
        for i, tok in enumerate(tokens):
            start = t
            end = t + WORD_MS
            utt_words.append({
                "text": tok,
                "start": start,
                "end": end,
                "confidence": 0.94,
                "speaker": None,
            })
            t = end + GAP_WITHIN_MS

            if anomaly is not None:
                kind, after_idx = anomaly
                if i == after_idx:
                    if kind == "big_gap":
                        t = end + BIG_GAP_MS
                    elif kind == "overlap":
                        t = end + OVERLAP_MS

        utt_end = utt_words[-1]["end"]
        utterances.append({
            "speaker": speaker,
            "text": text,
            "start": utt_start,
            "end": utt_end,
            "confidence": 0.93,
            "words": utt_words,
        })
        words_flat.extend(utt_words)
        t = utt_end + GAP_BETWEEN_MS

    full_text = " ".join(u["text"] for u in utterances)
    audio_duration_s = round(words_flat[-1]["end"] / 1000, 1)

    doc = {
        "_NOTE": (
            "CONSTRUCTED sample v3, not a real AssemblyAI API response -- no audio "
            "file exists behind it. Same base as sample_transcript_assemblyai_v2.json "
            "(JSON shape verified against AssemblyAI's real API reference; speaker A's "
            "sentence content is real, pulled from dan_error_analysis_master_v3.md and "
            "claude/all_grammar_errors_master.json; speaker B is invented filler "
            "dialogue), PLUS two additions built specifically to exercise FORMULAIC and "
            "FILLED_PAUSE now that both are wired into pipeline.py (2026-09-03): "
            "'Ich koche gern verschiedene Sachen.' is REAL -- dan_error_analysis_"
            "master_v3.md's own Idio-008 example verbatim, not invented. 'Ich, aeh, ich "
            "weiss nicht genau, aehm, wie ich das sagen soll.' is INVENTED, not "
            "corpus-sourced -- this project's real ASR transcripts do not preserve "
            "filler-sound tokens as their own words (see prompts/filled_pause.py's "
            "docstring for why FILLED_PAUSE accepts an accuracy ceiling because of "
            "exactly this), so there is no real corpus line to pull a filler example "
            "from the way there was for FORMULAIC. This sentence exists to exercise the "
            "plumbing end to end and, only if run against a real API key outside this "
            "sandbox, to see whether the prompt actually flags what it's designed to "
            "flag -- it is not evidence about real learner filler-sound frequency. See "
            "build_sample_transcript_v3.py's own module docstring for the full "
            "reasoning behind both additions."
        ),
        "id": "sample-transcript-v3-not-a-real-assemblyai-job",
        "status": "completed",
        "language_code": "de",
        "audio_url": "N/A -- synthetic sample, no real audio file behind this",
        "audio_duration": audio_duration_s,
        "confidence": 0.93,
        "text": full_text,
        "words": words_flat,
        "utterances": utterances,
        "speaker_labels": True,
    }
    return doc


if __name__ == "__main__":
    doc = build()
    out_path = "sample_transcript_assemblyai_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    a_turns = [u for u in doc["utterances"] if u["speaker"] == "A"]
    b_turns = [u for u in doc["utterances"] if u["speaker"] == "B"]
    print(f"Wrote {out_path}")
    print(f"Total turns: {len(doc['utterances'])}  (A: {len(a_turns)}, B: {len(b_turns)})")
    print(f"audio_duration: {doc['audio_duration']}s")
