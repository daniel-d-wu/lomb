"""
Builds sample_transcript_assemblyai_v2.json -- a longer companion to the
original 6-sentence sample, specifically to exercise as many of the 8
now-automatable metrics as possible (GVT-1 included, now that
speaker_filter.to_sentence_windows() exists).

Same honesty rules as the original sample: speaker A's (Dan's) sentences
are pulled verbatim from dan_error_analysis_master_v3.md and
claude/all_grammar_errors_master.json (real transcript errors/corrections
from real sessions), never invented. Speaker B (tutor/partner) is
plausible invented filler -- only A is analyzed. Word-level timestamps are
synthetic (evenly spaced, back-to-back within an utterance, ~400ms gap
between utterances) with two DELIBERATE anomalies inserted for
UNFILLED_PAUSE to have something to actually catch:
  1. a 1.8s gap inside "Ich lebe in die USA." (same technique/position as
     the original sample, reused deliberately for continuity)
  2. a -100ms overlap inside "dann wir sehen, wie lange wir auf Deutsch
     reden." -- the original sample only ever tested the "suspiciously
     large gap" failure mode; UNFILLED_PAUSE's own system instruction
     also asks it to catch "artificially small or... overlapping" gaps,
     which nothing has tested until now.

Metric coverage by design (turn index -> what it's there for):
  GDD-1   "Trotzdem kann ich nicht mit die Leute sprechen." (error)
          "Ich suche schon seit ein paar Monaten nach einem Tandempartner." (negative control, no error)
  GDD-2   "Wie viele Tage war ich in die Krankenhaus?" (error)
          "Ich lebe in die USA." (error, + UNFILLED_PAUSE big-gap anomaly)
  GVT-1   "und dann kommt letztes Jahr." + "Ich mache ein Urlaub in Hamburg
          und nach Koeln." -- split across TWO turns with a B turn between
          them, to specifically exercise to_sentence_windows() bridging a
          turn boundary (this is the exact real corpus example that
          motivated building that function)
          "fruher, als ich Kind war, spiele ich gern Basketball und
          schwimme." -- single-sentence drift, baseline case
          "weil ich damals kein Deutsch spreche, wuerde ein paar Frauen
          mit mir nicht reden." -- dual-tagged: also a weil_clause for
          STRUCTURE_BREADTH
  GVT-2   "dann wir sehen, wie lange wir auf Deutsch reden." (error, +
          UNFILLED_PAUSE overlap anomaly)
          "manchmal gibt es Problem, manchmal es ist ein Vorteil." (error)
  LPF     "Ich suche fuer Muttersprachler, um mein Deutsch zu verbessern." (error)
          "Ich freue mich fuer diese Reise, dass ich bald gehe." (error, +
          dass_clause for STRUCTURE_BREADTH)
  LP      "Ich sage dir ein Beispiel." (error -- one of only 2 confirmed
          LP examples in the whole corpus, per prompts/lp.py's own
          reliability caveat)
  STRUCTURE_BREADTH  konjunktiv_ii, dass_clause, weil_clause (3 distinct
          labels reachable, one more than the original sample's 2)
  UNFILLED_PAUSE  big-gap anomaly + overlap anomaly + otherwise-uniform
          timing elsewhere (should read as trustworthy)

Still NOT exercisable, same as before -- no code change fixes these:
  FORMULAIC    no reference word list exists in the project
  FILLED_PAUSE needs real audio bytes, not just a transcript
"""

import json

WORD_MS = 380          # duration of each synthetic word
GAP_WITHIN_MS = 0       # normal gap between words in the same utterance
GAP_BETWEEN_MS = 400    # normal gap between utterances
BIG_GAP_MS = 1800       # deliberate UNFILLED_PAUSE "suspiciously large" anomaly
OVERLAP_MS = -100       # deliberate UNFILLED_PAUSE "suspiciously small/overlapping" anomaly

# (speaker, text, anomaly) -- anomaly is None, or ("big_gap", after_word_index)
# or ("overlap", after_word_index), where after_word_index is the index of
# the word the anomaly gap follows (0-based, within that utterance's own
# word list).
TURNS = [
    ("B", "Was hast du diese Woche auf Deutsch gemacht?", None),
    ("A", "Ich suche schon seit ein paar Monaten nach einem Tandempartner.", None),
    ("B", "Ah interessant, erzaehlen Sie mehr.", None),
    ("A", "Trotzdem kann ich nicht mit die Leute sprechen.", None),
    ("B", "Wieso? Was ist das Problem?", None),
    ("A", "Ich sage dir ein Beispiel.", None),
    ("B", "Okay, verstehe.", None),
    ("A", "Wie viele Tage war ich in die Krankenhaus?", None),
    ("B", "Oh nein, warst du krank?", None),
    ("A", "Ich lebe in die USA.", ("big_gap", 1)),  # gap after "lebe", before "in"
    ("B", "Und wie war das Gespraech mit dem Partner?", None),
    ("A", "und dann kommt letztes Jahr.", None),
    ("B", "Wirklich? Was ist passiert?", None),
    ("A", "Ich mache ein Urlaub in Hamburg und nach Koeln.", None),
    ("B", "Klingt schoen!", None),
    ("A", "fruher, als ich Kind war, spiele ich gern Basketball und schwimme.", None),
    ("B", "Das ist toll.", None),
    ("A", "weil ich damals kein Deutsch spreche, wuerde ein paar Frauen mit mir nicht reden.", None),
    ("B", "Okay, gut.", None),
    ("A", "dann wir sehen, wie lange wir auf Deutsch reden.", ("overlap", 1)),  # "wir" / "sehen," overlap
    ("B", "Klingt gut.", None),
    ("A", "manchmal gibt es Problem, manchmal es ist ein Vorteil.", None),
    ("B", "Verstehe.", None),
    ("A", "Ich suche fuer Muttersprachler, um mein Deutsch zu verbessern.", None),
    ("B", "Verstehe.", None),
    ("A", "Ich freue mich fuer diese Reise, dass ich bald gehe.", None),
    ("B", "Das klingt aufregend!", None),
    ("A", "Das waere schoen, wenn ich mehr Zeit haette zum Ueben.", None),
]


def build():
    t = 400  # ms, matches original sample's opening offset
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
            "CONSTRUCTED sample v2, not a real AssemblyAI API response -- no audio "
            "file exists behind it. Same two facts as the original "
            "sample_transcript_assemblyai.json, true independently: (1) JSON SHAPE "
            "verified against AssemblyAI's real API reference "
            "(assemblyai.com/docs/api-reference/transcripts/get) -- ms units, "
            "text/start/end/confidence/speaker on words. (2) Speaker A's SENTENCE "
            "CONTENT is real -- pulled verbatim from dan_error_analysis_master_v3.md "
            "and claude/all_grammar_errors_master.json (real errors/corrections from "
            "leonie_tandem1/2/6, alicia_italki4, and kim_italki1). Speaker B is "
            "invented filler dialogue, since only A is analyzed. Built specifically "
            "to exercise GVT-1 after speaker_filter.to_sentence_windows() was added "
            "on 2026-09-03 -- see that function's docstring and this script's own "
            "module docstring for exactly which real error maps to which metric, "
            "and the two deliberate ASR-timestamp anomalies planted for "
            "UNFILLED_PAUSE (one large gap, one overlap) versus the otherwise-"
            "uniform synthetic word timing everywhere else. FORMULAIC and "
            "FILLED_PAUSE still cannot be exercised by any transcript -- both are "
            "blocked by missing infrastructure (a reference word list; real audio "
            "bytes), not by anything a transcript's content could fix."
        ),
        "id": "sample-transcript-v2-not-a-real-assemblyai-job",
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
    out_path = "sample_transcript_assemblyai_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    a_turns = [u for u in doc["utterances"] if u["speaker"] == "A"]
    b_turns = [u for u in doc["utterances"] if u["speaker"] == "B"]
    print(f"Wrote {out_path}")
    print(f"Total turns: {len(doc['utterances'])}  (A: {len(a_turns)}, B: {len(b_turns)})")
    print(f"audio_duration: {doc['audio_duration']}s")
