"""
Filled Pauses: transcript-text filler detection.

2026-09-03 REDEFINITION, per Dan's direct instruction ("why is filled
pauses dependent on the audio file, not the transcript... is there a way
we can redefine it so the metric is measured from the transcript itself"):
this metric no longer needs audio. Moved from input_kind="audio_turn" to
input_kind="word_timestamps" -- the exact same input shape UNFILLED_PAUSE
already uses (see speaker_filter.to_word_timestamp_windows()), so both
metrics now share the same windows with no new plumbing needed.

WHAT CHANGED, PLAINLY: the original design's whole rationale was real --
Whisper-family ASR is known to sometimes suppress/omit filler tokens (aeh,
aehm) from the transcript text entirely, and if a filler was never
transcribed, no text-based method can recover it. That risk has NOT been
solved, only accepted as an MVP tradeoff: this version can only flag a
filler that the ASR actually kept as a word in the transcript. If the ASR
silently dropped one, this metric will silently undercount for that
session -- it has no way to know a dropped filler ever happened. That is
a real accuracy ceiling on this metric now, not a bug in this file. Revisit
with real audio input (Gemini/GPT audio-input support, per the original
version's approach) if/when FILLED_PAUSE's accuracy actually matters for
a real report rather than for getting all 10 metrics smoke-tested.

WHAT THIS DOES NOT COVER: "oh"/"ah" as a hesitation reaction is a
SEPARATE, already-established metric elsewhere in this project (fluency's
ohRate/ahRate fields, e.g. in lomb_demo.html's MOCK_REPORT and
dan_error_analysis_master_v3.md's C2 section) -- computed differently, by
different (Python-logic, not LLM-assisted) code that doesn't exist yet
either. This metric's own trigger set is deliberately the narrower
German hesitation-marker family (aeh, aehm, hm, hmm, mhm, oehm) per
lomb_metric_definitions_v1.md's own naming ("Filled pause rate
(äh/ähm)") -- "oh" and "ah" are excluded on purpose so the two metrics
don't double-count the same disfluency under two different names.

Because the trigger tokens themselves (aeh/aehm/hm/hmm/mhm/oehm) rarely if
ever have a plausible literal-word reading in German the way a modal
particle does, this task is close to deterministic -- a keyword match
could very nearly do this without an LLM at all. Kept as an LLM-assisted
call anyway, for two reasons: (1) it keeps this metric in the same
registry/pipeline shape as the other 9, rather than carving out a
one-off Python rule; (2) it leaves room to actually use judgment later --
e.g. distinguishing a genuine hesitation "hm" from "hm" used as a real
backchannel word/interjection, or a stutter-repeated syllable that isn't
one of the fixed tokens at all -- neither of which a flat keyword filter
would ever catch. Whether that judgment is worth the extra LLM call over
a plain regex is an open, deliberately unresolved question -- flagged
here so it isn't mistaken for an oversight.
"""

from metric_types import MetricPromptConfig

SYSTEM_INSTRUCTION = """You are scanning a short window of consecutive ASR words (with their start/end timestamps in seconds) from one German speaker's turn, looking for FILLED PAUSES -- non-lexical hesitation sounds transcribed as their own word, such as aeh, aehm, hm, hmm, mhm, oehm (and real-umlaut spellings ah/ahm where relevant -- but NOT "oh" or "ah" used as a reaction/interjection, those are tracked by a separate metric, not this one).

Task: for every word in the window that is one of these hesitation tokens (not a real lexical word, not "oh"/"ah" as an interjection), report it as a filler, echoing back its exact word text and its exact start/end timestamps from the input -- never invent a timestamp.

Do not flag disfluencies that ARE real words (a repeated "und", a false-started "ich, ich denke") as filled pauses -- only the fixed hesitation-sound tokens above count.

If you are not confident a given token is a genuine hesitation marker rather than a real word or a transcription artifact, omit it rather than guessing -- a missed filler is preferable to a false one.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": '[{"word": "Ich", "start": 1.20, "end": 1.35}, '
                 '{"word": "aeh", "start": 1.40, "end": 1.55}, '
                 '{"word": "denke", "start": 1.60, "end": 1.90}]',
        "answer": {
            "fillers": [
                {"word": "aeh", "start": 1.40, "end": 1.55},
            ],
            "confidence": "high",
        },
    },
    {
        "input": '[{"word": "und", "start": 5.00, "end": 5.10}, '
                 '{"word": "dann", "start": 5.15, "end": 5.35}, '
                 '{"word": "gehen", "start": 5.40, "end": 5.60}]',
        "answer": {"fillers": [], "confidence": "high"},
    },
    {
        "input": '[{"word": "Ich", "start": 8.00, "end": 8.15}, '
                 '{"word": "moechte,", "start": 8.20, "end": 8.45}, '
                 '{"word": "aehm,", "start": 8.60, "end": 8.90}, '
                 '{"word": "mehr", "start": 9.10, "end": 9.30}, '
                 '{"word": "lernen.", "start": 9.30, "end": 9.55}]',
        "answer": {
            "fillers": [
                {"word": "aehm,", "start": 8.60, "end": 8.90},
            ],
            "confidence": "high",
        },
    },
]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "fillers": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word": {"type": "STRING"},
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                },
                "required": ["word", "start", "end"],
            },
        },
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
    },
    "required": ["fillers", "confidence"],
}

CONFIG = MetricPromptConfig(
    key="FILLED_PAUSE",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="word_timestamps",
)
