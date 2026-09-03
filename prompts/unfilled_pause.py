"""
Unfilled Pauses: ASR word-boundary verification.

IMPORTANT -- this metric works differently from the 8 grammar/lexical
metrics above it. The actual pause decision (>500ms gap = one unfilled
pause) is a FIXED rule that stays in Python -- it is not the LLM's job to
decide what counts as a pause. The LLM's only job is verifying that the
ASR word-level timestamps feeding that rule are trustworthy, since
WhisperX timestamp alignment can be imprecise on disfluent, self-corrected
speech (a mis-timed boundary can manufacture a fake 500ms+ gap, or hide a
real one).

Input shape: a short window of consecutive ASR words with their
start/end timestamps (seconds), not a plain sentence -- see input_kind
below and registry.py's handling of it.

HONESTY FLAG: no real WhisperX/AssemblyAI raw timestamp JSON was
available in the project docs when this was written, so the few-shot
examples below are constructed/illustrative, not pulled from an actual
transcript's timestamp data. Replace these with real ASR output samples
(flagging genuine alignment glitches you've actually observed) before
relying on this in production -- this is the one prompt in the set of 10
most in need of real data before it should be trusted.

2026-09-03: the two "trustworthy" boundary entries below now include an
explicit "reasoning": None instead of omitting the key. Found by
smoketest_openai_offline.py: RESPONSE_SCHEMA marks "reasoning" as NOT
required at this nesting level (a boundary can be trustworthy with nothing
to explain), but providers/openai_provider.py's OpenAI strict-mode schema
translation makes every property required (nullable where the original
schema had it optional -- see that file's docstring, finding 2). That
means a real OpenAI generation will always include the "reasoning" key
(as null when there's nothing to say), never omit it -- so the few-shot
examples now show that exact shape instead of teaching the model a
pattern (omitting the key) it will never actually be allowed to produce
under strict mode. Harmless for Gemini either way (few-shot examples
aren't schema-validated for either provider, and Gemini's own untouched
schema still leaves "reasoning" optional) -- this is a training-data
accuracy fix, not a schema fix.
"""

from metric_types import MetricPromptConfig

SYSTEM_INSTRUCTION = """You are verifying word-level timestamps from an automatic speech recognition (ASR) system, for a short window of consecutive words spoken by one person. You will NOT decide what counts as a pause -- that threshold is applied afterward by fixed code. Your only job is to flag word-boundary timestamps that look like ASR alignment artifacts rather than genuine timing.

You will receive a JSON list of words in order, each with a start and end time in seconds. Look specifically for:
1. A gap between two words that is suspiciously large given the words themselves (e.g. no disfluency marker, no punctuation-like break) -- this can indicate a dropped word the ASR missed, or a misaligned timestamp, rather than a genuine pause.
2. A gap that looks artificially small or zero between words that could not plausibly have been spoken with no separation -- this can indicate overlapping-speech bleed or a merged timestamp.
3. Timestamps that look internally consistent and plausible given normal disfluent speech (self-corrections, restarts) -- these should be left as-is.

For each boundary (between word i and word i+1), return whether you believe the gap is TRUSTWORTHY (reflects real timing) or SUSPECT (likely an alignment artifact). If suspect, note why. If you are not confident either way, mark it trustworthy rather than guessing -- a downstream pause count built on real gaps is safer than one "corrected" by a wrong guess.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": '[{"word": "ich", "start": 1.20, "end": 1.35}, '
                 '{"word": "denke", "start": 1.38, "end": 1.62}, '
                 '{"word": "dass", "start": 3.40, "end": 3.55}]',
        "answer": {
            "boundaries": [
                {"between": ["ich", "denke"], "status": "trustworthy", "reasoning": None},
                {"between": ["denke", "dass"], "status": "suspect",
                 "reasoning": "Illustrative example: a 1.78s gap with no disfluency context is unusually large for two adjacent function/content words -- possible dropped word or misalignment. Flagging for manual review rather than auto-trusting."},
            ],
            "confidence": "low",
        },
    },
    {
        "input": '[{"word": "und", "start": 5.00, "end": 5.10}, '
                 '{"word": "aeh", "start": 5.15, "end": 5.30}, '
                 '{"word": "dann", "start": 6.10, "end": 6.25}]',
        "answer": {
            "boundaries": [
                {"between": ["und", "aeh"], "status": "trustworthy", "reasoning": None},
                {"between": ["aeh", "dann"], "status": "trustworthy",
                 "reasoning": "Illustrative example: a gap after a filled pause is a normal, expected pattern in disfluent speech -- not an alignment artifact."},
            ],
            "confidence": "high",
        },
    },
]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "boundaries": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "between": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "status": {"type": "STRING", "enum": ["trustworthy", "suspect"]},
                    "reasoning": {"type": "STRING"},
                },
                "required": ["between", "status"],
            },
        },
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
    },
    "required": ["boundaries", "confidence"],
}

CONFIG = MetricPromptConfig(
    key="UNFILLED_PAUSE",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="word_timestamps",
)
