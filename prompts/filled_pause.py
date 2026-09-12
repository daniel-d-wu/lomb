"""
Filled Pauses: transcript-text filler detection.

2026-09-06, per Dan's explicit instruction: FILLED_PAUSE moves from an
LLM-assisted metric to a fully deterministic direct computation, same
treatment FORMULAIC got on 2026-09-05. This file now holds ONLY the fixed
token list -- SYSTEM_INSTRUCTION/FEW_SHOT_EXAMPLES/RESPONSE_SCHEMA/CONFIG
are gone, there's no LLM call left to configure. The actual detection logic
now lives in direct_computation.compute_filled_pause(), which imports
FILLER_TOKENS from here (same "reference data survives, the LLM scaffolding
around it doesn't" pattern as prompts/formulaic.py's BUNDLES).

Why this was safe to do (the reasoning already flagged in this file before
today, now acted on): the trigger tokens below rarely if ever have a
plausible literal-word reading in German, so matching them was already
close to deterministic. The two reasons this file's earlier version gave
for keeping an LLM call anyway -- consistency with the other registry
metrics, and "room to use judgment later" -- were judged not worth an LLM
call per word-window once FORMULAIC had already made the same tradeoff
for a genuinely more ambiguous case (BUNDLES entries CAN have a literal
reading; these tokens essentially never do).

2026-09-03 REDEFINITION (unchanged reasoning, still relevant): this metric
runs off the transcript text, not audio. Whisper-family ASR is known to
sometimes suppress/omit filler tokens (aeh, aehm) from the transcript text
entirely -- if a filler was never transcribed, no text-based method (LLM
or direct) can recover it. That is a real accuracy ceiling this direct
version inherits unchanged from the LLM version, not something this
change makes worse.

WHAT THIS DOES NOT COVER: "oh"/"ah" as a hesitation reaction is a
SEPARATE, already-established metric elsewhere in this project (fluency's
ohRate/ahRate fields) -- computed differently, by different (Python-logic)
code that doesn't exist yet either. FILLER_TOKENS below is deliberately the
narrower German hesitation-marker family (aeh/aehm/hm/hmm/mhm/oehm) per
lomb_metric_definitions_v1.md's own naming ("Filled pause rate
(äh/ähm)") -- "oh" and "ah" are excluded on purpose so the two metrics
don't double-count the same disfluency under two different names.
"""

# Exact-match only (not substring), case-insensitive, after stripping
# trailing punctuation (ASR transcripts often attach a comma or period to
# the token, e.g. "aehm,"). Real-umlaut spellings included since some
# ASR/transcript sources use them instead of the ae/oe digraphs.
FILLER_TOKENS = frozenset({
    "aeh", "ah", "aehm", "ahm", "ähm", "äh",
    "hm", "hmm", "mhm", "oehm", "öhm",
})
