"""
Unfilled Pauses: ASR word-boundary gap detection.

2026-09-06, per Dan's explicit instruction: UNFILLED_PAUSE moves from
LLM-assisted to a fully deterministic direct computation, same treatment
FORMULAIC (2026-09-05) and FILLED_PAUSE (2026-09-06) got. This file now
holds ONLY the fixed threshold constant -- SYSTEM_INSTRUCTION/
FEW_SHOT_EXAMPLES/RESPONSE_SCHEMA/CONFIG are gone. The actual detection
logic now lives in direct_computation.compute_unfilled_pause(), which
imports PAUSE_THRESHOLD_SECONDS from here.

HONEST TRADEOFF THIS ACCEPTS (stated plainly, not glossed over): the old
LLM version's real job was NOT deciding the pause threshold -- that was
already a fixed >500ms rule in the SYSTEM_INSTRUCTION, never actually the
model's call to make. Its job was verifying that the ASR word-level
timestamps feeding that rule were trustworthy, since a mis-timed boundary
could manufacture a fake large gap (masking a dropped word) or hide a real
one (masking overlapping-speech bleed). That verification step is GONE as
of this direct-computation version: a gap over the threshold is now always
counted as an unfilled pause, with no check for whether the underlying
timestamps look like an ASR alignment artifact. Concretely, this direct
version WILL overcount if WhisperX drops a word and leaves a large gap
behind, and WILL undercount if two words' timestamps overlap or collapse
when they shouldn't. Revisit with a real judgment layer (LLM-assisted or
a heuristic on timestamp-confidence scores, if AssemblyAI/WhisperX expose
one) if this accuracy ceiling turns out to matter for a real report,
rather than for a first working direct-computation pass.

This mirrors FILLED_PAUSE's own honesty flag about ASR silently dropping
filler tokens from the transcript text -- same shape of tradeoff (a real,
stated accuracy ceiling accepted for a simpler, cheaper, zero-LLM-call
computation), different underlying cause.
"""

# Same threshold the old LLM prompt's SYSTEM_INSTRUCTION already applied as
# a fixed rule, not a judgment call -- unchanged by this move to direct
# computation. A gap between two consecutive words (within the same turn --
# see direct_computation.compute_unfilled_pause()'s turn-boundary handling)
# strictly greater than this counts as one unfilled pause.
PAUSE_THRESHOLD_SECONDS = 0.5
