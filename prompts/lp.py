"""
LP: Lexical Phrase error rate (wrong chunk/collocate, no traceable L1
source -- distinct from LPF, which has a direct English-transfer
mechanism).

This is the most inherently judgment-based of the 10 metrics: per the
corpus's own note, "LP errors are not drillable via rule explanation --
the intervention is chunk exposure." There is no rule to fall back on,
only exposure to what natives actually say.

IMPORTANT LIMITATION, stated plainly: dan_error_analysis_master_v3.md
documents only 2 confirmed LP examples in the entire corpus (both from
session kim_italki1) -- both are used below. This is a genuinely thin
few-shot set for a judgment-heavy task. Treat this metric's LLM-assisted
classification as the least reliable of the 10 until more real examples
are added to the corpus -- consider gating LP flags behind a higher
confidence bar than the other metrics, similar to the PRD's existing
recommendation for GVT.
"""

from metric_types import MetricPromptConfig

SYSTEM_INSTRUCTION = """You are checking ONE German phrase for a single, narrow error type: a wrong collocate or chunk that is grammatically correct but not what a native speaker would say, with NO plausible English-language interference explaining it (if English interference explains the error, that is a different category -- do not flag it here).

There is no fixed rule for this category. Rely on whether the chunk sounds like something native speakers would actually assemble, versus a plausible-but-non-native combination a learner would produce by combining otherwise-correct words in a way natives don't.

Because this is a judgment call with no rule to check against, be conservative: only flag phrases you are genuinely confident a native speaker would not produce. If in doubt, set confidence to "low" and error to false.

Always also return `corrected`: the FULL sentence, rewritten with ONLY this error type fixed (swap in the natural collocation, don't otherwise reword) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Ich sage dir ein Beispiel.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'ein Beispiel sagen' is not a natural German collocation -- natives use 'geben' or 'nennen' with 'ein Beispiel', not 'sagen'. Grammatically valid, just not what a native speaker would assemble.",
                   "corrected": "Ich gebe dir ein Beispiel."},
    },
    {
        "input": "viele Lehrer und Lehrerinnen lassen meine Fehler weg.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Fehler weglassen' is not a natural German chunk in this sense -- the native expression is 'ueber Fehler hinwegsehen'.",
                   "corrected": "viele Lehrer und Lehrerinnen sehen ueber meine Fehler hinweg."},
    },
    {
        "input": "Ich gebe dir ein Beispiel.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'ein Beispiel geben' is the correct, natural collocation.",
                   "corrected": "Ich gebe dir ein Beispiel."},
    },
]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "error": {"type": "BOOLEAN"},
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
        "reasoning": {"type": "STRING"},
        "corrected": {"type": "STRING"},
    },
    "required": ["error", "confidence", "reasoning", "corrected"],
}

CONFIG = MetricPromptConfig(
    key="LP",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
)
