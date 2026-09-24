"""
GDD-6: Verb-Governed Case -- a verb's object in the wrong case.

~23+ of 618 real GDD errors in docs/all_grammar_errors_master.json (4%,
keyword count, likely undercounted), plus "vertraue das" in the
2026-09-24 test session. Added 2026-09-24.

Boundary with the other GDD checks: GDD-6 owns objects whose gender is
RIGHT but whose case is wrong for the verb. A determiner with the wrong
gender is GDD-3; objects of prepositions stay with GDD-1/GDD-2; choosing
the wrong preposition after a verb is LPF.

Few-shot inputs are verbatim real transcript errors from
all_grammar_errors_master.json (none invented). `corrected` narrows the
corpus correction to this one error type. None come from
benny_italki6_9_8_2026 (the current test session).
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import MetricPromptConfig

DATIVE_VERBS = [
    "helfen", "danken", "gefallen", "gehören", "vertrauen", "folgen", "antworten",
    "zustimmen", "gratulieren", "passen", "schmecken", "zuhören", "fehlen", "glauben (a person)",
]
DATIVE_RECIPIENT_VERBS = [
    "geben", "schenken", "zeigen", "erklären", "erzählen", "sagen", "empfehlen",
    "vorschlagen", "(eine Frage) stellen",
]

SYSTEM_INSTRUCTION = f"""You are checking ONE German sentence for a single, narrow grammar error type: an object of a VERB in the wrong CASE.

Flag:
- verbs that take a dative object, used with an accusative/nominative one: {", ".join(DATIVE_VERBS)} (e.g. "ich stimme das zu" -> "dem")
- the person receiving something with these verbs must be dative: {", ".join(DATIVE_RECIPIENT_VERBS)} (e.g. "die Leute empfehlen" -> "den Leuten empfehlen")
- a direct object that must be accusative but is nominative -- visible on masculine forms: den/einen/keinen/meinen, not der/ein/kein/mein. This includes "es gibt" + accusative.

Only flag when the determiner or pronoun has the RIGHT gender but the WRONG case.

NOT this check -- handled by other checks, do not flag here:
- a determiner with the wrong gender for its noun
- objects of prepositions (mit, von, in, auf, für, ...)
- choosing the wrong preposition after a verb
- adjective endings and singular/plural errors

If the sentence is too fragmented to tell which verb governs the object, set confidence to "low" and error to false -- do not guess.

Always also return `corrected`: the FULL sentence, rewritten with ONLY the object's case fixed (the determiner/pronoun form, change nothing else -- not word order, not other words, not restarts) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "dann würde ich das zustimmen",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'zustimmen' takes a dative object: 'dem zustimmen', not the accusative 'das'.",
                   "corrected": "dann würde ich dem zustimmen"},
    },
    {
        "input": "sollte die Leute nicht, ungesund essen vorschlagen oder, empfehlen.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "With 'vorschlagen'/'empfehlen' the people receiving the advice are dative: 'den Leuten', not 'die Leute'.",
                   "corrected": "sollte den Leuten nicht, ungesund essen vorschlagen oder, empfehlen."},
    },
    {
        "input": "habe ich es kein— kein Druck",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Druck' is masculine and the direct object of 'haben', so it must be accusative: 'keinen Druck', not the nominative 'kein'.",
                   "corrected": "habe ich es kein— keinen Druck"},
    },
    {
        "input": "Trotzdem kann ich nicht mit die Leute sprechen.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'die Leute' is governed by the preposition 'mit', not by the verb. Case after prepositions is a different check.",
                   "corrected": "Trotzdem kann ich nicht mit die Leute sprechen."},
    },
    {
        "input": "Ist das die Wort",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "The problem is gender ('Wort' is neuter), not the case the verb requires. A different check handles gender.",
                   "corrected": "Ist das die Wort"},
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
    key="GDD-6",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
    construct="ACCURACY",
    metric_key="gdd6_count",
    unit="count",
    formula="count of flagged sentences this session (placeholder -- not an error rate)",
    report1_tag="Case: object of the verb",
    report1_order=10,
)
