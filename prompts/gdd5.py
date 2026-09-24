"""
GDD-5: Plural / Number -- wrong noun number or plural form.

~75 of 618 real GDD errors (12%) in docs/all_grammar_errors_master.json.
Added 2026-09-24.

Scope is the NOUN's number: singular where plural is required (after
plural quantifiers/numbers), plural after a singular determiner, a wrong
plural form, viel/viele confusion with mass vs count nouns. The dative
plural -n after GDD-1/GDD-2 prepositions stays with those checks.
Subject-verb agreement is a verb error, not this check.

Few-shot inputs are verbatim real transcript errors from
all_grammar_errors_master.json (none invented). `corrected` narrows the
corpus correction to this one error type. None come from
benny_italki6_9_8_2026 (the current test session).
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import MetricPromptConfig
from prompts.gdd1 import ALWAYS_DATIVE_PREPOSITIONS
from prompts.gdd2 import WECHSELPRAEPOSITIONEN

SYSTEM_INSTRUCTION = f"""You are checking ONE German sentence for a single, narrow grammar error type: a noun in the wrong NUMBER (singular vs plural) or with a wrong plural form.

Flag:
- a singular noun where the phrase requires plural -- after numbers above one, "ein paar", "viele", "mehrere", "beide", or a plural determiner (e.g. "ein paar deutsche Lied" -> "Lieder")
- a plural noun after a singular determiner such as ein/eine, jede/jeder/jedes, dieser/diese/dieses used for one thing (e.g. "eine bessere Optionen" -> "Option")
- a wrongly formed plural (e.g. an English -s plural on a German noun that forms its plural differently)
- viel vs viele: "viel" with mass nouns (viel Energie), "viele" with countable plurals (viele Faktoren)

NOT this check -- handled by other checks, do not flag here:
- noun phrases governed by these prepositions (including the dative plural -n there): {", ".join(ALWAYS_DATIVE_PREPOSITIONS + WECHSELPRAEPOSITIONEN)}
- article gender, adjective endings, case
- subject-verb agreement (a verb conjugation error)

If singular and plural would both be natural in context, or the sentence is too fragmented to tell, set confidence to "low" and error to false -- do not guess.

Always also return `corrected`: the FULL sentence, rewritten with ONLY the noun's number/plural form fixed (and, only if unavoidable, the determiner directly attached to it) if error is true, or the input sentence completely unchanged if error is false. Change nothing else -- not word order, not other words, not restarts. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Bei YouTube gibt es ein paar deutsche Lied",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'ein paar' requires a plural noun: 'ein paar deutsche Lieder'.",
                   "corrected": "Bei YouTube gibt es ein paar deutsche Lieder"},
    },
    {
        "input": "bei mir ist es Ganz einfach, viele Repetition",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'viele' requires a plural noun: 'viele Repetitionen'.",
                   "corrected": "bei mir ist es Ganz einfach, viele Repetitionen"},
    },
    {
        "input": "das ist für mich persönlich, eine bessere Optionen",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "The singular article 'eine' needs a singular noun: 'eine bessere Option'.",
                   "corrected": "das ist für mich persönlich, eine bessere Option"},
    },
    {
        "input": "Trotzdem kann ich nicht mit die Leute sprechen.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'Leute' is correctly plural. The missing dative plural (den Leuten) after 'mit' is a case error that a different check handles.",
                   "corrected": "Trotzdem kann ich nicht mit die Leute sprechen."},
    },
    {
        "input": "Ich suche schon seit ein paar Monaten nach einem Tandempartner.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'ein paar Monaten' is correctly plural and 'einem Tandempartner' correctly singular.",
                   "corrected": "Ich suche schon seit ein paar Monaten nach einem Tandempartner."},
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
    key="GDD-5",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
    construct="ACCURACY",
    metric_key="gdd5_count",
    unit="count",
    formula="count of flagged sentences this session (placeholder -- not an error rate)",
    report1_tag="Plural / number",
    report1_order=9,
)
