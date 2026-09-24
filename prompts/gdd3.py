"""
GDD-3: Noun Gender -- wrong article/determiner gender for the noun.

The single largest error type in Dan's corpus: ~218 of 618 real GDD errors
(35%) in docs/all_grammar_errors_master.json are gender errors, none of
which GDD-1/GDD-2 can see (they only check case after specific
prepositions). Added 2026-09-24.

Boundary with the other GDD checks: GDD-3 owns anything where the
determiner shows the WRONG GENDER. Right gender but wrong case for the verb
is GDD-6; noun phrases after GDD-1/GDD-2 prepositions stay with those.

Few-shot inputs are verbatim real transcript errors from
all_grammar_errors_master.json (none invented). `corrected` narrows the
corpus correction to this one error type, per the house convention.
None come from benny_italki6_9_8_2026 (the current test session).
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import MetricPromptConfig
from prompts.gdd1 import ALWAYS_DATIVE_PREPOSITIONS
from prompts.gdd2 import WECHSELPRAEPOSITIONEN

SYSTEM_INSTRUCTION = f"""You are checking ONE German sentence for a single, narrow grammar error type: a determiner or pronoun that shows the wrong grammatical GENDER for its noun.

Determiners: definite and indefinite articles (der/die/das, ein/eine), kein-, dies-, jed-, welch-, and possessives (mein, dein, sein, ihr, unser, euer). A gender error is a form that signals a different gender than the noun actually has -- e.g. "das Wort" (neuter) said as "die Wort", "der Kopf" (masculine) said as "deine Kopf". Also flag a pronoun referring back to a specific noun with the wrong gender (e.g. "der Tee ... sie" instead of "er").

NOT this check -- handled by other checks, do not flag here:
- noun phrases governed by these prepositions: {", ".join(ALWAYS_DATIVE_PREPOSITIONS + WECHSELPRAEPOSITIONEN)}
- the right gender but the wrong CASE for what the verb requires (e.g. masculine "kein Druck" where the verb needs "keinen Druck")
- adjective endings on their own, and singular/plural errors
- plural noun phrases (German plural articles carry no gender)

English loanwords and anglicisms (e.g. Feature, Podcast, Leasing, Proof of Concept) often have unsettled gender. Unless the noun's German gender is standard and clear, set confidence to "low" and error to false.

If the sentence is too fragmented to tell which noun a determiner belongs to, set confidence to "low" and error to false -- do not guess.

Always also return `corrected`: the FULL sentence, rewritten with ONLY this error type fixed -- replace the wrongly-gendered determiner or pronoun with the correctly-gendered form in the case the sentence needs, and change nothing else (not word order, not other words, not restarts) -- if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Ist das die Wort",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Wort' is neuter (das Wort) -- 'die Wort' uses the feminine article.",
                   "corrected": "Ist das das Wort"},
    },
    {
        "input": "Deine Kopf ist ausgebrannt?",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Kopf' is masculine (der Kopf) -- the possessive must be 'Dein', not the feminine 'Deine'.",
                   "corrected": "Dein Kopf ist ausgebrannt?"},
    },
    {
        "input": "habe ich noch nicht diese Wort gehört",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Wort' is neuter -- the demonstrative must be 'dieses Wort', not the feminine 'diese Wort'.",
                   "corrected": "habe ich noch nicht dieses Wort gehört"},
    },
    {
        "input": "Trotzdem kann ich nicht mit die Leute sprechen.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'die Leute' is plural, so no gender is signalled; the problem here is the case after 'mit', which a different check handles.",
                   "corrected": "Trotzdem kann ich nicht mit die Leute sprechen."},
    },
    {
        "input": "Es ist nur einfach ein Ei, das man vom Supermarkt kaufen kann.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'Ei' is neuter: 'ein Ei' and the relative pronoun 'das' both match.",
                   "corrected": "Es ist nur einfach ein Ei, das man vom Supermarkt kaufen kann."},
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
    key="GDD-3",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
    construct="ACCURACY",
    metric_key="gdd3_count",
    unit="count",
    formula="count of flagged sentences this session (placeholder -- not an error rate)",
    report1_tag="Gender: wrong article for the noun",
    report1_order=7,
)
