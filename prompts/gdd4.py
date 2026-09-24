"""
GDD-4: Adjective Endings -- wrong adjective declension.

Second-largest uncovered error type in Dan's corpus: ~105 of 618 real GDD
errors (17%) in docs/all_grammar_errors_master.json. Added 2026-09-24.

Judged against the CORRECT gender/case/number of the noun phrase: if only
the article is wrong (GDD-3) and the adjective ending would be right with
the correct article, that is not a GDD-4 error. Noun phrases after
GDD-1/GDD-2 prepositions stay with those checks (they already cover the
adjective's declension there).

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

SYSTEM_INSTRUCTION = f"""You are checking ONE German sentence for a single, narrow grammar error type: a wrong ADJECTIVE ENDING.

Rules:
- An attributive adjective (before a noun) takes the ending required by the noun phrase's gender, case and number AND by what precedes it: weak endings after der-words (der, dies-, jed-, welch-), mixed endings after ein-words (ein, kein, possessives), strong endings when there is no determiner. E.g. masculine nominative "ein guter Punkt", neuter accusative "ein altes Auto", plural after "die": "die wichtigen Methoden".
- A predicate adjective (after sein, werden, bleiben) and an adjective used as an adverb take NO ending: "weil man dick ist", not "dicke ist".

Judge the ending against the CORRECT gender, case and number of the noun phrase. If only the article is wrong and the adjective's ending would be correct with the right article (e.g. "ein andere Blockade" -> "eine andere Blockade"), do NOT flag here -- that is an article/gender error handled elsewhere.

NOT this check -- handled by other checks, do not flag here:
- noun phrases governed by these prepositions: {", ".join(ALWAYS_DATIVE_PREPOSITIONS + WECHSELPRAEPOSITIONEN)}
- wrong article gender, wrong case of the article, singular/plural errors of the noun itself

If the sentence is too fragmented to tell which noun the adjective belongs to, set confidence to "low" and error to false -- do not guess.

Always also return `corrected`: the FULL sentence, rewritten with ONLY the adjective ending fixed (change nothing else -- not articles, not word order, not restarts) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Das ist ein richtig gut Punkt",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "Masculine nominative after 'ein' takes the mixed ending -er: 'ein richtig guter Punkt'.",
                   "corrected": "Das ist ein richtig guter Punkt"},
    },
    {
        "input": "die meisten Leute ein, alte Auto gekauft.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Auto' is neuter; accusative after 'ein' takes -es: 'ein altes Auto'.",
                   "corrected": "die meisten Leute ein, altes Auto gekauft."},
    },
    {
        "input": "Ich weiß nur die wichtige Methoden, einen Satz zu, zu sprechen.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "Plural after the definite article 'die' takes the weak ending -en: 'die wichtigen Methoden'.",
                   "corrected": "Ich weiß nur die wichtigen Methoden, einen Satz zu, zu sprechen."},
    },
    {
        "input": "Und jetzt Trump will ein, ein andere Blockade machen.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "With the correct feminine article this is 'eine andere Blockade', where 'andere' is already right. The error is the article's gender, which a different check handles.",
                   "corrected": "Und jetzt Trump will ein, ein andere Blockade machen."},
    },
    {
        "input": "das ist für mich persönlich, eine bessere Optionen",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'eine bessere' has the right ending for a feminine singular noun. The problem is the plural 'Optionen' after the singular 'eine', which a different check handles.",
                   "corrected": "das ist für mich persönlich, eine bessere Optionen"},
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
    key="GDD-4",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
    construct="ACCURACY",
    metric_key="gdd4_count",
    unit="count",
    formula="count of flagged sentences this session (placeholder -- not an error rate)",
    report1_tag="Adjective ending",
    report1_order=8,
)
