"""
GVT-1: Tense Drift and Error.

Past narrative frame set up correctly, then a subsequent clause reverts to
present tense -- most common after als/weil/dass or in open narrative.
Requires tracking the established tense frame across clauses, which is why
this stays LLM-assisted even after the 2026-08-28 override pass: it's not
a local, single-clause pattern.

Known false-positive risk (documented in lomb_cooccurrence_findings.md's
methodological notes on the regex-based GVT detector used for the
co-occurrence research): legitimate historical-present narration -- where
a speaker deliberately narrates consistently in present tense throughout,
never establishing a past frame to drift from -- must NOT be flagged.

4 of the 5 examples are real transcript errors from
dan_error_analysis_master_v3.md (GVT error pattern 1). The historical-
present contrast example is illustrative (constructed to teach the
negative case) since no validated real example of that specific pattern
was available in the corpus at the time this was written -- replace it
with a real one if/when the corpus documents one.
"""

from metric_types import MetricPromptConfig

SYSTEM_INSTRUCTION = """You are checking a short sequence of German clauses (in the order they were spoken) for ONE error type: tense drift -- a past narrative frame established in one clause that incorrectly reverts to present tense in a later clause.

A past frame can be established either by a past-tense verb (habe gemacht, war, kam) or by a temporal marker that clearly signals past time (frueher, damals, letztes Jahr, als ich Kind war, vorher). If a later clause in the SAME narrative continues describing that past event but uses a present-tense verb, that is a tense-drift error.

Do NOT flag: a narrative told consistently in present tense from the start as a deliberate stylistic choice (the "historical present") where no past frame was ever established to drift away from. Only flag an actual REVERSION -- past frame established, then broken.

If it's unclear whether a past frame was actually established or whether this is intentional historical-present narration, set confidence to "low" and error to false.

Always also return `corrected`: the FULL sequence, rewritten with ONLY the drifted verb(s) fixed to match the established past frame; change nothing else -- if error is false, return the input completely unchanged. This lets a caller diff `corrected` against the original text word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "frueher, als ich Kind war, spiele ich gern Basketball und schwimme.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'frueher, als ich Kind war' establishes a clear past frame, but 'spiele' and 'schwimme' are present tense -- should be 'habe gespielt' / 'bin geschwommen'.",
                   "corrected": "frueher, als ich Kind war, habe ich gern Basketball gespielt und bin geschwommen."},
    },
    {
        "input": "weil ich damals kein Deutsch spreche, wuerde ein paar Frauen mit mir nicht reden.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'damals' establishes a past frame, but 'spreche' is present tense -- should be 'gesprochen habe'.",
                   "corrected": "weil ich damals kein Deutsch gesprochen habe, wuerde ein paar Frauen mit mir nicht reden."},
    },
    {
        "input": "Vorher, spreche ich sehr wenig Deutsch, weil ich mache sehr wenig Uebung.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'Vorher' establishes a past frame, but 'spreche' and 'mache' are both present tense -- should be 'habe gesprochen' / 'habe gemacht'.",
                   "corrected": "Vorher, habe ich sehr wenig Deutsch gesprochen, weil ich habe sehr wenig Uebung gemacht."},
    },
    {
        "input": "und dann kommt letztes Jahr. Ich mache ein Urlaub in Hamburg und nach Koeln.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'letztes Jahr' establishes a past frame, but 'kommt' and 'mache' are present tense -- should be 'kam' / 'habe gemacht'.",
                   "corrected": "und dann kam letztes Jahr. Ich habe ein Urlaub in Hamburg und nach Koeln gemacht."},
    },
    {
        "input": "Ich erzaehle dir die Geschichte im Praesens: er geht ins Zimmer, er sieht die Tuer offen, er schreit.",
        "answer": {"error": False, "confidence": "low",
                   "reasoning": "Illustrative example (not from the transcript corpus) of deliberate historical-present narration -- no past frame is established anywhere in the sequence for the present tense to be reverting from, so this should not be flagged. Replace with a validated real corpus example when one is available.",
                   "corrected": "Ich erzaehle dir die Geschichte im Praesens: er geht ins Zimmer, er sieht die Tuer offen, er schreit."},
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
    key="GVT-1",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",  # actually a short clause sequence -- see build_request note in registry.py
)
