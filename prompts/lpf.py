"""
LPF: Lexical Phrase / False Friend error rate (wrong preposition or
collocate from L1-English transfer).

The verb-preposition lookup itself is mechanical (a fixed dictionary), but
locating the preposition actually governed by a target verb in a disfluent
utterance -- rather than one attached to something else nearby -- needs
judgment on messy input, which is why this stays LLM-assisted.

All 5 few-shot examples are real transcript errors/corrections from
dan_error_analysis_master_v3.md (LPF error pattern 1).
"""

from metric_types import MetricPromptConfig

SYSTEM_INSTRUCTION = """You are checking ONE German sentence for a single, narrow error type: a wrong preposition (or a missing/extra preposition) governed by a specific verb, caused by direct transfer from English.

Common examples of this transfer pattern (not exhaustive):
- suchen takes 'nach', not 'fuer' (English "search for")
- sich freuen (auf something upcoming) takes 'auf', not 'fuer' (English "look forward to/be happy for")
- passen (to suit someone) takes a plain dative object, no preposition at all (English "work for me" wrongly imports 'fuer')
- im Fernsehen / in den Nachrichten (in TV/the news), not 'auf' (English "on TV/on the news")

Task: given one sentence, identify the verb that is the source of a potential error, determine which preposition (if any) it actually governs in standard German, and check whether the sentence uses the correct one. Be careful to attach the preposition to the verb it's actually modifying, not one that happens to sit nearby in disfluent speech.

If you cannot confidently identify which verb governs the preposition in question, set confidence to "low" and error to false.

Always also return `corrected`: the FULL sentence, rewritten with ONLY this error type fixed (swap in the correct preposition/case, don't otherwise reword) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Ich suche fuer Muttersprachler, um mein Deutsch zu verbessern.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'suchen' governs 'nach', not 'fuer' -- should be 'suche nach Muttersprachlern'.",
                   "corrected": "Ich suche nach Muttersprachlern, um mein Deutsch zu verbessern."},
    },
    {
        "input": "Ich freue mich fuer diese Reise, dass ich bald gehe.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'sich freuen' (auf something upcoming) governs 'auf', not 'fuer' -- should be 'freue mich auf diese Reise'.",
                   "corrected": "Ich freue mich auf diese Reise, dass ich bald gehe."},
    },
    {
        "input": "Okay, ja, das passt fuer mich.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'passen' takes a plain dative object with no preposition -- should be 'das passt mir', not 'das passt fuer mich'.",
                   "corrected": "Okay, ja, das passt mir."},
    },
    {
        "input": "habe ich gesehen auf der Nachricht.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "The correct collocation is 'in den Nachrichten', not 'auf der Nachricht' -- direct transfer from English 'on the news'.",
                   "corrected": "habe ich gesehen in den Nachrichten."},
    },
    {
        "input": "Ich freue mich wirklich auf Berlin -- ich war noch nie da.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'sich freuen auf' is the correct preposition here.",
                   "corrected": "Ich freue mich wirklich auf Berlin -- ich war noch nie da."},
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
    key="LPF",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
)
