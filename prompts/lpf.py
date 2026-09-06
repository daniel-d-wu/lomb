"""
LPF: Lexical Phrase / False Friend error rate (wrong preposition or
collocate from L1-English transfer).

The verb-preposition lookup itself is mechanical (a fixed dictionary), but
locating the preposition actually governed by a target verb in a disfluent
utterance -- rather than one attached to something else nearby -- needs
judgment on messy input, which is why this stays LLM-assisted.

5 of the 6 few-shot examples are real transcript errors/corrections from
dan_error_analysis_master_v3.md (LPF error pattern 1); the 6th (added
2026-09-05, see below) is from all_grammar_errors_master.json.

2026-09-05, Dan's explicit instruction: `corrected` no longer follows the
"only fix this one error type" convention every other error-metric prompt
in this codebase uses (GDD-1, GDD-2, GVT-1, GVT-2, LP still all do -- this
change is LPF-specific, as literally requested, not applied elsewhere).
When a sentence has more than one error, `corrected` must now be a FULLY
corrected, natural sentence -- every error fixed, not just the
LPF-pattern preposition/case one that made `error` true. The `error` flag
itself is unchanged in scope: it still only answers "does an LPF-type
preposition-transfer error exist here," never anything about other error
types. Only what `corrected` shows changed. The new 6th few-shot example
below is real corpus data demonstrating this on a sentence with TWO
co-occurring LPF-pattern errors in the same utterance (alicia_italki3,
all_grammar_errors_master.json, itself explicitly documented there as a
"dual error in this utterance") -- both get fixed in `corrected`, not
just whichever one the reasoning happens to name first.
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

Always also return `corrected`. Unlike this codebase's other error-checking metrics (which only fix their own one target error type and leave everything else in the sentence untouched), LPF's `corrected` must be a FULLY corrected, natural-sounding version of the whole sentence: if the sentence has more than one error -- another LPF-pattern preposition error elsewhere in it, or a different kind of error entirely (case, word order, word choice) -- fix ALL of them, not only the one that made `error` true. `error` itself stays scoped to just the LPF preposition-transfer pattern described above; only what `corrected` shows is broader. If error is false, return the input sentence completely unchanged (do not "fix" unrelated errors in a sentence that has no LPF-pattern error at all). This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

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
    {
        "input": "Ueber die Nachricht oder auf dem Fernseher sehe ich viele diese Rassismus oder Hate Crimes.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "TWO LPF-pattern errors in this sentence, both fixed in `corrected` per the 2026-09-05 instruction, not just one of them: (1) 'ueber die Nachricht' should be 'in den Nachrichten' -- 'the news' (the broadcast) is idiomatically plural with 'in', not singular with 'ueber'; (2) 'auf dem Fernseher' should be 'im Fernsehen' -- 'im Fernsehen' (on television, the medium) is the correct collocation, 'auf dem Fernseher' literally means on top of the physical TV set. This is a real corpus utterance (alicia_italki3, all_grammar_errors_master.json) explicitly documented there as carrying both errors together.",
                   "corrected": "In den Nachrichten oder im Fernsehen sehe ich viele diese Rassismus oder Hate Crimes."},
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
