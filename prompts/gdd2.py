"""
GDD-2: Case Error -- Wechselpraepositionen (two-way prepositions).

in / an / auf / ueber / unter / vor / neben / zwischen take dative for
static location (Lage-Verben: sein, liegen, stehen, sitzen, bleiben,
wohnen, leben) or accusative for motion (Richtungsverben: gehen, fahren,
fliegen, ziehen, legen, stellen, setzen). Some verbs (haengen, stecken) are
genuinely ambiguous and resolve only from sentence context, not the verb
alone -- that ambiguity is the actual reason this metric is LLM-assisted.

4 of the 6 examples below are real transcript errors from
dan_error_analysis_master_v3.md (GDD error pattern 2) -- the "in die USA"
family is the single most frequent GDD-2 error in Dan's corpus (7
instances across 5 sessions). The haengen pair is illustrative (built to
teach the ambiguous-verb case explicitly), not pulled from a transcript --
flagged here so it's not mistaken for corpus data.
"""

from metric_types import MetricPromptConfig

# Fixed, closed vocabulary -- same rationale as ALWAYS_DATIVE_PREPOSITIONS
# in prompts/gdd1.py: exposed as a constant (mirroring STRUCTURE_LABELS in
# prompts/structure_breadth.py) so prefilter.py has one source of truth to
# import instead of a second, separately-typed copy of this list.
WECHSELPRAEPOSITIONEN = ["in", "an", "auf", "ueber", "unter", "vor", "neben", "zwischen"]

SYSTEM_INSTRUCTION = f"""You are checking ONE German sentence for a single, narrow grammar error type: incorrect case after a Wechselpraeposition (two-way preposition).

Wechselpraepositionen: {", ".join(WECHSELPRAEPOSITIONEN)}.

Rule:
- LOCATION (no movement toward a destination) -- verbs like sein, liegen, stehen, sitzen, bleiben, wohnen, leben -- takes DATIVE.
- MOTION toward a destination -- verbs like gehen, fahren, fliegen, ziehen, legen, stellen, setzen -- takes ACCUSATIVE.
- A few verbs (haengen, stecken, and similar) are genuinely ambiguous in isolation. Their reading depends on whether the sentence describes an existing state (dative) or the act of placing something (accusative). Decide from the sentence's own context; never guess from the verb alone.

Task: given one sentence, decide whether the case marking on the article/adjective after the Wechselpraeposition is correct.

If the sentence does not give enough context to determine the intended reading with confidence, set confidence to "low" and error to false. A missed error is preferable to a false alarm.

Always also return `corrected`: the FULL sentence, rewritten with ONLY this error type fixed (change the minimum necessary -- the article's case, not word order or wording) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Ich lebe in die USA.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'leben' is a Lage-Verb (static) -- dative plural 'in den USA' is required, not accusative 'in die USA'.",
                   "corrected": "Ich lebe in den USA."},
    },
    {
        "input": "Ich komme aus Austin -- ich fliege naechstes Jahr wieder in die USA.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'fliege' is a Richtungsverb (motion toward a destination) -- accusative 'in die USA' is correct here.",
                   "corrected": "Ich komme aus Austin -- ich fliege naechstes Jahr wieder in die USA."},
    },
    {
        "input": "Wuenschst du, langfristig in die Tuerkei zu bleiben?",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'bleiben' is a Lage-Verb -- feminine 'die Tuerkei' requires dative 'in der Tuerkei', not accusative.",
                   "corrected": "Wuenschst du, langfristig in der Tuerkei zu bleiben?"},
    },
    {
        "input": "Wie viele Tage war ich in die Krankenhaus?",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'war' (sein) is a Lage-Verb -- neuter 'das Krankenhaus' requires dative 'im Krankenhaus' (in+dem), not accusative.",
                   "corrected": "Wie viele Tage war ich im Krankenhaus?"},
    },
    {
        "input": "Ich habe das Bild an die Wand gehaengt.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "Motion reading -- the act of hanging it up -- so accusative 'an die Wand' is correct. Contrast with the next example.",
                   "corrected": "Ich habe das Bild an die Wand gehaengt."},
    },
    {
        "input": "Das Bild haengt an die Wand.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "Static reading -- 'haengt' describes the picture's current state, not the act of hanging it -- so dative 'an der Wand' is required, not accusative.",
                   "corrected": "Das Bild haengt an der Wand."},
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
    key="GDD-2",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
)
