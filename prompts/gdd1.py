"""
GDD-1: Case Error -- Always-Dative Prepositions.

mit / von / bei / zu / nach / aus / gegenueber / ausser / seit / ab always
take dative, regardless of motion or direction -- unlike GDD-2's
Wechselpraepositionen, there is no verb-based ambiguity to resolve. The
real reliability risk (per lomb_metric_definitions_v1.md) is identifying
which noun phrase the preposition actually governs in disfluent, restarted
speech -- not the case rule itself, which is a fixed lookup.

All 5 few-shot examples below are real transcript errors/corrections from
dan_error_analysis_master_v3.md (GDD error pattern 1) -- none invented,
per this project's own standing rule that drill examples anchor to real
transcript context.
"""

from metric_types import MetricPromptConfig

# Fixed, closed vocabulary -- German genuinely only has these always-dative
# prepositions, nothing "not exhaustive" about this list (unlike LPF's
# trigger list, which is explicitly non-exhaustive). Exposed as its own
# constant, same pattern prompts/structure_breadth.py already uses for
# STRUCTURE_LABELS, so prefilter.py can import this directly instead of
# re-typing the list somewhere else and risking the two copies drifting
# apart.
ALWAYS_DATIVE_PREPOSITIONS = [
    "mit", "von", "bei", "zu", "nach", "aus", "gegenueber", "ausser", "seit", "ab",
]

SYSTEM_INSTRUCTION = f"""You are checking ONE German sentence for a single, narrow grammar error type: wrong case after an always-dative preposition.

Always-dative prepositions: {", ".join(ALWAYS_DATIVE_PREPOSITIONS)}. These ALWAYS take dative case -- there is no motion/location distinction to resolve here (that ambiguity only applies to a different set of prepositions, handled elsewhere).

Task: given one sentence containing one of these prepositions, first identify the noun phrase the preposition actually governs (be careful in disfluent or restarted speech -- the nearest noun phrase is not always the one actually governed). Then check whether that noun phrase's article/adjective is correctly declined for dative case, given its gender and number.

Dative articles: masculine -> dem (contracts to vom = von+dem, beim = bei+dem, zum = zu+dem), feminine -> der (bei der, NEVER "beim" for feminine), neuter -> dem, plural -> den (and the noun itself takes -n unless it already ends in -n or -s).

Common trap: "beim" and "vom" are valid contractions ONLY for masculine or neuter nouns. A feminine noun after "bei" or "von" must be "bei der" / "von der", never "beim" / "vom".

If the sentence is too fragmented to identify which noun phrase is governed, set confidence to "low" and error to false -- do not guess.

Always also return `corrected`: the FULL sentence, rewritten with ONLY this error type fixed (change the minimum necessary -- the article/adjective ending, not word order or wording) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Trotzdem kann ich nicht mit die Leute sprechen.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'mit' governs 'die Leute' (plural) -- dative plural requires 'den Leuten', not 'die Leute'.",
                   "corrected": "Trotzdem kann ich nicht mit den Leuten sprechen."},
    },
    {
        "input": "Mit die Grammatik habe ich noch nicht so bewusst gelernt.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'mit' governs 'die Grammatik' (feminine) -- dative feminine requires 'der Grammatik', not 'die Grammatik'.",
                   "corrected": "Mit der Grammatik habe ich noch nicht so bewusst gelernt."},
    },
    {
        "input": "Es ist nur einfach ein Ei, das man vom Supermarkt kaufen kann.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'von' + masculine 'der Supermarkt' correctly contracts to 'vom Supermarkt'.",
                   "corrected": "Es ist nur einfach ein Ei, das man vom Supermarkt kaufen kann."},
    },
    {
        "input": "Frueher war ich ein Datenwissenschaftler beim Banken.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'bei' governs 'die Bank' (feminine). 'beim' (bei+dem) is valid only for masculine/neuter, so this must be 'bei der Bank', not 'beim Banken'.",
                   "corrected": "Frueher war ich ein Datenwissenschaftler bei der Bank."},
    },
    {
        "input": "Ich suche schon seit ein paar Monaten nach einem Tandempartner.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "'seit' + dative 'ein paar Monaten' is correctly declined.",
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
    key="GDD-1",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
)
