"""
Sentence Structure Breadth.

Per lomb_metric_definitions_v1.md, this metric has "no published measure --
requires dependency parser." breadth_score = count of DISTINCT structure
types the speaker used across a session.

CRITICAL DESIGN POINT (flagged during earlier design discussion, not
optional): the output labels MUST come from a closed, fixed list. If the
model is allowed to invent free-text structure names, it might call the
same real structure "subordinate clause with modal" in one call and "modal
+ subordinate construction" in another -- those would then get counted as
two different structure types instead of one, silently corrupting the
breadth_score. STRUCTURE_LABELS below is that fixed list, and the
response schema constrains the model to only ever pick from it.

Only 2 of the 5 example labels below (konjunktiv_ii, dass_clause) are
anchored to real corpus sentences (from dan_error_analysis_master_v3.md's
KII drills and GVT examples). The other 3 (ob_clause, relative_clause,
damit_clause) are illustrative placeholders -- no validated real-corpus
example sentence for those specific structure types was available when
this was written. Replace them with real tagged examples once the corpus
has them.
"""

from metric_types import MetricPromptConfig

# Fixed, closed vocabulary. Extend this list deliberately (a reviewed code
# change) if a genuinely new structure type needs tracking -- never let
# the model add to it on the fly.
STRUCTURE_LABELS = [
    "konjunktiv_ii",       # wäre, hätte, könnte, müsste, sollte
    "dass_clause",         # subordinate clause introduced by dass
    "ob_clause",           # indirect question introduced by ob
    "weil_clause",         # subordinate clause introduced by weil
    "damit_clause",        # purpose clause introduced by damit
    "relative_clause",     # clause introduced by der/die/das as relative pronoun
    "passive_voice",       # werden + past participle
    "sowohl_als_auch",     # sowohl...als auch construction
    "je_desto",            # je...desto construction
    "weder_noch",          # weder...noch construction
    "none",                # sentence exhibits none of the above
]

SYSTEM_INSTRUCTION = f"""You are classifying ONE German sentence by which syntactic structure(s) it exhibits, from this FIXED list only -- never invent a label outside this list:

{", ".join(STRUCTURE_LABELS)}

A sentence may exhibit more than one structure (e.g. a dass-clause containing a Konjunktiv II verb) -- return all that genuinely apply. If none of the listed structures apply, return ["none"].

Only return a label if you are confident the structure is genuinely present -- do not guess to fill in a label. If uncertain about a specific label, omit it rather than including it with low confidence.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Das waere schoen.",
        "answer": {"structures": ["konjunktiv_ii"], "confidence": "high"},
    },
    {
        "input": "Ich denke, dass mein Hoerverstaendnis ziemlich okay ist.",
        "answer": {"structures": ["dass_clause"], "confidence": "high"},
    },
    {
        "input": "Ich weiss nicht, ob ich das schaffe.",
        "answer": {"structures": ["ob_clause"], "confidence": "low",
                   "reasoning": "Illustrative example (not from the transcript corpus)."},
    },
    {
        "input": "Das ist der Mann, der mir geholfen hat.",
        "answer": {"structures": ["relative_clause"], "confidence": "low",
                   "reasoning": "Illustrative example (not from the transcript corpus)."},
    },
    {
        "input": "Ich lerne jeden Tag, damit ich schneller Fortschritte mache.",
        "answer": {"structures": ["damit_clause"], "confidence": "low",
                   "reasoning": "Illustrative example (not from the transcript corpus)."},
    },
    {
        "input": "Ich habe gestern mit Leonie gesprochen.",
        "answer": {"structures": ["none"], "confidence": "high"},
    },
]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "structures": {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": STRUCTURE_LABELS},
        },
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
    },
    "required": ["structures", "confidence"],
}

CONFIG = MetricPromptConfig(
    key="STRUCTURE_BREADTH",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
)
