"""
Formulaic Sequences.

2026-09-03: BUNDLES added below -- previously this docstring talked about
a reference list "anchoring what counts as a candidate" without the list
actually existing anywhere, which is exactly what pipeline.py and
prefilter.py both independently flagged as the real blocker (confirmed by
reading lomb_metric_definitions_v1.md directly: it specifies
`BUNDLES = load_reference_list()` as pseudocode, never an actual list).
Sourced from two places, not guessed:

1. German MODAL PARTICLES (Abtönungspartikeln) -- these are the textbook
   case of a word that is polysemous exactly the way this metric's own
   system instruction describes: a literal, content-bearing reading
   ("schon" = "already", "denn" = "because", "aber" = "but", "mal" = "a
   time/occurrence") versus a discourse-particle reading that adds
   speaker attitude rather than content (softening, reassurance,
   resignation, curiosity). Compiled from two independent sources so the
   list isn't one site's idiosyncratic take: elon.io's German-grammar
   reference (ja, doch, mal, halt, eben, wohl, schon, denn, eigentlich,
   also, naja, uebrigens, jedenfalls) and Lingoda's modal-particles guide
   (aber, ja, wohl, doch, halt, eben, schon, mal, denn, nun,
   schliesslich) -- both fetched 2026-09-03. The union of the two,
   deduplicated, is what's below.
2. Vague-language / hedge chunks -- "sozusagen", "irgendwie", "quasi",
   "so was", "und so" are standard German hedging vocabulary (see the
   general vague-language/hedge linguistics literature, e.g. Wikipedia's
   "Hedge (linguistics)" overview of the category this metric is
   targeting), plus "verschiedene Sachen" specifically, which stays
   corpus-confirmed (dan_error_analysis_master_v3.md, Idio-008 -- Dan's
   single highest-frequency filler chunk across the whole corpus, not a
   general-linguistics addition like the rest of this list).

HONESTY FLAG, explicit: this is a reasonable MVP candidate list, not a
linguistically exhaustive one -- German has more discourse particles and
hedge chunks than are listed here (irgendwas, sonstwie, gewissermassen,
and others exist too). Good enough to let this metric actually run and be
smoke-tested end to end, which it could not do at all before today. Treat
as a first cut to expand later with real corpus evidence, the same
"replace with validated real examples" caveat several other prompts/*.py
files already carry for their own illustrative content.

The "verschiedene Sachen" example is a real, corpus-documented filler
pattern (dan_error_analysis_master_v3.md, Idio-008 -- Dan's single
highest-frequency filler chunk across the whole corpus). The "denn"
particle-vs-conjunction pair and the new "mal" pair are illustrative (a
well-established German linguistic distinction, but not pulled from a
specific flagged transcript instance in the corpus) -- flagged here so
they aren't mistaken for corpus-verified examples the way "verschiedene
Sachen" is.
"""

from metric_types import MetricPromptConfig

# Candidate words/short phrases this metric is willing to classify as
# formulaic-or-literal. Deliberately NOT exhaustive (see docstring) --
# this is the list speaker_filter.to_formulaic_candidates() scans sentences
# against; a sentence containing none of these produces zero FORMULAIC
# calls, the same "skip what can't possibly trigger" logic prefilter.py
# already uses for GDD-1/GDD-2, just implemented as the candidate-scan
# step itself here rather than a separate prefilter module.
BUNDLES = [
    "verschiedene Sachen",  # corpus-confirmed (Idio-008), not a general addition
    "sozusagen",
    "irgendwie",
    "quasi",
    "so was",
    "und so",
    "ja",
    "doch",
    "mal",
    "halt",
    "eben",
    "wohl",
    "schon",
    "denn",
    "eigentlich",
    "aber",
    "nun",
    "schliesslich",
    "also",
    "naja",
    "uebrigens",
    "jedenfalls",
]

SYSTEM_INSTRUCTION = """You are checking ONE candidate word or short phrase, in the context of the sentence it appeared in, to decide whether it is being used as a FORMULAIC/discourse-marker chunk (vague filler, hedge, or discourse particle -- adds little specific content) versus a LITERAL, content-bearing use of the same word.

Examples of the distinction (not exhaustive):
- "verschiedene Sachen" (various things) used as a vague catch-all filler when the speaker could specify what they mean, versus used literally to actually enumerate/refer to genuinely distinct, specific items.
- "denn" used as an intensifying discourse particle (e.g. "was ist denn los?") versus used literally as the conjunction "because".
- "mal" used as a casual, softening discourse particle (e.g. "kommst du mal vorbei?") versus used literally to mean "a time/occurrence" (e.g. "ich war schon drei Mal dort").
- "schon" used as a reassurance/concession particle (e.g. "das wird schon klappen") versus used literally to mean "already" (e.g. "ich habe das schon gemacht").

Task: given the candidate word/phrase and its sentence, decide which use this is.

If the sentence doesn't give enough context to tell, set confidence to "low" and formulaic to false -- do not guess.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": 'Candidate: "verschiedene Sachen" | Sentence: "Ich koche gern verschiedene Sachen."',
        "answer": {"formulaic": True, "confidence": "high",
                   "reasoning": "Vague catch-all filler -- no specific items named, functions as a placeholder chunk rather than genuine content. This is Dan's single highest-frequency filler pattern in the corpus (Idio-008)."},
    },
    {
        "input": 'Candidate: "verschiedene Sachen" | Sentence: "Ich habe drei verschiedene Sachen in der Tasche: einen Stift, ein Buch und einen Schluessel."',
        "answer": {"formulaic": False, "confidence": "high",
                   "reasoning": "Used literally here -- the speaker genuinely enumerates distinct, specific items right after."},
    },
    {
        "input": 'Candidate: "denn" | Sentence: "Was ist denn los?"',
        "answer": {"formulaic": True, "confidence": "high",
                   "reasoning": "Illustrative example (not from the transcript corpus): 'denn' functions here as an intensifying discourse particle, not the conjunction 'because'."},
    },
    {
        "input": 'Candidate: "denn" | Sentence: "Ich bleibe hier, denn ich habe keine Zeit mehr."',
        "answer": {"formulaic": False, "confidence": "high",
                   "reasoning": "Illustrative example (not from the transcript corpus): 'denn' is used literally here as the causal conjunction 'because'."},
    },
    {
        "input": 'Candidate: "mal" | Sentence: "Komm doch mal vorbei, wenn du Zeit hast."',
        "answer": {"formulaic": True, "confidence": "high",
                   "reasoning": "Illustrative example (not from the transcript corpus): 'mal' softens the invitation here, a discourse particle, not a count of occurrences."},
    },
    {
        "input": 'Candidate: "mal" | Sentence: "Ich war schon drei Mal in Berlin."',
        "answer": {"formulaic": False, "confidence": "high",
                   "reasoning": "Illustrative example (not from the transcript corpus): 'Mal' here literally counts occurrences (three times), not a discourse particle."},
    },
]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "formulaic": {"type": "BOOLEAN"},
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
        "reasoning": {"type": "STRING"},
    },
    "required": ["formulaic", "confidence", "reasoning"],
}

CONFIG = MetricPromptConfig(
    key="FORMULAIC",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",  # actually "candidate + sentence" -- see speaker_filter.to_formulaic_candidates()
)
