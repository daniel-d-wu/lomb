"""
Formulaic Sequences.

2026-09-05, Dan's explicit instruction: FORMULAIC is now a deterministic
REGEX metric, not an LLM-assisted one -- removed from registry.
METRIC_PROMPTS entirely (see registry.py) and from pipeline.py's LLM call
path. This file now holds ONLY the BUNDLES reference list plus this
documentation; the LLM-specific scaffolding that used to live below
(SYSTEM_INSTRUCTION, FEW_SHOT_EXAMPLES, RESPONSE_SCHEMA, CONFIG) has been
deleted outright, not just left unused -- leaving it in place would have
risked a future accidental re-registration in registry.py.

WHAT THIS TRADES AWAY: the old LLM version's whole job was disambiguating
a literal reading of a BUNDLES word from a formulaic/discourse-particle
reading in context (see the "denn"/"mal"/"schon" contrast pairs the old
FEW_SHOT_EXAMPLES used to carry, now gone) -- e.g. telling literal "ich
habe das schon gemacht" (already) apart from formulaic "das wird schon
klappen" (reassurance particle). A regex match against BUNDLES can't make
that distinction at all: any sentence containing a BUNDLES word/phrase
now counts as a formulaic hit, full stop, whether the use is literal or
not. This is a real, accepted accuracy cost, not an oversight -- it buys
back a call-count/cost/determinism win (zero LLM calls for this metric,
same "regex, not LLM" treatment already given to prefilter.py's own
trigger-keyword matching) in exchange for it, which is what Dan asked for.
Revisit if FORMULAIC's false-positive rate on genuinely literal uses (e.g.
"ich war schon drei Mal in Berlin") turns out to matter in practice.

speaker_filter.find_formulaic_matches() (renamed 2026-09-05 from
to_formulaic_candidates(), which used to format an LLM-prompt-ready
"Candidate: ... | Sentence: ..." string per hit -- there is no LLM prompt
to format for anymore, so it now returns plain {candidate, sentence}
match records instead) is what actually scans sentences against BUNDLES;
this module no longer builds anything for a caller besides the list
itself.

BUNDLES sourced from two places, not guessed:

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

The "verschiedene Sachen" entry is a real, corpus-documented filler
pattern (dan_error_analysis_master_v3.md, Idio-008 -- Dan's single
highest-frequency filler chunk across the whole corpus). Every other entry
is a standard German discourse-particle/hedge word, included on general
linguistic grounds (see sources above), not pulled from a specific
flagged transcript instance -- flagged here so it isn't mistaken for a
corpus-verified example the way "verschiedene Sachen" is. This
distinction mattered more under the old LLM version (whose few-shot
examples needed to be labeled real vs. illustrative); it's noted here
mainly for provenance now that there's no LLM prompt for it to matter to.
"""

# Candidate words/short phrases this metric flags as formulaic on sight.
# Deliberately NOT exhaustive (see docstring) -- this is the list
# speaker_filter.find_formulaic_matches() scans sentences against; a
# sentence containing none of these produces zero FORMULAIC hits, the same
# "skip what can't possibly trigger" logic prefilter.py already uses for
# GDD-1/GDD-2, just implemented as the regex-scan step itself here rather
# than a separate prefilter module. As of 2026-09-05 a match against this
# list IS the metric's verdict (formulaic: true, no further judgment) --
# see the module docstring's "WHAT THIS TRADES AWAY" note for what that
# gives up relative to the old LLM-disambiguated version.
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
