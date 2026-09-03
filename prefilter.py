"""
Keyword pre-filter for the LLM-assisted metrics -- decides, per sentence,
whether a metric's own trigger condition can even be met before spending an
LLM call on it.

SCOPE, STATED PLAINLY: only 2 of the 10 registered metrics are safe to
pre-filter this way -- GDD-1 and GDD-2. Both check case marking on ONE
specific, closed set of German prepositions; if none of those prepositions
(or a contracted form of one) appear in a sentence, the metric's own rule
cannot possibly fire, so skipping the LLM call for that sentence loses
nothing. Everything else below is what was checked and deliberately
rejected as NOT safely filterable -- not an oversight. Read the per-metric
notes near the bottom of this docstring before adding anything else here.

WHY THIS SHAPE:
- GDD-1 (prompts/gdd1.py) and GDD-2 (prompts/gdd2.py) now expose their own
  closed preposition lists as module-level constants (ALWAYS_DATIVE_
  PREPOSITIONS, WECHSELPRAEPOSITIONEN), mirroring the STRUCTURE_LABELS
  pattern prompts/structure_breadth.py already used. This module imports
  those directly rather than re-typing them, so the filter can never
  silently drift out of sync with the rule it's approximating.

- Contracted forms (beim, vom, zum, zur, am, ans, im, ins, aufs) are
  included explicitly, not an afterthought: two of GDD-1's own five
  few-shot examples ("vom Supermarkt", "beim Banken") use a contraction,
  not the bare preposition. A filter that only matched the bare
  prepositions as literal words would have skipped both -- silently
  breaking on the metric's own training data. (A related but distinct
  real corpus error exists too -- all_grammar_errors_master.json,
  anja_italki1 session, "zu Fremdsprache" should be "zur Fremdsprache" --
  but that one is a MISSING contraction: the error sentence itself
  contains the bare word "zu", already caught by the base trigger list
  with no contraction-matching needed. Noted here so it isn't mistaken for
  evidence the contraction list depends on.) Checked here, not just
  claimed: __main__ below re-runs every GDD-1/GDD-2 few-shot example
  through the filter and asserts none of them get wrongly skipped.

- Real umlauts AND their ASCII transliterations are both matched, for
  "ueber"/"gegenueber"/"ausser" specifically. This is a genuine finding,
  not a guess either way: dan_error_analysis_master_v3.md and
  all_grammar_errors_master.json (the actual transcript corpus) are full
  of real umlauts ("über", "müssen", "würde"), and assemblyai_standard_
  prompt.txt explicitly instructs the ASR step to "preserve language-
  specific characters exactly as spoken, including German umlauts (ä, ö,
  ü, ß)" -- so real production sentences reaching this filter will very
  likely contain real umlauts. But every few-shot example INSIDE this
  codebase's own prompts/*.py files (the ones test_all_metrics_live.py
  actually exercises) is ASCII-transliterated instead ("Frueher", "wuerde",
  "waere") -- nothing in speaker_filter.py's to_sentences() or anywhere
  else in this codebase normalizes between the two forms. Rather than
  guess which spelling will actually reach this function at runtime, both
  are matched. This only matters for gegenueber/gegenüber, ausser/außer,
  and ueber/über -- none of GDD-1/GDD-2's other trigger words or
  contractions contain umlauts.

METRICS DELIBERATELY LEFT UNFILTERED, AND WHY (re-read this before adding a
keyword filter for any of these):

- GVT-2 (prompts/gvt2.py): checks TWO things in one LLM call -- subordinate
  clause verb-final position (closed trigger set: dass/weil/wenn/ob) AND
  main-clause V2 inversion after a fronted element. The V2 half has no
  closed trigger set: V2 inversion applies whenever ANYTHING other than
  the subject is fronted (an adverb, a time expression, an object, a whole
  clause) -- not a fixed vocabulary the way GDD-1/2's prepositions are. A
  filter keyed on dass/weil/wenn/ob would silently skip every V2-fronting
  error that doesn't happen to use one of those four words -- and GVT-2's
  OWN few-shot examples "vielleicht ich treffe..." and "dann wir sehen..."
  contain none of them and would be wrongly skipped. So GVT-2 is not
  filtered here at all, on either half of what it checks, even though the
  subordinate-clause half alone would look filterable in isolation.
- LP (prompts/lp.py): its own system instruction says plainly "there is no
  fixed rule for this category" -- no closed trigger set exists to filter on.
- LPF (prompts/lpf.py): its system instruction's own trigger list is
  explicitly flagged "not exhaustive" -- a keyword filter built from it
  would quietly cut real coverage every time a transfer error uses a
  verb-preposition pair not in that illustrative list.
- FORMULAIC (prompts/formulaic.py): 2026-09-03 UPDATE -- the BUNDLES
  reference list this note used to say didn't exist now DOES exist (added
  the same day, sourced from two web references plus the existing corpus
  finding; see that module's own docstring). That doesn't make FORMULAIC a
  candidate for THIS module, though: the reason isn't "no list to filter
  on" anymore, it's that scanning sentences against BUNDLES to decide which
  (candidate, sentence) pairs are even worth an LLM call IS the filtering
  step for this metric -- it already happens in
  speaker_filter.to_formulaic_candidates(), which is also where FORMULAIC's
  real unit of work (a candidate/sentence pair, not a bare sentence) gets
  built in the first place. Duplicating that scan here, keyed to a bare
  sentence with no candidate attached, wouldn't fit this module's
  should_run(metric_key, sentence) -> bool shape anyway. So FORMULAIC still
  isn't in _PATTERNS below -- now because its filtering already lives
  correctly upstream, not because nothing exists to filter with.
- STRUCTURE_BREADTH (prompts/structure_breadth.py): most of its 10 closed
  labels DO have a lexical cue (dass/ob/weil/damit for the four clause
  types, waere/haette/koennte/... for konjunktiv_ii, werden+participle for
  passive, sowohl/je/weder for the three correlative constructions) -- but
  relative_clause does not: der/die/das introducing a relative clause is
  lexically identical to der/die/das as a plain definite article, so no
  keyword rule can tell them apart. Filtering here would systematically
  blind the metric to relative_clause specifically, not just miss it at
  random -- worse than no filter, since it would look like coverage while
  quietly having a permanent hole.
- GVT-1 (prompts/gvt1.py): needs a short SEQUENCE of consecutive clauses to
  track whether an established past-tense frame is later broken (see its
  own docstring / input_kind note) -- it is not a single-sentence yes/no
  question, so it doesn't fit a per-sentence keyword filter's shape at all.
- UNFILLED_PAUSE and FILLED_PAUSE: input_kind is word_timestamps for both
  (FILLED_PAUSE was redefined off audio_turn onto word_timestamps on
  2026-09-03 -- see prompts/filled_pause.py's docstring), not sentence
  text -- there is no sentence for a keyword filter to even look at.
  FILLED_PAUSE's own trigger tokens (aeh/aehm/hm/hmm/mhm/oehm) ARE a closed
  set, unlike GDD-1/GDD-2's prepositions this module actually filters on,
  but the metric IS the act of finding those tokens -- a keyword match that
  decided whether to CALL the LLM would just be the whole answer computed
  twice, once for nothing.

WHAT "FILTERED" MEANS HERE: should_run() returning False means the LLM call
for THAT metric can be skipped for THAT sentence -- it says nothing about
whether the sentence is error-free, and nothing about any of the other 9
metrics, which still run on every sentence regardless of what this returns.
"""

import re

from prompts.gdd1 import ALWAYS_DATIVE_PREPOSITIONS
from prompts.gdd2 import WECHSELPRAEPOSITIONEN

# Contracted preposition+article forms. Each one folds a trigger
# preposition together with a definite article, so the bare preposition
# never appears as its own word in the sentence -- a filter that only
# matched the words in ALWAYS_DATIVE_PREPOSITIONS / WECHSELPRAEPOSITIONEN
# outright would miss every one of these. Standard-German contractions
# only (no colloquial "übers"/"unters"/"vors" -- rare enough, and absent
# from every corpus example checked here, that including them would be
# guessing rather than reporting a checked list; add them if a real
# transcript example ever surfaces one).
GDD1_CONTRACTIONS = {
    "beim": "bei",  # bei + dem
    "vom": "von",   # von + dem
    "zum": "zu",    # zu + dem
    "zur": "zu",    # zu + der
}

GDD2_CONTRACTIONS = {
    "am": "an",     # an + dem
    "ans": "an",    # an + das
    "im": "in",     # in + dem
    "ins": "in",    # in + das
    "aufs": "auf",  # auf + das
}

# ASCII transliteration <-> real umlaut, for the specific trigger words
# where the two forms differ (see docstring above for why both are
# matched). Only 3 of GDD-1/GDD-2's words need this.
_UMLAUT_VARIANTS = {
    "ueber": "über",
    "gegenueber": "gegenüber",
    "ausser": "außer",
}


def _with_umlaut_variants(words) -> list[str]:
    out = list(words)
    for w in words:
        if w in _UMLAUT_VARIANTS:
            out.append(_UMLAUT_VARIANTS[w])
    return out


def _make_pattern(words) -> re.Pattern:
    # Longest-first isn't strictly required -- \b at both ends of the
    # group already forces the regex engine to back off a short match
    # like "aus" if it lands mid-word inside "ausser" and retry the longer
    # alternative -- but sorting avoids relying on that backtracking.
    # \b...\b with IGNORECASE handles capitalization at sentence start
    # (e.g. "Mit die Grammatik...") without over-matching substrings
    # inside longer, unrelated words (\b won't fire inside "aber" for
    # "ab", since there's no boundary between the 'b' and the 'e' that
    # follows it).
    ordered = sorted(set(words), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in ordered) + r")\b", re.IGNORECASE)


_GDD1_TRIGGERS = _with_umlaut_variants(ALWAYS_DATIVE_PREPOSITIONS) + list(GDD1_CONTRACTIONS)
_GDD2_TRIGGERS = _with_umlaut_variants(WECHSELPRAEPOSITIONEN) + list(GDD2_CONTRACTIONS)

_GDD1_PATTERN = _make_pattern(_GDD1_TRIGGERS)
_GDD2_PATTERN = _make_pattern(_GDD2_TRIGGERS)

# Metrics not in this dict are simply not filtered -- see the docstring
# above for why, per metric. Absence here is a deliberate decision.
_PATTERNS = {
    "GDD-1": _GDD1_PATTERN,
    "GDD-2": _GDD2_PATTERN,
}

FILTERABLE_METRICS = frozenset(_PATTERNS)


def should_run(metric_key: str, sentence: str) -> bool:
    """True if metric_key's own LLM call is worth making for this sentence.

    For any metric not in FILTERABLE_METRICS this always returns True --
    this function only ever grants permission to SKIP a call, never
    permission to run one that wasn't already going to run. It also
    doesn't decide WHETHER to call a given metric on a given input_kind at
    all (e.g. it has nothing to say about FILLED_PAUSE, which isn't a
    sentence-input metric in the first place) -- that gating stays wherever
    it already lives (registry.py / speaker_filter.py), upstream of this.
    """
    pattern = _PATTERNS.get(metric_key)
    if pattern is None:
        return True
    return pattern.search(sentence) is not None


def prefilter_sentences(metric_key: str, sentences: list) -> list:
    """The subset of `sentences` worth sending to metric_key's LLM call, in
    original order. For an unfilterable metric this is just `sentences`
    back unchanged -- never a shorter list produced by guessing."""
    return [s for s in sentences if should_run(metric_key, s)]


if __name__ == "__main__":
    from prompts.gdd1 import CONFIG as GDD1_CONFIG
    from prompts.gdd2 import CONFIG as GDD2_CONFIG

    print(f"GDD-1 trigger pattern: {_GDD1_PATTERN.pattern}\n")
    print(f"GDD-2 trigger pattern: {_GDD2_PATTERN.pattern}\n")

    # The check this file's docstring promises: none of GDD-1/GDD-2's own
    # few-shot examples -- including the two built on a contraction --
    # get wrongly skipped by this filter.
    for label, config in (("GDD-1", GDD1_CONFIG), ("GDD-2", GDD2_CONFIG)):
        print(f"-- {label} few-shot examples --")
        for ex in config.few_shot_examples:
            sentence = ex["input"]
            kept = should_run(label, sentence)
            status = "KEPT (sent to LLM)" if kept else "SKIPPED"
            print(f"  {status:<20} {sentence!r}")
            assert kept, (
                f"{label} pre-filter wrongly skipped its own few-shot example: {sentence!r}"
            )
    print("\nAll GDD-1/GDD-2 few-shot examples correctly kept.\n")

    # Prove the umlaut-variant matching actually works, not just that it's
    # wired up: the ASCII-only ALWAYS_DATIVE_PREPOSITIONS/WECHSELPRAEPOSITIONEN
    # lists wouldn't catch these on their own.
    umlaut_checks = [
        ("GDD-1", "Ich wohne gegenüber der Kirche.", True),
        ("GDD-1", "Ich habe außer meiner Schwester niemanden gefragt.", True),
        ("GDD-2", "Das Bild hängt über dem Sofa.", True),
    ]
    print("-- real-umlaut sentences (not in any few-shot set) --")
    for label, sentence, expected in umlaut_checks:
        actual = should_run(label, sentence)
        print(f"  should_run({label!r}, {sentence!r}) -> {actual}")
        assert actual == expected, f"expected {expected}, got {actual}"
    print("\nUmlaut-variant matching confirmed.\n")

    # A sentence with none of GDD-1/GDD-2's trigger words, contractions, or
    # umlaut variants should be skippable -- proves the filter actually
    # filters something, not just that it never skips.
    control = "Ich lerne jeden Tag Deutsch."
    print(f"Control sentence with no GDD-1/GDD-2 triggers: {control!r}")
    print(f"  GDD-1 should_run -> {should_run('GDD-1', control)}")
    print(f"  GDD-2 should_run -> {should_run('GDD-2', control)}")
    assert not should_run("GDD-1", control)
    assert not should_run("GDD-2", control)

    # A metric not in FILTERABLE_METRICS always returns True, regardless
    # of content.
    print(f"\nLP (unfilterable) on the same control sentence -> "
          f"{should_run('LP', control)} (always True, by design)")
    assert should_run("LP", control)

    print("\nSelf-checks passed.")
