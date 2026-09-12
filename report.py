"""
Turns pipeline.py's output (a pipeline_result.json, per-metric raw LLM
answers) into the two things a session actually needs downstream:

1. A report-contract-shaped JSON, following the API shape
   claude/lomb_backend_prd_v1.md Section 5 already locked in
   (session/fluency/accuracy/complexity), as closely as this pipeline's
   CURRENT coverage actually supports -- see "WHAT THIS DOES NOT DO" below
   for the real, specific gaps between this and that full contract. Where
   this pipeline doesn't have data a contract field needs, this leaves
   that field null/absent with a `_coverage_notes` explanation instead of
   inventing a number.

2. A human-readable HTML report built from that same JSON.

2026-09-03 redesign of the error cards, per Dan's direct feedback on the
first version ("get rid of the explanations... only color the text where
the part is wrong, and then immediately after, rewrite the sentence in
the right form and show with green text what it should have been"):

- The `why` / reasoning text is no longer shown in the HTML card. It's
  still carried in the underlying JSON (report["accuracy"]["errors"][i]
  ["why"]) since the PRD contract itself has a `why` field for later use
  by a real frontend -- this file just stopped RENDERING it by default.
- Each card now shows the original sentence with ONLY the changed word(s)
  highlighted red, immediately followed by the corrected sentence with
  ONLY the changed word(s) highlighted green.

Getting that right needed a real fix, not a rendering trick: the 5
error-metric prompts (GDD-1, GDD-2, GVT-1, GVT-2, LPF, LP) previously only
returned an `error` flag plus prose that MENTIONED a fix inline -- there
was no clean, separate "here is the corrected sentence" field to diff
against. Scraping a correction out of free-form reasoning text with regex
was considered and rejected (flagged in this file's previous version) --
too fragile, would silently misquote corrections that don't follow the
same phrasing pattern. Instead, all 5 prompts (prompts/gdd1.py,
prompts/gdd2.py, prompts/gvt1.py, prompts/gvt2.py, prompts/lpf.py,
prompts/lp.py) now have a `corrected` field added to their response
schema: the model returns the FULL sentence with only that specific error
fixed (or the sentence unchanged if error is false). That's a real,
well-defined string this file can diff against the original
WORD-BY-WORD using Python's own difflib -- no guessing at which words
changed, no LLM-side span-labeling to trust, just a deterministic diff
between two full sentences the model already had to produce anyway.

CONSEQUENCE FOR EXISTING pipeline_result.json FILES: any pipeline_result.
json produced BEFORE this schema change (i.e. from the OpenAI package
Dan already had) will not have a `corrected` field in its output dicts.
_diff_html() below detects that and falls back to showing the original
sentence only, with a note that it needs a fresh run to get the new
red/green format -- it does NOT guess at a correction from the reasoning
text to paper over the gap.

WHAT THIS DOES NOT DO (gaps against the full PRD Section 5 contract,
stated plainly rather than silently painted over with fabricated
numbers):

- `fluency.wpm`, `ohRate`, `ahRate` and `complexity.ttr`, `vocdD`,
  `wuerde`, `hypothetical` are NOT computed here at all. Those are the 6
  Python-logic metrics (WPM, "Oh" rate, Lexical Diversity, Missing
  Active Vocabulary, Missing Syntactic Structures, Preterite Avoidance --
  see claude/lomb_metric_definitions_v1.md) -- a completely separate,
  not-yet-built module that this LLM-assisted pipeline was never meant
  to cover. Left null with a coverage note, not estimated.

- UNFILLED_PAUSE is now surfaced as a plain occurrence count (see
  _summarize_unfilled_pause()'s own docstring), not a rate -- word_count
  isn't threaded through this pipeline yet (that's a `sessions`-table
  field, per lomb_metric_architecture_v1.md, not something pipeline_result.
  json carries today), so `unfilled_pause_rate` can't be computed here
  without a number to divide by. Left as a raw count with a coverage note,
  not estimated against a guessed denominator.

- FORMULAIC and FILLED_PAUSE are not in ERROR_METRICS below and never
  contribute to accuracy.errors[] -- claude/lomb_reporting_requirements_v1.
  md's own 6-metric error cap (GDD-1, GDD-2, GVT-1, GVT-2, LPF, LP) never
  included either of them; that's a reporting-scope decision, unrelated to
  whether pipeline.py runs them. A fresh pipeline_result.json will have
  "FORMULAIC" and "FILLED_PAUSE" keys with real per-occurrence output in
  them -- this file just doesn't surface either into the HTML report yet,
  the same "not yet surfaced, not the same as zero found" distinction this
  module already applies to everything else it hasn't built a summarizer
  for. A pipeline_result.json produced before each metric's own move to
  this pipeline won't have that key at all, and (like every other "metric
  absent from results" case in this file) that still correctly reads as a
  coverage note here, never a false "zero found."

- 2026-09-05/2026-09-06: FORMULAIC, FILLED_PAUSE, and UNFILLED_PAUSE all
  moved from LLM-assisted to direct/deterministic computation (see
  pipeline.py's DIRECT_METRICS and direct_computation.py). FORMULAIC's own
  output dict shape was UNCHANGED by its move (still {"formulaic": bool,
  "confidence": str, "reasoning": str} per entry), so nothing in this file
  needed to change for that switch. FILLED_PAUSE and UNFILLED_PAUSE's
  shapes DID change -- both used to be one entry per word-timestamp
  window (a {"boundaries": [...]} or {"fillers": [...]} list covering
  every boundary/word checked in that window, flagged or not); both are
  now one entry per actual occurrence found (a filler word, or a gap over
  the fixed threshold), each carrying `sentence_indices` and
  `boundary_type` fields neither had before (see direct_computation.py's
  module docstring for why that granularity change was made, and
  speaker_filter.map_words_to_sentences() for where those two new fields
  come from). _summarize_unfilled_pause() below was rewritten for this new
  shape -- a pipeline_result.json from before 2026-09-06 will have the OLD
  {"boundaries": [...]} shape for UNFILLED_PAUSE, which this rewritten
  function no longer reads; re-run the pipeline to get the new shape
  rather than expecting old result files to still summarize correctly.
"""

import difflib
import json
import sys
from html import escape

TAGS = {
    "GDD-1": "Case: always-dative preposition",
    "GDD-2": "Case: two-way preposition (Wechselpräposition)",
    "GVT-1": "Verb tense drift",
    "GVT-2": "Verb position",
    "LPF": "Preposition (L1 transfer)",
    "LP": "Word choice / collocation",
}

# The 6 metrics claude/lomb_reporting_requirements_v1.md designated for
# Report 1's accuracy.errors[] cap, in the order that doc lists them --
# not every one of these is necessarily present in a given
# pipeline_result.json (GVT-1 in particular, see pipeline.py's own scope
# notes), and this module treats "designated but absent" as its own
# category rather than conflating it with "ran, found nothing."
ERROR_METRICS = ["GDD-1", "GDD-2", "GVT-1", "GVT-2", "LPF", "LP"]
CAP_PER_METRIC = 2
CAP_TOTAL = CAP_PER_METRIC * len(ERROR_METRICS)  # 12, per the 2026-09-02 decision


def _build_accuracy(results: dict) -> dict:
    errors = []
    coverage = []
    for metric in ERROR_METRICS:
        if metric not in results:
            coverage.append(f"{metric}: not yet wired into this pipeline run "
                             f"(see pipeline.py's scope notes) -- absent, not zero.")
            continue
        entries = results[metric]
        flagged = [
            e for e in entries
            if not e["skipped"] and e["output"] is not None and e["output"].get("error") is True
        ]
        shown = flagged[:CAP_PER_METRIC]
        for e in shown:
            errors.append({
                "tag": TAGS[metric],
                "bad": e["input"],
                # None when this pipeline_result.json predates the
                # `corrected` schema field -- see module docstring.
                "good": e["output"].get("corrected"),
                "why": e["output"]["reasoning"],
            })
        if len(flagged) > CAP_PER_METRIC:
            coverage.append(f"{metric}: {len(flagged)} flagged this session, "
                             f"only {CAP_PER_METRIC} shown per the 2026-09-02 display cap.")
        elif flagged:
            coverage.append(f"{metric}: {len(flagged)} flagged this session, all shown.")
        else:
            coverage.append(f"{metric}: ran, 0 flagged this session.")
    return {"errors": errors, "_coverage_notes": coverage}


def _summarize_structure_breadth(pipeline_result: dict) -> dict | None:
    if "STRUCTURE_BREADTH" not in pipeline_result.get("results", {}):
        return None
    return {
        "breadthScore": pipeline_result.get("structure_breadth_score"),
        "labels": pipeline_result.get("structure_breadth_labels", []),
    }


def _summarize_unfilled_pause(pipeline_result: dict) -> dict | None:
    """2026-09-06 rewrite for UNFILLED_PAUSE's new direct-computation shape
    (see module docstring) -- one entry per flagged pause now, not one per
    window covering every boundary. No more trustworthy-vs-suspect
    breakdown to report (that verification step doesn't exist anymore,
    see prompts/unfilled_pause.py's accuracy-tradeoff note); this is now a
    plain count plus a boundary_type breakdown (how many of the flagged
    pauses landed at a sentence boundary, a clause boundary, or mid-clause)
    since that's real information direct_computation.compute_unfilled_
    pause() now attaches to every occurrence.

    Takes the whole pipeline_result (not just results) because the count
    alone, with no session-level word_count to divide by yet, isn't worth
    turning into a rate here -- see module docstring's UNFILLED_PAUSE note.
    """
    results = pipeline_result.get("results", {})
    if "UNFILLED_PAUSE" not in results:
        return None
    occurrences = [e["output"] for e in results["UNFILLED_PAUSE"] if e["output"]]
    by_boundary = {"sentence": 0, "clause": 0, "none": 0}
    for out in occurrences:
        by_boundary[out.get("boundary_type", "none")] += 1
    return {
        "pauseCount": pipeline_result.get("unfilled_pause_count", len(occurrences)),
        "byBoundaryType": by_boundary,
        "_coverage_note": (
            "Raw count only -- no rate yet, since word_count isn't part of "
            "this pipeline's output (that's a sessions-table field per "
            "lomb_metric_architecture_v1.md, not computed here). No ASR-"
            "boundary trustworthiness check is performed as of 2026-09-06 "
            "(see prompts/unfilled_pause.py) -- a mis-timed ASR boundary "
            "could still inflate or hide a real pause."
        ),
    }


def build_report(pipeline_result: dict) -> dict:
    results = pipeline_result.get("results", {})
    return {
        "session": {
            "targetSpeakerId": pipeline_result.get("target_speaker_id"),
            "sentenceCount": pipeline_result.get("sentence_count"),
            # windowCount removed 2026-09-06: UNFILLED_PAUSE/FILLED_PAUSE no
            # longer use word-timestamp windows (direct computation scans
            # words directly) -- see pipeline.py's run_pipeline() docstring.
        },
        "fluency": {
            "wpm": pipeline_result.get("wpm"),
            "ohRate": None,
            "ahRate": None,
            "_coverage_note": (
                "ohRate/ahRate are Python-logic metrics, not yet computed by "
                "this pipeline (see claude/lomb_metric_definitions_v1.md). "
                "wpm IS computed as of 2026-09-07 (direct_computation."
                "compute_wpm()) -- but it's a convenience value derived "
                "here, not yet a real Metric Aggregator's stored output; "
                "None if this session had zero speaking time."
            ),
            "unfilledPause": _summarize_unfilled_pause(pipeline_result),
        },
        "accuracy": _build_accuracy(results),
        "complexity": {
            "ttr": None,
            "vocdD": None,
            "wuerde": None,
            "hypothetical": None,
            "_coverage_note": (
                "ttr/vocdD/wuerde/hypothetical are Python-logic metrics, not "
                "computed by this pipeline. structureBreadth below IS from "
                "this run (STRUCTURE_BREADTH, LLM-assisted)."
            ),
            "structureBreadth": _summarize_structure_breadth(pipeline_result),
        },
    }


def _tokenize(sentence: str) -> list[str]:
    # Whitespace-split, not stripped of punctuation -- punctuation stays
    # attached to its word (e.g. "USA." not "USA" + "."), matching how
    # these sentences already appear word-by-word elsewhere in this
    # codebase (see speaker_filter.to_word_timestamp_windows()'s own
    # whitespace-based word handling). Good enough for highlighting
    # purposes; not claiming linguistic tokenization correctness.
    return sentence.split()


def _diff_html(bad: str, good: str | None) -> tuple[str, str]:
    """Word-level diff between the original (bad) sentence and the
    model's own corrected (good) sentence, returning (bad_html,
    good_html) with ONLY the differing span(s) wrapped in a highlight
    span -- red for what's removed/changed in `bad`, green for what's
    added/changed in `good`. Unchanged words are plain text.

    Deliberately uses Python's own difflib.SequenceMatcher on WORD
    tokens (not characters, not an LLM-labeled span) -- a real,
    deterministic diff against two full sentences the model already had
    to produce, not a guess about which part is "the wrong part."

    If `good` is missing entirely (an older pipeline_result.json that
    predates the `corrected` schema field -- see module docstring),
    there's nothing to diff against: returns the bad sentence
    unhighlighted plus a placeholder explaining why, rather than
    inventing a "corrected" sentence that was never actually generated.
    """
    if good is None:
        return (escape(bad), '<span class="missing">(corrected form not available -- '
                             're-run with the updated prompts to get this)</span>')

    bad_words = _tokenize(bad)
    good_words = _tokenize(good)
    matcher = difflib.SequenceMatcher(None, bad_words, good_words)

    bad_parts, good_parts = [], []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        bad_span = escape(" ".join(bad_words[i1:i2]))
        good_span = escape(" ".join(good_words[j1:j2]))
        if op == "equal":
            bad_parts.append(bad_span)
            good_parts.append(good_span)
        else:  # replace, delete, insert
            if bad_span:
                bad_parts.append(f'<span class="bad-hl">{bad_span}</span>')
            if good_span:
                good_parts.append(f'<span class="good-hl">{good_span}</span>')

    return (" ".join(p for p in bad_parts if p), " ".join(p for p in good_parts if p))


def render_html(report: dict) -> str:
    session = report["session"]
    accuracy = report["accuracy"]
    fluency = report["fluency"]
    complexity = report["complexity"]

    cards = []
    for e in accuracy["errors"]:
        bad_html, good_html = _diff_html(e["bad"], e["good"])
        cards.append(f"""
        <div class="card">
          <span class="tag">{escape(e['tag'])}</span>
          <div class="sentence bad-sentence">{bad_html}</div>
          <div class="sentence good-sentence">{good_html}</div>
        </div>""")
    error_cards = "".join(cards) or "<p><em>No errors flagged (or no error metrics have results yet).</em></p>"

    coverage_items = "".join(f"<li>{escape(n)}</li>" for n in accuracy["_coverage_notes"])

    up = fluency.get("unfilledPause")
    if up:
        by_boundary = up.get("byBoundaryType", {})
        up_html = f"""
        <p><strong>{up['pauseCount']}</strong> unfilled pause(s) found
        (sentence boundary: {by_boundary.get('sentence', 0)},
        clause boundary: {by_boundary.get('clause', 0)},
        mid-clause: {by_boundary.get('none', 0)}).</p>
        <p class="notice">{escape(up.get('_coverage_note', ''))}</p>
        """
    else:
        up_html = "<p><em>Not run this session.</em></p>"

    wpm_value = fluency.get("wpm")
    wpm_html = (
        f"<p>Speed: <strong>{wpm_value}</strong> words per minute.</p>"
        if wpm_value is not None else
        "<p><em>Not available (zero speaking time detected this session).</em></p>"
    )

    sb = complexity.get("structureBreadth")
    if sb:
        sb_html = f"<p>Breadth score: <strong>{sb['breadthScore']}</strong> distinct structures used " \
                   f"({', '.join(sb['labels']) if sb['labels'] else 'none'}).</p>"
    else:
        sb_html = "<p><em>Not run this session.</em></p>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Lomb speech report (draft)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: .75rem 1rem; margin: .75rem 0; background: #fafafa; }}
.tag {{ display: inline-block; background: #eee; border-radius: 12px; padding: .1rem .6rem; font-size: .8rem; margin-bottom: .5rem; }}
.sentence {{ font-size: 1rem; line-height: 1.5; }}
.bad-sentence {{ margin-bottom: .25rem; }}
.bad-hl {{ color: #b3261e; font-weight: 600; }}
.good-hl {{ color: #1e7a34; font-weight: 600; }}
.missing {{ color: #888; font-style: italic; font-weight: 400; }}
.warn {{ color: #b3261e; }}
.notice {{ background: #fff8e1; border: 1px solid #f0d98a; border-radius: 8px; padding: .75rem 1rem; font-size: .9rem; }}
ul.coverage {{ font-size: .85rem; color: #555; }}
</style></head><body>

<h1>Lomb speech report (draft)</h1>
<p class="notice">This is a draft report built directly from a real pipeline run, not the final product UI.
Coverage is partial -- see the notes under each section for exactly what is and isn't included yet.</p>

<p>Speaker: <strong>{escape(str(session['targetSpeakerId']))}</strong> &middot;
{session['sentenceCount']} sentences analyzed.</p>

<h2>Accuracy — flagged errors</h2>
{error_cards}
<ul class="coverage">{coverage_items}</ul>

<h2>Fluency — speed &amp; unfilled pauses</h2>
{wpm_html}
{up_html}
<p class="notice">Hesitation ("oh"/"ah") rates are not shown here — they come from a
separate, not-yet-built module.</p>

<h2>Complexity — sentence structure breadth</h2>
{sb_html}
<p class="notice">Lexical diversity (TTR/vocd-D) and other complexity numbers are not shown here for the same reason.</p>

</body></html>"""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python report.py pipeline_result.json", file=sys.stderr)
        return 1

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        pipeline_result = json.load(f)

    report = build_report(pipeline_result)

    with open("report_contract.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Wrote report_contract.json")

    with open("report.html", "w", encoding="utf-8") as f:
        f.write(render_html(report))
    print("Wrote report.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
