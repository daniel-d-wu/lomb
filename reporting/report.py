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

- `fluency.ohRate`, `ahRate` and `complexity.ttr`, `vocdD`, `wuerde`,
  `hypothetical` are NOT computed here -- 6 fields, not "WPM plus 5"
  (this docstring previously said WPM belonged to this NOT-computed list;
  it doesn't, see the correction below). These are Python-logic metrics
  ("Oh"/"Äh" rate, Lexical Diversity, Missing Active Vocabulary, Missing
  Syntactic Structures, Preterite Avoidance -- see
  claude/lomb_metric_definitions_v1.md) -- a completely separate,
  not-yet-built module that this LLM-assisted pipeline was never meant
  to cover. Left null with a coverage note, not estimated.

- `fluency.wpm` IS computed here (direct_computation.compute_wpm(), as of
  2026-09-07) -- Dan confirmed 2026-09-23 this one is required, not
  deferrable like the 6 above. This paragraph used to lump it in with
  the NOT-computed list above; that was stale relative to the code below
  it, which already returns a real value (see the `fluency` dict's own
  `_coverage_note` for the caveat that it's a convenience value, not yet
  a stored Metric Aggregator output -- that caveat still stands, this
  correction is only about whether a number comes back at all).

- UNFILLED_PAUSE is now surfaced as a plain occurrence count (see
  _summarize_unfilled_pause()'s own docstring), not a rate -- word_count
  isn't threaded through this pipeline yet (that's a `sessions`-table
  field, per lomb_metric_architecture_v1.md, not something pipeline_result.
  json carries today), so `unfilled_pause_rate` can't be computed here
  without a number to divide by. Left as a raw count with a coverage note,
  not estimated against a guessed denominator.

- FORMULAIC and FILLED_PAUSE are not in ERROR_METRICS below and never
  contribute to accuracy.errors[] -- only fluencemes whose prompt file sets
  a report1_tag do (see ERROR_METRICS below); that's a reporting-scope decision, unrelated to
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
from collections import defaultdict
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.flag_quality import changed_word_count, fix_signature, rejection_reason  # noqa: E402
from pipeline.registry import METRIC_PROMPTS, REPORT1_ERROR_METRICS  # noqa: E402

# Report 1's accuracy.errors[] metrics and their labels come from each
# prompt file's report1_tag/report1_order (2026-09-24) -- a new fluenceme
# with a report1_tag shows up here automatically. A designated metric
# absent from a given pipeline_result.json is reported as "not wired",
# distinct from "ran, found nothing."
TAGS = {key: METRIC_PROMPTS[key].report1_tag for key in REPORT1_ERROR_METRICS}
ERROR_METRICS = REPORT1_ERROR_METRICS
CAP_PER_METRIC = 2
CAP_TOTAL = CAP_PER_METRIC * len(ERROR_METRICS)  # 2 per Report 1 metric (2026-09-02 decision)


def _card_rank(entry: dict) -> tuple:
    """Clearer lesson first: fewest words changed, then shorter sentence."""
    good = entry["output"].get("corrected") or entry["input"]
    return (changed_word_count(entry["input"], good), len(entry["input"].split()))


def _build_accuracy(results: dict) -> dict:
    """Pick which flagged errors a learner sees, per metric (2026-09-24,
    replacing "first 2 in sentence order"):

    1. Only trustworthy flags (pipeline/flag_quality.py) -- a wrong or empty
       correction teaches the wrong thing.
    2. Same mistake with the same word repeated -> ONE card, with
       `occurrences`; habits first, since they're the most useful to fix.
    3. Then the clearest lesson: fewest words changed, then shortest sentence.
    4. Each sentence appears once across all metrics; other error types found
       in it are listed in `alsoTags` instead of repeating the sentence.
    Metrics are visited in report1_order, so the higher-priority metric
    claims a shared sentence. Cards are ordered repeated-mistakes-first.
    """
    candidates: dict[str, list[dict]] = {}
    rejected: dict[str, int] = {}
    tags_by_sentence: dict[str, list[str]] = defaultdict(list)
    coverage = []
    for metric in ERROR_METRICS:
        if metric not in results:
            continue
        flagged = [e for e in results[metric]
                   if not e["skipped"] and e["output"] is not None and e["output"].get("error") is True]
        valid = [e for e in flagged if rejection_reason(e["output"], e["input"]) is None]
        candidates[metric], rejected[metric] = valid, len(flagged) - len(valid)
        for e in valid:
            tags_by_sentence[e["input"]].append(metric)

    cards, shown_sentences = [], set()
    for rank, metric in enumerate(ERROR_METRICS):
        if metric not in results:
            coverage.append(f"{metric}: not yet wired into this pipeline run "
                            f"(see pipeline.py's scope notes) -- absent, not zero.")
            continue
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for e in candidates[metric]:
            groups[fix_signature(e["input"], e["output"].get("corrected") or e["input"])].append(e)
        ranked_groups = sorted(groups.values(), key=lambda g: (-len(g), _card_rank(min(g, key=_card_rank))))

        shown = 0
        for group in ranked_groups:
            example = next((e for e in sorted(group, key=_card_rank) if e["input"] not in shown_sentences), None)
            if example is None:
                continue
            shown_sentences.add(example["input"])
            cards.append((-len(group), rank, {
                "tag": TAGS[metric],
                "bad": example["input"],
                # None when this pipeline_result.json predates the
                # `corrected` schema field -- see module docstring.
                "good": example["output"].get("corrected"),
                "why": example["output"]["reasoning"],
                "occurrences": len(group),
                "alsoTags": [TAGS[m] for m in tags_by_sentence[example["input"]] if m != metric],
            }))
            shown += 1
            if shown == CAP_PER_METRIC:
                break

        n_valid = len(candidates[metric])
        note = f"{metric}: {n_valid} flagged this session, {shown} shown"
        if rejected[metric]:
            note += f" ({rejected[metric]} more auto-rejected as untrustworthy, not counted)"
        coverage.append(note + ".")

    cards.sort(key=lambda c: (c[0], c[1]))
    return {"errors": [card for _, _, card in cards], "_coverage_notes": coverage}


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
        repeat = (f'<span class="tag repeat">Seen {e["occurrences"]}× this session</span>'
                  if e.get("occurrences", 1) > 1 else "")
        also = (f'<div class="also">Also in this sentence: {escape(", ".join(e["alsoTags"]))}</div>'
                if e.get("alsoTags") else "")
        cards.append(f"""
        <div class="card">
          <span class="tag">{escape(e['tag'])}</span>{repeat}
          <div class="sentence bad-sentence">{bad_html}</div>
          <div class="sentence good-sentence">{good_html}</div>
          {also}
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
.tag.repeat {{ background: #fdf0d5; margin-left: .4rem; }}
.also {{ font-size: .8rem; color: #666; margin-top: .35rem; }}
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

    # Write next to this script (reporting/), not whatever the caller's cwd
    # happens to be -- matters now that report.py doesn't live at repo root.
    here = Path(__file__).resolve().parent

    contract_path = here / "report_contract.json"
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {contract_path}")

    html_path = here / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(report))
    print(f"Wrote {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
