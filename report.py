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

- UNFILLED_PAUSE's raw result is surfaced but flagged, not reported as a
  clean pause rate. See _summarize_unfilled_pause()'s own docstring for
  why: a transcript with mechanically uniform, zero-gap word timestamps
  (like the synthetic sample_transcript_assemblyai.json this was tested
  against) makes the model flag nearly every boundary "suspect" for a
  real, defensible reason -- real ASR output essentially never has
  perfectly abutting timestamps between every consecutive word, so a
  transcript that does looks artificial to the model, correctly. That's
  a property of the INPUT DATA being synthetic, not a bug in the metric
  or the pipeline.

- FORMULAIC and FILLED_PAUSE are not in ERROR_METRICS below and never
  contribute to accuracy.errors[] -- neither was, before OR after
  2026-09-03's wiring work, because claude/lomb_reporting_requirements_v1.
  md's own 6-metric error cap (GDD-1, GDD-2, GVT-1, GVT-2, LPF, LP) never
  included either of them; that's a reporting-scope decision, unrelated to
  whether pipeline.py runs them. As of 2026-09-03 both DO run in
  pipeline.py (FORMULAIC via a candidate-scan, FILLED_PAUSE redefined onto
  transcript word-timestamps -- see pipeline.py and prompts/formulaic.py /
  prompts/filled_pause.py for the full reasoning), so a fresh
  pipeline_result.json will have "FORMULAIC" and "FILLED_PAUSE" keys with
  real per-pair / per-window output in them -- this file just doesn't
  surface either into the HTML report yet, the same "not yet surfaced,
  not the same as zero found" distinction this module already applies to
  everything else it hasn't built a summarizer for. A pipeline_result.json
  produced before 2026-09-03 won't have those keys at all, and (like every
  other "metric absent from results" case in this file) that still
  correctly reads as a coverage note here, never a false "zero found."
  GVT-1 went through this identical "not wired in" -> "wired in, not yet
  its own report section" progression one step earlier the same day, via
  speaker_filter.to_sentence_windows() -- it single-sentence-fed before
  that, which meant it could never show the model both halves of a tense
  drift in one request; GVT-1 IS one of the 6 ERROR_METRICS, so once it
  started producing "GVT-1" keys with `corrected` fields, this file's
  existing diff-and-card logic picked it up automatically, no separate
  summarizer needed the way FORMULAIC/FILLED_PAUSE would.
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


def _summarize_unfilled_pause(results: dict) -> dict | None:
    """See module docstring's UNFILLED_PAUSE section for the full
    reasoning. Short version: this counts trustworthy vs. suspect
    boundaries and flags -- loudly -- when the suspect rate is high
    enough that the result is more likely explained by unnaturally
    uniform input timestamps than by real disfluency, so a report
    consumer doesn't mistake a synthetic-data artifact for a real
    finding about the speaker.
    """
    if "UNFILLED_PAUSE" not in results:
        return None
    all_boundaries = []
    for entry in results["UNFILLED_PAUSE"]:
        if entry["output"]:
            all_boundaries.extend(entry["output"].get("boundaries", []))
    if not all_boundaries:
        return {"boundaryCount": 0, "suspectCount": 0, "reliable": None}
    suspect = [b for b in all_boundaries if b["status"] == "suspect"]
    suspect_rate = len(suspect) / len(all_boundaries)
    return {
        "boundaryCount": len(all_boundaries),
        "suspectCount": len(suspect),
        "suspectRate": round(suspect_rate, 3),
        # Real ASR output essentially never has EVERY consecutive word
        # boundary flagged suspect -- that pattern is the model correctly
        "reliable": suspect_rate < 0.5,
        "note": (
            "Over half of all word boundaries were flagged 'suspect' -- this "
            "almost always means the input timestamps were mechanically "
            "uniform (as in a synthetic test transcript), not that the "
            "speaker actually paused unnaturally often. Treat this run's "
            "pause data as a pipeline-wiring check, not a real fluency "
            "finding, until it's re-run against real ASR output."
            if suspect_rate >= 0.5 else
            "Suspect rate is low enough to plausibly reflect real timestamp "
            "quality rather than a uniform-input artifact -- still worth a "
            "spot check against real ASR output before trusting it fully."
        ),
    }


def build_report(pipeline_result: dict) -> dict:
    results = pipeline_result.get("results", {})
    return {
        "session": {
            "targetSpeakerId": pipeline_result.get("target_speaker_id"),
            "sentenceCount": pipeline_result.get("sentence_count"),
            "windowCount": pipeline_result.get("window_count"),
        },
        "fluency": {
            "wpm": None,
            "ohRate": None,
            "ahRate": None,
            "_coverage_note": (
                "Not computed by this pipeline -- these are Python-logic "
                "metrics (see claude/lomb_metric_definitions_v1.md), a "
                "separate module not yet built."
            ),
            "unfilledPause": _summarize_unfilled_pause(results),
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
        up_html = f"""
        <p>{up['boundaryCount']} word boundaries checked, {up.get('suspectCount', 0)} flagged suspect
        ({up.get('suspectRate', 0) * 100:.0f}%).</p>
        <p class="{'warn' if up.get('reliable') is False else ''}">{escape(up.get('note', ''))}</p>
        """
    else:
        up_html = "<p><em>Not run this session.</em></p>"

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
{session['sentenceCount']} sentences &middot; {session['windowCount']} timestamp windows analyzed.</p>

<h2>Accuracy — flagged errors</h2>
{error_cards}
<ul class="coverage">{coverage_items}</ul>

<h2>Fluency — unfilled pauses</h2>
{up_html}
<p class="notice">WPM and hesitation ("oh"/"ah") rates are not shown here — they come from a
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
