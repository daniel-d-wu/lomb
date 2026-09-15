"""
Orchestration layer: raw AssemblyAI transcript JSON -> per-metric
classifications, for all 11 registered metrics (7 LLM-assisted + 4
direct/deterministic) -- the full Phase 1 Fluenceme scope Dan locked in
(9 originally-LLM-assisted + FORMULAIC + WPM, with UNFILLED_PAUSE/
FILLED_PAUSE having since moved off the LLM -- see the 2026-09-06 and
2026-09-07 paragraphs below).

Ties together every piece built so far, in order:
  assemblyai_adapter.from_assemblyai_transcript()  raw JSON -> Turn objects
  speaker_filter.filter_to_target_speaker()        -> one speaker's turns only
  speaker_filter.to_sentences() / to_sentence_windows() /
    find_formulaic_matches()                       -> per-metric input shapes
  prefilter.should_run()                           skip LLM calls where safe
  registry.classify()                              the actual per-metric call
                                                    (7 metrics; FORMULAIC,
                                                    UNFILLED_PAUSE,
                                                    FILLED_PAUSE, WPM never
                                                    reach this)
  direct_computation.compute_filled_pause() /
    compute_unfilled_pause() / compute_wpm()       the other 4 direct metrics
  -> a session-level breadth_score aggregation for STRUCTURE_BREADTH

SCOPE, stated explicitly: as of 2026-09-07, this runs all 11 registered
metrics -- the full Phase 1 set. Four groups, by how they're computed:

  input_kind == "sentence", one sentence per call:
    GDD-1, GDD-2, GVT-2, LPF, LP, STRUCTURE_BREADTH
  input_kind == "sentence", but a WINDOW of several consecutive sentences
  per call (see speaker_filter.to_sentence_windows()):
    GVT-1
  NOT an LLM call at all -- direct/deterministic computation, zero cost:
    FORMULAIC (regex scan, speaker_filter.find_formulaic_matches())
    FILLED_PAUSE (fixed token lookup, direct_computation.compute_filled_pause())
    UNFILLED_PAUSE (fixed threshold check, direct_computation.compute_unfilled_pause())
    WPM (session-level word/duration tally, direct_computation.compute_wpm())

2026-09-07, per Dan's decision to build out Phase 1's last remaining
Fluenceme: WPM added, as a direct computation from the start (it was
never an LLM metric in the first place -- see lomb_metric_definitions_v1.
md's own Direct/LLM split). Structurally different from every other
Fluenceme here: its Information is session-level (word_count,
duration_seconds), not one entry per sentence/occurrence -- see
direct_computation.compute_wpm()'s own docstring for how that's still
made to fit this module's uniform per-metric list shape (always exactly
one entry, sentence_indices always empty). This module also computes a
convenience `wpm` value (word_count / duration-in-minutes) at the
top level of its return dict, the same status as structure_breadth_score
already has here -- a real number, useful now, but not the Metric
Aggregator's eventual stored output (see compute_wpm()'s docstring for
why that distinction matters).

2026-09-06, per Dan's explicit instruction: UNFILLED_PAUSE and
FILLED_PAUSE moved off the LLM entirely, same treatment FORMULAIC got the
day before. Both used to run once per word-timestamp window
(to_word_timestamp_windows()) through registry.classify(); now each has
its own function in direct_computation.py that scans the target speaker's
words directly (a fixed hesitation-token lookup for FILLED_PAUSE, a fixed
>500ms gap threshold for UNFILLED_PAUSE) and reports one entry per actual
occurrence found, not one per window checked -- same granularity change
FORMULAIC went through on 2026-09-05, for the same reason (see
direct_computation.py's own module docstring: this is what makes an
occurrence's shape match what information_items is actually meant to
hold). to_word_timestamp_windows() itself is now orphaned, same as
to_audio_turns() before it -- nothing in this module calls it anymore,
but it's left in speaker_filter.py rather than deleted, on the same
"still a correct, independently useful building block" reasoning that
function's own docstring already states for itself. The accuracy tradeoff
this accepts for UNFILLED_PAUSE specifically (no more ASR-boundary
trustworthiness verification) is real and stated plainly in
prompts/unfilled_pause.py's docstring, not glossed over here.

REGEX_METRICS was renamed DIRECT_METRICS accordingly -- it was never
strictly a "regex" group once it needs to also mean "a fixed-threshold
arithmetic check" and "a fixed-token lookup," and calling it DIRECT_METRICS
now matches the Direct-vs-LLM computation-path language this project
settled on in lomb_metric_architecture_v1.md.

2026-09-05, per Dan's explicit instruction: FORMULAIC moved from an
LLM-assisted candidate-scan metric to a fully deterministic regex one.
Previously it ran once per (candidate, sentence) pair through
registry.classify() same as every other metric, disambiguating a literal
reading of a BUNDLES word from a formulaic/discourse-particle one in
context; now a BUNDLES match IS the verdict, with zero LLM calls and zero
cost, at the price of losing that disambiguation (see prompts/formulaic.py's
module docstring for the full tradeoff). Concretely: FORMULAIC was removed
from registry.METRIC_PROMPTS entirely (registry.py now asserts it's
absent, not just present-but-unused), speaker_filter.to_formulaic_
candidates() was renamed to find_formulaic_matches() and now returns plain
{candidate, sentence} records instead of LLM-prompt-formatted strings.

2026-09-03, three earlier changes, each documented in the module/function
they actually live in rather than re-explained here:

  GVT-1 moved from excluded to running, via speaker_filter.to_sentence_
  windows() -- it was excluded before because to_sentences() output was
  being fed one sentence at a time, same as every other sentence metric,
  which meant GVT-1 could never actually see the past-tense frame and the
  later reversion in the same request. See to_sentence_windows()'s own
  docstring for the real corpus example that proved this.

  FORMULAIC moved from excluded to running (as an LLM metric at the time;
  see the 2026-09-05 paragraph above for what changed since), via
  prompts/formulaic.py's new BUNDLES reference list (previously
  nonexistent -- confirmed by reading lomb_metric_definitions_v1.md
  directly, not assumed).

  FILLED_PAUSE moved from excluded to running (as an LLM metric at the
  time; see the 2026-09-06 paragraph above for what changed since), by
  being redefined off input_kind "audio_turn" onto transcript-text-based
  detection -- per Dan's direct instruction to measure it from the
  transcript rather than the audio.

This module does NOT retry, rate-limit, batch, or parallelize calls --
that's production-hardening, out of scope for a first working pipeline
pass. It has NOT been run against a live API from inside this sandbox
(same api.openai.com egress block documented throughout this project,
in providers/openai_provider.py and elsewhere) -- see the __main__ block
and its FakeProvider for exactly what that block does and doesn't prove.
"""

import sys
from pathlib import Path

# repo root on sys.path -- prompts/ and transcript_processing/ are both
# siblings of this file's own new directory (pipeline/) post-reorg.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import prefilter
from pipeline.direct_computation import compute_filled_pause, compute_unfilled_pause, compute_wpm
from prompts.formulaic import BUNDLES as FORMULAIC_BUNDLES
from pipeline.registry import METRIC_PROMPTS, classify
from transcript_processing.assemblyai_adapter import from_assemblyai_transcript
from transcript_processing.speaker_filter import (
    filter_to_target_speaker,
    find_formulaic_matches,
    to_sentences,
    to_sentence_windows,
)

SENTENCE_METRICS = ["GDD-1", "GDD-2", "GVT-2", "LPF", "LP", "STRUCTURE_BREADTH"]
WINDOWED_SENTENCE_METRICS = ["GVT-1"]
_EXCLUDED = frozenset()

# All four direct/deterministic metrics -- zero calls to classify()/
# registry.py for any of them, see this module's docstring. Deliberately
# NOT unioned into _all_groups below: those groups partition
# registry.METRIC_PROMPTS (7 keys now), and none of these four live in
# that dict -- including them there would break the partition rather than
# prove it. DIRECT_METRICS is its own separate, explicit list instead,
# checked against the full 11-metric set in its own assertion right after.
DIRECT_METRICS = ["FORMULAIC", "FILLED_PAUSE", "UNFILLED_PAUSE", "WPM"]

# Catch a typo or scope drift here immediately (import time), not silently
# at runtime: every metric in the three groups below must actually have
# the input_kind this module assumes for it, and those three groups (run
# as single sentence, run as a sentence window, deliberately excluded)
# must partition all 7 keys in registry.METRIC_PROMPTS with nothing left
# over and nothing double-counted. _EXCLUDED is empty now, but stays in
# the partition (rather than being deleted) so a future metric that
# genuinely can't run yet has an obvious, already-wired place to go -- and
# so this assertion keeps proving "nothing was silently dropped," not just
# "nothing was silently dropped as of today."
assert all(METRIC_PROMPTS[k].input_kind == "sentence" for k in SENTENCE_METRICS)
assert all(METRIC_PROMPTS[k].input_kind == "sentence" for k in WINDOWED_SENTENCE_METRICS)
_all_groups = [SENTENCE_METRICS, WINDOWED_SENTENCE_METRICS, _EXCLUDED]
assert set().union(*_all_groups) == set(METRIC_PROMPTS)
assert sum(len(g) for g in _all_groups) == len(METRIC_PROMPTS), (
    "a metric key appears in more than one group -- that's double-counting, not just scope drift"
)

# The FULL metric set this module actually reports on -- the 7 LLM-backed
# keys registry.py knows about, PLUS the 4 direct ones, which deliberately
# aren't in that dict. This is the "11 registered metrics" count this
# module's docstring promises -- the complete Phase 1 scope -- proven here
# rather than just asserted.
for _direct_key in DIRECT_METRICS:
    assert _direct_key not in METRIC_PROMPTS, (
        f"{_direct_key} must not be registered as an LLM metric -- it's direct/deterministic now"
    )
_ALL_REPORTED_METRICS = set(METRIC_PROMPTS) | set(DIRECT_METRICS)
assert len(_ALL_REPORTED_METRICS) == 11, (
    f"expected 7 LLM metrics + 4 direct metrics == 11 total (the full Phase 1 scope), "
    f"got {sorted(_ALL_REPORTED_METRICS)}"
)


def run_pipeline(provider, assemblyai_json: dict, target_speaker_id: str) -> dict:
    """The one function this module exists to provide.

    provider: any llm_provider.LLMProvider instance -- OpenAIProvider() in
      production per providers/openai_provider.py's 2026-09-03 switch to
      primary, but deliberately accepted as a parameter rather than
      imported and hardcoded here, so this stays swappable (this module
      never needs to know or care which provider answered) and testable
      (the __main__ block below passes a FakeProvider instead).
    assemblyai_json: one decoded GET /v2/transcript/{id} response (a
      dict) -- see assemblyai_adapter.py for the shape this needs.
    target_speaker_id: which diarized speaker to run metrics against.
      lomb_backend_prd_v1.md Section 6.2 requires the visitor to confirm
      this before any metrics run -- this function does NOT do that
      confirmation step itself, only the resulting filtering; pass
      whatever speaker_id the confirmation step already validated.

    Returns:
      {
        "target_speaker_id": str,
        "sentence_count": int,
        "gvt1_window_count": int,          # sentence-windows for GVT-1
        "formulaic_candidate_count": int,  # regex matches found for FORMULAIC
        "filled_pause_count": int,         # occurrences found for FILLED_PAUSE
        "unfilled_pause_count": int,       # occurrences found for UNFILLED_PAUSE
        "word_count": int,                 # WPM's Information -- total words, this session
        "duration_seconds": float,         # WPM's Information -- speaking time, this session
        "wpm": float | None,               # convenience Metric (word_count / duration
                                            # in minutes), None if duration_seconds is 0 --
                                            # see compute_wpm()'s docstring for why this is
                                            # a convenience value, not the eventual Metric
                                            # Aggregator's stored output
        "results": {
          metric_key: [
            {"input": <sentence str, sentence-window str, or a short
                       display string for a direct-computation occurrence
                       ("Candidate: ... | Sentence: ...", "Word: ...",
                       "Gap: ... -> ...", or "session" for WPM)>,
             "skipped": bool,       # True only for prefilter-skipped GDD-1/GDD-2 calls
             "output": dict | None} # classify()'s return value, or the
                                     # equivalent fixed-shape dict for a
                                     # direct-computation path (FORMULAIC/
                                     # FILLED_PAUSE/UNFILLED_PAUSE/WPM, none
                                     # of which ever call classify()), or
                                     # None if skipped
            , ...
          ]
          for each of the 11 metric keys above (7 via classify(), 4 via
          direct computation -- see this module's docstring)
        },
        "structure_breadth_score": int,   # count of DISTINCT non-"none"
                                           # labels seen anywhere this
                                           # session (session-level, per
                                           # lomb_metric_definitions_v1.md
                                           # -- not summed per-sentence)
        "structure_breadth_labels": list[str],  # sorted, for inspection
      }

    NOTE on window_count (removed 2026-09-06): earlier versions of this
    dict had a "window_count" key -- word-timestamp windows shared by
    UNFILLED_PAUSE/FILLED_PAUSE when both were LLM-assisted. Neither
    metric uses windows anymore (direct computation scans the full word
    list directly, no batching needed), so there is no longer a concept
    of "window count" for anything this function computes. Removed rather
    than kept at a meaningless 0, per this project's stated preference for
    an absent field over a fabricated number.
    """
    turns = from_assemblyai_transcript(assemblyai_json)
    target_turns = filter_to_target_speaker(turns, target_speaker_id)
    sentences = to_sentences(target_turns)
    sentence_windows = to_sentence_windows(target_turns)
    formulaic_matches = find_formulaic_matches(target_turns, FORMULAIC_BUNDLES)
    filled_pause_occurrences = compute_filled_pause(target_turns)
    unfilled_pause_occurrences = compute_unfilled_pause(target_turns)
    wpm_occurrence = compute_wpm(target_turns)

    results = {}

    for metric_key in SENTENCE_METRICS:
        per_sentence = []
        for sentence in sentences:
            if not prefilter.should_run(metric_key, sentence):
                per_sentence.append({"input": sentence, "skipped": True, "output": None})
                continue
            output = classify(provider, metric_key, sentence)
            per_sentence.append({"input": sentence, "skipped": False, "output": output})
        results[metric_key] = per_sentence

    # GVT-1 is unfiltered (prefilter.py only ever safely covers GDD-1/
    # GDD-2's fixed-preposition lookup) and runs once per sentence-WINDOW,
    # not once per sentence -- see to_sentence_windows()'s docstring for
    # why a single sentence at a time could never have worked for this
    # metric.
    for metric_key in WINDOWED_SENTENCE_METRICS:
        per_window = []
        for window in sentence_windows:
            output = classify(provider, metric_key, window)
            per_window.append({"input": window, "skipped": False, "output": output})
        results[metric_key] = per_window

    # All four DIRECT_METRICS: NOT a classify() call at all -- each is
    # already a fully-built list of occurrence entries (find_formulaic_
    # matches() / compute_filled_pause() / compute_unfilled_pause() /
    # compute_wpm(), all computed above), zero LLM calls, zero cost. The
    # "input"/"skipped"/"output" shape is kept identical to every
    # classify()-backed metric's entries (same keys, same meaning) purely
    # so downstream consumers (report.py, this module's own __main__
    # checks) don't need a special case just to read these metrics'
    # results -- "skipped" is always False for all four (there's no
    # LLM-call decision to skip). See direct_computation.py's module
    # docstring for the granularity/shape reasoning this shares across all
    # four (WPM's own docstring explains its one exception: exactly one
    # session-level entry rather than one per occurrence), and
    # prompts/formulaic.py / prompts/filled_pause.py /
    # prompts/unfilled_pause.py for each pause metric's own accuracy
    # tradeoff.
    results["FORMULAIC"] = [
        {
            "input": f'Candidate: "{match["candidate"]}" | Sentence: "{match["sentence"]}"',
            "skipped": False,
            "output": {
                "formulaic": True,
                "confidence": "high",
                "reasoning": (
                    "Deterministic regex match against the BUNDLES reference list "
                    "(prompts/formulaic.py) -- no LLM literal-vs-formulaic "
                    "disambiguation as of 2026-09-05, per Dan's explicit instruction "
                    "to make FORMULAIC a regex-based metric. See that module's "
                    "docstring for the accuracy tradeoff this accepts."
                ),
            },
        }
        for match in formulaic_matches
    ]
    results["FILLED_PAUSE"] = filled_pause_occurrences
    results["UNFILLED_PAUSE"] = unfilled_pause_occurrences
    results["WPM"] = wpm_occurrence

    # Session-level STRUCTURE_BREADTH aggregation: the UNION of distinct
    # structure labels actually produced across every non-skipped sentence
    # call this session, excluding "none" -- that label means "no listed
    # structure detected in THIS sentence," it isn't itself a structure
    # type to count toward breadth. Using a set (not a running count) is
    # what makes this correctly session-level rather than per-sentence:
    # the same label appearing in 5 different sentences still counts once.
    breadth_labels = set()
    for entry in results["STRUCTURE_BREADTH"]:
        if entry["skipped"] or entry["output"] is None:
            continue
        for label in entry["output"].get("structures", []):
            if label != "none":
                breadth_labels.add(label)

    # WPM's convenience Metric: computed here, not in compute_wpm() itself
    # (which returns Information only -- see its docstring for why that
    # separation matters). Guarded against duration_seconds == 0 (an
    # empty/degenerate turn set) -- None, not a ZeroDivisionError or a
    # fabricated 0.0.
    wpm_info = wpm_occurrence[0]["output"]
    word_count = wpm_info["word_count"]
    duration_seconds = wpm_info["duration_seconds"]
    wpm_value = round(word_count / (duration_seconds / 60), 1) if duration_seconds > 0 else None

    return {
        "target_speaker_id": target_speaker_id,
        "sentence_count": len(sentences),
        "gvt1_window_count": len(sentence_windows),
        "formulaic_candidate_count": len(formulaic_matches),
        "filled_pause_count": len(filled_pause_occurrences),
        "unfilled_pause_count": len(unfilled_pause_occurrences),
        "word_count": word_count,
        "duration_seconds": duration_seconds,
        "wpm": wpm_value,
        "results": results,
        "structure_breadth_score": len(breadth_labels),
        "structure_breadth_labels": sorted(breadth_labels),
    }


if __name__ == "__main__":
    import json

    class FakeProvider:
        """NOT a stand-in for a real LLM -- makes no linguistic judgment
        at all, and must never be mistaken for one. Its only job is to
        prove the ORCHESTRATION logic above is wired correctly without
        needing network access or a real API key, the same gap
        smoketest_openai_offline.py exists to cover for the provider
        layer itself. What this actually checks, concretely:

          1. Every non-skipped sentence/window in the pipeline really
             reaches classify() exactly once (call counts are logged).
          2. prefilter.should_run()'s skip decisions are the ones that
             actually take effect in run_pipeline() -- proven by
             asserting GDD-1/GDD-2's skipped vs. called counts against
             should_run() computed independently, not just trusting the
             pipeline's own bookkeeping.
          3. structure_breadth_score is a DISTINCT-label union, not a
             raw count -- proven by deliberately returning a repeating,
             overlapping label sequence and asserting the aggregated
             score is smaller than the number of calls that produced it.

        What this does NOT check: whether any answer is linguistically
        correct. That is exactly what test_all_metrics_live.py already
        confirmed (7/7) for single calls -- this script's job is the
        wiring between calls, not the calls' own correctness.
        """

        def __init__(self):
            self.call_count = 0
            self.calls_by_metric = {}
            self._structure_cycle = [
                ["konjunktiv_ii"], ["dass_clause"], ["konjunktiv_ii"],
                ["none"], ["weil_clause"], ["dass_clause"],
            ]
            self._structure_i = 0

        def classify(self, config, input_data):
            self.call_count += 1
            self.calls_by_metric[config.key] = self.calls_by_metric.get(config.key, 0) + 1

            if config.key == "STRUCTURE_BREADTH":
                labels = self._structure_cycle[self._structure_i % len(self._structure_cycle)]
                self._structure_i += 1
                return {"structures": labels, "confidence": "high"}

            # FORMULAIC, UNFILLED_PAUSE, FILLED_PAUSE, and WPM are all
            # deliberately absent from this branch: all four are
            # direct/deterministic computations, none of them ever reach
            # classify() at all (see run_pipeline()'s DIRECT_METRICS
            # handling) -- if this ever fires with config.key equal to any
            # of them, that itself would mean the metric regressed back
            # onto the LLM path, which Check 2c/2d/2e below explicitly
            # guard against (asserting zero classify() calls for each).

            # GDD-1, GDD-2, GVT-2, LPF, LP all share the same
            # {error, confidence, reasoning} shape (verified against
            # registry.py's METRIC_PROMPTS in this same conversation --
            # grep '"required":' across prompts/*.py). Fake, fixed,
            # never-flagged answer: this provider is not trying to be
            # right, only to be reached correctly.
            return {"error": False, "confidence": "high",
                    "reasoning": "FakeProvider -- orchestration test only, not a real judgment."}

    # sample_transcript_assemblyai_v3.json lives in data/ since the
    # 2026-09 reorg; was a bare relative "sample_transcript_assemblyai.json"
    # (no _v3, and not actually present anywhere but archive/) before that.
    SAMPLE_TRANSCRIPT = Path(__file__).resolve().parent.parent / "data" / "sample_transcript_assemblyai_v3.json"
    with open(SAMPLE_TRANSCRIPT, "r", encoding="utf-8") as f:
        transcript_json = json.load(f)

    provider = FakeProvider()
    result = run_pipeline(provider, transcript_json, target_speaker_id="A")

    print(f"target_speaker_id:         {result['target_speaker_id']}")
    print(f"sentence_count:            {result['sentence_count']}")
    print(f"gvt1_window_count:         {result['gvt1_window_count']}")
    print(f"formulaic_candidate_count: {result['formulaic_candidate_count']}")
    print(f"filled_pause_count:        {result['filled_pause_count']}")
    print(f"unfilled_pause_count:      {result['unfilled_pause_count']}")
    print(f"word_count:                {result['word_count']}")
    print(f"duration_seconds:          {result['duration_seconds']}")
    print(f"wpm:                       {result['wpm']}")
    print(f"\nCalls made per metric: {provider.calls_by_metric}")
    print(f"Total classify() calls: {provider.call_count}")

    print(f"\nstructure_breadth_score: {result['structure_breadth_score']} "
          f"(labels: {result['structure_breadth_labels']})")

    print("\n--- Check 1: every metric key present, right entry count ---")
    for key in SENTENCE_METRICS:
        entries = result["results"][key]
        assert len(entries) == result["sentence_count"], f"{key}: expected one entry per sentence"
        print(f"  {key:<20} {len(entries)} entries")
    for key in WINDOWED_SENTENCE_METRICS:
        entries = result["results"][key]
        assert len(entries) == result["gvt1_window_count"], f"{key}: expected one entry per sentence-window"
        print(f"  {key:<20} {len(entries)} entries")
    _direct_expected_counts = {
        "FORMULAIC": result["formulaic_candidate_count"],
        "FILLED_PAUSE": result["filled_pause_count"],
        "UNFILLED_PAUSE": result["unfilled_pause_count"],
        "WPM": 1,  # always exactly one session-level entry -- see compute_wpm()'s docstring
    }
    for key in DIRECT_METRICS:
        entries = result["results"][key]
        assert len(entries) == _direct_expected_counts[key], (
            f"{key}: expected one entry per occurrence found"
        )
        print(f"  {key:<20} {len(entries)} entries")

    print("\n--- Check 2: prefilter skip decisions actually took effect ---")
    for key in ("GDD-1", "GDD-2"):
        entries = result["results"][key]
        actually_called = sum(1 for e in entries if not e["skipped"])
        expected_called = sum(1 for e in entries if prefilter.should_run(key, e["input"]))
        assert actually_called == expected_called == provider.calls_by_metric.get(key, 0), (
            f"{key}: pipeline called classify() {provider.calls_by_metric.get(key, 0)} times, "
            f"but {actually_called} entries are marked not-skipped and prefilter independently "
            f"says {expected_called} should have run -- these three numbers must all agree"
        )
        skipped = [e["input"] for e in entries if e["skipped"]]
        print(f"  {key:<8} called={actually_called}  skipped={len(skipped)}  "
              f"skipped sentences={skipped}")
    # Metrics with no pre-filter must never skip -- should_run() always
    # True for them, so every entry should have been called. GVT-1 is
    # unfiltered too, but its entries are sentence-WINDOWS, not single
    # sentences -- prefilter.should_run() was designed for single
    # sentences (see prefilter.py's docstring), so this checks GVT-1
    # separately rather than running should_run() against windowed input
    # it was never meant to see.
    for key in ("GVT-2", "LPF", "LP", "STRUCTURE_BREADTH"):
        entries = result["results"][key]
        assert all(not e["skipped"] for e in entries), f"{key} is unfiltered -- nothing should be skipped"
    print("  (GVT-2, LPF, LP, STRUCTURE_BREADTH correctly never skipped -- unfiltered by design)")

    print("\n--- Check 2b: GVT-1 sentence-windowing wiring ---")
    gvt1_entries = result["results"]["GVT-1"]
    assert all(not e["skipped"] for e in gvt1_entries), "GVT-1 is unfiltered -- nothing should be skipped"
    assert provider.calls_by_metric.get("GVT-1", 0) == result["gvt1_window_count"], (
        "GVT-1 should be called exactly once per sentence-window, no more, no fewer"
    )
    # Every window should span more than one sentence once there are more
    # sentences than the window size -- otherwise this degenerated back
    # into single-sentence feeding without anyone noticing.
    multi_sentence_windows = sum(1 for e in gvt1_entries if len(e["input"].split(". ")) > 1 or e["input"].count(".") > 1)
    print(f"  GVT-1 called={provider.calls_by_metric.get('GVT-1', 0)}  "
          f"windows={result['gvt1_window_count']}  "
          f"windows spanning >1 sentence={multi_sentence_windows}")
    if result["sentence_count"] > 3:
        assert multi_sentence_windows > 0, (
            "expected at least one GVT-1 window to bundle multiple sentences together -- "
            "if none do, this transcript can't actually exercise the windowing fix"
        )

    print("\n--- Check 2c: FORMULAIC regex wiring (no LLM call at all, as of 2026-09-05) ---")
    formulaic_entries = result["results"]["FORMULAIC"]
    assert all(not e["skipped"] for e in formulaic_entries), (
        "FORMULAIC is unfiltered -- the regex scan itself IS the filtering step"
    )
    # The whole point of this round's change: FORMULAIC must reach
    # classify() exactly ZERO times now, not once per match. If this ever
    # fails, FORMULAIC silently regressed back onto the LLM path.
    assert provider.calls_by_metric.get("FORMULAIC", 0) == 0, (
        "FORMULAIC must never call classify() -- it's a deterministic regex metric now, "
        "not an LLM-assisted one (see prompts/formulaic.py and this module's docstring)"
    )
    assert len(formulaic_entries) == result["formulaic_candidate_count"], (
        "FORMULAIC should produce exactly one entry per regex match, no more, no fewer"
    )
    assert all(e["input"].startswith('Candidate: "') for e in formulaic_entries), (
        "every FORMULAIC input should be a 'Candidate: ... | Sentence: ...' display string, "
        "not a bare sentence -- if this fails, the regex-match display formatting regressed"
    )
    assert all(e["output"]["formulaic"] is True for e in formulaic_entries), (
        "a regex match against BUNDLES should always report formulaic: True now -- "
        "there's no more LLM disambiguation step to ever return False for a real match"
    )
    print(f"  FORMULAIC classify() calls={provider.calls_by_metric.get('FORMULAIC', 0)} (expected 0)  "
          f"regex matches={result['formulaic_candidate_count']}")
    # This transcript's speaker A has a real corpus BUNDLES hit ("Ich suche
    # schon seit ein paar Monaten...") -- confirm the regex scan actually
    # found it, not just that the plumbing runs with zero matches.
    assert result["formulaic_candidate_count"] > 0, (
        "expected at least one BUNDLES match in this transcript (e.g. 'schon') -- "
        "if this is 0, either the transcript changed or the regex scan is broken"
    )
    for e in formulaic_entries:
        print(f"    {e['input']}")

    print("\n--- Check 2d: FILLED_PAUSE/UNFILLED_PAUSE direct-computation wiring (no LLM call at all, as of 2026-09-06) ---")
    filled_entries = result["results"]["FILLED_PAUSE"]
    unfilled_entries = result["results"]["UNFILLED_PAUSE"]
    assert all(not e["skipped"] for e in filled_entries), "FILLED_PAUSE is unfiltered -- the token scan itself IS the filter"
    assert all(not e["skipped"] for e in unfilled_entries), "UNFILLED_PAUSE is unfiltered -- the threshold check itself IS the filter"
    # The whole point of this round's change: neither must reach classify()
    # even once now. If this ever fails, the metric silently regressed
    # back onto the LLM path.
    assert provider.calls_by_metric.get("FILLED_PAUSE", 0) == 0, (
        "FILLED_PAUSE must never call classify() -- it's a direct token-lookup metric now "
        "(see prompts/filled_pause.py and direct_computation.compute_filled_pause())"
    )
    assert provider.calls_by_metric.get("UNFILLED_PAUSE", 0) == 0, (
        "UNFILLED_PAUSE must never call classify() -- it's a direct threshold-check metric now "
        "(see prompts/unfilled_pause.py and direct_computation.compute_unfilled_pause())"
    )
    # Every occurrence from either metric must carry the sentence_indices/
    # boundary_type fields the 2026-09-06 speaker_filter.
    # map_words_to_sentences() addition exists to provide -- this is the
    # actual point of building that function in the first place.
    for e in filled_entries + unfilled_entries:
        out = e["output"]
        assert "sentence_indices" in out and isinstance(out["sentence_indices"], list) and out["sentence_indices"], (
            f"every direct pause occurrence must carry a non-empty sentence_indices list, got {out}"
        )
        assert out["boundary_type"] in ("sentence", "clause", "none"), (
            f"boundary_type must be one of sentence/clause/none, got {out.get('boundary_type')!r}"
        )
    print(f"  FILLED_PAUSE classify() calls={provider.calls_by_metric.get('FILLED_PAUSE', 0)} (expected 0)  "
          f"occurrences={result['filled_pause_count']} "
          f"(0 is legitimate here -- this transcript has no planted filler tokens)")
    print(f"  UNFILLED_PAUSE classify() calls={provider.calls_by_metric.get('UNFILLED_PAUSE', 0)} (expected 0)  "
          f"occurrences={result['unfilled_pause_count']}")
    # This transcript's speaker A has one real >500ms gap ("lebe" -> "in")
    # -- confirm the direct scan actually found it, not just that the
    # plumbing runs with zero occurrences.
    assert result["unfilled_pause_count"] > 0, (
        "expected at least one real gap over threshold in this transcript -- "
        "if this is 0, either the transcript changed or the threshold scan is broken"
    )
    for e in unfilled_entries:
        print(f"    {e['input']}  sentence_indices={e['output']['sentence_indices']}  "
              f"boundary_type={e['output']['boundary_type']}")

    print("\n--- Check 2e: WPM direct-computation wiring (session-level, no LLM call) ---")
    wpm_entries = result["results"]["WPM"]
    assert len(wpm_entries) == 1, "WPM must produce exactly one session-level entry, not one per sentence"
    assert not wpm_entries[0]["skipped"], "WPM is unfiltered -- there's no LLM-call decision to skip"
    assert provider.calls_by_metric.get("WPM", 0) == 0, (
        "WPM must never call classify() -- it's a direct session-level tally "
        "(see direct_computation.compute_wpm())"
    )
    wpm_out = wpm_entries[0]["output"]
    assert wpm_out["sentence_indices"] == [], (
        "WPM's Information is session-level -- it has no sentence position, "
        "sentence_indices must be empty, not tied to any particular sentence"
    )
    # The top-level word_count/duration_seconds/wpm convenience fields must
    # agree with what's actually sitting inside the WPM occurrence itself --
    # two different views of the same numbers must never drift apart.
    assert result["word_count"] == wpm_out["word_count"]
    assert result["duration_seconds"] == wpm_out["duration_seconds"]
    if result["duration_seconds"] > 0:
        expected_wpm = round(result["word_count"] / (result["duration_seconds"] / 60), 1)
        assert result["wpm"] == expected_wpm, (
            f"convenience wpm value {result['wpm']} doesn't match word_count/duration_seconds "
            f"recomputed independently ({expected_wpm}) -- these must always agree"
        )
    print(f"  WPM classify() calls={provider.calls_by_metric.get('WPM', 0)} (expected 0)  "
          f"word_count={result['word_count']}  duration_seconds={result['duration_seconds']}  "
          f"wpm={result['wpm']}")

    print("\n--- Check 3: structure_breadth_score is a distinct-label union, not a raw count ---")
    total_structure_calls = provider.calls_by_metric["STRUCTURE_BREADTH"]
    assert total_structure_calls == result["sentence_count"]
    assert result["structure_breadth_score"] < total_structure_calls, (
        "FakeProvider's cycle deliberately repeats labels across calls -- the aggregated "
        "score must come out lower than the call count, or aggregation isn't deduplicating"
    )
    assert result["structure_breadth_labels"] == sorted(
        {"konjunktiv_ii", "dass_clause", "weil_clause"}
    ), result["structure_breadth_labels"]
    print(f"  {total_structure_calls} STRUCTURE_BREADTH calls -> "
          f"breadth_score={result['structure_breadth_score']} distinct labels "
          f"(not {total_structure_calls}) -- deduplication confirmed, 'none' correctly excluded")

    print("\nAll orchestration checks passed. "
          "Reminder: this proves the WIRING, not linguistic correctness -- "
          "that needs a real run with OpenAIProvider() and a live key.")
