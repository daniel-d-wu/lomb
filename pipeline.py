"""
Orchestration layer: raw AssemblyAI transcript JSON -> per-metric LLM
classifications, for all 10 registered metrics.

Ties together every piece built so far, in order:
  assemblyai_adapter.from_assemblyai_transcript()  raw JSON -> Turn objects
  speaker_filter.filter_to_target_speaker()        -> one speaker's turns only
  speaker_filter.to_sentences() / to_sentence_windows() /
    to_word_timestamp_windows() / to_formulaic_candidates()
                                                    -> per-metric input shapes
  prefilter.should_run()                           skip LLM calls where safe
  registry.classify()                              the actual per-metric call
  -> a session-level breadth_score aggregation for STRUCTURE_BREADTH

SCOPE, stated explicitly: as of 2026-09-03, this runs all 10 registered
metrics -- the first time that's been true. Four groups, by how they're fed:

  input_kind == "sentence", one sentence per call:
    GDD-1, GDD-2, GVT-2, LPF, LP, STRUCTURE_BREADTH
  input_kind == "sentence", but a WINDOW of several consecutive sentences
  per call (see speaker_filter.to_sentence_windows()):
    GVT-1
  input_kind == "sentence", but one (candidate, sentence) PAIR per call
  (see speaker_filter.to_formulaic_candidates()):
    FORMULAIC
  input_kind == "word_timestamps": UNFILLED_PAUSE, FILLED_PAUSE

2026-09-03, three changes landed the same day, each documented in the
module/function they actually live in rather than re-explained here:

  GVT-1 moved from excluded to running, via speaker_filter.to_sentence_
  windows() -- it was excluded before because to_sentences() output was
  being fed one sentence at a time, same as every other sentence metric,
  which meant GVT-1 could never actually see the past-tense frame and the
  later reversion in the same request. See to_sentence_windows()'s own
  docstring for the real corpus example that proved this.

  FORMULAIC moved from excluded to running, via prompts/formulaic.py's new
  BUNDLES reference list (previously nonexistent -- confirmed by reading
  lomb_metric_definitions_v1.md directly, not assumed) and the new
  speaker_filter.to_formulaic_candidates() candidate-scan builder. Per
  Dan's explicit MVP-over-perfection instruction, BUNDLES is a reasonable
  first cut, not a linguistically exhaustive list -- see that module's own
  HONESTY FLAG paragraph.

  FILLED_PAUSE moved from excluded to running, by being REDEFINED off
  input_kind "audio_turn" onto "word_timestamps" -- per Dan's direct
  instruction to measure it from the transcript rather than the audio.
  This accepts a real, stated accuracy ceiling (can't recover a filler the
  ASR silently dropped from the transcript text) -- see
  prompts/filled_pause.py's own docstring for the full reasoning. It now
  shares to_word_timestamp_windows()'s windows with UNFILLED_PAUSE rather
  than needing its own audio-slicing input builder (speaker_filter.
  to_audio_turns() is now orphaned -- see its own docstring).

This module does NOT retry, rate-limit, batch, or parallelize calls --
that's production-hardening, out of scope for a first working pipeline
pass. It has NOT been run against a live API from inside this sandbox
(same api.openai.com egress block documented throughout this project,
in providers/openai_provider.py and elsewhere) -- see the __main__ block
and its FakeProvider for exactly what that block does and doesn't prove.
"""

import prefilter
from assemblyai_adapter import from_assemblyai_transcript
from prompts.formulaic import BUNDLES as FORMULAIC_BUNDLES
from registry import METRIC_PROMPTS, classify
from speaker_filter import (
    filter_to_target_speaker,
    to_formulaic_candidates,
    to_sentences,
    to_sentence_windows,
    to_word_timestamp_windows,
)

SENTENCE_METRICS = ["GDD-1", "GDD-2", "GVT-2", "LPF", "LP", "STRUCTURE_BREADTH"]
WINDOWED_SENTENCE_METRICS = ["GVT-1"]
CANDIDATE_METRICS = ["FORMULAIC"]
WORD_TIMESTAMP_METRICS = ["UNFILLED_PAUSE", "FILLED_PAUSE"]
_EXCLUDED = frozenset()

# Catch a typo or scope drift here immediately (import time), not silently
# at runtime: every metric above must actually have the input_kind this
# module assumes for it, and the five groups (run as single sentence, run
# as a sentence window, run as a candidate/sentence pair, run as
# word_timestamps, deliberately excluded) must partition all 10 registered
# metrics with nothing left over and nothing double-counted. _EXCLUDED is
# empty now, but stays in the partition (rather than being deleted) so a
# future metric that genuinely can't run yet has an obvious, already-
# wired place to go -- and so this assertion keeps proving "nothing was
# silently dropped," not just "nothing was silently dropped as of today."
assert all(METRIC_PROMPTS[k].input_kind == "sentence" for k in SENTENCE_METRICS)
assert all(METRIC_PROMPTS[k].input_kind == "sentence" for k in WINDOWED_SENTENCE_METRICS)
assert all(METRIC_PROMPTS[k].input_kind == "sentence" for k in CANDIDATE_METRICS)
assert all(METRIC_PROMPTS[k].input_kind == "word_timestamps" for k in WORD_TIMESTAMP_METRICS)
_all_groups = [SENTENCE_METRICS, WINDOWED_SENTENCE_METRICS, CANDIDATE_METRICS, WORD_TIMESTAMP_METRICS, _EXCLUDED]
assert set().union(*_all_groups) == set(METRIC_PROMPTS)
assert sum(len(g) for g in _all_groups) == len(METRIC_PROMPTS), (
    "a metric key appears in more than one group -- that's double-counting, not just scope drift"
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
        "window_count": int,               # word-timestamp windows for
                                            # UNFILLED_PAUSE and FILLED_PAUSE
        "formulaic_candidate_count": int,  # (candidate, sentence) pairs for FORMULAIC
        "results": {
          metric_key: [
            {"input": <sentence str, sentence-window str, candidate/sentence
                       pair str, or word-timestamp window list>,
             "skipped": bool,       # True only for prefilter-skipped GDD-1/GDD-2 calls
             "output": dict | None} # classify()'s return value, or None if skipped
            , ...
          ]
          for each of the 10 metric keys above
        },
        "structure_breadth_score": int,   # count of DISTINCT non-"none"
                                           # labels seen anywhere this
                                           # session (session-level, per
                                           # lomb_metric_definitions_v1.md
                                           # -- not summed per-sentence)
        "structure_breadth_labels": list[str],  # sorted, for inspection
      }
    """
    turns = from_assemblyai_transcript(assemblyai_json)
    target_turns = filter_to_target_speaker(turns, target_speaker_id)
    sentences = to_sentences(target_turns)
    sentence_windows = to_sentence_windows(target_turns)
    windows = to_word_timestamp_windows(target_turns)
    formulaic_candidates = to_formulaic_candidates(target_turns, FORMULAIC_BUNDLES)

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

    # UNFILLED_PAUSE and FILLED_PAUSE are both unfiltered (there's no
    # closed keyword set prefilter.py could safely gate either on -- for
    # FILLED_PAUSE specifically, its own trigger tokens ARE a closed set,
    # but the metric IS the detection of those tokens, so pre-filtering on
    # them would just mean re-implementing the metric in the filter and
    # then never calling the LLM at all) and both run once per
    # word-timestamp window, sharing the exact same windows.
    for metric_key in WORD_TIMESTAMP_METRICS:
        per_window = []
        for window in windows:
            output = classify(provider, metric_key, window)
            per_window.append({"input": window, "skipped": False, "output": output})
        results[metric_key] = per_window

    # FORMULAIC is unfiltered in the "skip a call that can't possibly
    # trigger" sense (see to_formulaic_candidates()'s own docstring for why
    # candidate-scanning already IS that filtering step) and runs once per
    # (candidate, sentence) pair, not once per sentence.
    for metric_key in CANDIDATE_METRICS:
        per_candidate = []
        for candidate_pair in formulaic_candidates:
            output = classify(provider, metric_key, candidate_pair)
            per_candidate.append({"input": candidate_pair, "skipped": False, "output": output})
        results[metric_key] = per_candidate

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

    return {
        "target_speaker_id": target_speaker_id,
        "sentence_count": len(sentences),
        "gvt1_window_count": len(sentence_windows),
        "window_count": len(windows),
        "formulaic_candidate_count": len(formulaic_candidates),
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
        confirmed (9/9) for single calls -- this script's job is the
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

            if config.key == "UNFILLED_PAUSE":
                return {
                    "boundaries": [
                        {"between": [w["word"] for w in input_data[i:i + 2]] if i + 1 < len(input_data) else [],
                         "status": "trustworthy", "reasoning": None}
                        for i in range(max(len(input_data) - 1, 0))
                    ],
                    "confidence": "high",
                }

            if config.key == "FILLED_PAUSE":
                # Distinct shape from UNFILLED_PAUSE on purpose (see
                # prompts/filled_pause.py's RESPONSE_SCHEMA) -- fake, fixed
                # "no fillers found" answer. Real filler content isn't in
                # sample_transcript_assemblyai.json (a synthetic dummy
                # transcript with no aeh/aehm tokens planted in it), so
                # this FakeProvider isn't trying to simulate detecting one
                # -- only proving every word-timestamp window reaches
                # classify() and comes back in the right shape.
                return {"fillers": [], "confidence": "high"}

            if config.key == "FORMULAIC":
                # Distinct shape again (see prompts/formulaic.py's
                # RESPONSE_SCHEMA) -- fake, fixed "not formulaic" answer.
                return {"formulaic": False, "confidence": "high",
                        "reasoning": "FakeProvider -- orchestration test only, not a real judgment."}

            # GDD-1, GDD-2, GVT-2, LPF, LP all share the same
            # {error, confidence, reasoning} shape (verified against
            # registry.py's METRIC_PROMPTS in this same conversation --
            # grep '"required":' across prompts/*.py). Fake, fixed,
            # never-flagged answer: this provider is not trying to be
            # right, only to be reached correctly.
            return {"error": False, "confidence": "high",
                    "reasoning": "FakeProvider -- orchestration test only, not a real judgment."}

    with open("sample_transcript_assemblyai.json", "r", encoding="utf-8") as f:
        transcript_json = json.load(f)

    provider = FakeProvider()
    result = run_pipeline(provider, transcript_json, target_speaker_id="A")

    print(f"target_speaker_id:         {result['target_speaker_id']}")
    print(f"sentence_count:            {result['sentence_count']}")
    print(f"gvt1_window_count:         {result['gvt1_window_count']}")
    print(f"window_count:              {result['window_count']}")
    print(f"formulaic_candidate_count: {result['formulaic_candidate_count']}")
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
    for key in WORD_TIMESTAMP_METRICS:
        entries = result["results"][key]
        assert len(entries) == result["window_count"], f"{key}: expected one entry per window"
        print(f"  {key:<20} {len(entries)} entries")
    for key in CANDIDATE_METRICS:
        entries = result["results"][key]
        assert len(entries) == result["formulaic_candidate_count"], (
            f"{key}: expected one entry per (candidate, sentence) pair"
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

    print("\n--- Check 2c: FORMULAIC candidate-scan wiring ---")
    formulaic_entries = result["results"]["FORMULAIC"]
    assert all(not e["skipped"] for e in formulaic_entries), (
        "FORMULAIC is unfiltered -- the candidate scan itself IS the filtering step"
    )
    assert provider.calls_by_metric.get("FORMULAIC", 0) == result["formulaic_candidate_count"], (
        "FORMULAIC should be called exactly once per (candidate, sentence) pair, no more, no fewer"
    )
    assert all(e["input"].startswith('Candidate: "') for e in formulaic_entries), (
        "every FORMULAIC input should be a 'Candidate: ... | Sentence: ...' pair, "
        "not a bare sentence -- if this fails, FORMULAIC regressed back to sentence-only input"
    )
    print(f"  FORMULAIC called={provider.calls_by_metric.get('FORMULAIC', 0)}  "
          f"candidate/sentence pairs={result['formulaic_candidate_count']}")
    # This transcript's speaker A has a real corpus BUNDLES hit ("Ich suche
    # schon seit ein paar Monaten...") -- confirm the candidate scan
    # actually found it, not just that the plumbing runs with zero pairs.
    assert result["formulaic_candidate_count"] > 0, (
        "expected at least one BUNDLES candidate in this transcript (e.g. 'schon') -- "
        "if this is 0, either the transcript changed or the candidate scan is broken"
    )
    for e in formulaic_entries:
        print(f"    {e['input']}")

    print("\n--- Check 2d: FILLED_PAUSE shares UNFILLED_PAUSE's word-timestamp windows ---")
    filled_entries = result["results"]["FILLED_PAUSE"]
    unfilled_entries = result["results"]["UNFILLED_PAUSE"]
    assert all(not e["skipped"] for e in filled_entries), "FILLED_PAUSE is unfiltered -- nothing should be skipped"
    assert provider.calls_by_metric.get("FILLED_PAUSE", 0) == result["window_count"], (
        "FILLED_PAUSE should be called exactly once per word-timestamp window, same count as UNFILLED_PAUSE"
    )
    assert len(filled_entries) == len(unfilled_entries), (
        "FILLED_PAUSE and UNFILLED_PAUSE should be built from the exact same windows"
    )
    assert all(f["input"] == u["input"] for f, u in zip(filled_entries, unfilled_entries)), (
        "FILLED_PAUSE and UNFILLED_PAUSE are supposed to share to_word_timestamp_windows()'s "
        "output verbatim -- if any window differs, they've drifted onto separate builders"
    )
    print(f"  FILLED_PAUSE called={provider.calls_by_metric.get('FILLED_PAUSE', 0)}  "
          f"(matches UNFILLED_PAUSE's {provider.calls_by_metric.get('UNFILLED_PAUSE', 0)} calls "
          f"and identical window content -- confirmed shared, not duplicated, input building)")

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
