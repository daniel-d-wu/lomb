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
  speaker_filter.to_sentences() /
    find_formulaic_matches()                       -> per-metric input shapes
  prefilter.should_run()                           drop sentences from a batch where safe
  batching.classify_batch()                        ONE call per LLM metric per session
                                                    (7 metrics; FORMULAIC,
                                                    UNFILLED_PAUSE,
                                                    FILLED_PAUSE, WPM never
                                                    reach this)
  direct_computation.compute_filled_pause() /
    compute_unfilled_pause() / compute_wpm()       the other 4 direct metrics
  -> a session-level breadth_score aggregation for STRUCTURE_BREADTH

SCOPE, stated explicitly: as of 2026-09-07, this runs all 11 registered
metrics -- the full Phase 1 set. Four groups, by how they're computed:

  LLM-assisted, ONE call per metric covering every sentence (2026-09-24,
  see batching.py -- previously one call per sentence, 327 calls on a
  5-min session):
    GDD-1, GDD-2, GVT-1, GVT-2, LPF, LP, STRUCTURE_BREADTH
  (GVT-1 used to run on 3-sentence windows to see cross-sentence tense
  drift; the batch call gives it the whole ordered list, so windows are
  no longer used.)
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

This module does NOT retry or rate-limit calls. It does batch them: at
most one call per LLM metric per session (batching.py). It has NOT been run against a live API from inside this sandbox
(same api.openai.com egress block documented throughout this project,
in providers/openai_provider.py and elsewhere) -- see the __main__ block
and its FakeProvider for exactly what that block does and doesn't prove.
"""

import hashlib
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# repo root on sys.path -- prompts/ and transcript_processing/ are both
# siblings of this file's own new directory (pipeline/) post-reorg.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import prefilter
from pipeline.batching import BATCH_PROMPTS, classify_batch
from pipeline.registry import DIRECT_FLUENCEMES, METRIC_PROMPTS
from transcript_processing.assemblyai_adapter import from_assemblyai_transcript
from transcript_processing.speaker_filter import (
    filter_to_target_speaker,
    map_words_to_sentences,
    to_sentences,
)

# Both lists come from registry.py -- never edit them here. A new LLM
# fluenceme appears in SENTENCE_METRICS by adding its prompts/<name>.py.
SENTENCE_METRICS = list(METRIC_PROMPTS)
DIRECT_METRICS = list(DIRECT_FLUENCEMES)
assert all(METRIC_PROMPTS[k].input_kind == "sentence" for k in SENTENCE_METRICS), (
    "run_pipeline_from_turns() only knows how to feed sentence-input LLM fluencemes"
)

CHUNK_SECONDS = 15 * 60  # one LLM call per metric per 15 minutes of session audio
MAX_PARALLEL_CALLS = 8   # LLM calls in flight at once


def _fingerprint(payload) -> str:
    def stable(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=str)
        return str(o)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=stable).encode("utf-8")).hexdigest()[:12]


def fluenceme_versions() -> dict[str, str]:
    """Automatic version per fluenceme, stored on each analysis run so a
    metric change can be traced to a method change. LLM: fingerprint of the
    exact prompt sent (batch instructions, few-shot, schema). Direct:
    fingerprint of its compute() source + reference data. Nobody has to
    remember to bump a version number."""
    versions = {}
    for key, cfg in BATCH_PROMPTS.items():
        versions[key] = _fingerprint([cfg.system_instruction, cfg.few_shot_examples,
                                      cfg.response_schema, cfg.generation_config_overrides])
    for key, spec in DIRECT_FLUENCEMES.items():
        versions[key] = _fingerprint([inspect.getsource(spec.compute), spec.reference])
    return versions


def sentence_start_times(target_turns: list) -> list[float]:
    """Start time (session seconds) of each to_sentences() entry, from its
    first word. A sentence with no mapped words inherits the previous
    sentence's start (never observed; guards the alignment caveat in
    map_words_to_sentences())."""
    n = len(to_sentences(target_turns))
    starts: list[float | None] = [None] * n
    for w in map_words_to_sentences(target_turns):
        i = w["sentence_index"]
        if i < n and starts[i] is None:
            starts[i] = w["start"]
    last = 0.0
    for i, s in enumerate(starts):
        last = s if s is not None else last
        starts[i] = last
    return starts


def chunk_sentence_indices(start_times: list[float], chunk_seconds: float = CHUNK_SECONDS) -> list[list[int]]:
    """Group sentence indices into consecutive time chunks of chunk_seconds
    (by each sentence's start time). Any session length works: <15 min ->
    1 chunk, 30 min -> 2, 45 min -> 3, 47 min -> 4. Empty chunks (no
    target-speaker speech in that window) are dropped."""
    chunks: dict[int, list[int]] = {}
    for i, t in enumerate(start_times):
        chunks.setdefault(int(t // chunk_seconds), []).append(i)
    return [chunks[k] for k in sorted(chunks)]


def run_pipeline_from_turns(provider, turns: list, target_speaker_id: str, *,
                            chunk_seconds: float = CHUNK_SECONDS,
                            max_parallel_calls: int = MAX_PARALLEL_CALLS) -> dict:
    """The actual orchestration core -- engine-agnostic as of 2026-09-20.

    2026-09-20: split out of what used to be the only run_pipeline()
    function, so a transcription engine other than AssemblyAI (first
    concretely: whisperX+pyannote, see transcript_processing/
    whisperx_adapter.py) can reach every metric below without this module
    knowing or caring which engine produced its input. Before this split,
    run_pipeline() hardcoded assemblyai_adapter.from_assemblyai_transcript()
    as its very first line -- there was no seam for a second engine to
    plug into short of a second, near-duplicate copy of this entire
    function (and everything below it silently drifting out of sync with
    this one as it evolves). Everything from filter_to_target_speaker()
    onward never actually depended on AssemblyAI-shaped input in the
    first place; it only ever touched Turn/Word objects, which is exactly
    speaker_filter.py's own point (see that module's docstring). This
    function is the proof: it takes turns directly, already adapted by
    whichever *_adapter.py the caller used, and everything downstream is
    unchanged.

    provider: any llm_provider.LLMProvider instance -- OpenAIProvider() in
      production per providers/openai_provider.py's 2026-09-03 switch to
      primary, but deliberately accepted as a parameter rather than
      imported and hardcoded here, so this stays swappable (this module
      never needs to know or care which provider answered) and testable
      (the __main__ block below passes a FakeProvider instead).
    turns: list[transcript_processing.speaker_filter.Turn], already
      produced by an adapter (assemblyai_adapter.from_assemblyai_transcript(),
      whisperx_adapter.from_whisperx_transcript(), or any future engine's
      own adapter) -- this function has no idea which, and must never be
      given a reason to care.
    target_speaker_id: which diarized speaker to run metrics against.
      lomb_backend_prd_v1.md Section 6.2 requires the visitor to confirm
      this before any metrics run -- this function does NOT do that
      confirmation step itself, only the resulting filtering; pass
      whatever speaker_id the confirmation step already validated.

    Returns:
      {
        "target_speaker_id": str,
        "sentence_count": int,
        "chunk_count": int,                # 15-min slices of session audio with target speech
        "llm_call_count": int,             # at most (LLM fluencemes x chunk_count)
        "llm_missing": dict,               # metric -> sentence indices the model's batch answer omitted
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
    target_turns = filter_to_target_speaker(turns, target_speaker_id)
    sentences = to_sentences(target_turns)
    chunks = chunk_sentence_indices(sentence_start_times(target_turns), chunk_seconds)

    results = {}

    # One LLM call per (metric, 15-min chunk), all run in parallel. Each call
    # gets that chunk's sentences the prefilter doesn't rule out. Entries stay
    # one-per-sentence (the shape storage/report already read); a sentence
    # the model left out of its answer gets output None and is listed in
    # llm_missing, never silently treated as "no error".
    jobs = []
    for metric_key in SENTENCE_METRICS:
        for chunk in chunks:
            to_run = [(i, sentences[i]) for i in chunk if prefilter.should_run(metric_key, sentences[i])]
            if to_run:
                jobs.append((metric_key, to_run))
    with ThreadPoolExecutor(max_workers=max(1, max_parallel_calls)) as pool:
        futures = [pool.submit(classify_batch, provider, key, to_run) for key, to_run in jobs]
        job_outputs = [f.result() for f in futures]  # re-raises the first failed call

    outputs_by_metric: dict[str, dict[int, dict]] = {k: {} for k in SENTENCE_METRICS}
    sent_by_metric: dict[str, set[int]] = {k: set() for k in SENTENCE_METRICS}
    for (metric_key, to_run), outputs in zip(jobs, job_outputs):
        outputs_by_metric[metric_key].update(outputs)
        sent_by_metric[metric_key].update(i for i, _ in to_run)

    llm_missing = {}
    for metric_key in SENTENCE_METRICS:
        outputs, sent = outputs_by_metric[metric_key], sent_by_metric[metric_key]
        missing = sorted(sent - set(outputs))
        if missing:
            llm_missing[metric_key] = missing
        results[metric_key] = [
            {"input": s, "skipped": i not in sent, "output": outputs.get(i)}
            for i, s in enumerate(sentences)
        ]

    # Direct fluencemes: no LLM call, zero cost. Each one's compute() returns
    # entries in the same {"input","skipped","output"} shape as the LLM
    # ones, so storage/report never need a special case to read them.
    for key, spec in DIRECT_FLUENCEMES.items():
        results[key] = spec.compute(target_turns)

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
    wpm_info = results["WPM"][0]["output"]
    word_count = wpm_info["word_count"]
    duration_seconds = wpm_info["duration_seconds"]
    wpm_value = round(word_count / (duration_seconds / 60), 1) if duration_seconds > 0 else None

    return {
        "target_speaker_id": target_speaker_id,
        "sentence_count": len(sentences),
        "chunk_count": len(chunks),
        "chunk_seconds": chunk_seconds,
        "fluenceme_versions": fluenceme_versions(),
        "llm_call_count": len(jobs),
        "llm_missing": llm_missing,
        "formulaic_candidate_count": len(results["FORMULAIC"]),
        "filled_pause_count": len(results["FILLED_PAUSE"]),
        "unfilled_pause_count": len(results["UNFILLED_PAUSE"]),
        "word_count": word_count,
        "duration_seconds": duration_seconds,
        "wpm": wpm_value,
        "results": results,
        "structure_breadth_score": len(breadth_labels),
        "structure_breadth_labels": sorted(breadth_labels),
    }


def run_pipeline(provider, assemblyai_json: dict, target_speaker_id: str) -> dict:
    """Back-compat entry point -- the original signature, unchanged
    behavior. Adapts raw AssemblyAI JSON to Turn objects via
    assemblyai_adapter.from_assemblyai_transcript(), then delegates to
    run_pipeline_from_turns() for everything else (see that function's
    2026-09-20 docstring note for why this split happened -- a second
    engine, whisperX+pyannote, needed a way to reach the same metrics
    logic without going through AssemblyAI's response shape first).
    Existing callers (scripts/run_pipeline_live.py, this module's own
    __main__ self-test below) need zero changes -- this function's
    signature and return value are identical to before the split."""
    turns = from_assemblyai_transcript(assemblyai_json)
    return run_pipeline_from_turns(provider, turns, target_speaker_id)


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
            import threading
            self._lock = threading.Lock()  # calls now arrive in parallel
            self.call_count = 0
            self.calls_by_metric = {}
            self._structure_cycle = [
                ["konjunktiv_ii"], ["dass_clause"], ["konjunktiv_ii"],
                ["none"], ["weil_clause"], ["dass_clause"],
            ]

        def classify(self, config, input_data):
            with self._lock:
                self.call_count += 1
                self.calls_by_metric[config.key] = self.calls_by_metric.get(config.key, 0) + 1
            items = json.loads(input_data)  # batch input: [{"index", "text"}, ...]
            return {"results": [{"index": it["index"], **self._answer(config.key, it["index"])} for it in items]}

        def _answer(self, key, index):
            if key == "STRUCTURE_BREADTH":
                labels = self._structure_cycle[index % len(self._structure_cycle)]
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
    print(f"llm_call_count:            {result['llm_call_count']}")
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

    print("\n--- Check 1b: at most ONE provider call per LLM metric ---")
    for key in SENTENCE_METRICS:
        calls = provider.calls_by_metric.get(key, 0)
        ran_any = any(not e["skipped"] for e in result["results"][key])
        assert calls == (1 if ran_any else 0), f"{key}: expected {1 if ran_any else 0} call, got {calls}"
    assert result["chunk_count"] == 1, "this sample is under 15 min -- expected a single chunk"
    assert provider.call_count == result["llm_call_count"] <= len(SENTENCE_METRICS)
    assert result["llm_missing"] == {}, f"FakeProvider answers every index; missing={result['llm_missing']}"
    print(f"  total calls={provider.call_count} (was one per sentence before 2026-09-24)")

    print("\n--- Check 1c: chunking -- force 20-second chunks on the same transcript ---")
    chunked_provider = FakeProvider()
    turns_for_chunks = from_assemblyai_transcript(transcript_json)
    chunked = run_pipeline_from_turns(chunked_provider, turns_for_chunks, "A", chunk_seconds=20)
    assert chunked["chunk_count"] > 1, "20s chunks on a ~55s sample should give several chunks"
    for key in SENTENCE_METRICS:
        # identical per-sentence verdict shape and skip decisions as the unchunked run
        assert [e["skipped"] for e in chunked["results"][key]] == [e["skipped"] for e in result["results"][key]]
        assert chunked_provider.calls_by_metric.get(key, 0) <= chunked["chunk_count"]
    assert chunked_provider.call_count == chunked["llm_call_count"]
    print(f"  chunks={chunked['chunk_count']}  calls={chunked_provider.call_count} "
          f"(<= {len(SENTENCE_METRICS)} metrics x {chunked['chunk_count']} chunks), same per-sentence results")

    print("\n--- Check 2: prefilter skip decisions actually took effect ---")
    for key in ("GDD-1", "GDD-2"):
        entries = result["results"][key]
        actually_sent = sum(1 for e in entries if not e["skipped"])
        expected_sent = sum(1 for e in entries if prefilter.should_run(key, e["input"]))
        assert actually_sent == expected_sent, (
            f"{key}: {actually_sent} sentences sent in the batch, but prefilter independently "
            f"says {expected_sent} should have been -- these must agree"
        )
        skipped = [e["input"] for e in entries if e["skipped"]]
        print(f"  {key:<8} sent={actually_sent}  skipped={len(skipped)}  "
              f"skipped sentences={skipped}")
    for key in ("GVT-1", "GVT-2", "LPF", "LP", "STRUCTURE_BREADTH"):
        entries = result["results"][key]
        assert all(not e["skipped"] for e in entries), f"{key} is unfiltered -- nothing should be skipped"
    print("  (GVT-1, GVT-2, LPF, LP, STRUCTURE_BREADTH correctly never skipped -- unfiltered by design)")

    print("\n--- Check 2b: batch answers map back to the right sentence ---")
    for key in SENTENCE_METRICS:
        for i, e in enumerate(result["results"][key]):
            if not e["skipped"]:
                assert e["output"]["sentence_indices"] == [i], f"{key}: entry {i} got {e['output']['sentence_indices']}"
    print("  every non-skipped entry's output.sentence_indices == [its own position]")

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
    judged = sum(1 for e in result["results"]["STRUCTURE_BREADTH"] if not e["skipped"])
    assert judged == result["sentence_count"]
    assert result["structure_breadth_score"] < judged, (
        "FakeProvider's cycle deliberately repeats labels across sentences -- the aggregated "
        "score must come out lower than the sentence count, or aggregation isn't deduplicating"
    )
    assert result["structure_breadth_labels"] == sorted(
        {"konjunktiv_ii", "dass_clause", "weil_clause"}
    ), result["structure_breadth_labels"]
    print(f"  {judged} STRUCTURE_BREADTH sentence verdicts -> "
          f"breadth_score={result['structure_breadth_score']} distinct labels "
          f"(not {judged}) -- deduplication confirmed, 'none' correctly excluded")

    print("\nAll orchestration checks passed. "
          "Reminder: this proves the WIRING, not linguistic correctness -- "
          "that needs a real run with OpenAIProvider() and a live key.")
