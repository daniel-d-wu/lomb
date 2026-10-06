"""
Orchestration layer: a transcript's Turn objects -> every fluenceme's
result for the target speaker.

Flow (2026-09-24, find-then-sort):
  filter_to_target_speaker() / to_sentences()   the learner's sentences
  chunk_sentence_indices()                       15-minute chunks
  per chunk, in parallel:
    error_finder.find_errors()     call 1: EVERY grammar/word-choice error
    error_sorter.sort_errors()     call 2: one error-fluenceme tag per error
                                   (skipped when the finder found none)
    batching.classify_batch()      one call per labeler (STRUCTURE_BREADTH)
  direct_computation (DIRECT_FLUENCEMES)  FORMULAIC, FILLED_PAUSE,
                                   UNFILLED_PAUSE, WPM -- no LLM, zero cost
  -> session-level STRUCTURE_BREADTH aggregation

LLM calls per chunk: 2 for errors + 1 per labeler (3 today), independent of
how many error fluencemes exist -- adding one is a new tag in the sorter's
list, not a new call. Before 2026-09-24 it was one narrow detector call per
error fluenceme (11 calls per chunk); that design reported nothing that fell
between its narrow scopes. It was compared against this one on two real
sessions and removed (Dan, 2026-09-24; see docs/lomb_model_comparison_v1.md
and git history for the old prompts).

Count unit: one entry per ERROR in the error-tag results (two gender errors
in one sentence are two entries). The narrow method counted sentences, so
each result records `error_method` and `count_unit` as provenance --
stored runs from before 2026-09-24 mean something different.

History of the direct fluencemes (still accurate):
  2026-09-07  WPM added as a direct computation (session-level word_count /
              speaking time; one entry, empty sentence_indices).
  2026-09-06  UNFILLED_PAUSE and FILLED_PAUSE moved off the LLM (fixed
              >500ms gap threshold / fixed filler-token lookup), one entry
              per actual occurrence.
  2026-09-05  FORMULAIC moved off the LLM (BUNDLES regex match = verdict).

This module does NOT retry or rate-limit calls (docs/lomb_failure_modes_v1.md
M2). The __main__ block proves the wiring with a FakeProvider; linguistic
quality needs a real provider run.
"""

import hashlib
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.batching import build_batch_config, classify_batch
from pipeline.error_finder import FINDER_CONFIG, FINDER_KEY, find_errors
from pipeline.error_sorter import SORTER_CONFIG, SORTER_KEY, sort_errors
from pipeline.registry import DIRECT_FLUENCEMES, ERROR_TAGS, LLM_LABELERS, OTHER_ERROR_KEY
from transcript_processing.assemblyai_adapter import from_assemblyai_transcript
from transcript_processing.speaker_filter import (
    filter_to_target_speaker,
    map_words_to_sentences,
    to_sentences,
)

# All from registry.py -- never edit them here. A new fluenceme appears by
# adding its prompts/<name>.py (or a DIRECT_FLUENCEMES entry).
ERROR_METRICS = list(ERROR_TAGS)
LABELER_METRICS = list(LLM_LABELERS)
DIRECT_METRICS = list(DIRECT_FLUENCEMES)
LABELER_BATCH_PROMPTS = {k: build_batch_config(c) for k, c in LLM_LABELERS.items()}

CHUNK_SECONDS = 15 * 60  # each chunk gets its own finder/sorter/labeler calls
MAX_PARALLEL_CALLS = 8   # LLM calls in flight at once
ERROR_METHOD = "find_sort"  # provenance label stored on every analysis run
COUNT_UNIT = "error"        # one error-tag entry = one error (narrow runs before 2026-09-24 counted sentences)


def _fingerprint(payload) -> str:
    def stable(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=str)
        return str(o)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=stable).encode("utf-8")).hexdigest()[:12]


def _prompt_fingerprint(cfg) -> str:
    return _fingerprint([cfg.system_instruction, cfg.few_shot_examples, cfg.response_schema,
                         cfg.generation_config_overrides])


def fluenceme_versions() -> dict[str, str]:
    """Automatic version per prompt and fluenceme, stored on each analysis
    run so a metric change can be traced to a method change. Nobody has to
    remember to bump a version number.

    find_sort: ERROR_FINDER and ERROR_SORTER fingerprint the exact prompts
    sent (the sorter's includes every tag definition, so it changes when any
    tag does); each error tag also gets its own fingerprint (definition +
    examples) so you can see WHICH tag changed. Labelers: the batch prompt.
    Direct: compute() source + reference data."""
    versions = {FINDER_KEY: _prompt_fingerprint(FINDER_CONFIG), SORTER_KEY: _prompt_fingerprint(SORTER_CONFIG)}
    for key, tag in ERROR_TAGS.items():
        versions[key] = _fingerprint([tag.definition, tag.not_this, tag.examples])
    for key, cfg in LABELER_BATCH_PROMPTS.items():
        versions[key] = _prompt_fingerprint(cfg)
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
    1 chunk, 30 min -> 2, 47 min -> 4. Empty chunks (no target-speaker
    speech in that window) are dropped."""
    chunks: dict[int, list[int]] = {}
    for i, t in enumerate(start_times):
        chunks.setdefault(int(t // chunk_seconds), []).append(i)
    return [chunks[k] for k in sorted(chunks)]


def find_and_sort(provider, indexed_sentences: list[tuple[int, str]]) -> tuple[list[dict], dict[int, str], int]:
    """Both error calls for one chunk, in order. Returns (errors, tags by
    position in errors, calls made). No errors found -> no sorter call."""
    errors = find_errors(provider, indexed_sentences)
    if not errors:
        return errors, {}, 1
    text = dict(indexed_sentences)
    items = [{"id": n, "sentence": text[e["index"]], "said": e["said"], "corrected": e["corrected"]}
             for n, e in enumerate(errors)]
    return errors, sort_errors(provider, items), 2


def _error_entries(chunk_outputs: list, sentences: list[str]) -> tuple[dict, list[dict], list[int]]:
    """Turn every chunk's found+tagged errors into (results per error tag,
    found_errors, untagged error ids). Each error becomes ONE entry under its
    tag, in the {"input","skipped","output"} shape storage and Report 1
    read. An error the sorter didn't tag goes under OTHER_ERROR marked
    untagged -- stored for review, counted nowhere (flag_quality)."""
    results = {k: [] for k in ERROR_METRICS}
    found, untagged = [], []
    for errors, tags, _ in chunk_outputs:
        for position, e in enumerate(errors):
            error_id = len(found)
            tag = tags.get(position)
            found.append({"error_id": error_id, "tag": tag, **e})
            output = {
                "error": True,
                "confidence": e["confidence"],
                "reasoning": e["reasoning"],
                "corrected": e["corrected"],   # built in code: said -> fix
                "said": e["said"],
                "fix": e["fix"],
                "clean": e["clean"],           # fully corrected, filler-free sentence (shown on cards)
                "understandable": e["understandable"],
                "clean_ok": e["clean_ok"],     # clean exists and contains this fix
                "said_not_found": e["said_not_found"],
                "self_corrected": e["self_corrected"],
                "possible_transcription_error": e["possible_transcription_error"],
                "error_id": error_id,
                "sentence_indices": [e["index"]],
            }
            if tag is None:
                output["untagged"] = True
                untagged.append(error_id)
            results[tag or OTHER_ERROR_KEY].append(
                {"input": sentences[e["index"]], "skipped": False, "output": output})
    for entries in results.values():
        entries.sort(key=lambda en: (en["output"]["sentence_indices"][0], en["output"]["error_id"]))
    return results, found, untagged


def run_pipeline_from_turns(provider, turns: list, target_speaker_id: str, *,
                            chunk_seconds: float = CHUNK_SECONDS,
                            max_parallel_calls: int = MAX_PARALLEL_CALLS) -> dict:
    """The orchestration core -- engine-agnostic (2026-09-20): takes turns
    already adapted by any transcript_processing/*_adapter.py.

    provider: any llm_provider.LLMProvider (OpenAIProvider in production;
      a FakeProvider in tests) -- passed in, never hardcoded.
    target_speaker_id: the diarized speaker to analyse, already confirmed
      (voiceprint match or manual pick) by the caller.

    Returns:
      {
        "target_speaker_id", "sentence_count",
        "chunk_count", "chunk_seconds",
        "error_method": "find_sort", "count_unit": "error",   # provenance labels
        "fluenceme_versions": {...},
        "llm_call_count": int,     # find_sort: <= (2 + labelers) x chunks
        "llm_missing": dict,       # labeler -> omitted sentence indices;
                                   # ERROR_SORTER -> error ids left untagged
        "found_error_count": int,  # every error the finder reported
        "found_errors": [ {error_id, tag, index, said, corrected, reasoning,
                           confidence, self_corrected,
                           possible_transcription_error}, ... ],
        "formulaic_candidate_count", "filled_pause_count",
        "unfilled_pause_count", "word_count", "duration_seconds",
        "wpm": float | None,       # convenience value, None if no speaking time
        "results": {
          <error tag>:  one entry per error (find_sort) -- output carries
                        error=True, corrected, said, reasoning, confidence,
                        self_corrected, possible_transcription_error,
                        error_id, sentence_indices=[i]
          <labeler>:    one entry per sentence
          <direct>:     one entry per occurrence (WPM: one session entry)
          each entry: {"input": str, "skipped": bool, "output": dict | None}
        },
        "structure_breadth_score": int,   # DISTINCT non-"none" labels this session
        "structure_breadth_labels": list[str],
      }
    """
    target_turns = filter_to_target_speaker(turns, target_speaker_id)
    sentences = to_sentences(target_turns)
    chunks = chunk_sentence_indices(sentence_start_times(target_turns), chunk_seconds)
    indexed_chunks = [[(i, sentences[i]) for i in chunk] for chunk in chunks]

    results: dict[str, list] = {}
    llm_missing: dict[str, list] = {}
    found_errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, max_parallel_calls)) as pool:
        labeler_jobs = [(key, chunk, pool.submit(classify_batch, provider, LABELER_BATCH_PROMPTS[key], chunk))
                        for key in LABELER_METRICS for chunk in indexed_chunks]
        error_jobs = [pool.submit(find_and_sort, provider, chunk) for chunk in indexed_chunks]
        chunk_outputs = [f.result() for f in error_jobs]  # re-raises the first failed call
        error_results, found_errors, untagged = _error_entries(chunk_outputs, sentences)
        if untagged:
            llm_missing[SORTER_KEY] = untagged
        error_calls = sum(calls for _, _, calls in chunk_outputs)
        labeler_outputs = [(key, chunk, f.result()) for key, chunk, f in labeler_jobs]
    results.update(error_results)

    # Labelers: one entry per sentence; a sentence the model left out gets
    # output None and is listed in llm_missing, never treated as "none".
    for key in LABELER_METRICS:
        outputs, sent = {}, set()
        for k, chunk, out in labeler_outputs:
            if k == key:
                outputs.update(out)
                sent.update(i for i, _ in chunk)
        missing = sorted(sent - set(outputs))
        if missing:
            llm_missing[key] = missing
        results[key] = [{"input": s, "skipped": i not in sent, "output": outputs.get(i)}
                        for i, s in enumerate(sentences)]

    # Direct fluencemes: no LLM call. Each compute() returns entries in the
    # same {"input","skipped","output"} shape.
    for key, spec in DIRECT_FLUENCEMES.items():
        results[key] = spec.compute(target_turns)

    # Session-level STRUCTURE_BREADTH: the UNION of distinct labels, "none"
    # excluded -- the same label in 5 sentences counts once.
    breadth_labels = set()
    for entry in results.get("STRUCTURE_BREADTH", []):
        if entry["skipped"] or entry["output"] is None:
            continue
        breadth_labels.update(label for label in entry["output"].get("structures", []) if label != "none")

    wpm_info = results["WPM"][0]["output"]
    word_count = wpm_info["word_count"]
    duration_seconds = wpm_info["duration_seconds"]
    wpm_value = round(word_count / (duration_seconds / 60), 1) if duration_seconds > 0 else None

    return {
        "target_speaker_id": target_speaker_id,
        "sentence_count": len(sentences),
        "chunk_count": len(chunks),
        "chunk_seconds": chunk_seconds,
        "error_method": ERROR_METHOD,
        "count_unit": COUNT_UNIT,
        "fluenceme_versions": fluenceme_versions(),
        "llm_call_count": error_calls + len(labeler_jobs),
        "llm_missing": llm_missing,
        "found_error_count": len(found_errors),
        "found_errors": found_errors,
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


def run_pipeline(provider, assemblyai_json: dict, target_speaker_id: str, **kwargs) -> dict:
    """Back-compat entry point for raw AssemblyAI JSON: adapts it to Turn
    objects, then delegates to run_pipeline_from_turns()."""
    return run_pipeline_from_turns(provider, from_assemblyai_transcript(assemblyai_json), target_speaker_id, **kwargs)


if __name__ == "__main__":
    from pipeline.fake_provider import FakeProvider
    from pipeline.flag_quality import rejection_reason

    # data/sample_transcript_assemblyai_v3.json: a ~55s two-speaker sample.
    SAMPLE_TRANSCRIPT = Path(__file__).resolve().parent.parent / "data" / "sample_transcript_assemblyai_v3.json"
    with open(SAMPLE_TRANSCRIPT, "r", encoding="utf-8") as f:
        transcript_json = json.load(f)
    turns = from_assemblyai_transcript(transcript_json)

    provider = FakeProvider()
    result = run_pipeline_from_turns(provider, turns, "A")
    n_sent, n_chunks = result["sentence_count"], result["chunk_count"]
    print(f"sentences={n_sent}  chunks={n_chunks}  llm_calls={result['llm_call_count']}  "
          f"found_errors={result['found_error_count']}  calls by prompt={provider.calls_by_key}")

    print("\n--- Check 1: every fluenceme key present, right entry shape ---")
    for key in LABELER_METRICS:
        assert len(result["results"][key]) == n_sent, f"{key}: expected one entry per sentence"
    for key in ERROR_METRICS:
        assert key in result["results"], f"error tag {key} missing from results (must be present even if empty)"
    direct_expected = {"FORMULAIC": result["formulaic_candidate_count"], "FILLED_PAUSE": result["filled_pause_count"],
                       "UNFILLED_PAUSE": result["unfilled_pause_count"], "WPM": 1}
    for key in DIRECT_METRICS:
        assert len(result["results"][key]) == direct_expected[key], f"{key}: one entry per occurrence expected"
    print(f"  {len(ERROR_METRICS)} error tags, {len(LABELER_METRICS)} labeler(s), {len(DIRECT_METRICS)} direct -- all present")

    print("\n--- Check 1b: call budget -- finder once per chunk, sorter only when errors, labelers once per chunk ---")
    assert provider.calls_by_key.get(FINDER_KEY) == n_chunks
    assert provider.calls_by_key.get(SORTER_KEY) == (n_chunks if result["found_error_count"] else 0)
    for key in LABELER_METRICS:
        assert provider.calls_by_key.get(key) == n_chunks, f"{key}: one call per chunk"
    for key in ERROR_METRICS:
        assert key not in provider.calls_by_key, f"{key} must not have its own LLM call any more"
    assert provider.call_count == result["llm_call_count"] <= (2 + len(LABELER_METRICS)) * n_chunks
    print(f"  total calls={provider.call_count} (the old narrow design needed {len(ERROR_METRICS) - 1 + len(LABELER_METRICS)} per chunk)")

    print("\n--- Check 2: every found error lands exactly once, under its tag, on the right sentence ---")
    sentences = to_sentences(filter_to_target_speaker(turns, "A"))
    placed = [(tag, en) for tag in ERROR_METRICS for en in result["results"][tag]]
    assert len(placed) == result["found_error_count"] > 0
    assert sorted(en["output"]["error_id"] for _, en in placed) == list(range(result["found_error_count"]))
    for tag, en in placed:
        fe = result["found_errors"][en["output"]["error_id"]]
        assert fe["tag"] == tag, f"error {fe['error_id']} tagged {fe['tag']} but stored under {tag}"
        assert en["output"]["sentence_indices"] == [fe["index"]] and en["input"] == sentences[fe["index"]]
    assert result["llm_missing"] == {}, result["llm_missing"]
    print(f"  {len(placed)} errors placed; tags used: {sorted({t for t, _ in placed})}")

    print("\n--- Check 2a: an error the sorter doesn't tag is kept, flagged, and counted nowhere ---")
    p2 = FakeProvider(leave_untagged=True)
    r2 = run_pipeline_from_turns(p2, turns, "A")
    untagged = r2["llm_missing"][SORTER_KEY]
    assert untagged, "leave_untagged should leave at least one error without a tag"
    other = {en["output"]["error_id"]: en for en in r2["results"][OTHER_ERROR_KEY]}
    for eid in untagged:
        assert other[eid]["output"].get("untagged") is True
        assert rejection_reason(other[eid]["output"], other[eid]["input"]) == "sorter gave no tag"
    print(f"  untagged error ids {untagged} -> OTHER_ERROR, rejected as 'sorter gave no tag'")

    print("\n--- Check 2b: a possible transcription error is kept but never trusted ---")
    asr = [en for tag in ERROR_METRICS for en in result["results"][tag] if en["output"]["possible_transcription_error"]]
    assert asr, "FakeProvider marks every 9th sentence's error as a possible transcription error"
    assert all(rejection_reason(en["output"], en["input"]) == "possible transcription error" for en in asr)
    print(f"  {len(asr)} possible transcription error(s) rejected")

    print("\n--- Check 4: corrections are built in code; clean sentences checked against the fix ---")
    from pipeline.flag_quality import apply_fix
    for tag, en in placed:
        out = en["output"]
        assert out["corrected"] == apply_fix(en["input"], out["said"], out["fix"]), "corrected must be said -> fix"
        assert out["clean_ok"] == (out["understandable"] and "FAKEFIX" in out["clean"]), out
    unclear = [en for _, en in placed if not en["output"]["understandable"]]
    assert unclear and all(not en["output"]["clean_ok"] for en in unclear)
    p5 = FakeProvider(bad_said=True)
    r5 = run_pipeline_from_turns(p5, turns, "A")
    bad = [en for tag in ERROR_METRICS for en in r5["results"][tag]]
    assert bad and all(en["output"]["said_not_found"] and en["output"]["corrected"] == "" for en in bad)
    assert all(rejection_reason(en["output"], en["input"]) in ("flagged words not in sentence",
                                                             "possible transcription error") for en in bad)
    print(f"  {len(placed)} corrections rebuilt in code; {len(unclear)} not-understandable sentence(s) get no "
          f"clean sentence; {len(bad)} errors pointing at words not in the sentence rejected")

    print("\n--- Check 1c: 20-second chunks give the same errors with more calls ---")
    p3 = FakeProvider()
    r3 = run_pipeline_from_turns(p3, turns, "A", chunk_seconds=20)
    assert r3["chunk_count"] > 1
    assert p3.calls_by_key[FINDER_KEY] == r3["chunk_count"]
    key_of = lambda fe: (fe["index"], fe["said"], fe["corrected"])  # noqa: E731
    assert sorted(map(key_of, r3["found_errors"])) == sorted(map(key_of, result["found_errors"]))
    print(f"  chunks={r3['chunk_count']} calls={p3.call_count} same {r3['found_error_count']} errors")

    print("\n--- Check 2c-2e: direct fluencemes never call the LLM ---")
    for key in DIRECT_METRICS:
        assert key not in provider.calls_by_key, f"{key} must never call the provider"
        assert all(not e["skipped"] for e in result["results"][key])
    for e in result["results"]["FILLED_PAUSE"] + result["results"]["UNFILLED_PAUSE"]:
        assert e["output"]["sentence_indices"] and e["output"]["boundary_type"] in ("sentence", "clause", "none")
    assert result["formulaic_candidate_count"] > 0, "expected a BUNDLES match in this sample (e.g. 'schon')"
    assert result["unfilled_pause_count"] > 0, "expected a real >500ms gap in this sample"
    wpm_out = result["results"]["WPM"][0]["output"]
    assert wpm_out["sentence_indices"] == [] and result["word_count"] == wpm_out["word_count"]
    if result["duration_seconds"] > 0:
        assert result["wpm"] == round(result["word_count"] / (result["duration_seconds"] / 60), 1)
    print(f"  formulaic={result['formulaic_candidate_count']} filled={result['filled_pause_count']} "
          f"unfilled={result['unfilled_pause_count']} wpm={result['wpm']}")

    print("\n--- Check 3: structure_breadth_score is a distinct-label union ---")
    assert result["structure_breadth_score"] < n_sent
    assert result["structure_breadth_labels"] == sorted({"konjunktiv_ii", "dass_clause", "weil_clause"})
    print(f"  {n_sent} sentences -> breadth {result['structure_breadth_score']} {result['structure_breadth_labels']}")

    print("\nAll orchestration checks passed. This proves the WIRING, not linguistic "
          "correctness -- that needs a real run with OpenAIProvider.")
