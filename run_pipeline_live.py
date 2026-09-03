"""
Run the full pipeline (pipeline.py) against a real AssemblyAI-shaped
transcript file using the REAL OpenAI API -- the live counterpart to
pipeline.py's own __main__ block, which only proves the orchestration
wiring with a FakeProvider (no network, no real judgments). This script is
what actually answers "does the whole thing work end to end against real
German sentences," the same escalation test_openai_live.py ->
test_all_metrics_live.py already did for individual metric calls.

Requires (same as test_all_metrics_live.py):
  - OPENAI_API_KEY set in the environment (Windows cmd.exe: set OPENAI_API_KEY=sk-...)
  - Network access to api.openai.com (this sandbox is blocked at the proxy
    level -- confirmed multiple times while building this -- so this has
    only ever been run from your own machine, never from here)

Usage:
  python run_pipeline_live.py [transcript.json]
  Defaults to sample_transcript_assemblyai_v3.json (2026-09-03: v2 plus two
  additions -- a real corpus FORMULAIC candidate line and a synthetic
  FILLED_PAUSE filler-token line -- built specifically so all 10 metrics
  now wired into pipeline.py have something worth actually scanning; see
  build_sample_transcript_v3.py's own docstring for exactly what's real vs.
  invented in those two additions) -- pass a path to run against v2, the
  original 6-sentence sample, or a different file instead.

2026-09-03: this script now runs and prints ALL 10 registered metrics, not
8 -- GVT-1 (sentence-window), FORMULAIC (candidate-scan), and FILLED_PAUSE
(word-timestamps, redefined off audio) all newly wired in the same day;
see pipeline.py's own module docstring for the per-metric reasoning.

What this actually costs, for sample_transcript_assemblyai_v3.json
specifically: 16 sentences x 6 single-sentence metrics (minus whatever the
pre-filter skips for GDD-1/GDD-2) + 14 GVT-1 sentence-windows + 2 FORMULAIC
candidate/sentence pairs + 17 word-timestamp windows each for
UNFILLED_PAUSE and FILLED_PAUSE = up to ~155 live API calls (exact counts
per the FakeProvider structural smoke test run against this same file --
see this project's changelog), all against gpt-5.6-luna with reasoning
effort "none" on short inputs -- the same cheap-tier model already
confirmed working for all 9 single-call metrics individually (FORMULAIC
and FILLED_PAUSE's new/changed shapes have NOT yet been confirmed against
a live call, only structurally). Should still be a small fraction of a
dollar, but has not been separately priced out -- flagging that as
unverified rather than promising a number.

Writes the full result (every per-sentence/per-window answer, not just a
pass/fail summary) to pipeline_result.json next to this script, so you can
actually read what each metric said about each of your real sentences --
this is the point of the whole exercise, not just "did it run."
"""

import json
import os
import sys

from pipeline import (
    run_pipeline,
    SENTENCE_METRICS,
    WINDOWED_SENTENCE_METRICS,
    CANDIDATE_METRICS,
    WORD_TIMESTAMP_METRICS,
)
from providers.openai_provider import OpenAIProvider

TARGET_SPEAKER = "A"  # the learner, per the transcript's own dialogue
DEFAULT_TRANSCRIPT = "sample_transcript_assemblyai_v3.json"


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY is not set in this environment.")
        print('Windows cmd.exe:  set OPENAI_API_KEY=sk-...')
        return 1

    transcript_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_json = json.load(f)

    provider = OpenAIProvider()
    print(f"Transcript: {transcript_path}")
    print(f"Model: {provider.model}")
    print(f"Running {len(SENTENCE_METRICS)} single-sentence metrics + "
          f"{len(WINDOWED_SENTENCE_METRICS)} sentence-window metric + "
          f"{len(CANDIDATE_METRICS)} candidate-scan metric + "
          f"{len(WORD_TIMESTAMP_METRICS)} word-timestamp metrics against "
          f"speaker {TARGET_SPEAKER!r}'s turns...\n")

    result = run_pipeline(provider, transcript_json, target_speaker_id=TARGET_SPEAKER)

    print(f"sentence_count:            {result['sentence_count']}")
    print(f"gvt1_window_count:         {result['gvt1_window_count']}")
    print(f"window_count:              {result['window_count']}")
    print(f"formulaic_candidate_count: {result['formulaic_candidate_count']}\n")

    for metric_key in SENTENCE_METRICS:
        print(f"--- {metric_key} ---")
        for entry in result["results"][metric_key]:
            if entry["skipped"]:
                print(f"  SKIPPED (pre-filter)  {entry['input']!r}")
                continue
            out = entry["output"]
            flag = out.get("error", out.get("structures"))
            print(f"  {flag!s:<30} {entry['input']!r}")
            if out.get("reasoning"):
                print(f"      reasoning: {out['reasoning']}")
        print()

    for metric_key in WINDOWED_SENTENCE_METRICS:
        print(f"--- {metric_key} (sentence windows) ---")
        for entry in result["results"][metric_key]:
            out = entry["output"]
            flag = out.get("error")
            print(f"  {flag!s:<10} window: {entry['input']!r}")
            if out.get("reasoning"):
                print(f"      reasoning: {out['reasoning']}")
            if flag and out.get("corrected"):
                print(f"      corrected: {out['corrected']!r}")
        print()

    print("--- FORMULAIC (candidate/sentence pairs) ---")
    for entry in result["results"]["FORMULAIC"]:
        out = entry["output"]
        flag = out.get("formulaic")
        print(f"  {flag!s:<10} {entry['input']!r}")
        if out.get("reasoning"):
            print(f"      reasoning: {out['reasoning']}")
    if not result["results"]["FORMULAIC"]:
        print("  (no BUNDLES candidates found in this transcript -- zero calls made, not an error)")
    print()

    print("--- UNFILLED_PAUSE ---")
    for entry in result["results"]["UNFILLED_PAUSE"]:
        words = [w["word"] for w in entry["input"]]
        print(f"  window {words}")
        for b in entry["output"]["boundaries"]:
            marker = "  " if b["status"] == "trustworthy" else " SUSPECT ->"
            print(f"    {marker} {b['between']}  {b['reasoning'] or ''}")
    print()

    print("--- FILLED_PAUSE ---")
    for entry in result["results"]["FILLED_PAUSE"]:
        words = [w["word"] for w in entry["input"]]
        fillers = entry["output"].get("fillers", [])
        marker = f" FLAGGED -> {fillers}" if fillers else ""
        print(f"  window {words}{marker}")
    print()

    print(f"structure_breadth_score: {result['structure_breadth_score']} "
          f"(labels: {result['structure_breadth_labels']})")

    with open("pipeline_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nFull result written to pipeline_result.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
