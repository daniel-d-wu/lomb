"""
Trace one real Lomb pipeline run in LangSmith.

Runs the SAME code production runs (pipeline.run_pipeline_from_turns with
your OpenAIProvider) on a transcript you already have, and logs every LLM
call to LangSmith: one trace per run, with a child for each call --
ERROR_FINDER, CLEAN_RECHECK, ERROR_SORTER, STRUCTURE_BREADTH -- showing its
input, output, tokens, cost and time.

Put this file in lomb_prompts/ (the repo root) and run from there:

  python lomb_trace.py --transcript PATH --speaker A --minutes 15 --dry-run
  python lomb_trace.py --transcript PATH --speaker A --minutes 15
  python lomb_trace.py --transcript PATH --speaker A --minutes 15 --fake   (offline wiring check, no API calls)

Environment (same as the earlier ping test):
  OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT=lomb
  LOMB_LLM_MODEL, LOMB_REASONING_EFFORT   optional -- same variables and
      defaults as service/cloud.py, so set them to whatever your Cloud Run
      worker uses if you want this run to match production.

What this does NOT change: the pipeline, prompts, provider and storage are
untouched. The only differences from a production run are (1) the OpenAI
client is wrapped so calls get logged, (2) the pipeline's thread pool is
swapped for LangSmith's context-carrying one, so calls made in parallel
still nest under this run's trace, and (3) nothing is written to the
database -- the result is saved as JSON in test_runs/ instead.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline.pipeline as lomb_pipeline  # noqa: E402
from pipeline.flag_quality import rejection_reason  # noqa: E402
from pipeline.registry import ERROR_TAGS, LLM_LABELERS  # noqa: E402
from providers.openai_provider import OpenAIProvider  # noqa: E402
from transcript_processing.assemblyai_adapter import from_assemblyai_transcript  # noqa: E402
from transcript_processing.speaker_filter import filter_to_target_speaker, to_sentences  # noqa: E402


def _traceable(**kwargs):
    from langsmith import traceable
    return traceable(**kwargs)


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class TracedOpenAIProvider(OpenAIProvider):
    """Your production provider with one change: the OpenAI client is
    wrapped so every call is logged. Each classify() becomes a child run
    named after the prompt (ERROR_FINDER, ERROR_SORTER, ...)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None

    def call(self, request: dict) -> dict:
        if self._client is None:
            from langsmith import wrappers
            from openai import OpenAI
            self._client = wrappers.wrap_openai(OpenAI(api_key=os.environ[self.api_key_env]))
        return self._client.responses.create(**request).model_dump()

    def classify(self, config, input_data):
        parent = super()
        traced = _traceable(
            name=config.key,
            run_type="chain",
            metadata={"prompt_key": config.key, "model": self.model,
                      "reasoning": self.generation_config.get("reasoning")},
            # Log the learner text this call got, not the whole config object.
            process_inputs=lambda inputs: {"input": inputs.get("input_data")},
        )(lambda config, input_data: parent.classify(config, input_data))
        return traced(config, input_data)


def fake_provider():
    from pipeline.fake_provider import FakeProvider

    class TracedFake(FakeProvider):
        model = "fake"

        def classify(self, config, input_data):
            traced = _traceable(name=config.key, run_type="chain",
                                process_inputs=lambda inputs: {"input": inputs.get("input_data")})(
                lambda config, input_data: FakeProvider.classify(self, config, input_data))
            return traced(config, input_data)

    return TracedFake()


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------

def load_turns(path: str, start_min: float, minutes: float | None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    turns = from_assemblyai_transcript(data)   # raises a clear error if not a diarized AssemblyAI file
    total_min = max((t.end for t in turns), default=0) / 60
    if minutes is not None or start_min:
        lo = start_min * 60
        hi = (start_min + minutes) * 60 if minutes is not None else float("inf")
        turns = [t for t in turns if lo <= t.start < hi]
    return turns, total_min


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def summarise(result: dict, seconds: float) -> dict:
    counted, not_counted = Counter(), Counter()
    for tag in ERROR_TAGS:
        for entry in result["results"].get(tag, []):
            reason = rejection_reason(entry["output"], entry["input"])
            if reason is None:
                counted[tag] += 1
            else:
                not_counted[reason] += 1
    return {
        "seconds": round(seconds, 1),
        "llm_calls": result["llm_call_count"],
        "chunks": result["chunk_count"],
        "sentences": result["sentence_count"],
        "wpm": result["wpm"],
        "errors_found": result["found_error_count"],
        "errors_counted": sum(counted.values()),
        "counted_by_tag": dict(counted.most_common()),
        "not_counted_by_reason": dict(not_counted.most_common()),
        "llm_missing": result["llm_missing"],
        "structure_breadth_score": result["structure_breadth_score"],
    }


def main():
    p = argparse.ArgumentParser(description="Trace one Lomb pipeline run in LangSmith.")
    p.add_argument("--transcript", required=True, help="AssemblyAI transcript JSON (.txt or .json)")
    p.add_argument("--speaker", required=True, help="Your speaker label in that file (index sheet column D)")
    p.add_argument("--minutes", type=float, help="Keep only this many minutes, e.g. 15")
    p.add_argument("--start-min", type=float, default=0.0, help="Where the --minutes window starts (default 0)")
    p.add_argument("--session", help="Name shown in LangSmith (default: the file name)")
    p.add_argument("--dry-run", action="store_true", help="Check the transcript and speaker, no API calls")
    p.add_argument("--fake", action="store_true", help="Use FakeProvider: offline wiring check, no API calls")
    args = p.parse_args()

    turns, total_min = load_turns(args.transcript, args.start_min, args.minutes)
    speakers = sorted({t.speaker_id for t in turns})
    if args.speaker not in speakers:
        sys.exit(f"Speaker {args.speaker!r} not in this window. Speakers found: {speakers}")
    sentences = to_sentences(filter_to_target_speaker(turns, args.speaker))
    if not sentences:
        sys.exit("No sentences for that speaker in this window.")

    session = args.session or Path(args.transcript).stem
    if args.minutes is not None and not args.session:
        session += f"_min{args.start_min:g}-{args.start_min + args.minutes:g}"
    chunks = len(lomb_pipeline.chunk_sentence_indices(
        lomb_pipeline.sentence_start_times(filter_to_target_speaker(turns, args.speaker))))

    print(f"Transcript runs about {total_min:.1f} min.")
    if args.minutes is not None or args.start_min:
        end = f"{args.start_min + args.minutes:g}" if args.minutes is not None else "end"
        print(f"Keeping minutes {args.start_min:g} to {end}: {len(turns)} turns.")
    print(f"Session: {session}  |  speaker {args.speaker}  |  {len(sentences)} sentences  |  {chunks} chunk(s) of 15 min")
    print(f"First: {sentences[0]!r}\nLast:  {sentences[-1]!r}")
    print(f"Expect about {chunks * (3 + len(LLM_LABELERS))} LLM calls "
          f"(per chunk: finder, re-check, sorter + {len(LLM_LABELERS)} labeler).")

    if args.dry_run:
        print("\nDry run: no API calls made. Check the first and last sentences are yours.")
        return

    if args.fake:
        provider = fake_provider()
    else:
        for var in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
            if not os.environ.get(var):
                sys.exit(f"{var} is not set.")
        provider = TracedOpenAIProvider(model=os.environ.get("LOMB_LLM_MODEL", "gpt-5.6-luna"),
                                        reasoning_effort=os.environ.get("LOMB_REASONING_EFFORT") or None)
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        print("Warning: LANGSMITH_TRACING is not 'true' -- the run works but nothing is logged.")

    # Parallel calls run in a thread pool; LangSmith's version carries the
    # trace context into each thread so the calls nest under this run.
    from langsmith.utils import ContextThreadPoolExecutor
    lomb_pipeline.ThreadPoolExecutor = ContextThreadPoolExecutor

    reasoning = getattr(provider, "generation_config", {}).get("reasoning")
    metadata = {"session": session, "speaker": args.speaker, "model": provider.model,
                "reasoning": reasoning, "window_min": [args.start_min, args.minutes],
                "fluenceme_versions": lomb_pipeline.fluenceme_versions()}
    print(f"\nModel: {provider.model}  reasoning: {reasoning}")

    @_traceable(name=f"lomb_run {session}", run_type="chain", metadata=metadata,
                tags=["lomb", "fake" if args.fake else "live"],
                process_inputs=lambda inputs: {"session": session, "speaker": args.speaker,
                                               "sentences": len(sentences)},
                process_outputs=lambda out: out["summary"] if isinstance(out, dict) and "summary" in out else out)
    def traced_run(turns_in, speaker):
        start = time.time()
        result = lomb_pipeline.run_pipeline_from_turns(provider, turns_in, speaker)
        return {"result": result, "summary": summarise(result, time.time() - start)}

    out = traced_run(turns, args.speaker)
    result, summary = out["result"], out["summary"]

    run_dir = ROOT / "test_runs" / f"{datetime.now():%Y-%m-%d}_{session}_langsmith{'_fake' if args.fake else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({**summary, **metadata}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:<24} {v}")
    print(f"\nSaved to {run_dir}")
    if not args.fake:
        print(f"Open smith.langchain.com -> project '{os.environ.get('LANGSMITH_PROJECT', 'default')}' -> "
              f"trace 'lomb_run {session}'. Click each child call to see its input, output, tokens and cost.")


if __name__ == "__main__":
    main()
