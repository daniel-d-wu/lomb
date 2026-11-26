"""
LangSmith experiments on Lomb's CURRENT error detection (find-then-sort).

Replaces lomb_merge_test.py, which compared the retired one-call-per-metric
design. There is no alternative design left to compare against, so this
compares the pipeline with ITSELF (consistency) and with OTHER MODEL
SETTINGS (model choice):

  Consistency   the same 15-minute chunk, run N times with the same model.
                Do the error counts and tags stay put? If they move between
                identical runs, a change between two of your lessons can't
                be read as progress.
  Model choice  the same chunk under two model settings, e.g. gpt-5.6-luna
                (reasoning none) vs gpt-6-luna (reasoning low).

What runs is your real code: pipeline.find_and_sort() -- error finder,
clean re-check, sorter -- with your OpenAIProvider (wrapped for tracing by
lomb_trace.py). The structure labeler and the plain-Python fluencemes are
left out: the labeler isn't part of error counting, and the Python ones give
the same answer every time.

Put this file in lomb_prompts/ next to lomb_trace.py and run from there:

  1. Add a chunk to the dataset (no API calls):
     python lomb_experiment.py build --transcript PATH --speaker A --minutes 15 --session partner_lesson1_min0-15

  2. Consistency: run it 3 times with one model setting:
     python lomb_experiment.py run --repeats 3

  3. Model choice: run again with another setting, then compare the two
     experiments in LangSmith (or the CSVs this writes):
     python lomb_experiment.py run --repeats 3 --model gpt-6-luna --reasoning low

Environment: OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true,
LANGSMITH_PROJECT=lomb. Default model/reasoning: LOMB_LLM_MODEL /
LOMB_REASONING_EFFORT, the same variables service/cloud.py reads.

Cost: 3 calls per repeat per chunk (finder, re-check, sorter); a 15-minute
chunk is a few cents for 3 repeats.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline.pipeline as lomb_pipeline  # noqa: E402
from pipeline.flag_quality import rejection_reason  # noqa: E402
from pipeline.registry import ERROR_TAGS  # noqa: E402
from transcript_processing.speaker_filter import filter_to_target_speaker, to_sentences  # noqa: E402

from lomb_trace import TracedOpenAIProvider, fake_provider, load_turns  # noqa: E402

DEFAULT_DATASET = "lomb-error-detection"


# --------------------------------------------------------------------------
# The target: one chunk through find_and_sort, the way the pipeline does it
# --------------------------------------------------------------------------

def make_provider(model: str, reasoning: str | None, fake: bool):
    if fake:
        return fake_provider()
    return TracedOpenAIProvider(model=model, reasoning_effort=reasoning)


def detect(provider, sentences: list[str]) -> dict:
    """find_and_sort on the whole chunk, then the pipeline's own bookkeeping
    (_error_entries) and its own counting rule (flag_quality.rejection_reason)."""
    chunk_output = lomb_pipeline.find_and_sort(provider, list(enumerate(sentences)))
    results, found, untagged = lomb_pipeline._error_entries([chunk_output], sentences)
    errors = []
    for tag, entries in results.items():
        for entry in entries:
            out = entry["output"]
            reason = rejection_reason(out, entry["input"])
            errors.append({
                "index": out["sentence_indices"][0],
                "said": out["said"],
                "fix": out["fix"],
                "tag": None if out.get("untagged") else tag,
                "gravity": out["gravity"],
                "confidence": out["confidence"],
                "counted": reason is None,
                "not_counted_reason": reason,
            })
    errors.sort(key=lambda e: (e["index"], e["said"]))
    counted = Counter(e["tag"] for e in errors if e["counted"])
    return {
        "errors": errors,
        "found": len(errors),
        "counted": sum(counted.values()),
        "counted_by_tag": dict(sorted(counted.items())),
        "untagged": len(untagged),
        "calls": chunk_output[2],
    }


# --------------------------------------------------------------------------
# Matching the same error across runs
# --------------------------------------------------------------------------

def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def same_error(a: dict, b: dict) -> bool:
    """Same sentence and the flagged words mostly overlap. Runs often pick a
    slightly different span ('ein Urlaub' vs 'Urlaub'), so exact text
    matching would undercount agreement."""
    if a["index"] != b["index"]:
        return False
    wa, wb = _words(a["said"]), _words(b["said"])
    if not wa or not wb:
        return a["said"].strip() == b["said"].strip()
    return len(wa & wb) / len(wa | wb) >= 0.5 or wa <= wb or wb <= wa


def consistency(runs: list[list[dict]]) -> tuple[dict, list[dict]]:
    """runs: one error list per repeat of the same chunk. Groups matching
    errors across repeats, then scores how stable they are."""
    groups: list[dict] = []   # {"index", "said", "by_run": {run_no: error}}
    for run_no, errors in enumerate(runs):
        for e in errors:
            for g in groups:
                if run_no not in g["by_run"] and same_error(g["by_run"][min(g["by_run"])], e):
                    g["by_run"][run_no] = e
                    break
            else:
                groups.append({"index": e["index"], "said": e["said"], "by_run": {run_no: e}})

    n = len(runs)
    in_all = [g for g in groups if len(g["by_run"]) == n]
    same_tag = [g for g in in_all if len({e["tag"] for e in g["by_run"].values()}) == 1]
    same_counted = [g for g in in_all if len({e["counted"] for e in g["by_run"].values()}) == 1]
    per_tag = defaultdict(list)
    for errors in runs:
        c = Counter(e["tag"] for e in errors if e["counted"])
        for tag in ERROR_TAGS:
            per_tag[tag].append(c.get(tag, 0))
    scores = {
        "repeats": n,
        "found_per_run": [len(r) for r in runs],
        "counted_per_run": [sum(e["counted"] for e in r) for r in runs],
        "distinct_errors_any_run": len(groups),
        "found_in_every_run": len(in_all),
        "share_found_in_every_run": round(len(in_all) / len(groups), 3) if groups else None,
        "same_tag_every_run": round(len(same_tag) / len(in_all), 3) if in_all else None,
        "same_counted_decision_every_run": round(len(same_counted) / len(in_all), 3) if in_all else None,
        "counted_by_tag_per_run": {t: v for t, v in per_tag.items() if any(v)},
    }
    return scores, groups


# --------------------------------------------------------------------------
# Per-run evaluators (shown as columns in LangSmith)
# --------------------------------------------------------------------------

def errors_found(outputs: dict) -> dict:
    return {"key": "errors_found", "score": outputs.get("found", 0)}


def errors_counted(outputs: dict) -> dict:
    return {"key": "errors_counted", "score": outputs.get("counted", 0)}


def errors_untagged(outputs: dict) -> dict:
    return {"key": "errors_untagged", "score": outputs.get("untagged", 0)}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_build(args):
    turns, total_min = load_turns(args.transcript, args.start_min, args.minutes)
    speakers = sorted({t.speaker_id for t in turns})
    if args.speaker not in speakers:
        sys.exit(f"Speaker {args.speaker!r} not in this window. Speakers found: {speakers}")
    target = filter_to_target_speaker(turns, args.speaker)
    sentences = to_sentences(target)
    chunks = lomb_pipeline.chunk_sentence_indices(lomb_pipeline.sentence_start_times(target))
    if len(chunks) > 1:
        sys.exit(f"This window covers {len(chunks)} pipeline chunks. Use --minutes 15 (or less) so the test "
                 "matches one production chunk, and --start-min to pick which one (0, 15, 30, ...).")
    if not sentences:
        sys.exit("No sentences for that speaker in this window.")
    session = args.session or Path(args.transcript).stem
    if args.minutes is not None and not args.session:
        session += f"_min{args.start_min:g}-{args.start_min + args.minutes:g}"

    print(f"Transcript runs about {total_min:.1f} min; this window has {len(turns)} turns.")
    print(f"Session: {session}  |  speaker {args.speaker}  |  {len(sentences)} sentences")
    print(f"First: {sentences[0]!r}\nLast:  {sentences[-1]!r}")
    if args.dry_run:
        print("\nDry run: nothing uploaded. Check the first and last sentences are yours.")
        return

    from langsmith import Client
    ls = Client()
    try:
        dataset = ls.read_dataset(dataset_name=args.dataset)
    except Exception:
        dataset = ls.create_dataset(dataset_name=args.dataset,
                                    description="Lomb: one 15-min chunk per example, for find-then-sort experiments")
    ls.create_examples(dataset_id=dataset.id, examples=[{
        "inputs": {"session": session, "speaker": args.speaker, "sentences": sentences},
        "metadata": {"transcript": str(args.transcript), "window_min": [args.start_min, args.minutes]},
    }])
    print(f"\nAdded '{session}' to dataset '{args.dataset}'. Next: python lomb_experiment.py run --repeats 3")


def cmd_run(args):
    from langsmith import evaluate
    from langsmith.utils import ContextThreadPoolExecutor

    lomb_pipeline.ThreadPoolExecutor = ContextThreadPoolExecutor  # keep parallel calls nested in the trace
    model = args.model or os.environ.get("LOMB_LLM_MODEL", "gpt-5.6-luna")
    reasoning = args.reasoning or os.environ.get("LOMB_REASONING_EFFORT") or None
    provider = make_provider(model, reasoning, args.fake)
    label = "fake" if args.fake else f"{model}-{reasoning or 'default'}"
    print(f"Model: {provider.model}  reasoning: {getattr(provider, 'generation_config', {}).get('reasoning')}"
          f"  repeats: {args.repeats}")

    def target(inputs: dict) -> dict:
        return detect(provider, inputs["sentences"])

    results = evaluate(
        target, data=args.dataset, evaluators=[errors_found, errors_counted, errors_untagged],
        experiment_prefix=f"lomb-{label}", num_repetitions=args.repeats, max_concurrency=1,
        metadata={"model": provider.model, "reasoning": reasoning,
                  "fluenceme_versions": lomb_pipeline.fluenceme_versions()},
    )

    by_example: dict = defaultdict(lambda: {"session": None, "sentences": None, "runs": []})
    for row in results:
        ex, run = row["example"], row["run"]
        slot = by_example[ex.id]
        slot["session"], slot["sentences"] = ex.inputs["session"], ex.inputs["sentences"]
        slot["runs"].append((run.outputs or {}).get("errors", []))

    out_dir = ROOT / "test_runs" / f"{datetime.now():%Y-%m-%d_%H%M}_experiment_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_scores = {}
    for slot in by_example.values():
        scores, groups = consistency(slot["runs"])
        all_scores[slot["session"]] = scores
        with open(out_dir / f"{slot['session']}_errors_across_runs.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            n = len(slot["runs"])
            w.writerow(["sentence_no", "sentence", "said"] + [f"run{i + 1}_tag" for i in range(n)]
                       + [f"run{i + 1}_counted" for i in range(n)] + ["found_in_runs", "same_tag"])
            for g in sorted(groups, key=lambda g: g["index"]):
                tags = [g["by_run"][i]["tag"] if i in g["by_run"] else "" for i in range(n)]
                counted = [g["by_run"][i]["counted"] if i in g["by_run"] else "" for i in range(n)]
                present = [t for i, t in enumerate(tags) if i in g["by_run"]]
                w.writerow([g["index"] + 1, slot["sentences"][g["index"]], g["said"], *tags, *counted,
                            len(g["by_run"]), len(set(present)) == 1])
    (out_dir / "consistency.json").write_text(json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Consistency ===")
    for session, s in all_scores.items():
        print(f"\n{session}")
        for k, v in s.items():
            print(f"  {k:<34} {v}")
    print(f"\nSaved to {out_dir}  (one CSV per session: every error, which runs found it, its tag in each run)")
    print("How to read it: share_found_in_every_run and same_tag_every_run near 1.0 = stable. "
          "counted_by_tag_per_run shows how far each category's count moves between identical runs -- "
          "that's the noise any lesson-to-lesson change has to beat.")
    if not args.fake:
        print("In LangSmith: Datasets & Experiments -> "
              f"'{args.dataset}' -> pick experiments -> Compare.")


def main():
    p = argparse.ArgumentParser(description="Consistency and model-choice experiments on find-then-sort.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Add one 15-min chunk of a transcript to the dataset")
    b.add_argument("--transcript", required=True, help="AssemblyAI transcript JSON (.txt or .json)")
    b.add_argument("--speaker", required=True, help="Your speaker label (index sheet column D)")
    b.add_argument("--minutes", type=float, default=15.0, help="Window length, at most 15 (default 15)")
    b.add_argument("--start-min", type=float, default=0.0, help="Window start: 0, 15, 30, ... (default 0)")
    b.add_argument("--session", help="Name shown in LangSmith (default: file name + window)")
    b.add_argument("--dataset", default=DEFAULT_DATASET)
    b.add_argument("--dry-run", action="store_true", help="Check the transcript and speaker, upload nothing")

    r = sub.add_parser("run", help="Run find-then-sort on every chunk in the dataset, N times")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--model", help="Default: LOMB_LLM_MODEL or gpt-5.6-luna")
    r.add_argument("--reasoning", help="none | low | medium ... Default: LOMB_REASONING_EFFORT or the provider's")
    r.add_argument("--dataset", default=DEFAULT_DATASET)
    r.add_argument("--fake", action="store_true", help="FakeProvider, no API calls (wiring check)")

    args = p.parse_args()
    {"build": cmd_build, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
