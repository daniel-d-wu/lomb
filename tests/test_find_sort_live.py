"""
Live round-trip test for find-then-sort (replaces test_all_metrics_live.py,
which tested one narrow detector prompt per metric -- those no longer exist).

Two live calls: the error finder over a short list of known sentences, then
the sorter over whatever it found. The known sentences are each error tag's
own first example (verbatim corpus errors, from prompts/*.py) plus two
error-free real sentences, so every expectation has ground truth:

  - every error-tag example sentence gets at least one error found
  - at least one error found in it is tagged with that example's tag
  - the error-free sentences get no high-confidence error

A miss is printed as a MISMATCH, not raised -- the model can be wrong and
this is a quality probe, not a unit test. The wiring itself is covered
offline (pipeline/pipeline.py's self-test, tests/smoketest_openai_offline.py).

Setup: OPENAI_API_KEY in the environment, then from lomb_prompts/:
    python tests/test_find_sort_live.py [--model gpt-6-luna] [--reasoning-effort low]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pipeline import find_and_sort
from pipeline.registry import ERROR_TAGS
from providers.openai_provider import OpenAIProvider

CLEAN = [
    "Ja, ich habe noch viel Zeit.",
    "Mhm, ja, ich verstehe— jetzt verstehe ich, was du meinst.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default=None)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    expected = [(tag.key, tag.examples[0]["sentence"]) for tag in ERROR_TAGS.values() if tag.examples]
    sentences = [s for _, s in expected] + CLEAN
    provider = OpenAIProvider(model=args.model, reasoning_effort=args.reasoning_effort)
    print(f"Model: {provider.model}  settings: {provider.generation_config}\n")

    errors, tags, calls = find_and_sort(provider, list(enumerate(sentences)))
    by_sentence: dict[int, list[tuple[dict, str | None]]] = {}
    for pos, e in enumerate(errors):
        by_sentence.setdefault(e["index"], []).append((e, tags.get(pos)))

    misses = 0
    for i, (key, sentence) in enumerate(expected):
        found = by_sentence.get(i, [])
        got = [t for _, t in found]
        ok = key in got
        misses += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} expected {key:<12} got {got}  {sentence}")
        for e, t in found:
            print(f"           {t}: {e['said']!r} -> {e['corrected']!r} ({e['confidence']})")
    for j, sentence in enumerate(CLEAN, start=len(expected)):
        high = [e for e, _ in by_sentence.get(j, []) if e["confidence"] == "high"]
        misses += bool(high)
        print(f"{'OK      ' if not high else 'MISMATCH'} expected no error  {sentence}")
        for e in high:
            print(f"           flagged {e['said']!r} -> {e['corrected']!r}")
    print(f"\n{calls} live calls, {len(errors)} errors found, {misses} mismatch(es) of {len(sentences)} sentences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
