"""
Live round-trip test for 9 of the 10 LLM-assisted metrics -- everything
except FILLED_PAUSE. That one needs real audio bytes attached to the
request (the model has to listen, not just read text), and this codebase
doesn't implement audio-part construction yet -- it's excluded here
deliberately, not silently missed. Get a real, correct answer for the
other 9 before spending effort on the one that needs new code first.

For each of the 9, this reuses that metric's OWN first few-shot example as
the live test case. Every few-shot example already has a known-correct
answer -- that's the whole point of it, teaching the model the pattern --
so this checks the live model against the same known-right-answer standard
each metric's own design already established, rather than inventing a
fresh test sentence with no ground truth to check against.

2026-09-03: switched from GeminiProvider to OpenAIProvider -- OpenAI is
now the primary provider (see providers/openai_provider.py's module
docstring). Gemini's own version of this same test already ran
successfully before the switch; this file now exercises the new default.

Setup: identical to test_openai_live.py -- pip install openai,
set OPENAI_API_KEY, run from inside lomb_prompts/:
    python test_all_metrics_live.py

Nine live API calls happen here, not one -- there's a short pause between
each call as a courtesy against hitting a per-minute rate limit on a brand
new project (the same courtesy the Gemini version of this test used).
"""

import os
import sys
import time

from providers.openai_provider import OpenAIProvider
from registry import METRIC_PROMPTS

# What to compare, per metric -- pulls the one field worth checking out of
# either a live result or a few-shot example's own recorded answer. Six of
# the nine reduce to "does the error boolean match"; FORMULAIC has its own
# field name; STRUCTURE_BREADTH and UNFILLED_PAUSE have list-shaped answers.
METRIC_CHECKS = {
    "GDD-1": lambda r: r["error"],
    "GDD-2": lambda r: r["error"],
    "GVT-1": lambda r: r["error"],
    "GVT-2": lambda r: r["error"],
    "LPF": lambda r: r["error"],
    "LP": lambda r: r["error"],
    "FORMULAIC": lambda r: r["formulaic"],
    "STRUCTURE_BREADTH": lambda r: sorted(r["structures"]),
    "UNFILLED_PAUSE": lambda r: [b["status"] for b in r["boundaries"]],
}

SKIPPED = {
    "FILLED_PAUSE": "needs real audio bytes attached to the request -- not implemented yet",
}

SECONDS_BETWEEN_CALLS = 3


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in this terminal window.", file=sys.stderr)
        return 1

    provider = OpenAIProvider()
    print(f"Model: {provider.model}\n")

    results = []
    metrics_to_test = [k for k in METRIC_PROMPTS if k not in SKIPPED]

    for i, key in enumerate(METRIC_PROMPTS):
        if key in SKIPPED:
            print(f"SKIP      {key:<20} {SKIPPED[key]}")
            continue

        config = METRIC_PROMPTS[key]
        example = config.few_shot_examples[0]
        test_input = example["input"]
        expected = METRIC_CHECKS[key](example["answer"])

        request = provider.build_request(config, test_input)
        try:
            raw_response = provider.call(request)
            result = provider.parse_response(raw_response)
            actual = METRIC_CHECKS[key](result)
        except Exception as e:
            print(f"FAIL      {key:<20} live call errored: {type(e).__name__}: {e}")
            results.append((key, False))
            if metrics_to_test.index(key) < len(metrics_to_test) - 1:
                time.sleep(SECONDS_BETWEEN_CALLS)
            continue

        ok = actual == expected
        status = "OK       " if ok else "MISMATCH "
        print(f"{status} {key:<20} expected={expected!r}  got={actual!r}")
        results.append((key, ok))

        if metrics_to_test.index(key) < len(metrics_to_test) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    print()
    passed = sum(1 for _, ok in results if ok)
    print(f"{passed}/{len(results)} metrics matched their expected answer "
          f"(FILLED_PAUSE not counted -- see SKIP line above).")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
