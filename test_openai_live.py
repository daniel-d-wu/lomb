"""
Live round-trip test against the real OpenAI API -- run this on a machine
with actual internet access, not in the Claude Code sandbox this was
written in (that environment's egress proxy returns a 403 CONNECT
rejection for api.openai.com; there is no way to run this test from there,
confirmed by trying -- same failure mode test_gemini_live.py documents for
Google's endpoint).

Setup:
    pip install openai
    export OPENAI_API_KEY="<your key>"
    cd lomb_prompts/
    python3 test_openai_live.py

What this proves, that offline testing couldn't:
  - The OPENAI_API_KEY is actually valid and authorized to call the model.
  - build_request()'s output is not just locally well-formed (checked
    offline, 2026-09-03 -- see providers/openai_provider.py's module
    docstring for exactly what "locally well-formed" did and didn't cover)
    but genuinely accepted by OpenAI's live endpoint.
  - The model name (gpt-5.6-luna) is a real, currently-served model --
    this is the one thing that could NOT be checked without a live call,
    since it's OpenAI's side, not the SDK's.
  - Whether temperature/top_p are actually honored on this reasoning
    model, or silently ignored (flagged as unverified in
    providers/openai_provider.py, finding 3) -- this can't settle that
    definitively from one call, but a clean response at all is at least
    evidence the params aren't outright rejected.
  - parse_response() correctly unpacks a REAL response (including its
    output_text reconstruction -- see finding 4 in the provider's
    docstring for why that couldn't just call the SDK's own
    response.output_text property), not just a hand-constructed fake one.

Uses the same GDD-2 / "Das Bild haengt an die Wand." scenario
test_gemini_live.py uses, specifically so the two providers' real results
can be compared side by side on the identical input.
Expected answer per that sentence's own grammar (dative required after a
static-location "haengt", not accusative): error=true.
"""

import os
import sys

from providers.openai_provider import OpenAIProvider
from registry import METRIC_PROMPTS


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Run:\n"
            '    export OPENAI_API_KEY="<your key>"\n'
            "first, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    provider = OpenAIProvider()
    sentence = "Das Bild haengt an die Wand."

    print(f"Model:    {provider.model}")
    print(f"Metric:   GDD-2")
    print(f"Input:    {sentence!r}")
    print()

    request = provider.build_request(METRIC_PROMPTS["GDD-2"], sentence)
    print("Built request OK. Calling the live API...")

    try:
        raw_response = provider.call(request)
    except Exception as e:
        print(f"\nLive call failed: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nIf this is an auth error, double check OPENAI_API_KEY is "
            "correct and the account it belongs to has billing/access "
            "enabled for the Responses API and the gpt-5.6 family. If it's "
            "a 400/schema error, the most likely culprit is "
            "_to_openai_schema() in providers/openai_provider.py not "
            "matching some current strict-mode requirement -- check the "
            "error's own message first, it's usually specific about which "
            "schema field it didn't like.",
            file=sys.stderr,
        )
        return 1

    result = provider.parse_response(raw_response)

    print("\n--- Live result ---")
    print(result)

    print("\n--- Sanity check against the known-correct answer for this sentence ---")
    expected_error = True  # dative required ('an der Wand'), sentence uses accusative -- this IS an error
    if result.get("error") == expected_error:
        print(f"OK  -- error={result.get('error')} matches the expected {expected_error}")
    else:
        print(
            f"MISMATCH -- got error={result.get('error')!r}, expected {expected_error}. "
            "Not necessarily a bug (the model can be wrong or the prompt may "
            "need iteration) -- but worth a closer look at the reasoning field."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
