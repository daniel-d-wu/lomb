"""
Live round-trip test against the real Gemini API -- run this on a machine
with actual internet access, not in the Claude Code sandbox this was
written in (that environment's egress proxy returns 403 for
generativelanguage.googleapis.com; there is no way to run this test from
there, confirmed by trying).

Setup:
    pip install google-genai
    export GEMINI_API_KEY="<your gemini_layer_sandbox key>"
    cd lomb_prompts/
    python3 test_gemini_live.py

What this proves, that offline testing couldn't:
  - The GEMINI_API_KEY / AI Studio project ("gemini_layer_sandbox") is
    actually valid and authorized to call the model.
  - build_request()'s output is not just pydantic-valid (checked offline,
    2026-09-02) but genuinely accepted by Google's live endpoint.
  - The model name (gemini-2.5-flash-002) is a real, currently-served
    model version -- this is the one thing that could NOT be checked
    without a live call, since it's Google's side, not the SDK's.
  - parse_response() correctly unpacks a REAL response, not just a
    hand-constructed fake one.

Uses the exact same GDD-2 / "Das Bild haengt an die Wand." scenario as
sample_gemini_response.json in the project docs, specifically so the real
result here can be compared side by side against that constructed example.
Expected answer per that sentence's own grammar (dative required after a
static-location "haengt", not accusative): error=true.
"""

import os
import sys

from providers.gemini_provider import GeminiProvider
from registry import METRIC_PROMPTS


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set. Run:\n"
            '    export GEMINI_API_KEY="<your gemini_layer_sandbox key>"\n'
            "first, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    provider = GeminiProvider()
    sentence = "Das Bild haengt an die Wand."

    print(f"Model:    {provider.model_version}")
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
            "\nIf this is an auth error, double check GEMINI_API_KEY is the "
            "gemini_layer_sandbox key and that the AI Studio project it "
            "belongs to has the Generative Language API enabled.",
            file=sys.stderr,
        )
        return 1

    result = provider.parse_response(raw_response)

    print("\n--- Live result ---")
    print(result)

    print("\n--- Sanity check against sample_gemini_response.json's constructed example ---")
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
