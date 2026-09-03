"""
Lists the Gemini models your API key can actually use right now, and which
ones support generateContent (the method GeminiProvider.call() needs).
Run this the same way as test_gemini_live.py -- same folder, same
GEMINI_API_KEY already set in this terminal.
"""

import os
import sys

from google import genai


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set in this terminal window.", file=sys.stderr)
        return 1

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print("Models available to your key, that support generateContent:\n")
    found_any = False
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            found_any = True
            print(f"  {model.name}")

    if not found_any:
        print("  (none found -- printing every model returned, unfiltered, for debugging)\n")
        for model in client.models.list():
            print(f"  {model.name}  actions={getattr(model, 'supported_actions', None)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
