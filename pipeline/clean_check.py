"""
Grade the AI's own corrections: is a clean sentence really correct German?

Added 2026-10-07 after reviewing 22 Report 1 cards: the error finder writes
the clean (fully corrected) sentence the learner sees, and nothing checked
it. Two independent checks, cheapest first:

  1. added_content() (pipeline/flag_quality.py, code only): does the clean
     sentence contain real words the learner never said and no fix
     introduced? ('letztes Mal war es im Juni' -- the learner never said
     June.)
  2. recheck_clean() (one LLM call per chunk): send the clean sentences
     back through the error finder as if the learner had said them. A
     correct sentence comes back with no high-confidence error; anything
     the finder flags in its own correction means the correction isn't
     clean.

Each check's result is stored per error (correction_checks table, v1.4)
and a card is only shown when every check passed. Possible later additions (need a
download): a rule-based grammar checker (LanguageTool) and a German
language-model naturalness score.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.batching import batch_input
from pipeline.error_finder import FINDER_CONFIG

RECHECK_KEY = "CLEAN_RECHECK"
# The finder's own prompt under its own label, so the call is counted,
# fingerprinted and testable separately from the finder.
RECHECK_CONFIG = replace(FINDER_CONFIG, key=RECHECK_KEY)


def recheck_clean(provider, cleans: list[str]) -> dict[str, list[str]]:
    """One provider call over every distinct clean sentence. Returns
    {clean sentence: [problem, ...]} for each one the finder still finds a
    high-confidence error in (mishearing flags are ignored -- these
    sentences didn't come from speech recognition)."""
    unique = list(dict.fromkeys(c.strip() for c in cleans if c and c.strip()))
    if not unique:
        return {}
    raw = provider.classify(RECHECK_CONFIG, batch_input(list(enumerate(unique))))
    problems: dict[str, list[str]] = {}
    for e in raw.get("errors", []):
        i = e.get("index")
        if not (isinstance(i, int) and 0 <= i < len(unique)):
            continue
        if e.get("confidence") != "high" or e.get("possible_transcription_error"):
            continue
        problems.setdefault(unique[i], []).append(
            f"{e.get('said', '')!r} -> {e.get('fix', '')!r}: {e.get('reasoning', '')}")
    return problems
