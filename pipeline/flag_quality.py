"""
Is an LLM error flag trustworthy enough to count and to show a learner?

Shared by storage (metric counts) and reporting (Report 1 card selection) so
both apply the same rule. Deterministic, no extra LLM calls. Added
2026-09-24 after the first real run showed the model flagging sentences it
then "corrected" to exactly what was said (see docs/lomb_failure_modes_v1.md
M9). Rejected flags are still stored for review, just not counted or shown.

Word comparisons ignore case, punctuation and filler words (äh, ähm, ...),
so a "correction" that only drops a filler counts as no correction.
"""

import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.filled_pause import FILLER_TOKENS


def content_words(text: str) -> list[str]:
    words = (re.sub(r"[^\w]", "", w.lower()) for w in text.split())
    return [w for w in words if w and w not in FILLER_TOKENS]


def rejection_reason(output: dict | None, said: str) -> str | None:
    """None if this is a trustworthy error flag (or not an error flag at
    all); otherwise a short reason it can't be trusted."""
    if not output or output.get("error") is not True:
        return None
    if output.get("confidence") != "high":
        return "low confidence"
    if "corrected" not in output:
        return None  # legacy result predating the `corrected` field -- can't judge, don't reject
    corrected = output.get("corrected") or ""
    if not corrected.strip():
        return "no correction given"
    if content_words(corrected) == content_words(said):
        return "correction identical to what was said"
    return None


def _opcodes(said: str, corrected: str):
    a, b = content_words(said), content_words(corrected)
    return a, b, difflib.SequenceMatcher(None, a, b).get_opcodes()


def changed_word_count(said: str, corrected: str) -> int:
    """How many words the correction changes -- smaller = a more focused lesson."""
    _, _, ops = _opcodes(said, corrected)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def fix_signature(said: str, corrected: str) -> tuple:
    """Identifies 'the same mistake': the words removed and added, plus the
    word right after the change (usually the noun), so "die Wort -> das
    Wort" said twice groups together but "die Wort" and "die Essen" don't."""
    a, b, ops = _opcodes(said, corrected)
    sig = []
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal":
            context = a[i2] if i2 < len(a) else (a[i1 - 1] if i1 > 0 else "")
            sig.append((tuple(a[i1:i2]), tuple(b[j1:j2]), context))
    return tuple(sig)
