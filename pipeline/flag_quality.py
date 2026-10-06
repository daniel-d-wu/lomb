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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.filled_pause import FILLER_TOKENS


def content_words(text: str) -> list[str]:
    words = (re.sub(r"[^\w]", "", w.lower()) for w in text.split())
    return [w for w in words if w and w not in FILLER_TOKENS]


def apply_fix(sentence: str, said: str, fix: str) -> str | None:
    """The sentence with the flagged words `said` replaced by `fix`, or None
    if `said` isn't in the sentence. Built in code (2026-10-06) instead of
    asking the model to rewrite the whole sentence: the model got the fix
    right in its reasoning but garbled the rewrite ("mit der" -> wrote "mit
    den"; inserted "dem" without removing "das"; appended "benutzt.").
    First occurrence wins; the finder is told to make `said` unique."""
    if not said.strip():
        return None
    i = sentence.find(said)
    if i < 0:
        i = sentence.lower().find(said.lower())
        if i < 0:
            return None
    return re.sub(r" {2,}", " ", sentence[:i] + fix + sentence[i + len(said):]).strip()


# German separable-verb prefixes: a fix's "hängt ... ab" can legitimately
# reappear in the clean sentence as "abhängt" (and "zu ... auf" as "aufzu-").
SEPARABLE_PREFIXES = (
    "ab", "an", "auf", "aus", "bei", "ein", "fest", "fort", "her", "hin", "los", "mit", "nach", "vor",
    "weg", "weiter", "zu", "zurück", "zusammen", "dar", "heraus", "herein", "hinaus", "vorbei",
)


def _word_present(w: str, words: set[str]) -> bool:
    if w in words:
        return True
    for cw in words:
        for p in SEPARABLE_PREFIXES:
            if w == p and cw.startswith(p) and len(cw) > len(p) + 2:   # "ab" inside "abhängt"
                return True
            if cw in (p + w, p + "zu" + w) or (w == "zu" and cw.startswith(p + "zu")):  # "hängt"/"aufzunehmen"
                return True
    return False


def clean_has_fix(clean: str, said: str, fix: str) -> bool:
    """Does the clean (fully corrected) sentence contain the words this fix
    adds? Guards against a clean sentence that contradicts the fix -- e.g.
    the card says "die -> der" but the clean sentence still says "den".
    Separable verbs are matched in either shape (2026-10-06: "hängt ... davon
    ab" vs "davon abhängt" wrongly blocked real cards -- failure mode M25)."""
    if not clean.strip():
        return False
    added = Counter(content_words(fix)) - Counter(content_words(said))
    words = set(content_words(clean))
    return all(_word_present(w, words) for w in added)


def rejection_reason(output: dict | None, said: str) -> str | None:
    """None if this is a trustworthy error flag (or not an error flag at
    all); otherwise a short reason it can't be trusted. `said` is the full
    sentence the flag is about."""
    if not output or output.get("error") is not True:
        return None
    if output.get("possible_transcription_error"):
        return "possible transcription error"  # finder thinks Whisper misheard -- not the learner's mistake
    if output.get("said_not_found"):
        return "flagged words not in sentence"  # model pointed at words the learner didn't say -- never guessed
    if output.get("untagged"):
        return "sorter gave no tag"  # found, but never categorised -- counted nowhere rather than guessed
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
