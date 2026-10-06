"""
Call 2 of 2 in error detection: give every error the finder found exactly
one error-fluenceme tag (GDD-1 ... LP), or OTHER_ERROR.

The category list, definitions and examples are built from the registry's
ERROR_TAGS -- i.e. from each prompts/<name>.py ErrorTagConfig -- so a new
error fluenceme reaches the sorter by adding its file, nothing else.

The sorter never decides WHETHER something is an error (that was the
finder's job) and it can't drop one: every error it's given comes back
with a tag. An error it leaves out is reported by the caller, never
silently counted as a category.

Because the found errors are stored, a new or changed tag can be applied
to past sessions by re-running only this step (see sort_errors()).
"""

import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metric_types import PromptConfig
from pipeline.registry import ERROR_TAGS, OTHER_ERROR_KEY

SORTER_KEY = "ERROR_SORTER"

_TIE_BREAKS = """Tie-breaks:
- Article/determiner forms: if the form used would be right for the needed case with a different gender of noun, it is a gender error (GDD-3). If it is the wrong case for the noun's real gender, use the case category: GDD-1 after an always-dative preposition, GDD-2 after a two-way preposition, GDD-6 otherwise.
- Word choice: if English explains the wording (a literal translation, the preposition English would use, a false friend), LPF; otherwise LP.
- If one wrong form is wrong in two ways, tag the category the correction mainly fixes."""


def _category_block() -> str:
    lines = []
    for tag in ERROR_TAGS.values():
        line = f"{tag.key} -- {tag.definition}"
        if tag.not_this:
            line += f" Not this: {tag.not_this}"
        lines.append(line)
    return "\n".join(lines)


def _instruction() -> str:
    return f"""You are sorting errors made by an adult English speaker learning German. Each input item is ONE error that has already been found in a transcript of their speech: "sentence" is what they said, "said" is the wrong part, "corrected" is the sentence with only this error fixed. Do not judge whether it is really an error -- only choose its category.

Give every item exactly one tag from the categories below: the most specific one that fits what the correction changes. If none fits, use {OTHER_ERROR_KEY}.

Categories:
{_category_block()}

{_TIE_BREAKS}

Return exactly one entry per item, copying its "id".

Respond only in the fixed JSON shape you have been given."""


def said_part(sentence: str, corrected: str) -> str:
    """The words of `sentence` the correction removes or replaces (for an
    insertion, the word right after the insertion point) -- used to fill
    `said` in the examples, which only record sentence + correction."""
    a, b = sentence.split(), corrected.split()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("replace", "delete"):
            return " ".join(a[i1:i2])
        if tag == "insert":
            return a[i1] if i1 < len(a) else a[-1]
    return ""


def _example() -> dict:
    """One combined few-shot example: every tag's examples, interleaved
    round-robin so no tag sits in one block (avoids position bias)."""
    queues = [[(t.key, ex) for ex in t.examples] for t in ERROR_TAGS.values()]
    ordered = []
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.pop(0))
    items, answers = [], []
    for i, (key, ex) in enumerate(ordered):
        items.append({"id": i, "sentence": ex["sentence"], "said": said_part(ex["sentence"], ex["corrected"]),
                      "corrected": ex["corrected"]})
        answers.append({"id": i, "tag": key})
    return {"input": json.dumps(items, ensure_ascii=False), "answer": {"tags": answers}}


SORTER_CONFIG = PromptConfig(
    key=SORTER_KEY,
    system_instruction=_instruction(),
    few_shot_examples=[_example()],
    response_schema={
        "type": "OBJECT",
        "properties": {"tags": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"id": {"type": "INTEGER"},
                           "tag": {"type": "STRING", "enum": list(ERROR_TAGS)}},
            "required": ["id", "tag"],
        }}},
        "required": ["tags"],
    },
    input_kind="error_batch",
)


def sort_errors(provider, errors: list[dict]) -> dict[int, str]:
    """One provider call. errors: [{"id", "sentence", "said", "corrected"}].
    Returns {id: tag} for every id the model tagged with a known tag; an
    id it skipped is absent (the caller reports it)."""
    payload = [{k: e[k] for k in ("id", "sentence", "said", "corrected")} for e in errors]
    raw = provider.classify(SORTER_CONFIG, json.dumps(payload, ensure_ascii=False))
    wanted = {e["id"] for e in errors}
    tags: dict[int, str] = {}
    for item in raw.get("tags", []):
        if item.get("id") in wanted and item.get("id") not in tags and item.get("tag") in ERROR_TAGS:
            tags[item["id"]] = item["tag"]
    return tags


if __name__ == "__main__":
    print(SORTER_CONFIG.system_instruction)
    print("\nFew-shot items:")
    for item, ans in zip(json.loads(SORTER_CONFIG.few_shot_examples[0]["input"]),
                         SORTER_CONFIG.few_shot_examples[0]["answer"]["tags"]):
        assert item["said"], f"example has no detectable changed part: {item}"
        print(f"  {ans['tag']:<12} said={item['said']!r:<28} {item['sentence']}")
    assert {a["tag"] for a in SORTER_CONFIG.few_shot_examples[0]["answer"]["tags"]} == set(ERROR_TAGS), \
        "every tag, including the catch-all, should have at least one example"
    print("\nSorter config consistent: every tag has an example.")
