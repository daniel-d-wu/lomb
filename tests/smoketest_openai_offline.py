"""
Offline smoke test for the OpenAI provider -- everything checkable WITHOUT a
live OPENAI_API_KEY (a fake key is used; with network access the call fails
at authentication, without it at connection -- either way only after the
request was built and validated locally).

Covers every prompt the pipeline actually sends (2026-09-24 find-then-sort):
the error finder, the error sorter, and each labeler's batch prompt
(STRUCTURE_BREADTH). Error fluencemes (GDD-1 ... LP) are no longer prompts
of their own -- they reach the model only as categories inside the sorter.

1. build_request() reaches client.responses.create()'s local handling
   cleanly -- no TypeError, no local validation error.
2. _to_openai_schema()'s output is real, strict-mode-correct JSON Schema
   (jsonschema library; every property required, additionalProperties false,
   at every nesting level).
3. Each prompt's own few-shot answer, and an empty answer, are accepted by
   its converted schema and round-trip through parse_response().

What this does NOT prove: that OpenAI's live API accepts the requests or
answers sensibly -- that's tests/test_openai_live.py and
tests/test_find_sort_live.py.
"""

import json
import sys
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.batching import batch_input
from pipeline.error_finder import FINDER_CONFIG
from pipeline.error_sorter import SORTER_CONFIG
from pipeline.pipeline import LABELER_BATCH_PROMPTS
from providers.openai_provider import OpenAIProvider, _to_openai_schema

FAKE_KEY = "sk-test-dummy-key-for-smoketest-only"

SENTENCES = [
    (0, "Letztes Jahr reise ich nach Berlin und besuche meine Familie."),
    (1, "Obwohl ich muede bin, ich gehe heute noch joggen."),
    (2, "Wenn ich mehr Zeit haette, wuerde ich jeden Tag Deutsch ueben."),
]
SORTER_ITEMS = [{"id": 0, "sentence": "Ich habe mich gefreut fuer das Konzert.", "said": "fuer",
                 "corrected": "Ich habe mich gefreut auf das Konzert."}]

PROMPTS = {FINDER_CONFIG.key: FINDER_CONFIG, SORTER_CONFIG.key: SORTER_CONFIG, **LABELER_BATCH_PROMPTS}


def sample_for(config) -> str:
    if config.input_kind == "error_batch":
        return json.dumps(SORTER_ITEMS, ensure_ascii=False)
    if config.input_kind == "sentence_batch":
        return batch_input(SENTENCES)
    raise ValueError(f"unhandled input_kind {config.input_kind!r}")


def empty_answer(config) -> dict:
    """The smallest valid answer: no errors / no tags / no results."""
    (list_key,) = config.response_schema["properties"]
    return {list_key: []}


def check_schema_strict_invariants(schema, path="$"):
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        assert schema.get("additionalProperties") is False, f"{path}: missing additionalProperties: false"
        missing = set(schema.get("properties", {})) - set(schema.get("required", []))
        assert not missing, f"{path}: properties not in required: {missing}"
        for key, sub in schema.get("properties", {}).items():
            check_schema_strict_invariants(sub, f"{path}.{key}")
    if "items" in schema:
        check_schema_strict_invariants(schema["items"], f"{path}[]")


def wrap_as_openai_response(answer: dict) -> dict:
    return {"output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps(answer, ensure_ascii=False)}]}]}


def main() -> int:
    provider = OpenAIProvider()
    client = OpenAI(api_key=FAKE_KEY)
    failures = []

    print("=== 1. build_request() reaches the network layer cleanly ===\n")
    for key, config in PROMPTS.items():
        req = provider.build_request(config, sample_for(config))
        try:
            client.responses.create(**req)
            failures.append((key, "UNEXPECTED SUCCESS with a fake key"))
        except Exception as e:
            kind = type(e).__name__
            if kind in ("APIConnectionError", "AuthenticationError"):
                print(f"  {key:<20} OK (blocked only at {kind})")
            else:
                failures.append((key, f"LOCAL ERROR: {kind}: {e}"))
                print(f"  {key:<20} FAIL: {kind}: {e}")

    print("\n=== 2. converted schemas are valid, strict-mode-correct JSON Schema ===\n")
    for key, config in PROMPTS.items():
        converted = _to_openai_schema(config.response_schema)
        try:
            Draft202012Validator.check_schema(converted)
            check_schema_strict_invariants(converted)
            print(f"  {key:<20} OK")
        except (jsonschema.exceptions.SchemaError, AssertionError) as e:
            failures.append((key, f"SCHEMA INVALID: {e}"))
            print(f"  {key:<20} FAIL: {e}")

    print("\n=== 3. each prompt's own few-shot answer and an empty answer round-trip ===\n")
    for key, config in PROMPTS.items():
        validator = Draft202012Validator(_to_openai_schema(config.response_schema))
        for label, answer in (("few-shot", config.few_shot_examples[0]["answer"]), ("empty", empty_answer(config))):
            errors = list(validator.iter_errors(answer))
            if errors:
                failures.append((key, f"{label} answer rejected by its own schema: {errors[0].message}"))
                print(f"  {key:<20} {label:<9} REJECTED: {errors[0].message}")
            elif provider.parse_response(wrap_as_openai_response(answer)) != answer:
                failures.append((key, f"{label} answer: parse_response() round-trip mismatch"))
                print(f"  {key:<20} {label:<9} FAIL: round-trip mismatch")
            else:
                print(f"  {key:<20} {label:<9} OK")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for key, msg in failures:
            print(f"  - {key}: {msg}")
        return 1
    print("ALL OFFLINE SMOKE CHECKS PASSED (live round-trip still needs tests/test_find_sort_live.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
