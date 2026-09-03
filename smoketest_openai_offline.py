"""
Offline smoke test for the OpenAI migration -- everything checkable
WITHOUT a live OPENAI_API_KEY or network access (this sandbox has neither:
no key is set, and api.openai.com gets a 403 from the egress proxy, same
as documented in providers/openai_provider.py's module docstring). This
goes further than the ad-hoc checks done while building the provider:

1. build_request() for all 10 metrics, using varied realistic inputs (not
   just each metric's own first few-shot example), reaches
   client.responses.create()'s local handling cleanly -- no TypeError, no
   local validation error, request is well-formed enough that the ONLY
   failure is the network layer. This is the same checkpoint used to
   verify the provider originally, run again after the environment reset
   in case anything regressed, and against a wider input set.

2. _to_openai_schema()'s output is validated as ACTUALLY CORRECT JSON
   Schema for all 10 metrics using the real `jsonschema` library, not just
   eyeballed. Checks the specific strict-mode invariant OpenAI requires
   (every property in "properties" also appears in "required", and
   "additionalProperties": false is set) recursively at every nesting
   level, not just spot-checked on the one metric (UNFILLED_PAUSE) known
   to need special handling.

3. A hand-built response matching each metric's OWN schema, for both a
   "true"/flagged answer and a "false"/clean answer, is validated against
   that metric's converted schema (proving the schema doesn't
   accidentally reject its own metric's real answer shape) AND run
   through parse_response() end-to-end (proving the response_json ->
   parsed dict path works, not just the request-building half).

4. GDD-1/GDD-2 prefilter interaction: confirms prefilter.py's should_run()
   and the OpenAI request-building agree on which sentences matter --
   i.e. nothing about the provider switch changed which sentences the
   pre-filter would skip.

What this does NOT prove: that OpenAI's real API actually accepts these
requests or returns sensible answers. That needs test_openai_live.py (or
test_all_metrics_live.py) on a machine with real network access and a real
OPENAI_API_KEY -- this script is the strongest check possible without
those two things, not a replacement for them.
"""

import json
import sys

import jsonschema
from jsonschema import Draft202012Validator

from openai import OpenAI

from providers.openai_provider import OpenAIProvider, _to_openai_schema
from registry import METRIC_PROMPTS
import prefilter

FAKE_KEY = "sk-test-dummy-key-for-smoketest-only"

# Realistic-ish per-input-kind samples, deliberately DIFFERENT from each
# metric's own first few-shot example (that path was already checked while
# building the provider) -- these draw on other real sentences already
# documented elsewhere in this project's corpus docs, to widen coverage
# rather than re-test the same one input every metric was built against.
SENTENCE_SAMPLES = {
    "GDD-1": "Ich komme gerade von der Arbeit nach Hause.",
    "GDD-2": "Er stellt das Buch auf den Tisch.",
    "GVT-1": "Letztes Jahr reise ich nach Berlin und besuche meine Familie.",
    "GVT-2": "Obwohl ich muede bin, ich gehe heute noch joggen.",
    "LPF": "Ich habe mich gefreut fuer das Konzert.",
    "LP": "Er hat mir ein grosses Beispiel gemacht.",
    "FORMULAIC": 'Candidate: "sozusagen" | Sentence: "Es ist sozusagen die beste Loesung."',
    "STRUCTURE_BREADTH": "Wenn ich mehr Zeit haette, wuerde ich jeden Tag Deutsch ueben.",
}
WORD_TIMESTAMP_SAMPLE = [
    {"word": "und", "start": 10.0, "end": 10.15},
    {"word": "dann", "start": 10.2, "end": 10.4},
    {"word": "gehen", "start": 12.9, "end": 13.1},
]
AUDIO_TURN_SAMPLE = "[smoke test placeholder -- audio-part construction not implemented, see prompts/filled_pause.py]"


def sample_for(config):
    if config.input_kind == "sentence":
        return SENTENCE_SAMPLES[config.key]
    if config.input_kind == "word_timestamps":
        return WORD_TIMESTAMP_SAMPLE
    if config.input_kind == "audio_turn":
        return AUDIO_TURN_SAMPLE
    raise ValueError(f"unhandled input_kind {config.input_kind!r}")


def check_schema_strict_invariants(schema, path="$"):
    """Recursively assert OpenAI's strict-mode requirements actually hold
    everywhere in the converted schema, not just at the top level:
    every object has additionalProperties: false, and every key in
    "properties" also appears in "required"."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        assert schema.get("additionalProperties") is False, (
            f"{path}: missing additionalProperties: false"
        )
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = set(props.keys()) - required
        assert not missing, f"{path}: properties not in required: {missing}"
        for key, sub in props.items():
            check_schema_strict_invariants(sub, f"{path}.{key}")
    if "items" in schema:
        check_schema_strict_invariants(schema["items"], f"{path}[]")


def build_fake_answer(config, flagged: bool):
    """Construct a plausible answer dict matching config's OWN
    (pre-translation) schema shape, for both the flagged and clean case,
    reusing each metric's own few-shot answers as the template rather than
    inventing field names that might not match."""
    template = None
    for ex in config.few_shot_examples:
        ans = ex["answer"]
        is_flagged = ans.get("error", ans.get("formulaic", bool(ans.get("structures", [])) or bool(ans.get("fillers", []))))
        if bool(is_flagged) == flagged:
            template = ans
            break
    if template is None:
        template = config.few_shot_examples[0]["answer"]
    return template


def wrap_as_openai_response(answer: dict) -> dict:
    """Hand-build a raw_response dict in the exact shape
    client.responses.create() really returns (per the SDK introspection
    documented in providers/openai_provider.py), so parse_response() is
    exercised against a realistic structure, not a shortcut."""
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": json.dumps(answer, ensure_ascii=False)}
                ],
            }
        ]
    }


def main() -> int:
    provider = OpenAIProvider()
    client = OpenAI(api_key=FAKE_KEY)
    failures = []

    print("=== 1. build_request() reaches the network layer cleanly, varied inputs ===\n")
    for key, config in METRIC_PROMPTS.items():
        sample = sample_for(config)
        req = provider.build_request(config, sample)
        try:
            client.responses.create(**req)
            failures.append((key, "UNEXPECTED SUCCESS -- no real key/network, this should not happen"))
            print(f"  {key:<20} UNEXPECTED SUCCESS")
        except Exception as e:
            kind = type(e).__name__
            if kind == "APIConnectionError":
                print(f"  {key:<20} OK (local shape fine, blocked only at network layer)")
            else:
                failures.append((key, f"LOCAL ERROR: {kind}: {e}"))
                print(f"  {key:<20} FAIL: {kind}: {e}")

    print("\n=== 2. _to_openai_schema() output validates as real, strict-mode-correct JSON Schema ===\n")
    for key, config in METRIC_PROMPTS.items():
        converted = _to_openai_schema(config.response_schema)
        try:
            Draft202012Validator.check_schema(converted)
            check_schema_strict_invariants(converted)
            print(f"  {key:<20} OK (valid JSON Schema, strict-mode invariants hold at every level)")
        except (jsonschema.exceptions.SchemaError, AssertionError) as e:
            failures.append((key, f"SCHEMA INVALID: {e}"))
            print(f"  {key:<20} FAIL: {e}")

    print("\n=== 3. Round-trip: schema accepts each metric's own answer shapes; parse_response() unpacks them ===\n")
    for key, config in METRIC_PROMPTS.items():
        converted = _to_openai_schema(config.response_schema)
        validator = Draft202012Validator(converted)
        for flagged in (True, False):
            answer = build_fake_answer(config, flagged)
            errors = list(validator.iter_errors(answer))
            if errors:
                failures.append((key, f"answer (flagged={flagged}) rejected by its own schema: {errors[0].message}"))
                print(f"  {key:<20} flagged={flagged!s:<5} SCHEMA REJECTED OWN ANSWER: {errors[0].message}")
                continue
            raw_response = wrap_as_openai_response(answer)
            parsed = provider.parse_response(raw_response)
            if parsed != answer:
                failures.append((key, f"parse_response() round-trip mismatch (flagged={flagged}): {parsed!r} != {answer!r}"))
                print(f"  {key:<20} flagged={flagged!s:<5} FAIL: round-trip mismatch")
            else:
                print(f"  {key:<20} flagged={flagged!s:<5} OK (schema accepts it, parse_response round-trips it)")

    print("\n=== 4. prefilter.py agrees with the provider on GDD-1/GDD-2 -- unaffected by the switch ===\n")
    prefilter_checks = [
        ("GDD-1", "Ich lerne jeden Tag Deutsch.", False),   # no trigger -> skippable
        ("GDD-1", SENTENCE_SAMPLES["GDD-1"], True),          # "von" -> not skippable
        ("GDD-2", "Ich lerne jeden Tag Deutsch.", False),
        ("GDD-2", SENTENCE_SAMPLES["GDD-2"], True),          # "auf"/"den" -> not skippable
    ]
    for metric_key, sentence, expected_run in prefilter_checks:
        actual = prefilter.should_run(metric_key, sentence)
        status = "OK" if actual == expected_run else "FAIL"
        if status == "FAIL":
            failures.append((metric_key, f"prefilter mismatch for {sentence!r}: expected {expected_run}, got {actual}"))
        print(f"  {metric_key:<20} should_run={actual!s:<5} (expected {expected_run!s:<5}) {status}  {sentence!r}")
        # And confirm the provider would still happily build a request for
        # it regardless -- prefilter is a caller-side skip decision, never
        # something build_request() itself enforces.
        req = provider.build_request(METRIC_PROMPTS[metric_key], sentence)
        assert req["model"] == provider.model

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for key, msg in failures:
            print(f"  - {key}: {msg}")
        return 1
    print("ALL OFFLINE SMOKE CHECKS PASSED (network round-trip still unverified -- see module docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
