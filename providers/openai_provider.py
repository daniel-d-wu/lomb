"""
OpenAI adapter -- now the primary, verified provider (switched from Gemini
2026-09-03, per Dan's decision: "we are going to buy the openai api
solution"). Gemini's adapter (providers/gemini_provider.py) stays in this
package, tested and working -- it's just no longer the default the rest of
this project points at.

2026-09-03: build_request()/call()/parse_response() were rewritten and
checked against the ACTUAL installed openai SDK (pip install openai,
version 3.7.0), the same method used to verify gemini_provider.py: real
introspection of the SDK's own type definitions, since this environment
cannot reach OpenAI's API at all either (api.openai.com:443 gets a 403
CONNECT rejection from this sandbox's egress proxy, confirmed by testing --
the exact same failure mode generativelanguage.googleapis.com had). This
replaces the previous "unverified reference implementation" version, whose
call() didn't even attempt a real request (raised NotImplementedError) and
which had a real bug that would have failed the first time anyone actually
ran it.

Real findings from this pass, most significant first:

1. OpenAI's own current docs say plainly: "While Chat Completions remains
   supported, Responses is recommended for all new projects." The previous
   version of this file targeted Chat Completions (client.chat.completions
   .create, {"response_format": {"json_schema": ...}},
   choices[0].message.content) -- a different request/response shape from
   the Responses API this version now uses (client.responses.create,
   {"text": {"format": ...}}, output[].content[].text). Confirmed against
   the real SDK: client.responses.create's signature (checked via
   inspect.signature on a real client instance) has model, input,
   instructions, text, temperature, top_p, reasoning -- and NO chat-style
   "messages" parameter and NO "seed" parameter at all (Chat Completions
   had both). Determinism here rests on temperature=0 (whether this model
   family actually honors that is itself unverified -- see finding 3) plus
   reasoning.effort="none", not a seed.

2. The response_schema every prompts/<metric>.py file defines was written
   in Gemini's schema dialect -- uppercase type keywords ("OBJECT",
   "STRING", "BOOLEAN", "ARRAY", "NUMBER") -- and the previous version of
   this file passed that dict to OpenAI completely unmodified
   ("schema": config.response_schema). OpenAI's structured-output schema
   is standard JSON Schema: lowercase type keywords, "additionalProperties":
   false required on every object, and every key listed in "properties"
   must also appear in "required" under strict mode (confirmed against
   OpenAI's structured-outputs guide, cross-checked against the actual
   TypedDict fields on ResponseFormatTextJSONSchemaConfigParam). Passing
   the Gemini-dialect schema as-is would have been rejected by OpenAI's API
   the first time this was actually called -- never caught before because
   call() raised NotImplementedError and nothing ever exercised this path.
   _to_openai_schema() below does the translation. One real wrinkle it had
   to handle: prompts/unfilled_pause.py's nested per-boundary schema has
   "reasoning" as a property that's genuinely NOT in that level's
   "required" list, by that file's own design. OpenAI's strict mode has no
   notion of an optional property, so _to_openai_schema() makes it
   required but widens its type to ["string", "null"] so the model can
   still supply null instead of a value -- rather than silently forcing
   every "reasoning" to be non-empty (misrepresenting the schema's own
   intent) or leaving it out of "required" (which OpenAI's API would
   reject outright). Checked by grep across every prompts/*.py file: this
   is the only optional property among all 10 registered metrics' schemas,
   not assumed to be the only one.

3. gpt-5.6-luna (the model chosen below -- see its own comment) is a
   reasoning model (confirmed via its own page at developers.openai.com:
   it exposes reasoning.effort, including a "none" option documented there
   for "latency-critical tasks that do not benefit from any reasoning").
   Whether the API actually honors temperature/top_p on a reasoning model
   versus silently ignoring them was flagged here as unverified -- OpenAI's
   own reasoning-models guide doesn't say either way. PARTIALLY CLOSED
   2026-09-03: a real live call (Dan, on his own machine, via
   test_openai_live.py) succeeded with this exact request shape --
   temperature=0/top_p=0.1/reasoning.effort="none" together -- and returned
   a correct, real answer for GDD-2 on "Das Bild haengt an die Wand."
   (error=true, with reasoning identifying the dative requirement and
   correctly using "hängt" with a real umlaut in its own output text, not
   the ASCII "haengt" the prompt used). That proves the API accepts this
   parameter combination without error; it does NOT prove temperature=0
   is actually constraining sample variance the way it would on a
   non-reasoning model (that would need multiple live calls on the same
   input compared against each other, not yet done) -- so treat "accepted"
   as confirmed and "actually reduces variance" as still open.

4. Real-umlaut output, unprompted: the live call above returned "hängt"
   (real umlaut) even though every few-shot example and the system
   instruction text this file sends are ASCII-transliterated ("haengt",
   per this codebase's own convention -- see prefilter.py's docstring for
   the fuller umlaut-vs-ASCII discussion). The model did not mirror the
   prompt's ASCII spelling back -- it produced standard German orthography
   on its own. One live call is not proof this is reliable behavior across
   all 10 metrics, but it's a data point in favor of prefilter.py's
   defensive choice to match both spellings rather than assume ASCII.

5. call()'s response handling: Response.output_text (the SDK's own
   convenience accessor for the generated text) is a plain Python
   @property, not a pydantic computed field (confirmed: Response.
   model_computed_fields is empty) -- so it does NOT survive
   response.model_dump(), the same serialize-to-plain-dict step
   gemini_provider.py's call() uses. parse_response() below reconstructs
   the same logic output_text's own source performs (walk output[] for
   type=="message", then that message's content[] for
   type=="output_text", concatenate .text) directly against the dict
   call() returns, instead of relying on the property -- keeping the same
   "call() returns a plain serialized dict, parse_response() reads from
   it" shape gemini_provider.py already established.

CONFIRMED WORKING END TO END, live, 2026-09-03 (Dan, on his own machine --
this sandbox still can't reach api.openai.com at all, key or no key,
confirmed by testing with a real key here too):

- test_openai_live.py's GDD-2 / "Das Bild haengt an die Wand." case
  returned a real response, parsed cleanly by parse_response(), with the
  correct answer (error=true).
- test_all_metrics_live.py then ran all 9 text-based metrics (everything
  but FILLED_PAUSE, excluded by that script on purpose -- see its own
  docstring) against gpt-5.6-luna, live: GDD-1, GDD-2, GVT-1, GVT-2, LPF,
  LP, FORMULAIC, and STRUCTURE_BREADTH all matched their expected
  boolean/label answer; UNFILLED_PAUSE's list-shaped answer
  (['trustworthy', 'suspect']) matched exactly, confirming the nested
  per-boundary schema fix from finding 2 (the "reasoning": None few-shot
  fix in prompts/unfilled_pause.py) didn't just satisfy the schema
  validator offline -- it round-tripped through a real model correctly.
  9/9.

That closes the gap this docstring used to flag as unverified -- an
actual network round-trip -- for every text-based metric this codebase
currently implements. What's still open: FILLED_PAUSE specifically still
needs audio-part construction this codebase doesn't implement, live call
or not; and see finding 3 above for the narrower claim that IS still open
even after this success (params accepted on every call made, but "does
temperature=0 actually reduce variance the way it would on a
non-reasoning model" would need repeated calls on the same input compared
against each other, not yet done).
"""

import json
import os
from typing import Any

from llm_provider import LLMProvider
from metric_types import MetricPromptConfig

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 0.1,
    # See finding 3 above: this is the parameter actually documented as
    # the right lever for a task this small, not the sampling params.
    "reasoning": {"effort": "none"},
}

# Maps prompts/*.py's Gemini-dialect type keywords to standard JSON Schema
# ones. Anything not in this map is lowercased as a fallback rather than
# left alone, so a schema already written in lowercase JSON Schema still
# passes through unchanged instead of silently failing to match.
_GEMINI_TO_JSON_SCHEMA_TYPE = {
    "OBJECT": "object",
    "STRING": "string",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "NUMBER": "number",
    "INTEGER": "integer",
}


def _to_openai_schema(schema: dict) -> dict:
    """Translate one of prompts/*.py's Gemini-dialect response schemas into
    OpenAI strict-mode JSON Schema. Recursive -- every nested object and
    array gets the same treatment. See finding 2 in this module's
    docstring for what this has to get right and why.

    Never mutates its input: every dict encountered is shallow-copied
    before being changed, so the shared MetricPromptConfig.response_schema
    objects in registry.py -- which GeminiProvider also reads, unmodified
    -- are never touched by this.
    """

    def convert(node):
        if not isinstance(node, dict):
            return node
        node = dict(node)

        raw_type = node.get("type")
        if isinstance(raw_type, str):
            node["type"] = _GEMINI_TO_JSON_SCHEMA_TYPE.get(raw_type.upper(), raw_type.lower())

        if "items" in node:
            node["items"] = convert(node["items"])

        if "properties" in node:
            original_required = set(node.get("required", []))
            converted_properties = {}
            for key, prop_schema in node["properties"].items():
                converted = convert(prop_schema)
                if key not in original_required:
                    # OpenAI strict mode has no "optional" property -- make
                    # it required but nullable instead, so the model can
                    # supply null in place of a value. Preserves the
                    # original schema's actual intent rather than forcing
                    # every value to be present (which it wasn't meant to
                    # be) or leaving it out of "required" (which OpenAI's
                    # API rejects outright in strict mode).
                    prop_type = converted.get("type")
                    if isinstance(prop_type, str) and prop_type != "null":
                        converted["type"] = [prop_type, "null"]
                converted_properties[key] = converted
            node["properties"] = converted_properties
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False

        return node

    return convert(schema)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        # 2026-09-03, chosen after checking OpenAI's live pricing page
        # (openai.com/api/pricing) and cross-checking the exact API id
        # against developers.openai.com's own model docs: the GPT-5.6
        # family is the current lineup -- Sol ($5.00/$30.00 per 1M tokens
        # in/out), Terra ($2.00/$12.00), Luna ($0.20/$1.20). Luna is
        # described on OpenAI's own pricing page as "Fast, affordable model
        # for everyday work" -- the same "small task, cheap tier"
        # reasoning that picked gemini-3.1-flash-lite earlier (one short
        # sentence in, three small fields out). Real API id confirmed via
        # developers.openai.com/api/docs/models: "gpt-5.6-luna" (the
        # display name "GPT-5.6 Luna" is NOT the string the API expects).
        # NOT yet confirmed against a live call at this specific model name
        # -- this environment can't reach api.openai.com at all (see
        # module docstring) -- that's what test_openai_live.py is for,
        # once a real key is available.
        model: str = "gpt-5.6-luna",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model = model
        self.api_key_env = api_key_env

    def build_request(self, config: MetricPromptConfig, input_data: Any) -> dict:
        input_text = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)

        # Few-shot turns only -- the system-level instruction goes through
        # the Responses API's own top-level "instructions" param instead of
        # a system-role message mixed into "input" (confirmed as the
        # idiomatic split for this API, distinct from Chat Completions'
        # single flat messages list).
        turns = []
        for ex in config.few_shot_examples:
            turns.append({"role": "user", "content": ex["input"]})
            turns.append({"role": "assistant", "content": json.dumps(ex["answer"], ensure_ascii=False)})
        turns.append({"role": "user", "content": input_text})

        generation_config = {**DEFAULT_GENERATION_CONFIG, **config.generation_config_overrides}

        return {
            "model": self.model,
            "instructions": config.system_instruction,
            "input": turns,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"{config.key.lower().replace('-', '_')}_result",
                    "schema": _to_openai_schema(config.response_schema),
                    "strict": True,
                },
            },
            **generation_config,
        }

    def call(self, request: dict) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ[self.api_key_env])
        response = client.responses.create(**request)
        return response.model_dump()

    def parse_response(self, raw_response: dict) -> dict:
        # Reconstruct Response.output_text's own logic (see finding 4
        # above) -- that property doesn't survive model_dump(), so this
        # walks the same structure it would have read.
        text_parts = []
        for item in raw_response.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
        return json.loads("".join(text_parts))
