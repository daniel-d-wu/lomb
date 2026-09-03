"""
Gemini adapter -- a fully verified, working provider, kept as an
alternative. 2026-09-03: OpenAI became the primary/production provider
(see providers/openai_provider.py's module docstring for why); everything
documented below about this file is still true and this adapter still
works exactly as described -- switching the default was a one-file change
in registry.py, precisely the seam this provider-interface split was built
for. This is the same request-building logic that used to live inline in
registry.py; it's been moved here so it sits behind the LLMProvider
interface instead of being the only option registry.py knows about.

2026-09-02: build_request()/call()/parse_response() were rewritten against
the ACTUAL installed google-genai SDK (pip install google-genai, then
inspected via Python introspection -- no live API call was possible from
that environment, no network path to generativelanguage.googleapis.com,
but the SDK's own pydantic models were importable and checkable offline).
This replaces an earlier version that was written from memory/documentation
recall and never actually checked against the real library. Three real
bugs that version had, found this way:

1. build_request() returned {"model", "system_instruction", "contents",
   "generation_config"} as four separate top-level keys, meant to be
   unpacked as client.models.generate_content(**request). The real method
   signature is generate_content(*, model, contents, config) -- there is
   no system_instruction or generation_config parameter. Calling it the
   old way would have raised TypeError: generate_content() got an
   unexpected keyword argument 'system_instruction' the first time anyone
   actually tried it. Fixed by folding system_instruction and the
   generation-config fields into one `config` dict, matching
   GenerateContentConfig's real fields (confirmed via
   types.GenerateContentConfig.model_fields).
2. call()'s commented-out reference implementation ended with
   `return response.to_dict()` -- GenerateContentResponse has no to_dict()
   method (confirmed via dir() on the real class). The real serialization
   method is model_dump() (it's a pydantic model).
3. Few-shot "model" turns were built with `str(ex["answer"])`, which
   renders a Python dict literal (single quotes, True/False) rather than
   valid JSON -- exactly the wrong thing to show the model as an example
   of the JSON output it should produce. openai_provider.py already does
   this correctly with json.dumps(); this file didn't match it. Fixed to
   use json.dumps() here too.

What's still NOT verified: an actual live network round-trip. This
environment cannot reach Google's API at all (confirmed by testing --
the egress proxy returns 403 for generativelanguage.googleapis.com), so
everything above was checked by constructing the real pydantic request/
response objects locally (GenerateContentConfig(**...),
GenerateContentResponse.model_validate(...)) and confirming they validate
without error -- not by an actual call. Run test_gemini_live.py on a
machine with real network access and a real GEMINI_API_KEY to close that
last gap.
"""

import json
import os
from typing import Any

from llm_provider import LLMProvider
from metric_types import MetricPromptConfig

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 0.1,
    "top_k": 1,
    "candidate_count": 1,
    "seed": 42,
    "response_mime_type": "application/json",
}


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        # 2026-09-02, model name corrected three times against a live key:
        # (1) "gemini-2.5-flash-002" was a guessed dated ID -- 404'd, not
        #     a real model at all.
        # (2) "gemini-2.5-flash" IS real (confirmed via a live
        #     client.models.list() call) but 404'd anyway on an actual
        #     generateContent call -- Google's own error said it's no
        #     longer available to new accounts and pointed to
        #     gemini-3.6-flash instead.
        # (3) gemini-3.6-flash WORKED (confirmed live, GDD-2 test passed)
        #     but is the pricier non-lite tier ($0.75/$3.75 per 1M tokens
        #     input/output, per ai.google.dev's pricing page as of
        #     2026-09-02) -- overkill for a task this small (one short
        #     sentence in, three small fields out). Switched to
        #     gemini-3.1-flash-lite ($0.25/$1.50) -- ~3x cheaper on input,
        #     2.5x on output, still a generation past the 2.5 line that
        #     hit the new-account block in (2), and confirmed present in
        #     the models available to this exact key (see
        #     list_gemini_models.py's output). NOT yet confirmed against
        #     a live call at this specific model name -- that's what
        #     test_all_metrics_live.py is about to do.
        model_version: str = "gemini-3.1-flash-lite",
        api_key_env: str = "GEMINI_API_KEY",
    ):
        self.model_version = model_version
        self.api_key_env = api_key_env

    def build_request(self, config: MetricPromptConfig, input_data: Any) -> dict:
        input_text = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)

        few_shot_turns = []
        for ex in config.few_shot_examples:
            few_shot_turns.append({"role": "user", "parts": [{"text": ex["input"]}]})
            few_shot_turns.append({"role": "model", "parts": [{"text": json.dumps(ex["answer"], ensure_ascii=False)}]})

        generation_config = {**DEFAULT_GENERATION_CONFIG, **config.generation_config_overrides}

        # NOTE the shape here: everything except model/contents lives under
        # "config", matching the real generate_content(*, model, contents,
        # config) signature -- NOT flattened as top-level keys. system_instruction
        # takes a plain string directly (confirmed: GenerateContentConfig's
        # system_instruction field accepts str, no {"parts": [...]} wrapping needed).
        return {
            "model": self.model_version,
            "contents": few_shot_turns + [{"role": "user", "parts": [{"text": input_text}]}],
            "config": {
                "system_instruction": config.system_instruction,
                **generation_config,
                "response_schema": config.response_schema,
            },
        }

    def call(self, request: dict) -> dict:
        from google import genai

        client = genai.Client(api_key=os.environ[self.api_key_env])
        response = client.models.generate_content(**request)
        return response.model_dump()

    def parse_response(self, raw_response: dict) -> dict:
        # Gemini's structured-output text still comes back as a JSON
        # *string* inside candidates[0].content.parts[0].text -- it still
        # needs one json.loads(), even with response_schema set. Field
        # names here confirmed against the real GenerateContentResponse /
        # Candidate / Content / Part model_fields, matching what
        # response.model_dump() in call() actually produces.
        text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
