"""
One LLM call labels a whole list of sentences, instead of one call per
sentence. Used by labeler fluencemes (prompts/*.py with a
MetricPromptConfig -- today STRUCTURE_BREADTH).

Wraps a per-sentence MetricPromptConfig into a batch version: the input is
the target speaker's ordered, numbered sentence list, the output is one
result per sentence. The per-sentence instruction text and few-shot
examples are reused verbatim, so every provider adapter keeps working with
no changes -- build_request() only ever reads config fields.

Why (2026-09-24): the per-sentence loop made 327 calls on a 5-min session
and ~98% of every request was the same instruction + few-shot text resent
each time. Error detection no longer uses this at all -- see
pipeline/error_finder.py and pipeline/error_sorter.py.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metric_types import MetricPromptConfig

BATCH_INSTRUCTION = """

BATCH MODE: the input is a JSON array of the learner's sentences, in the order they were spoken, each with an "index". Apply the task above to EVERY sentence and return exactly one entry per sentence in "results", copying its "index" from the input. Judge each sentence by the rules above; the neighboring sentences are context only."""


def _batch_schema(item_schema: dict) -> dict:
    item = dict(item_schema)
    item["properties"] = {"index": {"type": "INTEGER"}, **item_schema["properties"]}
    item["required"] = ["index", *item_schema.get("required", [])]
    return {
        "type": "OBJECT",
        "properties": {"results": {"type": "ARRAY", "items": item}},
        "required": ["results"],
    }


def batch_input(indexed_sentences: list[tuple[int, str]]) -> str:
    return json.dumps([{"index": i, "text": s} for i, s in indexed_sentences], ensure_ascii=False)


def build_batch_config(config: MetricPromptConfig) -> MetricPromptConfig:
    examples = list(enumerate(config.few_shot_examples))
    combined_example = {
        "input": batch_input([(i, ex["input"]) for i, ex in examples]),
        "answer": {"results": [{"index": i, **ex["answer"]} for i, ex in examples]},
    }
    return replace(
        config,
        system_instruction=config.system_instruction + BATCH_INSTRUCTION,
        few_shot_examples=[combined_example],
        response_schema=_batch_schema(config.response_schema),
        input_kind="sentence_batch",
    )


def classify_batch(provider, batch_config: MetricPromptConfig,
                   indexed_sentences: list[tuple[int, str]]) -> dict[int, dict]:
    """One provider call for all of indexed_sentences. Returns
    {sentence_index: per-sentence output dict}; each output also carries
    sentence_indices=[index] so storage can record which sentence it's
    about. Indices the model omitted are simply absent -- the caller
    decides how to surface them."""
    raw = provider.classify(batch_config, batch_input(indexed_sentences))
    requested = {i for i, _ in indexed_sentences}
    outputs: dict[int, dict] = {}
    for item in raw.get("results", []):
        item = dict(item)
        idx = item.pop("index", None)
        if idx in requested and idx not in outputs:
            item["sentence_indices"] = [idx]
            outputs[idx] = item
    return outputs
