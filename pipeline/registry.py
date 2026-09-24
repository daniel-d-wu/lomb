"""
Registry for every fluenceme: LLM ones auto-discovered from prompts/*.py
(each file's CONFIG), direct ones from direct_computation.DIRECT_FLUENCEMES.
As of 2026-09-24 this is the single source of truth the pipeline, storage
taxonomy and Report 1 all read from -- adding an LLM fluenceme is adding one
prompts/<name>.py file, nothing else.

This file now holds ONLY provider-agnostic content -- the lookup table
mapping metric key -> (instruction text, few-shot examples, output
schema). It knows nothing about Gemini, OpenAI, or any other provider's
specific request format. That translation now lives in providers/ (see
llm_provider.py for the interface every provider adapter implements).

This split is what makes switching providers cheap: none of this file, and
none of the 7 files in prompts/, need to change if you swap
GeminiProvider for OpenAIProvider later. Only a new file in providers/
would be needed.

2026-09-05: FORMULAIC removed from this registry entirely, per Dan's
explicit instruction to make it a deterministic regex metric instead of an
LLM-assisted one. It's no longer imported here and has no CONFIG in
prompts/formulaic.py anymore (that file now holds only the BUNDLES
reference list -- see its own docstring). pipeline.py still computes and
reports a FORMULAIC result, via speaker_filter.find_formulaic_matches()
directly, with zero calls to classify()/this registry -- see pipeline.py's
module docstring and its DIRECT_METRICS group for that path.

2026-09-06: UNFILLED_PAUSE and FILLED_PAUSE removed from this registry too,
same treatment -- both are now direct computations
(direct_computation.compute_unfilled_pause() /
compute_filled_pause()), with no CONFIG left in prompts/unfilled_pause.py
or prompts/filled_pause.py (each now holds only its reference constant --
see those files' own docstrings). This dropped the metric count this file
is responsible for from 9 to 7; the assertion below was updated to match,
not silently loosened.
"""

import sys
from pathlib import Path

# repo root on sys.path -- prompts/ is a sibling of this file's own new
# directory (pipeline/) post-reorg, not something Python puts on the path
# for us automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
import pkgutil

import prompts
from pipeline.direct_computation import DIRECT_FLUENCEMES
from pipeline.metric_types import MetricPromptConfig

VALID_CONSTRUCTS = {"ACCURACY", "COMPLEXITY", "FLUENCY"}


def _discover_llm_fluencemes() -> dict[str, MetricPromptConfig]:
    """Every prompts/<name>.py that defines CONFIG = MetricPromptConfig(...)
    is an LLM fluenceme -- no list to maintain here. Files without a CONFIG
    (formulaic.py, filled_pause.py, unfilled_pause.py hold reference data
    for direct fluencemes) are skipped."""
    found: dict[str, MetricPromptConfig] = {}
    for module_info in pkgutil.iter_modules(prompts.__path__):
        config = getattr(importlib.import_module(f"prompts.{module_info.name}"), "CONFIG", None)
        if not isinstance(config, MetricPromptConfig):
            continue
        if config.key in found:
            raise ValueError(f"two prompt files both declare fluenceme key {config.key!r}")
        found[config.key] = config
    return dict(sorted(found.items()))


METRIC_PROMPTS: dict[str, MetricPromptConfig] = _discover_llm_fluencemes()

# Structural checks (not hardcoded counts): a new fluenceme must not reuse
# a key, a storage metric_key, or name an unknown construct.
_overlap = set(METRIC_PROMPTS) & set(DIRECT_FLUENCEMES)
assert not _overlap, f"fluenceme key(s) registered as both LLM and direct: {sorted(_overlap)}"
_all_specs = list(METRIC_PROMPTS.values()) + list(DIRECT_FLUENCEMES.values())
_metric_keys = [s.metric_key for s in _all_specs]
assert len(_metric_keys) == len(set(_metric_keys)), f"duplicate metric_key across fluencemes: {_metric_keys}"
for _spec in _all_specs:
    assert _spec.construct in VALID_CONSTRUCTS, f"{_spec.key}: unknown construct {_spec.construct!r}"

# Report 1's accuracy.errors[] metrics, in display order -- derived from
# each prompt file's report1_tag, not a separate list in report.py.
REPORT1_ERROR_METRICS: list[str] = [
    c.key for c in sorted(METRIC_PROMPTS.values(), key=lambda c: (c.report1_order, c.key))
    if c.report1_tag is not None
]


def taxonomy_rows() -> list[tuple[str, str, str, str, str, str]]:
    """(feature_key, construct_key, computation_path, metric_key, unit, formula)
    for every fluenceme -- what storage seeds its taxonomy tables from."""
    rows = [(c.key, c.construct, "llm", c.metric_key, c.unit, c.formula) for c in METRIC_PROMPTS.values()]
    rows += [(d.key, d.construct, "direct", d.metric_key, d.unit, d.formula) for d in DIRECT_FLUENCEMES.values()]
    return rows


def classify(provider, metric_key: str, input_data):
    """The one function the rest of the system should actually call.

    `provider` is any llm_provider.LLMProvider instance (GeminiProvider(),
    OpenAIProvider(), whatever gets added later). Swapping providers is
    passing a different object here -- nothing else about this call changes.
    """
    config = METRIC_PROMPTS[metric_key]  # KeyError on a typo is the point
    return provider.classify(config, input_data)


if __name__ == "__main__":
    import json

    from providers.gemini_provider import GeminiProvider
    from providers.openai_provider import OpenAIProvider

    print(f"Loaded {len(METRIC_PROMPTS)} metric configs: {list(METRIC_PROMPTS.keys())}\n")

    sample_inputs = {
        "sentence": "Ich lebe in die USA.",
        # word_timestamps/audio_turn samples removed 2026-09-06: no
        # remaining METRIC_PROMPTS entry uses either input_kind anymore
        # (UNFILLED_PAUSE/FILLED_PAUSE moved to direct_computation.py;
        # FORMULAIC moved off this registry entirely on 2026-09-05).
    }

    # 2026-09-03: OpenAI is now the primary/production provider (Dan's
    # decision -- see providers/openai_provider.py's module docstring for
    # the full rationale and what was verified). Gemini stays fully
    # supported and demonstrated here too -- switching which one is
    # primary is exactly the "one new file in providers/" seam this
    # portability design was built for, not a rewrite of this file.
    openai_ = OpenAIProvider()
    gemini = GeminiProvider()

    # Prove build_request() works for every LLM fluenceme, on BOTH providers,
    # from the exact same METRIC_PROMPTS content -- this is the actual
    # portability claim, demonstrated rather than just asserted.
    for key, config in METRIC_PROMPTS.items():
        sample = sample_inputs[config.input_kind]
        openai_req = openai_.build_request(config, sample)
        gemini_req = gemini.build_request(config, sample)
        # openai_req's shape: 2026-09-03 rewrite targets the Responses API,
        # not Chat Completions -- system_instruction now lives in the
        # top-level "instructions" string param, not messages[0].content.
        # See providers/openai_provider.py's module docstring (finding 1)
        # for why that's a different request shape, not just a rename.
        assert openai_req["instructions"] == config.system_instruction
        # gemini_req's shape: system_instruction lives under "config" (a
        # plain str), not a top-level {"parts":...} dict. See
        # providers/gemini_provider.py's module docstring for why (the old
        # shape didn't match the real google-genai SDK's generate_content()
        # signature and would have failed on a live call).
        assert gemini_req["config"]["system_instruction"] == config.system_instruction
        print(f"OK  {key:<20} input_kind={config.input_kind:<16} "
              f"openai_req_keys={list(openai_req.keys())}  gemini_req_keys={list(gemini_req.keys())}")

    print("\nSame metric (GDD-2), same input, two different provider request shapes:\n")
    print("--- OpenAI (primary) ---")
    print(json.dumps(openai_.build_request(METRIC_PROMPTS["GDD-2"], "Das Bild haengt an die Wand."),
                      indent=2, ensure_ascii=False)[:600] + "\n...(truncated)...")
    print("\n--- Gemini ---")
    print(json.dumps(gemini.build_request(METRIC_PROMPTS["GDD-2"], "Das Bild haengt an die Wand."),
                      indent=2, ensure_ascii=False)[:600] + "\n...(truncated)...")
