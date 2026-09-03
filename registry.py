"""
Registry for all 10 LLM-assisted metrics.

This file now holds ONLY provider-agnostic content -- the lookup table
mapping metric key -> (instruction text, few-shot examples, output
schema). It knows nothing about Gemini, OpenAI, or any other provider's
specific request format. That translation now lives in providers/ (see
llm_provider.py for the interface every provider adapter implements).

This split is what makes switching providers cheap: none of this file, and
none of the 10 files in prompts/, need to change if you swap
GeminiProvider for OpenAIProvider later. Only a new file in providers/
would be needed.
"""

from metric_types import MetricPromptConfig  # noqa: F401 (re-exported for convenience)

from prompts.gdd1 import CONFIG as GDD1_CONFIG
from prompts.gdd2 import CONFIG as GDD2_CONFIG
from prompts.gvt1 import CONFIG as GVT1_CONFIG
from prompts.gvt2 import CONFIG as GVT2_CONFIG
from prompts.lpf import CONFIG as LPF_CONFIG
from prompts.lp import CONFIG as LP_CONFIG
from prompts.unfilled_pause import CONFIG as UNFILLED_PAUSE_CONFIG
from prompts.filled_pause import CONFIG as FILLED_PAUSE_CONFIG
from prompts.formulaic import CONFIG as FORMULAIC_CONFIG
from prompts.structure_breadth import CONFIG as STRUCTURE_BREADTH_CONFIG

METRIC_PROMPTS: dict[str, MetricPromptConfig] = {
    "GDD-1": GDD1_CONFIG,
    "GDD-2": GDD2_CONFIG,
    "GVT-1": GVT1_CONFIG,
    "GVT-2": GVT2_CONFIG,
    "LPF": LPF_CONFIG,
    "LP": LP_CONFIG,
    "UNFILLED_PAUSE": UNFILLED_PAUSE_CONFIG,
    "FILLED_PAUSE": FILLED_PAUSE_CONFIG,
    "FORMULAIC": FORMULAIC_CONFIG,
    "STRUCTURE_BREADTH": STRUCTURE_BREADTH_CONFIG,
}

assert len(METRIC_PROMPTS) == 10, "expected exactly 10 LLM-assisted metrics"


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
        "word_timestamps": [
            {"word": "ich", "start": 1.20, "end": 1.35},
            {"word": "denke", "start": 1.38, "end": 1.62},
        ],
        "audio_turn": "[placeholder -- real call needs an attached audio part, see prompts/filled_pause.py]",
    }

    # 2026-09-03: OpenAI is now the primary/production provider (Dan's
    # decision -- see providers/openai_provider.py's module docstring for
    # the full rationale and what was verified). Gemini stays fully
    # supported and demonstrated here too -- switching which one is
    # primary is exactly the "one new file in providers/" seam this
    # portability design was built for, not a rewrite of this file.
    openai_ = OpenAIProvider()
    gemini = GeminiProvider()

    # Prove build_request() works for all 10 metrics, on BOTH providers,
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
