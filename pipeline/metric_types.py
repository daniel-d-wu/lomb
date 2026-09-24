"""
Shared type for the metric prompt registry. Kept in its own file so both
registry.py and every prompts/<metric>.py module can import it without a
circular dependency (registry.py imports the prompts/ modules; if
MetricPromptConfig lived inside registry.py, those modules would have to
import it back from there, creating an import cycle).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class MetricPromptConfig:
    """One LLM fluenceme, fully described in its own prompts/<name>.py file.
    registry.py discovers every CONFIG automatically; the pipeline, storage
    taxonomy and Report 1 all read from here -- adding a fluenceme is adding
    one file, nothing else."""
    key: str
    system_instruction: str
    few_shot_examples: list[dict]
    response_schema: dict
    # --- how this fluenceme is stored and reported ---
    construct: str        # "ACCURACY" | "COMPLEXITY" | "FLUENCY"
    metric_key: str       # metric_definitions / metric_values key for the session-level value
    unit: str
    formula: str          # plain-English definition of the stored session-level value
    report1_tag: str | None = None  # label in Report 1's accuracy.errors[]; None = not shown there
    report1_order: int = 100        # position among Report 1 error metrics (lower = first)
    generation_config_overrides: dict = field(default_factory=dict)
    # Documents what shape of input this metric expects, since not all 10
    # take a plain sentence -- the pause metrics take timestamped word
    # lists, and filled-pause detection needs actual audio, not just text.
    # "sentence" | "word_timestamps" | "audio_turn"
    input_kind: str = "sentence"
