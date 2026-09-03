"""
Shared type for the metric prompt registry. Kept in its own file so both
registry.py and every prompts/<metric>.py module can import it without a
circular dependency (registry.py imports the prompts/ modules; if
MetricPromptConfig lived inside registry.py, those modules would have to
import it back from there, creating an import cycle).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricPromptConfig:
    key: str
    system_instruction: str
    few_shot_examples: list[dict]
    response_schema: dict
    generation_config_overrides: dict = field(default_factory=dict)
    # Documents what shape of input this metric expects, since not all 10
    # take a plain sentence -- the pause metrics take timestamped word
    # lists, and filled-pause detection needs actual audio, not just text.
    # "sentence" | "word_timestamps" | "audio_turn"
    input_kind: str = "sentence"
