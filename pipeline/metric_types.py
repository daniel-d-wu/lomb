"""
Shared types for the fluenceme registry. Kept in their own file so both
registry.py and every prompts/<name>.py module can import them without a
circular dependency (registry.py imports the prompts/ modules).

Three kinds of LLM-side config (2026-09-24, find-then-sort redesign):

  PromptConfig       -- one prompt a provider can send (instruction,
                        few-shot, output schema). The error finder and the
                        error sorter are PromptConfigs; providers only ever
                        see this type.
  MetricPromptConfig -- a LABELER fluenceme: its own LLM call that labels
                        every sentence (STRUCTURE_BREADTH). A PromptConfig
                        plus how its result is stored and reported.
  ErrorTagConfig     -- an ERROR fluenceme: not its own call, but a tag the
                        sorter can give to an error the finder found. Holds
                        a definition and examples instead of a prompt.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class PromptConfig:
    """One LLM prompt, provider-agnostic. providers/*.py translate this
    into their own request format."""
    key: str
    system_instruction: str
    few_shot_examples: list[dict]
    response_schema: dict
    generation_config_overrides: dict = field(default_factory=dict)
    # What the input is: "sentence" (one sentence), "sentence_batch" (a
    # numbered sentence list) or "error_batch" (a list of found errors).
    input_kind: str = "sentence"


@dataclass(frozen=True, kw_only=True)
class MetricPromptConfig(PromptConfig):
    """A labeler fluenceme, fully described in its own prompts/<name>.py
    file (CONFIG = MetricPromptConfig(...)). registry.py discovers it; the
    pipeline gives it one call per chunk (pipeline/batching.py)."""
    construct: str        # "ACCURACY" | "COMPLEXITY" | "FLUENCY"
    metric_key: str       # metric_definitions / metric_values key for the session-level value
    unit: str
    formula: str          # plain-English definition of the stored session-level value
    report1_tag: str | None = None  # label in Report 1's accuracy.errors[]; None = not shown there
    report1_order: int = 100        # position among Report 1 error metrics (lower = first)


@dataclass(frozen=True, kw_only=True)
class ErrorTagConfig:
    """An error fluenceme, fully described in its own prompts/<name>.py file
    (CONFIG = ErrorTagConfig(...)). The error finder finds every error; the
    sorter gives each one exactly one tag from this list, or OTHER_ERROR.
    Adding an error fluenceme = adding one file with a definition and
    examples -- no new LLM call."""
    key: str
    definition: str       # what belongs under this tag, shown to the sorter
    not_this: str = ""    # nearby errors that belong under a different tag
    # Real examples, shown to the sorter: {"sentence": what was said,
    # "corrected": the sentence with only this error fixed}.
    examples: list[dict] = field(default_factory=list)
    construct: str = "ACCURACY"
    metric_key: str
    unit: str = "count"
    formula: str
    report1_tag: str | None = None  # label on Report 1 cards; None = stored and counted, never shown
    report1_order: int = 100        # position among Report 1 error metrics (lower = first)
