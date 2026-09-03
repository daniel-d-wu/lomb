"""
The seam that makes this portable across LLM providers.

Every provider adapter (Gemini today; OpenAI, Anthropic, or anything else
later) implements this same interface. Nothing outside this file and the
providers/ folder should ever need to know which provider is actually in
use -- the segmentation code, the aggregation code, the metrics module
all just call provider.classify(config, input_data) and get back the same
shape of answer no matter which provider answered.

Switching providers later means writing one new file in providers/, not
touching any of the 10 files in prompts/ and not touching registry.py.
"""

from abc import ABC, abstractmethod
from typing import Any

from metric_types import MetricPromptConfig


class LLMProvider(ABC):

    @abstractmethod
    def build_request(self, config: MetricPromptConfig, input_data: Any) -> dict:
        """Translate provider-agnostic content (instruction text, few-shot
        examples, output schema) into this provider's own native request
        format. This is the only place provider-specific field names
        (Gemini's `generation_config`, OpenAI's `response_format`, etc.)
        are allowed to appear."""
        raise NotImplementedError

    @abstractmethod
    def call(self, request: dict) -> dict:
        """Actually send the request to this provider's API and return its
        raw native response, unmodified."""
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, raw_response: dict) -> dict:
        """Translate this provider's native response back into the common
        schema-shaped dict every metric config declares (e.g.
        {"error": bool, "confidence": str, "reasoning": str}) -- so
        downstream code never has to know or care which provider answered."""
        raise NotImplementedError

    def classify(self, config: MetricPromptConfig, input_data: Any) -> dict:
        """Convenience: build -> call -> parse in one step. Adapters
        generally shouldn't need to override this -- override the three
        methods above instead."""
        request = self.build_request(config, input_data)
        raw_response = self.call(request)
        return self.parse_response(raw_response)
