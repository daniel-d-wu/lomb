"""
FakeProvider -- NOT a stand-in for a real LLM. It makes no linguistic
judgment at all and must never be mistaken for one. Its only job is to
prove the orchestration is wired correctly with no network access or API
key: every call is counted per prompt key, and its fake answers are
deterministic so tests can assert exactly where each one ends up.

Shared by pipeline/pipeline.py's self-test and
scripts/run_end_to_end_whisperx_test.py (it used to be copied into both).

Fake behaviour, by prompt:
  ERROR_FINDER  one fake error in every `error_every`-th sentence (by index):
                said = its first word, fix = "FAKEFIX"; a clean sentence
                for each, except every 6th index (not understandable);
                every 9th index's error is marked a possible transcription
                error; with bad_said=True `said` is never in the sentence
  ERROR_SORTER  tags cycle through every error tag in registry order; with
                leave_untagged=True the last error of each call gets no tag
  labelers      STRUCTURE_BREADTH gets a repeating, overlapping label cycle
                (so breadth de-duplication is testable); any other labeler
                gets a generic "no error" answer
"""

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.error_finder import FINDER_KEY
from pipeline.error_sorter import SORTER_KEY
from pipeline.registry import ERROR_TAGS

_STRUCTURE_CYCLE = [["konjunktiv_ii"], ["dass_clause"], ["konjunktiv_ii"], ["none"], ["weil_clause"], ["dass_clause"]]
NOTE = "FakeProvider -- orchestration test only, not a real judgment."


class FakeProvider:
    model = "FakeProvider"
    generation_config: dict = {}

    def __init__(self, *, error_every: int = 3, leave_untagged: bool = False, bad_said: bool = False):
        self.error_every = error_every
        self.leave_untagged = leave_untagged
        self.bad_said = bad_said  # point every error at words that aren't in the sentence
        self._lock = threading.Lock()  # calls arrive in parallel
        self.call_count = 0
        self.calls_by_key: dict[str, int] = {}

    def classify(self, config, input_data):
        with self._lock:
            self.call_count += 1
            self.calls_by_key[config.key] = self.calls_by_key.get(config.key, 0) + 1
        items = json.loads(input_data)
        if config.key == FINDER_KEY:
            hits = [it for it in items if it["index"] % self.error_every == 0]
            return {"errors": [self._fake_error(it) for it in hits],
                    "clean_sentences": [self._fake_clean(it) for it in hits]}
        if config.key == SORTER_KEY:
            tags = list(ERROR_TAGS)
            answers = [{"id": it["id"], "tag": tags[it["id"] % len(tags)]} for it in items]
            return {"tags": answers[:-1] if self.leave_untagged else answers}
        return {"results": [{"index": it["index"], **self._label(config.key, it)} for it in items]}

    def _fake_error(self, item: dict) -> dict:
        return {
            "index": item["index"],
            "said": "NOT-IN-SENTENCE" if self.bad_said else item["text"].split()[0],
            "reasoning": NOTE,
            "fix": "FAKEFIX",
            "confidence": "high",
            "self_corrected": False,
            "possible_transcription_error": item["index"] % 9 == 0 and item["index"] > 0,
        }

    @staticmethod
    def _fake_clean(item: dict) -> dict:
        # every 6th errored sentence is "not understandable" -> no clean sentence, no card
        if item["index"] % 6 == 0:
            return {"index": item["index"], "understandable": False, "clean": ""}
        return {"index": item["index"], "understandable": True,
                "clean": " ".join(["FAKEFIX"] + item["text"].split()[1:])}

    @staticmethod
    def _label(key: str, item: dict) -> dict:
        if key == "STRUCTURE_BREADTH":
            return {"structures": _STRUCTURE_CYCLE[item["index"] % len(_STRUCTURE_CYCLE)], "confidence": "high"}
        return {"error": False, "confidence": "high", "reasoning": NOTE, "corrected": item["text"]}
