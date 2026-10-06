"""
Registry for every fluenceme -- the single source of truth the pipeline,
storage taxonomy and Report 1 all read from. Three kinds:

  ERROR_TAGS      prompts/*.py whose CONFIG is an ErrorTagConfig (GDD-1..6,
                  GVT-1, GVT-2, LP, LPF), plus OTHER_ERROR below. Not LLM
                  calls of their own: the error finder finds every error,
                  the error sorter gives each one of these tags
                  (pipeline/error_finder.py, pipeline/error_sorter.py).
  LLM_LABELERS    prompts/*.py whose CONFIG is a MetricPromptConfig
                  (STRUCTURE_BREADTH): one LLM call per chunk that labels
                  every sentence (pipeline/batching.py).
  DIRECT_FLUENCEMES  computed in plain Python, no LLM
                  (pipeline/direct_computation.py).

Adding an error fluenceme = adding one prompts/<name>.py with an
ErrorTagConfig. It is discovered here, offered to the sorter, seeded into
the storage taxonomy and (if it sets report1_tag) shown in Report 1 --
nothing else to edit, and no new LLM call.

Provider-agnostic: nothing here knows which LLM answers (see providers/).

History: until 2026-09-24 every error fluenceme was its own narrow
detector prompt ("one call per fluenceme per chunk"). That design left
errors between the narrow scopes unreported and was removed after a
side-by-side comparison (docs/lomb_model_comparison_v1.md; old prompts in
git history).
"""

import importlib
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts
from pipeline.direct_computation import DIRECT_FLUENCEMES
from pipeline.metric_types import ErrorTagConfig, MetricPromptConfig

VALID_CONSTRUCTS = {"ACCURACY", "COMPLEXITY", "FLUENCY"}

# The catch-all tag: a real error the finder found that fits no error
# fluenceme. Stored and counted (so the uncovered error types are visible
# and can become the next fluenceme), never shown in Report 1.
OTHER_ERROR_KEY = "OTHER_ERROR"
OTHER_ERROR = ErrorTagConfig(
    key=OTHER_ERROR_KEY,
    definition=(
        "A real error that fits none of the categories above -- e.g. a wrongly formed verb (wrong participle, "
        "infinitive instead of a conjugated verb, 'zu aufnehmen' for 'aufzunehmen'), wrong verb ending or "
        "subject-verb agreement, wrong auxiliary (haben/sein), a missing or extra word, denn/dann."
    ),
    examples=[
        {"sentence": "ich habe die Sound wechseln zum Sh", "corrected": "ich habe die Sound gewechselt zum Sh"},
        {"sentence": "diese, diese Gespräch zu aufnehmen", "corrected": "diese, diese Gespräch aufzunehmen"},
    ],
    metric_key="other_errors",
    formula="count of real errors that fit no error fluenceme this session (not shown in Report 1)",
    report1_tag=None,
    report1_order=1000,
)


def _discover() -> tuple[dict[str, ErrorTagConfig], dict[str, MetricPromptConfig]]:
    """Every prompts/<name>.py that defines CONFIG. Files without a CONFIG
    (formulaic.py, filled_pause.py, unfilled_pause.py hold reference data
    for direct fluencemes) are skipped."""
    tags: dict[str, ErrorTagConfig] = {}
    labelers: dict[str, MetricPromptConfig] = {}
    for module_info in pkgutil.iter_modules(prompts.__path__):
        config = getattr(importlib.import_module(f"prompts.{module_info.name}"), "CONFIG", None)
        if isinstance(config, ErrorTagConfig):
            target = tags
        elif isinstance(config, MetricPromptConfig):
            target = labelers
        else:
            continue
        if config.key in tags or config.key in labelers:
            raise ValueError(f"two prompt files both declare fluenceme key {config.key!r}")
        target[config.key] = config
    return tags, labelers


_tags, _labelers = _discover()

# Sorted by Report 1 order, catch-all last: this is also the order the
# sorter sees the tags in.
ERROR_TAGS: dict[str, ErrorTagConfig] = {
    c.key: c for c in sorted(_tags.values(), key=lambda c: (c.report1_order, c.key))
}
ERROR_TAGS[OTHER_ERROR_KEY] = OTHER_ERROR
LLM_LABELERS: dict[str, MetricPromptConfig] = dict(sorted(_labelers.items()))

# Structural checks (not hardcoded counts): a new fluenceme must not reuse
# a key, a storage metric_key, or name an unknown construct.
_all_specs = list(ERROR_TAGS.values()) + list(LLM_LABELERS.values()) + list(DIRECT_FLUENCEMES.values())
_keys = [s.key for s in _all_specs]
assert len(_keys) == len(set(_keys)), f"fluenceme key registered twice: {sorted(_keys)}"
_metric_keys = [s.metric_key for s in _all_specs]
assert len(_metric_keys) == len(set(_metric_keys)), f"duplicate metric_key across fluencemes: {_metric_keys}"
for _spec in _all_specs:
    assert _spec.construct in VALID_CONSTRUCTS, f"{_spec.key}: unknown construct {_spec.construct!r}"
for _tag in ERROR_TAGS.values():
    assert _tag.definition.strip(), f"{_tag.key}: an error tag needs a definition for the sorter"

# Report 1's accuracy.errors[] metrics, in display order -- every error tag
# with a report1_tag (OTHER_ERROR has none).
REPORT1_ERROR_METRICS: list[str] = [k for k, c in ERROR_TAGS.items() if c.report1_tag is not None]


def taxonomy_rows() -> list[tuple[str, str, str, str, str, str]]:
    """(feature_key, construct_key, computation_path, metric_key, unit, formula)
    for every fluenceme -- what storage seeds its taxonomy tables from."""
    rows = [(c.key, c.construct, "llm_find_sort", c.metric_key, c.unit, c.formula) for c in ERROR_TAGS.values()]
    rows += [(c.key, c.construct, "llm", c.metric_key, c.unit, c.formula) for c in LLM_LABELERS.values()]
    rows += [(d.key, d.construct, "direct", d.metric_key, d.unit, d.formula) for d in DIRECT_FLUENCEMES.values()]
    return rows


if __name__ == "__main__":
    print(f"Error tags ({len(ERROR_TAGS)}): {list(ERROR_TAGS)}")
    print(f"LLM labelers ({len(LLM_LABELERS)}): {list(LLM_LABELERS)}")
    print(f"Direct fluencemes ({len(DIRECT_FLUENCEMES)}): {list(DIRECT_FLUENCEMES)}")
    print(f"Report 1 error metrics, in order: {REPORT1_ERROR_METRICS}")
    assert list(ERROR_TAGS)[-1] == OTHER_ERROR_KEY, "the catch-all must be offered to the sorter last"
    assert OTHER_ERROR_KEY not in REPORT1_ERROR_METRICS
    print("\nRegistry structural checks passed.")
