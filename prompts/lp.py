"""
LP: Lexical Phrase -- wrong chunk, no English source (error tag).

Per dan_error_analysis_master_v3.md: "Wrong multi-word chunk with no
traceable L1 source. Distinct from LPF: LPF errors have a direct English
interference mechanism; LP errors arise from chunk gaps." Includes invented
words and wrong collocates.

Examples are verbatim real transcript errors from
all_grammar_errors_master.json / the corpus notes.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="LP",
    definition=(
        "An unnatural WORD CHOICE or collocation that English does not explain: a wrong collocate ('Übung "
        "machen' for 'üben', 'mit Milch vorbereitet' for 'zubereitet'), a similar-sounding German word mixed "
        "up, or a word that doesn't exist in German."
    ),
    not_this="The wording is copied from English (preposition, idiom, false friend) -> LPF.",
    examples=[
        {"sentence": "wie lange möchtest du Chinesisch Übung machen?",
         "corrected": "wie lange möchtest du Chinesisch üben?"},
        {"sentence": "es wird— es mit Milch vorbereitet", "corrected": "es wird— es mit Milch zubereitet"},
    ],
    metric_key="lp_errors",
    formula="count of errors tagged LP this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Word choice / collocation",
    report1_order=6,
)
