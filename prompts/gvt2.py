"""
GVT-2: Verb Position (error tag).

Verb-final order in subordinate clauses and verb-second order in main
clauses -- the most frequent GVT pattern in the corpus.

Examples are real transcript errors from dan_error_analysis_master_v3.md
and all_grammar_errors_master.json.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="GVT-2",
    definition=(
        "A verb in the wrong POSITION: the conjugated verb not at the end of a subordinate clause (weil, "
        "dass, wenn, ob, als, relative and indirect-question clauses); the conjugated verb not in second "
        "position in a main clause, including after a fronted word ('dann wir sehen' -> 'dann sehen wir'); "
        "an infinitive or participle not at the end of its clause."
    ),
    not_this="Right position, wrongly formed verb -> OTHER_ERROR. Wrong tense -> GVT-1.",
    examples=[
        {"sentence": "ich denke, dass mein Hoerverstaendnis ist ziemlich okay.",
         "corrected": "ich denke, dass mein Hoerverstaendnis ziemlich okay ist."},
        {"sentence": "dann wir sehen, wie lange wir auf Deutsch reden.",
         "corrected": "dann sehen wir, wie lange wir auf Deutsch reden."},
        {"sentence": "Weil es ist fast Schlafzeit.", "corrected": "Weil es fast Schlafzeit ist."},
    ],
    metric_key="gvt2_errors",
    formula="count of errors tagged GVT-2 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Verb position",
    report1_order=4,
)
