"""
GDD-4: Adjective Endings (error tag).

~105 of 618 real GDD errors (17%) in docs/all_grammar_errors_master.json.
Added 2026-09-24. Since find-then-sort (2026-09-24) this also covers
adjectives after prepositions -- the old detector skipped them.

Examples are verbatim real transcript errors from
all_grammar_errors_master.json; none from the benny_italki5/6 test sessions.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="GDD-4",
    definition=(
        "A wrong ADJECTIVE ENDING while the article (if any) is right: an attributive adjective without the "
        "strong/weak/mixed ending its noun phrase needs (e.g. 'ein richtig gut Punkt' -> 'guter'), or an "
        "ending on a predicate adjective, which takes none ('weil man dicke ist' -> 'dick')."
    ),
    not_this=(
        "The article is wrong and fixing the article alone makes the phrase right -> that article's tag "
        "(GDD-3 or a case tag)."
    ),
    examples=[
        {"sentence": "Das ist ein richtig gut Punkt", "corrected": "Das ist ein richtig guter Punkt"},
        {"sentence": "Ich weiß nur die wichtige Methoden, einen Satz zu, zu sprechen.",
         "corrected": "Ich weiß nur die wichtigen Methoden, einen Satz zu, zu sprechen."},
    ],
    metric_key="gdd4_errors",
    formula="count of errors tagged GDD-4 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Adjective ending",
    report1_order=8,
)
