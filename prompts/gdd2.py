"""
GDD-2: Case Error -- Wechselpraepositionen (two-way prepositions) (error tag).

in / an / auf / ueber / unter / vor / neben / zwischen take dative for a
location and accusative for movement toward a destination.

2026-09-24 (find-then-sort): a tag definition for the sorter, not a
detector -- see prompts/gdd1.py's docstring.

Examples are real transcript errors from dan_error_analysis_master_v3.md
(GDD error pattern 2).
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

WECHSELPRAEPOSITIONEN = ["in", "an", "auf", "ueber", "unter", "vor", "neben", "zwischen"]

CONFIG = ErrorTagConfig(
    key="GDD-2",
    definition=(
        f"Wrong CASE after a two-way preposition ({', '.join(WECHSELPRAEPOSITIONEN)}; contractions im, ins, "
        "am, ans, aufs): a location (sein, wohnen, leben, bleiben, liegen, ...) takes dative; movement toward "
        "a destination (gehen, fahren, fliegen, legen, stellen, ...) takes accusative."
    ),
    not_this=(
        "The case is right but the form belongs to the wrong gender of the noun (e.g. 'in deiner Magen' -- "
        "dative, but feminine for masculine 'Magen') -> GDD-3."
    ),
    examples=[
        {"sentence": "Wuenschst du, langfristig in die Tuerkei zu bleiben?",
         "corrected": "Wuenschst du, langfristig in der Tuerkei zu bleiben?"},
        {"sentence": "Wie viele Tage war ich in die Krankenhaus?",
         "corrected": "Wie viele Tage war ich im Krankenhaus?"},
    ],
    metric_key="gdd2_errors",
    formula="count of errors tagged GDD-2 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Case: two-way preposition (Wechselpräposition)",
    report1_order=2,
)
