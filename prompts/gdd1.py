"""
GDD-1: Case Error -- Always-Dative Prepositions (error tag).

mit / von / bei / zu / nach / aus / gegenueber / ausser / seit / ab always
take dative, regardless of motion or direction.

2026-09-24 (find-then-sort): this file no longer holds a detector prompt.
The error finder (pipeline/error_finder.py) finds every error; the sorter
(pipeline/error_sorter.py) gives each one a tag using the definition and
examples below. The old narrow detector prompt was removed 2026-09-24
(see git history).

Examples are real transcript errors from dan_error_analysis_master_v3.md
(GDD error pattern 1), none invented.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

# Fixed, closed vocabulary -- German only has these always-dative
# prepositions. Other tag files quote it so the case/gender boundary is
# stated against one list.
ALWAYS_DATIVE_PREPOSITIONS = [
    "mit", "von", "bei", "zu", "nach", "aus", "gegenueber", "ausser", "seit", "ab",
]

CONFIG = ErrorTagConfig(
    key="GDD-1",
    definition=(
        f"Wrong CASE after an always-dative preposition ({', '.join(ALWAYS_DATIVE_PREPOSITIONS)}; "
        "contractions vom, beim, zum, zur). The noun phrase it governs must be dative: dem (masculine/neuter), "
        "der (feminine), den + noun ending -n (plural)."
    ),
    not_this=(
        "The form IS dative but for the wrong gender of the noun (e.g. 'von einer Gefühl' -- 'einer' is "
        "dative feminine, 'Gefühl' is neuter) -> GDD-3. The wrong preposition itself -> LPF."
    ),
    examples=[
        {"sentence": "Trotzdem kann ich nicht mit die Leute sprechen.",
         "corrected": "Trotzdem kann ich nicht mit den Leuten sprechen."},
        {"sentence": "Mit die Grammatik habe ich noch nicht so bewusst gelernt.",
         "corrected": "Mit der Grammatik habe ich noch nicht so bewusst gelernt."},
        {"sentence": "Frueher war ich ein Datenwissenschaftler beim Banken.",
         "corrected": "Frueher war ich ein Datenwissenschaftler bei der Bank."},
    ],
    metric_key="gdd1_errors",
    formula="count of errors tagged GDD-1 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Case: always-dative preposition",
    report1_order=1,
)
