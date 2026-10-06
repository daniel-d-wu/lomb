"""
GDD-3: Noun Gender -- a determiner or pronoun with the wrong gender (error tag).

~218 of 618 real GDD errors (35%) in docs/all_grammar_errors_master.json,
the largest single GDD error type. Added 2026-09-24.

2026-09-24 (find-then-sort): a tag definition for the sorter, not a
detector. Unlike the old narrow detector, gender errors AFTER a preposition
belong here too ("in deiner Magen", "von einer Gefühl") -- the old one was
told to skip them and GDD-1/GDD-2 only judged the case, so they were never
reported (docs/lomb_failure_modes_v1.md M17).

Examples are verbatim real transcript errors from
all_grammar_errors_master.json; none from the benny_italki5/6 test sessions.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="GDD-3",
    definition=(
        "An article, possessive or other determiner (der/die/das, ein/eine, kein-, dies-, jed-, welch-, "
        "mein/dein/sein/ihr/unser/euer) or a pronoun referring back to a noun, whose form shows the wrong "
        "GENDER for that noun -- anywhere in the sentence, including after a preposition. The form would be "
        "right for the case needed if the noun had a different gender."
    ),
    not_this=(
        "Right gender, wrong case -> GDD-1, GDD-2 or GDD-6. Right article but wrong adjective ending -> GDD-4. "
        "Plural noun phrases carry no gender -> GDD-5 or a case tag."
    ),
    examples=[
        {"sentence": "Ist das die Wort", "corrected": "Ist das das Wort"},
        {"sentence": "bleibt in deiner Magen und—", "corrected": "bleibt in deinem Magen und—"},
        {"sentence": "das hängt von einer Gefühl ab", "corrected": "das hängt von einem Gefühl ab"},
        {"sentence": "wenn man seine Job verkündigt hat", "corrected": "wenn man seinen Job verkündigt hat"},
    ],
    metric_key="gdd3_errors",
    formula="count of errors tagged GDD-3 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Gender: wrong article for the noun",
    report1_order=7,
)
