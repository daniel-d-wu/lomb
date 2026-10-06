"""
GDD-5: Plural / Number (error tag).

~75 of 618 real GDD errors (12%) in docs/all_grammar_errors_master.json.
Added 2026-09-24.

Examples are verbatim real transcript errors from
all_grammar_errors_master.json; none from the benny_italki5/6 test sessions.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="GDD-5",
    definition=(
        "A noun in the wrong NUMBER or with a wrongly formed plural: singular where plural is required "
        "(after numbers above one, ein paar, viele, mehrere, beide), plural after a singular determiner, an "
        "English-style -s plural or other wrong plural form, viel vs viele."
    ),
    not_this=(
        "The missing dative plural -n after a preposition (mit den Leute) -> that preposition's case tag. "
        "Subject-verb agreement -> OTHER_ERROR."
    ),
    examples=[
        {"sentence": "Bei YouTube gibt es ein paar deutsche Lied",
         "corrected": "Bei YouTube gibt es ein paar deutsche Lieder"},
        {"sentence": "das ist für mich persönlich, eine bessere Optionen",
         "corrected": "das ist für mich persönlich, eine bessere Option"},
    ],
    metric_key="gdd5_errors",
    formula="count of errors tagged GDD-5 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Plural / number",
    report1_order=9,
)
