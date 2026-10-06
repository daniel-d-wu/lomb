"""
GDD-6: Case for the noun's role in the clause -- verb objects, recipients,
reflexives (error tag).

~23+ of 618 real GDD errors in docs/all_grammar_errors_master.json (keyword
count, likely undercounted). Added 2026-09-24.

Boundary: GDD-6 owns case errors where NO preposition is involved and the
gender is right. Wrong gender is GDD-3; after a preposition is GDD-1/GDD-2.

Examples are verbatim real transcript errors from
all_grammar_errors_master.json; none from the benny_italki5/6 test sessions.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

DATIVE_VERBS = [
    "helfen", "danken", "gefallen", "gehören", "vertrauen", "folgen", "antworten",
    "zustimmen", "gratulieren", "passen", "schmecken", "zuhören", "fehlen", "glauben (a person)",
]
DATIVE_RECIPIENT_VERBS = [
    "geben", "schenken", "zeigen", "erklären", "erzählen", "sagen", "empfehlen",
    "vorschlagen", "(eine Frage) stellen",
]

CONFIG = ErrorTagConfig(
    key="GDD-6",
    definition=(
        "Wrong CASE for the noun phrase's role in its clause, with no preposition involved: the object of a "
        f"dative verb ({', '.join(DATIVE_VERBS)}); the person receiving something with "
        f"{', '.join(DATIVE_RECIPIENT_VERBS)}; a direct object that must be accusative (einen/keinen/meinen, "
        "including after 'es gibt'); a subject that must be nominative; a reflexive pronoun in the wrong case "
        "(mir vs mich)."
    ),
    not_this="Wrong gender -> GDD-3. After a preposition -> GDD-1 or GDD-2.",
    examples=[
        {"sentence": "dann würde ich das zustimmen", "corrected": "dann würde ich dem zustimmen"},
        {"sentence": "habe ich es kein— kein Druck", "corrected": "habe ich es kein— keinen Druck"},
        {"sentence": "Ich mache ein Urlaub in Hamburg", "corrected": "Ich mache einen Urlaub in Hamburg"},
    ],
    metric_key="gdd6_errors",
    formula="count of errors tagged GDD-6 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Case: object of the verb",
    report1_order=10,
)
