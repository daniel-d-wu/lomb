"""
GVT-1: Verb Tense Drift (error tag).

A verb tense that breaks the time frame the speaker set up -- present tense
inside a story about the past is the classic learner pattern. The finder
sees the whole ordered sentence list of a chunk, so drift across sentences
is visible to it.

Boundary: GVT-1 owns the choice of TENSE. A verb in the right tense but the
wrong form (wrong participle, infinitive instead of a conjugated verb,
wrong ending) is OTHER_ERROR until a verb-form fluenceme exists.

Examples are real transcript errors from dan_error_analysis_master_v3.md
and all_grammar_errors_master.json.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="GVT-1",
    definition=(
        "A verb in the wrong TENSE for the time frame: present tense in a story about the past (e.g. after "
        "'letztes Jahr', 'früher', 'als ich Kind war'), or switching tense mid-narrative. Judged against the "
        "surrounding sentences, not the sentence alone."
    ),
    not_this=(
        "Right tense, wrongly formed verb (wrong participle, infinitive where a conjugated verb is needed, "
        "wrong person ending) -> OTHER_ERROR. Verb in the wrong place -> GVT-2."
    ),
    examples=[
        {"sentence": "frueher, als ich Kind war, spiele ich gern Basketball und schwimme.",
         "corrected": "frueher, als ich Kind war, habe ich gern Basketball gespielt und bin geschwommen."},
        {"sentence": "als ich zurückkommen— nach Amerika zurückkommen, lerne ich Deutsch",
         "corrected": "als ich zurückkommen— nach Amerika zurückkommen, lernte ich Deutsch"},
    ],
    metric_key="gvt1_errors",
    formula="count of errors tagged GVT-1 this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="Verb tense drift",
    report1_order=3,
)
