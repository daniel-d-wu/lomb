"""
LPF: Lexical Phrase / False Friend -- English (L1) transfer (error tag).

2026-09-24: widened to Dan's own LPF definition in
dan_error_analysis_master_v3.md -- "LPF -- Lexical Phrase / False Friend
(L1 English transfer)", three sub-concepts: (1) wrong preposition from L1
transfer, (2) domain-specific verb confusion, (3) idiom transfer / calques
(false friends included). The old narrow detector only covered (1), so
English-caused word choice like "Beginner-Glück" belonged to no fluenceme
(docs/lomb_failure_modes_v1.md M17). Corpus: 247 LPF entries -- 97 wrong
preposition, 98 calque, 24 domain verb, 20 false friend.

Examples are verbatim real transcript errors from
all_grammar_errors_master.json; none from the benny_italki5/6 test sessions.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import ErrorTagConfig

CONFIG = ErrorTagConfig(
    key="LPF",
    definition=(
        "A word or phrase copied from ENGLISH: (1) a wrong, missing or extra preposition that follows English "
        "('suche für' -> 'suche', 'freue mich für' -> 'auf', 'passt für mich' -> 'passt mir'); (2) a verb "
        "confused the way English words overlap (wissen vs kennen, kochen vs backen); (3) a literal "
        "translation of an English idiom or compound, or a false friend (Kocher for Koch, Subjekt for Fach)."
    ),
    not_this="Unnatural word choice with no English source -> LP.",
    examples=[
        {"sentence": "Ich suche für Muttersprachler oder Muttersprachlerin",
         "corrected": "Ich suche Muttersprachler oder Muttersprachlerin"},
        {"sentence": "weil ich weiß deutsche Politik nicht so gut",
         "corrected": "weil ich kenne deutsche Politik nicht so gut"},
        {"sentence": "Ja, ich— ich mache einfach Dusche und, ja, ich riech gut.",
         "corrected": "Ja, ich— ich dusche einfach und, ja, ich riech gut."},
    ],
    metric_key="lpf_errors",
    formula="count of errors tagged LPF this session; each error counts once (placeholder -- not an error rate)",
    report1_tag="English transfer (preposition, phrase, false friend)",
    report1_order=5,
)
