"""
GVT-2: Verb Order Error.

Verb-final in subordinate clauses (dass/weil/wenn/ob) and V2 inversion
after a fronted element in main clauses. The rule itself is positional and
mechanical -- the reliability risk this metric was reclassified for is
that a written-text parser degrades on disfluent, self-corrected
spontaneous speech, which is most of this corpus.

5 of the 6 examples are real transcript errors/corrections from
dan_error_analysis_master_v3.md and all_grammar_errors_master.json (GVT
error patterns 2 and 3). The self-correction example is illustrative
(constructed to demonstrate the "evaluate only the final completed
clause" instruction) since a real transcript example combining a restart
with a verb-order error was not available in the corpus at the time this
was written.

2026-09-05: added a 6th example covering a fronted element that is
ITSELF a full clause with its own subject (a "compound Vorfeld") -- a gap
in coverage that a live pipeline run had already caught in the wild: the
model flagged "fruher, als ich Kind war, spiele ich gern Basketball und
schwimme." as a V2 inversion error, reasoning that "'spiele' follows
'ich' instead of preceding it" -- a factual misreading of the actual word
order (the string is "...spiele ich...", verb already correctly first).
Diagnosed root cause: none of the other 5 examples show a fronted element
containing its own embedded subject, so the model had nothing to learn
from about not confusing the embedded clause's subject ("ich" inside "als
ich Kind war") with the main clause's own subject when checking which
side of the verb it falls on. The new example below is real corpus data
(natasja_italki9) with exactly this structure, correctly classified as
error: false, with the reasoning spelling out how to tell the two "ich"s
apart.
"""

import sys
from pathlib import Path as _Path

# repo root on sys.path -- metric_types.py moved into pipeline/ post-reorg,
# a sibling of this file's own directory (prompts/), not on the path by default.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline.metric_types import MetricPromptConfig

SYSTEM_INSTRUCTION = """You are checking ONE German clause for ONE error type: verb order.

Rules:
- Subordinate clause (introduced by dass, weil, wenn, ob, and similar): the finite (conjugated) verb must go to the END of the clause.
- Main clause with a fronted element (something other than the subject placed first -- e.g. vielleicht, manchmal, dann, frueher): the finite verb must come in SECOND position, before the subject (V2 inversion).

If the speaker restarted or self-corrected mid-clause, evaluate ONLY the final, completed version of what they said -- ignore the abandoned false start.

If the clause is too fragmented to tell whether it's a genuine restart or a genuine error, set confidence to "low" and error to false.

Always also return `corrected`: the FULL sentence, rewritten with ONLY this error type fixed (move only the finite verb to its correct position; change nothing else -- if the speaker self-corrected, base this on the final completed clause only) if error is true, or the input sentence completely unchanged if error is false. This lets a caller diff `corrected` against the original sentence word-by-word to show exactly what changed, rather than parsing it out of the reasoning text.

Respond only in the fixed JSON shape you have been given."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "ich denke, dass mein Hoerverstaendnis ist ziemlich okay.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "Subordinate dass-clause: finite verb 'ist' is not in final position -- should be 'dass mein Hoerverstaendnis ziemlich okay ist'.",
                   "corrected": "ich denke, dass mein Hoerverstaendnis ziemlich okay ist."},
    },
    {
        "input": "vielleicht ich treffe mit einer grossen Wand irgendwann.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'vielleicht' is fronted, but the verb 'treffe' comes after the subject 'ich' instead of before it -- V2 inversion failure. Should be 'Vielleicht treffe ich...'.",
                   "corrected": "vielleicht treffe ich mit einer grossen Wand irgendwann."},
    },
    {
        "input": "dann wir sehen, wie lange wir auf Deutsch reden.",
        "answer": {"error": True, "confidence": "high",
                   "reasoning": "'dann' is fronted, but the verb 'sehen' comes after the subject 'wir' -- V2 inversion failure. Should be 'Dann sehen wir...'.",
                   "corrected": "dann sehen wir, wie lange wir auf Deutsch reden."},
    },
    {
        "input": "Manchmal gibt es ein Problem, manchmal ist es ein Vorteil.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "Both clauses correctly invert the verb before the subject after the fronted 'manchmal'.",
                   "corrected": "Manchmal gibt es ein Problem, manchmal ist es ein Vorteil."},
    },
    {
        "input": "ich denke, dass -- also, dass mein Deutsch ziemlich okay ist.",
        "answer": {"error": False, "confidence": "low",
                   "reasoning": "Illustrative example (not from the transcript corpus) of a self-correction: the speaker abandons the first 'dass' clause and restarts. Evaluating only the final completed clause ('dass mein Deutsch ziemlich okay ist'), the verb is correctly final -- no error. Replace with a validated real corpus example when one is available.",
                   "corrected": "ich denke, dass -- also, dass mein Deutsch ziemlich okay ist."},
    },
    {
        "input": "Ja, als ich Kind war, gehe ich zu einer Privatschule in Hongkong.",
        "answer": {"error": False, "confidence": "high",
                   "reasoning": "The fronted element here is itself a full clause with its own subject: 'als ich Kind war' -- that 'ich' belongs to the embedded als-clause, not the main clause. Don't confuse it with the MAIN clause's own subject, which is the second, separate 'ich' that comes after the verb. Checked that way: the finite verb 'gehe' correctly comes immediately after the fronted als-clause, before the main clause's own subject 'ich' -- V2 order is satisfied, no inversion error. (This sentence does have a real GVT-1 tense-drift error -- 'gehe' should be 'ging', since 'als ich Kind war' establishes a past frame -- but that's a different metric's job; this one checks ONLY word order.)",
                   "corrected": "Ja, als ich Kind war, gehe ich zu einer Privatschule in Hongkong."},
    },
]

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "error": {"type": "BOOLEAN"},
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
        "reasoning": {"type": "STRING"},
        "corrected": {"type": "STRING"},
    },
    "required": ["error", "confidence", "reasoning", "corrected"],
}

CONFIG = MetricPromptConfig(
    key="GVT-2",
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    response_schema=RESPONSE_SCHEMA,
    input_kind="sentence",
)
