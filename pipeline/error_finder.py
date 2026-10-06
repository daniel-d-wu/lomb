"""
Call 1 of 2 in error detection: find EVERY grammar and word-choice error
in a chunk of the learner's sentences, with no categories in mind.

Why (2026-09-24): the previous design asked one narrow question per error
fluenceme ("only check gender, skip anything after a preposition"). Real
errors that fell between the questions were never reported -- the models
saw them and followed the "not your job" rules (docs/lomb_failure_modes_v1.md
M17/M18). Finding first and categorising second (pipeline/error_sorter.py)
means an error can only go unreported if the finder misses it, never
because no category asked about it.

Policy decisions this prompt encodes (Dan, 2026-09-24):
- An error inside an abandoned or restarted phrase still counts. A restart
  that repeats the same mistake counts once. The restart/repeat itself is a
  disfluency, not a grammar error, and is not reported here.
- Grammar AND word choice (LP/LPF-type errors) are both in scope.
- One entry per error, not per sentence: two errors in one sentence are two
  entries, each with its own minimal correction.

Few-shot sentences are verbatim real transcript text from
docs/all_grammar_errors_master.json (errors, plus context lines as the
error-free examples); none from the benny_italki5/6 test sessions. The
annotations narrow each corpus correction to one error per entry.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.batching import batch_input
from pipeline.metric_types import PromptConfig

FINDER_KEY = "ERROR_FINDER"

SYSTEM_INSTRUCTION = """You are reviewing a transcript of an adult English speaker who is learning German, speaking in a conversation with a tutor. The input is a JSON array of the LEARNER's sentences, in the order spoken, each with an "index". Speech recognition produced the text, so punctuation and sentence breaks are approximate.

Find EVERY grammar and word-choice error the learner made. Report each error as its own entry: a sentence can have several errors, and many sentences have none.

Report:
- grammar errors of every kind: case, gender, adjective endings, plural, verb tense, verb form and conjugation, verb position, auxiliaries, reflexives, prepositions, missing or extra words
- word choice: a word or phrase a native speaker would not use there, including literal translations from English, false friends and words that don't exist
- errors inside incomplete, abandoned or restarted phrases. If the learner abandons a phrase and restarts, an error in the abandoned part still counts. If the restart repeats the same mistake, report it once. If the learner fixed it in the restart, still report it and set self_corrected to true.

Do NOT report:
- fillers (äh, ähm, hm, mhm), stutters, repeated words, false starts or restarts in themselves -- those are disfluencies, measured elsewhere; only a grammar or word error inside them counts
- punctuation, capitalization or spelling of the transcript
- normal spoken German: colloquial contractions (hab, gibt's, is, ne) and short answers or fragments that are normal in conversation
- an English word the learner uses because they lack the German one -- unless the German grammar around it is wrong

For each error return:
- index: the index of the sentence it is in
- said: the exact wrong word(s), copied from the sentence
- corrected: the FULL sentence with ONLY this error fixed. Change as few words as possible and keep everything else exactly as it was -- other errors, restarts and fillers included -- so a program can diff it against the original and show exactly what changed.
- reasoning: one short sentence naming the rule
- confidence: "high" if you are sure it is an error; "low" if a native speaker might accept it or you can't tell what the learner meant
- self_corrected: true if the learner corrected this error themselves later in the same sentence
- possible_transcription_error: true if the words look misheard by the speech recognizer (nonsense in context) rather than a learner mistake

Judge each sentence in context: the neighboring sentences set the time frame (for tense) and the topic. Report an error on the sentence where it occurs. If there are no errors, return an empty list.

Respond only in the fixed JSON shape you have been given."""

_EXAMPLE_SENTENCES = [
    "Ich bin ein bisschen müde.",
    "Weil es ist fast Schlafzeit.",
    "Um regelmäßig zu wiederholen, regelmäßig zu üben.",
    "Und, und dann kommt letztes Jahr.",
    "Ich mache ein Urlaub in Hamburg und— nach Hamburg und Köln.",
    "Mhm, ja, ich verstehe— jetzt verstehe ich, was du meinst.",
    "als ich zurückkommen— nach Amerika zurückkommen, lerne ich Deutsch",
    "diese, diese Gespräch zu aufnehmen",
    "Ja, ich habe noch viel Zeit.",
    "Wenn man 40er ist— 40ern ist, dann ist es egal.",
    "wenn man seine Job verkündigt hat",
    "Ja, es ist nur 30 Minuten von mir entfernt und wir fahren da oft spontan hin.",
    "Ich suche für Muttersprachler oder Muttersprachlerin",
]


def _err(index, said, corrected, reasoning, *, self_corrected=False):
    return {"index": index, "said": said, "corrected": corrected, "reasoning": reasoning,
            "confidence": "high", "self_corrected": self_corrected, "possible_transcription_error": False}


_EXAMPLE_ERRORS = [
    _err(1, "ist", "Weil es fast Schlafzeit ist.",
         "In a weil-clause the conjugated verb goes to the end."),
    _err(1, "Schlafzeit", "Weil es ist fast Schlafenszeit.",
         "The German word for bedtime is 'Schlafenszeit'."),
    _err(3, "kommt", "Und, und dann kam letztes Jahr.",
         "The story is about last year, so it needs the past tense, not present 'kommt'."),
    _err(4, "mache", "Ich habe ein Urlaub in Hamburg und— nach Hamburg und Köln gemacht.",
         "The previous sentence set a past time frame ('letztes Jahr'), so 'habe ... gemacht', not present 'mache'."),
    _err(4, "ein Urlaub", "Ich mache einen Urlaub in Hamburg und— nach Hamburg und Köln.",
         "'Urlaub' is masculine and the direct object, so accusative 'einen Urlaub'."),
    _err(6, "zurückkommen", "als ich zurückkam— nach Amerika zurückkam, lerne ich Deutsch",
         "An als-clause needs a conjugated verb ('zurückkam'), not the infinitive; the restart repeats the same mistake, so it counts once."),
    _err(6, "lerne", "als ich zurückkommen— nach Amerika zurückkommen, lernte ich Deutsch",
         "The als-clause sets a past time frame, so past tense 'lernte', not present 'lerne'."),
    _err(7, "diese", "dieses, dieses Gespräch zu aufnehmen",
         "'Gespräch' is neuter: 'dieses Gespräch'; the restart repeats the same mistake, so it counts once."),
    _err(7, "zu aufnehmen", "diese, diese Gespräch aufzunehmen",
         "With a separable verb, 'zu' goes between prefix and stem: 'aufzunehmen'."),
    _err(9, "40er", "Wenn man 40ern ist— 40ern ist, dann ist es egal.",
         "The plural form is '40ern'; the learner fixed this in the restart.", self_corrected=True),
    _err(9, "40ern ist", "Wenn man 40er ist— in den 40ern ist, dann ist es egal.",
         "Being in your forties is 'in den 40ern sein'; the preposition and article are missing."),
    _err(10, "seine Job", "wenn man seinen Job verkündigt hat",
         "'Job' is masculine, so the accusative possessive is 'seinen'."),
    _err(10, "verkündigt", "wenn man seine Job gekündigt hat",
         "Quitting a job is 'kündigen'; 'verkündigen' means to announce."),
    _err(12, "suche für", "Ich suche Muttersprachler oder Muttersprachlerin",
         "'suchen' takes a direct object; 'für' is copied from English 'look for'."),
    _err(12, "Muttersprachlerin", "Ich suche für Muttersprachler oder Muttersprachlerinnen",
         "Paired with plural 'Muttersprachler', the feminine form must be plural too: 'Muttersprachlerinnen'."),
]

_ERROR_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "index": {"type": "INTEGER"},
        "said": {"type": "STRING"},
        "corrected": {"type": "STRING"},
        "reasoning": {"type": "STRING"},
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
        "self_corrected": {"type": "BOOLEAN"},
        "possible_transcription_error": {"type": "BOOLEAN"},
    },
    "required": ["index", "said", "corrected", "reasoning", "confidence",
                 "self_corrected", "possible_transcription_error"],
}

FINDER_CONFIG = PromptConfig(
    key=FINDER_KEY,
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=[{
        "input": batch_input(list(enumerate(_EXAMPLE_SENTENCES))),
        "answer": {"errors": _EXAMPLE_ERRORS},
    }],
    response_schema={
        "type": "OBJECT",
        "properties": {"errors": {"type": "ARRAY", "items": _ERROR_ITEM_SCHEMA}},
        "required": ["errors"],
    },
    input_kind="sentence_batch",
)


def find_errors(provider, indexed_sentences: list[tuple[int, str]]) -> list[dict]:
    """One provider call for one chunk. Returns the found errors in the
    model's order, each {"index", "said", "corrected", "reasoning",
    "confidence", "self_corrected", "possible_transcription_error"}.
    Entries pointing at a sentence that wasn't sent are dropped (a model
    slip, never guessed onto a nearby sentence)."""
    raw = provider.classify(FINDER_CONFIG, batch_input(indexed_sentences))
    requested = {i for i, _ in indexed_sentences}
    # Strict-schema providers always return every field; the defaults only
    # matter for a provider without strict output, and err on the side of
    # "not trustworthy" (low confidence) rather than inventing certainty.
    defaults = {"said": "", "corrected": "", "reasoning": "", "confidence": "low",
                "self_corrected": False, "possible_transcription_error": False}
    return [{**defaults, **e} for e in raw.get("errors", []) if e.get("index") in requested]


if __name__ == "__main__":
    # The few-shot example must itself obey the rules the prompt states.
    for e in _EXAMPLE_ERRORS:
        sentence = _EXAMPLE_SENTENCES[e["index"]]
        assert e["said"] in sentence, f"'said' must be copied from the sentence: {e['said']!r} / {sentence!r}"
        assert e["corrected"] != sentence, f"a correction must change something: {sentence!r}"
    print(f"{len(_EXAMPLE_SENTENCES)} example sentences, {len(_EXAMPLE_ERRORS)} example errors, "
          f"{len(_EXAMPLE_SENTENCES) - len({e['index'] for e in _EXAMPLE_ERRORS})} error-free -- consistent.")
    print(json.dumps(FINDER_CONFIG.few_shot_examples[0]["answer"]["errors"][0], ensure_ascii=False))
