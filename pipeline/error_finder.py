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
  entries, each with its own fix.

2026-10-06, tuning round after the first retest: never merge two errors
into one entry (M27); clean sentences drop ALL repetition and add no
content the learner didn't say (M26); a form that is wrong in every
reading is high confidence even when a pronoun's referent is unknown
("vertraue das" had been dropped as low confidence).

2026-10-06 -- the model no longer rewrites the sentence (failure mode M21).
Its reasoning was right but its full-sentence rewrites were garbled ("mit
der" reasoned, "mit den" written; "dem" inserted without removing "das";
an unsaid word appended). Now:
- per error it returns only `said` (the exact wrong words) and `fix` (what
  replaces them), with `reasoning` BEFORE `fix` in the schema so it states
  the rule before committing to the fix; code builds `corrected` by
  swapping `fix` in for `said` (flag_quality.apply_fix) and rejects the
  error if `said` isn't in the sentence;
- per sentence with an error it returns a `clean` version -- fluent, fully
  correct, no fillers or restarts (Dan: what the learner sees must be a
  clean, fully correct sentence). Code checks the clean sentence contains
  every fix (flag_quality.clean_has_fix); only then can it be shown.

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
from pipeline.flag_quality import apply_fix, clean_has_fix
from pipeline.metric_types import PromptConfig

FINDER_KEY = "ERROR_FINDER"

SYSTEM_INSTRUCTION = """You are reviewing a transcript of an adult English speaker who is learning German, speaking in a conversation with a tutor. The input is a JSON array of the LEARNER's sentences, in the order spoken, each with an "index". Speech recognition produced the text, so punctuation and sentence breaks are approximate.

Find EVERY grammar and word-choice error the learner made. Report each error as its own entry: a sentence can have several errors, and many sentences have none. Never merge two errors into one entry, even when they are next to each other (e.g. a wrong article and a wrong verb form in the same phrase are two entries, each with its own short said and fix).

Report:
- grammar errors of every kind: case, gender, adjective endings, plural, verb tense, verb form and conjugation, verb position, auxiliaries, reflexives, prepositions, missing or extra words
- word choice: a word or phrase a native speaker would not use there, including literal translations from English, false friends and words that don't exist
- errors inside incomplete, abandoned or restarted phrases. If the learner abandons a phrase and restarts, an error in the abandoned part still counts. If the restart repeats the same mistake, report it once. If the learner fixed it in the restart, still report it and set self_corrected to true.

Do NOT report:
- fillers (äh, ähm, hm, mhm), stutters, repeated words, false starts or restarts in themselves -- those are disfluencies, measured elsewhere; only a grammar or word error inside them counts
- punctuation, capitalization or spelling of the transcript
- normal spoken German: colloquial contractions (hab, gibt's, is, ne) and short answers or fragments that are normal in conversation
- an English word the learner uses because they lack the German one -- unless the German grammar around it is wrong

For each error return, in this order:
- index: the index of the sentence it is in
- said: the wrong words, copied character for character from the sentence -- the shortest stretch that contains the whole error. It must appear only once in the sentence; add a neighbouring word if needed. If the fix needs changes in two places (e.g. "habe ... gemacht"), make said cover everything between them.
- reasoning: one short sentence naming the rule AND the correct form, e.g. "Karte is feminine; mit takes the dative, so 'der Karte'."
- fix: exactly the text that replaces said, with only this error fixed and everything else inside said unchanged (fillers and restarts included). Check it against your reasoning before writing it. Empty string if the words should simply be deleted.
- confidence: "high" if you are sure it is an error; "low" only if a native speaker might accept the words as said, or you can't tell what the learner meant. If the form is wrong in every possible reading (e.g. a dative verb with an accusative pronoun), it is "high" even when you don't know what the pronoun refers to.
- self_corrected: true if the learner corrected this error themselves later in the same sentence
- possible_transcription_error: true if the words look misheard by the speech recognizer (nonsense in context) rather than a learner mistake

Then, for EVERY sentence where you reported at least one error, return one entry in clean_sentences:
- index
- understandable: false if the sentence is too garbled or unfinished to tell what the learner meant
- clean: the sentence as a native speaker would say it -- every grammar and word-choice error fixed; fillers, stutters and false starts removed; and ALL repetition removed: if the learner restarted or repeated a phrase, keep only one complete version of it. Keep the learner's own words and meaning. Do not add content the learner did not say -- no new nouns, names, facts or numbers; you may only add small grammatical words a correct sentence needs (an article, pronoun, auxiliary, "zu", a verb ending). If it can't be made correct without guessing what the learner meant, set understandable to false and leave clean empty. A normal conversational fragment may stay a fragment. It must contain each of your fixes for that sentence. If writing it reveals an error you did not list, list that error too.

Judge each sentence in context: the neighboring sentences set the time frame (for tense) and the topic. Report an error on the sentence where it occurs. If there are no errors, return empty lists.

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


def _err(index, said, reasoning, fix, *, self_corrected=False):
    return {"index": index, "said": said, "reasoning": reasoning, "fix": fix,
            "confidence": "high", "self_corrected": self_corrected, "possible_transcription_error": False}


_EXAMPLE_ERRORS = [
    _err(1, "ist fast Schlafzeit", "In a weil-clause the conjugated verb goes to the end: 'fast Schlafzeit ist'.",
         "fast Schlafzeit ist"),
    _err(1, "Schlafzeit", "The German word for bedtime is 'Schlafenszeit'.", "Schlafenszeit"),
    _err(3, "kommt", "The story is about last year, so past tense 'kam', not present 'kommt'.", "kam"),
    _err(4, "mache", "The previous sentence set a past time frame ('letztes Jahr'), so past tense 'machte'.", "machte"),
    _err(4, "ein Urlaub", "'Urlaub' is masculine and the direct object, so accusative 'einen Urlaub'.", "einen Urlaub"),
    _err(6, "zurückkommen— nach Amerika zurückkommen",
         "An als-clause needs a conjugated past verb, 'zurückkam', not the infinitive; the restart repeats the same "
         "mistake, so it counts once.", "zurückkam— nach Amerika zurückkam"),
    _err(6, "lerne", "The als-clause sets a past time frame, so past tense 'lernte', not present 'lerne'.", "lernte"),
    _err(7, "diese, diese", "'Gespräch' is neuter: 'dieses Gespräch'; the restart repeats the same mistake, so it "
         "counts once.", "dieses, dieses"),
    _err(7, "zu aufnehmen", "With a separable verb, 'zu' goes between prefix and stem: 'aufzunehmen'.", "aufzunehmen"),
    _err(9, "man 40er ist", "The plural form is '40ern'; the learner fixed this in the restart.", "man 40ern ist",
         self_corrected=True),
    _err(9, "— 40ern ist", "Being in your forties is 'in den 40ern sein'; the preposition and article are missing.",
         "— in den 40ern ist"),
    _err(10, "seine Job", "'Job' is masculine, so the accusative possessive is 'seinen Job'.", "seinen Job"),
    _err(10, "verkündigt", "Quitting a job is 'kündigen', so 'gekündigt'; 'verkündigen' means to announce.", "gekündigt"),
    _err(12, "suche für", "'suchen' takes a direct object; 'für' is copied from English 'look for', so just 'suche'.",
         "suche"),
    _err(12, "Muttersprachlerin", "Paired with plural 'Muttersprachler', the feminine form must be plural too: "
         "'Muttersprachlerinnen'.", "Muttersprachlerinnen"),
]

_EXAMPLE_CLEAN = [
    {"index": 1, "understandable": True, "clean": "Weil es fast Schlafenszeit ist."},
    {"index": 3, "understandable": True, "clean": "Und dann kam das letzte Jahr."},
    {"index": 4, "understandable": True, "clean": "Ich machte einen Urlaub in Hamburg und Köln."},
    {"index": 6, "understandable": True, "clean": "Als ich nach Amerika zurückkam, lernte ich Deutsch."},
    {"index": 7, "understandable": True, "clean": "dieses Gespräch aufzunehmen"},
    {"index": 9, "understandable": True, "clean": "Wenn man in den 40ern ist, dann ist es egal."},
    {"index": 10, "understandable": True, "clean": "wenn man seinen Job gekündigt hat"},
    {"index": 12, "understandable": True, "clean": "Ich suche Muttersprachler oder Muttersprachlerinnen."},
]

_ERROR_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "index": {"type": "INTEGER"},
        "said": {"type": "STRING"},
        "reasoning": {"type": "STRING"},
        "fix": {"type": "STRING"},
        "confidence": {"type": "STRING", "enum": ["high", "low"]},
        "self_corrected": {"type": "BOOLEAN"},
        "possible_transcription_error": {"type": "BOOLEAN"},
    },
    "required": ["index", "said", "reasoning", "fix", "confidence", "self_corrected",
                 "possible_transcription_error"],
}
_CLEAN_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "index": {"type": "INTEGER"},
        "understandable": {"type": "BOOLEAN"},
        "clean": {"type": "STRING"},
    },
    "required": ["index", "understandable", "clean"],
}

FINDER_CONFIG = PromptConfig(
    key=FINDER_KEY,
    system_instruction=SYSTEM_INSTRUCTION,
    few_shot_examples=[{
        "input": batch_input(list(enumerate(_EXAMPLE_SENTENCES))),
        "answer": {"errors": _EXAMPLE_ERRORS, "clean_sentences": _EXAMPLE_CLEAN},
    }],
    response_schema={
        "type": "OBJECT",
        "properties": {
            "errors": {"type": "ARRAY", "items": _ERROR_ITEM_SCHEMA},
            "clean_sentences": {"type": "ARRAY", "items": _CLEAN_ITEM_SCHEMA},
        },
        "required": ["errors", "clean_sentences"],
    },
    input_kind="sentence_batch",
)


def find_errors(provider, indexed_sentences: list[tuple[int, str]]) -> list[dict]:
    """One provider call for one chunk. Returns the found errors in the
    model's order, each with the model's fields (index, said, reasoning,
    fix, confidence, self_corrected, possible_transcription_error) plus,
    built here in code:
      corrected       the sentence with `fix` swapped in for `said`
      said_not_found  True if `said` isn't in the sentence (then corrected "")
      clean           the model's clean version of that sentence ("" if none)
      understandable  False if the model couldn't tell what was meant
      clean_ok        clean exists, is understandable and contains this fix
    Entries pointing at a sentence that wasn't sent are dropped (a model
    slip, never guessed onto a nearby sentence)."""
    raw = provider.classify(FINDER_CONFIG, batch_input(indexed_sentences))
    text = dict(indexed_sentences)
    clean_by_index = {}
    for c in raw.get("clean_sentences", []):
        if c.get("index") in text and c.get("index") not in clean_by_index:
            clean_by_index[c["index"]] = c
    # Strict-schema providers always return every field; the defaults only
    # matter for a provider without strict output, and err on the side of
    # "not trustworthy" (low confidence) rather than inventing certainty.
    defaults = {"said": "", "reasoning": "", "fix": "", "confidence": "low",
                "self_corrected": False, "possible_transcription_error": False}
    errors = []
    for e in raw.get("errors", []):
        if e.get("index") not in text:
            continue
        e = {**defaults, **e}
        corrected = apply_fix(text[e["index"]], e["said"], e["fix"])
        c = clean_by_index.get(e["index"], {})
        clean = (c.get("clean") or "").strip()
        understandable = bool(c.get("understandable", False))
        e.update(corrected=corrected or "", said_not_found=corrected is None, clean=clean,
                 understandable=understandable,
                 clean_ok=understandable and clean_has_fix(clean, e["said"], e["fix"]))
        errors.append(e)
    return errors


if __name__ == "__main__":
    # The few-shot example must itself obey the rules the prompt states.
    clean = {c["index"]: c["clean"] for c in _EXAMPLE_CLEAN}
    for e in _EXAMPLE_ERRORS:
        sentence = _EXAMPLE_SENTENCES[e["index"]]
        assert sentence.count(e["said"]) == 1, f"'said' must appear exactly once: {e['said']!r} / {sentence!r}"
        assert apply_fix(sentence, e["said"], e["fix"]) != sentence, f"a fix must change something: {sentence!r}"
        assert clean_has_fix(clean[e["index"]], e["said"], e["fix"]), \
            f"clean sentence must contain the fix: {e['fix']!r} / {clean[e['index']]!r}"
    assert set(clean) == {e["index"] for e in _EXAMPLE_ERRORS}, "one clean sentence per sentence with an error"
    print(f"{len(_EXAMPLE_SENTENCES)} example sentences, {len(_EXAMPLE_ERRORS)} example errors, "
          f"{len(_EXAMPLE_SENTENCES) - len(clean)} error-free -- consistent.")
    for e in _EXAMPLE_ERRORS[:3]:
        print(f"  {e['said']!r} -> {e['fix']!r}: {apply_fix(_EXAMPLE_SENTENCES[e['index']], e['said'], e['fix'])}")
