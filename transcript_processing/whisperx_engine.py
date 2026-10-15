"""
Stage 1 as a reusable object: WhisperX large-v3 + alignment + pyannote
diarization, loaded ONCE and then used for any number of files.

Why (2026-10-11, production build): scripts/hardened_whisperx_batch_job.py
loads all three models inside its batch loop's function. A cloud GPU worker
(Cloud Run with an L4) handles one upload per request and stays warm between
requests, so the models must load once per container, not once per upload.
The batch script now uses this class too, so the settings below are the
single source of truth for both.

Transcription settings are the ones chosen by benchmark (2026-09-23):
large-v3, float16 on GPU, batch_size 1, language de, the verbatim
learner-speech initial prompt, prompt-leak filtering. Each result carries a
`_meta` block recorded from these same constants, so the provenance stored in
the transcripts table can't drift from what actually ran.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ASR_MODEL = "large-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LANGUAGE = "de"
ENGINE_VERSION = "2026-10-11"  # bump when a setting below changes (recorded in _meta)

INITIAL_PROMPT = (
    "Strict verbatim German-English learner speech transcription. "
    "Transcribe exactly what is acoustically spoken, even when the German is grammatically incorrect. "
    "Do not correct, normalize, complete, or infer grammar from context. "
    "When the acoustics conflict with what would be grammatically expected, prefer the acoustically supported form. "
    "Do not add sounds, syllables, words, or inflectional endings merely because they would make the sentence grammatical. "
    "Preserve incorrect articles, grammatical case, noun number, noun endings, "
    "adjective endings, verb conjugations, verb forms, pronouns, prepositions, "
    "word order, and nonstandard word forms exactly as spoken. "
    "Do not add, remove, or change inflectional endings such as -n, -en, -e, -er, or -s. "
    "Preserve incomplete, reduced, malformed, or learner-produced word forms instead of replacing them with standard German forms. "
    "For example, if the speaker says 'vor drei Monat', write 'vor drei Monat', not 'vor drei Monaten'. "
    "If the speaker says 'zwei Katze', write 'zwei Katze', not 'zwei Katzen'. "
    "If the speaker says 'mit die Frau', write 'mit die Frau', not 'mit der Frau'. "
    "Preserve hesitations such as äh, ähm, ah, um, uh, and hm, "
    "as well as repetitions, partial words, false starts, self-corrections, stutters, "
    "code-switching, discourse markers, and malformed or invented words. "
    "Do not collapse repeated words or self-corrections into a single fluent phrase. "
    "Never rewrite the speech into standard, fluent, or grammatically correct German."
)


class WhisperXTranscriber:
    """Load once, transcribe many. transcribe() returns the diarized
    WhisperX result dict (segments with word-level speakers) plus `_meta`,
    the exact shape transcript_processing.whisperx_adapter reads."""

    def __init__(self, hf_token: str, device: str | None = None):
        # Heavy imports here, not at module level, so code that only needs
        # the constants above (or a precomputed transcript) doesn't need a
        # GPU environment installed.
        import torch
        import whisperx
        from pyannote.audio import Pipeline

        self._torch, self._whisperx = torch, whisperx
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = whisperx.load_model(ASR_MODEL, self.device, compute_type=self.compute_type,
                                         asr_options={"temperatures": [0.0], "initial_prompt": INITIAL_PROMPT})
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code=LANGUAGE, device=self.device)
        self.diarize_model = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
        if self.device == "cuda":
            self.diarize_model.to(torch.device("cuda"))

    def transcribe(self, audio_path: str, *, num_speakers: int | None = None) -> dict:
        import pandas as pd

        from transcript_processing.whisperx_adapter import filter_prompt_leak

        whisperx, torch = self._whisperx, self._torch
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, language=LANGUAGE, batch_size=self.batch_size)
        result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device,
                                return_char_alignments=False)
        call_kwargs = {"num_speakers": num_speakers} if num_speakers is not None else {}
        diarization = self.diarize_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000},
                                         **call_kwargs)
        rows = [{"start": float(t.start), "end": float(t.end), "speaker": spk}
                for t, _, spk in diarization.exclusive_speaker_diarization.itertracks(yield_label=True)]
        result = whisperx.assign_word_speakers(pd.DataFrame(rows), result)
        kept, dropped = filter_prompt_leak(result["segments"], known_prompt_leak=INITIAL_PROMPT)
        result["segments"] = kept
        result["_meta"] = {
            "engine": "whisperx",
            "model": ASR_MODEL,
            "diarization_model": DIARIZATION_MODEL,
            "language": LANGUAGE,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "initial_prompt": INITIAL_PROMPT,
            "num_speakers_hint": num_speakers,
            "prompt_leak_segments_dropped": len(dropped),
            "source_audio_filename": Path(audio_path).name,
            "engine_version": ENGINE_VERSION,
        }
        return result


class PrecomputedTranscriber:
    """Stand-in that returns an already-produced WhisperX JSON -- for tests
    and for re-analysing transcripts made earlier, with no GPU."""

    def __init__(self, transcript_json_path: str):
        self.path = transcript_json_path

    def transcribe(self, audio_path: str, *, num_speakers: int | None = None) -> dict:
        import json
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)
