"""
Hardened version of whisperX_batch_job.ipynb (transcript_testing/, dated
9/9/2026) -- the script that already produced all 33 real whisperX+
pyannote transcripts this project's Phase 1 adapter/database work was
built and tested against. This file changes exactly three things, per
Dan's own explicit sign-off on scope (deferring everything else,
including the diarization-quality issue his own README.txt 9/9/2026 note
already flagged -- "attempted Diarization fix w/ pyannote package. not
effective" -- which this file does NOT attempt to fix):

  1. HF_TOKEN moves from a hardcoded string to an environment variable.
     The notebook had a real token typed directly into a cell -- fine for
     a one-person local notebook, not something that belongs copy-pasted
     into a script that might get committed, shared, or run from a
     different machine. Reads os.environ["HF_TOKEN"]; fails loudly and
     immediately (before loading any model) if it's unset, rather than
     failing confusingly deep inside Pipeline.from_pretrained().

  2. num_speakers is no longer hardcoded to 2. The original notebook
     passed num_speakers=2 to every file in the batch regardless of how
     many people were actually in that recording -- fine for this
     specific 33-file corpus (all 2-speaker conversations, as far as
     Dan's own file naming suggests), but wrong as a general default, and
     exactly the kind of hardcoded assumption claude/lomb_data_schema_v1.md's
     audio_assets.speaker_count_hint column was designed to replace (a
     per-file, not per-batch, hint). This script accepts an optional
     per-file hint file (--speaker-hints-file, a JSON object mapping
     filename stem -> speaker count) and an optional batch-wide default
     (--num-speakers), and passes NEITHER to pyannote (letting it
     auto-detect) when no hint is given for a file -- auto-detection is a
     more honest default than silently assuming 2 for every file.

  3. The known prompt-leak segments (confirmed, real, against this exact
     initial_prompt and this exact corpus -- see transcript_processing/
     whisperx_adapter.py's own module docstring for the confirmed case)
     are filtered out of the SAVED json, using whisperx_adapter.py's own
     filter_prompt_leak() function with the real initial_prompt text this
     script actually used -- not a re-implementation of that filtering
     logic here. The exact prompt text is also written into a new "_meta"
     block in the saved JSON (alongside num_speakers_hint and a few other
     provenance fields), so a transcript saved by an older, unhardened
     run of this notebook can still be filtered correctly later by
     passing that same prompt text to from_whisperx_transcript() by hand.

  4. compute_type moves from a flat "int8" to "float16" on CUDA (still
     "int8" on CPU, where float16 isn't well supported). A direct 8-combo
     benchmark of model x compute_type on this exact GPU (2026-09-22,
     transcript_testing/benchmark_results/) showed int8_float16 measurably
     dropping disfluencies and self-corrections relative to float16 on one
     of two test sessions -- collapsed repeats ("oh, oh" -> "oh"), a
     dropped stutter ("ein bis-, ein bisschen" -> "ein bisschen"), and one
     article normalized to the grammatically correct form ("eine" ->
     "einem") -- exactly the class of learner-error detail this pipeline
     exists to preserve. Dan's own call: take the slower runtime over that
     risk. batch_size stays at 1 -- unaffected by this change, already the
     most VRAM-conservative setting from the original notebook's OOM
     experience.

Everything else -- model loading, the transcribe/align/diarize/assign
sequence, and the per-file loop and its error handling -- is UNCHANGED.
This is a hardening pass, not a rewrite or a diarization-quality fix.

NOT RUN OR TESTED FROM THIS SANDBOX. whisperX, pyannote.audio, and torch
with real CUDA are not installed here, this sandbox has no GPU, and
(per Dan's own explicit decision) a fresh live verification run belongs
on his own machine, not here. The only check this file has actually had
is `python3 -m py_compile` (syntax-valid) and a manual read-through
against the original notebook cell it's based on. Run it for real on
your machine to confirm the three changes above didn't break anything
the original notebook did correctly:

    set HF_TOKEN=hf_...           (Windows cmd.exe)  or
    $env:HF_TOKEN = "hf_..."      (PowerShell)

    python hardened_whisperx_batch_job.py
    python hardened_whisperx_batch_job.py --num-speakers 2
    python hardened_whisperx_batch_job.py --speaker-hints-file hints.json
    python hardened_whisperx_batch_job.py --files lilli_tandem1_4_30_2026_5min.m4a
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

DEFAULT_AUDIO_DIR = r"C:\Users\Danie\Documents\Sound recordings\german_recordings\five_min_trunc"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac")

# Unchanged from the original notebook cell -- same instruction text,
# same reasoning (verbatim/error-preserving transcription for a learner-
# speech corpus). Kept as a module-level constant (not re-typed at the
# call site) specifically so it can also be passed to
# whisperx_adapter.filter_prompt_leak() below AND written into each
# saved file's "_meta" block -- one source of truth for what the prompt
# actually was, not three copies that could drift apart.
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


def load_speaker_hints(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        hints = json.load(f)
    if not isinstance(hints, dict):
        raise ValueError(
            f"{path} must be a JSON object mapping filename stem -> speaker count, "
            f"got {type(hints).__name__}"
        )
    return {str(k): int(v) for k, v in hints.items()}


def run_batch(
    audio_dir: str,
    output_dir: str,
    hf_token: str,
    num_speakers_default: int | None,
    speaker_hints: dict[str, int],
    only_files: list[str] | None,
) -> None:
    # Imported here, not at module level, so --help and argument parsing
    # work even on a machine that hasn't installed whisperx/torch/
    # pyannote yet -- matches this file's own "not run from this sandbox"
    # reality; a syntax check shouldn't require a GPU environment.
    import pandas as pd
    import torch
    import whisperx
    from pyannote.audio import Pipeline

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transcript_processing.whisperx_adapter import filter_prompt_leak

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    compute_type = "float16" if device == "cuda" else "int8"

    asr_options = {"temperatures": [0.0], "initial_prompt": INITIAL_PROMPT}

    os.makedirs(output_dir, exist_ok=True)

    print("Loading WhisperX model...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, asr_options=asr_options)

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="de", device=device)

    print("Loading diarization pipeline...")
    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
    if device == "cuda":
        diarize_model.to(torch.device("cuda"))

    if only_files:
        audio_files = sorted(only_files)
    else:
        audio_files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(AUDIO_EXTENSIONS))

    print(f"\nFound {len(audio_files)} audio file(s) in {audio_dir}")
    print(f"Output folder: {output_dir}\n")

    for i, filename in enumerate(audio_files, start=1):
        audio_path = os.path.join(audio_dir, filename)
        stem = os.path.splitext(filename)[0]
        header = f"[{i}/{len(audio_files)}] {filename}"
        print("=" * 80)
        print(header)
        print("=" * 80)

        num_speakers = speaker_hints.get(stem, num_speakers_default)
        if num_speakers is not None:
            print(f"speaker count hint: {num_speakers}")
        else:
            print("speaker count hint: none given -- letting pyannote auto-detect")

        start_time = time.time()
        try:
            audio = whisperx.load_audio(audio_path)

            result = model.transcribe(audio, language="de", batch_size=batch_size)
            print(f"Detected language: {result['language']}")

            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device,
                return_char_alignments=False,
            )

            waveform = torch.from_numpy(audio).unsqueeze(0)
            diarize_kwargs = {"waveform": waveform, "sample_rate": 16000}
            call_kwargs = {}
            if num_speakers is not None:
                call_kwargs["num_speakers"] = num_speakers
            diarization_output = diarize_model(diarize_kwargs, **call_kwargs)
            exclusive = diarization_output.exclusive_speaker_diarization

            rows = []
            for turn, _, speaker in exclusive.itertracks(yield_label=True):
                rows.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
            diarize_segments = pd.DataFrame(rows)

            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Prompt-leak filter -- see module docstring, change 3. Uses
            # the REAL initial_prompt this run actually used, not the
            # weaker no-known-prompt fallback whisperx_adapter.py falls
            # back to when this metadata isn't available.
            kept_segments, dropped_segments = filter_prompt_leak(
                result["segments"], known_prompt_leak=INITIAL_PROMPT
            )
            if dropped_segments:
                print(f"Dropped {len(dropped_segments)} prompt-leak segment(s):")
                for seg in dropped_segments:
                    print(f"  start={seg.get('start')}  text={seg.get('text')!r}")
            result["segments"] = kept_segments

            print("\n--- FINAL SEGMENTS ---")
            for segment in result["segments"]:
                print(segment)

            result["_meta"] = {
                "engine": "whisperx",
                "model": "large-v3",
                "compute_type": compute_type,
                "initial_prompt": INITIAL_PROMPT,
                "num_speakers_hint": num_speakers,
                "prompt_leak_segments_dropped": len(dropped_segments),
                "source_audio_filename": filename,
                "hardened_script_version": "2026-09-20",
            }

            output_file = os.path.join(output_dir, f"{stem}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nSaved to: {output_file}")

            elapsed = time.time() - start_time
            print(f"\n[{i}/{len(audio_files)}] DONE  ({elapsed:.1f}s)  {filename}\n")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[{i}/{len(audio_files)}] FAILED after {elapsed:.1f}s  {filename}")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            print()
            continue

    print("=" * 80)
    print(f"All {len(audio_files)} file(s) processed.")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR,
                         help=f"Directory of audio files to transcribe. Default: {DEFAULT_AUDIO_DIR}")
    parser.add_argument("--output-dir", default=None,
                         help="Where to write {stem}.json files. Default: {audio-dir}/whisperx_transcripts")
    parser.add_argument("--num-speakers", type=int, default=None,
                         help="Batch-wide default speaker count hint, used for any file not listed in "
                              "--speaker-hints-file. Omit entirely to let pyannote auto-detect by default.")
    parser.add_argument("--speaker-hints-file", default=None,
                         help="Path to a JSON object mapping filename stem -> speaker count, "
                              "e.g. {\"lilli_tandem1_4_30_2026_5min\": 2}. Overrides --num-speakers per file.")
    parser.add_argument("--files", nargs="*", default=None,
                         help="Process only these filenames (space-separated) instead of the whole audio-dir.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set in this environment.")
        print('Windows cmd.exe:   set HF_TOKEN=hf_...')
        print('PowerShell:        $env:HF_TOKEN = "hf_..."')
        return 1

    output_dir = args.output_dir or os.path.join(args.audio_dir, "whisperx_transcripts")
    speaker_hints = load_speaker_hints(args.speaker_hints_file)

    run_batch(
        audio_dir=args.audio_dir,
        output_dir=output_dir,
        hf_token=hf_token,
        num_speakers_default=args.num_speakers,
        speaker_hints=speaker_hints,
        only_files=args.files,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
