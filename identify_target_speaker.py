"""
Automated Dan-speaker identification for WhisperX + pyannote diarization
output -- a voiceprint-based check on (and eventual replacement for) the
manual TARGET_SPEAKER_BY_SESSION dict hand-filled in
build_alternative_errors_json.py.

WHY THIS EXISTS
  WhisperX's diarization is just pyannote.audio underneath, and pyannote's
  open-source pipeline does unsupervised clustering only: it hands back
  generic per-file labels (SPEAKER_00, SPEAKER_01, ...) with NO identity
  and NO stability across recordings -- confirmed empirically in this repo
  (see the "IMPORTANT" note above TARGET_SPEAKER_BY_SESSION: Dan is
  SPEAKER_00 in benny_italki1/2/4/5/6 but SPEAKER_01 in benny_italki3).
  So far that's been resolved by hand, reading each session's transcript
  for content clues (self-intro, being addressed by name, a Q&A
  direction). That mostly works, but it silently failed once already:
  natasja_italki6_4_14_2026 was hand-labeled "HIGH" confidence on a Q&A-
  direction heuristic (whoever answers a personal question is assumed to
  be Dan), and it was backwards -- confirmed by both a WPM check (the
  labeled "Dan" was faster than the tutor in every name-verified session
  in this corpus, no exceptions) and a filler-word check. Getting this
  wrong silently swaps "Dan's errors" for "the tutor's errors" for an
  entire session, with nothing downstream able to catch it.

  Real voice enrollment/identification (pyannoteAI's paid /voiceprint +
  /identify cloud endpoints) would solve this properly, but that's a
  separate paid product from the open-source pyannote.audio pip package
  this pipeline already uses. This script is the do-it-yourself version
  of the same idea, built entirely from things you already have:

    1. Build ONE reference voice embedding for Dan by averaging
       ECAPA-TDNN speaker embeddings (speechbrain/spkrec-ecapa-voxceleb --
       the same embedding family pyannote's own clustering step uses
       internally) extracted from his speech in a small set of sessions
       where TARGET_SPEAKER_BY_SESSION is backed by an EXPLICIT, unambiguous
       signal: a self-introduction ("Mein Name ist Daniel") or a direct
       name-address in one clear direction ("Hi, Benny" / "Hallo Daniel").
       See REFERENCE_SESSIONS below -- deliberately NOT sessions whose only
       evidence was the weaker Q&A-direction heuristic, since that's the
       exact heuristic that already produced one wrong label.
    2. For any session, embed each pyannote speaker cluster's own speech
       and cosine-match it against the Dan reference. Whichever cluster is
       closest is Dan. The similarity MARGIN (best match minus runner-up)
       is reported alongside every call: a small margin means "flag for
       manual review", not "trust blindly" -- same spirit as the existing
       HIGH/MEDIUM/LOW confidence comments, just computed instead of eyeballed.
    3. Every session's automated call is diffed against the existing
       TARGET_SPEAKER_BY_SESSION entry (parsed straight out of
       build_alternative_errors_json.py, not hand-copied, so the two files
       can't drift out of sync) and any disagreement is printed loudly.

THIS WON'T RUN IN THE CLOUD SANDBOX THIS WAS WRITTEN IN -- no GPU, no
torch/speechbrain, and no audio files there. Run it on your own machine,
same place you already run WhisperX itself.

REQUIREMENTS
  pip install speechbrain torchaudio torch soundfile
  ffmpeg on PATH (WhisperX already needs this, so you almost certainly
  have it) -- used here to decode .m4a/.mp3/etc. to 16kHz mono PCM.

INPUTS
  - whisperx_transcripts/{session}.json           (already in this repo)
  - the ORIGINAL AUDIO per session, downloaded locally from the
    google_recordings Drive folder (pipeline_README.md's naming
    convention: {session}.m4a) -- embeddings need real audio, word
    timestamps alone aren't enough. Point --audio-dir at wherever you've
    synced/downloaded them.

USAGE
  # Build the Dan reference embedding and audit every session that has
  # both a transcript and a local audio file:
  python identify_target_speaker.py --audio-dir /path/to/audio

  # Just check the sessions you're suspicious of:
  python identify_target_speaker.py --audio-dir /path/to/audio \
      --sessions natasja_italki6_4_14_2026 anja_italki3_5_5_2026

  # Cache the reference embedding (slow step) and reuse it next run:
  python identify_target_speaker.py --audio-dir /path/to/audio \
      --save-reference dan_reference_embedding.npy
  python identify_target_speaker.py --audio-dir /path/to/audio \
      --load-reference dan_reference_embedding.npy

OUTPUT
  Prints a per-session table (identified speaker, similarity, margin,
  agreement with the hand-labeled dict) and writes
  speaker_identification_report.json with the full detail, including
  every session flagged for manual review (margin below --flag-margin,
  default 0.10 on cosine similarity) or in outright disagreement with
  TARGET_SPEAKER_BY_SESSION.
"""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
WHISPERX_TRANSCRIPT_DIR = REPO_DIR / "whisperx_transcripts"
TARGET_SPEAKER_SOURCE = REPO_DIR / "build_alternative_errors_json.py"
AUDIO_EXTENSIONS = [".m4a", ".mp3", ".wav", ".webm", ".ogg", ".flac", ".mp4"]

SAMPLE_RATE = 16000          # what the ECAPA-TDNN model expects
MAX_SECONDS_PER_SPEAKER = 90  # cap on concatenated speech used per embedding
MIN_SECONDS_PER_SPEAKER = 3   # below this, an embedding is unreliable -- warn

# ---------------------------------------------------------------------
# Sessions used to build Dan's reference embedding -- deliberately
# restricted to entries in TARGET_SPEAKER_BY_SESSION whose comment cites
# an explicit self-introduction or a direct, one-directional name-address,
# NOT the weaker Q&A-direction heuristic (that's the one that already
# produced a wrong label once -- see natasja_italki6_4_14_2026). Spans
# four different tutors so the reference isn't accidentally keyed to one
# recording setup/microphone.
# ---------------------------------------------------------------------
REFERENCE_SESSIONS = [
    "anja_italki1_4_28_2026",     # VERY HIGH -- "Mein Name ist Daniel und ich komme... Texas" (self-intro)
    "anja_italki6_8_24_2026",     # HIGH -- Dan addresses "Anja!"
    "leonie_tandem1_4_18_2026",   # HIGH -- self-intro in Chinese ("wo jiao Daniel")
    "leonie_tandem2_4_22_2026",   # HIGH -- "Hey Leonie" (Dan addressing tutor)
    "benny_italki4_8_24_2026",    # HIGH -- "Hi, Benny" (Dan addressing tutor)
    "benny_italki5_9_1_2026",     # HIGH -- "Hi, hi, Benni"
    "benny_italki6_9_8_2026",     # HIGH -- "Hi, Benny"
    "natasja_italki5_4_10_2026",  # HIGH -- tutor addresses "Daniel." directly
    "natasja_italki7_4_16_2026",  # HIGH -- tutor addresses "Hallo Daniel, guten Morgen"
    "natasja_italki8_4_21_2026",  # HIGH -- Dan addresses "Natasja."; tutor addresses "Hi, Dani" back
    "dirk_italki2_5_21_2026",     # HIGH -- Dan addresses "Hi, Dirk"
]


def load_target_speaker_dict() -> dict[str, str]:
    """
    Parse TARGET_SPEAKER_BY_SESSION straight out of build_alternative_errors_json.py
    via ast, rather than importing that module (which pulls in the OpenAI
    provider and the rest of the error-classification pipeline) or hand-
    copying the dict (which drifts). Comments are fine -- they're not part
    of the AST.
    """
    tree = ast.parse(TARGET_SPEAKER_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TARGET_SPEAKER_BY_SESSION":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TARGET_SPEAKER_BY_SESSION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find TARGET_SPEAKER_BY_SESSION in {TARGET_SPEAKER_SOURCE}")


def fix_trailing_speaker_flips(words, gap_threshold=0.5, max_island_len=2):
    """Same heuristic used in the WPM/pause comparison script -- folds short,
    likely-misdiarized speaker islands back into the surrounding speaker
    when the gap on one side is small. Keeps turn boundaries here
    consistent with the rest of the analysis in this project."""
    i = 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and words[j + 1]["speaker"] == words[i]["speaker"]:
            j += 1
        run = words[i:j + 1]
        if i > 0 and words[i - 1]["speaker"] != run[0]["speaker"]:
            gap_before = run[0]["start"] - words[i - 1]["end"]
            if gap_before < gap_threshold:
                prev_speaker = words[i - 1]["speaker"]
                if len(run) <= max_island_len:
                    gap_after = words[j + 1]["start"] - run[-1]["end"] if j + 1 < len(words) else None
                    if gap_after is None or gap_after >= gap_threshold:
                        for w in run:
                            w["speaker"] = prev_speaker
                else:
                    for k in range(min(max_island_len, len(run) - 1)):
                        internal_gap = run[k + 1]["start"] - run[k]["end"]
                        if internal_gap >= gap_threshold:
                            for w in run[:k + 1]:
                                w["speaker"] = prev_speaker
                            break
        i = j + 1
    return words


def speaker_turns(session: str) -> dict[str, list[tuple[float, float]]]:
    """Return {speaker_label: [(turn_start, turn_end), ...]} for the WHOLE
    session (no 5-minute cap -- more audio makes a better embedding)."""
    with open(WHISPERX_TRANSCRIPT_DIR / f"{session}.json", encoding="utf-8") as f:
        result = json.load(f)
    words = [
        w for seg in result.get("segments", [])
        for w in seg.get("words", [])
        if "speaker" in w and "start" in w and "end" in w and w.get("word")
    ]
    words = fix_trailing_speaker_flips(words)

    turns: dict[str, list[tuple[float, float]]] = {}
    cur_speaker, cur_start, cur_end = None, None, None
    def flush():
        if cur_speaker is not None:
            turns.setdefault(cur_speaker, []).append((cur_start, cur_end))
    for w in words:
        if w["speaker"] != cur_speaker:
            flush()
            cur_speaker, cur_start, cur_end = w["speaker"], w["start"], w["end"]
        else:
            cur_end = w["end"]
    flush()
    return turns


def find_audio_file(audio_dir: Path, session: str) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        p = audio_dir / f"{session}{ext}"
        if p.exists():
            return p
    return None


def decode_speaker_audio(audio_path: Path, turns: list[tuple[float, float]], max_seconds: float) -> np.ndarray | None:
    """
    Use ffmpeg to pull out just this speaker's turns (concatenated, up to
    max_seconds total) as 16kHz mono float32 PCM. Building an ffmpeg
    filter_complex trim+concat graph avoids decoding the whole file once
    per turn.
    """
    import soundfile as sf

    picked, total = [], 0.0
    for start, end in turns:
        if total >= max_seconds:
            break
        dur = min(end - start, max_seconds - total)
        if dur <= 0:
            continue
        picked.append((start, start + dur))
        total += dur
    if total < MIN_SECONDS_PER_SPEAKER:
        return None

    filter_parts = []
    concat_inputs = []
    for idx, (start, end) in enumerate(picked):
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{idx}]"
        )
        concat_inputs.append(f"[a{idx}]")
    filter_complex = ";".join(filter_parts) + ";" + "".join(concat_inputs) + f"concat=n={len(picked)}:v=0:a=1[out]"

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            tmp.name,
        ]
        subprocess.run(cmd, check=True)
        audio, sr = sf.read(tmp.name, dtype="float32")
        assert sr == SAMPLE_RATE
    return audio


class EmbeddingModel:
    """Thin wrapper so the rest of the script doesn't care which speechbrain
    API generation is installed (the pretrained-model import path changed
    between speechbrain versions)."""

    def __init__(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # older speechbrain
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(audio).unsqueeze(0))
        return emb.squeeze().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_reference_embedding(model: EmbeddingModel, audio_dir: Path, existing_map: dict[str, str]) -> np.ndarray:
    embeddings = []
    for session in REFERENCE_SESSIONS:
        audio_path = find_audio_file(audio_dir, session)
        if audio_path is None:
            print(f"  [reference] SKIP {session}: no audio file found in {audio_dir}")
            continue
        turns = speaker_turns(session)
        dan_label = existing_map.get(session)
        if dan_label not in turns:
            print(f"  [reference] SKIP {session}: speaker label {dan_label!r} not found in transcript")
            continue
        audio = decode_speaker_audio(audio_path, turns[dan_label], MAX_SECONDS_PER_SPEAKER)
        if audio is None:
            print(f"  [reference] SKIP {session}: less than {MIN_SECONDS_PER_SPEAKER}s of speech for {dan_label}")
            continue
        embeddings.append(model.embed(audio))
        print(f"  [reference] OK {session} ({dan_label}, {len(audio)/SAMPLE_RATE:.1f}s)")
    if not embeddings:
        raise RuntimeError("Could not build any reference embeddings -- check --audio-dir")
    return np.mean(np.stack(embeddings), axis=0)


def identify_session(model: EmbeddingModel, audio_dir: Path, session: str, dan_reference: np.ndarray) -> dict | None:
    audio_path = find_audio_file(audio_dir, session)
    if audio_path is None:
        print(f"SKIP {session}: no audio file found in {audio_dir}")
        return None
    turns = speaker_turns(session)
    scores = {}
    for speaker, spk_turns in turns.items():
        audio = decode_speaker_audio(audio_path, spk_turns, MAX_SECONDS_PER_SPEAKER)
        if audio is None:
            continue
        emb = model.embed(audio)
        scores[speaker] = cosine(emb, dan_reference)
    if not scores:
        print(f"SKIP {session}: no speaker had enough audio to embed")
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_speaker, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else best_score
    return {
        "session": session,
        "scores": scores,
        "identified_dan_speaker": best_speaker,
        "similarity": round(best_score, 4),
        "margin": round(margin, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True, type=Path, help="Directory of local audio files, named {session}.m4a etc.")
    ap.add_argument("--sessions", nargs="*", default=None, help="Only check these sessions (default: every session with a transcript)")
    ap.add_argument("--flag-margin", type=float, default=0.10, help="Cosine-similarity margin below which a call is flagged for manual review")
    ap.add_argument("--save-reference", type=Path, default=None, help="Save the computed Dan reference embedding here (.npy) for reuse")
    ap.add_argument("--load-reference", type=Path, default=None, help="Load a previously saved reference embedding instead of recomputing it")
    ap.add_argument("--out", type=Path, default=REPO_DIR / "speaker_identification_report.json")
    args = ap.parse_args()

    existing_map = load_target_speaker_dict()
    model = EmbeddingModel()

    if args.load_reference:
        dan_reference = np.load(args.load_reference)
        print(f"Loaded Dan reference embedding from {args.load_reference}")
    else:
        print(f"Building Dan reference embedding from {len(REFERENCE_SESSIONS)} name-verified sessions...")
        dan_reference = build_reference_embedding(model, args.audio_dir, existing_map)
        if args.save_reference:
            np.save(args.save_reference, dan_reference)
            print(f"Saved reference embedding to {args.save_reference}")

    sessions = args.sessions or sorted(p.stem for p in WHISPERX_TRANSCRIPT_DIR.glob("*.json"))

    results = []
    print(f"\n{'session':<28} {'identified':<12} {'existing':<10} {'agree?':<7} {'sim':>6} {'margin':>7}")
    print("-" * 80)
    for session in sessions:
        result = identify_session(model, args.audio_dir, session, dan_reference)
        if result is None:
            continue
        existing = existing_map.get(session)
        agree = (existing == result["identified_dan_speaker"]) if existing else None
        result["existing_mapping"] = existing
        result["agrees_with_existing"] = agree
        result["flagged_low_margin"] = result["margin"] < args.flag_margin
        results.append(result)
        agree_str = "—" if agree is None else ("yes" if agree else "**NO**")
        print(f"{session:<28} {result['identified_dan_speaker']:<12} {existing or '—':<10} {agree_str:<7} "
              f"{result['similarity']:>6.3f} {result['margin']:>7.3f}"
              f"{'  <-- LOW MARGIN' if result['flagged_low_margin'] else ''}")

    disagreements = [r for r in results if r["agrees_with_existing"] is False]
    low_margin = [r for r in results if r["flagged_low_margin"]]
    print(f"\n{len(results)} sessions checked, {len(disagreements)} disagree with TARGET_SPEAKER_BY_SESSION, "
          f"{len(low_margin)} flagged for manual review (margin < {args.flag_margin}).")
    if disagreements:
        print("DISAGREEMENTS (fix these in build_alternative_errors_json.py after listening to confirm):")
        for r in disagreements:
            print(f"  {r['session']}: dict says {r['existing_mapping']}, voice match says {r['identified_dan_speaker']} "
                  f"(sim={r['similarity']}, margin={r['margin']})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "reference_sessions": REFERENCE_SESSIONS,
            "flag_margin_threshold": args.flag_margin,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
