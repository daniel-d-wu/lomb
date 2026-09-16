"""
test_speaker_identification.py -- generic accuracy test for the
voice_enrollment/enroll_api product path: enroll ANY user's voice once,
then check whether SpeakerResolutionService.resolve() correctly picks
them out across a batch of already-diarized, multi-speaker recordings.

This is the generic counterpart to identify_target_speaker.py.
identify_target_speaker.py hardcodes Dan's own session names, name-
address evidence, and a diff against Dan's hand-filled
TARGET_SPEAKER_BY_SESSION dict, because it audits one specific personal
corpus -- that hardcoding is about WHICH FILES belong to Dan, not a
limitation of the underlying approach. This script has none of that:
--user-id, --transcript-dir, and --audio-dir are all arguments, and
ground truth (if you have it) is a plain {session: speaker_label} JSON
file you provide, not parsed out of anyone's pipeline code.

Two other differences from identify_target_speaker.py, both deliberate:
  - It calls the REAL SpeakerResolutionService.resolve() from
    voice_enrollment.py -- the exact function and threshold policy
    (MATCH_THRESHOLD, MARGIN_THRESHOLD) that runs in production --
    instead of reimplementing its own cosine-matching loop. A pass here
    is a real accuracy signal, not a proxy for one.
  - The reference voiceprint comes from the real enrollment path
    (SQLiteVoiceprintRepository, the same store enroll_api.py's
    POST /enroll writes to), not a one-shot average computed fresh on
    every run. --enroll-from lets you seed it from a clip in the same
    invocation; drop that flag on later runs to reuse the stored one.

What IS duplicated from identify_target_speaker.py, unavoidably: the
WhisperX-turn-extraction and ffmpeg atrim+concat speaker-audio-decode
logic, since that script's versions read its own REPO_DIR-relative
module constants tied to Dan's corpus layout rather than taking a path
argument. Reimplemented here in parameterized form; the diarization
logic itself is unchanged.

PREREQUISITE: --user-id must already be enrolled (via POST /enroll,
or pass --enroll-from to enroll from a clip in this same run).

USAGE
  # Enroll from a clean sample clip, then test against every session
  # with both a transcript and a matching audio file:
  python test_speaker_identification.py --user-id dan \
      --db-path voiceprints.sqlite \
      --transcript-dir whisperx_transcripts --audio-dir /path/to/audio \
      --enroll-from dan_sample.wav

  # Already enrolled -- just test, scoring against known ground truth:
  python test_speaker_identification.py --user-id dan \
      --db-path voiceprints.sqlite \
      --transcript-dir whisperx_transcripts --audio-dir /path/to/audio \
      --ground-truth ground_truth.json

ground_truth.json format: {"session_name": "SPEAKER_00", ...} -- the
diarizer's label for --user-id in that session, whatever you already
know it to be.

OUTPUT
  Prints a per-session table (identified speaker, similarity, margin,
  correct/wrong/flagged against ground truth if given) and writes
  --out (default speaker_identification_report.json) with full detail.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# repo root on sys.path -- voice_enrollment.py lives in voice_enrollment/,
# a sibling of this file's own directory (tests/), same convention as
# test_voice_enrollment.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_enrollment.audio_decode import decode_audio_bytes
from voice_enrollment.speechbrain_embedder import EcapaSpeakerEmbedder
from voice_enrollment.voice_enrollment import (
    SAMPLE_RATE,
    SpeakerAudio,
    SpeakerResolutionService,
    SQLiteVoiceprintRepository,
)

AUDIO_EXTENSIONS = [".m4a", ".mp3", ".wav", ".webm", ".ogg", ".flac", ".mp4"]
MAX_SECONDS_PER_SPEAKER = 90  # cap on concatenated speech used per embedding
MIN_SECONDS_PER_SPEAKER = 3   # below this, a speaker's embedding is unreliable -- skip


def fix_trailing_speaker_flips(words, gap_threshold=0.5, max_island_len=2):
    """Folds short, likely-misdiarized speaker islands back into the
    surrounding speaker when the gap on one side is small. Same heuristic
    identify_target_speaker.py uses -- generic diarization cleanup, not
    corpus-specific, just not importable from that file without dragging
    in its module-level constants."""
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


def speaker_turns(transcript_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return {speaker_label: [(turn_start, turn_end), ...]} for the whole
    session, from a WhisperX-format transcript JSON."""
    with open(transcript_path, encoding="utf-8") as f:
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
    """ffmpeg atrim+concat one speaker's turns (up to max_seconds total)
    into 16kHz mono float32 PCM, without decoding the whole file once per
    turn."""
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
        filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{idx}]")
        concat_inputs.append(f"[a{idx}]")
    filter_complex = ";".join(filter_parts) + ";" + "".join(concat_inputs) + f"concat=n={len(picked)}:v=0:a=1[out]"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            tmp_path,
        ]
        subprocess.run(cmd, check=True)
        audio, sr = sf.read(tmp_path, dtype="float32")
        assert sr == SAMPLE_RATE
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return audio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-id", required=True, help="Whichever user_id the voiceprint is/will be stored under")
    ap.add_argument("--db-path", required=True, type=Path, help="SQLite voiceprint store (same file enroll_api.py writes to)")
    ap.add_argument("--transcript-dir", required=True, type=Path, help="Directory of WhisperX-format {session}.json transcripts")
    ap.add_argument("--audio-dir", required=True, type=Path, help="Directory of matching original audio files, named {session}.<ext>")
    ap.add_argument("--sessions", nargs="*", default=None, help="Only check these sessions (default: every session with a transcript)")
    ap.add_argument("--ground-truth", type=Path, default=None, help="JSON file: {session: correct_speaker_label} -- enables accuracy scoring")
    ap.add_argument("--enroll-from", type=Path, default=None, help="Audio clip to enroll --user-id from before testing (any format ffmpeg reads)")
    ap.add_argument("--force-enroll", action="store_true", help="Overwrite an existing voiceprint for --user-id when using --enroll-from")
    ap.add_argument("--out", type=Path, default=Path("speaker_identification_report.json"))
    args = ap.parse_args()

    embedder = EcapaSpeakerEmbedder()
    repo = SQLiteVoiceprintRepository(args.db_path)
    service = SpeakerResolutionService(embedder, repo)

    if args.enroll_from:
        waveform = decode_audio_bytes(args.enroll_from.read_bytes())
        service.enroll_active(args.user_id, waveform, force=args.force_enroll)
        print(f"Enrolled {args.user_id!r} from {args.enroll_from} ({len(waveform) / SAMPLE_RATE:.1f}s)")

    if repo.get(args.user_id) is None:
        raise SystemExit(f"user_id {args.user_id!r} has no voiceprint on file -- enroll first (see --enroll-from)")

    ground_truth = json.loads(args.ground_truth.read_text()) if args.ground_truth else {}
    sessions = args.sessions or sorted(p.stem for p in args.transcript_dir.glob("*.json"))

    results = []
    correct = wrong = flagged = 0
    print(f"\n{'session':<28} {'identified':<12} {'expected':<10} {'result':<9} {'sim':>6} {'margin':>7}")
    print("-" * 80)
    for session in sessions:
        audio_path = find_audio_file(args.audio_dir, session)
        if audio_path is None:
            print(f"SKIP {session}: no audio file found in {args.audio_dir}")
            continue

        turns = speaker_turns(args.transcript_dir / f"{session}.json")
        speakers = []
        for label, spk_turns in turns.items():
            audio = decode_speaker_audio(audio_path, spk_turns, MAX_SECONDS_PER_SPEAKER)
            if audio is not None:
                speakers.append(SpeakerAudio(label, audio))
        if not speakers:
            print(f"SKIP {session}: no speaker had enough audio to embed")
            continue

        resolution = service.resolve(args.user_id, speakers)
        expected = ground_truth.get(session)
        best = resolution.candidates[0] if resolution.candidates else None
        margin = (best.similarity - resolution.candidates[1].similarity) if best and len(resolution.candidates) > 1 else None

        if resolution.needs_manual_confirmation:
            outcome = "flagged"
            if expected:
                flagged += 1
        elif expected:
            outcome = "correct" if resolution.resolved_speaker_label == expected else "WRONG"
            if outcome == "correct":
                correct += 1
            else:
                wrong += 1
        else:
            outcome = "resolved"

        results.append({
            "session": session,
            "resolved_speaker_label": resolution.resolved_speaker_label,
            "expected": expected,
            "reason": resolution.reason,
            "outcome": outcome,
            "candidates": [{"speaker_label": c.speaker_label, "similarity": round(c.similarity, 4)} for c in resolution.candidates],
        })

        sim_str = f"{best.similarity:.3f}" if best else "  --"
        margin_str = f"{margin:.3f}" if margin is not None else "  --"
        print(f"{session:<28} {str(resolution.resolved_speaker_label):<12} {str(expected):<10} {outcome:<9} {sim_str:>6} {margin_str:>7}")

    if ground_truth:
        scored = correct + wrong + flagged
        print(f"\n{correct}/{scored} correct auto-matches, {wrong} WRONG, {flagged} flagged for manual review")
        if wrong:
            print("WRONG means a confident auto-match that disagreed with ground truth -- the failure mode the threshold policy exists to prevent. Investigate before trusting these thresholds in production.")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
