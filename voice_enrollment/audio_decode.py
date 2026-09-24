"""
audio_decode.py -- the one piece voice_enrollment.py deliberately doesn't
have: turning an arbitrary uploaded audio BLOB (whatever a browser's
MediaRecorder produced -- typically webm/opus, but this doesn't assume
that) into the 16kHz mono float32 waveform every SpeakerAudio/embed() call
in this project expects.

Kept as one small, framework-free function (not baked into enroll_api.py)
so it's independently testable and reusable by the future /analyze
diarization-glue step too -- that step needs the exact same
bytes-in/16kHz-float32-out conversion, just applied to slices of a longer
recording instead of a whole enrollment clip.

Shells out to ffmpeg rather than a Python audio-decoding library on
purpose: ffmpeg already has to be on the machine (WhisperX needs it, and
identify_target_speaker.py's atrim+concat trick already depends on it), so
this doesn't add a new dependency, and it handles whatever container
format a browser's MediaRecorder happens to produce without this project
needing an opinion about which one.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000  # matches voice_enrollment.SAMPLE_RATE and the ECAPA model's expected input


class AudioDecodeError(ValueError):
    """Raised when ffmpeg can't make sense of the uploaded bytes at all --
    e.g. an empty upload, a truncated recording, or a file that isn't
    audio. Distinct from voice_enrollment.py's "too short" rejection,
    which fires on VALID audio that's just not long enough."""


def decode_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """raw_bytes: whatever came in over the wire (webm/opus, mp3, wav,
    m4a -- ffmpeg auto-detects from content, not the filename). Returns
    16kHz mono float32 PCM, ready for SpeakerEmbedder.embed() or
    SpeakerResolutionService.enroll_active()."""
    if not raw_bytes:
        raise AudioDecodeError("empty upload -- no audio data received")

    # Deliberately NOT `with tempfile.NamedTemporaryFile(...) as f: ...` --
    # that keeps Python's own handle open for the file's whole lifetime,
    # which is harmless on Linux/Mac (multiple handles to the same path are
    # fine) but fails on Windows: a second process (ffmpeg) can't open a
    # file Python still holds open, and ffmpeg -i / the output write both
    # errored with "Permission denied" as a result. mkstemp() + closing our
    # own handle immediately, before ffmpeg ever touches the path, avoids
    # that -- cross-platform, not just a Linux workaround.
    src_fd, src_path = tempfile.mkstemp(suffix=".bin")
    dst_fd, dst_path = tempfile.mkstemp(suffix=".wav")
    os.close(dst_fd)  # ffmpeg (-y) creates/overwrites this; we only needed the reserved name

    try:
        with os.fdopen(src_fd, "wb") as src:
            src.write(raw_bytes)
        # src's handle is now closed -- safe for ffmpeg to open on Windows too

        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path, "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise AudioDecodeError(
                f"ffmpeg could not decode the uploaded audio: {result.stderr.decode(errors='replace').strip()[:300]}"
            )
        audio, sr = sf.read(dst_path, dtype="float32")
        assert sr == SAMPLE_RATE
    finally:
        for path in (src_path, dst_path):
            try:
                os.unlink(path)
            except OSError:
                pass  # best-effort cleanup -- a leaked temp file is not worth failing the request over

    if audio.ndim > 1:  # sf.read can hand back (n, channels) even for ac=1 in some containers -- collapse defensively
        audio = audio.mean(axis=1).astype(np.float32)
    return audio


MAX_SECONDS_PER_SPEAKER = 90  # same cap identify_target_speaker.py uses per embedding
MIN_SECONDS_PER_SPEAKER = 3   # below this an ECAPA embedding is unreliable


def extract_speaker_audio(
    audio_path: str,
    turns,
    max_seconds_per_speaker: float = MAX_SECONDS_PER_SPEAKER,
    min_seconds_per_speaker: float = MIN_SECONDS_PER_SPEAKER,
) -> dict[str, np.ndarray]:
    """The /analyze diarization-glue step: slice one recording into each
    diarized speaker's concatenated speech, 16kHz mono float32, ready to
    wrap in voice_enrollment.SpeakerAudio for SpeakerResolutionService.

    turns: any objects with .speaker_id/.start/.end (speaker_filter.Turn).
    Speakers with less than min_seconds_per_speaker of speech are omitted
    from the result, not padded -- the caller sees them missing by label.
    """
    spans: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        spans.setdefault(t.speaker_id, []).append((t.start, t.end))

    result: dict[str, np.ndarray] = {}
    for label, label_spans in spans.items():
        picked, total = [], 0.0
        for start, end in label_spans:
            if total >= max_seconds_per_speaker:
                break
            dur = min(end - start, max_seconds_per_speaker - total)
            if dur <= 0:
                continue
            picked.append((start, start + dur))
            total += dur
        if total < min_seconds_per_speaker:
            continue

        filter_parts = [
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
            for i, (s, e) in enumerate(picked)
        ]
        concat_inputs = "".join(f"[a{i}]" for i in range(len(picked)))
        filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(picked)}:v=0:a=1[out]"

        dst_fd, dst_path = tempfile.mkstemp(suffix=".wav")
        os.close(dst_fd)  # same Windows-safe pattern as decode_audio_bytes()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
                 "-filter_complex", filter_complex, "-map", "[out]",
                 "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
                capture_output=True,
            )
            if proc.returncode != 0:
                raise AudioDecodeError(
                    f"ffmpeg could not slice {label} from {audio_path}: "
                    f"{proc.stderr.decode(errors='replace').strip()[:300]}"
                )
            audio, sr = sf.read(dst_path, dtype="float32")
            assert sr == SAMPLE_RATE
        finally:
            try:
                os.unlink(dst_path)
            except OSError:
                pass
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        result[label] = audio
    return result
