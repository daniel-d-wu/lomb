"""
run_dev_server.py -- local manual-testing entry point for the active
voice-enrollment slice (enroll_api.py + enroll_widget.html).

NOT production wiring. Defaults to a placeholder embedder instead of the
real speechbrain_embedder.EcapaSpeakerEmbedder, so you can test the
actual mechanics (record -> upload -> ffmpeg decode -> enroll_active() ->
SQLite storage) with your own microphone, without installing
torch/speechbrain first.

Pass --real-embedder to use the real EcapaSpeakerEmbedder instead. You
need this for any voiceprint you intend to cross-reference against real
conversation audio later (see tests/test_speaker_identification.py) --
the fake embedder's output has no relationship to actual voice
characteristics, so two recordings of the same person produce unrelated
"embeddings" under it. No code changes needed either way, since
SpeakerResolutionService only depends on the SpeakerEmbedder interface.

Storage is a real (not temp) SQLite file, voiceprints_dev.sqlite, next to
this script -- so state persists across restarts. Delete that file to
reset from scratch, or point tests/test_speaker_identification.py's
--db-path at it directly to test against whatever you enroll here.

Prerequisites:
  pip install fastapi uvicorn numpy soundfile python-multipart
  ffmpeg installed and on PATH (test with: ffmpeg -version)
  --real-embedder additionally needs: pip install speechbrain torch

Run:
  python run_dev_server.py                  # fake embedder, plumbing only
  python run_dev_server.py --real-embedder   # real voice matching
Then open http://127.0.0.1:8000/ in a real browser and use your real mic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# repo root on sys.path -- enroll_api.py and voice_enrollment.py both moved
# into voice_enrollment/ post-reorg, a sibling of this file's own new
# directory (scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import uvicorn

from voice_enrollment.enroll_api import create_app
from voice_enrollment.voice_enrollment import SpeakerEmbedder, SQLiteVoiceprintRepository


class FixedLengthFakeEmbedder(SpeakerEmbedder):
    """Deterministic placeholder -- NOT a real voice embedding. Good enough
    to prove enrollment's plumbing works end-to-end; not good enough to
    trust for actual speaker-matching quality. See module docstring."""

    model_id = "fake-fixed-length-v1"

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(waveform.tobytes())) % (2**32))
        vec = rng.normal(size=192).astype(np.float32)
        vec[0] += float(waveform.mean())
        vec[1] += float(waveform.std())
        return vec


# Path(__file__)-relative, not a bare relative string -- so the db always
# lands next to this script regardless of which directory it's launched
# from, matching what the module docstring above already promises.
DB_PATH = Path(__file__).resolve().parent / "voiceprints_dev.sqlite"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--real-embedder", action="store_true",
        help="Use the real speechbrain ECAPA embedder instead of the fixed-length fake one.",
    )
    args = parser.parse_args()

    if args.real_embedder:
        from voice_enrollment.speechbrain_embedder import EcapaSpeakerEmbedder
        embedder: SpeakerEmbedder = EcapaSpeakerEmbedder()
    else:
        embedder = FixedLengthFakeEmbedder()

    app = create_app(embedder, SQLiteVoiceprintRepository(str(DB_PATH)))

    print("Starting dev server at http://127.0.0.1:8000/ (Ctrl+C to stop)")
    if args.real_embedder:
        print(f"Using the REAL speechbrain ECAPA embedder -- voiceprints written to {DB_PATH} are usable for real matching.")
    else:
        print("Using a FAKE embedder -- this tests enrollment mechanics, not real voice matching. Pass --real-embedder for a usable voiceprint.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
