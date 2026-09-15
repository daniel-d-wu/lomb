"""
run_dev_server.py -- local manual-testing entry point for the active
voice-enrollment slice (enroll_api.py + enroll_widget.html).

NOT production wiring. Uses a placeholder embedder instead of the real
speechbrain_embedder.EcapaSpeakerEmbedder, so you can test the actual
mechanics (record -> upload -> ffmpeg decode -> enroll_active() -> SQLite
storage) with your own microphone, without installing torch/speechbrain
first. Once you want to test real voice-matching quality (not just that
enrollment mechanically works), swap FixedLengthFakeEmbedder below for
speechbrain_embedder.EcapaSpeakerEmbedder -- no other code changes needed,
since SpeakerResolutionService only depends on the SpeakerEmbedder
interface.

Storage is a real (not temp) SQLite file, voiceprints_dev.sqlite, next to
this script -- so state persists across restarts. Delete that file to
reset from scratch.

Prerequisites:
  pip install fastapi uvicorn numpy soundfile python-multipart
  ffmpeg installed and on PATH (test with: ffmpeg -version)

Run:
  python run_dev_server.py
Then open http://127.0.0.1:8000/ in a real browser and use your real mic.
"""

from __future__ import annotations

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

app = create_app(FixedLengthFakeEmbedder(), SQLiteVoiceprintRepository(str(DB_PATH)))

if __name__ == "__main__":
    print("Starting dev server at http://127.0.0.1:8000/ (Ctrl+C to stop)")
    print("Using a FAKE embedder -- this tests enrollment mechanics, not real voice matching.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
