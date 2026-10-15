"""
Runs once while the worker image is built: downloads every model the worker
uses into the image, so an instance starting up loads them from disk instead
of pulling ~4 GB from Hugging Face (Cloud Run's disk lives in memory). The
image sets HF_HUB_OFFLINE=1, so if anything here was missed the worker fails
loudly at startup instead of silently downloading.

Loads the real WhisperXTranscriber on CPU -- the exact code path the worker
runs -- rather than a hand-kept list of repos that could drift from it.
Needs HF_TOKEN (pyannote's diarization model is gated); the build passes it
as a build secret, so it is not stored in any image layer.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk  # noqa: E402

from transcript_processing.whisperx_engine import WhisperXTranscriber  # noqa: E402

nltk.download("punkt_tab", download_dir=os.environ["NLTK_DATA"], quiet=True)  # used by whisperx alignment
WhisperXTranscriber(os.environ["HF_TOKEN"].strip(), device="cpu")
print("models baked")
