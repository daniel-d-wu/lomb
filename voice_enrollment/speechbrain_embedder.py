"""
speechbrain_embedder.py -- the one concrete SpeakerEmbedder implementation,
kept in its own file (not inside voice_enrollment.py) specifically so that
importing voice_enrollment.py -- and testing SpeakerResolutionService's
threshold/enrollment logic -- never requires speechbrain/torch to be
installed. Import THIS module only where you actually have those
dependencies (i.e. the real backend process, not the unit tests).

Uses the same speechbrain/spkrec-ecapa-voxceleb model
identify_target_speaker.py uses for Dan's personal-corpus diagnostics, on
purpose -- see voice_enrollment.py's "EMBEDDING MODEL" docstring section
for why keeping these on the same model matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# repo root on sys.path -- makes the dotted import below resolve whether
# this file is run directly or imported as voice_enrollment.speechbrain_embedder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_enrollment.voice_enrollment import SpeakerEmbedder


class EcapaSpeakerEmbedder(SpeakerEmbedder):
    model_id = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # older speechbrain
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            emb = self._model.encode_batch(torch.from_numpy(waveform).unsqueeze(0))
        return emb.squeeze().cpu().numpy()
