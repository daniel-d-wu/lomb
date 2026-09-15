# The active voice-enrollment slice: record a voice sample, turn it into a
# stored voiceprint, and later auto-recognize the same speaker.
#
# voice_enrollment.py    -- core logic: SpeakerResolutionService, the
#                            match/margin decision, SQLiteVoiceprintRepository.
# speechbrain_embedder.py -- the real SpeakerEmbedder (ECAPA-TDNN via
#                            speechbrain); needs torch/speechbrain installed.
# audio_decode.py         -- ffmpeg-based blob -> 16kHz mono float32 PCM,
#                            the one piece voice_enrollment.py deliberately
#                            doesn't have.
# enroll_api.py           -- the FastAPI app (POST /enroll, GET /status,
#                            GET / for the widget) wiring the above together.
# enroll_widget.html      -- the mic-capture widget enroll_api.py serves.
# identify_target_speaker.py -- a separate, standalone diagnostic script
#                            over Dan's personal corpus (no dependency on
#                            the other files here -- self-contained).
#
# See ../lomb_voice_enrollment_design_v1.md for the design this implements.
