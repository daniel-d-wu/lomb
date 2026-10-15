# The production service layer: functions the web API and the GPU worker
# call. Everything below is written against seams (StorageRepository,
# BlobStore, LLMProvider, a transcriber, a speaker-resolution service) so the
# same code runs locally (SQLite, a folder, a precomputed transcript) and on
# Google Cloud (Cloud SQL, Cloud Storage, the GPU worker).
#
# lesson_job.py -- one lesson end to end: upload -> transcribe -> identify the
#                  learner -> analyse -> reports, with the "which one is
#                  you?" pause when the voiceprint match isn't confident.
