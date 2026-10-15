"""
LessonDeps for Google Cloud (2026-10-14, build steps 3-4): Cloud SQL,
Cloud Storage, OpenAI, and -- on the GPU worker only -- WhisperX. Shared by
the worker and (step 5) the web API so both read the same settings.

Settings come from environment variables, set on the Cloud Run service:

  LOMB_BUCKET          Cloud Storage bucket                    (required)
  LOMB_DB_USER         Cloud SQL IAM user: the service account's email
                       minus '.gserviceaccount.com'            (required)
  LOMB_DB_NAME         database                                 default lomb
  LOMB_DB_INSTANCE     project:region:instance                  default lomb-prod:us-east4:lomb-db
  LOMB_LLM_MODEL       OpenAI model                             default gpt-5.6-luna
  LOMB_REASONING_EFFORT                                         default: the provider's
  OPENAI_API_KEY       from Secret Manager (open-ai-key)
  HF_TOKEN             from Secret Manager (hf-token); worker only

No voiceprints exist in the cloud yet (enrollment is step 6), so
speaker_service is None and every lesson asks "which one is you?".
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.lesson_job import LessonDeps  # noqa: E402


def _secret(name: str) -> str:
    # A secret pasted with a trailing newline or space would fail as a key.
    value = os.environ[name].strip()
    os.environ[name] = value
    return value


def cloud_deps(*, with_transcriber: bool, with_llm: bool = True) -> LessonDeps:
    """The worker needs everything; the web API only the database and storage."""
    from storage.blob_store import GCSBlobStore
    from storage.postgres_repo import INSTANCE, PostgresStorageRepository, cloud_sql_connect

    repo = PostgresStorageRepository(cloud_sql_connect(
        os.environ["LOMB_DB_USER"], os.environ.get("LOMB_DB_NAME", "lomb"),
        instance=os.environ.get("LOMB_DB_INSTANCE", INSTANCE)))
    provider = None
    if with_llm:
        from providers.openai_provider import OpenAIProvider
        _secret("OPENAI_API_KEY")
        provider = OpenAIProvider(model=os.environ.get("LOMB_LLM_MODEL", "gpt-5.6-luna"),
                                  reasoning_effort=os.environ.get("LOMB_REASONING_EFFORT") or None)
    transcriber = None
    if with_transcriber:
        from transcript_processing.whisperx_engine import WhisperXTranscriber
        transcriber = WhisperXTranscriber(_secret("HF_TOKEN"))
    return LessonDeps(repo=repo, blobs=GCSBlobStore(os.environ["LOMB_BUCKET"]), transcriber=transcriber,
                      provider=provider, speaker_service=None,
                      llm_model=provider.model if provider else "none",
                      llm_settings=provider.generation_config if provider else {})
