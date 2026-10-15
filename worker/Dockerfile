# syntax=docker/dockerfile:1.7
# Lomb GPU worker: WhisperX + pyannote + the analysis pipeline on an NVIDIA
# L4 (Cloud Run). Built by Cloud Build from the repo root:
#   gcloud builds submit --config worker/cloudbuild.yaml
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    NLTK_DATA=/models/nltk

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv libpython3.12 ffmpeg libsndfile1 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv

WORKDIR /app
COPY worker/requirements.txt worker/requirements.txt
RUN pip install --upgrade pip && pip install -r worker/requirements.txt

# Only the code the worker runs (tests, demos, data and docs stay out).
COPY pipeline pipeline
COPY prompts prompts
COPY providers providers
COPY reporting reporting
COPY service service
COPY storage storage
COPY transcript_processing transcript_processing
COPY voice_enrollment voice_enrollment
COPY worker worker

RUN --mount=type=secret,id=hf_token HF_TOKEN="$(cat /run/secrets/hf_token)" python worker/bake_models.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn worker.main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75"]
