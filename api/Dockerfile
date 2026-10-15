# Lomb web API + app page (CPU, Cloud Run). Built by Cloud Build from the
# repo root:  gcloud builds submit --config api/cloudbuild.yaml
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080

# ffprobe measures each uploaded lesson's length
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

COPY pipeline pipeline
COPY prompts prompts
COPY providers providers
COPY reporting reporting
COPY service service
COPY storage storage
COPY transcript_processing transcript_processing
COPY voice_enrollment voice_enrollment
COPY api api

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
