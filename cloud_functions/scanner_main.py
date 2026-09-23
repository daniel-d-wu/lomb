"""
Cloud Function 1: Scanner
─────────────────────────
Triggered by Cloud Scheduler every 5 minutes.
- Lists audio files in the german_recordings Drive folder
- Compares against the index sheet (column A) to find new files
- For each new file:
    1. Writes row immediately with status=pending (prevents duplicate submissions)
    2. Downloads from Drive, uploads to AssemblyAI, submits job
    3. Updates row with transcript_id and status=submitted
    4. On failure, updates status=failed

Sheet columns: A=filename | B=transcript_id | C=prompt | D=Dan | E=Notes | F=status
"""

import io
import json
import os
import re

import functions_framework
import requests
from google.auth import default
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── CONFIG ────────────────────────────────────────────────────────────────────
ASSEMBLYAI_API_KEY      = os.environ["ASSEMBLYAI_API_KEY"].strip()
WEBHOOK_URL             = os.environ["WEBHOOK_URL"]
DRIVE_AUDIO_FOLDER      = "110lmAYYAuD8ja03CDdifPJCemijCjsA_"
DRIVE_TRANSCRIPT_FOLDER = "1JcKXIMEoZ3YPc0yUZCQ5mP2tm1NVb5is"
SHEET_ID                = "1x7-UqFnta4AMJoH2KawuQE9vtdUj-Zd3e5Q1lpURrKc"
AUDIO_EXTENSIONS        = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
AAI_BASE                = "https://api.assemblyai.com/v2"
AAI_HEADERS             = {"authorization": ASSEMBLYAI_API_KEY}

# ── GOOGLE AUTH ───────────────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

def get_google_services():
    creds, _ = default(scopes=SCOPES)
    creds.refresh(Request())
    drive  = build("drive",  "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)
    return drive, sheets

# ── SHEET HELPERS ─────────────────────────────────────────────────────────────
def get_processed_filenames(sheets) -> set:
    """Return set of filenames already in column A (with or without extension)."""
    result = sheets.spreadsheets().values().get(
        spreadsheetId=SHEET_ID, range="A:A"
    ).execute()
    rows = result.get("values", [])
    stems = set()
    for row in rows:
        if row:
            name = row[0]
            stem = name.rsplit(".", 1)[0] if "." in name else name
            stems.add(stem)
    return stems

def append_pending_row(sheets, stem: str) -> int:
    """
    Write a new row with status=pending immediately on file detection.
    Returns the 1-indexed row number of the new row.
    """
    row  = [stem, "", "auto_pipeline_v1", "TBD", "", "pending"]
    body = {"values": [row]}
    result = sheets.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range="A:F",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
        includeValuesInResponse=True,
    ).execute()
    updated_range = result.get("updates", {}).get("updatedRange", "")
    try:
        row_num = int(updated_range.split("!")[1].split(":")[0][1:])
    except Exception:
        row_num = None
    print(f"  Row written (pending): {stem} -> row {row_num}")
    return row_num

def update_row_submitted(sheets, row_num: int, stem: str, transcript_id: str):
    """Update transcript_id (col B) and status=submitted (col F)."""
    sheets.spreadsheets().values().batchUpdate(
        spreadsheetId=SHEET_ID,
        body={
            "valueInputOption": "RAW",
            "data": [
                {"range": f"B{row_num}", "values": [[transcript_id]]},
                {"range": f"F{row_num}", "values": [["submitted"]]},
            ]
        }
    ).execute()
    print(f"  Row updated (submitted): {stem} -> {transcript_id}")

def update_row_failed(sheets, row_num: int, stem: str, error: str):
    """Update status=failed in col F."""
    sheets.spreadsheets().values().batchUpdate(
        spreadsheetId=SHEET_ID,
        body={
            "valueInputOption": "RAW",
            "data": [
                {"range": f"F{row_num}", "values": [[f"failed: {error[:100]}"]]},
            ]
        }
    ).execute()
    print(f"  Row updated (failed): {stem}")

# ── DRIVE HELPERS ─────────────────────────────────────────────────────────────
def list_audio_files(drive) -> list:
    """List all audio files in the german_recordings folder."""
    results = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=f"'{DRIVE_AUDIO_FOLDER}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
        for f in resp.get("files", []):
            ext = "." + f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
            if ext in AUDIO_EXTENSIONS:
                results.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_from_drive(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

# ── ASSEMBLYAI ────────────────────────────────────────────────────────────────
def parse_speaker_count(filename: str, default: int = 2) -> int:
    stem = filename.rsplit(".", 1)[0]
    match = re.search(r'_(\d+)spk', stem)
    return int(match.group(1)) if match else default

def upload_to_assemblyai(audio_bytes: bytes) -> str:
    resp = requests.post(f"{AAI_BASE}/upload", headers=AAI_HEADERS, data=audio_bytes)
    resp.raise_for_status()
    return resp.json()["upload_url"]

def submit_transcription(upload_url: str, filename: str) -> str:
    speakers = parse_speaker_count(filename)
    payload = {
        "audio_url":                 upload_url,
        "speech_models":             ["universal-3-pro", "universal-2"],
        "speaker_labels":            True,
        "speakers_expected":         speakers,
        "language_detection":        True,
        "keyterms_prompt":           ["aehm", "aeh", "mhm"],
        "webhook_url":               WEBHOOK_URL,
        "webhook_auth_header_name":  "X-Webhook-Secret",
        "webhook_auth_header_value": os.environ.get("WEBHOOK_SECRET", "").strip(),
    }
    resp = requests.post(
        f"{AAI_BASE}/transcript",
        headers={**AAI_HEADERS, "content-type": "application/json"},
        json=payload
    )
    if not resp.ok:
        print(f"AssemblyAI error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()["id"]

# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
@functions_framework.http
def scanner(request):
    print("Scanner triggered")
    drive, sheets = get_google_services()

    processed = get_processed_filenames(sheets)
    all_audio = list_audio_files(drive)
    new_files = [f for f in all_audio
                 if f["name"].rsplit(".", 1)[0] not in processed]

    print(f"Found {len(all_audio)} audio files, {len(new_files)} unprocessed")

    results = []
    for f in new_files:
        name = f["name"]
        stem = name.rsplit(".", 1)[0]
        print(f"Processing: {name}")

        # Step 1: write pending row immediately - prevents duplicate submissions
        row_num = append_pending_row(sheets, stem)

        try:
            audio_bytes   = download_from_drive(drive, f["id"])
            upload_url    = upload_to_assemblyai(audio_bytes)
            transcript_id = submit_transcription(upload_url, name)

            # Step 2: update row with transcript_id and status=submitted
            if row_num:
                update_row_submitted(sheets, row_num, stem, transcript_id)

            results.append({"file": name, "transcript_id": transcript_id, "status": "submitted"})
            print(f"  Submitted -> {transcript_id}")

        except Exception as e:
            if row_num:
                update_row_failed(sheets, row_num, stem, str(e))
            results.append({"file": name, "error": str(e), "status": "failed"})
            print(f"  Failed: {e}")

    return (json.dumps(results), 200, {"Content-Type": "application/json"})
