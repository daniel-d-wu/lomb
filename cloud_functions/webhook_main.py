"""
Cloud Function 2: Webhook Receiver
────────────────────────────────────
AssemblyAI calls this URL when a transcript job completes.
- Validates the webhook secret
- Fetches the full transcript JSON from AssemblyAI
- Saves it to the transcript_api_resp Drive folder using personal OAuth credentials
- Finds the row in the index sheet by transcript_id
- Updates status to completed in column F

Sheet columns: A=filename | B=transcript_id | C=prompt | D=Dan | E=Notes | F=status
"""

import io
import json
import os

import functions_framework
import requests
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ── CONFIG ────────────────────────────────────────────────────────────────────
ASSEMBLYAI_API_KEY      = os.environ["ASSEMBLYAI_API_KEY"].strip()
WEBHOOK_SECRET          = os.environ.get("WEBHOOK_SECRET", "").strip()
OAUTH_CLIENT_ID         = os.environ["OAUTH_CLIENT_ID"].strip()
OAUTH_CLIENT_SECRET     = os.environ["OAUTH_CLIENT_SECRET"].strip()
OAUTH_REFRESH_TOKEN     = os.environ["OAUTH_REFRESH_TOKEN"].strip()
DRIVE_TRANSCRIPT_FOLDER = "1JcKXIMEoZ3YPc0yUZCQ5mP2tm1NVb5is"
SHEET_ID                = "1x7-UqFnta4AMJoH2KawuQE9vtdUj-Zd3e5Q1lpURrKc"
AAI_BASE                = "https://api.assemblyai.com/v2"
AAI_HEADERS             = {"authorization": ASSEMBLYAI_API_KEY}

# ── GOOGLE AUTH ───────────────────────────────────────────────────────────────
SA_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

def get_drive_service():
    """Use personal OAuth credentials for Drive (to write files as the user)."""
    creds = Credentials(
        token=None,
        refresh_token=OAUTH_REFRESH_TOKEN,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds)

def get_sheets_service():
    """Use service account for Sheets (already works fine)."""
    creds, _ = default(scopes=SA_SCOPES)
    creds.refresh(Request())
    return build("sheets", "v4", credentials=creds)

# ── SHEET HELPERS ─────────────────────────────────────────────────────────────
def find_row_by_transcript_id(sheets, transcript_id: str) -> int:
    """Find the row number in column B matching the transcript_id."""
    result = sheets.spreadsheets().values().get(
        spreadsheetId=SHEET_ID, range="B:B"
    ).execute()
    rows = result.get("values", [])
    for i, row in enumerate(rows):
        if row and row[0] == transcript_id:
            return i + 1
    return None

def update_status_completed(sheets, row_num: int):
    """Update column F to completed for the given row."""
    sheets.spreadsheets().values().update(
        spreadsheetId=SHEET_ID,
        range=f"F{row_num}",
        valueInputOption="RAW",
        body={"values": [["completed"]]}
    ).execute()
    print(f"  Status updated to completed for row {row_num}")

# ── ASSEMBLYAI ────────────────────────────────────────────────────────────────
def fetch_transcript(transcript_id: str) -> dict:
    resp = requests.get(f"{AAI_BASE}/transcript/{transcript_id}", headers=AAI_HEADERS)
    resp.raise_for_status()
    return resp.json()

# ── DRIVE ─────────────────────────────────────────────────────────────────────
def save_to_drive(drive, transcript_id: str, data: dict) -> str:
    """Save transcript as {transcript_id}.txt to transcript_api_resp folder."""
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(json_bytes), mimetype="text/plain")
    meta  = {"name": f"{transcript_id}.txt", "parents": [DRIVE_TRANSCRIPT_FOLDER]}
    result = drive.files().create(
        body=meta,
        media_body=media,
        fields="id"
    ).execute()
    print(f"  File saved to Drive: {transcript_id}.txt (id: {result['id']})")
    return result["id"]

# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
@functions_framework.http
def webhook_receiver(request):
    """HTTP entry point called by AssemblyAI when a transcript completes."""

    # Validate webhook secret
    if WEBHOOK_SECRET:
        incoming = request.headers.get("X-Webhook-Secret", "").strip()
        if incoming != WEBHOOK_SECRET:
            print("Webhook secret mismatch - rejecting")
            return ("Unauthorized", 401)

    payload = request.get_json(silent=True)
    if not payload:
        return ("Bad request", 400)

    transcript_id = payload.get("transcript_id")
    status        = payload.get("status")
    print(f"Webhook received: transcript_id={transcript_id}, status={status}")

    if status != "completed":
        print(f"Non-completed status '{status}' - skipping")
        return ("OK", 200)

    try:
        drive  = get_drive_service()
        sheets = get_sheets_service()
        data   = fetch_transcript(transcript_id)

        # Save to Drive as {transcript_id}.txt
        save_to_drive(drive, transcript_id, data)

        # Update sheet status to completed
        row_num = find_row_by_transcript_id(sheets, transcript_id)
        if row_num:
            update_status_completed(sheets, row_num)
        else:
            print(f"  Warning: transcript_id {transcript_id} not found in sheet")

        print(f"Pipeline complete for {transcript_id}")
        return ("OK", 200)

    except Exception as e:
        print(f"Error processing transcript {transcript_id}: {e}")
        return (str(e), 500)
