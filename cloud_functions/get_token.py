"""
Run this once locally to generate an OAuth refresh token for Google Drive access.
It will open a browser window asking you to authorize access to your Google Drive.
Once authorized, it prints the refresh token which you store in Secret Manager.
"""

import json
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_SECRET_FILE = "oauth_client.json"

flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
creds = flow.run_local_server(port=0)

print("\n========================================")
print("REFRESH TOKEN (copy this):")
print(creds.refresh_token)
print("========================================\n")
