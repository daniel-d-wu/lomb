"""
Manage who may sign in to Lomb (the invites table, schema v1.6). Logs in to
Cloud SQL as the gcloud account, like scripts/cloud_smoke_test.py.

Usage (from lomb_prompts/):
  python scripts/invite.py add someone@gmail.com [--note "tutor friend"]
  python scripts/invite.py remove someone@gmail.com
  python scripts/invite.py list

Removing an invite stops new sign-ins and uploads; it doesn't delete the
person's lessons (they can still be deleted from their account page, or by
deleting the account).
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.oauth2.credentials import Credentials  # noqa: E402

from storage.postgres_repo import PostgresStorageRepository, cloud_sql_connect  # noqa: E402


class TokenOnly(Credentials):  # gcloud's token is short-lived and this run is short; nothing to refresh
    def refresh(self, request):
        pass


def gcloud(*args: str) -> str:
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=True,
                          shell=sys.platform == "win32").stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["add", "remove", "list"])
    ap.add_argument("email", nargs="?")
    ap.add_argument("--note")
    ap.add_argument("--db", default="lomb")
    args = ap.parse_args()
    if args.action != "list" and not args.email:
        ap.error("an email is required")

    expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(minutes=50)
    creds = TokenOnly(gcloud("auth", "print-access-token"), expiry=expiry)
    connect = cloud_sql_connect(gcloud("config", "get", "account"), args.db, credentials=creds)
    repo = PostgresStorageRepository(connect)
    try:
        if args.action == "add":
            repo.add_invite(args.email, args.note)
        elif args.action == "remove":
            repo.remove_invite(args.email)
        for row in repo.list_invites():
            print(f"{row['email']:<40} {row['invited_at'][:10]}  {row['note'] or ''}")
    finally:
        repo.close()
        connect.connector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
