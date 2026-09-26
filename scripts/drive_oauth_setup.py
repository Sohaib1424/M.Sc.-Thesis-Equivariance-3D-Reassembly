#!/usr/bin/env python
"""
One-time OAuth setup, to be run ON YOUR OWN MACHINE (it opens a browser).

    pip install google-auth-oauthlib
    python -m scripts.drive_oauth_setup --client_secrets client_secret.json

Prints a JSON blob to paste into a Kaggle Secret. Unlike a service account,
the resulting credentials act as YOU, so uploaded files are owned by your
account and count against your normal storage -- which is what makes this work
where a service account fails with `storageQuotaExceeded`.

GETTING client_secret.json
--------------------------
1. console.cloud.google.com -> select or create a project.
2. APIs & Services -> Library -> "Google Drive API" -> Enable.
3. APIs & Services -> OAuth consent screen:
     - User type: External
     - Fill in app name / support email / developer email
     - Add YOUR OWN Google account under "Test users"
4. APIs & Services -> Credentials -> Create credentials -> OAuth client ID
     - Application type: Desktop app
   Download the JSON; that is `client_secret.json`.

A NOTE ON TOKEN LIFETIME
------------------------
While the consent screen's publishing status is "Testing", Google expires
refresh tokens after SEVEN DAYS, and mirroring will start failing after that.
For a run spanning more than a week, set the publishing status to "In
production" on the OAuth consent screen (an unverified app still works; you
will see a warning screen during consent, which you can pass via "Advanced ->
Go to <app> (unsafe)"). Re-run this script whenever the token stops working.
"""
from __future__ import annotations

import argparse
import json
import sys

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Obtain a Drive refresh token for Kaggle.")
    p.add_argument("--client_secrets", default="client_secret.json",
                   help="OAuth client JSON downloaded from Google Cloud console.")
    p.add_argument("--out", default="", help="Also write the result to this path.")
    args = p.parse_args(argv)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Install the helper first:  pip install google-auth-oauthlib", file=sys.stderr)
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secrets, SCOPES)
    # access_type=offline + prompt=consent is what actually returns a refresh
    # token; without them Google may hand back only a one-hour access token.
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    if not creds.refresh_token:
        print("No refresh token was returned. Revoke the app's access at "
              "https://myaccount.google.com/permissions and run this again.", file=sys.stderr)
        return 1

    blob = {
        "type": "authorized_user",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri or "https://oauth2.googleapis.com/token",
    }
    text = json.dumps(blob, indent=2)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")

    print("\n" + "=" * 70)
    print("Paste EVERYTHING between the lines into a Kaggle Secret")
    print("(Add-ons -> Secrets -> Add a new secret), label it GDRIVE_SA,")
    print("and attach it to the notebook.")
    print("=" * 70)
    print(text)
    print("=" * 70)
    print("\nThen verify from the notebook:")
    print("  python -m scripts.check_drive --folder_id <ID> --credentials GDRIVE_SA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
