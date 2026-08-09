"""
Google Drive mirroring for checkpoints.

WHY THIS EXISTS
---------------
Kaggle wipes /kaggle/working when a session ends, so a checkpoint that only
exists locally does not survive to the next 12-hour session. This module
mirrors the checkpoint (and the loss history) into a Drive folder after each
save, and pulls it back at startup, which is what makes a multi-session run
resumable.

AUTHENTICATION -- read this before configuring
----------------------------------------------
Kaggle has no `google.colab.drive` mount, so authentication has to be
non-interactive. Two supported routes:

1. SERVICE ACCOUNT (recommended).
   a. Google Cloud console -> create a project -> enable the Drive API.
   b. Create a service account, create a JSON key, download it.
   c. In your own Drive, create a folder for checkpoints. Share it with the
      service account's email address (the `client_email` field of the JSON),
      giving Editor access.
      This step is NOT optional: a service account has no Drive storage quota
      of its own, so it can only write into a folder shared from a real
      account. Skipping it produces a confusing
      "storageQuotaExceeded" error on the first upload.
   d. Copy the folder id out of its URL
      (drive.google.com/drive/folders/<THIS_PART>) into `drive_folder_id`.
   e. Paste the JSON into a Kaggle Secret (Add-ons -> Secrets), then pass its
      name as `drive_credentials`.

2. OAUTH REFRESH TOKEN. Store a JSON containing `client_id`,
   `client_secret` and `refresh_token` in the same place. Files then land in
   your own Drive under your own quota. Obtain it once locally with any OAuth
   flow for the `drive.file` scope.

Everything degrades gracefully: if Drive is not configured, or the libraries
are missing, or a call fails, training continues with local-only checkpoints
and a warning. Losing the mirror must never take the run down with it.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Optional

from ..utils.progress import write

_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _load_credentials_blob(spec: str) -> Optional[dict]:
    """Resolve `spec` to a credentials dict from a path, a Kaggle Secret, an
    env var, or raw JSON."""
    candidates = [spec] if spec else []
    candidates += [
        os.environ.get("VNGAT_DRIVE_CREDENTIALS", ""),
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
    ]
    for cand in [c for c in candidates if c]:
        # A path on disk
        if Path(cand).is_file():
            with open(cand) as fh:
                return json.load(fh)
        # Raw JSON pasted straight into the config
        stripped = cand.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        # A Kaggle Secret label
        try:  # pragma: no cover - Kaggle-only path
            from kaggle_secrets import UserSecretsClient

            return json.loads(UserSecretsClient().get_secret(cand))
        except Exception:  # noqa: BLE001
            continue
    return None


class DriveSync:
    """
    Minimal Drive client: upload (replacing in place), download, exists.

    Uploads keep the SAME file id when a file of that name already exists in
    the folder, so the checkpoint is genuinely replaced rather than
    accumulating hundreds of same-named copies (Drive allows duplicate names,
    which makes "just upload it again" quietly wrong).
    """

    def __init__(self, folder_id: str = "", credentials: str = "", enabled: bool = True):
        self.folder_id = folder_id
        self.service = None
        self.enabled = bool(enabled and folder_id)
        if not self.enabled:
            return
        try:
            self.service = self._build_service(credentials)
        except Exception as exc:  # noqa: BLE001
            write(f"  [drive] disabled -- could not authenticate: {type(exc).__name__}: {exc}")
            self.enabled = False

    # ------------------------------------------------------------------
    @staticmethod
    def _build_service(credentials_spec: str):
        from googleapiclient.discovery import build

        blob = _load_credentials_blob(credentials_spec)
        if blob is None:
            raise RuntimeError(
                "no Drive credentials found (checked the given path/secret name, "
                "VNGAT_DRIVE_CREDENTIALS and GOOGLE_APPLICATION_CREDENTIALS)"
            )
        if blob.get("type") == "service_account":
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_info(blob, scopes=_SCOPES)
        else:
            from google.oauth2.credentials import Credentials

            creds = Credentials(
                token=None,
                refresh_token=blob["refresh_token"],
                client_id=blob["client_id"],
                client_secret=blob["client_secret"],
                token_uri=blob.get("token_uri", "https://oauth2.googleapis.com/token"),
                scopes=_SCOPES,
            )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # ------------------------------------------------------------------
    def _find(self, name: str) -> Optional[str]:
        query = (
            f"name = '{name}' and '{self.folder_id}' in parents and trashed = false"
        )
        resp = self.service.files().list(
            q=query, fields="files(id, name)", pageSize=10,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    def upload(self, local_path: str, remote_name: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        from googleapiclient.http import MediaFileUpload

        remote_name = remote_name or Path(local_path).name
        try:
            media = MediaFileUpload(local_path, resumable=True)
            existing = self._find(remote_name)
            if existing:
                self.service.files().update(
                    fileId=existing, media_body=media, supportsAllDrives=True,
                ).execute()
            else:
                self.service.files().create(
                    body={"name": remote_name, "parents": [self.folder_id]},
                    media_body=media, fields="id", supportsAllDrives=True,
                ).execute()
            return True
        except Exception as exc:  # noqa: BLE001
            write(f"  [drive] upload of {remote_name} failed: {type(exc).__name__}: {exc}")
            return False

    def download(self, remote_name: str, local_path: str) -> bool:
        if not self.enabled:
            return False
        from googleapiclient.http import MediaIoBaseDownload

        try:
            file_id = self._find(remote_name)
            if not file_id:
                return False
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            buffer = io.FileIO(local_path, "wb")
            downloader = MediaIoBaseDownload(buffer, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _status, done = downloader.next_chunk()
            buffer.close()
            return True
        except Exception as exc:  # noqa: BLE001
            write(f"  [drive] download of {remote_name} failed: {type(exc).__name__}: {exc}")
            return False

    def exists(self, remote_name: str) -> bool:
        if not self.enabled:
            return False
        try:
            return self._find(remote_name) is not None
        except Exception:  # noqa: BLE001
            return False
