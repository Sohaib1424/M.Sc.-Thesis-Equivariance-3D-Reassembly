#!/usr/bin/env python
"""
Check that Google Drive checkpoint mirroring actually works, end to end,
before committing to a long run.

    python -m scripts.check_drive --folder_id <ID> --credentials GDRIVE_SA

Does a real round trip -- upload, find, delete -- because authentication
succeeding tells you almost nothing. The common failure has `auth: True` and
fails only at the first upload, which during training is ten epochs in.

A script rather than a heredoc: `!python - << 'EOF'` does not work in a Jupyter
cell (only the first line reaches bash; the rest is executed as notebook
Python, and the closing token raises NameError).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vngat.training.drive import DriveSync  # noqa: E402

QUOTA_HELP = """
DIAGNOSIS: service accounts have no Drive storage of their own.

Google removed the storage quota from service accounts. A file a service
account creates is OWNED by that service account, so the upload fails with
'storageQuotaExceeded' even when the target folder is shared with it and even
though authentication succeeded. Sharing the folder does NOT fix this.

Three ways forward:

  1. Do not use Drive at all (simplest). Kaggle persists /kaggle/working when
     the notebook is run with "Save Version -> Save & Run All (Commit)". To
     continue in a later session, attach that run's output as an input dataset
     and copy the checkpoint back in. See docs/kaggle.md.

  2. Use OAuth credentials from your own Google account instead of a service
     account. Files are then owned by you and count against your normal 15 GB.
     Run  python -m scripts.drive_oauth_setup  ON YOUR OWN MACHINE (it needs a
     browser), then paste the JSON it prints into the Kaggle Secret.

  3. Use a Shared Drive, where the drive owns the files rather than the
     account. This requires Google Workspace; personal Gmail accounts do not
     have Shared Drives.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify Google Drive checkpoint mirroring.")
    p.add_argument("--folder_id", required=True, help="From the folder URL: .../folders/<THIS>")
    p.add_argument("--credentials", default="GDRIVE_SA",
                   help="Kaggle Secret label, path to a JSON file, or raw JSON.")
    p.add_argument("--keep", action="store_true", help="Do not delete the probe file.")
    args = p.parse_args(argv)

    probe = Path("/tmp/vngat_drive_probe.txt")
    probe.write_text("vngat drive probe\n")

    drive = DriveSync(args.folder_id, args.credentials)
    print(f"  authenticated : {drive.enabled}")
    if not drive.enabled:
        print("\nCould not authenticate. Check that the secret exists, is attached to this\n"
              "notebook, and contains the whole JSON document (starting with '{').")
        return 1

    kind = drive.credential_kind or "unknown"
    print(f"  credential    : {kind}")

    uploaded = drive.upload(str(probe), "vngat_probe.txt")
    print(f"  upload        : {uploaded}")
    if not uploaded:
        if drive.last_error and "storagequota" in drive.last_error.lower():
            print(QUOTA_HELP)
        else:
            print("\nUpload failed. If the folder id is right and the credential is a service\n"
                  "account, see the storage-quota note in vngat/training/drive.py.")
        return 1

    found = drive.exists("vngat_probe.txt")
    print(f"  found in folder: {found}")
    if not found:
        print("\nUploaded but not visible in that folder -- check the folder id.")
        return 1

    if not args.keep:
        print(f"  cleaned up    : {drive.delete('vngat_probe.txt')}")

    print("\nDrive mirroring is working. Pass the same --drive_folder_id and\n"
          "--drive_credentials to scripts.train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
