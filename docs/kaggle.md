# Persisting checkpoints across Kaggle sessions

Kaggle deletes `/kaggle/working` when a session ends, so a checkpoint that
only exists there does not survive. Two ways to keep it. The first needs no
Google account at all and is the one to reach for first.

---

## Option 1 (recommended): Kaggle's own output persistence

**Save the run.** Instead of running cells interactively, use
**Save Version -> Save & Run All (Commit)**. The notebook runs start to finish
in the background and everything left in `/kaggle/working` is stored as that
version's output. Interactive sessions only persist `/kaggle/working` if you
click Save Version before the session closes.

**Resume in a later session.**

1. Open the notebook -> **Add Input** -> **Your Work / Notebook Output** ->
   pick the previous version. It mounts read-only under
   `/kaggle/input/<notebook-slug>/`.
2. Copy the checkpoint back into place and resume:

```bash
mkdir -p /kaggle/working/checkpoints
cp /kaggle/input/<notebook-slug>/checkpoints/checkpoint.pt /kaggle/working/checkpoints/
cp /kaggle/input/<notebook-slug>/checkpoints/history.json  /kaggle/working/checkpoints/

python -m scripts.train --config configs/kaggle_2xt4_full.yaml --root_dir data \
    --checkpoint_dir /kaggle/working/checkpoints --resume auto
```

`--resume auto` picks up the stored epoch, optimiser state, LR schedule, AMP
scaler and RNG state, and truncates the history so the curves stay aligned.

Trade-off: the copy is manual, and output is capped (currently ~20 GB, far
more than this project needs).

---

## Option 2: Google Drive mirroring

Automatic once configured -- every checkpoint save is mirrored, and
`--resume auto` pulls it back with no manual copying.

**Do not use a service account with a personal Google account.** Google
removed storage quota from service accounts, so files they create have no
owner with space and every upload fails with `storageQuotaExceeded` -- even
when the folder is shared with the service account, and even though
authentication succeeds. Service accounts only work against a Workspace
**Shared Drive**, where the drive owns the files.

For a personal Gmail account, use OAuth credentials instead:

```bash
# on your own machine, once -- it needs a browser
pip install google-auth-oauthlib
python -m scripts.drive_oauth_setup --client_secrets client_secret.json
```

Paste the printed JSON into a Kaggle Secret (Add-ons -> Secrets), label it
`GDRIVE_SA`, and attach it to the notebook. Then verify before a long run:

```bash
python -m scripts.check_drive --folder_id <FOLDER_ID> --credentials GDRIVE_SA
```

`<FOLDER_ID>` is the last path element of the folder's URL:
`drive.google.com/drive/folders/<FOLDER_ID>`.

Once that passes:

```bash
python -m scripts.train --config configs/kaggle_2xt4_full.yaml --root_dir data \
    --checkpoint_dir /kaggle/working/checkpoints \
    --drive_folder_id <FOLDER_ID> --drive_credentials GDRIVE_SA
```

Training verifies the round trip at startup and prints
`[drive] mirroring verified` or a loud warning; it never dies because Drive is
unavailable, it just falls back to local-only checkpoints.

While the OAuth consent screen's publishing status is **Testing**, Google
expires refresh tokens after seven days. For longer campaigns set it to
"In production" (an unverified app still works) or re-run the setup script.

---

## What gets written either way

| file | when |
|---|---|
| `checkpoint.pt` | every `save_every` epochs, replaced unless the stored one is better on **both** train and val loss |
| `best.pt` | whenever validation improves |
| `history.json` | every epoch |
| `config.yaml` | at startup |

Remote copies are prefixed with `tag`, e.g. `vngat_full_checkpoint.pt`.
