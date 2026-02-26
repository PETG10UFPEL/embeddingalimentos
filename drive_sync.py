import os
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def sync_folder(folder_id: str, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    service = get_drive_service()

    q = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=q, fields="files(id, name, mimeType, modifiedTime)").execute()
    files = results.get("files", [])

    downloaded = []
    for f in files:
        name = f["name"]
        file_id = f["id"]
        mime = f["mimeType"]

        # Baixa só arquivos "baixáveis" direto
        # (Docs/Sheets exigem export; dá pra acrescentar depois)
        if mime.startswith("application/vnd.google-apps"):
            # pular por enquanto (ou implementar export)
            continue

        dest = out / name
        request = service.files().get_media(fileId=file_id)

        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        downloaded.append(dest)

    return downloaded