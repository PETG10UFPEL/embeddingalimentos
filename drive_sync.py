import os
import streamlit as stcd "C:\Users\aljma\G10\aplicativos\diet_app"
git add app.py drive_sync.py
git commit -m "fix: credenciais via st.secrets; feat: banner maior e senha admin na sidebar"
git push origin main
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
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