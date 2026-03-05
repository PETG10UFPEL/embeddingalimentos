import os
import json
import streamlit as st
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Caminho para o JSON de service account local (desenvolvimento)
# Coloque o arquivo em .streamlit/service_account.json  OU  defina a variável
# de ambiente GOOGLE_SERVICE_ACCOUNT_JSON com o conteúdo JSON como string.
LOCAL_SA_PATHS = [
    Path(__file__).parent / ".streamlit" / "service_account.json",
    Path(__file__).parent / "service_account.json",
]


def _load_service_account_info() -> dict:
    """
    Tenta carregar as credenciais na seguinte ordem de prioridade:
    1. st.secrets["gcp_service_account"]  (Streamlit Cloud)
    2. Variável de ambiente GOOGLE_SERVICE_ACCOUNT_JSON  (string JSON)
    3. Arquivo local service_account.json  (desenvolvimento)
    """
    # 1. Streamlit Cloud secrets
    try:
        info = st.secrets["gcp_service_account"]
        return dict(info)
    except Exception:
        pass

    # 2. Variável de ambiente com o JSON completo como string
    env_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if env_json.strip():
        try:
            return json.loads(env_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON contém JSON inválido."
            ) from e

    # 3. Arquivo JSON local
    for p in LOCAL_SA_PATHS:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)

    raise RuntimeError(
        "Credenciais do Google Drive não encontradas.\n\n"
        "Para rodar localmente, escolha UMA das opções:\n"
        "  a) Salve o JSON da service account em:\n"
        "       .streamlit/service_account.json\n"
        "  b) Defina a variável de ambiente:\n"
        "       GOOGLE_SERVICE_ACCOUNT_JSON='{...json...}'\n\n"
        "Para o Streamlit Cloud, configure st.secrets['gcp_service_account']."
    )


def get_drive_service():
    info = _load_service_account_info()
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def sync_folder(folder_id: str, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    service = get_drive_service()

    q = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=q, fields="files(id, name, mimeType, modifiedTime)"
    ).execute()
    files = results.get("files", [])

    downloaded = []
    for f in files:
        name = f["name"]
        file_id = f["id"]
        mime = f["mimeType"]

        if mime.startswith("application/vnd.google-apps"):
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
