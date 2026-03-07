import os
import json
import streamlit as st
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",  # permite criar/atualizar arquivos
]

INDEX_FOLDER_NAME = "_chroma_index"

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


# ==============================
# FUNÇÕES DE PERSISTÊNCIA DO ÍNDICE
# ==============================

def _get_drive_service_rw():
    """Serviço com permissão de leitura e escrita (drive.file)."""
    info = _load_service_account_info()
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def _list_children(service, folder_id: str) -> list:
    """Lista itens diretamente dentro de folder_id (com paginação)."""
    q = f"'{folder_id}' in parents and trashed = false"
    items, page_token = [], None
    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _get_or_create_index_folder(service, parent_folder_id: str) -> str:
    """Retorna o ID da pasta '_chroma_index', criando se não existir."""
    q = (
        f"'{parent_folder_id}' in parents "
        f"and name = '{INDEX_FOLDER_NAME}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    resp = service.files().list(q=q, fields="files(id)").execute()
    folders = resp.get("files", [])
    if folders:
        return folders[0]["id"]

    folder_meta = {
        "name": INDEX_FOLDER_NAME,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=folder_meta, fields="id").execute()
    print(f"[Drive] Pasta '{INDEX_FOLDER_NAME}' criada no Drive.")
    return folder["id"]


def upload_index_to_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    """
    Faz upload de todos os arquivos do diretório db_dir
    para a pasta '_chroma_index' no Google Drive.
    Retorna True em caso de sucesso.
    """
    from googleapiclient.http import MediaFileUpload

    db_path = Path(db_dir)
    if not db_path.exists():
        print(f"[ERRO] Pasta do índice não existe: {db_path}")
        return False

    files_to_upload = [f for f in db_path.rglob("*") if f.is_file()]
    if not files_to_upload:
        print("[AVISO] Nenhum arquivo encontrado para upload.")
        return False

    try:
        service = _get_drive_service_rw()
        index_folder_id = _get_or_create_index_folder(service, gdrive_folder_id)

        existing = {
            f["name"]: f["id"]
            for f in _list_children(service, index_folder_id)
            if f.get("mimeType") != "application/vnd.google-apps.folder"
        }

        for file_path in files_to_upload:
            name = file_path.name
            media = MediaFileUpload(str(file_path), resumable=False)
            if name in existing:
                service.files().update(fileId=existing[name], media_body=media).execute()
            else:
                service.files().create(
                    body={"name": name, "parents": [index_folder_id]},
                    media_body=media,
                    fields="id",
                ).execute()

        print(f"[Drive] Índice salvo: {len(files_to_upload)} arquivo(s) em '{INDEX_FOLDER_NAME}'.")
        return True

    except Exception as e:
        print(f"[ERRO] Falha ao salvar índice no Drive: {e}")
        return False


def download_index_from_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    """
    Baixa os arquivos da pasta '_chroma_index' do Drive para db_dir.
    Retorna True se o índice existia e foi baixado com sucesso.
    """
    try:
        service = get_drive_service()  # somente leitura é suficiente para baixar

        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{INDEX_FOLDER_NAME}' "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        resp = service.files().list(q=q, fields="files(id)").execute()
        folders = resp.get("files", [])

        if not folders:
            print("[Drive] Pasta '_chroma_index' não encontrada no Drive.")
            return False

        index_folder_id = folders[0]["id"]
        items = _list_children(service, index_folder_id)

        if not items:
            print("[Drive] Pasta '_chroma_index' está vazia.")
            return False

        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)

        for item in items:
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                continue
            dest = db_path / item["name"]
            request = service.files().get_media(fileId=item["id"])
            with open(dest, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

        print(f"[Drive] Índice baixado: {len(items)} arquivo(s) para '{db_path}'.")
        return True

    except Exception as e:
        print(f"[ERRO] Falha ao baixar índice do Drive: {e}")
        return False
