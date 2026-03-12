import os
import io
import streamlit as st
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/drive",  # leitura + escrita; usa cota do dono da pasta
]
FOLDER_MIME = "application/vnd.google-apps.folder"

# Nome da pasta onde o índice Chroma será salvo no Drive
INDEX_FOLDER_NAME = "_chroma_index"


def get_drive_service():
    """
    Autentica no Google Drive de forma flexível:
    1. Tenta usar st.secrets (Streamlit Cloud)
    2. Tenta carregar o arquivo JSON local definido em variáveis de ambiente
    3. Tenta procurar o arquivo JSON padrão na pasta raiz
    """
    creds = None

    # 1) Streamlit Secrets (Produção)
    try:
        if "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        pass

    # 2) Arquivo Local (Desenvolvimento)
    if not creds:
        json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "disco-retina-371501-525385725005.json")
        if os.path.exists(json_path):
            creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)

    if not creds:
        raise RuntimeError(
            "Credenciais do Google Cloud não encontradas. "
            "Configure 'gcp_service_account' nos Secrets do Streamlit ou tenha o arquivo JSON localmente."
        )

    return build("drive", "v3", credentials=creds)


def _list_children(service, folder_id: str):
    """Lista itens diretamente dentro de folder_id (com paginação)."""
    q = f"'{folder_id}' in parents and trashed = false"
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _download_file(service, file_id: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _sync_folder_recursive(service, folder_id: str, out_root: Path, rel: Path = Path(".")) -> list[Path]:
    """Sincroniza recursivamente: percorre subpastas e baixa arquivos preservando a estrutura."""
    downloaded: list[Path] = []
    items = _list_children(service, folder_id)

    for it in items:
        name = it["name"]
        mime = it.get("mimeType", "")
        it_id = it["id"]

        if mime == FOLDER_MIME:
            downloaded.extend(_sync_folder_recursive(service, it_id, out_root, rel / name))
            continue

        if mime.startswith("application/vnd.google-apps"):
            continue

        dest = out_root / rel / name

        if not dest.exists():
            _download_file(service, it_id, dest)
            downloaded.append(dest)

    return downloaded


def sync_folder(folder_id: str, out_dir: str, recursive: bool = True) -> list[Path]:
    """
    Sincroniza uma pasta do Google Drive com um diretório local.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        service = get_drive_service()

        if recursive:
            downloaded = _sync_folder_recursive(service, folder_id, out, Path("."))
        else:
            downloaded = []
            items = _list_children(service, folder_id)
            for f in items:
                name = f["name"]
                file_id = f["id"]
                mime = f.get("mimeType", "")

                if mime == FOLDER_MIME or mime.startswith("application/vnd.google-apps"):
                    continue

                dest = out / name
                if not dest.exists():
                    _download_file(service, file_id, dest)
                    downloaded.append(dest)

        print(f"Baixados {len(downloaded)} arquivos para {out.resolve()}")
        return downloaded

    except Exception as e:
        print(f"Erro na sincronização: {e}")
        return []


# ==============================
# FUNÇÕES DE PERSISTÊNCIA DO ÍNDICE
# ==============================

def _get_or_create_index_folder(service, parent_folder_id: str) -> str:
    """
    Retorna o ID da pasta '_chroma_index' dentro de parent_folder_id.
    Cria a pasta se não existir.
    """
    q = (
        f"'{parent_folder_id}' in parents "
        f"and name = '{INDEX_FOLDER_NAME}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    resp = service.files().list(q=q, fields="files(id, name)").execute()
    files = resp.get("files", [])

    if files:
        return files[0]["id"]

    # Cria a pasta
    folder_meta = {
        "name": INDEX_FOLDER_NAME,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(
        body=folder_meta,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    print(f"[Drive] Pasta '{INDEX_FOLDER_NAME}' criada no Drive.")
    return folder["id"]


def upload_index_to_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    """
    Compacta o índice Chroma em um único ZIP e faz upload para o Google Drive.
    Usar um arquivo único evita o erro 403 de cota de Service Accounts.
    Retorna True em caso de sucesso.
    """
    import zipfile
    import tempfile

    db_path = Path(db_dir)
    if not db_path.exists():
        print(f"[ERRO] Pasta do índice não existe: {db_path}")
        return False

    files_to_upload = [f for f in db_path.rglob("*") if f.is_file()]
    if not files_to_upload:
        print("[AVISO] Nenhum arquivo encontrado para upload.")
        return False

    try:
        # Cria ZIP temporário com todos os arquivos do índice
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = Path(tmp.name)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_upload:
                zf.write(file_path, file_path.relative_to(db_path))

        print(f"[Drive] ZIP criado: {zip_path.stat().st_size // 1024} KB")

        service = get_drive_service()

        # Busca arquivo ZIP existente na pasta pai (não em subpasta)
        zip_name = "chroma_index.zip"
        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{zip_name}' "
            f"and trashed = false"
        )
        resp = service.files().list(q=q, fields="files(id)").execute()
        existing = resp.get("files", [])

        media = MediaFileUpload(str(zip_path), mimetype="application/zip", resumable=False)

        if existing:
            # Atualiza arquivo existente — usa cota do dono da pasta
            service.files().update(
                fileId=existing[0]["id"],
                media_body=media,
            ).execute()
            print(f"[Drive] ZIP atualizado: '{zip_name}'.")
        else:
            # Cria novo arquivo na pasta do usuário
            service.files().create(
                body={"name": zip_name, "parents": [gdrive_folder_id]},
                media_body=media,
                fields="id",
            ).execute()
            print(f"[Drive] ZIP criado no Drive: '{zip_name}'.")

        zip_path.unlink(missing_ok=True)
        return True

    except Exception as e:
        print(f"[ERRO] Falha ao salvar índice no Drive: {e}")
        return False


def download_index_from_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    """
    Baixa o ZIP do índice Chroma do Drive e extrai para db_dir.
    Retorna True se o índice existia e foi baixado com sucesso.
    """
    import zipfile
    import tempfile

    try:
        service = get_drive_service()
        zip_name = "chroma_index.zip"

        # Busca o ZIP na pasta pai
        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{zip_name}' "
            f"and trashed = false"
        )
        resp = service.files().list(q=q, fields="files(id, name)").execute()
        files = resp.get("files", [])

        if not files:
            print("[Drive] Arquivo 'chroma_index.zip' não encontrado no Drive.")
            return False

        # Baixa o ZIP para arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = Path(tmp.name)

        _download_file(service, files[0]["id"], zip_path)
        print(f"[Drive] ZIP baixado: {zip_path.stat().st_size // 1024} KB")

        # Extrai para db_dir
        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(db_path)

        zip_path.unlink(missing_ok=True)
        print(f"[Drive] Índice extraído para '{db_path}'.")
        return True

    except Exception as e:
        print(f"[ERRO] Falha ao baixar índice do Drive: {e}")
        return False


def index_exists_on_drive(gdrive_folder_id: str) -> bool:
    """
    Verifica rapidamente se o ZIP do índice existe no Drive (sem baixar).
    """
    try:
        service = get_drive_service()
        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = 'chroma_index.zip' "
            f"and trashed = false"
        )
        resp = service.files().list(q=q, fields="files(id)").execute()
        return len(resp.get("files", [])) > 0
    except Exception:
        return False


if __name__ == "__main__":
    pass
