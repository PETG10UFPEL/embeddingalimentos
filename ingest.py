# ingest.py
"""
Indexador (RAG) para o projeto PET-Saúde G10.
Usa HuggingFace Embeddings locais (sem API paga, sem cota).
Modelo multilíngue: suporta PT, EN, ES e +50 línguas.
"""

from __future__ import annotations

import os
import ftfy
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
except Exception:
    UnstructuredWordDocumentLoader = None  # type: ignore

load_dotenv()

RAW_DIR_DEFAULT = "data/raw_docs"
DB_DIR_DEFAULT  = "data/chroma_db"
COLLECTION_NAME = "diet_knowledge"

# Modelo multilíngue — roda local, sem API key, sem limite de cota
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def _fix_encoding(docs: List[Document]) -> List[Document]:
    """
    Corrige problemas de encoding em textos extraídos de PDFs e DOCX.
    Usa ftfy para detectar e reparar bytes mal interpretados (ex: ç, ã, é
    salvos em latin-1 mas lidos como UTF-8, comum em documentos PT-BR).
    """
    for doc in docs:
        try:
            doc.page_content = ftfy.fix_text(doc.page_content)
        except Exception:
            # Fallback: força re-encode latin-1 → utf-8 se ftfy falhar
            try:
                doc.page_content = (
                    doc.page_content.encode("latin-1", errors="replace")
                    .decode("utf-8", errors="replace")
                )
            except Exception:
                pass
    return docs


def _load_pdf(path: Path) -> List[Document]:
    try:
        docs = PyPDFLoader(str(path)).load()
        return _fix_encoding(docs)
    except Exception as e:
        print(f"[AVISO] Erro ao carregar PDF {path}: {e}")
        return []


def _load_docx(path: Path) -> List[Document]:
    if UnstructuredWordDocumentLoader is None:
        print(f"[AVISO] Pulando DOCX (loader indisponível): {path}")
        return []
    try:
        docs = UnstructuredWordDocumentLoader(str(path)).load()
        return _fix_encoding(docs)
    except Exception as e:
        print(f"[AVISO] Erro ao carregar DOCX {path}: {e}")
        return []


def load_file(path: Path) -> List[Document]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _load_pdf(path)
    if suf == ".docx":
        return _load_docx(path)
    return []


def load_all_docs(raw_dir: str) -> Tuple[List[Document], List[str]]:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    docs: List[Document] = []
    skipped: List[str] = []
    for p in raw.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".pdf", ".docx"]:
            skipped.append(str(p))
            continue
        try:
            docs.extend(load_file(p))
        except Exception as e:
            skipped.append(f"{p} (erro: {e})")
    return docs, skipped


def build_index(
    raw_dir: str = RAW_DIR_DEFAULT,
    db_dir: str = DB_DIR_DEFAULT,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    gdrive_folder_id: str = "",
) -> Tuple[int, Optional[Chroma]]:
    """
    Indexa os documentos localmente com HuggingFace (sem API paga, sem cota).
    Suporta documentos em português, inglês e outras línguas simultaneamente.
    Corrige automaticamente problemas de encoding em documentos PT-BR.

    gdrive_folder_id — se informado, faz upload do índice ao Drive após criar,
                       permitindo recuperar após sleep do Streamlit.
    """
    docs, skipped = load_all_docs(raw_dir)
    if not docs:
        print(f"AVISO: Nenhum documento válido encontrado em '{raw_dir}'")
        return 0, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Embeddings locais — sem API key, sem limite de cota
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[OK] Gerando embeddings para {len(chunks)} chunks (modelo local)...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=COLLECTION_NAME,
    )

    # Compatibilidade com versões antigas do Chroma
    try:
        vectordb.persist()
    except Exception:
        pass

    if skipped:
        print(f"Arquivos ignorados/erro: {len(skipped)}")

    print(f"Sucesso! {len(chunks)} trechos indexados.")

    # Salva índice no Drive para sobreviver ao sleep do Streamlit
    if gdrive_folder_id:
        print("[Drive] Salvando índice no Google Drive...")
        try:
            from drive_sync import upload_index_to_drive
            ok = upload_index_to_drive(db_dir, gdrive_folder_id)
            if ok:
                print("[Drive] Índice salvo com sucesso.")
            else:
                print("[Drive] Falha ao salvar índice (verifique permissões).")
        except Exception as e:
            print(f"[Drive] Erro ao salvar índice: {e}")

    return len(chunks), vectordb


if __name__ == "__main__":
    n, _ = build_index()
    print(f"{n} trechos indexados.")
