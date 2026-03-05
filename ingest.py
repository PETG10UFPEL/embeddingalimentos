# ingest.py
"""
Indexador (RAG) para o projeto PET-Saúde G10.
Usa Google Generative AI Embeddings (text-embedding-004).

NOTA: No Streamlit Cloud o filesystem é efêmero.
      build_index() agora retorna também o objeto vectordb em memória,
      que deve ser guardado no st.session_state pelo app.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from docx import Document as DocxDocument

load_dotenv()

RAW_DIR_DEFAULT  = "data/raw_docs"
DB_DIR_DEFAULT   = "data/chroma_db"
COLLECTION_NAME  = "diet_knowledge"


def _load_pdf(path: Path) -> List[Document]:
    return PyPDFLoader(str(path)).load()


def _load_docx(path: Path) -> List[Document]:
    doc = DocxDocument(str(path))
    parts: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(path), "page": "DOCX"})]


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
    db_dir: str = DB_DIR_DEFAULT,       # ignorado no modo in-memory
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    in_memory: bool = True,             # ← NOVO: padrão True para Streamlit Cloud
) -> Tuple[int, Optional[Chroma]]:
    """
    Indexa os documentos e retorna (n_chunks, vectordb).

    - Se in_memory=True  → cria o Chroma em RAM (ideal para Streamlit Cloud).
    - Se in_memory=False → persiste em db_dir (ideal para rodar local).

    O objeto vectordb deve ser salvo em st.session_state para reutilização.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrada.")

    # 1. Carregamento
    docs, skipped = load_all_docs(raw_dir)
    if not docs:
        print(f"AVISO: Nenhum documento válido encontrado em '{raw_dir}'")
        return 0, None

    # 2. Divisão em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 3. Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # 4. Cria o vectorstore
    if in_memory:
        # Sem persist_directory → fica só na RAM desta sessão
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
        )
    else:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_dir,
            collection_name=COLLECTION_NAME,
        )

    if skipped:
        print(f"Arquivos ignorados/erro: {len(skipped)}")

    print(f"Sucesso! {len(chunks)} trechos indexados.")
    return len(chunks), vectordb


if __name__ == "__main__":
    n, _ = build_index(in_memory=False)   # local: persiste em disco
    print(f"{n} trechos indexados.")
