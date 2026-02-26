# ingest.py
"""
Indexador (RAG) para PDFs e DOCX.

O que faz:
- Lê arquivos em data/raw_docs (PDF e DOCX)
- Extrai texto + metadados de origem (arquivo e página quando houver)
- Quebra em chunks
- Gera embeddings
- Salva no Chroma em data/chroma_db

Uso:
  python ingest.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from docx import Document as DocxDocument

load_dotenv()


RAW_DIR_DEFAULT = "data/raw_docs"
DB_DIR_DEFAULT = "data/chroma_db"
COLLECTION_NAME = "diet_knowledge"


def _load_pdf(path: Path) -> List[Document]:
    # PyPDFLoader já devolve Document por página com metadata: {"source": "...", "page": N}
    return PyPDFLoader(str(path)).load()


def _load_docx(path: Path) -> List[Document]:
    doc = DocxDocument(str(path))

    # Texto de parágrafos
    parts: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    # Texto de tabelas (muito comum em protocolos)
    for table in doc.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                ct = (cell.text or "").strip()
                if ct:
                    cells.append(ct.replace("\n", " ").strip())
            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts).strip()

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path),   # rastreabilidade
                "page": "DOCX",        # não existe página real
            },
        )
    ]


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
    embedding_model: str = "text-embedding-3-large",
) -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrado. Coloque no .env.")

    docs, skipped = load_all_docs(raw_dir)

    if not docs:
        raise RuntimeError(
            f"Nenhum PDF/DOCX encontrado em {raw_dir}. Coloque arquivos em data/raw_docs."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Recria a coleção do zero (simples e robusto)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=COLLECTION_NAME,
    )
    vectordb.persist()

    if skipped:
        print("\nArquivos ignorados/erro (ok se não forem PDF/DOCX):")
        for s in skipped[:50]:
            print(" -", s)
        if len(skipped) > 50:
            print(f" ... e mais {len(skipped) - 50}")

    print(f"\nIndex criado: {len(chunks)} chunks em {db_dir} (coleção: {COLLECTION_NAME})")
    return len(chunks)


if __name__ == "__main__":
    build_index()