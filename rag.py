# rag.py
from __future__ import annotations

import os
from typing import Tuple, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DB_DIR_DEFAULT = "data/chroma_db"
COLLECTION_NAME = "diet_knowledge"

SYSTEM_RULES = """Você é um assistente de planejamento alimentar clínico.
REGRAS OBRIGATÓRIAS:
- Use APENAS o CONTEXTO fornecido (trechos recuperados).
- NÃO use conhecimento externo, diretrizes gerais ou “bom senso” fora do contexto.
- Se o contexto não for suficiente, diga claramente que não há evidência nos arquivos para responder.
- Sempre cite a(s) fonte(s) em formato: [arquivo | página/trecho].
- Seja objetivo e acionável (itens, checklist, plano).
"""


def _get_db(db_dir: str = DB_DIR_DEFAULT) -> Chroma:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrado. Coloque no .env e reinicie o terminal.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def answer(question: str, patient_summary: str, k: int = 6, db_dir: str = DB_DIR_DEFAULT) -> Tuple[str, List]:
    """
    Retorna (resposta_texto, hits)
    hits é uma lista de Documents recuperados (com metadata source/page).
    """
    db = _get_db(db_dir=db_dir)

    query = (question or "").strip() + "\n" + (patient_summary or "").strip()
    hits = db.similarity_search(query, k=k)

    if not hits:
        return (
            "Não encontrei trechos relevantes nos arquivos indexados para responder com segurança. "
            "Sugestão: refine a pergunta ou verifique se os documentos corretos foram indexados.",
            [],
        )

    context_blocks = []
    for d in hits:
        src = d.metadata.get("source", "arquivo_desconhecido")
        page = d.metadata.get("page", "trecho")
        txt = (d.page_content or "").strip()
        context_blocks.append(f"FONTE: {src} | p/trecho: {page}\nTRECHO:\n{txt}")

    context = "\n\n---\n\n".join(context_blocks)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    user_prompt = f"""DADOS DO PACIENTE:
{patient_summary}

PERGUNTA:
{question}

CONTEXTO RECUPERADO (use somente isso):
{context}

INSTRUÇÃO DE SAÍDA:
- Responda em tópicos curtos.
- Para cada recomendação relevante, cite pelo menos 1 fonte no formato [arquivo | página/trecho].
- Se alguma parte não estiver suportada no contexto, declare explicitamente.
"""

    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": user_prompt},
        ]
    )

    return resp.content, hits