# rag.py
"""
Módulo de Recuperação e Resposta (RAG) para o projeto PET-Saúde G10.
Modo híbrido: prioriza documentos indexados, complementa com conhecimento geral.
Embeddings: HuggingFace local (sem API paga).
LLM: Groq (gratuito, rápido, suporte a português).
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

from typing import Tuple, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

DB_DIR_DEFAULT      = "data/chroma_db"
COLLECTION_NAME     = "diet_knowledge"
RELEVANCE_THRESHOLD = 0.4
EMBED_MODEL         = "paraphrase-multilingual-mpnet-base-v2"

SYSTEM_RULES = """Você é um assistente de planejamento alimentar clínico do projeto PET-Saúde G10, \
especialista em nutrição clínica e dietoterapia.

MODO DE OPERAÇÃO HÍBRIDO:
1. PRIORIDADE MÁXIMA: Use os trechos do CONTEXTO DOS DOCUMENTOS como base principal da resposta.
   - Sempre cite as fontes no formato [Nome do Arquivo | Página] quando usar o contexto.
2. COMPLEMENTAÇÃO: Se o contexto não cobrir completamente a situação clínica do paciente,
   complemente com seu conhecimento em nutrição clínica baseado em evidências.
   - Neste caso, indique claramente: "(conhecimento geral - não consta nos documentos indexados)"
3. TRANSPARÊNCIA: Sempre deixe claro o que veio dos documentos e o que veio do conhecimento geral.
4. NUNCA invente fontes ou cite documentos que não estejam no contexto fornecido.
5. Seja objetivo e organize a resposta em tópicos/checklists.
6. Adapte sempre ao perfil específico do paciente informado.
"""


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Embeddings locais — sem API key, sem cota."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_db_from_disk(db_dir: str = DB_DIR_DEFAULT) -> Chroma:
    """Carrega um índice já persistido em disco (uso local)."""
    return Chroma(
        persist_directory=db_dir,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def answer(
    question: str,
    patient_summary: str,
    k: int = 5,
    vectordb: Optional[Chroma] = None,
) -> Tuple[str, List]:
    """
    Busca nos documentos e gera resposta híbrida via Groq.

    Parâmetros:
      vectordb  – instância Chroma já criada (session_state). Se None,
                  tenta carregar do disco (modo local).
    Retorna (texto_da_resposta, lista_de_documentos_encontrados).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY não encontrada. Configure em .streamlit/secrets.toml."
        )

    # 1. Decide qual banco usar
    if vectordb is not None:
        db = vectordb
    else:
        if not os.path.exists(DB_DIR_DEFAULT):
            return (
                "⚠️ Índice não encontrado. No Streamlit Cloud o índice precisa ser "
                "criado na sessão atual: use a sidebar para **Sincronizar** e depois "
                "**Recriar índice** antes de gerar uma resposta.",
                [],
            )
        db = _get_db_from_disk()

    # 2. Query combinada
    full_query = f"Paciente: {patient_summary}\nPergunta: {question}"

    # 3. Busca com score
    hits_with_score = db.similarity_search_with_relevance_scores(full_query, k=k)

    # 4. Filtra relevantes
    hits     = [doc for doc, score in hits_with_score if score >= RELEVANCE_THRESHOLD]
    all_hits = [doc for doc, _ in hits_with_score]

    # 5. Monta contexto
    if hits_with_score:
        context_parts = []
        for doc, score in hits_with_score:
            source    = os.path.basename(doc.metadata.get("source", "desconhecido"))
            page      = doc.metadata.get("page", "-")
            content   = doc.page_content.strip()
            relevance = "alta" if score >= RELEVANCE_THRESHOLD else "baixa"
            context_parts.append(
                f"FONTE: {source} (p. {page}) [relevância: {relevance}]\nCONTEÚDO: {content}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        context_note = "" if hits else (
            "\n⚠️ NOTA: Os documentos indexados têm baixa relevância para esta consulta. "
            "Use seu conhecimento clínico para complementar, indicando claramente o que é conhecimento geral."
        )
    else:
        context_text = "Nenhum documento relevante encontrado no índice."
        context_note = (
            "\n⚠️ NOTA: Não há documentos indexados para esta consulta. "
            "Responda com seu conhecimento clínico em nutrição, indicando que a resposta é baseada em conhecimento geral."
        )

    # 6. LLM via Groq
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=groq_key,
        temperature=0.2,
    )

    user_message = f"""DADOS DO PACIENTE:
{patient_summary}

PERGUNTA DO USUÁRIO:
{question}

CONTEXTO DOS DOCUMENTOS:
{context_text}
{context_note}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user",   "content": user_message},
    ])

    return response.content, all_hits
