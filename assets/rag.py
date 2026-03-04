# rag.py
"""
Módulo de Recuperação e Resposta (RAG) para o projeto PET-Saúde G10.
Modo híbrido: prioriza documentos indexados, complementa com conhecimento geral quando necessário.
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Tenta carregar do Streamlit Secrets (quando em produção no Streamlit Cloud)
try:
    import streamlit as st
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
except Exception:
    pass

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

from typing import Tuple, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# Configurações que DEVEM ser idênticas às do ingest.py
DB_DIR_DEFAULT = "data/chroma_db"
COLLECTION_NAME = "diet_knowledge"

# Limiar de relevância: score abaixo disso indica contexto fraco
RELEVANCE_THRESHOLD = 0.4

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


def _get_db(db_dir: str = DB_DIR_DEFAULT) -> Chroma:
    """Instancia o banco de dados Chroma."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrada. Verifique o arquivo .env.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def answer(question: str, patient_summary: str, k: int = 5) -> Tuple[str, List]:
    """
    Busca nos documentos e gera resposta híbrida.
    Retorna (texto_da_resposta, lista_de_documentos_encontrados).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrada. Verifique o arquivo .env.")

    # 1. Conecta ao banco
    db = _get_db()

    # 2. Query combinando perfil do paciente + pergunta
    full_query = f"Paciente: {patient_summary}\nPergunta: {question}"

    # 3. Busca com score de relevância
    hits_with_score = db.similarity_search_with_relevance_scores(full_query, k=k)

    # 4. Filtra hits relevantes
    hits = [doc for doc, score in hits_with_score if score >= RELEVANCE_THRESHOLD]
    all_hits = [doc for doc, _ in hits_with_score]  # para retornar ao app

    # 5. Prepara contexto (mesmo que fraco, envia o que há)
    if hits_with_score:
        context_parts = []
        for doc, score in hits_with_score:
            source = os.path.basename(doc.metadata.get("source", "desconhecido"))
            page = doc.metadata.get("page", "-")
            content = doc.page_content.strip()
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

    # 6. Configura o modelo
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=api_key,
        temperature=0.2  # leve flexibilidade para complementar com conhecimento geral
    )

    user_message = f"""DADOS DO PACIENTE:
{patient_summary}

PERGUNTA DO USUÁRIO:
{question}

CONTEXTO DOS DOCUMENTOS:
{context_text}
{context_note}
"""

    # 7. Gera resposta
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": user_message}
    ])

    return response.content, all_hits
