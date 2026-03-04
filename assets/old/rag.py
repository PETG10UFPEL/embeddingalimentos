# rag.py
"""
Módulo de Recuperação e Resposta (RAG) para o projeto PET-Saúde G10.
Sincronizado com o modelo gemini-embedding-001 do ingest.py.
"""

import os
from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

from typing import Tuple, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# Configurações que DEVEM ser idênticas às do ingest.py
DB_DIR_DEFAULT = "data/chroma_db"
COLLECTION_NAME = "diet_knowledge"

# Regras de comportamento para o especialista em nutrição
SYSTEM_RULES = """Você é um assistente de planejamento alimentar clínico do projeto PET-Saúde G10.
REGRAS OBRIGATÓRIAS:
- Use APENAS o CONTEXTO fornecido (trechos recuperados dos documentos).
- NÃO use conhecimentos externos ou suposições.
- Se o contexto não tiver a resposta, diga que não há evidência nos arquivos.
- Sempre cite a fonte no formato: [Nome do Arquivo | Página].
- Seja objetivo e organize a resposta em tópicos/checklists.
"""

def _get_db(db_dir: str = DB_DIR_DEFAULT) -> Chroma:
    """Instancia o banco de dados Chroma com as configurações sincronizadas."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrada no .env.")

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
    Realiza a busca e gera a resposta baseada no contexto.
    Retorna (texto_da_resposta, lista_de_documentos_encontrados).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrada.")

    # 1. Conecta ao banco criado pelo ingest.py
    db = _get_db()

    # 2. Cria a query combinando a pergunta com o perfil do paciente
    full_query = f"Paciente: {patient_summary}\nPergunta: {question}"

    # 3. Busca os trechos mais relevantes
    hits = db.similarity_search(full_query, k=k)

    if not hits:
        return "Nenhuma informação relevante encontrada nos documentos indexados.", []

    # 4. Prepara o contexto para o Gemini
    context_parts = []
    for d in hits:
        source = os.path.basename(d.metadata.get("source", "desconhecido"))
        page = d.metadata.get("page", "-")
        content = d.page_content.strip()
        context_parts.append(f"FONTE: {source} (p. {page})\nCONTEÚDO: {content}")

    context_text = "\n\n---\n\n".join(context_parts)

    # 5. Configura o modelo de chat
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=api_key,
        temperature=0
    )

    user_message = f"""DADOS DO PACIENTE:
{patient_summary}

PERGUNTA DO USUÁRIO:
{question}

CONTEXTO DOS DOCUMENTOS:
{context_text}
"""

    # 6. Gera a resposta final
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": user_message}
    ])

    return response.content, hits
