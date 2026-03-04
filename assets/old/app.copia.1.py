import os
from dotenv import load_dotenv
load_dotenv(override=True)  # DEVE ser antes de qualquer outro import

import streamlit as st

# Streamlit Cloud: carrega secrets se disponíveis
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
    if "GDRIVE_FOLDER_ID" in st.secrets:
        os.environ["GDRIVE_FOLDER_ID"] = st.secrets["GDRIVE_FOLDER_ID"]
except Exception:
    pass

from drive_sync import sync_folder
from ingest import build_index
from rag import answer


st.set_page_config(page_title="Planejador de Dieta (RAG)", layout="wide")
st.title("🥗 Planejador de Dieta - Patrícia")

st.caption(
    "**Patrícia Xavier Bittencourt**, estudante · Disciplina 15001103 M1 — Princípios de Inteligência Artificial Aplicados · UFPel (2025/6-2) · "
    "Prof. Alejandro Martins R. · "
    "Sistema elaborado em parceria junto à P&D do Projeto PET UFPel Saúde Digital — Telemonitoramento de Feridas Crônicas."
)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Base de conhecimento (Google Drive)")
    folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
    st.text_input("Folder ID", value=folder_id, key="folder_id")

    if st.button("1) Sincronizar arquivos do Drive"):
        with st.spinner("Baixando..."):
            files = sync_folder(st.session_state.folder_id, "data/raw_docs")
        st.success(f"Baixados {len(files)} arquivos para data/raw_docs")

    if st.button("2) Recriar índice (embeddings)"):
        with st.spinner("Indexando..."):
            n = build_index("data/raw_docs", "data/chroma_db")
        st.success(f"Índice criado com {n} trechos.")

# ==============================
# Entrada do usuário
# ==============================
st.subheader("Dados do paciente")
patient = st.text_area(
    "Cole aqui um resumo estruturado (idade, sexo, comorbidades, alergias, preferências, objetivos, etc.)",
    height=160
)

st.subheader("Pergunta / objetivo")
q = st.text_area(
    "Ex.: sugerir plano alimentar de 7 dias, ou ajustar dieta para cicatrização, etc.",
    height=100
)

# ==============================
# Cache para evitar chamadas repetidas
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def cached_answer(q, patient, k):
    return answer(q, patient, k=k)

# ==============================
# Botão principal
# ==============================
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🗑️ Limpar cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

if st.button("Gerar resposta"):

    if not patient.strip() or not q.strip():
        st.error("Preencha dados do paciente e a pergunta.")
    else:
        with st.spinner("Buscando nos documentos e gerando resposta..."):
            patient_short = patient[:2000]
            resp, hits = cached_answer(q, patient_short, 3)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Resposta")
            st.markdown(resp)

        with col2:
            st.markdown("### Fontes recuperadas")
            if hits:
                for d in hits:
                    src = d.metadata.get("source", "arquivo_desconhecido")
                    page = d.metadata.get("page", "trecho")
                    st.write(f"- {src} | {page}")
            else:
                st.info("Nenhuma fonte indexada utilizada — resposta baseada em conhecimento geral.")
