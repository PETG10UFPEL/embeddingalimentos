import os
import streamlit as st
from dotenv import load_dotenv

from drive_sync import sync_folder
from ingest import build_index
from rag import answer

load_dotenv()

st.set_page_config(page_title="Planejador de Dieta (RAG)", layout="wide")
st.title("🥗 Planejador de Dieta baseado nos seus arquivos")

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

st.subheader("Dados do paciente")
patient = st.text_area(
    "Cole aqui um resumo estruturado (idade, sexo, comorbidades, alergias, preferências, objetivos, etc.)",
    height=160
)

st.subheader("Pergunta / objetivo")
q = st.text_area("Ex.: sugerir plano alimentar de 7 dias, ou ajustar dieta para cicatrização, etc.", height=100)

if st.button("Gerar resposta"):
    if not patient.strip() or not q.strip():
        st.error("Preencha dados do paciente e a pergunta.")
    else:
        with st.spinner("Buscando nos documentos e gerando resposta..."):
            resp, hits = answer(q, patient, k=6)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Resposta")
            st.write(resp)

        with col2:
            st.markdown("### Fontes recuperadas")
            for d in hits:
                src = d.metadata.get("source", "arquivo_desconhecido")
                page = d.metadata.get("page", "s/página")
                st.write(f"- {src} | p. {page}")