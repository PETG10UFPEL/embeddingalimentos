import os
from dotenv import load_dotenv
load_dotenv(override=True)  # DEVE ser antes de qualquer outro import

import base64
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud: carrega secrets se disponíveis
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GDRIVE_FOLDER_ID" in st.secrets:
        os.environ["GDRIVE_FOLDER_ID"] = st.secrets["GDRIVE_FOLDER_ID"]
except Exception:
    pass

from drive_sync import sync_folder, download_index_from_drive
from ingest import build_index
from rag import answer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

DB_DIR          = "data/chroma_db"
COLLECTION_NAME = "diet_knowledge"


st.set_page_config(page_title="Diet - PET G10 UFPel", layout="wide")

# ==============================
# CSS global
# ==============================
st.markdown("""
<style>
  header {visibility: hidden;}
  [data-testid="stToolbar"] {visibility: hidden;}
  [data-testid="stHeader"] {display:none;}
  [data-testid="stDecoration"] {display:none;}
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .block-container { padding-top: 0.25rem !important; }

  .field-label {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    margin-top: 0.8rem;
    color: #1a1a2e;
  }

  .info-text {
    font-size: 1.0rem;
    line-height: 1.6;
    color: #222;
    margin: 0.3rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ==============================
# Helpers de imagem
# ==============================
def img_b64(filename: str) -> str:
    p = Path(__file__).parent / "assets" / filename
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""

# ==============================
# Auto-restauração do índice após sleep do Streamlit
# Roda uma única vez por sessão (cache_resource garante isso).
# Se o chroma_db local estiver ausente, tenta baixar do Drive.
# CORREÇÃO: esta função deve rodar ANTES de load_vectordb_from_disk,
# e load_vectordb_from_disk deve chamá-la internamente para garantir
# a ordem de execução mesmo após st.cache_resource.clear().
# ==============================
@st.cache_resource(show_spinner=False)
def _auto_restore_index() -> str:
    db_path = Path(DB_DIR)
    already_local = db_path.exists() and any(db_path.iterdir())
    if already_local:
        return "local"

    folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
    if not folder_id:
        return "no_folder_id"

    print("[Init] Índice local não encontrado. Tentando restaurar do Drive...")
    ok = download_index_from_drive(DB_DIR, folder_id)
    return "restored" if ok else "not_found"


# ==============================
# Carrega o vectordb do disco (uma vez por sessão)
# CORREÇÃO: chama _auto_restore_index() internamente para garantir
# que o download do Drive sempre precede a leitura do disco,
# independentemente da ordem de execução dos cache_resource.
# ==============================
# _restore_status executa _auto_restore_index UMA vez:
# baixa o índice do Drive se necessário, e retorna o status para a sidebar.
_restore_status = _auto_restore_index()


@st.cache_resource
def load_vectordb_from_disk():
    """
    Carrega o índice Chroma persistido em data/chroma_db/.
    Retorna None se o índice ainda não foi criado.
    Sobrevive a reruns sem reindexar.
    Nota: _auto_restore_index() já foi executada antes desta função
    via _restore_status, garantindo que o Drive foi consultado primeiro.
    """
    if not Path(DB_DIR).exists():
        return None
    # Embeddings locais — multilíngue PT/EN/ES e +50 línguas, sem API key
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# ==============================
# Banner — 40% da largura, centralizado
# ==============================
banner_b64 = img_b64("banner.png")
if banner_b64:
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:8px;">
      <img src="data:image/png;base64,{banner_b64}"
           style="width:40%; height:auto; border-radius:6px;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🩹 Feridas Crônicas - PET G10 UFPel")

# ==============================
# Links institucionais
# ==============================
insta_b64 = img_b64("instagram.png")
enf_b64   = img_b64("logo.enfermagem.png")

insta_tag = f'<img src="data:image/png;base64,{insta_b64}" width="22" style="vertical-align:middle; margin-right:4px;">' if insta_b64 else "📷"
enf_tag   = f'<img src="data:image/png;base64,{enf_b64}" width="22" style="vertical-align:middle; margin-right:4px;">'   if enf_b64   else "🎓"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;
            margin-top:4px; margin-bottom:10px; font-size:0.95rem;">
  {insta_tag}
  <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank"
     style="text-decoration:none; font-weight:500;">Amor à Pele</a>
  <span style="color:#aaa;">|</span>
  <a href="https://www.instagram.com/g10petsaude/" target="_blank"
     style="text-decoration:none; font-weight:500;">PET G10</a>
  <span style="color:#aaa;">|</span>
  {enf_tag}
  <a href="https://wp.ufpel.edu.br/fen/" target="_blank"
     style="text-decoration:none; font-weight:500;">Faculdade de Enfermagem – UFPel</a>
</div>
""", unsafe_allow_html=True)

# ==============================
# Assinatura
# ==============================
st.markdown("""
<p class="info-text" style="margin-bottom:0.8rem;">
  Sistema de apoio ao ensino &amp; aprendizagem sobre <strong>Feridas Crônicas</strong> &mdash; baseado em RAG-AI
  sobre o material das disciplinas oferecidas pela professora
  <a href="https://institucional.ufpel.edu.br/servidores/id/2858" target="_blank">Adrize Rutz Porto</a>,
  da Faculdade de Enfermagem &ndash; UFPel.<br>
  Desenvolvimento conjunto em P&amp;D do Projeto
  <a href="https://www.instagram.com/g10petsaude/" target="_blank">PET UFPel Saúde Digital</a>.
</p>
""", unsafe_allow_html=True)

# ==============================
# Texto explicativo RAG + embeddings
# ==============================
def _html_com_imagens_embutidas(html_path: Path) -> str:
    import re
    html = html_path.read_text(encoding="utf-8", errors="replace")
    assets_dir = html_path.parent

    def substituir(match):
        src = match.group(1)
        if src.startswith("http"):
            return match.group(0)
        img_path = assets_dir / src
        if img_path.exists():
            mime = "image/png" if src.lower().endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'src="data:{mime};base64,{b64}"'
        return match.group(0)

    html = re.sub(r'src="([^"]+)"', substituir, html)
    return html

st.markdown("""
<p class="info-text" style="margin-top:0.8rem;">
  <strong>Retrieval-Augmented Generation (RAG-AI)</strong> melhora a precisão dos modelos de linguagem (LLMs)
  ao recuperar dados externos — como documentos e bases de dados — para responder perguntas,
  reduzindo as chamadas alucinações.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  <strong>Embeddings</strong> são representações numéricas (vetores) de textos que capturam seu significado
  semântico, permitindo que a IA encontre informações relevantes por similaridade de sentido,
  e não apenas por palavras-chave.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  Quer visualizar a representação gráfica do conhecimento?
</p>
""", unsafe_allow_html=True)

_emb_path = Path(__file__).parent / "assets" / "embeddings_feridas.html"
if _emb_path.exists():
    with st.expander("📊 Ver Embeddings da Base de Conhecimento", expanded=False):
        _html_content = _html_com_imagens_embutidas(_emb_path)
        components.html(_html_content, height=700, scrolling=True)
else:
    st.caption("_(arquivo embeddings_feridas.html não encontrado em assets/)_")

st.divider()

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Base de conhecimento (Google Drive)")

    # ── Status da restauração automática ──
    _db_loaded = load_vectordb_from_disk() is not None
    if _db_loaded or _restore_status in ("restored", "local"):
        st.success("✅ Índice carregado e pronto.")
    elif _restore_status == "not_found":
        st.error(
            "⚠️ Nenhum índice encontrado no Drive. "
            "Para começar: faça login como admin → "
            "**1) Sincronizar** → **2) Recriar índice**."
        )
    elif _restore_status == "no_folder_id":
        st.error("⚠️ GDRIVE_FOLDER_ID não configurado nos secrets.")

    admin_pass = st.text_input("Senha admin", type="password")
    is_admin = admin_pass == st.secrets.get("ADMIN_PASSWORD", "")

    if is_admin:
        folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
        st.text_input("Folder ID", value=folder_id, key="folder_id")

        if st.button("1) Sincronizar arquivos do Drive"):
            with st.spinner("Baixando..."):
                files = sync_folder(st.session_state.folder_id, "data/raw_docs")
            st.success(f"Baixados {len(files)} arquivos para data/raw_docs")

        if st.button("2) Recriar índice (embeddings)"):
            with st.spinner("Indexando e salvando no Drive..."):
                n, vectordb = build_index(
                    "data/raw_docs",
                    DB_DIR,
                    gdrive_folder_id=st.session_state.folder_id,
                )
                if vectordb is not None:
                    # CORREÇÃO: limpa apenas o cache do vectordb,
                    # não o da restauração automática (_auto_restore_index).
                    load_vectordb_from_disk.clear()
                    st.success(f"✅ Índice criado com {n} trechos e salvo no Drive.")
                else:
                    st.error("Nenhum documento encontrado em data/raw_docs. Sincronize primeiro.")

        if Path(DB_DIR).exists():
            st.info("✅ Índice disponível em disco.")
        else:
            st.warning("⚠️ Índice não encontrado. Execute os passos 1 e 2 acima.")
    else:
        if admin_pass:
            st.error("Senha incorreta")

# ==============================
# Componente de voz (reutilizável)
# ==============================
def mic_component(target_label: str, field_index: int):
    key = f"mic_{field_index}"
    components.html(f"""
    <style>
      .mic-row {{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }}
      .mic-btn {{
        background:#0066cc; color:white; border:none;
        width:36px; height:36px; border-radius:50%;
        font-size:1.1rem; cursor:pointer; flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        transition:background .2s;
      }}
      .mic-btn.listening {{ background:#e53935; animation:pulse 1s infinite; }}
      @keyframes pulse {{
        0%,100%{{box-shadow:0 0 0 0 rgba(229,57,53,.35);}}
        50%{{box-shadow:0 0 0 8px rgba(229,57,53,0);}}
      }}
      #micStatus_{key} {{ font-size:.80rem; color:#555; }}
    </style>
    <div class="mic-row">
      <button class="mic-btn" id="micBtn_{key}" title="Falar">🎤</button>
      <span id="micStatus_{key}">Toque em 🎤 para falar (Chrome/Edge)</span>
    </div>
    <script>
    (function(){{
      const btn    = document.getElementById('micBtn_{key}');
      const status = document.getElementById('micStatus_{key}');
      let listening = false;

      function fillTextarea(text) {{
        try {{
          const doc = window.parent.document;
          const allTa = doc.querySelectorAll('[data-testid="stTextArea"] textarea');
          const ta = allTa[{field_index}] || doc.querySelector('textarea[aria-label="{target_label}"]');
          if (!ta) {{ status.textContent = '⚠️ Campo não encontrado.'; return; }}
          const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
          setter.call(ta, text);
          ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
          ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
          ta.focus();
          status.textContent = '✅ Texto inserido — edite se quiser.';
        }} catch(e) {{
          status.textContent = '⚠️ Erro: ' + e.message;
        }}
      }}

      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {{
        status.textContent = '⚠️ Voz indisponível — use Chrome ou Edge.';
        btn.style.opacity = '.4';
        return;
      }}
      const rec = new SR();
      rec.lang = 'pt-BR';
      rec.continuous = false;
      rec.interimResults = true;

      rec.onstart = () => {{
        listening = true;
        btn.classList.add('listening');
        btn.innerHTML = '🔴';
        status.textContent = '🎙️ Ouvindo…';
      }};
      rec.onresult = (e) => {{
        let interim = '', final = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {{
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) final += t; else interim += t;
        }}
        if (final) fillTextarea(final.trim());
        else status.textContent = '…' + interim;
      }};
      rec.onerror = (e) => {{
        status.textContent = '❌ Erro: ' + e.error;
        btn.classList.remove('listening'); btn.innerHTML = '🎤'; listening = false;
      }};
      rec.onend = () => {{
        listening = false; btn.classList.remove('listening'); btn.innerHTML = '🎤';
      }};
      btn.addEventListener('click', () => {{ if (listening) rec.stop(); else rec.start(); }});
    }})();
    </script>
    """, height=48, scrolling=False)


# ==============================
# Campos de entrada
# ==============================
st.markdown('<p class="field-label">📋 Dados do paciente</p>', unsafe_allow_html=True)
mic_component("Dados do paciente", 0)
patient = st.text_area(
    "Dados do paciente",
    height=160,
    placeholder="Cole aqui um resumo estruturado (idade, sexo, comorbidades, alergias, preferências, objetivos, etc.)",
    label_visibility="collapsed",
)

st.markdown('<p class="field-label">❓ Pergunta / objetivo</p>', unsafe_allow_html=True)
mic_component("Pergunta / objetivo", 1)
q = st.text_area(
    "Pergunta / objetivo",
    height=100,
    placeholder="Ex.: Quais são as fases da cicatrização? Como classificar uma úlcera por pressão? Qual o curativo indicado para ferida com exsudato abundante?",
    label_visibility="collapsed",
)

# ==============================
# Botões
# ==============================
col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    gerar = st.button("🚀 Gerar resposta", type="primary")
with col_btn2:
    if st.button("🗑️ Limpar cache"):
        # CORREÇÃO: limpa apenas o cache do vectordb, preservando
        # o cache da restauração automática para não redownlodar do Drive.
        load_vectordb_from_disk.clear()
        st.success("Cache limpo!")

# ==============================
# Execução principal
# ==============================
if gerar:
    if not patient.strip() or not q.strip():
        st.error("Preencha os dados do paciente e a pergunta.")
    else:
        with st.spinner("Buscando nos documentos e gerando resposta..."):
            patient_short = patient[:2000]
            vectordb = load_vectordb_from_disk()
            resp, hits = answer(q, patient_short, k=3, vectordb=vectordb)

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
