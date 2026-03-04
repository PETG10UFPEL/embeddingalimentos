import os
from dotenv import load_dotenv
load_dotenv(override=True)  # DEVE ser antes de qualquer outro import

import base64
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

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


st.set_page_config(page_title="Embeddings Alimentos - PET G10 UFPel", layout="wide")

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
    font-size: 1.0rem;      /* equivalente a ~12pt no Word */
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
    st.title("🥗 Embeddings Alimentos - PET G10 UFPel")

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
# Assinatura — tamanho ~12pt
# ==============================
st.markdown("""
<p class="info-text" style="margin-bottom:0.8rem;">
  <strong>Patrícia Xavier Bittencourt</strong>, estudante &middot;
  Disciplina 15001103 M1 &mdash; Princípios de Inteligência Artificial Aplicados &middot;
  UFPel (2025/6-2) &middot; Prof. Alejandro Martins R. &middot;
  Sistema elaborado em parceria junto à P&amp;D do Projeto PET UFPel Saúde Digital &mdash;
  Telemonitoramento de Feridas Crônicas.
</p>
""", unsafe_allow_html=True)

# ==============================
# Texto explicativo RAG + link embeddings
# ==============================

def _html_com_imagens_embutidas(html_path: Path) -> str:
    """Lê o HTML e substitui src de imagens locais por base64."""
    import re
    html = html_path.read_text(encoding="utf-8", errors="replace")
    assets_dir = html_path.parent

    def substituir(match):
        src = match.group(1)
        # só processa caminhos relativos (não http)
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

_emb_path = Path(__file__).parent / "assets" / "embeddings_alimentos.html"
if _emb_path.exists():
    with st.expander("📊 Ver Embeddings dos Alimentos", expanded=False):
        _html_content = _html_com_imagens_embutidas(_emb_path)
        components.html(_html_content, height=700, scrolling=True)
else:
    st.caption("_(arquivo embeddings_alimentos.html não encontrado em assets/)_")

st.divider()

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
# Componente de voz (reutilizável)
# ==============================
def mic_component(target_label: str, field_index: int):
    """Injeta botão de microfone que preenche o textarea pelo índice."""
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
    placeholder="Ex.: sugerir plano alimentar de 7 dias, ou ajustar dieta para cicatrização, etc.",
    label_visibility="collapsed",
)

# ==============================
# Cache
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def cached_answer(q, patient, k):
    return answer(q, patient, k=k)

# ==============================
# Botões
# ==============================
col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    gerar = st.button("🚀 Gerar resposta", type="primary")
with col_btn2:
    if st.button("🗑️ Limpar cache"):
        st.cache_data.clear()
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
