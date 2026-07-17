import streamlit as st
import os
from datetime import datetime
import json
import gc

from main import LeitorXLS
from main import MotorV1Completo
from main import adicionar_a_base_longo_prazo
from main import carregar_modelo_longo_prazo
from main import motor_unificado
from main import EngineMatematicoAvancado
from main import NOME_BASE_DEFINITIVA
from main import MotorContagensProjetivas

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="MOTOR V1 • DEEP LEARNING",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ESTILO VISUAL
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            min-width: 310px;
        }
        .soft-card {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.9rem;
        }
        .hero {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(135deg, rgba(90,90,90,0.10), rgba(30,30,30,0.02));
            margin-bottom: 1rem;
        }
        .hero h1, .hero h2, .hero h3, .hero p {
            margin: 0;
        }
        .compact-label {
            font-size: 0.84rem;
            opacity: 0.75;
            letter-spacing: 0.02em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FUNÇÕES DE APOIO VISUAL
# ============================================================
def card_html(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="compact-label">{title}</div>
            <h3 style="margin: 0.25rem 0 0.15rem 0;">{value}</h3>
            <div style="opacity: 0.78;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_badge(ok: bool, label: str) -> str:
    return f"✅ {label}" if ok else f"⚠️ {label}"

# ============================================================
# INICIALIZAÇÃO DO MOTOR NO SESSION STATE
# ============================================================
if "motor_v1" not in st.session_state:
    st.session_state.motor_v1 = motor_unificado
    with st.spinner("🧠 Inicializando MLP, Gradient Boosting, Markov e memórias contextuais..."):
        try:
            st.session_state.motor_v1.carregar_tudo()
            st.success("Motor de Machine Learning carregado e pronto.")
        except Exception as e:
            st.error(f"Erro protegido durante o boot inicial: {e}")

motor = st.session_state.motor_v1

# ============================================================
# TELA PRINCIPAL
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🧠 MOTOR V1</h1>
        <p>Interface repaginada para operação, leitura e análise do sistema.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAINEL LATERAL
# ============================================================
st.sidebar.markdown("## ⚙️ Status de Operação")

try:
    status_motor = motor.status()
except Exception:
    status_motor = {}
    st.sidebar.error("Erro ao ler status do motor.")

st.sidebar.markdown("### Núcleo")
st.sidebar.write(status_badge(bool(status_motor.get("ia_carregada")), "IA Preditiva"))
st.sidebar.write(status_badge(bool(status_motor.get("base_longa_carregada")), "Base Longa"))
st.sidebar.write(status_badge(bool(status_motor.get("recencia_injetada")), "Recência"))
st.sidebar.write(status_badge(bool(status_motor.get("motor_contagens_carregado")), "Contagens Projetivas"))

st.sidebar.markdown("### Memórias")
st.sidebar.write(status_badge(bool(status_motor.get("memoria_curta")), "Memória Curta"))
st.sidebar.write(status_badge(bool(status_motor.get("memoria_media")), "Memória Média"))
st.sidebar.write(status_badge(bool(status_motor.get("memoria_longa")), "Memória Longa"))

st.sidebar.markdown("### Base")
st.sidebar.write(f"**{NOME_BASE_DEFINITIVA}**")

# ============================================================
# RESUMO EM CARDS
# ============================================================
col1, col2, col3 = st.columns(3)
with col1:
    card_html("IA Preditiva", "Ativa" if status_motor.get("ia_carregada") else "Inativa", "Estado geral do modelo")
with col2:
    card_html("Base Longa", "Carregada" if status_motor.get("base_longa_carregada") else "Ausente", "Histórico consolidado")
with col3:
    card_html("Recência", "Injetada" if status_motor.get("recencia_injetada") else "Não detectada", "Ajuste de peso recente")

st.divider()

# ============================================================
# ÁREA PRINCIPAL EM DUAS COLUNAS
# ============================================================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("### 📥 Entrada e Operação")
    arquivo = st.file_uploader("Carregar arquivo XLS", type=["xls", "xlsx"])

    acao = st.selectbox(
        "Escolha a ação",
        [
            "Analisar arquivo",
            "Adicionar à base longa",
            "Recarregar modelo",
            "Limpar memória",
        ],
    )

    executar = st.button("Executar ação", use_container_width=True)

    if executar:
        if acao == "Analisar arquivo":
            if arquivo is None:
                st.warning("Envie um arquivo XLS/XLSX para continuar.")
            else:
                try:
                    leitor = LeitorXLS(arquivo)
                    dados = leitor.ler()
                    st.success("Arquivo lido com sucesso.")
                    st.write(dados[:5] if hasattr(dados, "__getitem__") else dados)
                except Exception as e:
                    st.error(f"Falha ao analisar o arquivo: {e}")

        elif acao == "Adicionar à base longa":
            st.info("Operação acionada: adicionar à base longa.")
            try:
                resultado = adicionar_a_base_longo_prazo()
                st.success(f"Resultado: {resultado}")
            except Exception as e:
                st.error(f"Erro ao adicionar à base: {e}")

        elif acao == "Recarregar modelo":
            try:
                motor.carregar_tudo()
                st.success("Modelo recarregado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao recarregar modelo: {e}")

        elif acao == "Limpar memória":
            try:
                gc.collect()
                st.success("Memória otimizada com gc.collect().")
            except Exception as e:
                st.error(f"Erro ao limpar memória: {e}")

with right:
    st.markdown("### 📊 Diagnóstico do Sistema")
    if status_motor:
        st.json(status_motor)
    else:
        st.info("Sem dados de status disponíveis no momento.")

st.divider()

# ============================================================
# SEÇÃO INFERIOR
# ============================================================
tab1, tab2, tab3 = st.tabs(["Resumo", "Motor", "Logs"])

with tab1:
    st.markdown("### Resumo operacional")
    st.write("Interface reorganizada para leitura rápida, status e execução central.")

with tab2:
    st.markdown("### Estado do motor")
    try:
        st.write(motor.status())
    except Exception as e:
        st.error(f"Não foi possível obter o status: {e}")

with tab3:
    st.markdown("### Log visual")
    st.caption("Use esta área para mensagens, exceções e eventos operacionais.")
    st.write("Pronto para integrar novos logs ou relatórios.")

# ============================================================
# RODAPÉ
# ============================================================
st.caption(f"© {datetime.now().year} • MOTOR V1 • Layout repaginado")
