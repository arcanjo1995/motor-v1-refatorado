import streamlit as st
import os
from datetime import datetime
import json
import gc

# ---------------------------------------------------------
# IMPORTAÇÕES
# ---------------------------------------------------------
from config.settings import NOME_BASE_DEFINITIVA
from data.leitor_xls import LeitorXLS
from data.persistence import carregar_modelo_longo_prazo
from utils.math_engine import EngineMatematicoAvancado
from rules.contagens import MotorContagensProjetivas

from services.motor_unificado import motor_unificado
from services.auditoria import MotorV1Completo
from services.treinador import adicionar_a_base_longo_prazo

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="MOTOR V1 - Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS – TELA CHEIA, SEM CORTES
# ============================================================
st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none !important; }
    .main .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0.5rem;
        max-width: 1400px;
        margin: 0 auto;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.5rem; font-weight: 600; }
    .app-header .sub { color: rgba(255,255,255,0.7); font-size: 0.8rem; }
    .app-header .version { color: rgba(255,255,255,0.5); font-size: 0.7rem; background: rgba(255,255,255,0.1); padding: 0.2rem 0.8rem; border-radius: 20px; }
    .status-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-bottom: 0.8rem;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.3rem 0.6rem;
        border: 1px solid #e9ecef;
    }
    .status-item { flex: 1 1 120px; text-align: center; padding: 0.1rem 0.2rem; }
    .status-item .label { font-size: 0.55rem; text-transform: uppercase; color: #6c757d; font-weight: 600; }
    .status-item .value { font-size: 0.85rem; font-weight: 700; color: #1a1a2e; }
    .signal-badge {
        display: inline-block;
        padding: 0.3rem 1.4rem;
        border-radius: 40px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .signal-vermelho { background: #dc3545; color: white; }
    .signal-preto { background: #212529; color: white; }
    .signal-no-call { background: #ffc107; color: #212529; }
    .signal-neutro { background: #6c757d; color: white; }
    .streamlit-expanderHeader { font-weight: 600; font-size: 0.9rem; padding: 0.2rem 0.5rem; border-radius: 6px; background: #f8f9fa; }
    .streamlit-expanderContent { padding: 0.3rem 0.5rem 0.5rem 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.2rem; background: #f1f3f5; border-radius: 10px; padding: 0.2rem 0.5rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.2rem 1rem; font-weight: 500; background: transparent; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .stButton button { border-radius: 8px; font-weight: 600; transition: all 0.2s; }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .footer { margin-top: 1.5rem; padding-top: 0.6rem; border-top: 1px solid #e9ecef; text-align: center; font-size: 0.7rem; color: #6c757d; }
    .json-container { background: #f8f9fa; padding: 0.5rem; border-radius: 6px; border: 1px solid #e9ecef; }
    .progress-container { margin: 0.5rem 0; }
    @media (max-width: 768px) {
        .app-header h1 { font-size: 1.1rem; }
        .status-item { flex: 1 1 70px; }
        .status-item .value { font-size: 0.7rem; }
        .signal-badge { font-size: 0.9rem; padding: 0.2rem 0.8rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZAÇÃO DO MOTOR
# ============================================================
if "motor_v1" not in st.session_state:
    st.session_state.motor_v1 = motor_unificado
    with st.spinner("🧠 Inicializando..."):
        try:
            st.session_state.motor_v1.carregar_tudo()
        except Exception as e:
            st.error(f"Erro: {e}")
motor = st.session_state.motor_v1

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <div>
        <h1>🧠 MOTOR V1</h1>
        <div class="sub">Deep Learning · Q-Learning · Markov Multiescala</div>
    </div>
    <div class="version">v2.0</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STATUS BAR – DINÂMICA
# ============================================================
try:
    status = motor.status()
except:
    status = {}

def status_badge(ativo, texto_ativo, texto_inativo, cor_ativo="#28a745", cor_inativo="#dc3545"):
    if ativo:
        return f'<span style="color:{cor_ativo};font-weight:700;">{texto_ativo}</span>'
    else:
        return f'<span style="color:{cor_inativo};font-weight:700;">{texto_inativo}</span>'

st.markdown(f"""
<div class="status-grid">
    <div class="status-item">
        <div class="label">🤖 IA</div>
        <div class="value">{status_badge(status.get('ia_carregada', False), 'ATIVA', 'INATIVA')}</div>
    </div>
    <div class="status-item">
        <div class="label">📊 Base Longa</div>
        <div class="value">{status_badge(status.get('base_longa_carregada', False), 'CARREGADA', 'NÃO DETECTADA', '#28a745', '#ffc107')}</div>
    </div>
    <div class="status-item">
        <div class="label">⚡ Recência</div>
        <div class="value">{status_badge(status.get('recencia_injetada', False), 'ATIVA', 'AGUARDANDO', '#17a2b8', '#6c757d')}</div>
    </div>
    <div class="status-item">
        <div class="label">📈 Mestra</div>
        <div class="value">{status.get('volume_longo_prazo', 0)} Giros</div>
    </div>
    <div class="status-item">
        <div class="label">🧠 Imediata</div>
        <div class="value">{status.get('volume_recencia', 0)} Giros</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ABAS
# ============================================================
aba_tipo_b, aba_feedback, aba_tipo_d, aba_padroes, aba_matematica = st.tabs([
    "🎯 Sinal Real",
    "✅ Feedback & Correção",
    "📊 Auditoria & Treino",
    "📈 Padrões Aprendidos",
    "🧮 Cálculos Avançados"
])

# ============================================================
# ABA 1 — SINAL REAL (TODOS OS RELATÓRIOS RESTAURADOS)
# ============================================================
with aba_tipo_b:
    st.header("🎯 Sinal Real — Predição Neural")
    st.caption("Insira a sequência de 12 números e receba a predição da rede neural híbrida.")

    col_entrada, col_metrica = st.columns([3, 1], gap="medium")
    with col_entrada:
        entrada_numeros = st.text_input(
            "Digite os 12 números (separados por vírgula):",
            placeholder="Ex: 2,11,14,4,9,12,12,7,3,9,5,12",
            key="input_sequencia_real",
            label_visibility="collapsed"
        )
        btn_gerar = st.button("🚀 Executar Rede Neural", use_container_width=True, type="primary")
    with col_metrica:
        st.caption("📊 Pré-análise")
        if entrada_numeros:
            try:
                nums = [int(x) for x in entrada_numeros.replace(';', ',').replace('-', ',').replace(' ', '').split(",") if x.isdigit()]
                if len(nums) == 12:
                    pol = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in nums]
                    entropia = EngineMatematicoAvancado.calcular_entropia_shannon(pol)
                    st.metric("Entropia Estimada", f"{entropia:.2f} Bits", delta="Caos" if entropia > 1.52 else "Estruturado", delta_color="inverse")
            except:
                pass

    if btn_gerar:
        if not entrada_numeros:
            st.error("Por favor, insira uma sequência válida.")
        else:
            try:
                raw = entrada_numeros.replace(';', ',').replace('-', ',').replace(' ', '')
                lista_numeros = [int(x) for x in raw.split(",") if x.isdigit()]
                if len(lista_numeros) != 12:
                    st.error(f"Erro: Exatamente 12 números são necessários. Você enviou {len(lista_numeros)}.")
                else:
                    resultado = motor.gerar_sinal_tipo_b(lista_numeros)
                    polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in lista_numeros]

                    st.session_state.ultimo_sinal = {
                        "sequencia": lista_numeros,
                        "sinal": resultado.get("sinal", "NEUTRO"),
                        "justificativa": resultado.get("justificativa", ""),
                        "regra_id": resultado.get("regra_id", "DESCONHECIDO"),
                        "entropia": resultado.get("entropia", 0.0),
                        "probabilidade_markov": resultado.get("probabilidade_markov", {"V":0, "P":0, "B":0})
                    }

                    st.markdown("---")
                    st.markdown("### 🔮 CARD DE DECISÃO OPERACIONAL DA IA")

                    with st.container():
                        if resultado.get("no_call"):
                            st.error(f"🚨 **SINAL VETADO PELO AGENTE DE RISCO: NO CALL**")
                            st.write(f"**Justificativa Analítica:** {resultado.get('justificativa')}")
                        else:
                            sinal_final = resultado.get("sinal")
                            if sinal_final == "VERMELHO":
                                st.markdown("<h2 style='color: #FF4B4B;'>🔴 SINAL: ENTRAR NO VERMELHO</h2>", unsafe_allow_html=True)
                            elif sinal_final == "PRETO":
                                st.markdown("<h2 style='color: #1E1E1E; background-color: #F0F2F6; padding: 10px; border-radius: 5px;'>⚫ SINAL: ENTRAR NO PRETO</h2>", unsafe_allow_html=True)
                            else:
                                st.warning(f"⚪ **SINAL: {sinal_final}**")

                            st.write(f"**Direcionamento Matemático:** {resultado.get('justificativa')}")
                            st.write(f"**Confiança Estatística da Rede:** {resultado.get('confianca_ia')}%")

                            # Radar Numérico
                            radar = resultado.get("radar_numerico") or resultado.get("radar") or {}
                            with st.expander("📡 Relatório do Radar Numérico", expanded=False):
                                if radar:
                                    st.write("### Contexto Detectado")
                                    for chave, valor in radar.items():
                                        st.write(f"**{str(chave).replace('_',' ').title()}:** {valor}")
                                else:
                                    st.info("Nenhum dado detalhado do Radar Numérico disponível para esta janela.")

                            # Kelly
                            if resultado.get("kelly") is not None:
                                st.success(f"💰 **Gestão de Risco (Half-Kelly):** Aporte sugerido de **{resultado.get('kelly')}% da sua banca** para esta entrada.")

                    # Métricas lado a lado
                    if resultado.get("entropia") is not None:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            entropia_val = resultado.get('entropia', 0)
                            st.metric(label="Entropia de Shannon", value=f"{entropia_val} Bits", delta="Alta Imprevisibilidade" if entropia_val > 1.52 else "Mercado Estruturado", delta_color="inverse")
                        with col_m2:
                            markov = resultado.get("probabilidade_markov", {"V": 0, "P": 0, "B": 0})
                            st.metric(label="Cadeia de Markov", value=f"V: {markov.get('V', 0)}% | P: {markov.get('P', 0)}%")

                    # Expansores – todos os relatórios
                    col_exp1, col_exp2 = st.columns(2, gap="medium")
                    
                    with col_exp1:
                        with st.expander("📊 Regime de Recência Proporcional", expanded=False):
                            if resultado.get("regime_recencia"):
                                st.json(resultado["regime_recencia"])
                            else:
                                st.info("Nenhum regime disponível.")

                        with st.expander("🧮 Análise de Raridade", expanded=False):
                            raridade = EngineMatematicoAvancado.calcular_raridade_sequencia(polaridades)
                            st.write(f"**Streak Atual:** {raridade.get('streak')}x da cor {raridade.get('cor_sequencia')}")
                            st.write(f"**Prob. Continuação:** {raridade.get('probabilidade')}%")
                            st.info(f"**Status:** {raridade.get('status')}")

                        with st.expander("🔍 Auditoria de Raciocínio (Camadas)", expanded=False):
                            if resultado.get("raciocinio_trace"):
                                for camada in resultado["raciocinio_trace"]:
                                    st.markdown(f"**Camada {camada.get('camada')} — {camada.get('nome')}**")
                                    st.write(f"• *Saída:* `{camada.get('resultado')}`")
                                    st.write(f"• *Detalhe:* {camada.get('detalhe')}")
                                    st.markdown("---")
                            else:
                                st.info("Nenhum trace disponível.")

                        with st.expander("📌 Validação Contextual da Autoridade", expanded=False):
                            validacao = resultado.get("validacao_contextual_autoridade", {})
                            if validacao:
                                st.json(validacao)
                            else:
                                st.info("Nenhuma validação contextual disponível.")

                        with st.expander("📋 Auditoria Contrafactual", expanded=False):
                            auditoria = resultado.get("auditoria_contrafactual_autorizacao", {})
                            if auditoria:
                                st.json(auditoria)
                            else:
                                st.info("Nenhuma auditoria contrafactual registrada.")
                    
                    with col_exp2:
                        with st.expander("🧠 Regras Oficiais e Contagens Ativas", expanded=True):
                            try:
                                regras = MotorContagensProjetivas.mapear_janela(
                                    lista_numeros, polaridades, None, getattr(motor, "ia", None)
                                )
                                contagens = MotorContagensProjetivas._mapear_contagens(lista_numeros, polaridades)

                                st.caption("Evidências estruturais analisadas pelo motor.")

                                st.markdown("### 📌 Evidências estruturais")
                                if regras:
                                    for idx, r in enumerate(regras, 1):
                                        direcao = r.get("direcao", "NEUTRO")
                                        simbolo = "🔴" if direcao == "VERMELHO" else ("⚫" if direcao == "PRETO" else "⚪")
                                        st.markdown(f"**{idx}. {simbolo} {r.get('tipo_regra', 'REGRA')}**")
                                        st.write(f"• Família: `{r.get('familia', 'N/D')}` | Origem: `{r.get('origem', 'N/D')}`")
                                        st.write(f"• Direção: **{direcao}** | Peso: **{r.get('peso', 'N/D')}** | Validade: **{r.get('validade', 'N/D')}**")
                                        detalhes = {k: v for k, v in r.items() if k not in ("direcao", "tipo_regra", "origem", "peso", "familia", "validade")}
                                        if detalhes:
                                            st.json(detalhes)
                                        st.markdown("---")
                                else:
                                    st.info("Nenhuma regra ativa.")

                                st.markdown("### 🔢 Contagens projetivas")
                                if contagens:
                                    resumo = {}
                                    for c in contagens:
                                        st_cont = c.get("status", "DESCONHECIDO")
                                        resumo[st_cont] = resumo.get(st_cont, 0) + 1

                                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                                    with col_c1: st.metric("Abertas", resumo.get("ABERTA", 0))
                                    with col_c2: st.metric("Fechadas/Vivas", resumo.get("FECHADA", 0) + resumo.get("VIVA", 0))
                                    with col_c3: st.metric("Pagas", resumo.get("PAGA", 0))
                                    with col_c4: st.metric("Mortas", resumo.get("MORTA", 0))

                                    for idx, c in enumerate(contagens, 1):
                                        st.markdown(f"**Contagem {idx} — Número {c.get('numero')} — Status: `{c.get('status')}`**")
                                        st.write(f"• Origem: R{c.get('origem_posicao')} | Fechamento: R{c.get('fechamento_posicao')} | Casas: {c.get('casas_exigidas')}")
                                        st.write(f"• Coexistente: {'SIM' if c.get('coexistente') else 'NÃO'} | Transicional: {'SIM' if c.get('transicional') else 'NÃO'} | Assumida por: {c.get('assumida_por') or 'NÃO'}")
                                        st.markdown("---")
                                else:
                                    st.info("Nenhuma contagem projetiva aberta.")

                                # Leitura de arbitragem
                                familias = sorted({r.get("familia", "N/D") for r in regras})
                                st.markdown("### ⚖️ Arbitragem")
                                st.write(f"**Famílias ativas:** {', '.join(familias) if familias else 'NENHUMA'}")
                                st.write(f"**Regra vencedora:** `{resultado.get('regra_id', 'DESCONHECIDO')}`")
                                st.info("Evidências coexistem e apontam direções diferentes.")
                            except Exception as e:
                                st.warning(f"Erro ao carregar regras: {e}")

                        with st.expander("📈 Simulação de Rotas", expanded=False):
                            sim = resultado.get("simulacao_rotas_proximos_resultados", {})
                            if sim.get("ativo"):
                                st.json(sim)
                            else:
                                st.info("Simulação não disponível para esta janela.")

                        with st.expander("🧩 Confluência de Camadas Ampliadas", expanded=False):
                            confluencia = resultado.get("confluencia_camadas_ampliadas", {})
                            if confluencia:
                                st.json(confluencia)
                            else:
                                st.info("Nenhuma confluência disponível.")

                        with st.expander("⚖️ Oposição Causal (Streak)", expanded=False):
                            oposicao = resultado.get("oposicao_causal_consolidada", {})
                            if oposicao:
                                st.json(oposicao)
                            else:
                                st.info("Nenhuma oposição causal registrada.")

            except Exception as e:
                st.error(f"Erro ao gerar sinal: {e}")

# ============================================================
# ABA 2 — FEEDBACK
# ============================================================
with aba_feedback:
    st.header("✅ Reforço Preditivo (Q-Learning)")
    st.caption("A IA usa o feedback para atualizar recompensa e contexto. Somente os novos resultados são anexados cronologicamente.")

    if "ultimo_sinal" in st.session_state:
        st.info(f"📍 **Último sinal:** {st.session_state.ultimo_sinal['sinal']} | Sequência: `{st.session_state.ultimo_sinal['sequencia']}`")
    else:
        st.warning("⚠️ Nenhum sinal pendente. Gere um sinal na aba 'Sinal' primeiro.")

    col_f1, col_f2 = st.columns(2, gap="medium")
    with col_f1:
        entrada_feedback = st.text_input("Números reais saídos (separados por vírgula):", placeholder="Ex: 14, 0, 5", key="input_feedback")
    with col_f2:
        resultado_feedback = st.selectbox("Resultado final:", ["G0 (Acerto Alvo)", "G1 (Acerto 1º Gale)", "G2 (Risco Alto)", "LOSS / FALHA"])

    if st.button("💾 Enviar Recompensa/Punição", use_container_width=True):
        if not entrada_feedback:
            st.error("Insira os números reais.")
        else:
            try:
                raw = entrada_feedback.replace(';', ',').replace('-', ',').replace(' ', '')
                nums = [int(x) for x in raw.split(",") if x.isdigit()]
                with st.spinner("Registrando feedback..."):
                    if "ultimo_sinal" in st.session_state:
                        rel = motor.processar_feedback_real(
                            sequencia_12=st.session_state.ultimo_sinal["sequencia"],
                            sinal_indicado=st.session_state.ultimo_sinal["sinal"],
                            regra_id=st.session_state.ultimo_sinal["regra_id"],
                            numeros_saidos=nums,
                            classificacao=resultado_feedback,
                            entropia_shannon=st.session_state.ultimo_sinal.get("entropia", 0.0),
                            probabilidade_markov=st.session_state.ultimo_sinal.get("probabilidade_markov", {})
                        )
                        del st.session_state["ultimo_sinal"]
                    else:
                        dados = [{"numero": n, "cor": 'B' if n == 0 else ('V' if 1 <= n <= 7 else 'P')} for n in nums]
                        rel = adicionar_a_base_longo_prazo(dados)
                        motor.carregar_tudo()
                    
                    if rel and isinstance(rel, dict) and rel.get("sucesso"):
                        st.success("✅ Feedback absorvido com sucesso!")
                    else:
                        st.error(f"Erro: {rel.get('mensagem', 'Falha desconhecida')}")
            except Exception as e:
                st.error(f"Erro: {e}")

# ============================================================
# ABA 3 — AUDITORIA (OTIMIZADA)
# ============================================================
with aba_tipo_d:
    st.header("📊 Auditoria Dinâmica e Treinamento")
    st.caption("Gerenciamento de bases de dados e recálculo das matrizes de transição.")

    arquivo_upload = st.file_uploader("Envie seu arquivo Excel (.xlsx) com resultados históricos:", type=["xlsx"])

    # Armazena os dados lidos em session_state para evitar releitura
    if "dados_upload" not in st.session_state:
        st.session_state.dados_upload = None

    if arquivo_upload is not None:
        # Lê o arquivo apenas uma vez
        if st.session_state.dados_upload is None:
            with st.spinner("Lendo arquivo..."):
                caminho_temp = "temp_recencia.xlsx"
                with open(caminho_temp, "wb") as f:
                    f.write(arquivo_upload.getbuffer())
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados:
                    st.session_state.dados_upload = dados
                    st.session_state.arquivo_nome = arquivo_upload.name
                else:
                    st.error("Erro: Não foi possível ler os dados do arquivo.")
                if os.path.exists(caminho_temp):
                    try:
                        os.remove(caminho_temp)
                    except:
                        pass

        dados = st.session_state.dados_upload

        if dados:
            st.info(f"📄 Arquivo carregado: {st.session_state.arquivo_nome} ({len(dados)} registros)")
        else:
            st.error("Dados inválidos. Envie outro arquivo.")
            # Reseta o cache para permitir nova tentativa
            st.session_state.dados_upload = None

        if dados and len(dados) >= 20:
            col_a1, col_a2, col_a3 = st.columns(3, gap="medium")
            with col_a1:
                btn_recencia = st.button("⚡ Injetar como Recência", use_container_width=True)
            with col_a2:
                btn_substituir = st.button("💾 Substituir Base", use_container_width=True)
            with col_a3:
                btn_adicionar = st.button("➕ Encadear (Anexar)", use_container_width=True)

            # ===== RECÊNCIA =====
            if btn_recencia:
                try:
                    with st.spinner("Processando recência..."):
                        # Salva o arquivo de recência ativa
                        with open("base_recencia_ativa.xlsx", "wb") as f_rec:
                            f_rec.write(arquivo_upload.getvalue())
                        
                        resultado = motor.processar_recencia(dados)
                        st.success("✅ Recência injetada com sucesso!")

                        if resultado and isinstance(resultado, dict) and resultado.get("regime_recencia"):
                            with st.expander("📊 Relatório do Regime Injetado", expanded=True):
                                st.json(resultado["regime_recencia"])

                    # Auditoria (opcional, mas mantida)
                    with st.spinner("Auditando janelas..."):
                        motor_antigo = MotorV1Completo(dados)
                        output = motor_antigo.processar_auditoria()
                        linhas = output.split("\n")
                        janelas = [linha for linha in linhas if "Janela" in linha]
                        num_janelas = len(janelas)
                    
                    st.subheader(f"📝 Memória de Cálculo — {num_janelas} janelas analisadas")
                    st.text_area("Log Completo", output, height=500, key="auditoria_log_recencia")
                except Exception as e:
                    st.error(f"Erro: {e}")

            # ===== SUBSTITUIR BASE =====
            if btn_substituir:
                try:
                    with st.spinner("Substituindo base definitiva..."):
                        rel = motor.absorver_base_longa(dados)
                    if rel and rel.get("sucesso"):
                        st.success("✅ Base substituída e modelo retreinado!")
                        st.json(rel)
                    else:
                        st.error(f"Erro: {rel.get('mensagem')}")
                except Exception as e:
                    st.error(f"Erro: {e}")

            # ===== ENCADEAR =====
            if btn_adicionar:
                try:
                    with st.spinner("Anexando novos dados..."):
                        rel = motor.processar_novo_lote(dados)
                    if rel and rel.get("sucesso"):
                        st.success("✅ Dados anexados!")
                        st.json(rel)
                    else:
                        st.error(f"Erro: {rel.get('mensagem')}")
                except Exception as e:
                    st.error(f"Erro: {e}")
                    st.exception(e)
        else:
            if dados and len(dados) < 20:
                st.error("Mínimo 20 registros válidos são necessários.")
            # Se dados for None, já exibimos erro acima

    else:
        # Reset do cache quando o arquivo é removido
        st.session_state.dados_upload = None

# ============================================================
# ABA 4 — PADRÕES (COMPLETA)
# ============================================================
with aba_padroes:
    st.header("📈 Padrões Aprendidos e Memórias")
    st.caption("Varredura profunda da Q-Table e destrinchadores numéricos.")

    if st.button("🔄 Extrair Memória do Modelo", use_container_width=True, type="primary"):
        try:
            ia = carregar_modelo_longo_prazo()
            if ia is None:
                st.warning("Modelo não encontrado. Treine a base primeiro.")
            else:
                st.success("✅ Modelo carregado!")
                
                with st.expander("🤖 Q-Table (Agente RL)", expanded=False):
                    if ia.q_table:
                        st.write(f"**Estados contextuais aprendidos:** {len(ia.q_table)}")
                        st.json(ia.q_table)
                    else:
                        st.info("Q-Table vazia.")
                
                with st.expander("📐 Competência das Regras", expanded=False):
                    if hasattr(ia, 'regras_competencia_cronologica') and ia.regras_competencia_cronologica:
                        for regra, stats in ia.regras_competencia_cronologica.items():
                            st.write(f"**{regra}:** {stats.get('taxa_g0_g1', 0):.2f}% (n={stats.get('total_validacao', 0)})")
                    else:
                        st.info("Nenhuma competência registrada.")
                
                with st.expander("🔢 Comportamento Pós-Número (Completo)", expanded=False):
                    if hasattr(ia, 'analisar_comportamento_pos_numero'):
                        rel = ia.analisar_comportamento_pos_numero()
                        for num, dados in rel.items():
                            st.markdown(f"**Número {num}**")
                            st.write(f"• Total: {dados.get('total_aparicoes')}")
                            st.write(f"• Cor predominante: {dados.get('cor_mais_frequente_apos')} ({dados.get('frequencia_cor_dominante_%')}%)")
                            st.write(f"• Estabilidade: {dados.get('estabilidade')} | Saturação: {dados.get('saturacao')}")
                            st.write(f"• Tendência: {dados.get('tendencia_recente')}")
                            st.json(dados.get("distribuicao_pos"))
                            st.markdown("---")
        except Exception as e:
            st.error(f"Erro na extração: {e}")

# ============================================================
# ABA 5 — CÁLCULOS
# ============================================================
with aba_matematica:
    st.header("🧮 Engine Estatístico Avançado")
    st.caption("Análise de viés e simulador de gestão de banca.")

    if not os.path.exists(NOME_BASE_DEFINITIVA):
        st.warning(f"Arquivo base não encontrado: `{NOME_BASE_DEFINITIVA}`")
    else:
        try:
            janela = st.slider("Janela de análise (Surfe):", 20, 500, 100, 10)
            vies = EngineMatematicoAvancado.calcular_vies_surfe(NOME_BASE_DEFINITIVA, janela=janela)
            
            col_v1, col_v2 = st.columns(2, gap="medium")
            with col_v1:
                st.metric("Freq. Vermelho", f"{vies.get('frequencia_v')}%", delta=f"{vies.get('desvio_v')}%")
                st.metric("Freq. Preto", f"{vies.get('frequencia_p')}%", delta=f"{vies.get('desvio_p')}%")
            with col_v2:
                st.metric("Freq. Branco", f"{vies.get('frequencia_b')}%")
                st.info(f"**Parecer:** {vies.get('vies')}")
        except Exception as e:
            st.error(f"Erro: {e}")

    st.markdown("---")
    st.subheader("💰 Split Stake (Cobertura)")
    try:
        stake = st.number_input("Stake principal (R$):", 1.0, 5000.0, 10.0, 5.0)
        sim = EngineMatematicoAvancado.simular_split_stake_cobertura(stake)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Aporte na cor", f"R$ {sim.get('stake_cor'):.2f}")
            st.metric("Custo total", f"R$ {sim.get('custo_total_operacao'):.2f}")
        with col_s2:
            st.metric("Cobertura Branco (1/7)", f"R$ {sim.get('cobertura_b_ideal_1_7'):.2f}")
            st.metric("Cobertura Branco (1/10)", f"R$ {sim.get('cobertura_b_matematica_1_10'):.2f}")
        with col_s3:
            st.metric("Lucro se Branco", f"R$ {sim.get('lucro_liquido_se_der_branco'):.2f}")
            st.metric("House Edge", sim.get("house_edge_estatico"))
    except Exception as e:
        st.error(f"Erro: {e}")

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("""
<div class="footer">
    MOTOR V1 · Deep Learning para Roleta · v2.0 · 🧠
</div>
""", unsafe_allow_html=True)
