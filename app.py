import streamlit as st
import os
from datetime import datetime
import json
import gc

# ---------------------------------------------------------
# IMPORTAÇÕES DA NOVA ARQUITETURA MODULAR
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
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    /* Reset de margens */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Cards de status horizontal */
    .status-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #4CAF50;
        text-align: center;
    }
    .status-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
    }
    .status-card .value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    /* Card de resultado do sinal */
    .signal-result {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .signal-badge {
        display: inline-block;
        padding: 0.4rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .signal-vermelho { background: #dc3545; color: white; }
    .signal-preto { background: #212529; color: white; }
    .signal-no-call { background: #ffc107; color: #212529; }
    .signal-neutro { background: #6c757d; color: white; }
    /* Expanders com título em negrito */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a1a2e;
    }
    /* Footer */
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
        text-align: center;
        font-size: 0.75rem;
        color: #6c757d;
    }
    /* Remover espaçamento extra do sidebar */
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZAÇÃO DO MOTOR
# ============================================================
if "motor_v1" not in st.session_state:
    st.session_state.motor_v1 = motor_unificado
    with st.spinner("🧠 Inicializando MLP, Gradient Boosting, Markov e memórias contextuais..."):
        try:
            st.session_state.motor_v1.carregar_tudo()
        except Exception as e:
            st.error(f"Erro durante o boot: {e}")
motor = st.session_state.motor_v1

# ============================================================
# 1. BARRA DE STATUS HORIZONTAL (TOPO)
# ============================================================
try:
    status = motor.status()
except:
    status = {}

st.markdown("### 🧠 Motor V1 — Status Operacional")
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
with col_s1:
    status_text = "🟢 ATIVA" if status.get("ia_carregada") else "🔴 INATIVA"
    st.markdown(f"""
    <div class="status-card" style="border-left-color: {'#28a745' if status.get('ia_carregada') else '#dc3545'};">
        <div class="label">IA Preditiva</div>
        <div class="value">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)
with col_s2:
    status_text = "✅ CARREGADA" if status.get("base_longa_carregada") else "❌ NÃO DETECTADA"
    st.markdown(f"""
    <div class="status-card" style="border-left-color: {'#28a745' if status.get('base_longa_carregada') else '#ffc107'};">
        <div class="label">Base Longa</div>
        <div class="value">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)
with col_s3:
    status_text = "⚡ ATIVA" if status.get("recencia_injetada") else "⏳ AGUARDANDO"
    st.markdown(f"""
    <div class="status-card" style="border-left-color: {'#17a2b8' if status.get('recencia_injetada') else '#6c757d'};">
        <div class="label">Recência (Peso 6)</div>
        <div class="value">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)
with col_s4:
    st.markdown(f"""
    <div class="status-card" style="border-left-color: #6f42c1;">
        <div class="label">Base Mestra</div>
        <div class="value">{status.get('volume_longo_prazo', 0)} Giros</div>
    </div>
    """, unsafe_allow_html=True)
with col_s5:
    st.markdown(f"""
    <div class="status-card" style="border-left-color: #fd7e14;">
        <div class="label">Memória Imediata</div>
        <div class="value">{status.get('volume_recencia', 0)} Giros</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 2. SIDEBAR (LIMPA E COMPACTA)
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Navegação")
    aba_tipo_b, aba_feedback, aba_tipo_d, aba_padroes, aba_matematica = st.tabs([
        "🎯 Sinal", "✅ Feedback", "📊 Auditoria", "📈 Padrões", "🧮 Cálculos"
    ])
    
    st.markdown("---")
    if status.get("ultima_atualizacao"):
        st.caption(f"🕒 Último sync: {status.get('ultima_atualizacao')}")
    st.caption("v2.0 · Motor Refatorado")

# ============================================================
# 3. ABA 1 — SINAL REAL (REFORMULADA)
# ============================================================
with aba_tipo_b:
    st.header("🎯 Sinal Real — Predição Neural")
    
    # Layout de entrada em duas colunas
    col_entrada, col_metrica = st.columns([2, 1])
    with col_entrada:
        entrada_numeros = st.text_input(
            "Digite os 12 últimos números (separados por vírgula):",
            placeholder="Ex: 2,11,14,4,9,12,12,7,3,9,5,12",
            key="input_sequencia_real",
            label_visibility="collapsed"
        )
        btn_gerar = st.button("🚀 Executar Rede Neural", use_container_width=True, type="primary")
    
    with col_metrica:
        st.caption("📊 Pré-análise rápida")
        if entrada_numeros:
            try:
                nums = [int(x) for x in entrada_numeros.replace(';', ',').replace('-', ',').replace(' ', '').split(",") if x.isdigit()]
                if len(nums) == 12:
                    polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in nums]
                    entropia = EngineMatematicoAvancado.calcular_entropia_shannon(polaridades)
                    st.metric("Entropia Estimada", f"{entropia:.2f} Bits", delta="Caos" if entropia > 1.52 else "Estruturado", delta_color="inverse")
            except:
                pass

    # Processamento do sinal
    if btn_gerar:
        if not entrada_numeros:
            st.error("Por favor, insira uma sequência válida.")
        else:
            try:
                raw_str = str(entrada_numeros).replace(';', ',').replace('-', ',').replace(' ', '')
                lista_numeros = [int(x) for x in raw_str.split(",") if x.isdigit()]
                
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
                    
                    # Card de resultado
                    st.markdown("### 🔮 Card de Decisão")
                    with st.container():
                        col_res1, col_res2 = st.columns([2, 1])
                        with col_res1:
                            if resultado.get("no_call"):
                                st.markdown("<div class='signal-badge signal-no-call'>⚠️ NO CALL</div>", unsafe_allow_html=True)
                                st.warning(f"**Motivo:** {resultado.get('justificativa')}")
                            else:
                                sinal = resultado.get("sinal")
                                if sinal == "VERMELHO":
                                    st.markdown("<div class='signal-badge signal-vermelho'>🔴 ENTRAR NO VERMELHO</div>", unsafe_allow_html=True)
                                elif sinal == "PRETO":
                                    st.markdown("<div class='signal-badge signal-preto'>⚫ ENTRAR NO PRETO</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div class='signal-badge signal-neutro'>⚪ {sinal}</div>", unsafe_allow_html=True)
                                
                                st.write(f"**Direcionamento:** {resultado.get('justificativa')}")
                                st.write(f"**Confiança da Rede:** {resultado.get('confianca_ia')}%")
                        with col_res2:
                            if resultado.get("entropia") is not None:
                                st.metric("Entropia (Shannon)", f"{resultado.get('entropia', 0):.2f} Bits")
                            markov = resultado.get("probabilidade_markov", {})
                            st.caption(f"Markov: V: {markov.get('V', 0)}% | P: {markov.get('P', 0)}%")
                            
                            if resultado.get("regime_recencia"):
                                regime = resultado["regime_recencia"]
                                st.caption(f"Regime: {regime.get('modo_dominante', 'N/D')} | Viés: {regime.get('viés_atual', 'N/D')}")

                    # Expansores organizados em duas colunas
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        with st.expander("📊 Regime de Recência", expanded=False):
                            if resultado.get("regime_recencia"):
                                st.json(resultado["regime_recencia"])
                            else:
                                st.info("Nenhum regime disponível.")
                        
                        with st.expander("🧮 Análise de Raridade", expanded=False):
                            raridade = EngineMatematicoAvancado.calcular_raridade_sequencia(polaridades)
                            st.write(f"**Streak:** {raridade.get('streak')}x da cor {raridade.get('cor_sequencia')}")
                            st.write(f"**Probabilidade de continuação:** {raridade.get('probabilidade')}%")
                            st.info(f"**Status:** {raridade.get('status')}")
                        
                        with st.expander("🔍 Auditoria de Raciocínio (Camadas)", expanded=False):
                            if resultado.get("raciocinio_trace"):
                                for camada in resultado["raciocinio_trace"][-3:]:  # Últimas camadas são as mais importantes
                                    st.markdown(f"**Camada {camada.get('camada')} — {camada.get('nome')}**")
                                    st.write(f"*{camada.get('resultado')}*")
                                    st.caption(camada.get('detalhe', ''))
                                    st.markdown("---")
                            else:
                                st.info("Nenhum trace disponível.")
                    
                    with col_exp2:
                        with st.expander("🧠 Regras e Contagens Ativas", expanded=False):
                            try:
                                regras = MotorContagensProjetivas.mapear_janela(
                                    lista_numeros, polaridades, None, getattr(motor, "ia", None)
                                )
                                if regras:
                                    for r in regras[:5]:  # Limita a 5 para não poluir
                                        direcao = r.get("direcao", "NEUTRO")
                                        emoji = "🔴" if direcao == "VERMELHO" else ("⚫" if direcao == "PRETO" else "⚪")
                                        st.write(f"{emoji} **{r.get('tipo_regra')}** — *{r.get('familia')}*")
                                else:
                                    st.info("Nenhuma regra ativa.")
                            except Exception as e:
                                st.warning(f"Erro ao carregar regras: {e}")
                        
                        with st.expander("📈 Simulação de Rotas (Próximos Resultados)", expanded=False):
                            if resultado.get("simulacao_rotas_proximos_resultados", {}).get("ativo"):
                                st.json(resultado["simulacao_rotas_proximos_resultados"])
                            else:
                                st.info("Simulação não disponível para esta janela.")

            except Exception as e:
                st.error(f"Erro ao gerar sinal: {e}")

# ============================================================
# 4. ABA 2 — FEEDBACK (MANTIDA, MAS COM LAYOUT MAIS LIMPO)
# ============================================================
with aba_feedback:
    st.header("✅ Reforço Preditivo (Q-Learning)")
    st.caption("A IA usa o feedback para atualizar recompensa e contexto. Somente os novos resultados são anexados cronologicamente.")

    if "ultimo_sinal" in st.session_state:
        st.info(f"📍 **Último sinal:** {st.session_state.ultimo_sinal['sinal']} | Sequência: `{st.session_state.ultimo_sinal['sequencia']}`")
    else:
        st.warning("⚠️ Nenhum sinal pendente. Gere um sinal na aba 'Sinal' primeiro.")

    col_f1, col_f2 = st.columns(2)
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
# 5. ABA 3 — AUDITORIA (MANTIDA)
# ============================================================
with aba_tipo_d:
    st.header("📊 Auditoria Dinâmica e Treinamento")
    st.caption("Gerenciamento de bases de dados e recálculo das matrizes de transição.")

    arquivo_upload = st.file_uploader("Envie seu arquivo Excel (.xlsx) com resultados históricos:", type=["xlsx"])

    if arquivo_upload is not None:
        caminho_temp = "temp_recencia.xlsx"
        with open(caminho_temp, "wb") as f:
            f.write(arquivo_upload.getbuffer())
        st.info("Arquivo recebido. Selecione a ação:")

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            btn_recencia = st.button("⚡ Injetar como Recência", use_container_width=True)
        with col_a2:
            btn_substituir = st.button("💾 Substituir Base", use_container_width=True)
        with col_a3:
            btn_adicionar = st.button("➕ Encadear (Anexar)", use_container_width=True)

        if btn_recencia:
            try:
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados and len(dados) >= 20:
                    with open("base_recencia_ativa.xlsx", "wb") as f:
                        f.write(arquivo_upload.getvalue())
                    with st.spinner("Processando recência..."):
                        resultado = motor.processar_recencia(dados)
                    st.success("✅ Recência injetada!")
                    if resultado.get("regime_recencia"):
                        st.json(resultado["regime_recencia"])
                else:
                    st.error("Mínimo 20 registros válidos.")
            except Exception as e:
                st.error(f"Erro: {e}")

        if btn_substituir:
            try:
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados:
                    with st.spinner("Substituindo base..."):
                        rel = motor.absorver_base_longa(dados)
                    if rel and rel.get("sucesso"):
                        st.success("✅ Base substituída!")
                        st.json(rel)
                    else:
                        st.error(f"Erro: {rel.get('mensagem')}")
                else:
                    st.error("Dados inválidos.")
            except Exception as e:
                st.error(f"Erro: {e}")

        if btn_adicionar:
            try:
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados:
                    with st.spinner("Anexando dados..."):
                        rel = motor.processar_novo_lote(dados)
                    if rel and rel.get("sucesso"):
                        st.success("✅ Dados anexados!")
                        st.json(rel)
                    else:
                        st.error(f"Erro: {rel.get('mensagem')}")
                else:
                    st.error("Dados inválidos.")
            except Exception as e:
                st.error(f"Erro: {e}")
                st.exception(e)
        
        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)

# ============================================================
# 6. ABA 4 — PADRÕES (MANTIDA, MAS COM BOTÃO MAIS DESTACADO)
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
                        for regra, stats in list(ia.regras_competencia_cronologica.items())[:20]:
                            st.write(f"**{regra}:** {stats.get('taxa_g0_g1', 0):.2f}% (n={stats.get('total_validacao', 0)})")
                    else:
                        st.info("Nenhuma competência registrada.")
                
                with st.expander("🔢 Comportamento Pós-Número", expanded=False):
                    if hasattr(ia, 'analisar_comportamento_pos_numero'):
                        rel = ia.analisar_comportamento_pos_numero()
                        for num, dados in list(rel.items())[:10]:
                            st.write(f"**Número {num}:** {dados.get('cor_mais_frequente_apos')} ({dados.get('frequencia_cor_dominante_%')}%)")
        except Exception as e:
            st.error(f"Erro na extração: {e}")

# ============================================================
# 7. ABA 5 — CÁLCULOS (MANTIDA)
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
            
            col_v1, col_v2 = st.columns(2)
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
