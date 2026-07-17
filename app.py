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

# Importação dos Serviços
from services.motor_unificado import motor_unificado
from services.auditoria import MotorV1Completo
from services.treinador import adicionar_a_base_longo_prazo

# ============================================================
# CONFIGURAÇÃO DA PÁGINA (com tema claro/escuro personalizado)
# ============================================================
st.set_page_config(
    page_title="MOTOR V1 - Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONALIZADO PARA UM LAYOUT MAIS MODERNO
# ============================================================
st.markdown("""
<style>
    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.2rem 0 0 0;
        opacity: 0.85;
        font-size: 1rem;
    }
    /* Cards de métricas */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border-left: 4px solid #4CAF50;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    /* Botões estilizados */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    /* Expanders com borda suave */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a1a2e;
    }
    /* Badge de sinal */
    .signal-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .signal-vermelho {
        background: #ff4b4b;
        color: white;
    }
    .signal-preto {
        background: #1e1e1e;
        color: white;
    }
    .signal-no-call {
        background: #ffa500;
        color: white;
    }
    /* Rodapé */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
        text-align: center;
        font-size: 0.8rem;
        color: #6c757d;
    }
    /* Sidebar com fundo diferenciado */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    /* Ajuste para telas menores */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5rem; }
        .metric-card .value { font-size: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZAÇÃO DO MOTOR NO SESSION STATE
# ============================================================
if "motor_v1" not in st.session_state:
    st.session_state.motor_v1 = motor_unificado
    with st.spinner("🧠 Inicializando MLP, Gradient Boosting, Markov e memórias contextuais..."):
        try:
            st.session_state.motor_v1.carregar_tudo()
            st.success("Motor de Machine Learning Carregado e Pronto!")
        except Exception as e:
            st.error(f"Erro protegido durante o boot inicial: {e}")

motor = st.session_state.motor_v1

# ============================================================
# CABEÇALHO PRINCIPAL
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧠 MOTOR V1 — Deep Learning &amp; Q-Learning</h1>
    <p>Arquitetura ativa: MLP · Gradient Boosting · Markov multiescala · Memórias contextuais · Meta-confluência adaptativa</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAINEL LATERAL — STATUS DO SISTEMA (REFORMULADO)
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Status de Operação")
    
    try:
        status_motor = motor.status()
    except Exception as e:
        status_motor = {}
        st.error("Erro ao ler status do motor.")

    col1, col2 = st.columns(2)
    with col1:
        if status_motor.get("ia_carregada"):
            st.success("🤖 IA Preditiva: **ATIVA**")
        else:
            st.error("🤖 IA Preditiva: **INATIVA**")
    with col2:
        if status_motor.get("base_longa_carregada"):
            st.success("📊 Base Longa: **CARREGADA**")
        else:
            st.warning("📊 Base Longa: **NÃO DETECTADA**")

    if status_motor.get("recencia_injetada"):
        st.success("⚡ Recência: **INJETADA** (Peso 6)")
    else:
        st.info("⚡ Recência: **AGUARDANDO**")

    st.markdown("---")
    st.markdown("### 🧠 Tamanho do Cérebro Ativo")
    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            label="Base Mestra",
            value=f"{status_motor.get('volume_longo_prazo', 0)} Giros"
        )
    with col4:
        st.metric(
            label="Memória Imediata",
            value=f"{status_motor.get('volume_recencia', 0)} Giros"
        )

    if status_motor.get("ultima_atualizacao"):
        st.caption(f"🕒 Último sincronismo: {status_motor.get('ultima_atualizacao')}")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem; color:#6c757d; text-align:center;">
        Versão 2.0 · Motor Refatorado
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# INTERFACE PRINCIPAL — ABAS RENOMEADAS COM ÍCONES
# ============================================================
aba_tipo_b, aba_feedback, aba_tipo_d, aba_padroes, aba_matematica = st.tabs([
    "🎯 Sinal Real",
    "✅ Feedback & Correção",
    "📊 Auditoria & Treino",
    "📈 Padrões Aprendidos",
    "🧮 Cálculos Avançados"
])

# =========================================================================
# ABA 1 — SINAL REAL (TIPO B)
# =========================================================================
with aba_tipo_b:
    st.header("🎯 Predição via Redes Neurais e XGBoost")
    st.caption("O Juiz Preditivo utiliza pesos dinâmicos gerados por Machine Learning. O Agente RL (Q-Learning) gerencia ativamente as decisões para maximizar lucro.")

    with st.container():
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            entrada_numeros = st.text_input(
                "Digite os 12 últimos números da sequência separados por vírgula:",
                placeholder="Ex: 2,11,14,4,9,12,12,7,3,9,5,12",
                key="input_sequencia_real"
            )
        with col_btn:
            st.write("")  # espaçamento
            st.write("")
            btn_gerar = st.button("🚀 Executar Rede Neural", key="btn_gerar_sinal", use_container_width=True)

    if btn_gerar:
        if not entrada_numeros:
            st.error("Por favor, insira uma sequência válida de números.")
        else:
            try:
                raw_str = str(entrada_numeros).replace(';', ',').replace('-', ',').replace(' ', '')
                lista_numeros = [int(x) for x in raw_str.split(",") if x.isdigit()]
                
                if len(lista_numeros) != 12:
                    st.error(f"Erro operacional: São necessários exatamente 12 números. Você enviou {len(lista_numeros)}.")
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
                    st.subheader("🔮 Card de Decisão Operacional")

                    # Exibição do sinal com badge estilizado
                    if resultado.get("no_call"):
                        st.error("🚨 **SINAL VETADO PELO AGENTE DE RISCO: NO CALL**")
                        st.info(f"**Justificativa:** {resultado.get('justificativa')}")
                    else:
                        sinal_final = resultado.get("sinal")
                        if sinal_final == "VERMELHO":
                            st.markdown("<div class='signal-badge signal-vermelho'>🔴 SINAL: ENTRAR NO VERMELHO</div>", unsafe_allow_html=True)
                        elif sinal_final == "PRETO":
                            st.markdown("<div class='signal-badge signal-preto'>⚫ SINAL: ENTRAR NO PRETO</div>", unsafe_allow_html=True)
                        else:
                            st.warning(f"⚪ **SINAL: {sinal_final}**")

                        st.write(f"**Direcionamento Matemático:** {resultado.get('justificativa')}")
                        st.write(f"**Confiança Estatística da Rede:** {resultado.get('confianca_ia')}%")
                        
                        if resultado.get("kelly") is not None:
                            st.success(f"💰 **Gestão de Risco (Half-Kelly):** Aporte sugerido de **{resultado.get('kelly')}% da sua banca** para esta entrada.")

                    # Métricas lado a lado
                    if resultado.get("entropia") is not None:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            entropia_val = resultado.get('entropia', 0)
                            st.metric(
                                label="Entropia de Shannon (Nível de Caos)",
                                value=f"{entropia_val} Bits",
                                delta="Alta Imprevisibilidade" if entropia_val > 1.52 else "Mercado Estruturado",
                                delta_color="inverse"
                            )
                        with col_m2:
                            markov = resultado.get("probabilidade_markov", {"V": 0, "P": 0, "B": 0})
                            st.metric(
                                label="Probabilidade Pura (Cadeia de Markov)",
                                value=f"V: {markov.get('V', 0)}% | P: {markov.get('P', 0)}%"
                            )

                    if resultado.get("regime_recencia"):
                        with st.expander("📊 Regime de Recência Proporcional (Filtro Dinâmico)", expanded=True):
                            st.json(resultado["regime_recencia"])

                    # Raridade
                    raridade_atual = EngineMatematicoAvancado.calcular_raridade_sequencia(polaridades)
                    with st.expander("🧮 Análise de Raridade da Janela Atual", expanded=False):
                        st.write(f"**Streak Atual Detectado:** {raridade_atual.get('streak')}x da cor {raridade_atual.get('cor_sequencia')}")
                        st.write(f"**Probabilidade Teórica de Continuação:** {raridade_atual.get('probabilidade')}%")
                        st.info(f"**Status Estrutural:** {raridade_atual.get('status')}")

                    # Auditoria de regras
                    try:
                        regras_oficiais_ativas = MotorContagensProjetivas.mapear_janela(
                            lista_numeros,
                            polaridades,
                            None,
                            getattr(motor, "ia", None)
                        )
                        contagens_mapeadas = MotorContagensProjetivas._mapear_contagens(
                            lista_numeros,
                            polaridades
                        )

                        with st.expander("🧠 Regras Oficiais e Contagens Ativas", expanded=True):
                            st.caption("Auditoria visual das evidências estruturais já analisadas pelo motor e encaminhadas à arbitragem do sinal.")

                            st.markdown("### 📌 Evidências estruturais detectadas")
                            if regras_oficiais_ativas:
                                for idx, regra in enumerate(regras_oficiais_ativas, 1):
                                    direcao = regra.get("direcao", "NEUTRO")
                                    simbolo = "🔴" if direcao == "VERMELHO" else ("⚫" if direcao == "PRETO" else "⚪")
                                    st.markdown(f"**{idx}. {simbolo} {regra.get('tipo_regra', 'REGRA_NAO_IDENTIFICADA')}**")
                                    st.write(f"• Família: `{regra.get('familia', 'N/D')}` | Origem: `{regra.get('origem', 'N/D')}`")
                                    st.write(f"• Direção: **{direcao}** | Peso: **{regra.get('peso', 'N/D')}** | Validade: **{regra.get('validade', 'N/D')}**")
                                    detalhes = {k: v for k, v in regra.items() if k not in ("direcao", "tipo_regra", "origem", "peso", "familia", "validade")}
                                    if detalhes:
                                        st.json(detalhes)
                                    st.markdown("---")
                            else:
                                st.info("Nenhuma regra oficial com consequência ativa no fechamento desta janela.")

                            st.markdown("### 🔢 Mapa completo das contagens projetivas")
                            if contagens_mapeadas:
                                resumo_status = {}
                                for c in contagens_mapeadas:
                                    status = c.get("status", "DESCONHECIDO")
                                    resumo_status[status] = resumo_status.get(status, 0) + 1

                                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                                with col_c1: st.metric("Abertas", resumo_status.get("ABERTA", 0))
                                with col_c2: st.metric("Fechadas/Vivas", resumo_status.get("FECHADA", 0) + resumo_status.get("VIVA", 0))
                                with col_c3: st.metric("Pagas", resumo_status.get("PAGA", 0))
                                with col_c4: st.metric("Mortas", resumo_status.get("MORTA", 0))

                                for idx, c in enumerate(contagens_mapeadas, 1):
                                    st.markdown(f"**Contagem {idx} — Número {c.get('numero')} — Status: `{c.get('status')}`**")
                                    st.write(f"• Origem: R{c.get('origem_posicao')} | Fechamento: R{c.get('fechamento_posicao')} | Casas: {c.get('casas_exigidas')}")
                                    st.write(f"• Coexistente: {'SIM' if c.get('coexistente') else 'NÃO'} | Transicional: {'SIM' if c.get('transicional') else 'NÃO'} | Assumida por: {c.get('assumida_por') or 'NÃO'}")
                                    st.markdown("---")
                            else:
                                st.info("Nenhuma contagem projetiva de 1 a 7 foi aberta nesta janela.")

                            familias = sorted({r.get("familia", "N/D") for r in regras_oficiais_ativas})
                            st.markdown("### ⚖️ Leitura de arbitragem")
                            st.write(f"**Famílias estruturais ativas:** {', '.join(familias) if familias else 'NENHUMA'}")
                            st.write(f"**Regra vencedora registrada pelo sinal:** `{resultado.get('regra_id', 'DESCONHECIDO')}`")
                            st.info("As evidências podem coexistir e apontar direções diferentes. O painel não elimina regras concorrentes: mostra o mapa estrutural analisado pelo motor.")
                    except Exception as e:
                        st.warning(f"A geração do sinal foi concluída, mas a auditoria visual das regras não pôde ser exibida. Detalhe: {e}")

                    if resultado.get("raciocinio_trace"):
                        with st.expander("🔍 Auditoria de Raciocínio por Camadas Neurais", expanded=False):
                            for camada in resultado["raciocinio_trace"]:
                                st.markdown(f"**Camada {camada.get('camada')} — {camada.get('nome')}**")
                                st.write(f"• *Saída do Módulo:* `{camada.get('resultado')}`")
                                st.write(f"• *Interpretação:* {camada.get('detalhe')}")
                                st.markdown("---")

            except Exception as e:
                st.error(f"🚨 Proteção de Crash Ativada: Ocorreu um erro ao gerar o sinal. Detalhe: {e}")

# =========================================================================
# ABA 2 — FEEDBACK E CORREÇÃO
# =========================================================================
with aba_feedback:
    st.header("✅ Reforço Preditivo (Q-Learning)")
    st.caption("A IA usa o feedback para atualizar recompensa e contexto. A janela de 12 permanece apenas como contexto; somente os novos resultados reais são anexados cronologicamente à recência e à base.")

    if "ultimo_sinal" in st.session_state:
        st.info(f"📍 **Último Estado Computado:** Sequência `{st.session_state.ultimo_sinal['sequencia']}` ➔ Sinal Gerado: **{st.session_state.ultimo_sinal['sinal']}**")
    else:
        st.warning("⚠️ O Agente RL precisa que um sinal seja gerado na aba 'Sinal Real' para receber uma punição ou recompensa.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        entrada_feedback = st.text_input("Números Reais Saídos (separados por vírgula):", placeholder="Ex: 14, 0, 5", key="input_feedback")
    with col_f2:
        resultado_feedback = st.selectbox("Qual foi o resultado final?", ["G0 (Acerto no Alvo)", "G1 (Acerto 1º Gale)", "G2 (Risco Alto)", "LOSS / FALHA"])

    if st.button("💾 Enviar Sinal de Recompensa/Punição ao Agente", key="btn_absorver_feedback", use_container_width=True):
        if not entrada_feedback:
            st.error("Insira as correções da mesa.")
        else:
            try:
                raw_str_fb = str(entrada_feedback).replace(';', ',').replace('-', ',').replace(' ', '')
                lista_nums_feedback = [int(x) for x in raw_str_fb.split(",") if x.isdigit()]
                
                with st.spinner("Registrando feedback contextual e preservando a cronologia da base..."):
                    if "ultimo_sinal" in st.session_state:
                        rel = motor.processar_feedback_real(
                            sequencia_12=st.session_state.ultimo_sinal["sequencia"],
                            sinal_indicado=st.session_state.ultimo_sinal["sinal"],
                            regra_id=st.session_state.ultimo_sinal["regra_id"],
                            numeros_saidos=lista_nums_feedback,
                            classificacao=resultado_feedback,
                            entropia_shannon=st.session_state.ultimo_sinal.get("entropia", 0.0),
                            probabilidade_markov=st.session_state.ultimo_sinal.get("probabilidade_markov", {"V":0,"P":0,"B":0})
                        )
                        del st.session_state["ultimo_sinal"]
                    else:
                        dados_novos = [{"numero": n, "cor": 'B' if n == 0 else ('V' if 1 <= n <= 7 else 'P')} for n in lista_nums_feedback]
                        rel = adicionar_a_base_longo_prazo(dados_novos)
                        motor.carregar_tudo()
                    
                    if rel and isinstance(rel, dict) and rel.get("sucesso"):
                        st.success("✅ Feedback absorvido com integridade cronológica. Somente os resultados reais novos foram anexados à base.")
                    else:
                        erro_msg = rel.get('mensagem') if isinstance(rel, dict) else 'Falha desconhecida'
                        st.error(f"Erro na propagação da rede: {erro_msg}")
            except Exception as e:
                st.error(f"🚨 Proteção de Crash Ativada: Formato inválido no feedback. Detalhe: {e}")

# =========================================================================
# ABA 3 — AUDITORIA E TREINAMENTO (TIPO D)
# =========================================================================
with aba_tipo_d:
    st.header("📊 Auditoria Dinâmica e Treinamento Profundo")
    st.caption("Gerenciamento de bases de dados e recálculo das matrizes de transição probabilísticas.")

    arquivo_upload = st.file_uploader("Arraste ou envie seu arquivo Excel (.xlsx) contendo os resultados históricos:", type=["xlsx"])

    if arquivo_upload is not None:
        caminho_temp = "temp_recencia.xlsx"
        try:
            with open(caminho_temp, "wb") as f:
                f.write(arquivo_upload.getbuffer())
        except Exception as e:
            st.error(f"Erro ao salvar arquivo temporário: {e}")

        st.info("Arquivo recebido com sucesso. Selecione a diretriz de processamento abaixo:")

        col1, col2, col3 = st.columns(3)
        with col1:
            btn_recencia = st.button("⚡ Injetar como Recência Ativa", use_container_width=True)
        with col2:
            btn_substituir = st.button("💾 Substituir Base Definitiva", use_container_width=True)
        with col3:
            btn_adicionar = st.button("➕ Encadeamento Dinâmico", use_container_width=True)

        if btn_recencia:
            try:
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados and len(dados) >= 20:
                    with open("base_recencia_ativa.xlsx", "wb") as f_rec:
                        f_rec.write(arquivo_upload.getvalue())
                        
                    with st.spinner("Injetando pesos na recência e absorvendo para o Longo Prazo..."):
                        resultado = motor.processar_recencia(dados)
                    st.success("✅ Recência acoplada com sucesso e absorvida pela Base Mestra!")

                    if resultado and isinstance(resultado, dict) and resultado.get("regime_recencia"):
                        with st.expander("📊 Relatório de Análise do Regime Injetado", expanded=True):
                            st.json(resultado["regime_recencia"])

                    with st.spinner("Simulando auditoria cronológica das janelas móveis..."):
                        motor_antigo = MotorV1Completo(dados)
                        output = motor_antigo.processar_auditoria()
                    
                    st.subheader("📝 Memória de Cálculo e Taxas de Assertividade")
                    st.text_area("Log Completo da Auditoria Executada", output, height=400)
                else:
                    st.error("Erro: A base de dados fornecida é muito pequena para estruturar um regime de recência consistente (Mínimo: 20 registros válidos).")
            except Exception as e:
                st.error(f"🚨 Proteção de Crash Ativada na Recência: {e}")

        if btn_substituir:
            try:
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados:
                    with st.spinner("Substituindo a base definitiva e retreinando os modelos contextuais..."):
                        rel = motor.absorver_base_longa(dados)
                    if rel and isinstance(rel, dict) and rel.get("sucesso"):
                        st.success("✅ Base definitiva substituída no XLS e modelos retreinados com sucesso!")
                        st.json(rel)
                    else:
                        st.error(f"Falha ao substituir base: {rel.get('mensagem') if isinstance(rel, dict) else 'Erro desconhecido'}")
                else:
                    st.error("Erro ao validar ou ler os registros do arquivo fornecido.")
            except Exception as e:
                st.error(f"🚨 Proteção de Crash Ativada na Substituição: {e}")

        if btn_adicionar:
            try:
                dados = LeitorXLS(caminho_temp).ler_e_validar()
                if dados:
                    with st.spinner("Processando lote incremental sem recarregar a base XLS no motor..."):
                        rel = motor.processar_novo_lote(dados)

                    del dados
                    gc.collect()

                    if rel and isinstance(rel, dict) and rel.get("sucesso"):
                        st.success("✅ Registros acoplados à base definitiva e persistidos no modelo pkl com sucesso!")
                        st.json(rel)
                    else:
                        erro_ms = rel.get("mensagem") if isinstance(rel, dict) else "Retorno nulo da camada de salvamento."
                        st.error(f"Falha no Encadeamento: {erro_ms}")
                else:
                    st.error("Erro: Nenhum dado numérico válido encontrado no arquivo enviado para encadeamento.")
            except Exception as e:
                gc.collect()
                st.error(f"🚨 Erro crítico no Encadeamento Dinâmico: {type(e).__name__}: {e}")
                st.exception(e)
        if os.path.exists(caminho_temp):
            try:
                os.remove(caminho_temp)
            except:
                pass

# =========================================================================
# ABA 4 — PADRÕES APRENDIDOS
# =========================================================================
with aba_padroes:
    st.header("📈 Arquitetura Neural e Padrões Aprendidos")
    st.caption("Varredura profunda das memórias da Q-Table e dos Destrinchadores Numéricos Ativos.")

    if st.button("🔄 Extrair Dumps de Memória da Rede Neural", use_container_width=True):
        try:
            ia = carregar_modelo_longo_prazo()
            if ia is None:
                st.warning("O pickle neural não pôde ser ativado. Falta histórico para treino.")
            else:
                st.success("Acesso ao kernel do modelo concluído.")
                
                with st.expander("🤖 Agente de Reinforcement Learning Autônomo (Q-Table)", expanded=True):
                    if hasattr(ia, 'q_table') and ia.q_table:
                        st.write(f"O Agente de Risco já mapeou autonomamente os Payouts de **{len(ia.q_table)} estados contextuais** diferentes na Roleta.")
                        metricas_q = getattr(ia, "q_learning_contextual_metricas", {})
                        if metricas_q:
                            st.caption("Origem do aprendizado Q-Learning: base longa cronológica, recência com peso oficial 6 e correção online pelo Feedback Tipo B.")
                            st.json(metricas_q)
                        st.json(ia.q_table)
                    else:
                        st.info("A Q-Table contextual ainda não possui estados treinados. Execute o treinamento da base longa ou processe a recência para gerar as janelas de aprendizado.")

                with st.expander("📐 Competência das Regras Oficiais e Contagens", expanded=False):
                    competencia_regras = getattr(ia, "regras_competencia_cronologica", {}) or {}
                    if competencia_regras:
                        grupos_regras = {
                            "🔢 Contagens Projetivas e Dinâmicas": {},
                            "🏛️ Hierarquias de Contagem": {},
                            "4️⃣ Regra Oficial do 4": {},
                            "🔟 Regra Oficial do 10": {},
                            "5️⃣ Regra Oficial 5-10": {},
                            "2️⃣ e 3️⃣ Ativadores e Regras V3": {},
                            "📚 Outras Regras Oficiais": {}
                        }
                        for nome_regra, stats_regra in competencia_regras.items():
                            nome_upper = str(nome_regra).upper()
                            if "COEXISTENCIA_CONTAGENS" in nome_upper or "TRANSICAO_CONTAGENS" in nome_upper or "CHANCE_DUPLA" in nome_upper or "FINALIZACAO_CONJUNTA" in nome_upper:
                                grupo = "🔢 Contagens Projetivas e Dinâmicas"
                            elif "HIERARQUIA_CONTAGEM" in nome_upper:
                                grupo = "🏛️ Hierarquias de Contagem"
                            elif "REGRA_4_" in nome_upper:
                                grupo = "4️⃣ Regra Oficial do 4"
                            elif "REGRA_10_" in nome_upper:
                                grupo = "🔟 Regra Oficial do 10"
                            elif "REGRA_5_10_" in nome_upper:
                                grupo = "5️⃣ Regra Oficial 5-10"
                            elif "V3_ATIVADOR_2" in nome_upper or "V3_ATIVADOR_3" in nome_upper:
                                grupo = "2️⃣ e 3️⃣ Ativadores e Regras V3"
                            elif "V3_ATIVADOR_" in nome_upper:
                                grupo = "🔢 Contagens Projetivas e Dinâmicas"
                            else:
                                grupo = "📚 Outras Regras Oficiais"
                            grupos_regras[grupo][nome_regra] = stats_regra

                        for nome_grupo, regras_grupo in grupos_regras.items():
                            if not regras_grupo: continue
                            st.markdown(f"### {nome_grupo}")
                            for nome_regra, stats_regra in sorted(regras_grupo.items()):
                                total_validacao = int(stats_regra.get("total_validacao", 0))
                                g0 = int(stats_regra.get("g0", 0))
                                g1 = int(stats_regra.get("g1", 0))
                                acertos_g0_g1 = int(stats_regra.get("acertos_g0_g1", g0 + g1))
                                taxa_g0_g1 = float(stats_regra.get("taxa_g0_g1", (acertos_g0_g1 / total_validacao * 100) if total_validacao > 0 else 0.0))
                                st.write(f"**{nome_regra}:** Total Validação: {total_validacao} | G0: {g0} | G1: {g1} | Acertos G0/G1: {acertos_g0_g1} ➔ **Taxa G0/G1: {taxa_g0_g1:.2f}%**")
                            st.markdown("---")
                    else:
                        st.info("A competência cronológica das regras oficiais e contagens ainda não foi treinada na base longa.")

                with st.expander("🎯 Assertividade Global das Projeções (Volume 3)", expanded=False):
                    if hasattr(ia, 'estatisticas_projecoes_globais') and ia.estatisticas_projecoes_globais:
                        for num_proj, stats in ia.estatisticas_projecoes_globais.items():
                            if stats["total"] > 0:
                                taxa = ((stats["g0"] + stats["g1"]) / stats["total"]) * 100
                                st.write(f"**Projeção do {num_proj}:** Total Analisado: {stats['total']} | G0: {stats['g0']} | G1: {stats['g1']} | Falhas: {stats['falha']} ➔ **Taxa de Assertividade Exata (Até G1): {taxa:.1f}%**")
                    else:
                        st.info("Nenhuma estatística global de projeção computada.")

                with st.expander("📊 Estatísticas Globais de Bigramas (Até G1)", expanded=False):
                    if hasattr(ia, 'estatisticas_bigramas_globais') and ia.estatisticas_bigramas_globais:
                        contagem_bi = 0
                        for bigrama, stats in ia.estatisticas_bigramas_globais.items():
                            if stats["total"] >= 10:
                                taxa_v = ((stats["V_g0"] + stats["V_g1"]) / stats["total"]) * 100
                                taxa_p = ((stats["P_g0"] + stats["P_g1"]) / stats["total"]) * 100
                                st.write(f"**Bigrama {bigrama}:** Total Ocorrências: {stats['total']} ➔ **Chamou Vermelho (Até G1): {taxa_v:.1f}%** | **Chamou Preto (Até G1): {taxa_p:.1f}%**")
                                contagem_bi += 1
                        if contagem_bi == 0: st.info("Sem quorum estatístico suficiente para bigramas.")
                    else:
                        st.info("Nenhuma estatística global de bigramas computada.")

                with st.expander("📊 Estatísticas Globais de Trigramas (Até G1)", expanded=False):
                    if hasattr(ia, 'estatisticas_trigramas_globais') and ia.estatisticas_trigramas_globais:
                        contagem_tri = 0
                        for trigrama, stats in ia.estatisticas_trigramas_globais.items():
                            if stats["total"] >= 5:
                                taxa_v = ((stats["V_g0"] + stats["V_g1"]) / stats["total"]) * 100
                                taxa_p = ((stats["P_g0"] + stats["P_g1"]) / stats["total"]) * 100
                                st.write(f"**Trigrama {trigrama}:** Total Ocorrências: {stats['total']} ➔ **Chamou Vermelho (Até G1): {taxa_v:.1f}%** | **Chamou Preto (Até G1): {taxa_p:.1f}%**")
                                contagem_tri += 1
                        if contagem_tri == 0: st.info("Sem quorum estatístico suficiente para trigramas.")
                    else:
                        st.info("Nenhuma estatística global de trigramas computada.")

                with st.expander("♟️ Padrões de Alternância Contínua (Xadrez)", expanded=False):
                    if hasattr(ia, 'padroes_xadrez_detalhado') and ia.padroes_xadrez_detalhado:
                        contagem_padroes = 0
                        for padrao, info in ia.padroes_xadrez_detalhado.items():
                            if info.get("total", 0) >= 5:
                                st.markdown(f"**Identificador:** `{padrao}` — Ocorrências Computadas: **{info.get('total')}x**")
                                st.json(info)
                                contagem_padroes += 1
                        if contagem_padroes == 0:
                            st.info("Nenhum padrão de xadrez atingiu a recorrência mínima necessária (>= 5 aparições).")
                    else:
                        st.info("Nenhum padrão estrutural de xadrez catalogado no modelo.")

                with st.expander("🔥 Padrões de Repetição Seguida (Streak)", expanded=False):
                    if hasattr(ia, 'padroes_streak_detalhado') and ia.padroes_streak_detalhado:
                        contagem_streaks = 0
                        for padrao, info in ia.padroes_streak_detalhado.items():
                            if info.get("total", 0) >= 5:
                                st.markdown(f"**Identificador:** `{padrao}` — Ocorrências Computadas: **{info.get('total')}x**")
                                st.json(info)
                                contagem_streaks += 1
                        if contagem_streaks == 0:
                            st.info("Nenhum padrão de streak atingiu a recorrência mínima necessária (>= 5 aparições).")
                    else:
                        st.info("Nenhum padrão estrutural de streak catalogado no modelo.")

                with st.expander("🌌 Arquitetura Combinatória de Destrinchador", expanded=False):
                    if hasattr(ia, 'padroes_gerais_detalhado') and ia.padroes_gerais_detalhado:
                        contagem_gerais = 0
                        for padrao, info in ia.padroes_gerais_detalhado.items():
                            if info.get("total", 0) >= 5:
                                st.markdown(f"**Estrutura Combinatória:** `{padrao}` — Ocorrências Computadas: **{info.get('total')}x**")
                                st.json(info)
                                contagem_gerais += 1
                        if contagem_gerais == 0:
                            st.info("Nenhum padrão geral ou espelho complexo atingiu a recorrência mínima necessária (>= 5 aparições).")
                    else:
                        st.info("Nenhum padrão estrutural complexo catalogado no modelo.")

                with st.expander("🔢 Distribuição Estatística de Comportamento Pós-Número (0 a 14)", expanded=False):
                    if hasattr(ia, 'analisar_comportamento_pos_numero'):
                        with st.spinner("Compilando relatórios probabilísticos por número individual..."):
                            relatorio_numeros = ia.analisar_comportamento_pos_numero()
                        
                        for numero, dados_num in relatorio_numeros.items():
                            st.markdown(f"### 📍 Número: {numero}")
                            col_n1, col_n2, col_n3 = st.columns(3)
                            with col_n1:
                                st.write(f"• **Aparições Totais:** {dados_num.get('total_aparicoes')}")
                                st.write(f"• **Cor Predominante Posterior:** `{dados_num.get('cor_mais_frequente_apos')}`")
                            with col_n2:
                                st.write(f"• **Frequência da Dominante:** {dados_num.get('frequencia_cor_dominante_%')}%")
                                st.write(f"• **Estabilidade Histórica:** `{dados_num.get('estabilidade')}`")
                            with col_n3:
                                st.write(f"• **Saturação de Volumetria:** `{dados_num.get('saturacao')}`")
                                st.write(f"• **Tendência de Fluxo:** `{dados_num.get('tendencia_recente')}`")
                            st.write("**Distribuição Real de Frequências (V/P/B):**")
                            st.json(dados_num.get("distribuicao_pos"))
                            st.markdown("---")
                    else:
                        st.info("O modelo carregado não possui suporte ou histórico para extração de comportamento pós-número.")
        except Exception as e:
            st.error(f"🚨 Proteção de Crash Ativada na extração de memória: {e}")

# =========================================================================
# ABA 5 — CÁLCULOS MATEMÁTICOS AVANÇADOS
# =========================================================================
with aba_matematica:
    st.header("🧮 Engine Estatístico Independente")
    st.caption("Acesso cru às varreduras de Viés e Gestão de Banca.")

    st.subheader("🌊 Algoritmo de Viés de Surfe (Macrofrequência)")
    if not os.path.exists(NOME_BASE_DEFINITIVA):
        st.warning(f"Matriz Estática Ausente: `{NOME_BASE_DEFINITIVA}`")
    else:
        try:
            janela_surfe = st.slider("Janela de Volumetria Retroativa para o Surfe:", min_value=20, max_value=500, value=100, step=10)
            vies_macro = EngineMatematicoAvancado.calcular_vies_surfe(NOME_BASE_DEFINITIVA, janela=janela_surfe)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(label="Frequência Real Vermelho (Alvo: 46.67%)", value=f"{vies_macro.get('frequencia_v')}%", delta=f"{vies_macro.get('desvio_v')}%")
                st.metric(label="Frequência Real Preto (Alvo: 46.67%)", value=f"{vies_macro.get('frequencia_p')}%", delta=f"{vies_macro.get('desvio_p')}%")
            with col_m2:
                st.metric(label="Frequência Real Branco (Alvo: 6.67%)", value=f"{vies_macro.get('frequencia_b')}%")
                st.info(f"**Parecer do Motor:**\n\n{vies_macro.get('vies')}")
        except Exception as e:
            st.error(f"🚨 Proteção de Crash Ativada no Motor Matemático: {e}")

    st.markdown("---")

    st.subheader("💰 Simulador de Cobertura e Divisão de Aportes (Split Stake)")
    st.caption("Cálculo exato das frações de capital de cobertura com base na margem matemática estática.")
    
    try:
        stake_base = st.number_input("Insira o valor da sua Stake Principal na cor escolhida (R$):", min_value=1.0, max_value=5000.0, value=10.0, step=5.0)
        simulacao_stake = EngineMatematicoAvancado.simular_split_stake_cobertura(stake_base)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric(label="Aporte na Cor Principal", value=f"R$ {simulacao_stake.get('stake_cor'):.2f}")
            st.metric(label="Custo Total da Operação", value=f"R$ {simulacao_stake.get('custo_total_operacao'):.2f}")
        with col_s2:
            st.metric(label="Cobertura de Branco Ideal (Proporção 1/7)", value=f"R$ {simulacao_stake.get('cobertura_b_ideal_1_7'):.2f}")
            st.metric(label="Cobertura Conservadora (Proporção 1/10)", value=f"R$ {simulacao_stake.get('cobertura_b_matematica_1_10'):.2f}")
        with col_s3:
            st.metric(label="Lucro Líquido Real (Se bater o Branco)", value=f"R$ {simulacao_stake.get('lucro_liquido_se_der_branco'):.2f}")
            st.metric(label="House Edge Mapeado", value=simulacao_stake.get("house_edge_estatico"))
    except Exception as e:
        st.error(f"🚨 Proteção de Crash Ativada no Simulador de Cobertura: {e}")

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("""
<div class="footer">
    MOTOR V1 · Deep Learning para Roleta · Versão 2.0 · 🧠 Todos os direitos reservados
</div>
""", unsafe_allow_html=True)
