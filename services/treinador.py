# services/treinador.py

import os
import time
import pandas as pd
from datetime import datetime

from config.settings import NOME_BASE_DEFINITIVA
from data.leitor_xls import LeitorXLS
from data.persistence import salvar_modelo_longo_prazo, carregar_modelo_longo_prazo
from ml_engine.preditor_base import IAPreditivaV1
from services.auditoria import MotorV1Completo
from rules.contagens import MotorContagensProjetivas
from rules.analisador import AnalisadorContextoAvancado

def analisar_regime_recencia(dados_recencia):
    if not dados_recencia or len(dados_recencia) < 20:
        return {
            "viés_atual": "INDEFINIDO",
            "modo_dominante": "NEUTRO",
            "xadrez_frequencia": 0.0,
            "streak_medio": 0,
            "confianca_regime": 0
        }
    
    janela_termometro = min(len(dados_recencia), 100)
    recorte_termometro = dados_recencia[-janela_termometro:]
    
    cores = [d['cor'] for d in recorte_termometro]
    total = len(cores)
    
    if total == 0:
        return {
            "viés_atual": "INDEFINIDO",
            "modo_dominante": "NEUTRO",
            "xadrez_frequencia": 0.0,
            "streak_medio": 0,
            "confianca_regime": 0
        }
        
    v = cores.count('V')
    p = cores.count('P')
    pct_v = (v / total) * 100
    pct_p = (p / total) * 100
    
    if pct_v >= 52.5:
        viés = "VERMELHO"
    elif pct_p >= 52.5:
        viés = "PRETO"
    else:
        viés = "EQUILIBRADO"
        
    alternancias = sum(1 for i in range(1, total) if cores[i] != cores[i-1] and cores[i] != 'B' and cores[i-1] != 'B')
    xadrez_freq = (alternancias / (total - 1)) * 100 if total > 1 else 0
    
    streaks = []
    atual = 1
    for i in range(1, total):
        if cores[i] == cores[i-1] and cores[i] != 'B':
            atual += 1
        else:
            if atual >= 2:
                streaks.append(atual)
            atual = 1
            
    streak_medio = sum(streaks) / len(streaks) if streaks else 0
    
    if xadrez_freq >= 55.0:
        modo = "XADREZ_DOMINANTE"
    elif streak_medio >= 3.0:
        modo = "STREAK_DOMINANTE"
    else:
        modo = "MISTO"
        
    confianca = min(85, int(abs(pct_v - 50) + abs(pct_p - 50)))
    
    return {
        "viés_atual": viés,
        "modo_dominante": modo,
        "xadrez_frequencia": round(xadrez_freq, 1),
        "streak_medio": round(streak_medio, 1),
        "confianca_regime": confianca,
        "pct_vermelho_recencia": round(pct_v, 1),
        "pct_preto_recencia": round(pct_p, 1)
    }

def integrar_recencia_no_modelo(dados_recencia, multiplicador=6):
    """Integra recência na instância ativa com peso oficial 6, sem persistir reinjeções."""
    multiplicador = 6
    ia = carregar_modelo_longo_prazo()
    if ia is None:
        ia = IAPreditivaV1([], [])
    ia.dados_recencia = []
    ia.injetar_aprendizado_imediato(
        dados_recencia, multiplicador_peso=multiplicador, salvar_na_recencia=True
    )
    ia.treinar_q_learning_contextual(
        dados_recencia,
        multiplicador_peso=6,
        origem="RECENCIA"
    )
    ia.analise_recencia = ia.analisar_comportamento_pos_numero_recencia(ia.dados_recencia)
    ia.regime_recencia = analisar_regime_recencia(ia.dados_recencia)
    ia.recencia_foi_injetada_na_sessao = True
    return ia

def adicionar_a_base_longo_prazo(novos_dados, origem_feedback_ao_vivo=False):
    if not novos_dados:
        return {"sucesso": False, "mensagem": "Nenhum dado novo foi fornecido."}
    base_existente = []
    if os.path.exists(NOME_BASE_DEFINITIVA):
        try:
            base_existente = LeitorXLS(NOME_BASE_DEFINITIVA).ler_e_validar() or []
        except:
            return {"sucesso": False, "mensagem": "Não foi possível ler a base antiga."}

    # AUDITORIA WALK-FORWARD: mede os dados novos com o modelo histórico congelado.
    # Somente depois da medição os novos dados são absorvidos pelo treinamento.
    auditoria_walk_forward = None
    modelo_historico = carregar_modelo_longo_prazo()

    if modelo_historico is None and len(base_existente) >= 30:
        modelo_historico = IAPreditivaV1(base_existente, [])

    # MAIN 133 — feedback ao vivo não dispara auditoria walk-forward.
    # A reconstrução cronológica do MAIN 132 pode produzir 13+ registros novos
    # no primeiro segmento, mas isso continua sendo feedback operacional curto,
    # não uma nova carga XLS destinada à auditoria histórica.
    if (
        not origem_feedback_ao_vivo
        and modelo_historico is not None
        and len(novos_dados) >= 13
    ):
        contexto_historico = base_existente[-12:] if base_existente else []
        dados_auditoria = contexto_historico + novos_dados
        motor_auditoria = MotorV1Completo(dados_auditoria, ia_existente=modelo_historico)
        motor_auditoria.processar_auditoria(aprender_durante_auditoria=False)
        stats_walk = dict(motor_auditoria.stats)
        total_walk = sum(stats_walk.values())
        taxa_walk = (((stats_walk.get("G0", 0) + stats_walk.get("G1", 0)) / total_walk) * 100) if total_walk > 0 else 0
        auditoria_walk_forward = {
            "registros_novos_avaliados": len(novos_dados),
            "janelas_analisadas": total_walk,
            "G0": stats_walk.get("G0", 0),
            "G1": stats_walk.get("G1", 0),
            "G2": stats_walk.get("G2", 0),
            "FALHA": stats_walk.get("FALHA", 0),
            "NO CALL": stats_walk.get("NO CALL", 0),
            "assertividade_g0_g1_percent": round(taxa_walk, 2),
            "modelo_congelado_durante_auditoria": True
        }

    dados_combinados = base_existente + novos_dados
    if os.path.exists(NOME_BASE_DEFINITIVA):
        try:
            backup_name = NOME_BASE_DEFINITIVA.replace(".xlsx", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            os.rename(NOME_BASE_DEFINITIVA, backup_name)
        except:
            pass
    try:
        df = pd.DataFrame([{"numero": d["numero"], "cor": d["cor"]} for d in dados_combinados])
        # Persistência oficial NOVO -> ANTIGO; LeitorXLS reconstrói ANTIGO -> NOVO.
        df = df.iloc[::-1].reset_index(drop=True)
        df.to_excel(NOME_BASE_DEFINITIVA, index=False)
    except Exception as e:
        return {"sucesso": False, "mensagem": f"Erro ao salvar o arquivo: {e}"}

    stats_incrementais = None
    if isinstance(auditoria_walk_forward, dict):
        stats_incrementais = {
            "G0": auditoria_walk_forward.get("G0", 0),
            "G1": auditoria_walk_forward.get("G1", 0),
            "G2": auditoria_walk_forward.get("G2", 0),
            "FALHA": auditoria_walk_forward.get("FALHA", 0),
            "NO CALL": auditoria_walk_forward.get("NO CALL", 0)
        }

    relatorio_treinamento = treinar_base_longo_prazo_incremental(
        modelo_historico,
        base_existente,
        novos_dados,
        dados_combinados,
        stats_incrementais=stats_incrementais
    )
    if relatorio_treinamento is None:
        relatorio_treinamento = treinar_base_longo_prazo_com_janelas(dados_combinados)
    if isinstance(relatorio_treinamento, dict):
        relatorio_treinamento["registros_base_acumulada"] = len(dados_combinados)
        relatorio_treinamento["registros_novos_absorvidos"] = len(novos_dados)
        relatorio_treinamento["auditoria_walk_forward"] = auditoria_walk_forward
    return relatorio_treinamento

def _atualizar_metricas_cartografia_incremental(ia):
    proj_suporte20 = sum(
        1 for st in ia.cartografia_projecoes_trajetoria.values()
        if st.get("total", 0) >= 20
    )
    padroes_suporte20 = sum(
        1 for st in ia.cartografia_padroes_xls.values()
        if st.get("total", 0) >= 20
    )
    contextual_suporte20 = sum(
        1 for st in ia.cartografia_padroes_contextual.values()
        if st.get("total", 0) >= 20
    )
    streak_traj_suporte20 = sum(
        1 for st in ia.cartografia_trajetoria_streak.values()
        if st.get("total", 0) >= 20
    )
    morfologia_suporte20 = sum(
        1 for st in ia.cartografia_morfologia_estrutural.values()
        if st.get("total", 0) >= 20
    )
    ia.cartografia_morfologia_estrutural_metricas = {
        "ativo": True,
        "versao": 1,
        "metodo": "MORFOLOGIA_BLOCOS_NORMALIZADA_REPETICAO_INVERSAO_ESPELHO_ATE_G1_INCREMENTAL",
        "contextos_aprendidos": len(ia.cartografia_morfologia_estrutural),
        "contextos_suporte_minimo_20": morfologia_suporte20,
        "dimensoes": ["BLOCOS", "MORFOLOGIA", "TRAJETORIA", "REPETICAO", "INVERSAO_CROMATICA", "ESPELHO"],
        "nomes_didaticos_convertidos_em_regras": False,
        "altera_direcao": False,
        "processamento_incremental": True,
    }
    ia.cartografia_trajetoria_streak_metricas = {
        "ativo": True,
        "versao": 2,
        "metodo": "TRAJETORIA_CAUSAL_STREAK_MORFOLOGICA_BILATERAL_V_P_ATE_G1_INCREMENTAL",
        "contextos_aprendidos": len(ia.cartografia_trajetoria_streak),
        "contextos_suporte_minimo_20": streak_traj_suporte20,
        "cores_estudadas": ["VERMELHO", "PRETO"],
        "estagios": ["NASCIMENTO", "CONFIRMACAO", "STREAK", "EXPANSAO", "RETOMADA"],
        "cruza_respiro_e_contagens": True,
        "processamento_incremental": True,
    }
    ia.cartografia_padroes_contextual_metricas = {
        "ativo": True,
        "versao": 1,
        "metodo": "PADRAO_RAIZ_COM_CONDICIONANTES_INTERNAS_G0_G1",
        "contextos_aprendidos": len(ia.cartografia_padroes_contextual),
        "contextos_suporte_minimo_20": contextual_suporte20,
        "dimensoes": [
            "PADRAO_RAIZ", "ULTIMO_NUMERO", "BIGRAMA", "TRIGRAMA",
            "REGIME", "MARKOV", "GEOMETRIA", "TRANSICAO_GEOMETRIA",
            "REGRAS_ATIVAS", "CONTAGENS_ATIVAS"
        ],
        "prioridade": "G0_COM_CONFIRMACAO_G0_G1",
        "altera_regras": False,
        "altera_recencia": False
    }
    metricas_antigas = getattr(ia, "cartografia_xls_metricas", {}) or {}
    ia.cartografia_xls_metricas = {
        "ativo": True,
        "versao": 2,
        "metodo": "CARTOGRAFIA_CRONOLOGICA_COMPLETA_CASO_A_CASO_ATE_G1_INCREMENTAL",
        "eventos_padrao_analisados": int(metricas_antigas.get("eventos_padrao_analisados", 0)),
        "eventos_projecoes_1_a_7_analisados": int(metricas_antigas.get("eventos_projecoes_1_a_7_analisados", 0)),
        "contextos_padroes_aprendidos": len(ia.cartografia_padroes_xls),
        "contextos_projecoes_aprendidos": len(ia.cartografia_projecoes_trajetoria),
        "contextos_padroes_suporte_minimo_20": padroes_suporte20,
        "contextos_projecoes_suporte_minimo_20": proj_suporte20,
        "processamento_incremental": True
    }


def _absorver_cartografia_completa_incremental(ia, dados_combinados, inicio_novos):
    total = len(dados_combinados)
    eventos_padrao_novos = 0
    eventos_projecao_novos = 0

    # Eventos de padrão que ainda não tinham G0/G1 disponíveis na base antiga,
    # mais todos os eventos cujo fechamento está nos dados novos.
    for i in range(max(0, inicio_novos - 2), total - 2):
        inicio = max(0, i - 11)
        sub = dados_combinados[inicio:i + 1]
        sub_num = [int(d["numero"]) for d in sub]
        sub_pol = [str(d["cor"]).upper() for d in sub]
        c0 = str(dados_combinados[i + 1]["cor"]).upper()
        c1 = str(dados_combinados[i + 2]["cor"]).upper()

        for chave in ia._chaves_cartografia_padrao(sub_num, sub_pol):
            st = ia.cartografia_padroes_xls[chave]
            st["total"] += 1
            if c0 == "B":
                st["B_g0"] += 1
            rv = ia._resultado_ate_g1(c0, c1, "V")
            rp = ia._resultado_ate_g1(c0, c1, "P")
            if rv == "G0":
                st["V_g0"] += 1
            elif rv == "G1":
                st["V_g1"] += 1
            if rp == "G0":
                st["P_g0"] += 1
            elif rp == "G1":
                st["P_g1"] += 1

        ia._registrar_cartografia_contextual_padrao(sub_num, sub_pol, c0, c1)
        ia._registrar_trajetoria_streak(sub_num, sub_pol, c0, c1)
        ia._registrar_morfologia_estrutural(sub_num, sub_pol, c0, c1)
        eventos_padrao_novos += 1

    # Projeções iniciadas no fim da base antiga podem terminar nos dados novos.
    for i in range(max(0, inicio_novos - 8), total):
        num_gatilho = int(dados_combinados[i]["numero"])
        if not 1 <= num_gatilho <= 7:
            continue
        alvo_idx = i + num_gatilho
        if alvo_idx + 1 >= total or alvo_idx + 1 < inicio_novos:
            continue
        caminho_tem_branco = any(
            str(dados_combinados[k]["cor"]).upper() == "B"
            for k in range(i + 1, alvo_idx)
        )
        if caminho_tem_branco:
            continue
        traj = dados_combinados[i:alvo_idx + 1]
        traj_num = [int(d["numero"]) for d in traj]
        traj_pol = [str(d["cor"]).upper() for d in traj]
        cor_alvo = str(dados_combinados[alvo_idx]["cor"]).upper()
        cor_g1 = str(dados_combinados[alvo_idx + 1]["cor"]).upper()
        resultado = ia._resultado_ate_g1(cor_alvo, cor_g1, "V")
        for chave in ia._chaves_trajetoria_projecao(num_gatilho, traj_num, traj_pol):
            st = ia.cartografia_projecoes_trajetoria[chave]
            st["total"] += 1
            if resultado == "G0":
                st["respeitada_g0"] += 1
            elif resultado == "G1":
                st["respeitada_g1"] += 1
            else:
                st["nao_respeitada"] += 1
        eventos_projecao_novos += 1

    metricas = getattr(ia, "cartografia_xls_metricas", {}) or {}
    metricas["eventos_padrao_analisados"] = int(metricas.get("eventos_padrao_analisados", 0)) + eventos_padrao_novos
    metricas["eventos_projecoes_1_a_7_analisados"] = int(metricas.get("eventos_projecoes_1_a_7_analisados", 0)) + eventos_projecao_novos
    ia.cartografia_xls_metricas = metricas
    _atualizar_metricas_cartografia_incremental(ia)


def _absorver_regras_contextuais_incremental(ia, dados_combinados, inicio_novos):
    numeros = [int(d["numero"]) for d in dados_combinados]
    cores = [str(d["cor"]).upper() for d in dados_combinados]
    familias_diretas = {"REGRA_OFICIAL_4", "REGRA_OFICIAL_10", "REGRA_OFICIAL_5_10"}
    posicoes = 0
    ocorrencias = 0
    diretas = 0
    detector_12 = 0

    # A base antiga já processou até len(base_antiga)-4. Começamos exatamente
    # nas três posições que antes não possuíam G0/G1/G2 completos.
    for i in range(max(0, inicio_novos - 3), len(dados_combinados) - 3):
        inicio = max(0, i - 11)
        sub_num = numeros[inicio:i + 1]
        sub_pol = cores[inicio:i + 1]
        posicoes += 1

        eventos_diretos = ia._eventos_regras_oficiais_cronologicos_no_indice(
            numeros, cores, i
        )
        eventos_detector = []
        if len(sub_num) >= 12 and len(sub_pol) >= 12:
            eventos_detector = [
                evento
                for evento in ia._eventos_regras_contagens_contextuais(
                    sub_num[-12:], sub_pol[-12:]
                )
                if evento.get("familia") not in familias_diretas
            ]
        eventos = eventos_diretos + eventos_detector
        if not eventos:
            continue

        ocorrencias += ia._registrar_cartografia_contextual_regra(
            sub_num, sub_pol,
            cores[i + 1], cores[i + 2], cores[i + 3],
            eventos_override=eventos
        )
        diretas += len(eventos_diretos)
        detector_12 += len(eventos_detector)

    bases = [
        st for chave, st in ia.cartografia_regras_contextual.items()
        if chave.count("|") == 1
    ]
    total_base = sum(int(st.get("total", 0)) for st in bases)
    g0 = sum(int(st.get("g0", 0)) for st in bases)
    g1 = sum(int(st.get("g1", 0)) for st in bases)
    g2 = sum(int(st.get("g2", 0)) for st in bases)
    falha = sum(int(st.get("falha", 0)) for st in bases)
    antigas = getattr(ia, "cartografia_regras_contextual_metricas", {}) or {}
    ia.cartografia_regras_contextual_metricas = {
        "ativo": True,
        "versao": 3,
        "metodo": "VARREDURA_CRONOLOGICA_DIRETA_INDICE_A_INDICE_INCREMENTAL",
        "detector_oficial_alterado": False,
        "janela_12_operacional_alterada": False,
        "regras_diretas": ["REGRA_4", "REGRA_10", "REGRA_5_10"],
        "posicoes_varridas": int(antigas.get("posicoes_varridas", 0)) + posicoes,
        "ocorrencias_catalogadas": int(antigas.get("ocorrencias_catalogadas", 0)) + ocorrencias,
        "ocorrencias_regras_diretas": int(antigas.get("ocorrencias_regras_diretas", 0)) + diretas,
        "ocorrencias_detector_12": int(antigas.get("ocorrencias_detector_12", 0)) + detector_12,
        "contextos": len(ia.cartografia_regras_contextual),
        "total_ocorrencias_base": total_base,
        "G0": g0, "G1": g1, "G2": g2, "FALHA": falha,
        "resolucao_ate_g1_percent": round(((g0 + g1) / max(1, total_base)) * 100.0, 2),
        "processamento_incremental": True
    }


def _absorver_markov_incremental(ia, dados_combinados, inicio_novos):
    cores = [str(d.get("cor", "B")).upper() for d in dados_combinados]
    for ordem in range(1, 7):
        inicio = max(ordem, inicio_novos)
        for i in range(inicio, len(cores)):
            estado = tuple(cores[i-ordem:i])
            proxima = cores[i]
            if proxima not in ("V", "P", "B"):
                continue
            stats = ia.markov_ordens[ordem][estado]
            stats[proxima] += 1
            stats["total"] += 1



def _absorver_estatisticas_globais_incremental(ia, dados_combinados, inicio_novos):
    total_dados = len(dados_combinados)

    # Projeções 1..7 cuja resolução passou a existir com a chegada do novo bloco.
    for i in range(max(0, inicio_novos - 8), total_dados):
        num = int(dados_combinados[i]["numero"])
        if not 1 <= num <= 7:
            continue
        alvo_idx = i + num
        if alvo_idx + 1 >= total_dados or alvo_idx + 1 < inicio_novos:
            continue
        caminho_tem_branco = any(
            str(dados_combinados[k]["cor"]).upper() == "B"
            for k in range(i + 1, alvo_idx)
        )
        if caminho_tem_branco:
            continue
        cor_alvo = str(dados_combinados[alvo_idx]["cor"]).upper()
        cor_g1 = str(dados_combinados[alvo_idx + 1]["cor"]).upper()
        if cor_alvo in ("V", "B"):
            resultado = "G0"
        elif cor_g1 in ("V", "B"):
            resultado = "G1"
        else:
            resultado = "NAO_RESPEITADA"

        stats_global = ia.estatisticas_projecoes_globais[num]
        stats_respeito = ia.estatisticas_projecoes_respeito[num]
        stats_global["total"] += 1
        stats_respeito["total"] += 1
        if resultado == "G0":
            stats_global["g0"] += 1
            stats_respeito["respeitada_g0"] += 1
        elif resultado == "G1":
            stats_global["g1"] += 1
            stats_respeito["respeitada_g1"] += 1
        else:
            stats_global["falha"] += 1
            stats_respeito["nao_respeitada"] += 1

        inicio_ctx = max(0, alvo_idx - 11)
        pol_ctx = [
            str(d["cor"]).upper()
            for d in dados_combinados[inicio_ctx:alvo_idx + 1]
        ]
        if len(pol_ctx) >= 3:
            chave_ctx = ia._chave_respeito_projecao(num, pol_ctx)
            stats_ctx = ia.projecoes_respeito_contextual[chave_ctx]
            stats_ctx["total"] += 1
            if resultado == "G0":
                stats_ctx["respeitada_g0"] += 1
            elif resultado == "G1":
                stats_ctx["respeitada_g1"] += 1
            else:
                stats_ctx["nao_respeitada"] += 1

    # Bigramas e trigramas: começa exatamente nos fechamentos que não tinham
    # G0/G1 completos na base antiga.
    for i in range(max(0, inicio_novos - 2), total_dados - 2):
        c0 = str(dados_combinados[i + 1]["cor"]).upper()
        c1 = str(dados_combinados[i + 2]["cor"]).upper()
        if i >= 1:
            bigrama = f"{dados_combinados[i-1]['numero']}-{dados_combinados[i]['numero']}"
            st = ia.estatisticas_bigramas_globais[bigrama]
            st["total"] += 1
            if c0 in ("V", "B"):
                st["V_g0"] += 1
            elif c1 in ("V", "B"):
                st["V_g1"] += 1
            if c0 in ("P", "B"):
                st["P_g0"] += 1
            elif c1 in ("P", "B"):
                st["P_g1"] += 1
        if i >= 2:
            trigrama = f"{dados_combinados[i-2]['numero']}-{dados_combinados[i-1]['numero']}-{dados_combinados[i]['numero']}"
            st = ia.estatisticas_trigramas_globais[trigrama]
            st["total"] += 1
            if c0 in ("V", "B"):
                st["V_g0"] += 1
            elif c1 in ("V", "B"):
                st["V_g1"] += 1
            if c0 in ("P", "B"):
                st["P_g0"] += 1
            elif c1 in ("P", "B"):
                st["P_g1"] += 1

    # Regras oficiais auditadas até G1.
    for fim in range(max(11, inicio_novos - 2), total_dados - 2):
        janela_num = [int(d["numero"]) for d in dados_combinados[fim - 11:fim + 1]]
        janela_pol = [str(d["cor"]).upper() for d in dados_combinados[fim - 11:fim + 1]]
        regras = MotorContagensProjetivas.mapear_janela(
            janela_num, janela_pol,
            AnalisadorContextoAvancado.mapear_padroes_geometria(janela_pol),
            None
        )
        c0 = str(dados_combinados[fim + 1]["cor"]).upper()
        c1 = str(dados_combinados[fim + 2]["cor"]).upper()
        vistos = set()
        for regra in regras:
            direcao = regra.get("direcao")
            tipo = regra.get("tipo_regra")
            if direcao not in ("VERMELHO", "PRETO") or not tipo:
                continue
            chave_vista = (tipo, direcao)
            if chave_vista in vistos:
                continue
            vistos.add(chave_vista)
            st = ia.estatisticas_regras_oficiais[tipo]
            st["total"] += 1
            if direcao == "VERMELHO":
                if c0 in ("V", "B"):
                    st["V_g0"] += 1
                elif c1 in ("V", "B"):
                    st["V_g1"] += 1
            else:
                if c0 in ("P", "B"):
                    st["P_g0"] += 1
                elif c1 in ("P", "B"):
                    st["P_g1"] += 1

    # Especialista espelho/inversão na mesma fronteira cronológica.
    for i in range(max(11, inicio_novos - 2), total_dados - 2):
        janela = dados_combinados[i-11:i+1]
        sub_num = [d["numero"] for d in janela]
        sub_pol = [d["cor"] for d in janela]
        chave = ia._identificar_contexto_espelho_inversao(sub_num, sub_pol)
        if not chave:
            continue
        c0 = str(dados_combinados[i+1]["cor"]).upper()
        c1 = str(dados_combinados[i+2]["cor"]).upper()
        st = ia.especialista_espelho_inversao[chave]
        st["total"] += 1
        if c0 in ("V", "B"):
            st["V_g0"] += 1
        elif c1 in ("V", "B"):
            st["V_g1"] += 1
        if c0 in ("P", "B"):
            st["P_g0"] += 1
        elif c1 in ("P", "B"):
            st["P_g1"] += 1

    ia.regras_oficiais_metricas = {
        "ativo": True,
        "metodo": "REGRAS_OFICIAIS_AUDITADAS_ATE_G1_INCREMENTAL",
        "regras_mapeadas": len(ia.estatisticas_regras_oficiais),
        "ocorrencias_mapeadas": sum(v["total"] for v in ia.estatisticas_regras_oficiais.values()),
        "participa_geracao_sinal": True,
        "processamento_incremental": True
    }
    total_projecoes = sum(
        stats["total"] for stats in ia.estatisticas_projecoes_respeito.values()
    )
    total_respeitadas = sum(
        stats["respeitada_g0"] + stats["respeitada_g1"]
        for stats in ia.estatisticas_projecoes_respeito.values()
    )
    ia.projecoes_respeito_metricas = {
        "ativo": True,
        "leitura_bilateral_v_p": False,
        "metodo": "RESPEITADA_VS_NAO_RESPEITADA_ATE_G1_INCREMENTAL",
        "total_contagens_mapeadas": total_projecoes,
        "contagens_respeitadas_ate_g1": total_respeitadas,
        "contagens_nao_respeitadas": total_projecoes - total_respeitadas,
        "taxa_respeito_g0_g1_percent": round(
            (total_respeitadas / max(total_projecoes, 1)) * 100, 2
        ),
        "contextos_respeito_aprendidos": len(ia.projecoes_respeito_contextual),
        "regra_v3_direcao_original": "VERMELHO",
        "processamento_incremental": True
    }


def _montar_relatorio_incremental(ia, dados_combinados, stats_incrementais, sucesso_salvar):
    stats = stats_incrementais or {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0, "NO CALL": 0}
    total_janelas = sum(stats.values())
    regras_boas = int(
        getattr(ia, "regras_competencia_metricas", {}).get(
            "regras_com_boa_performance", 0
        )
    )
    taxa_g0_g1 = ((stats.get("G0", 0) + stats.get("G1", 0)) / total_janelas * 100) if total_janelas > 0 else 0
    return {
        "sucesso": True,
        "registros_processados": len(dados_combinados),
        "janelas_analisadas": total_janelas,
        "G0": stats.get("G0", 0),
        "G1": stats.get("G1", 0),
        "G2": stats.get("G2", 0),
        "FALHA": stats.get("FALHA", 0),
        "NO CALL": stats.get("NO CALL", 0),
        "regras_com_boa_performance": regras_boas,
        "assertividade_g0_g1_percent": round(taxa_g0_g1, 2),
        "assertividade_sinais_liberados_g0_g1_percent": round(
            ((stats.get("G0", 0) + stats.get("G1", 0)) /
             max(1, stats.get("G0", 0) + stats.get("G1", 0) + stats.get("G2", 0) + stats.get("FALHA", 0))) * 100, 2
        ),
        "modelo_salvo_com_sucesso": sucesso_salvar,
        "analise_comportamento_numeros": ia.analisar_comportamento_pos_numero(),
        "evolucao_modelo": {
            "markov_multiescala_ordens": [1, 2, 3, 4, 5, 6],
            "contextos_conflito_aprendidos": len(getattr(ia, "memoria_conflitos", {})),
            "memoria_conflitos_metricas": getattr(ia, "memoria_conflitos_metricas", {}),
            "competencia_regras_posicionais": getattr(ia, "regras_competencia_cronologica", {}),
            "competencia_regras_posicionais_metricas": getattr(ia, "regras_competencia_metricas", {}),
            "ml_validacao_cronologica": getattr(ia, "ml_metricas", {}),
            "ml_pesos_adaptativos": getattr(ia, "ml_pesos", {}),
            "q_learning_contextual": getattr(ia, "q_learning_contextual_metricas", {}),
            "aprendizado_temporal_adaptativo": getattr(ia, "temporal_metricas", {}),
            "ml_atualizacao_controlada": getattr(ia, "ml_atualizacao_incremental_metricas", {}),
            "matriz_evolutiva": getattr(ia, "matriz_evolutiva", {}),
            "matriz_deriva_comportamental": ia.mapear_deriva_comportamental_numeros(),
            "consequencia_futura_ultima": getattr(ia, "ultima_consequencia_futura", {}),
            "hierarquia_oficial_motor_v1": getattr(ia, "hierarquia_oficial_metricas", {}),
            "especialista_espelho_inversao_contextos": len(getattr(ia, "especialista_espelho_inversao", {})),
            "competencia_especialistas_cronologica": getattr(ia, "competencia_especialistas", {}),
            "competencia_especialistas_metricas": getattr(ia, "competencia_metricas", {}),
            "competencia_especialistas_contextual": getattr(ia, "competencia_contextual_metricas", {}),
            "competencia_camadas_ampliadas": getattr(ia, "camadas_ampliadas_competencia", {}),
            "camadas_ampliadas_metricas": getattr(ia, "camadas_ampliadas_metricas", {}),
            "contagens_projetivas_respeito": getattr(ia, "projecoes_respeito_metricas", {}),
            "cartografia_completa_xls": getattr(ia, "cartografia_xls_metricas", {}),
            "cartografia_trajetoria_streak": getattr(ia, "cartografia_trajetoria_streak_metricas", {}),
            "cartografia_morfologia_estrutural": getattr(ia, "cartografia_morfologia_estrutural_metricas", {}),
            "cartografia_contextual_interna_padroes": getattr(ia, "cartografia_padroes_contextual_metricas", {}),
            "cartografia_contextual_interna_regras_contagens": getattr(ia, "cartografia_regras_contextual_metricas", {}),
            "filtro_discriminativo_g0_g1_vs_g2_mais": getattr(ia, "filtro_discriminativo_metricas", {}),
            "configuracao_filtro_discriminativo": getattr(ia, "filtro_discriminativo_config", {}),
            "objetivo_operacional_g0_g1": {
                "ativo": True, "G0": "SUCESSO_PRIORITARIO", "G1": "SUCESSO_ACEITAVEL",
                "G2": "RISCO_OPERACIONAL", "FALHA": "RISCO_OPERACIONAL"
            },
            "especialista_risco_g2_mais": getattr(ia, "risco_g2_mais_metricas", {}),
            "configuracao_risco_g2_mais": getattr(ia, "risco_g2_mais_config", {}),
            "projecoes_bilaterais_ativas": False,
            "projecoes_respeito_ativas": True,
            "regra_v3_original_preservada": True,
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH,
            "treinamento_base_longa_incremental": True
        },
        "ia_treinada": ia,
        "mensagem": "Absorção incremental da base longa concluída com sucesso."
    }


def treinar_base_longo_prazo_incremental(modelo_existente, base_existente, novos_dados, dados_combinados, stats_incrementais=None):
    if modelo_existente is None or len(getattr(modelo_existente, "dados_longo", []) or []) != len(base_existente):
        return None
    if not novos_dados:
        return None

    ia = modelo_existente
    inicio_novos = len(base_existente)

    # Pré-condição do fluxo incremental: modelos persistidos de versões
    # anteriores podem ter unidade_analise incompleta ou chaves numéricas como
    # texto. Normaliza somente essa estrutura antes de reutilizá-la.
    ia._normalizar_unidade_analise_compatibilidade()

    # Preserva toda a memória histórica já treinada. Somente os registros novos
    # são absorvidos pelas estruturas aditivas; as fronteiras cronológicas são
    # tratadas separadamente nas cartografias abaixo.
    ia.dados_longo = dados_combinados
    ia._processar_bloco_dados(novos_dados, 1, True)
    ia._calcular_probabilidades_globais_cache()
    _absorver_markov_incremental(ia, dados_combinados, inicio_novos)
    _absorver_estatisticas_globais_incremental(ia, dados_combinados, inicio_novos)
    _absorver_cartografia_completa_incremental(ia, dados_combinados, inicio_novos)
    _absorver_regras_contextuais_incremental(ia, dados_combinados, inicio_novos)

    ia.treinar_q_learning_contextual(
        novos_dados,
        multiplicador_peso=1,
        origem="BASE_LONGA"
    )
    reforcar_aprendizado_tipo_d(ia)
    ia.mapear_padroes_avancados(novos_dados)

    # MAIN 114 — atualização controlada das memórias que antes ficavam
    # congeladas no treinamento inicial. A memória temporal é reconstruída
    # em fluxo único sobre a base acumulada e a ML usa somente uma cauda
    # cronológica limitada, evitando retreinar brutalmente toda a base.
    ia._treinar_memoria_temporal_adaptativa()
    ia._atualizar_ml_controlada_incremental(dados_combinados)
    ia.atualizar_matriz_evolutiva()

    seen = set()
    unique_patterns = []
    for p in ia.memoria_padroes_vencedores:
        try:
            key = json.dumps(p, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)
        except:
            unique_patterns.append(p)
    ia.memoria_padroes_vencedores = unique_patterns

    sucesso_salvar = False
    for _ in range(3):
        sucesso_salvar = salvar_modelo_longo_prazo(ia)
        if sucesso_salvar and carregar_modelo_longo_prazo() is not None:
            break
        sucesso_salvar = False
        time.sleep(0.6)

    return _montar_relatorio_incremental(
        ia, dados_combinados, stats_incrementais, sucesso_salvar
    )


def reforcar_aprendizado_tipo_d(ia):
    for padrao, qtd in ia.controladores_fortes.items():
        if qtd >= 8:
            ia.padroes_fortes.append({"tipo": "CONTROLADOR_MUITO_FORTE", "padrao": padrao, "peso": qtd * 2})
    ia.padroes_fortes = sorted(ia.padroes_fortes, key=lambda x: x.get("peso", 0), reverse=True)[:30]

def treinar_base_longo_prazo_com_janelas(dados_completos):
    if not dados_completos or len(dados_completos) < 30:
        return {"sucesso": False, "mensagem": "Base muito pequena para treinamento profundo."}
    motor = MotorV1Completo(dados_completos)
    motor.processar_auditoria()
    motor.ia.treinar_q_learning_contextual(
        dados_completos,
        multiplicador_peso=1,
        origem="BASE_LONGA"
    )
    reforcar_aprendizado_tipo_d(motor.ia)
    motor.ia.mapear_padroes_avancados(dados_completos)
    motor.ia.atualizar_matriz_evolutiva()
    seen = set()
    unique_patterns = []
    for p in motor.ia.memoria_padroes_vencedores:
        try:
            key = json.dumps(p, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)
        except:
            unique_patterns.append(p)
    motor.ia.memoria_padroes_vencedores = unique_patterns
    sucesso_salvar = False
    for _ in range(3):
        sucesso_salvar = salvar_modelo_longo_prazo(motor.ia)
        if sucesso_salvar:
            ia_verif = carregar_modelo_longo_prazo()
            if ia_verif is not None:
                break
            else:
                sucesso_salvar = False
        time.sleep(0.6)
    stats = getattr(motor, 'stats', {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0, "NO CALL": 0})
    total_janelas = sum(stats.values()) if stats else 0
    regras_boas = int(
        getattr(motor.ia, "regras_competencia_metricas", {}).get(
            "regras_com_boa_performance", 0
        )
    )
    taxa_g0_g1 = ((stats.get("G0", 0) + stats.get("G1", 0)) / total_janelas * 100) if total_janelas > 0 else 0
    analise_numeros = motor.ia.analisar_comportamento_pos_numero()
    return {
        "sucesso": True,
        "registros_processados": len(dados_completos),
        "janelas_analisadas": total_janelas,
        "G0": stats.get("G0", 0),
        "G1": stats.get("G1", 0),
        "G2": stats.get("G2", 0),
        "FALHA": stats.get("FALHA", 0),
        "NO CALL": stats.get("NO CALL", 0),
        "regras_com_boa_performance": regras_boas,
        "assertividade_g0_g1_percent": round(taxa_g0_g1, 2),
        "assertividade_sinais_liberados_g0_g1_percent": round(
            ((stats.get("G0", 0) + stats.get("G1", 0)) /
             max(1, stats.get("G0", 0) + stats.get("G1", 0) + stats.get("G2", 0) + stats.get("FALHA", 0))) * 100, 2
        ),
        "modelo_salvo_com_sucesso": sucesso_salvar,
        "analise_comportamento_numeros": analise_numeros,
        "evolucao_modelo": {
            "markov_multiescala_ordens": [1, 2, 3, 4, 5, 6],
            "contextos_conflito_aprendidos": len(getattr(motor.ia, "memoria_conflitos", {})),
            "memoria_conflitos_metricas": getattr(motor.ia, "memoria_conflitos_metricas", {}),
            "competencia_regras_posicionais": getattr(motor.ia, "regras_competencia_cronologica", {}),
            "competencia_regras_posicionais_metricas": getattr(motor.ia, "regras_competencia_metricas", {}),
            "ml_validacao_cronologica": getattr(motor.ia, "ml_metricas", {}),
            "ml_pesos_adaptativos": getattr(motor.ia, "ml_pesos", {}),
            "q_learning_contextual": getattr(motor.ia, "q_learning_contextual_metricas", {}),
            "aprendizado_temporal_adaptativo": getattr(motor.ia, "temporal_metricas", {}),
            "ml_atualizacao_controlada": getattr(motor.ia, "ml_atualizacao_incremental_metricas", {}),
            "matriz_evolutiva": getattr(motor.ia, "matriz_evolutiva", {}),
            "matriz_deriva_comportamental": motor.ia.mapear_deriva_comportamental_numeros(),
            "consequencia_futura_ultima": getattr(motor.ia, "ultima_consequencia_futura", {}),
            "hierarquia_oficial_motor_v1": getattr(motor.ia, "hierarquia_oficial_metricas", {}),
            "especialista_espelho_inversao_contextos": len(getattr(motor.ia, "especialista_espelho_inversao", {})),
            "competencia_especialistas_cronologica": getattr(motor.ia, "competencia_especialistas", {}),
            "competencia_especialistas_metricas": getattr(motor.ia, "competencia_metricas", {}),
            "competencia_especialistas_contextual": getattr(motor.ia, "competencia_contextual_metricas", {}),
            "competencia_camadas_ampliadas": getattr(motor.ia, "camadas_ampliadas_competencia", {}),
            "camadas_ampliadas_metricas": getattr(motor.ia, "camadas_ampliadas_metricas", {}),
            "contagens_projetivas_respeito": getattr(motor.ia, "projecoes_respeito_metricas", {}),
            "cartografia_completa_xls": getattr(motor.ia, "cartografia_xls_metricas", {}),
            "cartografia_trajetoria_streak": getattr(motor.ia, "cartografia_trajetoria_streak_metricas", {}),
            "cartografia_morfologia_estrutural": getattr(motor.ia, "cartografia_morfologia_estrutural_metricas", {}),
            "cartografia_contextual_interna_padroes": getattr(
                motor.ia, "cartografia_padroes_contextual_metricas", {}
            ),
            "cartografia_contextual_interna_regras_contagens": getattr(
                motor.ia, "cartografia_regras_contextual_metricas", {}
            ),
            "filtro_discriminativo_g0_g1_vs_g2_mais": getattr(motor.ia, "filtro_discriminativo_metricas", {}),
            "auditoria_contrafactual_filtro_discriminativo": getattr(motor, "auditoria_contrafactual_filtro_discriminativo", {}),
            "risk_coverage_sinais_liberados": getattr(motor, "risk_coverage_metricas", {}),
            "configuracao_filtro_discriminativo": getattr(motor.ia, "filtro_discriminativo_config", {}),
            "objetivo_operacional_g0_g1": {
                "ativo": True,
                "G0": "SUCESSO_PRIORITARIO",
                "G1": "SUCESSO_ACEITAVEL",
                "G2": "RISCO_OPERACIONAL",
                "FALHA": "RISCO_OPERACIONAL"
            },
            "especialista_risco_g2_mais": getattr(motor.ia, "risco_g2_mais_metricas", {}),
            "configuracao_risco_g2_mais": getattr(motor.ia, "risco_g2_mais_config", {}),
            "projecoes_bilaterais_ativas": False,
            "projecoes_respeito_ativas": True,
            "regra_v3_original_preservada": True,
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        },
        "ia_treinada": motor.ia,
        "mensagem": "Treinamento profundo concluído com sucesso."
    }
