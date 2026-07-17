# orquestrador.py
import os
import pandas as pd
import pickle
import json
from datetime import datetime
import time
import tempfile
import math
import gc
from collections import defaultdict

# Importações internas (certifique-se de que os caminhos estão corretos)
from core.preditor_base import IAPreditivaV1
from core.motor_analise import MotorAnalise
from core.juiz_hierarquico import JuizHierarquicoModificado
from core.leitor_xls import LeitorXLS
from core.persistence import salvar_modelo_longo_prazo, carregar_modelo_longo_prazo
from core.settings import NOME_BASE_DEFINITIVA, VERSAO_CHAVES_HASH
from core.helpers import fabrica_historico_regras_auditado

# ============================================================
# FUNÇÕES GLOBAIS (estavam no main 152)
# ============================================================

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
    auditoria_walk_forward = None
    modelo_historico = carregar_modelo_longo_prazo()

    if modelo_historico is None and len(base_existente) >= 30:
        modelo_historico = IAPreditivaV1(base_existente, [])

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

    ia._normalizar_unidade_analise_compatibilidade()

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


# ============================================================
# CLASSES AUXILIARES
# ============================================================

class SequenciaOperacional:
    def __init__(self, lista_resultados):
        self.cronologia = lista_resultados
        self.numerica = [int(r['numero']) for r in self.cronologia]
        self.polaridades = [str(r['cor']).upper() for r in self.cronologia]
        self.total = len(self.numerica)


class MotorV1Completo:
    def __init__(self, lista_dados_xls, ia_existente=None):
        self.seq = SequenciaOperacional(lista_dados_xls)
        self.dados_longo = lista_dados_xls
        
        global motor_unificado
        if ia_existente is not None: 
            self.ia = ia_existente
        elif 'motor_unificado' in globals() and motor_unificado.ia is not None and len(lista_dados_xls) <= 1000:
            self.ia = motor_unificado.ia
        else:
            base_recencia = None
            if os.path.exists("base_recencia_ativa.xlsx"):
                try: base_recencia = LeitorXLS("base_recencia_ativa.xlsx").ler_e_validar()
                except: pass
            self.ia = IAPreditivaV1(self.dados_longo, base_recencia if base_recencia else [])
            
        self.historico_regras = defaultdict(fabrica_historico_regras_auditado)
        self.stats = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0, "NO CALL": 0}
        self.auditoria_contrafactual_filtro_discriminativo = {}

    def processar_auditoria(self, aprender_durante_auditoria=False):
        idx = 0
        memorias = []
        stats = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0, "NO CALL": 0}

        cf_total_vetado = 0
        cf_total_preservado = 0
        cf_vetados = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0}
        cf_preservados = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0}
        cf_fontes = {
            "TRIGRAMA": 0,
            "BIGRAMA": 0,
            "PROJETIVA": 0,
            "MARKOV": 0,
            "ESPELHO_INVERSAO": 0,
            "CARTOGRAFIA_NUMERO": 0,
            "CARTOGRAFIA_STREAK": 0,
            "CARTOGRAFIA_XADREZ": 0,
            "CARTOGRAFIA_TRAJETORIA": 0,
            "CARTOGRAFIA_PADRAO": 0
        }
        cf_contextos_cartografia_consultados = 0
        cf_contextos_cartografia_risco_alto = 0
        cf_vetos_por_cartografia = 0
        cf_cartografia_vetados = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0}
        cf_riscos_vetados = []
        cf_riscos_preservados = []

        while idx + 12 < self.seq.total:
            sub_num = self.seq.numerica[idx:idx+12]
            sub_pol = self.seq.polaridades[idx:idx+12]
            if self.ia is not None:
                self.ia._ultima_avaliacao_filtro_discriminativo = None
                self.ia._ultima_direcao_pre_filtro_discriminativo = None
            analise = MotorAnalise.analisar_janela(
                sub_num, sub_pol, self.ia, eh_sinal_real=True
            )

            regra_id = "NENHUMA"

            if analise["no_call"]["ativo"]:
                sinal = "NO CALL"
                justificativa = analise["no_call"]["motivo"]
                regra_id = "SISTEMA_TRAVADO"
            else:
                geometria = analise["geometria"]
                expectativas = analise["regras_posicionais"]
                direcao_ia = analise["ia"]["direcao"]
                conf_ia = analise["ia"]["confianca"]
                raciocinio_ia = analise["ia"]["raciocinio"]
                streak = analise["contexto_reversao"]["streak"]
                xadrez_len = analise["contexto_reversao"]["xadrez_len"]
                xadrez_quebrou = analise["contexto_reversao"]["xadrez_quebrou"]
                contexto_exaustao = analise["contexto_reversao"]["exaustao"]
                modo_mercado = analise["contexto_avancado"].get("modo_mercado", "NEUTRO")
                
                sinal, justificativa, regra_id = JuizHierarquicoModificado.arbitrar_sinal(
                    no_call_ativo=False, motivo_nc="", expectations=expectativas, inclinacao_num=None, 
                    geometria_mercado=geometria, previsao_ia=(direcao_ia, conf_ia, raciocinio_ia), 
                    status_inversao=None, historico_regras=self.historico_regras, modo_mercado=modo_mercado,
                    streak_atual=streak, xadrez_len=xadrez_len, xadrez_quebrou=xadrez_quebrou,
                    contexto_exaustao=contexto_exaustao, probabilidade_markov=analise.get("probabilidade_markov"),
                    ia_modelo=self.ia, entropia_shannon=analise.get("entropia", 0.0)
                )

            correcoes = self.seq.polaridades[idx+12 : idx+15]
            classificacao = "FALHA"
            salto = 3

            if sinal == "NO CALL":
                classificacao = "NO CALL"
                salto = 1
            else:
                letra = "V" if sinal == "VERMELHO" else "P"
                for g, cor in enumerate(correcoes):
                    if cor == letra or cor == "B":
                        classificacao = f"G{g}"
                        salto = g + 1
                        break

            avaliacao_cf = getattr(self.ia, "_ultima_avaliacao_filtro_discriminativo", None)
            direcao_cf = getattr(self.ia, "_ultima_direcao_pre_filtro_discriminativo", None)

            if (
                isinstance(avaliacao_cf, dict)
                and avaliacao_cf.get("ativo")
                and direcao_cf in ("VERMELHO", "PRETO")
            ):
                letra_cf = "V" if direcao_cf == "VERMELHO" else "P"
                classificacao_cf = "FALHA"
                for g_cf, cor_cf in enumerate(correcoes):
                    if cor_cf == letra_cf or cor_cf == "B":
                        classificacao_cf = f"G{g_cf}"
                        break

                risco_cf = avaliacao_cf.get("risco_estimado")
                if isinstance(risco_cf, (int, float)):
                    if avaliacao_cf.get("vetar"):
                        cf_riscos_vetados.append(float(risco_cf))
                    else:
                        cf_riscos_preservados.append(float(risco_cf))

                cf_contextos_cartografia_consultados += int(
                    avaliacao_cf.get("CONTEXTOS_CARTOGRAFIA_CONSULTADOS", 0)
                )
                cf_contextos_cartografia_risco_alto += int(
                    avaliacao_cf.get("CONTEXTOS_CARTOGRAFIA_RISCO_ALTO", 0)
                )

                if avaliacao_cf.get("vetar"):
                    cf_total_vetado += 1
                    cf_vetados[classificacao_cf] = cf_vetados.get(classificacao_cf, 0) + 1

                    if avaliacao_cf.get("VETO_POR_CARTOGRAFIA"):
                        cf_vetos_por_cartografia += 1
                        cf_cartografia_vetados[classificacao_cf] = (
                            cf_cartografia_vetados.get(classificacao_cf, 0) + 1
                        )

                    for fonte_cf in avaliacao_cf.get("fontes_risco_alto", []):
                        fonte_cf = str(fonte_cf).upper()
                        if "TRIGRAMA" in fonte_cf:
                            cf_fontes["TRIGRAMA"] += 1
                        elif "BIGRAMA" in fonte_cf:
                            cf_fontes["BIGRAMA"] += 1
                        elif "PROJETIVA" in fonte_cf:
                            cf_fontes["PROJETIVA"] += 1
                        elif "MARKOV" in fonte_cf:
                            cf_fontes["MARKOV"] += 1
                        elif "ESPELHO_INVERSAO" in fonte_cf:
                            cf_fontes["ESPELHO_INVERSAO"] += 1
                        elif "CARTOGRAFIA_NUMERO" in fonte_cf:
                            cf_fontes["CARTOGRAFIA_NUMERO"] += 1
                        elif "CARTOGRAFIA_STREAK" in fonte_cf:
                            cf_fontes["CARTOGRAFIA_STREAK"] += 1
                        elif "CARTOGRAFIA_XADREZ" in fonte_cf:
                            cf_fontes["CARTOGRAFIA_XADREZ"] += 1
                        elif "CARTOGRAFIA_TRAJETORIA" in fonte_cf:
                            cf_fontes["CARTOGRAFIA_TRAJETORIA"] += 1
                        elif "CARTOGRAFIA_PADRAO" in fonte_cf:
                            cf_fontes["CARTOGRAFIA_PADRAO"] += 1
                else:
                    cf_total_preservado += 1
                    cf_preservados[classificacao_cf] = cf_preservados.get(classificacao_cf, 0) + 1

            stats[classificacao] = stats.get(classificacao, 0) + 1

            if aprender_durante_auditoria:
                if self.ia is not None:
                    entropia_shannon_atual = analise.get("entropia", 0.0)

                    estado_rl_historico = self.ia.construir_estado_q_contextual(
                        sub_num,
                        sub_pol,
                        analise=analise,
                        entropia_shannon=entropia_shannon_atual,
                        probabilidade_markov=analise.get("probabilidade_markov")
                    )
                    acao_rl_historico = "APOSTAR" if sinal != "NO CALL" else "NO_CALL"
                    
                    recompensa = 0.0
                    if classificacao in ["G0", "G1"]:
                        recompensa = 1.0
                    elif classificacao == "G2":
                        recompensa = -0.5
                    elif classificacao == "FALHA":
                        recompensa = -2.0
                        
                    self.ia.atualizar_q_learning(estado_rl_historico, acao_rl_historico, recompensa)
                
                if classificacao in ["G0", "G1"]:
                    contexto_analise = {
                        "geometria": analise.get("geometria"),
                        "regras_posicionais": analise.get("regras_posicionais"),
                        "controlador_retardador": analise.get("controlador_retardador", {}),
                        "contexto_avancado": {"modo_mercado": analise.get("contexto_avancado", {}).get("modo_mercado", "NEUTRO")}
                    }
                    self.ia.registrar_padrao_vencedor(contexto_analise, classificacao)
    
                if regra_id not in ["NENHUMA", "SISTEMA_TRAVADO"]:
                    self.historico_regras[regra_id]["total"] += 1
                    if classificacao in ["G0", "G1"]:
                        self.historico_regras[regra_id]["acertos"] += 1
    
                if self.ia is not None and hasattr(self.ia, "registrar_resultado_conflito"):
                    self.ia.registrar_resultado_conflito(
                        analise.get("regras_posicionais", []),
                        analise.get("geometria", "ESTÁVEL"),
                        analise.get("contexto_avancado", {}).get("modo_mercado", "NEUTRO"),
                        analise.get("probabilidade_markov", {}),
                        correcoes
                    )

                bloco = [{"numero": self.seq.numerica[k], "cor": self.seq.polaridades[k]} for k in range(idx, min(idx + 12 + salto, self.seq.total))]
                contexto_injecao = {
                    "regras_posicionais": analise.get("regras_posicionais", []),
                    "controlador_retardador": analise.get("controlador_retardador", {}),
                    "geometria": analise.get("geometria", "ESTÁVEL")
                }
                self.ia.injetar_aprendizado_imediato(bloco, 4, contexto_injecao, salvar_na_recencia=False)
            memorias.append(f"Janela {len(memorias)+1}: {sub_num} -> {sinal} | {justificativa} | {classificacao}")
            idx += 12 + salto

        g0_g1_vetados = cf_vetados.get("G0", 0) + cf_vetados.get("G1", 0)
        g2_falha_evitados = cf_vetados.get("G2", 0) + cf_vetados.get("FALHA", 0)
        eficiencia_veto = (
            (g2_falha_evitados / cf_total_vetado) * 100
            if cf_total_vetado > 0 else 0.0
        )

        self.auditoria_contrafactual_filtro_discriminativo = {
            "ativo": True,
            "metodo": "CONTRAFACTUAL_DIRECAO_PRE_VETO_SEM_ALTERAR_DECISAO",
            "TOTAL AVALIADO": cf_total_vetado + cf_total_preservado,
            "TOTAL VETADO": cf_total_vetado,
            "TOTAL PRESERVADO": cf_total_preservado,
            "VETADOS_G0": cf_vetados.get("G0", 0),
            "VETADOS_G1": cf_vetados.get("G1", 0),
            "VETADOS_G2": cf_vetados.get("G2", 0),
            "VETADOS_FALHA": cf_vetados.get("FALHA", 0),
            "PRESERVADOS_G0": cf_preservados.get("G0", 0),
            "PRESERVADOS_G1": cf_preservados.get("G1", 0),
            "PRESERVADOS_G2": cf_preservados.get("G2", 0),
            "PRESERVADOS_FALHA": cf_preservados.get("FALHA", 0),
            "G0_G1_VETADOS": g0_g1_vetados,
            "G2_FALHA_EVITADOS": g2_falha_evitados,
            "EFICIENCIA_VETO_G2_FALHA_PERCENT": round(eficiencia_veto, 2),
            "FONTES_RISCO_ALTO": cf_fontes,
            "VETOS_POR_FONTE": dict(cf_fontes),
            "CONTEXTOS_CARTOGRAFIA_CONSULTADOS": cf_contextos_cartografia_consultados,
            "CONTEXTOS_CARTOGRAFIA_RISCO_ALTO": cf_contextos_cartografia_risco_alto,
            "VETOS_POR_CARTOGRAFIA": cf_vetos_por_cartografia,
            "G0_G1_VETADOS_CARTOGRAFIA": (
                cf_cartografia_vetados.get("G0", 0) + cf_cartografia_vetados.get("G1", 0)
            ),
            "G2_FALHA_EVITADOS_CARTOGRAFIA": (
                cf_cartografia_vetados.get("G2", 0) + cf_cartografia_vetados.get("FALHA", 0)
            ),
            "CARTOGRAFIA_VETADOS_G0": cf_cartografia_vetados.get("G0", 0),
            "CARTOGRAFIA_VETADOS_G1": cf_cartografia_vetados.get("G1", 0),
            "CARTOGRAFIA_VETADOS_G2": cf_cartografia_vetados.get("G2", 0),
            "CARTOGRAFIA_VETADOS_FALHA": cf_cartografia_vetados.get("FALHA", 0),
            "RISCO_MEDIO_VETADOS_PERCENT": round(
                sum(cf_riscos_vetados) / len(cf_riscos_vetados), 2
            ) if cf_riscos_vetados else 0.0,
            "RISCO_MEDIO_PRESERVADOS_PERCENT": round(
                sum(cf_riscos_preservados) / len(cf_riscos_preservados), 2
            ) if cf_riscos_preservados else 0.0,
            "altera_sinal_operacional": False
        }

        self.stats = stats
        sinais_liberados = stats.get("G0", 0) + stats.get("G1", 0) + stats.get("G2", 0) + stats.get("FALHA", 0)
        sucessos_ate_g1 = stats.get("G0", 0) + stats.get("G1", 0)
        riscos_g2_falha = stats.get("G2", 0) + stats.get("FALHA", 0)
        self.risk_coverage_metricas = {
            "ativo": True,
            "sinais_liberados": sinais_liberados,
            "cobertura_percent": round((sinais_liberados / max(1, sum(stats.values()))) * 100, 2),
            "assertividade_seletiva_g0_g1_percent": round((sucessos_ate_g1 / max(1, sinais_liberados)) * 100, 2),
            "risco_seletivo_g2_falha_percent": round((riscos_g2_falha / max(1, sinais_liberados)) * 100, 2),
            "objetivo": "MEDIR_RISCO_DOS_SINAIS_LIBERADOS_SEM_CRIAR_NO_CALL"
        }
        total_janelas = sum(stats.values())
        denom = total_janelas if total_janelas > 0 else 1
        sinais_liberados = stats.get("G0", 0) + stats.get("G1", 0) + stats.get("G2", 0) + stats.get("FALHA", 0)
        g0_g1_liberados = stats.get("G0", 0) + stats.get("G1", 0)
        g2_falha_liberados = stats.get("G2", 0) + stats.get("FALHA", 0)
        baseline_g0_g1 = 78.22
        self.metricas_risco_cobertura = {
            "COBERTURA_PERCENT": round((sinais_liberados / total_janelas) * 100, 2) if total_janelas else 0.0,
            "RISCO_SELETIVO_G2_FALHA_PERCENT": round((g2_falha_liberados / sinais_liberados) * 100, 2) if sinais_liberados else 0.0,
            "ASSERTIVIDADE_G0_G1_LIBERADOS_PERCENT": round((g0_g1_liberados / sinais_liberados) * 100, 2) if sinais_liberados else 0.0,
            "BASELINE_TEORICO_G0_G1_PERCENT": baseline_g0_g1,
            "LIFT_SOBRE_BASELINE_PONTOS_PERCENTUAIS": round(((g0_g1_liberados / sinais_liberados) * 100) - baseline_g0_g1, 2) if sinais_liberados else 0.0,
            "SINAIS_LIBERADOS": sinais_liberados,
            "TOTAL_OPORTUNIDADES": total_janelas
        }
        
        output = "[MEMÓRIA DE CÁLCULO DAS JANELAS MÓVEIS]\n"
        output += "\n".join(memorias) + "\n\n"
        output += "[RESULTADO FINAL TIPO D]\n"
        output += f"CRONOLOGIA VALIDADA: {self.seq.total} Resultados\n"
        output += f"TOTAL DE JANELAS AUDITADAS: {len(memorias)}\n"
        output += f" - Taxa G0: {stats.get('G0',0)} Ocorrências ({(stats.get('G0',0)/denom)*100:.2f}%)\n"
        output += f" - Taxa G1: {stats.get('G1',0)} Ocorrências ({(stats.get('G1',0)/denom)*100:.2f}%)\n"
        output += f" - Taxa G2: {stats.get('G2',0)} Ocorrências ({(stats.get('G2',0)/denom)*100:.2f}%)\n"
        output += f" - Taxa de Falha: {stats.get('FALHA',0)} Ocorrências ({(stats.get('FALHA',0)/denom)*100:.2f}%)\n"
        output += f" - Taxa de NO CALL: {stats.get('NO CALL',0)} Ocorrências ({(stats.get('NO CALL',0)/denom)*100:.2f}%)\n\n"
        
        if stats.get("FALHA", 0) >= 25:
            condicao = "MERCADO EM DEGRADAÇÃO"
        elif stats.get("G0", 0) >= 50:
            condicao = "MERCADO PAGADOR"
        else:
            condicao = "MERCADO INSTÁVEL"
            
        output += f"ESTADO ATUAL DO MERCADO: {condicao}\n"
        return output


class ProcessadorTipoB:
    def __init__(self, sequencia_12_numeros, caminho_base_dados):
        self.entrada = sequencia_12_numeros
        self.caminho_base = caminho_base_dados
        self.polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in sequencia_12_numeros]

    def executar_sinal_real(self):
        if len(self.entrada) != 12: return {"erro": "Necessário exatamente 12 números"}
        ia = carregar_modelo_longo_prazo()
        if ia is None:
            base = LeitorXLS(self.caminho_base).ler_e_validar()
            if not base: return {"erro": "Base de dados não encontrada"}
            ia = IAPreditivaV1(base, None)
        regime_rec = None
        if os.path.exists("base_recencia_ativa.xlsx"):
            base_rec = LeitorXLS("base_recencia_ativa.xlsx").ler_e_validar()
            if base_rec:
                ia = integrar_recencia_no_modelo(base_rec, 6)
                regime_rec = ia.regime_recencia
                
        analise = MotorAnalise.analisar_janela(self.entrada, self.polaridades, ia, eh_sinal_real=True)
        
        nc_ativo = analise["no_call"]["ativo"]
        motivo_nc = analise["no_call"]["motivo"]
        geometria = analise["geometria"]
        expectativas = analise["regras_posicionais"]
        direcao_ia = analise["ia"]["direcao"] if not nc_ativo else "NEUTRO"
        conf_ia = analise["ia"]["confianca"] if not nc_ativo else 0.0
        raciocinio_ia = analise["ia"]["raciocinio"] if not nc_ativo else motivo_nc
        streak = analise["contexto_reversao"]["streak"]
        xadrez_len = analise["contexto_reversao"]["xadrez_len"]
        xadrez_quebrou = analise["contexto_reversao"]["xadrez_quebrou"]
        contexto_exaustao = analise["contexto_reversao"]["exaustao"]
        modo_mercado = analise["contexto_avancado"].get("modo_mercado", "NEUTRO")
        raciocinio_trace = analise["camadas"]
        
        if nc_ativo:
            return {
                "sinal": "NO CALL", "justificativa": motivo_nc, "no_call": True, "regime_recencia": regime_rec,
                "motivo_real": f"NO CALL: {motivo_nc}", "regra_id": "SISTEMA_TRAVADO",
                "entropia": analise.get("entropia"), "probabilidade_markov": analise.get("probabilidade_markov")
            }
            
        sinal_final, justificativa_final, regra_id_final = JuizHierarquicoModificado.arbitrar_sinal(
            no_call_ativo=False, motivo_nc="", expectations=expectativas, inclinacao_num=None, geometria_mercado=geometria,
            previsao_ia=(direcao_ia, conf_ia, raciocinio_ia), status_inversao=None, historico_regras=ia.historico_regras if ia else {},
            modo_mercado=modo_mercado, streak_atual=streak, xadrez_len=xadrez_len, xadrez_quebrou=xadrez_quebrou,
            contexto_exaustao=contexto_exaustao, probabilidade_markov=analise.get("probabilidade_markov"),
            ia_modelo=ia, entropia_shannon=analise.get("entropia", 0.0)
        )

        if sinal_final != "NO CALL" and streak >= 6:
            sinal_final = "NO CALL"
            justificativa_final = f"Veto de streak {streak}x (segurança anti-tendência)"
            regra_id_final = "VETO_STREAK"
            
        return {
            "sinal": sinal_final, "justificativa": justificativa_final, "confianca_ia": round(conf_ia, 2),
            "no_call": False, "regime_recencia": regime_rec, "motivo_real": justificativa_final,
            "raciocinio_trace": raciocinio_trace, "decisao_final": {"sinal": sinal_final, "justificativa": justificativa_final, "regra_id": regra_id_final},
            "entropia": analise.get("entropia"), "probabilidade_markov": analise.get("probabilidade_markov")
        }


class MotorUnificadoV1:
    def __init__(self):
        self.ia = None
        self.regime_recencia = None
        self.ultima_atualizacao = None
        self.base_longa_carregada = False
        self.recencia_injetada = False

    def carregar_tudo(self, forcar_recencia=True):
        print("[MOTOR UNIFICADO] Iniciando carregamento completo...")
        self.ia = carregar_modelo_longo_prazo()

        if self.ia and len(self.ia.dados_recencia) > 200:
            self.ia.dados_recencia = self.ia.dados_recencia[-200:]

        if self.ia is None:
            if os.path.exists(NOME_BASE_DEFINITIVA):
                dados_longos = LeitorXLS(NOME_BASE_DEFINITIVA).ler_e_validar()
                if dados_longos and len(dados_longos) >= 50:
                    relatorio = treinar_base_longo_prazo_com_janelas(dados_longos)
                    self.ia = relatorio.get("ia_treinada")
                    self.base_longa_carregada = self.ia is not None
                    if self.ia is not None:
                        salvar_modelo_longo_prazo(self.ia)
                del dados_longos
                gc.collect()
        else:
            self.base_longa_carregada = True

        if forcar_recencia:
            if self.ia and not getattr(self.ia, "recencia_foi_injetada_na_sessao", False):
                self._carregar_e_injetar_recencia()

        self.ultima_atualizacao = datetime.now()
        gc.collect()
        print("[MOTOR UNIFICADO] Carregamento concluído.")

    def _carregar_e_injetar_recencia(self):
        if getattr(self, "recencia_injetada", False): return
        if not os.path.exists("base_recencia_ativa.xlsx"): return
        dados_rec = LeitorXLS("base_recencia_ativa.xlsx").ler_e_validar()
        if not dados_rec or len(dados_rec) < 20: return

        dados_rec = dados_rec[-200:]
        print(f"[MOTOR UNIFICADO] Injetando {len(dados_rec)} registros de recência com peso alto...")
        if self.ia is None:
            self.ia = IAPreditivaV1([], [])

        self.ia.dados_recencia = list(dados_rec)
        self.ia.injetar_aprendizado_imediato(
            dados_rec,
            multiplicador_peso=6,
            salvar_na_recencia=False
        )

        self.ia.treinar_q_learning_contextual(
            dados_rec,
            multiplicador_peso=6,
            origem="RECENCIA"
        )
        self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
        self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
        self.ia.regime_recencia = self.regime_recencia
        self.ia.recencia_foi_injetada_na_sessao = True
        self.recencia_injetada = True

    def absorver_base_longa(self, dados_novos):
        if not dados_novos or len(dados_novos) < 30: return {"sucesso": False, "mensagem": "Base muito pequena."}
        try:
            if os.path.exists(NOME_BASE_DEFINITIVA):
                backup_name = NOME_BASE_DEFINITIVA.replace(".xlsx", f"_backup_substituicao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                os.replace(NOME_BASE_DEFINITIVA, backup_name)
            df_base = pd.DataFrame([{"numero": d["numero"], "cor": d["cor"]} for d in dados_novos])
            df_base.iloc[::-1].reset_index(drop=True).to_excel(NOME_BASE_DEFINITIVA, index=False)
        except Exception as e:
            return {"sucesso": False, "mensagem": f"Falha ao substituir a base definitiva: {e}"}
        relatorio = treinar_base_longo_prazo_com_janelas(dados_novos)
        self.ia = relatorio.get("ia_treinada")
        self.base_longa_carregada = True
        if os.path.exists("base_recencia_ativa.xlsx"): self._carregar_e_injetar_recencia()
        sucesso = salvar_modelo_longo_prazo(self.ia)
        return {"sucesso": True, "registros_absorvidos": len(dados_novos), "modelo_salvo": sucesso, "mensagem": "Absorvido."}

    def processar_novo_lote(self, novos_dados):
        if not novos_dados:
            return {"sucesso": False, "mensagem": "Nenhum dado novo foi fornecido."}
        if self.ia is None:
            self.carregar_tudo(forcar_recencia=False)
        if self.ia is None:
            return {
                "sucesso": False,
                "mensagem": "Modelo persistido não encontrado. Substitua/treine a base definitiva uma vez antes do encadeamento incremental."
            }

        relatorio = adicionar_a_base_longo_prazo(novos_dados)

        if isinstance(relatorio, dict) and relatorio.get("sucesso"):
            ia_atualizada = relatorio.get("ia_treinada")
            if ia_atualizada is not None:
                self.ia = ia_atualizada
            self.base_longa_carregada = self.ia is not None
            self.ultima_atualizacao = datetime.now()

        gc.collect()
        return relatorio

    def processar_recencia(self, dados_recencia):
        if not dados_recencia or len(dados_recencia) < 20: return {"sucesso": False, "mensagem": "Base de recência muito pequena."}
        if self.ia is None: self.carregar_tudo(forcar_recencia=False)

        dados_recencia_ativos = [
            {"numero": d["numero"], "cor": d["cor"]}
            for d in dados_recencia[-200:]
        ]

        self.ia.dados_recencia = list(dados_recencia_ativos)
        self.ia.injetar_aprendizado_imediato(
            dados_recencia_ativos,
            multiplicador_peso=6,
            salvar_na_recencia=False
        )
        self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
        self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
        self.ia.regime_recencia = self.regime_recencia
        self.recencia_injetada = True

        self.ia.treinar_q_learning_contextual(
            dados_recencia_ativos,
            multiplicador_peso=6,
            origem="RECENCIA"
        )
        self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
        self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
        self.ia.regime_recencia = self.regime_recencia
        self.ia.atualizar_matriz_evolutiva()

        salvar_modelo_longo_prazo(self.ia)

        return {"sucesso": True, "registros_processados": len(dados_recencia), "registros_recencia_ativos": len(dados_recencia_ativos), "recencia_separada_base_mestra": True, "regime_recencia": self.regime_recencia, "matriz_evolutiva": self.ia.matriz_evolutiva, "mensagem": "Recência processada com sucesso em buffer separado da Base Mestra."}

    def gerar_sinal_tipo_b(self, sequencia_12):
        if len(sequencia_12) != 12: return {"erro": "Necessário exatamente 12 números"}
        if self.ia is None: self.carregar_tudo()
        polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in sequencia_12]
        
        analise = MotorAnalise.analisar_janela(sequencia_12, polaridades, self.ia, eh_sinal_real=True)
        
        if analise["no_call"]["ativo"]:
            return {"sinal": "NO CALL", "justificativa": analise["no_call"]["motivo"], "no_call": True, "regime_recencia": self.regime_recencia, "motivo_real": f"NO CALL: {analise['no_call']['motivo']}", "regra_id": "SISTEMA_TRAVADO", "entropia": analise.get("entropia"), "probabilidade_markov": analise.get("probabilidade_markov")}
            
        geometria = analise["geometria"]
        expectativas = analise["regras_posicionais"]
        direcao_ia = analise["ia"]["direcao"]
        conf_ia = analise["ia"]["confianca"]
        raciocinio_ia = analise["ia"]["raciocinio"]
        streak = analise["contexto_reversao"]["streak"]
        xadrez_len = analise["contexto_reversao"]["xadrez_len"]
        xadrez_quebrou = analise["contexto_reversao"]["xadrez_quebrou"]
        contexto_exaustao = analise["contexto_reversao"]["exaustao"]
        modo_mercado = analise["contexto_avancado"].get("modo_mercado", "NEUTRO")
        
        sinal_final, justificativa_final, regra_id_final = JuizHierarquicoModificado.arbitrar_sinal(
            no_call_ativo=False, motivo_nc="", expectations=expectativas, inclinacao_num=None, geometria_mercado=geometria,
            previsao_ia=(direcao_ia, conf_ia, raciocinio_ia), status_inversao=None, historico_regras=self.ia.historico_regras if self.ia else {},
            modo_mercado=modo_mercado, streak_atual=streak, xadrez_len=xadrez_len, xadrez_quebrou=xadrez_quebrou,
            contexto_exaustao=contexto_exaustao, probabilidade_markov=analise.get("probabilidade_markov"),
            ia_modelo=self.ia, entropia_shannon=analise.get("entropia", 0.0)
        )

        validacao_contextual = getattr(self.ia, "_ultima_validacao_autoridade_contextual", {}) or {}
        if validacao_contextual.get("ativo"):
            componentes_ctx = validacao_contextual.get("componentes", {}) or {}
            resumo_componentes = [
                (
                    f"{nome}: macro {item.get('taxa_direcao_g0_g1', 0):.2f}% direção vs "
                    f"{item.get('taxa_contraria_g0_g1', 0):.2f}% contrária (n={item.get('suporte', 0)})"
                    + (
                        f" | recente {item.get('taxa_direcao_recente_g0_g1', 0):.2f}% direção vs "
                        f"{item.get('taxa_contraria_recente_g0_g1', 0):.2f}% contrária "
                        f"(n={item.get('suporte_recente', 0)})"
                        if item.get('suporte_recente', 0) else ""
                    )
                )
                for nome, item in componentes_ctx.items()
            ]
            analise["camadas"].append({
                "camada": 7.5,
                "nome": "Validação Contextual da Autoridade Hierárquica",
                "resultado": validacao_contextual.get("status", "SEM_VALIDACAO"),
                "detalhe": (
                    f"Regra: {validacao_contextual.get('regra')} | Direção preservada: {validacao_contextual.get('direcao')} | "
                    f"G0/G1 direção: {validacao_contextual.get('taxa_direcao_g0_g1', 0):.2f}% | "
                    f"Contrária: {validacao_contextual.get('taxa_contraria_g0_g1', 0):.2f}% | "
                    f"Deriva número final: {validacao_contextual.get('estado_deriva_numero_final', 'SEM_SUPORTE')} | "
                    f"Fragmentação: {validacao_contextual.get('fragmentacao_contextual', False)} | "
                    f"Contextos: {'; '.join(resumo_componentes) if resumo_componentes else 'sem componentes com suporte'}"
                ),
                "impacto": "BLOQUEIO" if validacao_contextual.get("vetar") else "VALIDACAO"
            })

        if sinal_final != "NO CALL" and streak >= 6:
            sinal_final = "NO CALL"
            justificativa_final = f"Veto de streak {streak}x (segurança anti-tendência)"
            regra_id_final = "VETO_STREAK"
            
        return {
            "sinal": sinal_final,
            "justificativa": justificativa_final,
            "confianca_ia": round(conf_ia, 2),
            "no_call": False,
            "regime_recencia": self.regime_recencia,
            "motivo_real": justificativa_final,
            "raciocinio_trace": analise["camadas"],
            "regra_id": regra_id_final,
            "entropia": analise.get("entropia"),
            "probabilidade_markov": analise.get("probabilidade_markov"),
            "simulacao_rotas_proximos_resultados": getattr(self.ia, "ultima_simulacao_rotas", {}),
            "confluencia_camadas_ampliadas": getattr(self.ia, "ultima_confluencia_camadas_ampliadas", {}),
            "validacao_contextual_autoridade": getattr(self.ia, "_ultima_validacao_autoridade_contextual", {}),
            "oposicao_causal_consolidada": getattr(self.ia, "_ultima_oposicao_causal_consolidada", {}),
            "auditoria_contrafactual_autorizacao": getattr(self.ia, "auditoria_contrafactual_autorizacao", {})
        }

    def processar_feedback_real(self, sequencia_12, sinal_indicado, regra_id, numeros_saidos, classificacao, entropia_shannon=0.0, probabilidade_markov=None):
        if self.ia is None: self.carregar_tudo()
        polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in sequencia_12]
        analise = MotorAnalise.analisar_janela(sequencia_12, polaridades, self.ia)
        
        modo_mercado = analise.get("contexto_avancado", {}).get("modo_mercado", "NEUTRO")
        geometria = analise.get("geometria", "ESTÁVEL")
        expectativas = analise.get("regras_posicionais", [])
        
        classificacao_limpa = classificacao.split(" ")[0].upper()
        if "LOSS" in classificacao_limpa or "FALHA" in classificacao_limpa: classificacao_limpa = "FALHA"
            
        estado_rl = self.ia.construir_estado_q_contextual(
            sequencia_12,
            polaridades,
            analise=analise,
            entropia_shannon=entropia_shannon,
            probabilidade_markov=probabilidade_markov or analise.get("probabilidade_markov")
        )
        acao_rl = "APOSTAR" if sinal_indicado != "NO CALL" else "NO_CALL"
        if classificacao_limpa in ["G0", "G1"]: recompensa = 1.0
        elif classificacao_limpa == "G2": recompensa = -0.5 
        elif classificacao_limpa == "FALHA": recompensa = -2.0 
        else: recompensa = 0.0
        
        self.ia.atualizar_q_learning(estado_rl, acao_rl, recompensa)
            
        contexto_analise = {
            "geometria": geometria, "regras_posicionais": expectativas, "controlador_retardador": analise.get("controlador_retardador", {}),
            "contexto_avancado": {"modo_mercado": modo_mercado}, "entropia_shannon": entropia_shannon, "monte_carlo_indicou": probabilidade_markov
        }
        
        if classificacao_limpa in ["G0", "G1"]: self.ia.registrar_padrao_vencedor(contexto_analise, classificacao_limpa)

        try:
            memoria_cf = getattr(self.ia, "auditoria_contrafactual_autorizacao", None)
            if not isinstance(memoria_cf, dict):
                memoria_cf = {
                    "total": 0, "oficial_g0_g1": 0, "oposta_g0_g1": 0,
                    "no_call_protegeria_g2_falha": 0, "eventos": []
                }
                self.ia.auditoria_contrafactual_autorizacao = memoria_cf

            direcao_oficial = sinal_indicado if sinal_indicado in ("VERMELHO", "PRETO") else None
            direcao_oposta = (
                "PRETO" if direcao_oficial == "VERMELHO"
                else "VERMELHO" if direcao_oficial == "PRETO"
                else None
            )
            cores_reais = [
                "BRANCO" if int(n) == 0 else ("VERMELHO" if 1 <= int(n) <= 7 else "PRETO")
                for n in numeros_saidos[:2]
            ]
            oficial_g0_g1 = bool(
                direcao_oficial and any(c in (direcao_oficial, "BRANCO") for c in cores_reais)
            )
            oposta_g0_g1 = bool(
                direcao_oposta and any(c in (direcao_oposta, "BRANCO") for c in cores_reais)
            )
            risco_real = classificacao_limpa in ("G2", "FALHA")

            memoria_cf["total"] = int(memoria_cf.get("total", 0)) + 1
            if oficial_g0_g1:
                memoria_cf["oficial_g0_g1"] = int(memoria_cf.get("oficial_g0_g1", 0)) + 1
            if oposta_g0_g1:
                memoria_cf["oposta_g0_g1"] = int(memoria_cf.get("oposta_g0_g1", 0)) + 1
            if risco_real:
                memoria_cf["no_call_protegeria_g2_falha"] = int(
                    memoria_cf.get("no_call_protegeria_g2_falha", 0)
                ) + 1

            eventos_cf = list(memoria_cf.get("eventos", []))
            eventos_cf.append({
                "regra": regra_id,
                "direcao_oficial": direcao_oficial,
                "direcao_oposta": direcao_oposta,
                "classificacao_real": classificacao_limpa,
                "oficial_g0_g1": oficial_g0_g1,
                "oposta_g0_g1": oposta_g0_g1,
                "no_call_protegeria": risco_real,
                "validacao_contextual": dict(
                    getattr(self.ia, "_ultima_validacao_autoridade_contextual", {}) or {}
                ),
                "oposicao_causal": dict(
                    getattr(self.ia, "_ultima_oposicao_causal_consolidada", {}) or {}
                )
            })
            memoria_cf["eventos"] = eventos_cf[-500:]
        except Exception as e:
            print(f"[AUDITORIA CONTRAFACTUAL] Registro ignorado por segurança: {e}")

        if regra_id and regra_id not in ["NENHUMA", "SISTEMA_TRAVADO"]:
            self.ia.historico_regras[regra_id]["total"] += 1
            if classificacao_limpa in ["G0", "G1"]: self.ia.historico_regras[regra_id]["acertos"] += 1
                
        evento_ao_vivo = [int(n) for n in list(sequencia_12) + list(numeros_saidos)]
        cronologia_ao_vivo = list(getattr(self.ia, "cronologia_ao_vivo", []) or [])

        maior_sobreposicao = 0
        limite_sobreposicao = min(len(cronologia_ao_vivo), len(evento_ao_vivo))
        for tamanho_sobreposicao in range(limite_sobreposicao, 0, -1):
            if cronologia_ao_vivo[-tamanho_sobreposicao:] == evento_ao_vivo[:tamanho_sobreposicao]:
                maior_sobreposicao = tamanho_sobreposicao
                break

        if cronologia_ao_vivo and maior_sobreposicao == 0:
            cronologia_ao_vivo = []

        numeros_cronologicamente_novos = evento_ao_vivo[maior_sobreposicao:]
        cronologia_ao_vivo.extend(numeros_cronologicamente_novos)
        self.ia.cronologia_ao_vivo = cronologia_ao_vivo[-5000:]

        dados_novos_completos = []
        for n in numeros_cronologicamente_novos:
            cor = 'B' if n == 0 else ('V' if 1 <= n <= 7 else 'P')
            dados_novos_completos.append({"numero": n, "cor": cor})

        contexto_injecao = {
            "regras_posicionais": expectativas,
            "controlador_retardador": analise.get("controlador_retardador", {}),
            "geometria": geometria
        }
        self.ia.injetar_aprendizado_imediato(
            dados_novos_completos,
            multiplicador_peso=6,
            analise_contexto=contexto_injecao,
            salvar_na_recencia=False
        )
        self.ia.dados_recencia.extend([
            {"numero": n, "cor": ('B' if n == 0 else ('V' if 1 <= n <= 7 else 'P'))}
            for n in numeros_saidos
        ])
        self.ia.dados_recencia = self.ia.dados_recencia[-200:]

        dados_novos_para_arquivo = []
        for n in numeros_cronologicamente_novos:
            cor = 'B' if n == 0 else ('V' if 1 <= n <= 7 else 'P')
            dados_novos_para_arquivo.append({"numero": n, "cor": cor})
            
        recencia_atual = self.ia.dados_recencia.copy() if self.ia else []
        cronologia_ao_vivo_atual = list(getattr(self.ia, "cronologia_ao_vivo", []) or [])
        rel = adicionar_a_base_longo_prazo(dados_novos_para_arquivo, origem_feedback_ao_vivo=True)
        self.carregar_tudo(forcar_recencia=False)
        
        if self.ia:
            self.ia.cronologia_ao_vivo = cronologia_ao_vivo_atual[-5000:]
            self.ia.dados_recencia = recencia_atual[-200:]
            self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
            self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
            self.ia.regime_recencia = self.regime_recencia
            self.ia.atualizar_matriz_evolutiva()
            self.ia.mapear_deriva_comportamental_numeros()
            salvar_modelo_longo_prazo(self.ia)
        return rel

    def status(self):
        volume_longo = len(self.ia.dados_longo) if self.ia and self.ia.dados_longo else 0
        volume_rec = len(self.ia.dados_recencia) if self.ia and self.ia.dados_recencia else 0
        return {
            "ia_carregada": self.ia is not None,
            "base_longa_carregada": self.base_longa_carregada,
            "recencia_injetada": self.recencia_injetada,
            "regime_recencia": self.regime_recencia,
            "volume_longo_prazo": volume_longo,
            "volume_recencia": volume_rec,
            "ultima_atualizacao": self.ultima_atualizacao.isoformat() if self.ultima_atualizacao else None
        }


# ============================================================
# INSTÂNCIA GLOBAL (esperada pelo app.py)
# ============================================================
motor_unificado = MotorUnificadoV1()
