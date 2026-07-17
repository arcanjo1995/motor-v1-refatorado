import os
from collections import defaultdict

from data.leitor_xls import LeitorXLS
from utils.helpers import fabrica_historico_regras_auditado
from core.motor_analise import MotorAnalise
from core.juiz_hierarquico import JuizHierarquicoModificado
from ml_engine.preditor_base import IAPreditivaV1


class SequenciaOperacional:
    def __init__(self, lista_resultados):
        self.cronologia = lista_resultados
        self.numerica = [int(r['numero']) for r in self.cronologia]
        self.polaridades = [str(r['cor']).upper() for r in self.cronologia]
        self.total = len(self.numerica)


class MotorV1Completo:
    """
    Motor completo para auditoria cronológica (Walk-Forward).
    """
    def __init__(self, lista_dados_xls, ia_existente=None):
        self.seq = SequenciaOperacional(lista_dados_xls)
        self.dados_longo = lista_dados_xls

        # Import local para evitar circularidade com motor_unificado
        from services.motor_unificado import motor_unificado

        if ia_existente is not None:
            self.ia = ia_existente
        elif 'motor_unificado' in globals() and motor_unificado.ia is not None and len(lista_dados_xls) <= 1000:
            self.ia = motor_unificado.ia
        else:
            base_recencia = None
            if os.path.exists("base_recencia_ativa.xlsx"):
                try:
                    base_recencia = LeitorXLS("base_recencia_ativa.xlsx").ler_e_validar()
                except:
                    pass
            self.ia = IAPreditivaV1(self.dados_longo, base_recencia if base_recencia else [])

        self.historico_regras = defaultdict(fabrica_historico_regras_auditado)
        self.stats = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0, "NO CALL": 0}
        self.auditoria_contrafactual_filtro_discriminativo = {}

    def processar_auditoria(self, aprender_durante_auditoria=False):
        idx = 0
        memorias = []
        stats = {"G0": 0, "G1": 0, "G2": 0, "FALHA": 0, "NO CALL": 0}

        # MAIN 72: auditoria contrafactual do filtro discriminativo.
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
            sub_num = self.seq.numerica[idx:idx + 12]
            sub_pol = self.seq.polaridades[idx:idx + 12]
            if self.ia is not None:
                self.ia._ultima_avaliacao_filtro_discriminativo = None
                self.ia._ultima_direcao_pre_filtro_discriminativo = None
            # Replay com as mesmas camadas analíticas do sinal Tipo B real.
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

            correcoes = self.seq.polaridades[idx + 12: idx + 15]
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

            # =========================================================
            # MAIN 72 — AUDITORIA CONTRAFACTUAL DO FILTRO DISCRIMINATIVO
            # =========================================================
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

            # =========================================================
            # APRENDIZADO DURANTE AUDITORIA
            # =========================================================
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

                bloco = [{"numero": self.seq.numerica[k], "cor": self.seq.polaridades[k]} for k in
                         range(idx, min(idx + 12 + salto, self.seq.total))]
                contexto_injecao = {
                    "regras_posicionais": analise.get("regras_posicionais", []),
                    "controlador_retardador": analise.get("controlador_retardador", {}),
                    "geometria": analise.get("geometria", "ESTÁVEL")
                }
                self.ia.injetar_aprendizado_imediato(bloco, 4, contexto_injecao, salvar_na_recencia=False)

            memorias.append(f"Janela {len(memorias) + 1}: {sub_num} -> {sinal} | {justificativa} | {classificacao}")
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
        output += f" - Taxa G0: {stats.get('G0', 0)} Ocorrências ({(stats.get('G0', 0) / denom) * 100:.2f}%)\n"
        output += f" - Taxa G1: {stats.get('G1', 0)} Ocorrências ({(stats.get('G1', 0) / denom) * 100:.2f}%)\n"
        output += f" - Taxa G2: {stats.get('G2', 0)} Ocorrências ({(stats.get('G2', 0) / denom) * 100:.2f}%)\n"
        output += f" - Taxa de Falha: {stats.get('FALHA', 0)} Ocorrências ({(stats.get('FALHA', 0) / denom) * 100:.2f}%)\n"
        output += f" - Taxa de NO CALL: {stats.get('NO CALL', 0)} Ocorrências ({(stats.get('NO CALL', 0) / denom) * 100:.2f}%)\n\n"

        if stats.get("FALHA", 0) >= 25:
            condicao = "MERCADO EM DEGRADAÇÃO"
        elif stats.get("G0", 0) >= 50:
            condicao = "MERCADO PAGADOR"
        else:
            condicao = "MERCADO INSTÁVEL"

        output += f"ESTADO ATUAL DO MERCADO: {condicao}\n"
        return output
