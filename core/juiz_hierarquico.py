# core/juiz_hierarquico.py
from utils.helpers import hash_chave
from core.consenso import ConsensoPonderado

class JuizHierarquicoModificado:
    """
    Juiz da Hierarquia Oficial do MOTOR V1.
    Agora utiliza Consenso Ponderado por Evidência (PDE) – substitui a hierarquia fixa.
    """

    @staticmethod
    def arbitrar_sinal(no_call_ativo, motivo_nc, expectations, inclinacao_num, geometria_mercado,
                       previsao_ia, status_inversao, historico_regras,
                       modo_mercado="NEUTRO",
                       streak_atual=0, xadrez_len=0, xadrez_quebrou=False,
                       contexto_exaustao=False, sintese_evidencias=None,
                       probabilidade_markov=None, ia_modelo=None, entropia_shannon=0.0,
                       influencia_radar=None):
        """
        Coleta todas as evidências, calcula pesos dinâmicos e decide por consenso.
        Vetores de NO CALL soberanos são mantidos (Q-Learning, entropia, riscos G2+).
        """
        # ============================================================
        # NÍVEL 1 – NO CALL é soberano (mantido)
        # ============================================================
        if no_call_ativo:
            return "NO CALL", motivo_nc, "SISTEMA_TRAVADO"

        # Q-Learning veto (mantido)
        if ia_modelo and hasattr(ia_modelo, "q_table"):
            estado_rl = None
            if hasattr(ia_modelo, "construir_estado_q_contextual"):
                try:
                    estado_rl = ia_modelo.construir_estado_q_contextual(
                        getattr(ia_modelo, "_ultima_janela_numeros", []),
                        getattr(ia_modelo, "_ultima_janela_cores", []),
                        analise={
                            "geometria": geometria_mercado,
                            "regras_posicionais": expectations,
                            "contexto_avancado": {"modo_mercado": modo_mercado},
                            "entropia": entropia_shannon,
                            "probabilidade_markov": probabilidade_markov
                        },
                        entropia_shannon=entropia_shannon,
                        probabilidade_markov=probabilidade_markov
                    )
                except Exception:
                    estado_rl = None
            estado_rl_legado = f"HMM_{modo_mercado}_Entropia_{round(entropia_shannon, 1)}"
            chave_estado_rl = hash_chave(estado_rl)
            chave_estado_legado = hash_chave(estado_rl_legado)
            estado_consulta = chave_estado_rl if chave_estado_rl in ia_modelo.q_table else chave_estado_legado
            if estado_consulta in ia_modelo.q_table:
                q_apostar = ia_modelo.q_table[estado_consulta]["APOSTAR"]
                q_no_call = ia_modelo.q_table[estado_consulta]["NO_CALL"]
                if q_no_call > q_apostar + 0.3:
                    return "NO CALL", (
                        f"Agente RL Autônomo VETOU: operar nesse cenário tem ROI "
                        f"negativo histórico (Q-Value: {q_apostar:.2f})"
                    ), "VETO_RL"

        # ============================================================
        # COLETA TODAS AS EVIDÊNCIAS E CALCULA CONSENSO
        # ============================================================
        sub_num = getattr(ia_modelo, "_ultima_janela_numeros", []) if ia_modelo else []
        sub_pol = getattr(ia_modelo, "_ultima_janela_cores", []) if ia_modelo else []

        votos = ConsensoPonderado.coletar_votos(
            ia_modelo=ia_modelo,
            expectations=expectations,
            geometria_mercado=geometria_mercado,
            previsao_ia=previsao_ia,
            probabilidade_markov=probabilidade_markov,
            influencia_radar=influencia_radar,
            sub_num=sub_num,
            sub_pol=sub_pol
        )

        if not votos:
            return "NO CALL", "Nenhuma evidência válida para formar consenso.", "FALLBACK_SEM_EVIDENCIAS"

        direcao, score_v, score_p, detalhes = ConsensoPonderado.calcular_consenso(votos, limiar_minimo=0.05)

        if direcao == "NO_CALL":
            return "NO CALL", (
                f"Consenso inconclusivo: Vermelho={detalhes['score_vermelho']:.3f}, "
                f"Preto={detalhes['score_preto']:.3f} (diferença {detalhes['diferenca']:.3f} < 0.05). "
                f"{detalhes['total_votos']} evidências analisadas."
            ), "NO_CALL_CONSENSO_INCONCLUSIVO"

        # ============================================================
        # APLICA OS VETORES DE RISCO (sobre a direção escolhida)
        # ============================================================
        if ia_modelo is not None:
            ia_modelo._ultima_direcao_pre_filtro_discriminativo = direcao
            ia_modelo._ultima_avaliacao_filtro_discriminativo = None

        # Filtro discriminativo
        if ia_modelo and hasattr(ia_modelo, "avaliar_filtro_discriminativo_g0_g1"):
            try:
                filtro = ia_modelo.avaliar_filtro_discriminativo_g0_g1(
                    sub_num, sub_pol, direcao
                )
                ia_modelo._ultima_avaliacao_filtro_discriminativo = dict(filtro)
                if filtro.get("vetar"):
                    return (
                        "NO CALL",
                        f"Veto discriminativo G0/G1 x G2+: {filtro.get('risco_estimado', 0):.2f}% de risco "
                        f"com {filtro.get('contextos_risco_alto', 0)} evidências de risco "
                        f"({', '.join(filtro.get('fontes_risco_alto', []))}). "
                        f"Consenso original: {direcao} (score {detalhes['score_vermelho'] if direcao == 'VERMELHO' else detalhes['score_preto']:.3f}).",
                        "VETO_DISCRIMINATIVO_G0_G1"
                    )
            except Exception as e:
                print(f"[FILTRO DISCRIMINATIVO] Ignorado: {e}")

        # Especialista de risco G2+
        if ia_modelo and hasattr(ia_modelo, "avaliar_risco_g2_mais"):
            try:
                risco = ia_modelo.avaliar_risco_g2_mais(
                    sub_num, sub_pol, direcao
                )
                if risco.get("vetar"):
                    return (
                        "NO CALL",
                        f"Veto de risco G2+ [{risco.get('tipo_veto', 'ABSOLUTO')}]: "
                        f"{risco.get('risco_estimado', 0):.2f}% de não resolução até G1 "
                        f"em {risco.get('contextos_risco_alto', 0)} contextos históricos concordantes.",
                        "VETO_RISCO_G2_MAIS"
                    )
            except Exception as e:
                print(f"[RISCO G2+] Ignorado: {e}")

        # Streak consolidada (vetor soberano)
        if ia_modelo and hasattr(ia_modelo, "obter_voto_streak_consolidado"):
            try:
                voto_streak = ia_modelo.obter_voto_streak_consolidado(sub_num, sub_pol)
                if voto_streak and voto_streak.get("ativo") and voto_streak.get("direcao") in ("VERMELHO", "PRETO"):
                    if voto_streak.get("direcao") != direcao:
                        margem = float(voto_streak.get("margem", 0.0))
                        suporte = int(voto_streak.get("suporte", 0))
                        if margem >= 0.06 and suporte >= 30:
                            return (
                                "NO CALL",
                                f"Veto por oposição de Streak consolidada: {voto_streak.get('streak')}x {voto_streak.get('cor_streak')} "
                                f"aponta {voto_streak.get('direcao')} contra a direção {direcao}. "
                                f"Margem={margem*100:.2f} p.p., suporte={suporte}.",
                                "NO_CALL_STREAK_OPOSICAO"
                            )
            except Exception as e:
                print(f"[STREAK] Ignorado: {e}")

        # Morfologia estrutural (vetor soberano)
        if ia_modelo and hasattr(ia_modelo, "obter_voto_morfologia_estrutural"):
            try:
                voto_morf = ia_modelo.obter_voto_morfologia_estrutural(sub_num, sub_pol)
                if voto_morf and voto_morf.get("ativo") and voto_morf.get("direcao") in ("VERMELHO", "PRETO"):
                    if voto_morf.get("direcao") != direcao:
                        margem = float(voto_morf.get("margem", 0.0))
                        suporte = int(voto_morf.get("suporte", 0))
                        if margem >= 0.07 and suporte >= 40:
                            return (
                                "NO CALL",
                                f"Veto por oposição morfológica: forma {voto_morf.get('morfologia', {}).get('morfologia', '')} "
                                f"aponta {voto_morf.get('direcao')} contra a direção {direcao}. "
                                f"Margem={margem*100:.2f} p.p., suporte={suporte}.",
                                "NO_CALL_MORFOLOGIA_OPOSICAO"
                            )
            except Exception as e:
                print(f"[MORFOLOGIA] Ignorado: {e}")

        # ============================================================
        # MONTAGEM DO MOTIVO FINAL
        # ============================================================
        fontes_vencedoras = [v["fonte"] for v in votos if v["direcao"] == direcao]
        fontes_contrarias = [v["fonte"] for v in votos if v["direcao"] != direcao and v["direcao"] in ("VERMELHO", "PRETO")]

        motivo = (
            f"Consenso Ponderado por Evidência: {direcao} venceu com score "
            f"{detalhes['score_vermelho'] if direcao == 'VERMELHO' else detalhes['score_preto']:.3f} "
            f"contra {detalhes['score_preto'] if direcao == 'VERMELHO' else detalhes['score_vermelho']:.3f} "
            f"(margem {detalhes['diferenca']:.3f}). "
            f"{len(fontes_vencedoras)} evidências a favor ({', '.join(fontes_vencedoras[:4])})"
            f"{' e ' + ', '.join(fontes_contrarias[:2]) + ' contra' if fontes_contrarias else ''}."
        )

        # Guarda os detalhes do consenso para o relatório
        if ia_modelo:
            ia_modelo._ultimo_consenso = detalhes

        return direcao, motivo, "CONSENSO_PONDERADO"
