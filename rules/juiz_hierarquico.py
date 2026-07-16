from utils.hashing import hash_chave
from rules.motor_nocall import MotorNoCall

class JuizHierarquicoModificado:
    """
    MAIN 114 — Juiz da Hierarquia Oficial do MOTOR V1.
    Não soma votos, famílias ou quantidade de evidências. O primeiro nível
    hierárquico válido e resolvido encerra a busca direcional.
    """

    @staticmethod
    def _nivel_regra(regra, ia_modelo=None):
        if ia_modelo is not None and hasattr(ia_modelo, "_nivel_hierarquico_regra"):
            return ia_modelo._nivel_hierarquico_regra(regra)
        tipo = str((regra or {}).get("tipo_regra", "")).upper()
        if tipo == "COEXISTENCIA_CONTAGENS_ATIVA":
            return 4, "COEXISTENCIAS"
        if tipo in ("TRANSICAO_CONTAGENS_ATIVA", "CHANCE_DUPLA_ATIVA"):
            return 5, "TRANSICOES"
        if "ASSUNCAO" in tipo:
            return 6, "ASSUNCOES"
        if tipo.startswith("V3_ATIVADOR_") or tipo.startswith("HIERARQUIA_CONTAGEM_") or tipo == "FINALIZACAO_CONJUNTA_ATIVA":
            return 3, "CONTAGENS"
        return 2, "REGRAS_POSICIONAIS"

    @staticmethod
    def _resolver_nivel_sem_votacao(itens, ia_modelo=None):
        """
        Resolve autoridade dentro de UM nível. Nunca soma regras.
        Em conflito real sem diferença de autoridade, retorna conflito.
        """
        candidatos = []
        peso_manual = {"BAIXO": 0.25, "MEDIO": 0.50, "MÉDIO": 0.50, "ALTO": 0.75}
        for item in itens:
            direcao = item.get("direcao")
            if direcao not in ("VERMELHO", "PRETO"):
                continue
            tipo = str(item.get("tipo_regra", "REGRA"))
            autoridade_recente = 0.0
            estado_evolutivo = "SEM_MEDICAO"
            if ia_modelo is not None:
                evo = (
                    (getattr(ia_modelo, "matriz_evolutiva", {}) or {})
                    .get("regras", {})
                    .get(tipo, {})
                )
                autoridade_recente = float(evo.get("autoridade_atual", 0.0) or 0.0)
                estado_evolutivo = str(evo.get("estado_evolutivo", "SEM_MEDICAO"))
            base = peso_manual.get(str(item.get("peso", "MEDIO")).upper(), 0.50)
            if estado_evolutivo == "DEGRADACAO_CRITICA":
                continue
            autoridade = (base * 0.40) + (autoridade_recente * 0.60)
            penalidade_degradacao = {
                "DEGRADACAO_LEVE": 0.96,
                "DEGRADACAO_MODERADA": 0.88,
                "DEGRADACAO_FORTE": 0.72
            }.get(estado_evolutivo, 1.0)
            autoridade *= penalidade_degradacao
            candidatos.append({
                "direcao": direcao,
                "tipo_regra": tipo,
                "autoridade": autoridade,
                "autoridade_recente": autoridade_recente,
                "estado_evolutivo": estado_evolutivo,
                "item": item
            })

        if not candidatos:
            return None, "SEM_AUTORIDADE_VALIDA"

        candidatos.sort(key=lambda x: x["autoridade"], reverse=True)
        topo = candidatos[0]
        opostos = [c for c in candidatos if c["direcao"] != topo["direcao"]]
        if opostos:
            melhor_oposto = opostos[0]
            if abs(topo["autoridade"] - melhor_oposto["autoridade"]) < 0.08:
                return None, (
                    f"CONFLITO_HIERARQUICO_INTERNO: {topo['tipo_regra']} "
                    f"x {melhor_oposto['tipo_regra']}"
                )
        return topo, "RESOLVIDO_POR_AUTORIDADE"

    @staticmethod
    def arbitrar_sinal(no_call_ativo, motivo_nc, expectations, inclinacao_num, geometria_mercado,
                       previsao_ia, status_inversao, historico_regras,
                       modo_mercado="NEUTRO",
                       streak_atual=0, xadrez_len=0, xadrez_quebrou=False,
                       contexto_exaustao=False, sintese_evidencias=None,
                       probabilidade_markov=None, ia_modelo=None, entropia_shannon=0.0):
        # NÍVEL 1 — NO CALL é soberano.
        if no_call_ativo:
            return "NO CALL", motivo_nc, "SISTEMA_TRAVADO"

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
            chave_estado_rl = hash_chave(estado_rl) if estado_rl else None
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

        direcao_ia, confianca_ia, raciocinio_ia = previsao_ia
        if direcao_ia == "NO CALL":
            return "NO CALL", raciocinio_ia, "COLISAO_MOTORES"

        expectations = list(expectations or [])
        niveis = {2: [], 3: [], 4: [], 5: [], 6: []}
        nomes = {
            2: "REGRAS_POSICIONAIS",
            3: "CONTAGENS",
            4: "COEXISTENCIAS",
            5: "TRANSICOES",
            6: "ASSUNCOES"
        }
        for item in expectations:
            nivel, _ = JuizHierarquicoModificado._nivel_regra(item, ia_modelo)
            if nivel in niveis:
                niveis[nivel].append(item)

        sinal_preliminar = "NEUTRO"
        motivo_preliminar = ""
        regra_preliminar = ""

        # NÍVEIS 2..6 — primeira autoridade válida encerra a busca.
        for nivel in (2, 3, 4, 5, 6):
            if not niveis[nivel]:
                continue
            vencedor, status = JuizHierarquicoModificado._resolver_nivel_sem_votacao(
                niveis[nivel], ia_modelo
            )
            if vencedor is None:
                if "CONFLITO_HIERARQUICO_INTERNO" in status:
                    return "NO CALL", (
                        f"Conflito no nível {nivel} ({nomes[nivel]}). "
                        f"A Hierarquia Oficial recusou votação por quantidade. {status}"
                    ), "NO_CALL_CONFLITO_HIERARQUICO"
                continue
            sinal_preliminar = vencedor["direcao"]
            motivo_preliminar = (
                f"Hierarquia Oficial Nível {nivel} — {nomes[nivel]}: "
                f"{vencedor['tipo_regra']} | autoridade={vencedor['autoridade']:.4f} | "
                f"evolução={vencedor['estado_evolutivo']}"
            )
            regra_preliminar = vencedor["tipo_regra"]
            break

        # NÍVEL 7 — consequência futura viva.
        if sinal_preliminar == "NEUTRO" and ia_modelo is not None:
            try:
                consequencia = ia_modelo.construir_cadeia_causal_consequencia(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", []),
                    expectations
                )
                if (
                    consequencia.get("status") == "VIVA"
                    and consequencia.get("direcao") in ("VERMELHO", "PRETO")
                ):
                    sinal_preliminar = consequencia["direcao"]
                    motivo_preliminar = (
                        "Hierarquia Oficial Nível 7 — CONSEQUÊNCIA FUTURA: "
                        f"origem={consequencia.get('origem')} | "
                        f"sustentador={consequencia.get('sustentador')} | "
                        f"assunção={consequencia.get('assuncao')} | "
                        f"controlador={consequencia.get('controlador')} | "
                        f"estado={consequencia.get('estado_evolutivo')}"
                    )
                    regra_preliminar = "CONSEQUENCIA_FUTURA"
            except Exception:
                pass

        # NÍVEL 8 — padrões visuais/observacionais.
        if sinal_preliminar == "NEUTRO":
            if geometria_mercado == "CICLO_FECHADO_PVVP":
                sinal_preliminar = "VERMELHO"
                motivo_preliminar = "Hierarquia Oficial Nível 8 — padrão visual PVVP"
                regra_preliminar = "PADRAO_VISUAL_PVVP"
            elif geometria_mercado == "CICLO_FECHADO_VPPV":
                sinal_preliminar = "PRETO"
                motivo_preliminar = "Hierarquia Oficial Nível 8 — padrão visual VPPV"
                regra_preliminar = "PADRAO_VISUAL_VPPV"

        # ML é inteligência observacional: somente auxilia no nível visual quando
        # nenhum nível estrutural anterior produziu autoridade.
        if (
            sinal_preliminar == "NEUTRO"
            and direcao_ia in ("VERMELHO", "PRETO")
            and float(confianca_ia) >= 52.5
        ):
            sinal_preliminar = direcao_ia
            motivo_preliminar = (
                f"Hierarquia Oficial Nível 8 — IA observacional ({confianca_ia:.1f}%): "
                f"{raciocinio_ia}"
            )
            regra_preliminar = "IA_OBSERVACIONAL_NIVEL_8"

        if sinal_preliminar == "NEUTRO" and isinstance(probabilidade_markov, dict):
            try:
                markov_v = float(probabilidade_markov.get("V", 0.0))
                markov_p = float(probabilidade_markov.get("P", 0.0))
                if abs(markov_v - markov_p) >= 2.0:
                    sinal_preliminar = "VERMELHO" if markov_v > markov_p else "PRETO"
                    motivo_preliminar = (
                        "Hierarquia Oficial Nível 8 — Markov observacional "
                        f"V={markov_v:.2f}% P={markov_p:.2f}%"
                    )
                    regra_preliminar = "MARKOV_OBSERVACIONAL_NIVEL_8"
            except (TypeError, ValueError):
                pass

        if sinal_preliminar == "NEUTRO":
            return "NO CALL", (
                "Ausência de autoridade hierárquica válida. "
                "O sistema recusou votação/confluência."
            ), "FALLBACK_NO_CALL_HIERARQUIA"

        # MAIN 139 — correção direcional por oposição causal consolidada.
        if ia_modelo is not None and hasattr(ia_modelo, "obter_voto_contagens_consolidado"):
            try:
                autoridade_vencedora = float(
                    ia_modelo._autoridade_evolutiva_regra(regra_preliminar)
                    if hasattr(ia_modelo, "_autoridade_evolutiva_regra")
                    else 0.0
                )
                raiz_contagens = ia_modelo.obter_voto_contagens_consolidado(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", []),
                    expectations
                )
                direcao_raiz = raiz_contagens.get("direcao", "NEUTRO")
                peso_raiz_bruto = float(raiz_contagens.get("peso", 0.0) or 0.0)
                autoridade_raiz = max(0.0, min(1.0, peso_raiz_bruto / 18.0))
                correcao_direcional_causal = bool(
                    sinal_preliminar in ("VERMELHO", "PRETO")
                    and direcao_raiz in ("VERMELHO", "PRETO")
                    and direcao_raiz != sinal_preliminar
                    and autoridade_vencedora < 0.35
                    and autoridade_raiz >= 0.70
                    and (autoridade_raiz - autoridade_vencedora) >= 0.35
                )
                direcao_hierarquica_original = sinal_preliminar
                regra_hierarquica_original = regra_preliminar

                if correcao_direcional_causal:
                    sinal_preliminar = direcao_raiz
                    regra_preliminar = "CORRECAO_DIRECIONAL_RAIZ_CONTAGENS"
                    motivo_preliminar = (
                        "Correção direcional por oposição causal consolidada: "
                        f"{regra_hierarquica_original} venceu inicialmente a hierarquia "
                        f"em {direcao_hierarquica_original} com autoridade "
                        f"{autoridade_vencedora:.4f}, mas a raiz CONTAGENS consolidada "
                        f"aponta {direcao_raiz} com autoridade normalizada "
                        f"{autoridade_raiz:.4f} (peso bruto={peso_raiz_bruto:.4f}). "
                        "A oposição robusta corrigiu a direção antes das validações finais."
                    )

                ia_modelo._ultima_oposicao_causal_consolidada = {
                    "ativo": True,
                    "vetar": False,
                    "regra_vencedora": regra_hierarquica_original,
                    "direcao_vencedora": direcao_hierarquica_original,
                    "autoridade_vencedora": round(autoridade_vencedora, 4),
                    "raiz_oposta": "CONTAGENS",
                    "direcao_raiz": direcao_raiz,
                    "peso_raiz_bruto": round(peso_raiz_bruto, 4),
                    "autoridade_raiz": round(autoridade_raiz, 4),
                    "tipos_raiz": raiz_contagens.get("tipos", []),
                    "altera_direcao": correcao_direcional_causal,
                    "direcao_corrigida": sinal_preliminar if correcao_direcional_causal else None
                }
            except Exception as e:
                ia_modelo._ultima_oposicao_causal_consolidada = {
                    "ativo": False,
                    "vetar": False,
                    "status": "ERRO_PROTEGIDO",
                    "erro": f"{type(e).__name__}: {e}"
                }
                print(f"[OPOSIÇÃO CAUSAL CONSOLIDADA] Ignorada por segurança: {e}")

        # MAIN 115 — validação contextual da autoridade.
        if ia_modelo is not None and hasattr(ia_modelo, "validar_autoridade_hierarquica_contextual"):
            try:
                validacao_contextual = ia_modelo.validar_autoridade_hierarquica_contextual(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", []),
                    regra_preliminar, sinal_preliminar
                )
                if validacao_contextual.get("ativo"):
                    motivo_preliminar += (
                        f" | validação contextual={validacao_contextual.get('status')} "
                        f"[G0/G1 direção={validacao_contextual.get('taxa_direcao_g0_g1', 0):.2f}% "
                        f"vs contrária={validacao_contextual.get('taxa_contraria_g0_g1', 0):.2f}% "
                        f"| suporte={validacao_contextual.get('suporte', 0)}]"
                    )
                    if validacao_contextual.get("vetar"):
                        return (
                            "NO CALL",
                            "Validação Contextual da Autoridade VETOU a operação: "
                            f"{regra_preliminar} permaneceu {validacao_contextual.get('status')} no contexto atual; "
                            f"G0/G1 da direção={validacao_contextual.get('taxa_direcao_g0_g1', 0):.2f}% vs "
                            f"contrária={validacao_contextual.get('taxa_contraria_g0_g1', 0):.2f}%. "
                            f"Fontes contrárias robustas: {', '.join(validacao_contextual.get('fontes_contrarias_fortes', []))}.",
                            "NO_CALL_VALIDACAO_CONTEXTUAL_AUTORIDADE"
                        )
            except Exception as e:
                ia_modelo._ultima_validacao_autoridade_contextual = {
                    "ativo": False, "status": "ERRO_PROTEGIDO", "vetar": False,
                    "erro": f"{type(e).__name__}: {e}"
                }
                print(f"[VALIDAÇÃO CONTEXTUAL DA AUTORIDADE] Ignorada por segurança: {e}")

        # MAIN 129 — continuidade estrutural aprendida (morfologia).
        if ia_modelo is not None and hasattr(ia_modelo, "obter_voto_morfologia_estrutural"):
            try:
                voto_morfologia = ia_modelo.obter_voto_morfologia_estrutural(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", [])
                )
                oposicao_morfologica_robusta = bool(
                    voto_morfologia.get("ativo")
                    and voto_morfologia.get("direcao") in ("VERMELHO", "PRETO")
                    and voto_morfologia.get("direcao") != sinal_preliminar
                    and int(voto_morfologia.get("suporte", 0) or 0) >= 40
                    and float(voto_morfologia.get("margem", 0.0) or 0.0) >= 0.07
                    and float(voto_morfologia.get("peso", 0.0) or 0.0) >= 0.32
                )
                ia_modelo._ultima_oposicao_morfologia_estrutural = {
                    **dict(voto_morfologia),
                    "vetar": oposicao_morfologica_robusta,
                    "direcao_preliminar": sinal_preliminar,
                    "regra_preliminar": regra_preliminar,
                }
                if (
                    voto_morfologia.get("ativo")
                    and voto_morfologia.get("direcao") == sinal_preliminar
                    and int(voto_morfologia.get("suporte", 0) or 0) >= 30
                ):
                    morfo = voto_morfologia.get("morfologia", {}) or {}
                    motivo_preliminar += (
                        f" | morfologia={morfo.get('morfologia')} "
                        f"trajetória={morfo.get('trajetoria')} alinhada "
                        f"[margem={float(voto_morfologia.get('margem', 0.0))*100:.2f} p.p. "
                        f"| suporte={voto_morfologia.get('suporte')}]"
                    )
                if oposicao_morfologica_robusta:
                    morfo = voto_morfologia.get("morfologia", {}) or {}
                    return (
                        "NO CALL",
                        "Veto de oposição morfológica estrutural: "
                        f"forma {morfo.get('morfologia')} / {morfo.get('trajetoria')} "
                        f"possui memória contextual favorável a {voto_morfologia.get('direcao')} "
                        f"contra a direção oficial {sinal_preliminar}; "
                        f"margem={float(voto_morfologia.get('margem', 0.0))*100:.2f} p.p. | "
                        f"suporte={voto_morfologia.get('suporte')}. "
                        "A direção não foi invertida; a operação foi recusada por conflito estrutural.",
                        "NO_CALL_OPOSICAO_MORFOLOGIA_ESTRUTURAL"
                    )
            except Exception as e:
                ia_modelo._ultima_oposicao_morfologia_estrutural = {
                    "ativo": False, "vetar": False, "status": "ERRO_PROTEGIDO",
                    "erro": f"{type(e).__name__}: {e}",
                }
                print(f"[MORFOLOGIA ESTRUTURAL] Ignorada por segurança: {e}")

        # MAIN 126 — conflito causal consolidado da família STREAK.
        if ia_modelo is not None and hasattr(ia_modelo, "obter_voto_streak_consolidado"):
            try:
                voto_streak = ia_modelo.obter_voto_streak_consolidado(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", [])
                )
                oposicao_streak_robusta = bool(
                    voto_streak.get("ativo")
                    and int(voto_streak.get("streak", 0) or 0) >= 1
                    and voto_streak.get("direcao") in ("VERMELHO", "PRETO")
                    and voto_streak.get("direcao") != sinal_preliminar
                    and int(voto_streak.get("suporte", 0) or 0) >= 30
                    and float(voto_streak.get("margem", 0.0) or 0.0) >= 0.06
                    and float(voto_streak.get("peso", 0.0) or 0.0) >= 0.28
                )
                ia_modelo._ultima_oposicao_streak_consolidada = {
                    **dict(voto_streak),
                    "vetar": oposicao_streak_robusta,
                    "direcao_preliminar": sinal_preliminar,
                    "regra_preliminar": regra_preliminar,
                }
                if oposicao_streak_robusta:
                    return (
                        "NO CALL",
                        "Veto de oposição causal STREAK consolidada: "
                        f"trajetória {voto_streak.get('tipo_trajetoria')} "
                        f"{voto_streak.get('streak')}x {voto_streak.get('cor_streak')} "
                        f"possui memória contextual favorável a {voto_streak.get('direcao')} "
                        f"contra a direção oficial {sinal_preliminar}; "
                        f"margem={float(voto_streak.get('margem', 0.0))*100:.2f} p.p. | "
                        f"suporte={voto_streak.get('suporte')}. "
                        "A direção não foi invertida; a operação foi recusada por conflito causal.",
                        "NO_CALL_OPOSICAO_STREAK_CONSOLIDADA"
                    )
            except Exception as e:
                ia_modelo._ultima_oposicao_streak_consolidada = {
                    "ativo": False,
                    "vetar": False,
                    "status": "ERRO_PROTEGIDO",
                    "erro": f"{type(e).__name__}: {e}",
                }
                print(f"[STREAK CONSOLIDADA] Ignorada por segurança: {e}")

        # As duas proteções de G2+ permanecem intactas.
        if ia_modelo is not None:
            ia_modelo._ultima_direcao_pre_filtro_discriminativo = sinal_preliminar
            ia_modelo._ultima_avaliacao_filtro_discriminativo = None

        if ia_modelo and hasattr(ia_modelo, "avaliar_filtro_discriminativo_g0_g1"):
            try:
                filtro_discriminativo = ia_modelo.avaliar_filtro_discriminativo_g0_g1(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", []),
                    sinal_preliminar
                )
                ia_modelo._ultima_avaliacao_filtro_discriminativo = dict(filtro_discriminativo)
                if filtro_discriminativo.get("vetar"):
                    return (
                        "NO CALL",
                        f"Veto discriminativo G0/G1 x G2+: "
                        f"{filtro_discriminativo.get('risco_estimado', 0):.2f}% de risco estimado G2/FALHA "
                        f"com {filtro_discriminativo.get('contextos_risco_alto', 0)} evidências contextuais de risco "
                        f"({', '.join(filtro_discriminativo.get('fontes_risco_alto', []))}).",
                        "VETO_DISCRIMINATIVO_G0_G1"
                    )
            except Exception as e:
                print(f"[FILTRO DISCRIMINATIVO] Ignorado por segurança: {e}")

        if ia_modelo and hasattr(ia_modelo, "avaliar_risco_g2_mais"):
            try:
                risco_g2_mais = ia_modelo.avaliar_risco_g2_mais(
                    getattr(ia_modelo, "_ultima_janela_numeros", []),
                    getattr(ia_modelo, "_ultima_janela_cores", []),
                    sinal_preliminar
                )
                if risco_g2_mais.get("vetar"):
                    return (
                        "NO CALL",
                        f"Veto de risco G2+ [{risco_g2_mais.get('tipo_veto', 'ABSOLUTO')}]: "
                        f"{risco_g2_mais.get('risco_estimado', 0):.2f}% de não resolução até G1 "
                        f"em {risco_g2_mais.get('contextos_risco_alto', 0)} contextos históricos concordantes.",
                        "VETO_RISCO_G2_MAIS"
                    )
            except Exception as e:
                print(f"[RISCO G2+] Especialista ignorado por segurança: {e}")

        if ia_modelo is not None:
            ia_modelo.hierarquia_oficial_metricas = {
                "ativo": True,
                "versao": 1,
                "metodo": "AUTORIDADE_HIERARQUICA_SEM_VOTACAO",
                "ordem": [
                    "NO_CALL", "REGRAS_POSICIONAIS", "CONTAGENS", "COEXISTENCIAS",
                    "TRANSICOES", "ASSUNCOES", "CONSEQUENCIA_FUTURA",
                    "PADROES_VISUAIS", "SINAL"
                ],
                "ultima_regra_autoridade": regra_preliminar,
                "ultima_direcao": sinal_preliminar
            }

        return sinal_preliminar, motivo_preliminar, regra_preliminar
