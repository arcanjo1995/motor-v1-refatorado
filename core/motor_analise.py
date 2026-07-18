# core/motor_analise.py
from rules.no_call import MotorNoCall
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from utils.math_engine import EngineMatematicoAvancado

class MotorAnalise:
    @staticmethod
    def analisar_janela(sub_num, sub_pol, ia_modelo, base_longa=None, eh_sinal_real=False):
        resultado = {
            "camadas": [],
            "no_call": None,
            "geometria": None,
            "regras_posicionais": [],
            "contexto_avancado": {},
            "ia": {},
            "contexto_reversao": {},
            "controlador_retardador": {}
        }

        # Contexto efêmero para especialistas adaptativos; não entra na recência.
        if ia_modelo is not None:
            ia_modelo._ultima_janela_cores = list(sub_pol)
            ia_modelo._ultima_janela_numeros = list(sub_num)
        
        nc_ativo, motivo_nc = MotorNoCall.checar_no_call(sub_num, sub_pol)
        
        if eh_sinal_real and not nc_ativo and len(sub_num) >= 2:
            risco_ativo, motivo_risco = MotorNoCall.checar_risco_preditivo_g0(sub_num, ia_modelo)
            if risco_ativo:
                nc_ativo = True
                motivo_nc = motivo_risco
        
        entropia = 0.0
        if eh_sinal_real:
            entropia = EngineMatematicoAvancado.calcular_entropia_shannon(sub_pol)
            resultado["entropia"] = entropia
            if entropia > 1.52 and not nc_ativo:
                nc_ativo = True
                motivo_nc = f"Bloqueio de Segurança HMM: Entropia de Shannon em Nível Crítico de Caos ({entropia} Bits). O Mercado está Aleatório."
            
        resultado["no_call"] = {"ativo": nc_ativo, "motivo": motivo_nc}
        resultado["camadas"].append({
            "camada": 1, "nome": "Segurança e Proteção de Saldo (NO CALL)",
            "resultado": f"Ativo={nc_ativo}", "detalhe": motivo_nc,
            "impacto": "BLOQUEIO" if nc_ativo else "APROVADO"
        })
        
        if nc_ativo:
            resultado["probabilidade_markov"] = ia_modelo.calcular_probabilidade_exata_markov(sub_pol)
            return resultado
            
        modo_mercado = AnalisadorContextoAvancado.detectar_modo_mercado(sub_pol, eh_sinal_real, ia_modelo)
        geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
        
        resultado["geometria"] = geometria
        resultado["camadas"].append({
            "camada": 2, "nome": "Geometria de Mercado",
            "resultado": geometria, "detalhe": "Padrão geométrico detectado no gráfico",
            "impacto": "FORTE" if geometria in ["CICLO_FECHADO_VPPV", "CICLO_FECHADO_PVVP"] else "NEUTRO"
        })
        
        expectativas = MotorContagensProjetivas.mapear_janela(sub_num, sub_pol, geometria, ia_modelo)
        resultado["regras_posicionais"] = expectativas
        resultado["camadas"].append({
            "camada": 3, "nome": "Regras Posicionais Ativadas (Volumes 2 e 3)",
            "resultado": f"{len(expectativas)} regras ativas",
            "detalhe": [e["tipo_regra"] for e in expectativas] if expectativas else "Nenhuma detecção volumétrica",
            "impacto": "ALTO" if expectativas else "BAIXO"
        })
        
        resultado["contexto_avancado"] = {"modo_mercado": modo_mercado}
        resultado["camadas"].append({
            "camada": 4, "nome": "Contexto Avançado de Regime (HMM quando disponível)",
            "resultado": f"Modo: {modo_mercado}", "detalhe": "Detecção matemática de fase do algoritmo", "impacto": "MÉDIO"
        })
        
        contexto_para_ia = {
            "geometria": geometria,
            "regras_posicionais": expectativas,
            "controlador_retardador": {},
            "contexto_avancado": resultado["contexto_avancado"]
        }
        
        direcao_ia, conf_ia, raciocinio_ia = ia_modelo.predizer_proxima_casa(sub_num, sub_pol, contexto_para_ia)
        resultado["ia"] = {
            "direcao": direcao_ia,
            "confianca": conf_ia,
            "raciocinio": raciocinio_ia
        }
        resultado["camadas"].append({
            "camada": 5, "nome": "IA Preditiva Híbrida (Gradient Boosting + MLP + Memórias Contextuais)",
            "resultado": f"{direcao_ia} ({conf_ia}%)",
            "detalhe": raciocinio_ia,
            "impacto": "ALTO" if conf_ia >= 52 else "MÉDIO"
        })
        
        probabilidade_markov = ia_modelo.calcular_probabilidade_exata_markov(sub_pol)
        resultado["probabilidade_markov"] = probabilidade_markov
        
        if eh_sinal_real:
            resultado["camadas"].append({
                "camada": 5.5, "nome": "Validação Determinística (Cadeia de Markov Exata)",
                "resultado": f"V: {probabilidade_markov['V']}% | P: {probabilidade_markov['P']}%",
                "detalhe": "Cálculo matemático fechado do arquivo base (sem simulações aleatórias)",
                "impacto": "ALTO"
            })
            
        streak, xadrez_len, xadrez_quebrou, exaustao = MotorAnalise._calcular_contexto_reversao(sub_pol)
        resultado["contexto_reversao"] = {
            "streak": streak, "xadrez_len": xadrez_len,
            "xadrez_quebrou": xadrez_quebrou, "exaustao": exaustao
        }
        resultado["camadas"].append({
            "camada": 6, "nome": "Contexto de Reversão e Exaustão",
            "resultado": f"Streak atual: {streak}x | Tamanho do Xadrez: {xadrez_len}",
            "detalhe": f"Sinal de Exaustão de Padrão: {exaustao}",
            "impacto": "ALTO" if exaustao else "BAIXO"
        })
        
        consequencia_futura = ia_modelo.construir_cadeia_causal_consequencia(
            sub_num, sub_pol, expectativas
        )
        resultado["consequencia_futura"] = consequencia_futura
        contexto_para_ia["consequencia_futura"] = consequencia_futura

        ctrl_ret = MotorAnalise._detectar_controlador_retardador(
            sub_num, sub_pol, expectativas, geometria, modo_mercado,
            eh_sinal_real, ia_modelo=ia_modelo,
            consequencia_futura=consequencia_futura
        )
        resultado["controlador_retardador"] = ctrl_ret
        resultado["camadas"].append({
            "camada": 7, "nome": "Balança de Forças Operacionais (Controlador vs Retardador)",
            "resultado": ctrl_ret["dominancia"],
            "detalhe": f"Forças de Controle: {ctrl_ret['controladores']} | Forças de Retardo: {ctrl_ret['retardadores']}",
            "impacto": "ALTO"
        })

        # =========================================================
        # <-- RADAR: Obter influência do Radar Numérico e adicionar ao resultado
        # =========================================================
        if ia_modelo is not None and hasattr(ia_modelo, 'obter_influencia_radar'):
            try:
                influencia_radar = ia_modelo.obter_influencia_radar(
                    sub_num, sub_pol, analise_contexto={
                        "geometria": geometria,
                        "regras_posicionais": expectativas,
                        "contexto_avancado": {"modo_mercado": modo_mercado}
                    }
                )
                resultado["influencia_radar"] = influencia_radar
                resultado["camadas"].append({
                    "camada": 7.5,
                    "nome": "Radar Numérico",
                    "resultado": f"Número dominante: {influencia_radar.get('numero_dominante')} (consenso {influencia_radar.get('consenso', 0)*100:.1f}%)",
                    "detalhe": f"Fator de influência: {influencia_radar.get('fator_influencia', 0):.3f} | Confiabilidade: {influencia_radar.get('confiabilidade', 0)*100:.1f}%",
                    "impacto": "ALTO" if influencia_radar.get('fator_influencia', 0) >= 0.15 else "MÉDIO" if influencia_radar.get('fator_influencia', 0) >= 0.08 else "BAIXO"
                })
            except Exception as e:
                resultado["influencia_radar"] = None
                resultado["camadas"].append({
                    "camada": 7.5,
                    "nome": "Radar Numérico",
                    "resultado": "INDISPONÍVEL",
                    "detalhe": f"Erro ao obter influência do Radar: {type(e).__name__}",
                    "impacto": "NEUTRO"
                })
        else:
            resultado["influencia_radar"] = None

        return resultado

    @staticmethod
    def _calcular_contexto_reversao(sub_pol):
        streak = 0
        if sub_pol:
            ultima_cor = sub_pol[-1]
            for c in reversed(sub_pol):
                if c == ultima_cor:
                    streak += 1
                else:
                    break
        
        xadrez_len = 0
        for i in range(len(sub_pol)-1, 0, -1):
            if sub_pol[i] != sub_pol[i-1]:
                xadrez_len += 1
            else:
                break
        xadrez_quebrou = (sub_pol[-1] == sub_pol[-2]) if len(sub_pol) >= 2 else False
        exaustao = (streak >= 4) or (xadrez_len >= 5 and xadrez_quebrou)
        return streak, xadrez_len, xadrez_quebrou, exaustao

    @staticmethod
    def _detectar_controlador_retardador(
        sub_num, sub_pol, expectativas, geometria, modo_mercado,
        eh_sinal_real=False, ia_modelo=None, consequencia_futura=None
    ):
        """
        Controlador = maior autoridade estrutural válida, não quantidade de rótulos.
        Retardador altera tempo/risco e não cria direção nem invalida sozinho.
        """
        expectativas = list(expectativas or [])
        consequencia_futura = consequencia_futura or {}
        candidatos = []
        for regra in expectativas:
            direcao = regra.get("direcao")
            if direcao not in ("VERMELHO", "PRETO"):
                continue
            if ia_modelo is not None:
                nivel, nome_nivel = ia_modelo._nivel_hierarquico_regra(regra)
                autoridade = ia_modelo._autoridade_evolutiva_regra(regra.get("tipo_regra", ""))
            else:
                nivel, nome_nivel = (8, "PADROES_VISUAIS")
                autoridade = 0.0
            candidatos.append({
                "fonte": regra.get("tipo_regra", "REGRA"),
                "direcao": direcao,
                "nivel": nivel,
                "nivel_nome": nome_nivel,
                "autoridade_atual": autoridade
            })

        candidatos.sort(key=lambda c: (c["nivel"], -c["autoridade_atual"]))
        dominante = candidatos[0] if candidatos else None
        secundarios = candidatos[1:3] if len(candidatos) > 1 else []

        retardadores = []
        if "RECOLHIMENTO" in str(modo_mercado) or "CAOS" in str(modo_mercado) or modo_mercado == "CHUVA":
            retardadores.append({
                "fonte": "REGIME_ALTA_ALTERNANCIA",
                "efeito": "ALTERACAO_DE_TEMPO",
                "autoridade": "RISCO"
            })
        if geometria in ("SATURAÇÃO ESTRUTURAL (V)", "SATURAÇÃO ESTRUTURAL (P)"):
            retardadores.append({
                "fonte": "SATURACAO_ESTRUTURAL",
                "efeito": "ALERTA_ESTRUTURAL_SEM_DIRECAO",
                "autoridade": "RISCO"
            })

        # MAIN 126 — a família STREAK passa a ter voz causal consolidada na
        # balança. Ela não cria direção e não veta sozinha nesta camada.
        if (
            ia_modelo is not None
            and dominante is not None
            and hasattr(ia_modelo, "obter_voto_streak_consolidado")
        ):
            try:
                voto_streak = ia_modelo.obter_voto_streak_consolidado(sub_num, sub_pol)
                if (
                    voto_streak.get("ativo")
                    and voto_streak.get("direcao") in ("VERMELHO", "PRETO")
                    and voto_streak.get("direcao") != dominante.get("direcao")
                    and float(voto_streak.get("peso", 0.0) or 0.0) >= 0.20
                ):
                    retardadores.append({
                        "fonte": "STREAK_CONSOLIDADA",
                        "efeito": "OPOSICAO_CAUSAL_CONTEXTUAL",
                        "direcao": voto_streak.get("direcao"),
                        "streak": voto_streak.get("streak"),
                        "cor_streak": voto_streak.get("cor_streak"),
                        "estagio": voto_streak.get("estagio"),
                        "tipo_trajetoria": voto_streak.get("tipo_trajetoria"),
                        "autoridade": voto_streak.get("peso"),
                        "suporte": voto_streak.get("suporte"),
                        "margem": voto_streak.get("margem"),
                    })
            except Exception as e:
                print(f"[STREAK CONSOLIDADA - BALANÇA] Ignorada por segurança: {e}")

        estado_ctrl = "SEM_CONTROLADOR"
        if dominante:
            estado_evolutivo = "SEM_MEDICAO"
            if ia_modelo is not None:
                estado_evolutivo = (
                    (getattr(ia_modelo, "matriz_evolutiva", {}) or {})
                    .get("regras", {})
                    .get(dominante["fonte"], {})
                    .get("estado_evolutivo", "SEM_MEDICAO")
                )
            if str(estado_evolutivo).startswith("DEGRADACAO"):
                estado_ctrl = "CONTROLADOR_DEGRADADO"
            else:
                estado_ctrl = "CONTROLADOR_VALIDADO"

        return {
            "controladores": candidatos,
            "retardadores": retardadores,
            "dominancia": estado_ctrl,
            "controlador_dominante": dominante,
            "controlador_secundario": secundarios[0] if secundarios else None,
            "controlador_escondido": consequencia_futura.get("controlador") if consequencia_futura.get("status") == "VIVA" else None,
            "consequencia_futura": consequencia_futura,
            "metodo": "MAIOR_AUTORIDADE_ESTRUTURAL_HIERARQUICA"
        }
