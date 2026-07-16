from utils.helpers import hash_chave
from utils.math_engine import EngineMatematicoAvancado
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas

class QLearningMixin:
    """
    Mixin que isola a lógica de Reinforcement Learning e Q-Table
    da classe principal da IA.
    """
    
    def atualizar_q_learning(self, estado_str, acao, recompensa):
        if not hasattr(self, 'q_table'):
            self.q_table = {}
        chave_estado = hash_chave(estado_str)
        if chave_estado not in self.q_table:
            self.q_table[chave_estado] = {"APOSTAR": 0.0, "NO_CALL": 0.0}
            
        alpha = 0.1
        current_q = self.q_table[chave_estado][acao]
        self.q_table[chave_estado][acao] = current_q + alpha * (recompensa - current_q)

    def construir_estado_q_contextual(self, sub_num, sub_pol, analise=None, entropia_shannon=None, probabilidade_markov=None):
        nums = [int(n) for n in (sub_num or [])]
        pol = [str(c).upper() for c in (sub_pol or [])]

        if not nums or not pol:
            return "QCTX|SEM_JANELA"

        analise = analise if isinstance(analise, dict) else {}
        contexto_avancado = analise.get("contexto_avancado", {}) or {}
        modo_mercado = contexto_avancado.get("modo_mercado")
        
        if not modo_mercado:
            modo_mercado = AnalisadorContextoAvancado.detectar_modo_mercado(pol, False, self)

        geometria = analise.get("geometria")
        if not geometria:
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)

        expectativas = analise.get("regras_posicionais")
        if expectativas is None:
            try:
                expectativas = MotorContagensProjetivas.mapear_janela(nums[-12:], pol[-12:], geometria, self)
            except Exception:
                expectativas = []

        if entropia_shannon is None:
            entropia_shannon = analise.get("entropia")
        if entropia_shannon is None:
            entropia_shannon = EngineMatematicoAvancado.calcular_entropia_shannon(pol)

        if probabilidade_markov is None:
            probabilidade_markov = analise.get("probabilidade_markov")
        if not isinstance(probabilidade_markov, dict):
            probabilidade_markov = self.calcular_probabilidade_exata_markov(pol)

        mv = float(probabilidade_markov.get("V", 0.0))
        mp = float(probabilidade_markov.get("P", 0.0))
        diferenca_markov = abs(mv - mp)
        
        if diferenca_markov < 0.50:
            faixa_markov = "NEUTRO"
            direcao_markov = "N"
        elif mv > mp:
            faixa_markov = "FORTE" if diferenca_markov >= 2.0 else "LEVE"
            direcao_markov = "V"
        else:
            faixa_markov = "FORTE" if diferenca_markov >= 2.0 else "LEVE"
            direcao_markov = "P"

        tipos_regras = sorted({str(r.get("tipo_regra")) for r in (expectativas or []) if r.get("tipo_regra")})
        tipos_contagens = sorted({
            str(r.get("tipo_regra")) for r in (expectativas or [])
            if (str(r.get("familia", "")).upper() in ("CONTAGENS_PROJETIVAS", "DINAMICA_CONTAGENS", "HIERARQUIA_CONTAGENS")
                or "CONTAGEM" in str(r.get("tipo_regra", "")).upper()
                or str(r.get("tipo_regra", "")).upper().startswith("V3_ATIVADOR_"))
        })

        votos_v = sum(1 for r in (expectativas or []) if str(r.get("direcao", "")).upper() == "VERMELHO")
        votos_p = sum(1 for r in (expectativas or []) if str(r.get("direcao", "")).upper() == "PRETO")
        direcao_regras = "V" if votos_v > votos_p else ("P" if votos_p > votos_v else "N")
        conflito_markov_familias = (direcao_markov in ("V", "P") and direcao_regras in ("V", "P") and direcao_markov != direcao_regras)

        padrao_raiz = "".join(pol[-4:]) if len(pol) >= 4 else "".join(pol)
        ultimo = nums[-1]
        bigrama = "-".join(str(n) for n in nums[-2:]) if len(nums) >= 2 else str(ultimo)
        trigrama = "-".join(str(n) for n in nums[-3:]) if len(nums) >= 3 else bigrama

        leitura_padrao = {}
        if hasattr(self, "obter_voto_padrao_contextual"):
            try:
                leitura_padrao = self.obter_voto_padrao_contextual(nums, pol) or {}
            except Exception:
                pass
        
        direcao_padrao = str(leitura_padrao.get("direcao", "NEUTRO")).upper()
        peso_padrao = float(leitura_padrao.get("peso", 0.0) or 0.0)
        faixa_padrao = "ALTO" if peso_padrao >= 3.0 else ("MEDIO" if peso_padrao >= 1.5 else ("BAIXO" if peso_padrao > 0 else "SEM_DADO"))

        leitura_regra_ctx = {}
        if hasattr(self, "obter_voto_regra_contextual"):
            try:
                leitura_regra_ctx = self.obter_voto_regra_contextual(nums, pol) or {}
            except Exception:
                pass
        
        direcao_regra_ctx = str(leitura_regra_ctx.get("direcao", "NEUTRO")).upper()
        peso_regra_ctx = float(leitura_regra_ctx.get("peso", 0.0) or 0.0)
        faixa_regra_ctx = "ALTO" if peso_regra_ctx >= 2.0 else ("MEDIO" if peso_regra_ctx >= 1.0 else ("BAIXO" if peso_regra_ctx > 0 else "SEM_DADO"))

        regras_estado = ",".join(tipos_regras) if tipos_regras else "SEM_REGRA"
        contagens_estado = ",".join(tipos_contagens) if tipos_contagens else "SEM_CONTAGEM"
        regime_hmm = self._obter_regime_hmm_contextual(pol) if hasattr(self, '_obter_regime_hmm_contextual') else "HMM_INDISPONIVEL"

        return (
            f"QCTX|REG={modo_mercado}|HMM={regime_hmm}|ENT={round(float(entropia_shannon), 1)}"
            f"|PAD={padrao_raiz}|PADCTX={direcao_padrao}:{faixa_padrao}"
            f"|REGRACTX={direcao_regra_ctx}:{faixa_regra_ctx}|ULT={ultimo}"
            f"|BI={bigrama}|TRI={trigrama}|GEO={geometria}"
            f"|REGRAS={regras_estado}|CONTAGENS={contagens_estado}"
            f"|MK={direcao_markov}:{faixa_markov}|CONFLITO_MK_FAM={int(conflito_markov_familias)}"
        )

    def treinar_q_learning_contextual(self, dados, multiplicador_peso=1, origem="BASE_LONGA"):
        # Importação interna para evitar dependências circulares pesadas na inicialização
        from core.motor_analise import MotorAnalise
        from core.juiz_hierarquico import JuizHierarquicoModificado
        
        if not dados or len(dados) < 13:
            return {"ativo": False, "origem": origem, "janelas_processadas": 0, "peso": int(max(1, multiplicador_peso))}

        nums = [int(d.get("numero")) for d in dados]
        pol = [str(d.get("cor", "B")).upper() for d in dados]
        total = len(nums)
        idx = 0
        janelas_processadas = 0
        atualizacoes_q = 0
        peso = int(max(1, multiplicador_peso))

        while idx + 12 < total:
            sub_num = nums[idx:idx + 12]
            sub_pol = pol[idx:idx + 12]

            analise = MotorAnalise.analisar_janela(sub_num, sub_pol, self, eh_sinal_real=True)

            if analise["no_call"]["ativo"]:
                sinal = "NO CALL"
            else:
                geometria = analise["geometria"]
                expectativas = analise["regras_posicionais"]
                direcao_ia = analise["ia"]["direcao"]
                conf_ia = analise["ia"]["confianca"]
                raciocinio_ia = analise["ia"]["raciocinio"]
                contexto_reversao = analise["contexto_reversao"]
                modo_mercado = analise["contexto_avancado"].get("modo_mercado", "NEUTRO")

                sinal, _, _ = JuizHierarquicoModificado.arbitrar_sinal(
                    no_call_ativo=False, motivo_nc="", expectations=expectativas,
                    inclinacao_num=None, geometria_mercado=geometria,
                    previsao_ia=(direcao_ia, conf_ia, raciocinio_ia),
                    status_inversao=None, historico_regras=getattr(self, 'historico_regras', {}),
                    modo_mercado=modo_mercado, streak_atual=contexto_reversao["streak"],
                    xadrez_len=contexto_reversao["xadrez_len"], xadrez_quebrou=contexto_reversao["xadrez_quebrou"],
                    contexto_exaustao=contexto_reversao["exaustao"],
                    probabilidade_markov=analise.get("probabilidade_markov"),
                    ia_modelo=self, entropia_shannon=analise.get("entropia", 0.0)
                )

            correcoes = pol[idx + 12:idx + 15]
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

            estado_q = self.construir_estado_q_contextual(
                sub_num, sub_pol, analise=analise,
                entropia_shannon=analise.get("entropia", 0.0),
                probabilidade_markov=analise.get("probabilidade_markov")
            )
            acao_q = "APOSTAR" if sinal != "NO CALL" else "NO_CALL"

            recompensa = 1.0 if classificacao in ("G0", "G1") else (-0.5 if classificacao == "G2" else (-2.0 if classificacao == "FALHA" else 0.0))

            for _ in range(peso):
                self.atualizar_q_learning(estado_q, acao_q, recompensa)
                atualizacoes_q += 1

            janelas_processadas += 1
            idx += 12 + salto

        metricas = {
            "ativo": True, "metodo": "Q_STATE_CONTEXTUAL_CRONOLOGICO",
            "origem": str(origem), "janelas_processadas": janelas_processadas,
            "atualizacoes_q": atualizacoes_q, "peso": peso, "estados_q_total": len(self.q_table),
        }
        self.q_learning_contextual_metricas = metricas
        return metricas
