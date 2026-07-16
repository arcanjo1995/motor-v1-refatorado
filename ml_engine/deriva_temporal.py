from collections import defaultdict
from rules.analisador import AnalisadorContextoAvancado

class DerivaTemporalMixin:
    """
    Mixin para Deriva Comportamental e Validação Hierárquica Contextual.
    """
    def mapear_deriva_comportamental_numeros(self):
        """
        MAIN 119 — diagnóstico multi-horizonte dentro da recência oficial.
        Não cria direção, não inverte sinal e não altera o peso 6.
        """
        longo = list(getattr(self, "dados_longo", []) or [])
        recente = list(getattr(self, "dados_recencia", []) or [])[-200:]
        horizontes = (200, 100, 50, 25)

        def validar_autoridade_hierarquica_contextual(self, sub_num, sub_pol, regra_id, direcao):
        """
        MAIN 115 — prova contextual da autoridade hierárquica.

        Não cria regra, não troca direção e não soma famílias. Usa a cartografia
        cronológica já treinada para verificar se a REGRA VENCEDORA continua
        saudável no contexto atual (número final, bigrama, trigrama, padrão,
        regime, geometria e Markov). Contexto forte contrário pode somente
        degradar a autoridade ou converter a operação em NO CALL.
        """
        neutro = {
            "ativo": False, "status": "SEM_VALIDACAO", "vetar": False,
            "regra": str(regra_id or ""), "direcao": direcao,
            "autoridade_contextual": None, "taxa_direcao_g0_g1": None,
            "taxa_contraria_g0_g1": None, "suporte": 0, "evidencias": [],
            "componentes": {}
        }
        if (
            len(sub_num or []) < 12 or len(sub_pol or []) < 12
            or not regra_id or direcao not in ("VERMELHO", "PRETO")
        ):
            self._ultima_validacao_autoridade_contextual = neutro
            return neutro

        mapa = getattr(self, "cartografia_regras_contextual", {}) or {}
        if not mapa:
            self._ultima_validacao_autoridade_contextual = neutro
            return neutro
        mapa_recente = self._cartografia_recente_regra_atual(regra_id)
        try:
            matriz_deriva = self.mapear_deriva_comportamental_numeros()
        except Exception:
            matriz_deriva = {"numeros": {}}

        evento = [{
            "tipo": str(regra_id), "direcao": direcao,
            "familia": "AUTORIDADE_HIERARQUICA"
        }]
        chaves = self._chaves_cartografia_contextual_eventos(
            list(sub_num)[-12:], list(sub_pol)[-12:], evento
        )
        prefixo = f"REGRA_CTX|E={reg    # 3. _chaves_deriva_temporal_cenario
    def _mapear_deriva_temporal_basica(self, dados, chaves_alvo):
        """Conta os cenários relevantes separando resolução em G0 e até G1."""
        resultado = {
            chave: {
                "total": 0, "V_g0": 0, "V_g1": 0,
                "P_g0": 0, "P_g1": 0, "B": 0
            }
            for chave in chaves_alvo
        }
        if not dados or len(dados) < 14 or not chaves_alvo:
            return resultado

        alvo = set(chaves_alvo)
        numeros = [int(d["numero"]) for d in dados]
        cores = [str(d["cor"]).upper() for d in dados]

        for i in range(11, len(dados) - 2):
            chaves = self._chaves_deriva_temporal_cenario(
                numeros[max(0, i - 11):i + 1],
                cores[max(0, i - 11):i + 1]
            )
            correspondentes = alvo.intersection(chaves)
            if not correspondentes:
                continue

            c0 = cores[i + 1]
            c1 = cores[i + 2]
            for chave in correspondentes:
                st = resultado[chave]
                st["total"] += 1
                if c0 == "B":
                    st["B"] += 1
                if c0 in ("V", "B"):
                    st["V_g0"] += 1
                elif c1 in ("V", "B"):
                    st["V_g1"] += 1
                if c0 in ("P", "B"):
                    st["P_g0"] += 1
                elif c1 in ("P", "B"):
                    st["P_g1"] += 1
        return resultado

    def _mapear_deriva_temporal_regras(self, dados, tipos_alvo):
        """
        Mede na RECÊNCIA cada regra atualmente ativa, ocorrência por ocorrência,
        separando resolução em G0 e até G1. O detector oficial não é alterado.
        """
        resultado = {
            f"REGRA|{tipo}": {
                "total": 0, "V_g0": 0, "V_g1": 0,
                "P_g0": 0, "P_g1": 0, "B": 0
            }
            for tipo in tipos_alvo
        }
        if not dados or len(dados) < 14 or not tipos_alvo:
            return resultado

        alvo = set(tipos_alvo)
        numeros = [int(d["numero"]) for d in dados]
        cores = [str(d["cor"]).upper() for d in dados]

        for i in range(11, len(dados) - 2):
            sub_num = numeros[i - 11:i + 1]
            sub_pol = cores[i - 11:i + 1]
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
            regras = MotorContagensProjetivas.mapear_janela(
                sub_num, sub_pol, geometria, None
            )
            tipos_encontrados = {
                str(regra.get("tipo_regra", ""))
                for regra in regras
                if str(regra.get("tipo_regra", "")) in alvo
            }
            c0 = cores[i + 1]
            c1 = cores[i + 2]
            for tipo in tipos_encontrados:
                st = resultado[f"REGRA|{tipo}"]
                st["total"] += 1
                if c0 == "B":
                    st["B"] += 1
                if c0 in ("V", "B"):
                    st["V_g0"] += 1
                elif c1 in ("V", "B"):
                    st["V_g1"] += 1
                if c0 in ("P", "B"):
                    st["P_g0"] += 1
                elif c1 in ("P", "B"):
                    st["P_g1"] += 1
        return resultado

    def _resumir_trajetoria_temporal_cenario(self, pontos, historico_v_g0, historico_p_g0):
        """
        MAIN 113 — resume a evolução cronológica de um cenário na RECÊNCIA.
        Não cria regra nem altera detector: apenas classifica a trajetória já observada.
        """
        validos = [p for p in pontos if int(p.get("suporte", 0)) >= 2]
        if len(validos) < 2:
            return {
                "estado": "TRAJETORIA_INSUFICIENTE",
                "direcao_atual": "NEUTRO",
                "direcao_historica": "V" if historico_v_g0 >= historico_p_g0 else "P",
                "persistencia_blocos": 0,
                "velocidade": "INDEFINIDA",
                "confianca_longitudinal": 0.0,
                "fator_longitudinal": 0.75,
                "pontos": pontos,
            }

        taxas_v = [float(p.get("V_g0_taxa", 0.5)) for p in validos]
        direcoes = ["V" if taxa >= 0.5 else "P" for taxa in taxas_v]
        direcao_hist = "V" if historico_v_g0 >= historico_p_g0 else "P"
        direcao_atual = direcoes[-1]

        persistencia = 1
        for direcao in reversed(direcoes[:-1]):
            if direcao != direcao_atual:
                break
            persistencia += 1

        n = len(taxas_v)
        xs = list(range(n))
        media_x = sum(xs) / n
        media_y = sum(taxas_v) / n
        denominador = sum((x - media_x) ** 2 for x in xs)
        inclinacao = (
            sum((x - media_x) * (y - media_y) for x, y in zip(xs, taxas_v))
            / denominador
            if denominador > 0 else 0.0
        )

        if inclinacao >= 0.04:
            velocidade = "ACELERANDO_V"
        elif inclinacao <= -0.04:
            velocidade = "ACELERANDO_P"
        else:
            velocidade = "ESTAVEL"

        cruzou = any(d != direcao_hist for d in direcoes)
        voltou = cruzou and direcao_atual == direcao_hist
        if voltou:
            estado = "RETORNO_AO_HISTORICO"
        elif direcao_atual != direcao_hist and persistencia >= 3:
            estado = "INVERSAO_PERSISTENTE"
        elif direcao_atual != direcao_hist:
            estado = "MIGRACAO_EM_OBSERVACAO"
        elif persistencia >= 3 and (
            (direcao_atual == "V" and inclinacao >= 0.04)
            or (direcao_atual == "P" and inclinacao <= -0.04)
        ):
            estado = "ACELERACAO_NA_DIRECAO_ATUAL"
        else:
            estado = "ESTAVEL_COM_HISTORICO"

        suporte_total = sum(int(p.get("suporte", 0)) for p in validos)
        confianca_suporte = min(1.0, suporte_total / 30.0)
        confianca_persistencia = min(1.0, persistencia / 4.0)
        confianca_longitudinal = (
            0.60 * confianca_suporte + 0.40 * confianca_persistencia
        )

        if estado == "INVERSAO_PERSISTENTE":
            fator = 1.0 + (0.35 * confianca_longitudinal)
        elif estado == "ACELERACAO_NA_DIRECAO_ATUAL":
            fator = 1.0 + (0.20 * confianca_longitudinal)
        elif estado == "MIGRACAO_EM_OBSERVACAO":
            fator = 0.75 + (0.20 * confianca_longitudinal)
        elif estado == "RETORNO_AO_HISTORICO":
            fator = 0.65 + (0.20 * confianca_longitudinal)
        else:
            fator = 0.90 + (0.10 * confianca_longitudinal)

        return {
            "estado": estado,
            "direcao_atual": direcao_atual,
            "direcao_historica": direcao_hist,
            "persistencia_blocos": persistencia,
            "velocidade": velocidade,
            "inclinacao_V_por_bloco": round(inclinacao, 6),
            "confianca_longitudinal": round(confianca_longitudinal, 6),
            "fator_longitudinal": round(fator, 6),
            "pontos": pontos,
        }

    def _mapear_trajetoria_deriva_temporal(
        self, dados_rec, chaves_basicas, tipos_regras, longo
    ):
        """
        MAIN 113 — cartografia temporal longitudinal caso a caso.
        Divide a RECÊNCIA em blocos cronológicos e acompanha a trajetória de cada
        número, bigrama, trigrama, padrão, streak, geometria, Markov, regime e
        regra ativa atual: ANTES -> MUDOU -> ACELEROU -> ESTABILIZOU -> VOLTOU.
        """
        if not dados_rec or len(dados_rec) < 40:
            return {}

        quantidade_blocos = min(6, max(2, len(dados_rec) // 40))
        tamanho_bloco = max(20, len(dados_rec) // quantidade_blocos)
        blocos = []
        inicio = 0
        while inicio < len(dados_rec):
            fim = min(len(dados_rec), inicio + tamanho_bloco)
            bloco = dados_rec[inicio:fim]
            if len(bloco) >= 14:
                blocos.append((inicio, fim, bloco))
            inicio = fim

        if len(blocos) < 2:
            return {}

        trajetorias = {
            chave: [] for chave in (
                list(chaves_basicas) + [f"REGRA|{tipo}" for tipo in tipos_regras]
            )
        }

        for indice_bloco, (inicio, fim, bloco) in enumerate(blocos):
            mapa_basico = self._mapear_deriva_temporal_basica(
                bloco, chaves_basicas
            )
            mapa_regras = self._mapear_deriva_temporal_regras(
                bloco, tipos_regras
            )

            for chave in trajetorias:
                st = (
                    mapa_regras.get(chave, {})
                    if chave.startswith("REGRA|")
                    else mapa_basico.get(chave, {})
                )
                total = int(st.get("total", 0))
                validos_g0 = int(st.get("V_g0", 0) + st.get("P_g0", 0))
                if total <= 0 or validos_g0 <= 0:
                    continue
                trajetorias[chave].append({
                    "bloco": indice_bloco + 1,
                    "inicio_recencia": inicio,
                    "fim_recencia": fim,
                    "suporte": total,
                    "V_g0_taxa": float(st.get("V_g0", 0)) / validos_g0,
                    "P_g0_taxa": float(st.get("P_g0", 0)) / validos_g0,
                    "V_ate_g1_taxa": float(
                        st.get("V_g0", 0) + st.get("V_g1", 0)
                    ) / max(float(total), 1.0),
                    "P_ate_g1_taxa": float(
                        st.get("P_g0", 0) + st.get("P_g1", 0)
                    ) / max(float(total), 1.0),
                })

        resumo = {}
        for chave, pontos in trajetorias.items():
            hist = longo.get(chave, {})
            hist_validos = float(hist.get("V_g0", 0) + hist.get("P_g0", 0))
            if hist_validos <= 0:
                continue
            resumo[chave] = self._resumir_trajetoria_temporal_cenario(
                pontos,
                float(hist.get("V_g0", 0)) / hist_validos,
                float(hist.get("P_g0", 0)) / hist_validos,
            )
        return resumo

    def obter_ajuste_deriva_temporal(self, sub_num, sub_pol, analise_contexto=None):
        """
        Compara LONGO PRAZO x RECÊNCIA por cenário e calibra a influência recente
        pela competência observada de cada família, separando G0 de resolução até G1.
        A camada continua gerando uma única força temporal consolidada.
        """
        dados_rec = list(getattr(self, "dados_recencia", []) or [])
        if len(dados_rec) < 20:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "motivo": "RECENCIA_INSUFICIENTE", "leituras": [],
                "competencia_familias": {}
            }

        dados_longo = list(getattr(self, "dados_longo", []) or [])
        if len(dados_longo) >= len(dados_rec):
            cauda = dados_longo[-len(dados_rec):]
            if [int(d["numero"]) for d in cauda] == [int(d["numero"]) for d in dados_rec]:
                dados_longo = dados_longo[:-len(dados_rec)]

        chaves_basicas = self._chaves_deriva_temporal_cenario(
            sub_num[-12:], sub_pol[-12:]
        )
        longo = self._mapear_deriva_temporal_basica(dados_longo, chaves_basicas)
        recente = self._mapear_deriva_temporal_basica(dados_rec, chaves_basicas)

        regras_ativas = (analise_contexto or {}).get("regras_posicionais", [])
        tipos_regras = sorted({
            str(regra.get("tipo_regra", ""))
            for regra in regras_ativas
            if str(regra.get("tipo_regra", ""))
        })
        recente_regras = self._mapear_deriva_temporal_regras(dados_rec, tipos_regras)

        # MAIN 113 — acompanha a trajetória cronológica da mudança de comportamento
        # dos cenários atualmente relevantes, sem alterar detectores ou regras.
        trajetoria_temporal = self._mapear_trajetoria_deriva_temporal(
            dados_rec, chaves_basicas, tipos_regras, longo
        )

        # Competência temporal é validada cronologicamente dentro da própria recência:
        # a parte inicial sugere a mudança e a parte final mede se segui-la funcionou.
        corte_competencia = max(20, int(len(dados_rec) * 0.60))
        dados_calibracao = dados_rec[:corte_competencia]
        dados_validacao = dados_rec[corte_competencia:]
        calibracao = self._mapear_deriva_temporal_basica(dados_calibracao, chaves_basicas)
        validacao = self._mapear_deriva_temporal_basica(dados_validacao, chaves_basicas)
        calibracao_regras = self._mapear_deriva_temporal_regras(dados_calibracao, tipos_regras)
        validacao_regras = self._mapear_deriva_temporal_regras(dados_validacao, tipos_regras)

        for tipo in tipos_regras:
            chave = f"REGRA|{tipo}"
            st = getattr(self, "estatisticas_regras_oficiais", {}).get(tipo, {})
            longo[chave] = {
                "total": int(st.get("total", 0)),
                "V_g0": int(st.get("V_g0", 0)),
                "V_g1": int(st.get("V_g1", 0)),
                "P_g0": int(st.get("P_g0", 0)),
                "P_g1": int(st.get("P_g1", 0)),
                "B": 0,
            }
            recente[chave] = recente_regras.get(
                chave,
                {
                    "total": 0, "V_g0": 0, "V_g1": 0,
                    "P_g0": 0, "P_g1": 0, "B": 0
                }
            )
            calibracao[chave] = calibracao_regras.get(
                chave,
                {
                    "total": 0, "V_g0": 0, "V_g1": 0,
                    "P_g0": 0, "P_g1": 0, "B": 0
                }
            )
            validacao[chave] = validacao_regras.get(
                chave,
                {
                    "total": 0, "V_g0": 0, "V_g1": 0,
                    "P_g0": 0, "P_g1": 0, "B": 0
                }
            )
