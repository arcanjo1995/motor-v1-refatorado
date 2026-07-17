from collections import defaultdict
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas

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

        def medir(dados, numero):
            cont = {"V": 0, "P": 0, "B": 0}
            total = 0
            for i in range(len(dados) - 1):
                if int(dados[i].get("numero", -1)) != numero:
                    continue
                cor = str(dados[i + 1].get("cor", "B")).upper()
                if cor in cont:
                    cont[cor] += 1
                    total += 1
            taxas = {c: (cont[c] / total if total else 0.0) for c in cont}
            return {"total": total, "contagens": cont, "taxas": taxas}

        resultado = {}
        for numero in range(15):
            macro = medir(longo, numero)
            por_horizonte = {
                str(h): medir(recente[-h:], numero) for h in horizontes
            }
            macro_vp = {c: macro["taxas"][c] for c in ("V", "P")}
            dom_macro = max(macro_vp, key=macro_vp.get) if macro["total"] else None
            m50 = por_horizonte["50"]
            m25 = por_horizonte["25"]
            dom50 = max(("V", "P"), key=lambda c: m50["taxas"][c]) if m50["total"] else None
            dom25 = max(("V", "P"), key=lambda c: m25["taxas"][c]) if m25["total"] else None
            margem50 = abs(m50["taxas"]["V"] - m50["taxas"]["P"])
            estado = "SEM_SUPORTE"
            if macro["total"] >= 30 and m50["total"] >= 8:
                if (
                    dom_macro in ("V", "P") and dom50 in ("V", "P")
                    and dom50 != dom_macro and margem50 >= 0.20
                    and m25["total"] >= 4 and dom25 == dom50
                ):
                    estado = "INVERSAO_COMPORTAMENTAL"
                elif dom50 != dom_macro and margem50 >= 0.12:
                    estado = "MUDANCA_COMPORTAMENTAL"
                else:
                    delta_dom = m50["taxas"].get(dom_macro, 0.0) - macro["taxas"].get(dom_macro, 0.0)
                    estado = "ENFRAQUECENDO" if delta_dom <= -0.12 else "PRESERVADO"
            resultado[numero] = {
                "macro": macro,
                "horizontes_recencia": por_horizonte,
                "direcao_macro": dom_macro,
                "direcao_50": dom50,
                "direcao_25": dom25,
                "estado": estado
            }
        self.matriz_deriva_comportamental = {
            "ativo": True,
            "metodo": "MACRO_X_RECENCIA_200_100_50_25",
            "numeros": resultado,
            "altera_direcao": False,
            "altera_peso_recencia": False
        }
        return self.matriz_deriva_comportamental

    def validar_autoridade_hierarquica_contextual(self, sub_num, sub_pol, regra_id, direcao):
        """
        MAIN 115 — prova contextual da autoridade hierárquica.
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
        prefixo = f"REGRA_CTX|E={regra_id}"

        def dimensao(chave):
            if chave == prefixo:
                return "REGRA_GLOBAL", 50, 0.42
            if "|TRI=" in chave and "|PAD=" in chave and "|REG=" in chave:
                return "TRIGRAMA_PADRAO_REGIME", 20, 1.00
            if "|ULT=" in chave and "|BI=" in chave:
                return "NUMERO_BIGRAMA", 20, 0.96
            if "|TRI=" in chave and "|PAD=" in chave:
                return "TRIGRAMA_PADRAO", 20, 0.92
            if "|PAD=" in chave and "|REG=" in chave:
                return "PADRAO_REGIME", 25, 0.88
            if "|REG=" in chave and "|HMM=" in chave:
                return "REGIME_HMM", 25, 0.90
            if "|HMM=" in chave:
                return "HMM", 30, 0.82
            if "|REG=" in chave and "|MK=" in chave:
                return "REGIME_MARKOV", 30, 0.84
            if "|TRI=" in chave:
                return "TRIGRAMA", 20, 0.80
            if "|BI=" in chave:
                return "BIGRAMA", 25, 0.76
            if "|ULT=" in chave:
                return "ULTIMO_NUMERO", 30, 0.72
            if "|PAD=" in chave:
                return "PADRAO", 30, 0.70
            if "|REG=" in chave:
                return "REGIME", 40, 0.62
            if "|GEO=" in chave:
                return "GEOMETRIA", 35, 0.62
            if "|MK=" in chave:
                return "MARKOV", 40, 0.58
            return "OUTRO_CONTEXTO", 40, 0.50

        evidencias = []
        componentes = {}
        for chave in chaves:
            if chave != prefixo and not chave.startswith(prefixo + "|"):
                continue
            stats = mapa.get(chave)
            if not stats:
                continue
            nome, minimo, especificidade = dimensao(chave)
            total = int(stats.get("total", 0) or 0)
            if total <= 0:
                continue
            taxa_v = (float(stats.get("V_g0", 0)) + float(stats.get("V_g1", 0))) / max(total, 1)
            taxa_p = (float(stats.get("P_g0", 0)) + float(stats.get("P_g1", 0))) / max(total, 1)
            taxa_dir = taxa_v if direcao == "VERMELHO" else taxa_p
            taxa_contra = taxa_p if direcao == "VERMELHO" else taxa_v
            margem = taxa_dir - taxa_contra

            shrink = total / (total + minimo)
            maturidade_suporte = min(1.0, total / max(float(minimo), 1.0))
            peso = especificidade * shrink * maturidade_suporte

            if total < 5:
                faixa_suporte = "RARO"
            elif total < minimo:
                faixa_suporte = "EM_FORMACAO"
            else:
                faixa_suporte = "CONSOLIDADO"

            stats_rec = mapa_recente.get(chave, {}) or {}
            total_rec = int(stats_rec.get("total", 0) or 0)
            taxa_v_rec = (
                float(stats_rec.get("V_g0", 0)) + float(stats_rec.get("V_g1", 0))
            ) / max(total_rec, 1)
            taxa_p_rec = (
                float(stats_rec.get("P_g0", 0)) + float(stats_rec.get("P_g1", 0))
            ) / max(total_rec, 1)
            taxa_dir_rec = taxa_v_rec if direcao == "VERMELHO" else taxa_p_rec
            taxa_contra_rec = taxa_p_rec if direcao == "VERMELHO" else taxa_v_rec

            item = {
                "fonte": nome, "suporte": total,
                "suporte_minimo_consolidacao": minimo,
                "faixa_suporte": faixa_suporte,
                "taxa_direcao_g0_g1": round(taxa_dir * 100.0, 2),
                "taxa_contraria_g0_g1": round(taxa_contra * 100.0, 2),
                "margem": round(margem * 100.0, 2),
                "suporte_recente": total_rec,
                "taxa_direcao_recente_g0_g1": round(taxa_dir_rec * 100.0, 2),
                "taxa_contraria_recente_g0_g1": round(taxa_contra_rec * 100.0, 2),
                "delta_margem_macro_recente": round(
                    ((taxa_dir_rec - taxa_contra_rec) - margem) * 100.0, 2
                ) if total_rec else None,
                "shrinkage_suporte": round(shrink, 4),
                "maturidade_suporte": round(maturidade_suporte, 4),
                "peso": round(peso, 4)
            }
            evidencias.append((taxa_dir, taxa_contra, peso, item))
            anterior = componentes.get(nome)
            if anterior is None or total > anterior["suporte"]:
                componentes[nome] = item

        if not evidencias:
            self._ultima_validacao_autoridade_contextual = neutro
            return neutro

        soma_pesos = sum(x[2] for x in evidencias)
        taxa_dir = sum(x[0] * x[2] for x in evidencias) / max(soma_pesos, 1e-9)
        taxa_contra = sum(x[1] * x[2] for x in evidencias) / max(soma_pesos, 1e-9)
        margem = taxa_dir - taxa_contra
        suporte = max(x[3]["suporte"] for x in evidencias)
        fortes_contrarias = [x[3] for x in evidencias if x[3]["suporte"] >= 25 and x[3]["margem"] <= -8.0]
        fontes_pro = [x[3] for x in evidencias if x[3]["suporte"] >= 20 and x[3]["margem"] >= 5.0]
        fontes_contra = [x[3] for x in evidencias if x[3]["suporte"] >= 20 and x[3]["margem"] <= -5.0]
        fragmentacao = bool(fontes_pro and fontes_contra)

        ultimo_numero = int(list(sub_num)[-1])
        deriva_numero = (
            (matriz_deriva.get("numeros", {}) or {}).get(ultimo_numero, {})
        )
        estado_deriva = str(deriva_numero.get("estado", "SEM_SUPORTE"))

        recentes_contrarias = [
            x[3] for x in evidencias
            if x[3].get("suporte_recente", 0) >= 5
            and (
                float(x[3].get("taxa_contraria_recente_g0_g1", 0.0))
                - float(x[3].get("taxa_direcao_recente_g0_g1", 0.0))
            ) >= 15.0
        ]

        if estado_deriva == "INVERSAO_COMPORTAMENTAL" and recentes_contrarias:
            status = "INVERSAO_CONTEXTUAL"
        elif fragmentacao:
            status = "FRAGMENTACAO_CONTEXTUAL"
        elif margem >= 0.10:
            status = "FORTALECIDA"
        elif margem >= -0.04:
            status = "NEUTRA"
        elif margem > -0.10:
            status = "DEGRADADA"
        else:
            status = "CONTRARIA"

        margem_abs = abs(margem)
        fragmentacao_estreita = bool(
            status == "FRAGMENTACAO_CONTEXTUAL"
            and suporte >= 30
            and margem_abs <= 0.02
            and len(fontes_contra) >= 1
        )
        fragmentacao_aguarda_prova_g2_mais = bool(fragmentacao_estreita)

        degradacao_contraria_robusta = bool(
            status in ("DEGRADADA", "CONTRARIA")
            and suporte >= 30
            and margem <= -0.04
            and len(fortes_contrarias) >= 2
        )

        vetar = bool(
            degradacao_contraria_robusta
            or (
                status == "INVERSAO_CONTEXTUAL"
                and len(recentes_contrarias) >= 2
            )
        )
        motivo_veto = None
        if degradacao_contraria_robusta:
            motivo_veto = "DEGRADACAO_CONTRARIA_ROBUSTA"
        elif status == "INVERSAO_CONTEXTUAL" and len(recentes_contrarias) >= 2:
            motivo_veto = "INVERSAO_CONFIRMADA_PELA_MICRO_RECENCIA"

        resultado = {
            "ativo": True, "status": status, "vetar": vetar,
            "regra": str(regra_id), "direcao": direcao,
            "autoridade_contextual": round(max(0.0, min(1.0, taxa_dir)), 4),
            "taxa_direcao_g0_g1": round(taxa_dir * 100.0, 2),
            "taxa_contraria_g0_g1": round(taxa_contra * 100.0, 2),
            "margem_contextual": round(margem * 100.0, 2),
            "suporte": suporte,
            "evidencias": [x[3] for x in sorted(evidencias, key=lambda y: (-y[2], -y[3]["suporte"]))],
            "componentes": componentes,
            "fontes_contrarias_fortes": [x["fonte"] for x in fortes_contrarias],
            "fontes_fragmentacao_pro": [x["fonte"] for x in fontes_pro],
            "fontes_fragmentacao_contra": [x["fonte"] for x in fontes_contra],
            "fontes_recentes_contrarias": [x["fonte"] for x in recentes_contrarias],
            "fragmentacao_contextual": fragmentacao,
            "fragmentacao_estreita": fragmentacao_estreita,
            "fragmentacao_aguarda_prova_g2_mais": fragmentacao_aguarda_prova_g2_mais,
            "motivo_veto": motivo_veto,
            "estado_deriva_numero_final": estado_deriva,
            "numero_final": ultimo_numero,
            "altera_direcao": False
        }
        self._ultima_validacao_autoridade_contextual = resultado
        return resultado

    def _mapear_deriva_temporal_basica(self, dados, chaves_alvo):
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

        trajetoria_temporal = self._mapear_trajetoria_deriva_temporal(
            dados_rec, chaves_basicas, tipos_regras, longo
        )

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

        def familia_da_chave(chave):
            if chave.startswith("REGRA|"):
                tipo = chave.split("|", 1)[1]
                if tipo.startswith("V3_ATIVADOR_") or tipo.startswith("HIERARQUIA_CONTAGEM_"):
                    return "CONTAGENS"
                if tipo in {"COEXISTENCIA_CONTAGENS_ATIVA", "FINALIZACAO_CONJUNTA_ATIVA"}:
                    return "CONTAGENS"
                if tipo.startswith("REGRA_4_"):
                    return "REGRA_4"
                if tipo.startswith("REGRA_10_"):
                    return "REGRA_10"
                if tipo.startswith("REGRA_5_10_"):
                    return "REGRA_5_10"
                return f"REGRA:{tipo}"
            prefixo = chave.split("|", 1)[0]
            return {
                "NUM": "NUMERO", "BI": "BIGRAMA", "TRI": "TRIGRAMA",
                "DELTA_TRAJ": "NUMERO", "BLOCOS": "GEOMETRIA",
                "ALT_BLOCOS": "GEOMETRIA",
                "PAD": "PADRAO", "STREAK": "STREAK", "GEO": "GEOMETRIA",
                "MARKOV": "MARKOV", "REGIME": "REGIME"
            }.get(prefixo, prefixo)

        leituras = []
        leituras_validas = []
        familias_acumuladas = defaultdict(
            lambda: {
                "peso": 0.0, "competencia_g0": 0.0,
                "competencia_g1": 0.0, "suporte_recencia": 0
            }
        )

        todas_chaves = list(chaves_basicas) + [f"REGRA|{t}" for t in tipos_regras]
        for chave in todas_chaves:
            hist = longo.get(chave, {})
            rec = recente.get(chave, {})
            suporte_longo = int(hist.get("total", 0))
            suporte_rec = int(rec.get("total", 0))
            if suporte_longo < 20 or suporte_rec < 4:
                continue

            hist_g0_validos = float(hist.get("V_g0", 0) + hist.get("P_g0", 0))
            rec_g0_validos = float(rec.get("V_g0", 0) + rec.get("P_g0", 0))
            if hist_g0_validos <= 0 or rec_g0_validos <= 0:
                continue

            hist_v_g0 = float(hist.get("V_g0", 0)) / hist_g0_validos
            hist_p_g0 = float(hist.get("P_g0", 0)) / hist_g0_validos
            rec_v_g0 = float(rec.get("V_g0", 0)) / rec_g0_validos
            rec_p_g0 = float(rec.get("P_g0", 0)) / rec_g0_validos

            hist_total = max(float(suporte_longo), 1.0)
            rec_total = max(float(suporte_rec), 1.0)
            hist_v_g1 = float(hist.get("V_g0", 0) + hist.get("V_g1", 0)) / hist_total
            hist_p_g1 = float(hist.get("P_g0", 0) + hist.get("P_g1", 0)) / hist_total
            rec_v_g1 = float(rec.get("V_g0", 0) + rec.get("V_g1", 0)) / rec_total
            rec_p_g1 = float(rec.get("P_g0", 0) + rec.get("P_g1", 0)) / rec_total

            confianca_rec = min(1.0, suporte_rec / 30.0)

            cal = calibracao.get(chave, {})
            val = validacao.get(chave, {})
            suporte_cal = int(cal.get("total", 0))
            suporte_val = int(val.get("total", 0))
            cal_g0_validos = float(cal.get("V_g0", 0) + cal.get("P_g0", 0))
            val_g0_validos = float(val.get("V_g0", 0) + val.get("P_g0", 0))

            competencia_g0 = 0.50
            competencia_g1 = 0.50
            if suporte_cal >= 4 and suporte_val >= 3 and cal_g0_validos > 0 and val_g0_validos > 0:
                cal_v_g0 = float(cal.get("V_g0", 0)) / cal_g0_validos
                cal_p_g0 = float(cal.get("P_g0", 0)) / cal_g0_validos
                cal_v_g1 = float(cal.get("V_g0", 0) + cal.get("V_g1", 0)) / max(float(suporte_cal), 1.0)
                cal_p_g1 = float(cal.get("P_g0", 0) + cal.get("P_g1", 0)) / max(float(suporte_cal), 1.0)
                confianca_cal = min(1.0, suporte_cal / 30.0)

                cal_v_adaptado_g0 = hist_v_g0 + ((cal_v_g0 - hist_v_g0) * confianca_cal)
                cal_p_adaptado_g0 = hist_p_g0 + ((cal_p_g0 - hist_p_g0) * confianca_cal)
                cal_v_adaptado_g1 = hist_v_g1 + ((cal_v_g1 - hist_v_g1) * confianca_cal)
                cal_p_adaptado_g1 = hist_p_g1 + ((cal_p_g1 - hist_p_g1) * confianca_cal)

                direcao_cal_g0 = "V" if cal_v_adaptado_g0 >= cal_p_adaptado_g0 else "P"
                direcao_cal_g1 = "V" if cal_v_adaptado_g1 >= cal_p_adaptado_g1 else "P"

                acertos_val_g0 = float(val.get(f"{direcao_cal_g0}_g0", 0))
                acertos_val_g1 = float(
                    val.get(f"{direcao_cal_g1}_g0", 0)
                    + val.get(f"{direcao_cal_g1}_g1", 0)
                )

                competencia_g0 = (acertos_val_g0 + 2.0) / (val_g0_validos + 4.0)
                competencia_g1 = (acertos_val_g1 + 2.0) / (float(suporte_val) + 4.0)

            familia = familia_da_chave(chave)
            peso_competencia = (
                min(1.0, suporte_val / 20.0)
                * min(1.0, suporte_longo / 100.0)
            )
            fam = familias_acumuladas[familia]
            fam["peso"] += peso_competencia
            fam["competencia_g0"] += competencia_g0 * peso_competencia
            fam["competencia_g1"] += competencia_g1 * peso_competencia
            fam["suporte_recencia"] += suporte_rec

            leituras_validas.append({
                "chave": chave,
                "familia": familia,
                "hist_v_g0": hist_v_g0, "hist_p_g0": hist_p_g0,
                "rec_v_g0": rec_v_g0, "rec_p_g0": rec_p_g0,
                "hist_v_g1": hist_v_g1, "hist_p_g1": hist_p_g1,
                "rec_v_g1": rec_v_g1, "rec_p_g1": rec_p_g1,
                "confianca_rec": confianca_rec,
                "suporte_longo": suporte_longo,
                "suporte_rec": suporte_rec,
                "competencia_g0_cenario": competencia_g0,
                "competencia_g1_cenario": competencia_g1,
                "suporte_calibracao": suporte_cal,
                "suporte_validacao": suporte_val,
                "trajetoria_temporal": trajetoria_temporal.get(chave, {}),
            })

        competencia_familias = {}
        for familia, dados_fam in familias_acumuladas.items():
            peso_fam = dados_fam["peso"]
            if peso_fam <= 0:
                continue
            competencia_familias[familia] = {
                "G0": dados_fam["competencia_g0"] / peso_fam,
                "G1": dados_fam["competencia_g1"] / peso_fam,
                "suporte_recencia": dados_fam["suporte_recencia"],
            }

        score_v = 0.0
        score_p = 0.0
        soma_pesos = 0.0

        for item in leituras_validas:
            chave = item["chave"]
            familia = item["familia"]
            comp_fam = competencia_familias.get(familia, {"G0": 0.5, "G1": 0.5})
            competencia_g0 = float(comp_fam.get("G0", 0.5))
            competencia_g1 = float(comp_fam.get("G1", 0.5))

            fator_g0 = max(0.0, min(1.0, (competencia_g0 - 0.50) / 0.20))
            fator_g1 = max(0.0, min(1.0, (competencia_g1 - 0.50) / 0.20))

            v_adaptado_g0 = item["hist_v_g0"] + (
                (item["rec_v_g0"] - item["hist_v_g0"]) * item["confianca_rec"]
            )
            p_adaptado_g0 = item["hist_p_g0"] + (
                (item["rec_p_g0"] - item["hist_p_g0"]) * item["confianca_rec"]
            )
            v_adaptado_g1 = item["hist_v_g1"] + (
                (item["rec_v_g1"] - item["hist_v_g1"]) * item["confianca_rec"]
            )
            p_adaptado_g1 = item["hist_p_g1"] + (
                (item["rec_p_g1"] - item["hist_p_g1"]) * item["confianca_rec"]
            )

            delta_v_g0 = v_adaptado_g0 - item["hist_v_g0"]
            delta_p_g0 = p_adaptado_g0 - item["hist_p_g0"]
            delta_v_g1 = v_adaptado_g1 - item["hist_v_g1"]
            delta_p_g1 = p_adaptado_g1 - item["hist_p_g1"]

            trajetoria = item.get("trajetoria_temporal", {}) or {}
            fator_longitudinal = float(
                trajetoria.get("fator_longitudinal", 0.75)
            )
            delta_v_g0 *= fator_longitudinal
            delta_p_g0 *= fator_longitudinal
            delta_v_g1 *= fator_longitudinal
            delta_p_g1 *= fator_longitudinal

            if chave.startswith("REGRA|"):
                especificidade = 1.00
            elif chave.startswith("TRI|") or chave.startswith("STREAK|"):
                especificidade = 0.95
            elif chave.startswith("DELTA_TRAJ|") or chave.startswith("ALT_BLOCOS|"):
                especificidade = 0.93
            elif chave.startswith("BLOCOS|"):
                especificidade = 0.91
            elif chave.startswith("BI|") or chave.startswith("PAD|"):
                especificidade = 0.90
            elif chave.startswith("GEO|") or chave.startswith("MARKOV|"):
                especificidade = 0.85
            elif chave.startswith("REGIME|"):
                especificidade = 0.80
            else:
                especificidade = 0.75

            peso_leitura = (
                especificidade
                * item["confianca_rec"]
                * min(1.0, item["suporte_longo"] / 100.0)
            )

            score_v += (
                max(0.0, delta_v_g0) * fator_g0
                + max(0.0, delta_v_g1) * fator_g1 * 0.55
            ) * peso_leitura
            score_p += (
                max(0.0, delta_p_g0) * fator_g0
                + max(0.0, delta_p_g1) * fator_g1 * 0.55
            ) * peso_leitura
            soma_pesos += peso_leitura

            direcao_hist_g0 = "V" if item["hist_v_g0"] >= item["hist_p_g0"] else "P"
            direcao_rec_g0 = "V" if item["rec_v_g0"] >= item["rec_p_g0"] else "P"
            atraso = (
                max(item["rec_v_g1"], item["rec_p_g1"])
                > max(item["rec_v_g0"], item["rec_p_g0"]) + 0.10
            )

            leituras.append({
                "cenario": chave,
                "familia": familia,
                "suporte_longo": item["suporte_longo"],
                "suporte_recencia": item["suporte_rec"],
                "longo_G0_V_percent": round(item["hist_v_g0"] * 100.0, 2),
                "longo_G0_P_percent": round(item["hist_p_g0"] * 100.0, 2),
                "recencia_G0_V_percent": round(item["rec_v_g0"] * 100.0, 2),
                "recencia_G0_P_percent": round(item["rec_p_g0"] * 100.0, 2),
                "longo_ate_G1_V_percent": round(item["hist_v_g1"] * 100.0, 2),
                "longo_ate_G1_P_percent": round(item["hist_p_g1"] * 100.0, 2),
                "recencia_ate_G1_V_percent": round(item["rec_v_g1"] * 100.0, 2),
                "recencia_ate_G1_P_percent": round(item["rec_p_g1"] * 100.0, 2),
                "competencia_familia_G0": round(competencia_g0, 4),
                "competencia_familia_G1": round(competencia_g1, 4),
                "suporte_calibracao_temporal": item["suporte_calibracao"],
                "suporte_validacao_temporal": item["suporte_validacao"],
                "confianca_recencia": round(item["confianca_rec"], 4),
                "direcao_historica_G0": direcao_hist_g0,
                "direcao_recente_G0": direcao_rec_g0,
                "mudanca_direcao_G0": direcao_hist_g0 != direcao_rec_g0,
                "risco_atraso": atraso,
                "estado_trajetoria_temporal": trajetoria.get(
                    "estado", "TRAJETORIA_INSUFICIENTE"
                ),
                "direcao_trajetoria_atual": trajetoria.get(
                    "direcao_atual", "NEUTRO"
                ),
                "persistencia_trajetoria_blocos": int(
                    trajetoria.get("persistencia_blocos", 0)
                ),
                "velocidade_trajetoria": trajetoria.get(
                    "velocidade", "INDEFINIDA"
                ),
                "confianca_longitudinal": round(
                    float(trajetoria.get("confianca_longitudinal", 0.0)), 4
                ),
                "fator_longitudinal": round(fator_longitudinal, 4),
                "trajetoria_blocos": trajetoria.get("pontos", []),
            })

        if soma_pesos <= 0:
            retorno = {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "motivo": "SEM_SUPORTE_COMPARAVEL", "leituras": leituras,
                "competencia_familias": competencia_familias
            }
            self.ultima_deriva_temporal = retorno
            return retorno

        score_v /= soma_pesos
        score_p /= soma_pesos
        margem = abs(score_v - score_p)

        if margem < 0.01:
            direcao = "NEUTRO"
            peso = 0.0
        else:
            direcao = "VERMELHO" if score_v > score_p else "PRETO"
            peso = min(18.0, 18.0 * min(1.0, margem / 0.20))

        competencia_saida = {
            familia: {
                "G0": round(dados["G0"], 4),
                "G1": round(dados["G1"], 4),
                "suporte_recencia": int(dados["suporte_recencia"])
            }
            for familia, dados in competencia_familias.items()
        }

        retorno = {
            "ativo": direcao in ("VERMELHO", "PRETO"),
            "direcao": direcao,
            "peso": round(peso, 4),
            "score_deriva_V": round(score_v, 6),
            "score_deriva_P": round(score_p, 6),
            "margem": round(margem, 6),
            "leituras": sorted(
                leituras,
                key=lambda x: (
                    max(x["competencia_familia_G0"], x["competencia_familia_G1"]),
                    x["confianca_recencia"],
                    x["suporte_recencia"]
                ),
                reverse=True
            ),
            "competencia_familias": competencia_saida,
            "horizontes_separados": ["G0", "ATE_G1"],
            "competencia_validada_cronologicamente": True,
            "cartografia_temporal_longitudinal_ativa": True,
            "estados_temporais": [
                "ESTAVEL_COM_HISTORICO",
                "MIGRACAO_EM_OBSERVACAO",
                "INVERSAO_PERSISTENTE",
                "ACELERACAO_NA_DIRECAO_ATUAL",
                "RETORNO_AO_HISTORICO",
            ],
            "altera_regras": False,
            "altera_detectores": False,
            "peso_recencia_oficial_alterado": False,
        }
        self.ultima_deriva_temporal = retorno
        return retorno
