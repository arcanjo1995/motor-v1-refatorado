from collections import defaultdict

class ProbabilidadesMixin:
    def obter_voto_temporal(self, ultimas_cores):
        """
        Retorna uma evidência adicional conservadora. Não veta sinais e não
        substitui Markov, ML, regras, geometria ou recência.
        """
        if not ultimas_cores or not getattr(self, "markov_temporal", None):
            return {"direcao": "NEUTRO", "peso": 0.0, "total": 0.0}

        regime = self._detectar_regime_temporal(ultimas_cores)
        acumulado_v = 0.0
        acumulado_p = 0.0
        peso_total = 0.0
        amostra_total = 0.0
        ordens = []

        min_amostra = {6: 8.0, 5: 10.0, 4: 14.0, 3: 20.0, 2: 28.0, 1: 35.0}
        peso_ordem = {6: 6.0, 5: 5.0, 4: 4.0, 3: 3.0, 2: 2.0, 1: 1.0}

        for ordem in range(min(6, len(ultimas_cores)), 0, -1):
            estado = tuple(ultimas_cores[-ordem:])
            stats_regime = self.markov_temporal_regime.get(ordem, {}).get((regime, estado))
            stats_global = self.markov_temporal.get(ordem, {}).get(estado)

            stats = None
            origem = None
            if stats_regime and stats_regime.get("total", 0.0) >= min_amostra[ordem]:
                stats = stats_regime
                origem = "REGIME"
            elif stats_global and stats_global.get("total", 0.0) >= min_amostra[ordem]:
                stats = stats_global
                origem = "GLOBAL"

            if not stats:
                continue

            total = float(stats.get("total", 0.0))
            denom = total + 3.0
            prob_v = (float(stats.get("V", 0.0)) + 1.0) / denom
            prob_p = (float(stats.get("P", 0.0)) + 1.0) / denom
            confianca_amostra = min(1.0, total / (min_amostra[ordem] * 4.0))
            peso = peso_ordem[ordem] * (0.5 + 0.5 * confianca_amostra)

            acumulado_v += prob_v * peso
            acumulado_p += prob_p * peso
            peso_total += peso
            amostra_total += total
            ordens.append({"ordem": ordem, "origem": origem, "massa": round(total, 2)})

        if peso_total <= 0:
            return {"direcao": "NEUTRO", "peso": 0.0, "total": 0.0, "regime": regime}

        prob_v = (acumulado_v / peso_total) * 100
        prob_p = (acumulado_p / peso_total) * 100
        margem = abs(prob_v - prob_p)
        direcao = "VERMELHO" if prob_v > prob_p else ("PRETO" if prob_p > prob_v else "NEUTRO")

        # Peso limitado: a memória temporal é especialista adicional,
        # nunca autoridade absoluta sobre as camadas já existentes.
        if margem >= 8.0:
            peso_voto = 2.5
        elif margem >= 5.0:
            peso_voto = 2.0
        elif margem >= 3.0:
            peso_voto = 1.0
        else:
            peso_voto = 0.0
            direcao = "NEUTRO"

        peso_voto = min(
            peso_voto,
            float(self.temporal_config.get("peso_maximo_no_juiz", 2.5))
        )
        return {
            "direcao": direcao,
            "peso": peso_voto,
            "total": round(amostra_total, 2),
            "regime": regime,
            "V": round(prob_v, 2),
            "P": round(prob_p, 2),
            "margem": round(margem, 2),
            "ordens_utilizadas": ordens
        }
    def _chave_respeito_projecao(self, numero, sub_pol):
        regime = self._detectar_regime_temporal(sub_pol)
        geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
        padrao = "".join(sub_pol[-3:]) if len(sub_pol) >= 3 else "".join(sub_pol)
        return f"N={int(numero)}|REG={regime}|GEO={geometria}|PAD={padrao}"
   def _obter_respeito_projecao_contextual(self, numero, sub_num, sub_pol):
        """
        Mede se a contagem V3 costuma ser RESPEITADA até G1.
        Nunca compara V contra P e nunca transforma projeção em sinal PRETO.
        """
        global_stats = self.estatisticas_projecoes_respeito.get(int(numero), {})
        total_global = int(global_stats.get("total", 0))
        respeitadas_global = int(global_stats.get("respeitada_g0", 0)) + int(global_stats.get("respeitada_g1", 0))
        taxa_global = respeitadas_global / max(total_global, 1)

        chave = self._chave_respeito_projecao(numero, sub_pol)
        ctx = self.projecoes_respeito_contextual.get(chave)
        if ctx and ctx.get("total", 0) >= 20:
            total_ctx = int(ctx["total"])
            respeitadas_ctx = int(ctx.get("respeitada_g0", 0)) + int(ctx.get("respeitada_g1", 0))
            taxa_ctx = respeitadas_ctx / max(total_ctx, 1)
            forca = total_ctx / (total_ctx + 20.0)
            taxa = (taxa_ctx * forca) + (taxa_global * (1.0 - forca))
            return {
                "taxa_respeito": taxa,
                "suporte": total_ctx,
                "fonte": "CONTEXTO",
                "chave": chave
            }

        return {
            "taxa_respeito": taxa_global,
            "suporte": total_global,
            "fonte": "GLOBAL",
            "chave": f"N={int(numero)}"
        }
    def _mapear_projecoes_globais(self, dados):
        if not dados or len(dados) < 10:
            return

        self.estatisticas_projecoes_globais = {
            n: {"total": 0, "g0": 0, "g1": 0, "falha": 0}
            for n in range(1, 8)
        }
        self.estatisticas_projecoes_bilaterais = {}
        self.estatisticas_projecoes_respeito = {
            n: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0}
            for n in range(1, 8)
        }
        self.projecoes_respeito_contextual = defaultdict(
            lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0}
        )
        self.especialista_espelho_inversao = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0}
        )
        self.estatisticas_bigramas_globais = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0}
        )
        self.estatisticas_trigramas_globais = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0}
        )
        self.estatisticas_regras_oficiais = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0}
        )

        total_dados = len(dados)
        for i in range(total_dados):
            num = dados[i]["numero"]

            # Projeções V3: alvo original VERMELHO. O aprendizado agora mede
            # somente se a CONTAGEM FOI RESPEITADA ou NÃO RESPEITADA até G1.
            if 1 <= num <= 7:
                alvo_idx = i + num
                if alvo_idx + 1 < total_dados:
                    caminho_tem_branco = any(
                        dados[k]["cor"] == "B" for k in range(i + 1, alvo_idx)
                    )
                    if not caminho_tem_branco:
                        cor_alvo = dados[alvo_idx]["cor"]
                        cor_g1 = dados[alvo_idx + 1]["cor"]

                        if cor_alvo in ("V", "B"):
                            resultado = "G0"
                        elif cor_g1 in ("V", "B"):
                            resultado = "G1"
                        else:
                            resultado = "NAO_RESPEITADA"

                        stats_global = self.estatisticas_projecoes_globais[num]
                        stats_respeito = self.estatisticas_projecoes_respeito[num]
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
                            for d in dados[inicio_ctx:alvo_idx + 1]
                        ]
                        if len(pol_ctx) >= 3:
                            chave_ctx = self._chave_respeito_projecao(num, pol_ctx)
                            stats_ctx = self.projecoes_respeito_contextual[chave_ctx]
                            stats_ctx["total"] += 1
                            if resultado == "G0":
                                stats_ctx["respeitada_g0"] += 1
                            elif resultado == "G1":
                                stats_ctx["respeitada_g1"] += 1
                            else:
                                stats_ctx["nao_respeitada"] += 1

            # Bigramas Globais (Até G1 para ambas as cores)
            if i >= 1 and i + 2 < total_dados:
                bigrama = f"{dados[i-1]['numero']}-{dados[i]['numero']}"
                c0 = dados[i+1]["cor"]
                c1 = dados[i+2]["cor"]
                self.estatisticas_bigramas_globais[bigrama]["total"] += 1
                if c0 in ("V", "B"):
                    self.estatisticas_bigramas_globais[bigrama]["V_g0"] += 1
                elif c1 in ("V", "B"):
                    self.estatisticas_bigramas_globais[bigrama]["V_g1"] += 1
                if c0 in ("P", "B"):
                    self.estatisticas_bigramas_globais[bigrama]["P_g0"] += 1
                elif c1 in ("P", "B"):
                    self.estatisticas_bigramas_globais[bigrama]["P_g1"] += 1

            # Trigramas Globais (Até G1 para ambas as cores)
            if i >= 2 and i + 2 < total_dados:
                trigrama = f"{dados[i-2]['numero']}-{dados[i-1]['numero']}-{dados[i]['numero']}"
                c0 = dados[i+1]["cor"]
                c1 = dados[i+2]["cor"]
                self.estatisticas_trigramas_globais[trigrama]["total"] += 1
                if c0 in ("V", "B"):
                    self.estatisticas_trigramas_globais[trigrama]["V_g0"] += 1
                elif c1 in ("V", "B"):
                    self.estatisticas_trigramas_globais[trigrama]["V_g1"] += 1
                if c0 in ("P", "B"):
                    self.estatisticas_trigramas_globais[trigrama]["P_g0"] += 1
                elif c1 in ("P", "B"):
                    self.estatisticas_trigramas_globais[trigrama]["P_g1"] += 1

        # Auditoria cronológica das regras oficiais estruturais, exatamente
        # como bigramas/trigramas: janela de 12 -> consequência em G0/G1.
        for fim in range(11, total_dados - 2):
            janela_num = [int(d["numero"]) for d in dados[fim - 11:fim + 1]]
            janela_pol = [str(d["cor"]).upper() for d in dados[fim - 11:fim + 1]]
            regras = MotorContagensProjetivas.mapear_janela(
                janela_num, janela_pol,
                AnalisadorContextoAvancado.mapear_padroes_geometria(janela_pol),
                None
            )
            c0 = str(dados[fim + 1]["cor"]).upper()
            c1 = str(dados[fim + 2]["cor"]).upper()
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
                stats_regra = self.estatisticas_regras_oficiais[tipo]
                stats_regra["total"] += 1
                if direcao == "VERMELHO":
                    if c0 in ("V", "B"):
                        stats_regra["V_g0"] += 1
                    elif c1 in ("V", "B"):
                        stats_regra["V_g1"] += 1
                else:
                    if c0 in ("P", "B"):
                        stats_regra["P_g0"] += 1
                    elif c1 in ("P", "B"):
                        stats_regra["P_g1"] += 1

        self.regras_oficiais_metricas = {
            "ativo": True,
            "metodo": "REGRAS_OFICIAIS_AUDITADAS_ATE_G1",
            "regras_mapeadas": len(self.estatisticas_regras_oficiais),
            "ocorrencias_mapeadas": sum(v["total"] for v in self.estatisticas_regras_oficiais.values()),
            "participa_geracao_sinal": True
        }

        total_projecoes = sum(
            stats["total"] for stats in self.estatisticas_projecoes_respeito.values()
        )
        total_respeitadas = sum(
            stats["respeitada_g0"] + stats["respeitada_g1"]
            for stats in self.estatisticas_projecoes_respeito.values()
        )
        self.projecoes_respeito_metricas = {
            "ativo": True,
            "leitura_bilateral_v_p": False,
            "metodo": "RESPEITADA_VS_NAO_RESPEITADA_ATE_G1",
            "total_contagens_mapeadas": total_projecoes,
            "contagens_respeitadas_ate_g1": total_respeitadas,
            "contagens_nao_respeitadas": total_projecoes - total_respeitadas,
            "taxa_respeito_g0_g1_percent": round(
                (total_respeitadas / max(total_projecoes, 1)) * 100, 2
            ),
            "contextos_respeito_aprendidos": len(self.projecoes_respeito_contextual),
            "regra_v3_direcao_original": "VERMELHO"
        }

        self._mapear_especialista_espelho_inversao(dados)
    @staticmethod
    def _identificar_contexto_espelho_inversao(sub_num, sub_pol):
        if not sub_num or not sub_pol:
            return None
        limite, ultimo_num = min(10, len(sub_pol)), int(sub_num[-1])
        for tam in range(limite, 2, -1):
            janela = list(sub_pol[-tam:])
            if 'B' in janela: continue
            eh_streak = len(set(janela)) == 1
            eh_xadrez = all(janela[j] != janela[j-1] for j in range(1, tam))
            eh_espelho_normal = all(janela[j] == janela[tam-1-j] for j in range(tam)) and not eh_streak
            eh_espelho_invertido = all(janela[j] != janela[tam-1-j] for j in range(tam)) and not eh_xadrez
            if eh_espelho_normal: return f"ESPELHO_NORMAL_{tam}|N={ultimo_num}|F={janela[-1]}"
            if eh_espelho_invertido: return f"ESPELHO_INVERTIDO_{tam}|N={ultimo_num}|F={janela[-1]}"
            eh_duplo = False
            if tam >= 4 and tam % 2 == 0:
                metade = tam // 2
                eh_duplo = len(set(janela[:metade])) == 1 and len(set(janela[metade:])) == 1 and janela[0] != janela[metade]
            if eh_streak: return f"INV_STREAK_{tam}|N={ultimo_num}|F={janela[-1]}"
            if eh_xadrez: return f"INV_XADREZ_{tam}|N={ultimo_num}|F={janela[-1]}"
            if eh_duplo: return f"INV_DUPLO_{tam}|N={ultimo_num}|F={janela[-1]}"
        return None

  def _mapear_especialista_espelho_inversao(self, dados):
        if not dados or len(dados) < 15: return
        for i in range(11, len(dados) - 2):
            janela = dados[i-11:i+1]
            sub_num, sub_pol = [d['numero'] for d in janela], [d['cor'] for d in janela]
            chave = self._identificar_contexto_espelho_inversao(sub_num, sub_pol)
            if not chave: continue
            c0, c1 = dados[i+1]['cor'], dados[i+2]['cor']
            stats = self.especialista_espelho_inversao[chave]
            stats["total"] += 1
            if c0 in ['V', 'B']: stats["V_g0"] += 1
            elif c1 in ['V', 'B']: stats["V_g1"] += 1
            if c0 in ['P', 'B']: stats["P_g0"] += 1
            elif c1 in ['P', 'B']: stats["P_g1"] += 1

    def obter_voto_espelho_inversao(self, sub_num, sub_pol):
        chave = self._identificar_contexto_espelho_inversao(sub_num, sub_pol)
        if not chave: return {"direcao": "NEUTRO", "peso": 0.0, "total": 0}
        stats = self.especialista_espelho_inversao.get(chave)
        if not stats or stats.get("total", 0) < 20:
            return {"direcao": "NEUTRO", "peso": 0.0, "total": stats.get("total", 0) if stats else 0, "chave": chave}
        total = stats["total"]
        taxa_v = ((stats["V_g0"] + stats["V_g1"]) / total) * 100
        taxa_p = ((stats["P_g0"] + stats["P_g1"]) / total) * 100
        margem, melhor = abs(taxa_v - taxa_p), max(taxa_v, taxa_p)
        if melhor < 58.0 or margem < 6.0:
            return {"direcao": "NEUTRO", "peso": 0.0, "total": total, "V": round(taxa_v, 2), "P": round(taxa_p, 2), "chave": chave}
        peso = 18.0 if melhor >= 70.0 and margem >= 12.0 else (12.0 if melhor >= 63.0 and margem >= 8.0 else 8.0)
        return {"direcao": "VERMELHO" if taxa_v > taxa_p else "PRETO", "peso": peso, "total": total, "V": round(taxa_v, 2), "P": round(taxa_p, 2), "margem": round(margem, 2), "chave": chave}

    def _calcular_probabilidades_globais_cache(self):
        self.probabilidades_globais["streak_v_5"] = self.calcular_probabilidade_streak_empirica('V', 5)
        self.probabilidades_globais["streak_p_5"] = self.calcular_probabilidade_streak_empirica('P', 5)
        self.probabilidades_globais["xadrez_5"] = self.calcular_probabilidade_xadrez_empirica(5)

def obter_voto_contagens_consolidado(self, sub_num, sub_pol, regras_posicionais):
        """
        Consolida V3, coexistência, finalização conjunta e hierarquia em UMA única
        raiz causal CONTAGENS. A direção/força é calibrada exclusivamente pela
        cartografia contextual histórica já existente, sem alterar os detectores.
        """
        tipos_consolidados = {
            "COEXISTENCIA_CONTAGENS_ATIVA",
            "FINALIZACAO_CONJUNTA_ATIVA",
        }

        regras_contagens = []
        for regra in regras_posicionais or []:
            tipo = str(regra.get("tipo_regra", ""))
            if (
                tipo.startswith("V3_ATIVADOR_")
                or tipo.startswith("HIERARQUIA_CONTAGEM_")
                or tipo in tipos_consolidados
            ):
                regras_contagens.append(regra)

        if not regras_contagens:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "suporte": 0, "tipos": []
            }

        mapa = getattr(self, "cartografia_regras_contextual", {})
        if not mapa or len(sub_num) < 12 or len(sub_pol) < 12:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "suporte": 0,
                "tipos": sorted({str(r.get("tipo_regra", "")) for r in regras_contagens})
            }

        # Uma ocorrência repetida do mesmo ativador não vira um novo voto causal.
        regras_unicas = {}
        for regra in regras_contagens:
            tipo = str(regra.get("tipo_regra", ""))
            regras_unicas.setdefault(tipo, regra)

        eventos_contagens = [
            {
                "tipo": tipo,
                "direcao": regra.get("direcao", "NEUTRO"),
                "familia": regra.get("familia", "CONTAGENS"),
            }
            for tipo, regra in regras_unicas.items()
        ]
        chaves_familia = self._chaves_cartografia_contextual_eventos(
            sub_num[-12:], sub_pol[-12:], eventos_contagens
        )

        leituras_por_tipo = []
        for tipo in regras_unicas:
            prefixo_tipo = f"REGRA_CTX|E={tipo}"
            chaves = [
                chave for chave in chaves_familia
                if chave == prefixo_tipo or chave.startswith(f"{prefixo_tipo}|")
            ]

            leituras_tipo = []
            for chave in chaves:
                st = mapa.get(chave)
                if not st:
                    continue
                suporte = int(st.get("total", 0))
                if suporte < 20:
                    continue

                v_g0 = float(st.get("V_g0", 0)) / suporte
                p_g0 = float(st.get("P_g0", 0)) / suporte
                v_g01 = (
                    float(st.get("V_g0", 0)) + float(st.get("V_g1", 0))
                ) / suporte
                p_g01 = (
                    float(st.get("P_g0", 0)) + float(st.get("P_g1", 0))
                ) / suporte

                score_v = (0.70 * v_g0) + (0.30 * v_g01)
                score_p = (0.70 * p_g0) + (0.30 * p_g01)

                if "|TRI=" in chave and "|PAD=" in chave and "|REG=" in chave:
                    especificidade = 1.00
                elif "|ULT=" in chave and "|BI=" in chave:
                    especificidade = 0.98
                elif "|REG=" in chave and "|MK=" in chave:
                    especificidade = 0.95
                elif "|COEX=" in chave or "|CONT=" in chave:
                    especificidade = 0.92
                elif "|TRI=" in chave or "|PAD=" in chave:
                    especificidade = 0.88
                elif "|BI=" in chave:
                    especificidade = 0.82
                elif "|ULT=" in chave:
                    especificidade = 0.78
                elif "|REG=" in chave or "|MK=" in chave or "|GEO=" in chave:
                    especificidade = 0.74
                else:
                    especificidade = 0.58

                shrink = suporte / (suporte + 30.0)
                peso_ctx = especificidade * shrink
                leituras_tipo.append((score_v, score_p, peso_ctx, suporte))

            if leituras_tipo:
                peso_tipo = sum(x[2] for x in leituras_tipo)
                score_v_tipo = sum(x[0] * x[2] for x in leituras_tipo) / max(peso_tipo, 1e-9)
                score_p_tipo = sum(x[1] * x[2] for x in leituras_tipo) / max(peso_tipo, 1e-9)
                suporte_tipo = sum(x[3] for x in leituras_tipo) / len(leituras_tipo)
                leituras_por_tipo.append(
                    (score_v_tipo, score_p_tipo, suporte_tipo)
                )

        if not leituras_por_tipo:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "suporte": 0,
                "tipos": sorted(regras_unicas)
            }

        # Cada TIPO estrutural participa uma vez; a família produz um único voto.
        score_v = sum(x[0] for x in leituras_por_tipo) / len(leituras_por_tipo)
        score_p = sum(x[1] for x in leituras_por_tipo) / len(leituras_por_tipo)
        suporte_medio = sum(x[2] for x in leituras_por_tipo) / len(leituras_por_tipo)
        margem = abs(score_v - score_p)

        if margem < 0.025:
            direcao = "NEUTRO"
            peso = 0.0
        else:
            direcao = "VERMELHO" if score_v > score_p else "PRETO"
            # Um único bônus causal, limitado ao peso ALTO já usado pelo motor.
            confianca_ctx = min(1.0, margem / 0.20)
            confianca_suporte = min(1.0, suporte_medio / 80.0)
            peso = 18.0 * confianca_ctx * confianca_suporte

        return {
            "ativo": direcao in ("VERMELHO", "PRETO"),
            "direcao": direcao,
            "peso": round(peso, 4),
            "margem": round(margem, 6),
            "suporte": int(round(suporte_medio)),
            "score_v": round(score_v, 6),
            "score_p": round(score_p, 6),
            "tipos": sorted(regras_unicas),
        }


    def calcular_probabilidade_streak_empirica(self, cor, k):
        todos = (self.dados_longo or []) + (self.dados_recencia or [])
        if len(todos) < k + 1: return 0.0
        total = len(todos) - k
        count = sum(1 for i in range(total) if all(d['cor'] == cor for d in todos[i:i+k]))
        return round((count / total) * 100, 2) if total > 0 else 0.0

    def calcular_probabilidade_xadrez_empirica(self, k):
        todos = (self.dados_longo or []) + (self.dados_recencia or [])
        if len(todos) < k + 1: return 0.0
        total = len(todos) - k
        count = 0
        for i in range(total):
            janela = [d['cor'] for d in todos[i:i+k]]
            if all(janela[j] != janela[j-1] for j in range(1, len(janela))):
                count += 1
        return round((count / total) * 100, 2) if total > 0 else 0.0
