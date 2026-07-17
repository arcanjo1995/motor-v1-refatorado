import math
from collections import defaultdict
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas

class PadroesMixin:
    """
    Mixin para leitura de Z-Score, Quebradores Históricos e Simulação de Rotas.
    """

    def _garantir_padrao_info(self, dicionario, chave):
        """
        Garante que a chave exista no dicionário com a estrutura padrão.
        """
        if chave not in dicionario:
            dicionario[chave] = {
                "total": 0,
                "apos_V": 0,
                "apos_P": 0,
                "apos_B": 0,
                "quebradores": defaultdict(int),
                "g0": 0,
                "g1": 0,
                "_futuros": []
            }
        return dicionario[chave]

    def _garantir_quebradores_defaultdict(self, info):
        """
        Garante que o campo 'quebradores' seja um defaultdict(int) e exista.
        """
        if info is None:
            return {}
        if 'quebradores' not in info or not isinstance(info['quebradores'], defaultdict):
            info['quebradores'] = defaultdict(int)
        return info

    def mapear_padroes_avancados(self, dados):
        if not dados or len(dados) < 10: return
        cores = [d['cor'] for d in dados]
        numeros = [d['numero'] for d in dados]
        rastreio_numeros = {}
        for i in range(len(dados) - 1):
            num = numeros[i]
            cor_post = cores[i+1]
            if num not in rastreio_numeros:
                rastreio_numeros[num] = {"sequencia_atual": cor_post, "contagem": 1}
            else:
                if rastreio_numeros[num]["sequencia_atual"] == cor_post:
                    rastreio_numeros[num]["contagem"] += 1
                else:
                    cor_quebrada = rastreio_numeros[num]["sequencia_atual"]
                    tamanho_ciclo = rastreio_numeros[num]["contagem"]
                    if cor_quebrada == "V": self.saturacao_ciclica[num]["ciclos_V"].append(tamanho_ciclo)
                    elif cor_quebrada == "P": self.saturacao_ciclica[num]["ciclos_P"].append(tamanho_ciclo)
                    rastreio_numeros[num] = {"sequencia_atual": cor_post, "contagem": 1}

        i = 0
        while i < len(cores) - 4:
            janela = cores[i:i+4]
            if all(janela[j] != janela[j-1] for j in range(1, 4)) and 'B' not in janela:
                if i + 4 < len(cores):
                    proximo_cor = cores[i+4]
                    num_quebra = numeros[i+3]
                    chave = "XADREZ_4"
                    info = self._garantir_padrao_info(self.padroes_xadrez_detalhado, chave)
                    info["total"] += 1
                    info[f"apos_{proximo_cor}"] += 1
                    if proximo_cor != janela[-1]:
                        info["quebradores"][num_quebra] += 1
                    if "_futuros" not in info: info["_futuros"] = []
                    info["_futuros"].append((cores[i+4] if i+4 < len(cores) else None,
                                             cores[i+5] if i+5 < len(cores) else None,
                                             cores[i+6] if i+6 < len(cores) else None))
            i += 1

        i = 0
        while i < len(cores) - 3:
            if cores[i] == cores[i+1] == cores[i+2] and cores[i] != 'B':
                if i + 3 < len(cores):
                    proximo_cor = cores[i+3]
                    num_quebra = numeros[i+3]
                    chave = "STREAK_3"
                    info = self._garantir_padrao_info(self.padroes_streak_detalhado, chave)
                    info["total"] += 1
                    info[f"apos_{proximo_cor}"] += 1
                    if proximo_cor != cores[i]:
                        info["quebradores"][num_quebra] += 1
                    if "_futuros" not in info: info["_futuros"] = []
                    info["_futuros"].append((cores[i+3] if i+3 < len(cores) else None,
                                             cores[i+4] if i+4 < len(cores) else None,
                                             cores[i+5] if i+5 < len(cores) else None))
            i += 1

        for tam in range(3, 11):
            i = 0
            while i <= len(cores) - tam - 1:
                janela_cores = cores[i:i+tam]
                janela_nums = numeros[i:i+tam]
                if 'B' in janela_cores:
                    i += 1
                    continue
                proxima_cor = cores[i+tam]
                num_quebra = numeros[i+tam-1]
                janela_str = "-".join(janela_cores)
                
                chave_dna = "-".join(map(str, janela_nums))
                self.dna_padroes[chave_dna]["total"] += 1
                self.dna_padroes[chave_dna][proxima_cor] += 1
                
                eh_streak = len(set(janela_cores)) == 1
                eh_xadrez = all(janela_cores[j] != janela_cores[j-1] for j in range(1, tam))
                eh_duplo = False
                if tam >= 4 and tam % 2 == 0:
                    metade = tam // 2
                    if len(set(janela_cores[:metade])) == 1 and len(set(janela_cores[metade:])) == 1 and janela_cores[0] != janela_cores[metade]:
                        eh_duplo = True
                eh_espelho_normal = all(janela_cores[j] == janela_cores[tam-1-j] for j in range(tam)) and not eh_streak
                eh_espelho_invertido = all(janela_cores[j] != janela_cores[tam-1-j] for j in range(tam)) and not eh_xadrez
                if eh_streak: tipo_prefix = f"STREAK_{tam}"
                elif eh_xadrez: tipo_prefix = f"XADREZ_{tam}"
                elif eh_duplo: tipo_prefix = f"DUPLO_{tam}"
                elif eh_espelho_normal: tipo_prefix = f"ESPELHO_NORMAL_{tam}"
                elif eh_espelho_invertido: tipo_prefix = f"ESPELHO_INVERTIDO_{tam}"
                else: tipo_prefix = f"PADRAO_GERAL_{tam}"
                
                chave = f"{tipo_prefix} [{janela_str}]"
                info = self._garantir_padrao_info(self.padroes_gerais_detalhado, chave)
                info["total"] += 1
                info[f"apos_{proxima_cor}"] += 1
                if eh_streak and proxima_cor != janela_cores[-1]:
                    info["quebradores"][num_quebra] += 1
                elif eh_xadrez and proxima_cor == janela_cores[-1]:
                    info["quebradores"][num_quebra] += 1
                elif (eh_duplo or eh_espelho_normal or eh_espelho_invertido) and proxima_cor != janela_cores[-1]:
                    info["quebradores"][num_quebra] += 1
                
                if "_futuros" not in info: info["_futuros"] = []
                info["_futuros"].append((cores[i+tam] if i+tam < len(cores) else None,
                                         cores[i+tam+1] if i+tam+1 < len(cores) else None,
                                         cores[i+tam+2] if i+tam+2 < len(cores) else None))
                i += 1

        for i in range(len(cores)):
            self.color_ngrams[1][cores[i]] += 1
            if i + 1 < len(cores): self.color_ngrams[2][f"{cores[i]}-{cores[i+1]}"] += 1
            if i + 2 < len(cores): self.color_ngrams[3][f"{cores[i]}-{cores[i+1]}-{cores[i+2]}"] += 1

        for dic in [self.padroes_xadrez_detalhado, self.padroes_streak_detalhado, self.padroes_gerais_detalhado]:
            for chave, info in list(dic.items()):
                v = info.get("apos_V", 0)
                p = info.get("apos_P", 0)
                if v == 0 and p == 0: continue
                cor_alvo = "V" if v >= p else "P"
                for c1, c2, c3 in info.get("_futuros", []):
                    if c1 == cor_alvo or c1 == "B": info["g0"] += 1
                    elif c2 == cor_alvo or c2 == "B": info["g1"] += 1
                if "_futuros" in info: del info["_futuros"]

    # ============================================================
    # MÉTODOS EXISTENTES (mantidos integralmente)
    # ============================================================
    def _processar_bloco_dados(self, dados, multiplicador_peso, treinamento_profundo=False):
        if not dados or len(dados) < 3: return
        total_dados = len(dados)
        
        for i in range(total_dados - 4):
            estado_2 = (dados[i+2]['cor'], dados[i+3]['cor'])
            estado_4 = (dados[i]['cor'], dados[i+1]['cor'], dados[i+2]['cor'], dados[i+3]['cor'])
            prox = dados[i+4]['cor']
            num = dados[i+3]['numero']
            fator_temporal = 1.0 + ((i / total_dados) * 1.5) if treinamento_profundo and total_dados > 1000 else 1.0
            peso_final = max(1, int(multiplicador_peso * fator_temporal))
            for _ in range(peso_final):
                self.modelo_transicao[estado_2].append(prox)
                self.modelo_transicao_profundo[estado_4].append(prox)
                self.modelo_numerico[num].append(prox)

        for i in range(total_dados - 1):
            num = int(dados[i]['numero'])
            cor_post = str(dados[i+1]['cor']).upper()
            if 0 <= num <= 14 and cor_post in ['V', 'P', 'B']:
                self.unidade_analise[num]["ocorrencias"] += multiplicador_peso
                self.unidade_analise[num][cor_post] += multiplicador_peso
                self.unidade_analise[num][f"pos_numero_{cor_post}"] += multiplicador_peso
                self.unidade_analise[num]["ultimas_cores"].append(cor_post)
                if len(self.unidade_analise[num]["ultimas_cores"]) > 10:
                    self.unidade_analise[num]["ultimas_cores"].pop(0)
                    
        for i in range(total_dados - 1):
            numero_atual = int(dados[i]['numero'])
            proximo_numero = int(dados[i + 1]['numero'])
            if 0 <= numero_atual <= 14 and 0 <= proximo_numero <= 14:
                fator_temporal = 1.0 + ((i / total_dados) * 1.5) if treinamento_profundo and total_dados > 1000 else 1.0
                peso_final = max(1, int(multiplicador_peso * fator_temporal))
                self.transicoes_numericas[numero_atual]["total"] += peso_final
                self.transicoes_numericas[numero_atual]["proximos"][proximo_numero] += peso_final

        for i in range(total_dados - 2):
            num1 = int(dados[i]['numero'])
            num2 = int(dados[i+1]['numero'])
            prox_num = int(dados[i+2]['numero'])
            if num1 != num2 and 0 <= num1 <= 14 and 0 <= num2 <= 14:
                cor_proxima = str(dados[i+2]['cor']).upper()
                if cor_proxima in ['V', 'P', 'B']:
                    chave_dupla = f"{num1}-{num2}"
                    fator_temporal = 1.0 + ((i / total_dados) * 1.5) if treinamento_profundo and total_dados > 1000 else 1.0
                    peso_final = max(1, int(multiplicador_peso * fator_temporal))
                    self.bigramas_numericos[chave_dupla]["total"] += peso_final
                    self.bigramas_numericos[chave_dupla][cor_proxima] += peso_final
                    self.bigramas_numericos[chave_dupla]["prox_numero"][prox_num] += peso_final
                    
        for tam_padrao in [3, 4, 5]:
            for i in range(total_dados - tam_padrao):
                janela_cores = [dados[k]['cor'] for k in range(i, i+tam_padrao)]
                if 'B' not in janela_cores:
                    padrao_str = "".join(janela_cores)
                    ultimo_num = dados[i+tam_padrao-1]['numero']
                    penultimo_num = dados[i+tam_padrao-2]['numero']
                    cor_post = dados[i+tam_padrao]['cor']
                    chave_num = f"PADRAO_{padrao_str}_{ultimo_num}"
                    chave_bigrama = f"PADRAO_{padrao_str}_{penultimo_num}-{ultimo_num}"
                    fator_temporal = 1.0 + ((i / total_dados) * 1.5) if treinamento_profundo and total_dados > 1000 else 1.0
                    peso_final = max(1, int(multiplicador_peso * fator_temporal))
                    self.padroes_fechamento_numerico[chave_num]["total"] += peso_final
                    self.padroes_fechamento_numerico[chave_num][cor_post] += peso_final
                    self.padroes_fechamento_numerico[chave_bigrama]["total"] += peso_final
                    self.padroes_fechamento_numerico[chave_bigrama][cor_post] += peso_final
                    
        for n in range(15):
            total = self.unidade_analise[n]["ocorrencias"]
            if total > 0:
                self.unidade_analise[n]["freq_v"] = round((self.unidade_analise[n]["V"] / total) * 100, 2)
                self.unidade_analise[n]["freq_p"] = round((self.unidade_analise[n]["P"] / total) * 100, 2)
                self.unidade_analise[n]["freq_b"] = round((self.unidade_analise[n]["B"] / total) * 100, 2)
                self.unidade_analise[n]["comportamento_dominante"] = self._calcular_comportamento_dominante(n)
                self.unidade_analise[n]["estabilidade"] = self._calcular_estabilidade(n)
                self.unidade_analise[n]["enfraquecimento"] = self._calcular_enfraquecimento(n)
                self.unidade_analise[n]["saturacao"] = self._calcular_saturacao(n)

    def _calcular_comportamento_dominante(self, num):
        freq_v = self.unidade_analise[num]["freq_v"]
        freq_p = self.unidade_analise[num]["freq_p"]
        if freq_v > freq_p + 8: return "VERMELHO"
        elif freq_p > freq_v + 8: return "PRETO"
        return "NEUTRO"

    def _calcular_estabilidade(self, num):
        ultimas = self.unidade_analise[num]["ultimas_cores"]
        if len(ultimas) < 5: return "NEUTRO"
        dominante = self.unidade_analise[num]["comportamento_dominante"]
        if dominante == "NEUTRO": return "NEUTRO"
        count = sum(1 for c in ultimas if c == ('V' if dominante == "VERMELHO" else 'P'))
        taxa = count / len(ultimas)
        if taxa >= 0.7: return "ESTÁVEL"
        elif taxa <= 0.4: return "INSTÁVEL"
        return "NEUTRO"

    def _calcular_enfraquecimento(self, num):
        return "ENFRAQUECIDO" if self.unidade_analise[num]["estabilidade"] == "INSTÁVEL" else "ESTÁVEL"

    def _calcular_saturacao(self, num):
        total = self.unidade_analise[num]["ocorrencias"]
        if total > 800: return "ALTA"
        elif total > 400: return "MÉDIA"
        return "BAIXA"

    def injetar_aprendizado_imediato(self, sub_dados, multiplicador_peso=4, analise_contexto=None, salvar_na_recencia=True):
        if salvar_na_recencia:
            self.dados_recencia.extend(sub_dados)
        self._processar_bloco_dados(sub_dados, multiplicador_peso, True)
        if hasattr(self, "markov_ordens") and sub_dados:
            cores_markov = [str(d.get("cor", "B")).upper() for d in sub_dados]
            for ordem in range(1, min(6, len(cores_markov) - 1) + 1):
                for i in range(ordem, len(cores_markov)):
                    estado = tuple(cores_markov[i-ordem:i])
                    proxima = cores_markov[i]
                    if proxima not in ("V", "P", "B"):
                        continue
                    stats_markov = self.markov_ordens[ordem][estado]
                    stats_markov[proxima] += multiplicador_peso
                    stats_markov["total"] += multiplicador_peso
        if analise_contexto:
            for regra in analise_contexto.get("regras_posicionais", []):
                self.historico_regras[regra.get("tipo_regra", "DESCONHEVIDO")]["total"] += 1

    def registrar_padrao_vencedor(self, analise_contexto, resultado):
        if resultado not in ["G0", "G1"]: return
        padrao = {
            "geometria": analise_contexto.get("geometria"),
            "regras_ativas": [r.get("tipo_regra") for r in analise_contexto.get("regras_posicionais", [])],
            "controlador_dominante": analise_contexto.get("controlador_retardador", {}).get("dominancia"),
            "modo_mercado": analise_contexto.get("contexto_avancado", {}).get("modo_mercado"),
            "entropia_shannon": analise_contexto.get("entropia_shannon", 0.0),
            "monte_carlo_indicou": analise_contexto.get("monte_carlo_indicou"),
            "resultado": resultado,
            "peso": 1
        }
        if padrao not in self.memoria_padroes_vencedores:
            self.memoria_padroes_vencedores.append(padrao)
        if len(self.memoria_padroes_vencedores) > 50:
            self.memoria_padroes_vencedores.pop(0)

    def calcular_bonus_memoria(self, analise_contexto):
        bonus = 0
        entropia_atual = analise_contexto.get("entropia_shannon", 0.0)
        for padrao in self.memoria_padroes_vencedores:
            match = 0
            if padrao.get("geometria") == analise_contexto.get("geometria"): match += 8
            comuns = set(padrao.get("regras_ativas", [])) & set([r.get("tipo_regra") for r in analise_contexto.get("regras_posicionais", [])])
            match += len(comuns) * 3
            if padrao.get("entropia_shannon", 0.0) > 0:
                if abs(padrao.get("entropia_shannon", 0.0) - entropia_atual) <= 0.2: match += 5
            if match >= 12: bonus += 4
        return min(bonus, 22)

    def _calcular_z_score(self, sucessos, tentativas, probabilidade_esperada=0.4667):
        if tentativas == 0: return 0.0
        proporcao_real = sucessos / tentativas
        desvio_padrao = math.sqrt((probabilidade_esperada * (1 - probabilidade_esperada)) / tentativas)
        if desvio_padrao == 0: return 0.0
        z_score = (proporcao_real - probabilidade_esperada) / desvio_padrao
        return z_score

    def verificar_quebrador_historico(self, sub_num, sub_pol):
        if len(sub_num) < 3: return False, "NEUTRO", ""
        ultimo_num = sub_num[-1]
        for tam in range(10, 2, -1):
            if len(sub_pol) < tam: continue
            janela_cores = sub_pol[-tam:]
            janela_str = "-".join(janela_cores)
            eh_streak = len(set(janela_cores)) == 1
            eh_xadrez = all(janela_cores[j] != janela_cores[j-1] for j in range(1, tam))
            eh_duplo = False
            if tam >= 4 and tam % 2 == 0:
                metade = tam // 2
                if len(set(janela_cores[:metade])) == 1 and len(set(janela_cores[metade:])) == 1 and janela_cores[0] != janela_cores[metade]:
                    eh_duplo = True
            eh_espelho_normal = all(janela_cores[j] == janela_cores[tam-1-j] for j in range(tam)) and not eh_streak
            eh_espelho_invertido = all(janela_cores[j] != janela_cores[tam-1-j] for j in range(tam)) and not eh_xadrez
            if eh_streak: tipo_prefix = f"STREAK_{tam}"
            elif eh_xadrez: tipo_prefix = f"XADREZ_{tam}"
            elif eh_duplo: tipo_prefix = f"DUPLO_{tam}"
            elif eh_espelho_normal: tipo_prefix = f"ESPELHO_NORMAL_{tam}"
            elif eh_espelho_invertido: tipo_prefix = f"ESPELHO_INVERTIDO_{tam}"
            else: tipo_prefix = f"PADRAO_GERAL_{tam}"
            chave = f"{tipo_prefix} [{janela_str}]"
            info = self.padroes_gerais_detalhado.get(chave)
            if info:
                info = self._garantir_quebradores_defaultdict(info)
                quebras = info["quebradores"].get(ultimo_num, 0)
                if quebras >= 2 and (quebras / info["total"]) >= 0.4:
                    direcao_inversao = "PRETO" if janela_cores[-1] == "V" else "VERMELHO"
                    return True, direcao_inversao, f"Quebrador Histórico Detectado: Número {ultimo_num} rompe o {chave}"
        return False, "NEUTRO", ""

    def simular_rotas_proximos_resultados(self, sub_num, sub_pol, limite_rotas=6):
        if len(sub_num) < 3 or len(sub_pol) < 3:
            return {"ativo": False, "direcao": "NEUTRO", "peso": 0.0, "rotas": []}

        ultimo = int(sub_num[-1])
        penultimo = int(sub_num[-2])
        candidatos = defaultdict(float)
        suporte_numero = 0.0
        suporte_bigrama = 0.0

        trans_num = getattr(self, "transicoes_numericas", {}).get(ultimo)
        if trans_num:
            suporte_numero = float(trans_num.get("total", 0.0))
            if suporte_numero > 0:
                for numero, qtd in trans_num.get("proximos", {}).items():
                    candidatos[int(numero)] += 0.55 * (float(qtd) / suporte_numero)

        chave_bi_atual = f"{penultimo}-{ultimo}"
        stats_bi_atual = getattr(self, "bigramas_numericos", {}).get(chave_bi_atual)
        if stats_bi_atual:
            suporte_bigrama = float(stats_bi_atual.get("total", 0.0))
            if suporte_bigrama > 0:
                for numero, qtd in stats_bi_atual.get("prox_numero", {}).items():
                    candidatos[int(numero)] += 0.45 * (float(qtd) / suporte_bigrama)

        if not candidatos:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0, "rotas": [],
                "motivo": "SEM_DISTRIBUICAO_NUMERICA"
            }

        deriva_rota_y = {
            "aplicada": False,
            "numero_referencia": ultimo,
            "estado": "SEM_SUPORTE",
            "multiplicador_vermelho": 1.0,
            "multiplicador_preto": 1.0
        }
        try:
            matriz_deriva = getattr(self, "matriz_deriva_comportamental", {}) or {}
            if not matriz_deriva.get("ativo"):
                matriz_deriva = self.mapear_deriva_comportamental_numeros()

            deriva_numero = (
                (matriz_deriva.get("numeros", {}) or {}).get(ultimo)
                or (matriz_deriva.get("numeros", {}) or {}).get(str(ultimo))
                or {}
            )
            estado_deriva = str(deriva_numero.get("estado", "SEM_SUPORTE")).upper()
            deriva_rota_y["estado"] = estado_deriva

            if estado_deriva in (
                "ENFRAQUECENDO",
                "MUDANCA_COMPORTAMENTAL",
                "INVERSAO_COMPORTAMENTAL"
            ):
                horizontes_deriva = deriva_numero.get("horizontes_recencia", {}) or {}
                pesos_horizonte = ((200, 0.15), (100, 0.20), (50, 0.30), (25, 0.35))
                soma_pesos = 0.0
                taxa_v_recente = 0.0
                taxa_p_recente = 0.0

                for horizonte, peso_h in pesos_horizonte:
                    medicao = (
                        horizontes_deriva.get(str(horizonte))
                        or horizontes_deriva.get(horizonte)
                        or {}
                    )
                    suporte_h = int(medicao.get("total", 0) or 0)
                    suporte_minimo_h = {200: 8, 100: 6, 50: 4, 25: 3}[horizonte]
                    if suporte_h < suporte_minimo_h:
                        continue
                    taxas_h = medicao.get("taxas", {}) or {}
                    taxa_v_recente += float(taxas_h.get("V", 0.0)) * peso_h
                    taxa_p_recente += float(taxas_h.get("P", 0.0)) * peso_h
                    soma_pesos += peso_h

                macro = deriva_numero.get("macro", {}) or {}
                taxas_macro = macro.get("taxas", {}) or {}
                macro_v = float(taxas_macro.get("V", 0.0))
                macro_p = float(taxas_macro.get("P", 0.0))

                if soma_pesos > 0 and int(macro.get("total", 0) or 0) >= 30:
                    taxa_v_recente /= soma_pesos
                    taxa_p_recente /= soma_pesos

                    mult_v = taxa_v_recente / max(macro_v, 0.01)
                    mult_p = taxa_p_recente / max(macro_p, 0.01)
                    mult_v = max(0.65, min(1.35, mult_v))
                    mult_p = max(0.65, min(1.35, mult_p))

                    for numero in list(candidatos):
                        numero_int = int(numero)
                        if 1 <= numero_int <= 7:
                            candidatos[numero] *= mult_v
                        elif 8 <= numero_int <= 14:
                            candidatos[numero] *= mult_p

                    deriva_rota_y.update({
                        "aplicada": True,
                        "taxa_vermelho_recente": round(taxa_v_recente, 6),
                        "taxa_preto_recente": round(taxa_p_recente, 6),
                        "taxa_vermelho_macro": round(macro_v, 6),
                        "taxa_preto_macro": round(macro_p, 6),
                        "multiplicador_vermelho": round(mult_v, 6),
                        "multiplicador_preto": round(mult_p, 6)
                    })
        except Exception:
            deriva_rota_y["estado"] = "FALLBACK_MACRO"

        total_massa = sum(candidatos.values())
        if total_massa <= 0:
            return {"ativo": False, "direcao": "NEUTRO", "peso": 0.0, "rotas": []}
        for numero in list(candidatos):
            candidatos[numero] /= total_massa

        ordenados = sorted(candidatos.items(), key=lambda x: x[1], reverse=True)
        selecionados = []
        massa = 0.0
        for numero, prob in ordenados:
            if len(selecionados) >= int(limite_rotas):
                break
            if prob < 0.025 and massa >= 0.75:
                break
            selecionados.append((numero, prob))
            massa += prob
            if massa >= 0.85 and len(selecionados) >= 3:
                break

        score_v_total = 0.0
        score_p_total = 0.0
        rotas = []

        for numero_y, prob_rota in selecionados:
            cor_y = "B" if numero_y == 0 else ("V" if 1 <= numero_y <= 7 else "P")
            score_v = 0.0
            score_p = 0.0
            evidencias = []

            if cor_y == "V":
                score_v += 1.50
            elif cor_y == "P":
                score_p += 1.50
            else:
                score_v += 0.75
                score_p += 0.75
            evidencias.append(f"G0_NUMERO_{numero_y}_{cor_y}")

            chave_tri = f"{penultimo}-{ultimo}-{numero_y}"
            st_tri = getattr(self, "estatisticas_trigramas_globais", {}).get(chave_tri)
            if st_tri and int(st_tri.get("total", 0)) >= 5:
                total = float(st_tri["total"])
                tv = (float(st_tri.get("V_g0", 0)) + float(st_tri.get("V_g1", 0))) / total
                tp = (float(st_tri.get("P_g0", 0)) + float(st_tri.get("P_g1", 0))) / total
                score_v += tv * 0.90
                score_p += tp * 0.90
                evidencias.append(f"TRI_{chave_tri}_V{tv:.3f}_P{tp:.3f}")

            chave_bi = f"{ultimo}-{numero_y}"
            st_bi = getattr(self, "estatisticas_bigramas_globais", {}).get(chave_bi)
            if st_bi and int(st_bi.get("total", 0)) >= 5:
                total = float(st_bi["total"])
                tv = (float(st_bi.get("V_g0", 0)) + float(st_bi.get("V_g1", 0))) / total
                tp = (float(st_bi.get("P_g0", 0)) + float(st_bi.get("P_g1", 0))) / total
                score_v += tv * 0.75
                score_p += tp * 0.75
                evidencias.append(f"BI_{chave_bi}_V{tv:.3f}_P{tp:.3f}")

            st_num = self.unidade_analise.get(numero_y, {})
            ocorrencias = float(st_num.get("ocorrencias", 0.0))
            if ocorrencias >= 5:
                tv = float(st_num.get("freq_v", 0.0)) / 100.0
                tp = float(st_num.get("freq_p", 0.0)) / 100.0
                score_v += tv * 0.65
                score_p += tp * 0.65
                evidencias.append(f"NUM_{numero_y}_V{tv:.3f}_P{tp:.3f}")

            pol_simulada = (list(sub_pol) + [cor_y])[-12:]
            markov_sim = self.calcular_probabilidade_exata_markov(pol_simulada)
            score_v += (float(markov_sim.get("V", 0.0)) / 100.0) * 0.55
            score_p += (float(markov_sim.get("P", 0.0)) / 100.0) * 0.55
            evidencias.append(
                f"MARKOV_POS_Y_V{float(markov_sim.get('V', 0.0)):.2f}_P{float(markov_sim.get('P', 0.0)):.2f}"
            )

            num_simulada = (list(sub_num) + [numero_y])[-12:]
            geo_simulada = AnalisadorContextoAvancado.mapear_padroes_geometria(pol_simulada)
            regras_simuladas = MotorContagensProjetivas.mapear_janela(
                num_simulada, pol_simulada, geo_simulada, None
            )
            for regra in regras_simuladas:
                peso_txt = str(regra.get("peso", "MEDIO")).upper()
                peso_regra = {"BAIXO": 0.15, "MEDIO": 0.30, "MÉDIO": 0.30, "ALTO": 0.45}.get(peso_txt, 0.30)
                if regra.get("direcao") == "VERMELHO":
                    score_v += peso_regra
                elif regra.get("direcao") == "PRETO":
                    score_p += peso_regra
            if regras_simuladas:
                evidencias.append(
                    "REGRAS_POS_Y=" + ",".join(r.get("tipo_regra", "") for r in regras_simuladas)
                )

            voto_padrao_y = self.obter_voto_padrao_contextual(
                num_simulada, pol_simulada
            )
            if voto_padrao_y.get("ativo"):
                margem_ctx = float(voto_padrao_y.get("margem", 0.0))
                forca_ctx = min(1.10, 0.35 + (margem_ctx * 4.0))
                if voto_padrao_y.get("direcao") == "VERMELHO":
                    score_v += forca_ctx
                elif voto_padrao_y.get("direcao") == "PRETO":
                    score_p += forca_ctx
                evidencias.append(
                    f"PADRAO_CTX_POS_Y_{voto_padrao_y.get('direcao', 'NEUTRO')}_"
                    f"M{margem_ctx:.4f}_CTX{int(voto_padrao_y.get('contextos', 0))}"
                )

            voto_regra_y = self.obter_voto_regra_contextual(
                num_simulada, pol_simulada
            )
            if voto_regra_y.get("ativo"):
                margem_regra = float(voto_regra_y.get("margem", 0.0))
                forca_regra = min(1.10, 0.35 + (margem_regra * 4.0))
                if voto_regra_y.get("direcao") == "VERMELHO":
                    score_v += forca_regra
                elif voto_regra_y.get("direcao") == "PRETO":
                    score_p += forca_regra
                evidencias.append(
                    f"REGRA_CTX_POS_Y_{voto_regra_y.get('direcao', 'NEUTRO')}_"
                    f"M{margem_regra:.4f}_CTX{int(voto_regra_y.get('contextos', 0))}"
                )

            score_v_total += prob_rota * score_v
            score_p_total += prob_rota * score_p
            rotas.append({
                "numero_g0_possivel": numero_y,
                "cor_g0_possivel": cor_y,
                "probabilidade_rota_percent": round(prob_rota * 100.0, 2),
                "score_vermelho": round(score_v, 4),
                "score_preto": round(score_p, 4),
                "direcao_rota": "VERMELHO" if score_v > score_p else ("PRETO" if score_p > score_v else "NEUTRO"),
                "evidencias": evidencias
            })

        soma_scores = score_v_total + score_p_total
        if soma_scores <= 0:
            direcao = "NEUTRO"
            margem = 0.0
        else:
            prob_v = score_v_total / soma_scores
            prob_p = score_p_total / soma_scores
            margem = abs(prob_v - prob_p)
            if margem < 0.035:
                direcao = "NEUTRO"
            else:
                direcao = "VERMELHO" if prob_v > prob_p else "PRETO"

        if direcao == "NEUTRO":
            peso = 0.0
        elif margem >= 0.14 and massa >= 0.65:
            peso = 3.0
        elif margem >= 0.08:
            peso = 2.0
        else:
            peso = 1.0

        return {
            "ativo": True,
            "direcao": direcao,
            "peso": peso,
            "score_vermelho": round(score_v_total, 6),
            "score_preto": round(score_p_total, 6),
            "margem": round(margem, 6),
            "massa_rotas_analisada_percent": round(massa * 100.0, 2),
            "suporte_numero": round(suporte_numero, 2),
            "suporte_bigrama": round(suporte_bigrama, 2),
            "deriva_temporal_rotas_y": deriva_rota_y,
            "rotas": rotas,
            "altera_regras": False,
            "altera_recencia": False
        }
