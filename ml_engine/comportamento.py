class ComportamentoMixin:
    def analisar_comportamento_pos_numero_recencia(self, dados_recencia):
        if not dados_recencia or len(dados_recencia) < 30:
            return {"mensagem": "Base muito pequena"}
        relatorio = {}
        dados_ativos = list(dados_recencia)[-200:]
        horizontes = (200, 100, 50, 25)

        def medir(bloco, numero, deslocamento):
            cont = {"V": 0, "P": 0, "B": 0}
            total = 0
            for i in range(len(bloco) - deslocamento):
                if int(bloco[i].get("numero", -1)) != numero:
                    continue
                cor = str(bloco[i + deslocamento].get("cor", "B")).upper()
                if cor in cont:
                    cont[cor] += 1
                    total += 1
            return {
                "total": total,
                "V": round(100.0 * cont["V"] / total, 2) if total else 0.0,
                "P": round(100.0 * cont["P"] / total, 2) if total else 0.0,
                "B": round(100.0 * cont["B"] / total, 2) if total else 0.0
            }

        for num in range(15):
            por_horizonte = {}
            for h in horizontes:
                bloco = dados_ativos[-h:]
                por_horizonte[str(h)] = {
                    "+1": medir(bloco, num, 1),
                    "+2": medir(bloco, num, 2),
                    "+3": medir(bloco, num, 3)
                }
            base = por_horizonte["200"]["+1"]
            if base["total"] == 0:
                continue
            cores = {"VERMELHO": base["V"], "PRETO": base["P"], "BRANCO": base["B"]}
            cor_dominante = max(cores, key=cores.get)
            freq = float(cores[cor_dominante])
            relatorio[num] = {
                "total_aparicoes_recencia": base["total"],
                "cor_mais_frequente_apos": cor_dominante,
                "frequencia_cor_dominante_%": round(freq, 2),
                "tendencia_recente": "FORTE" if freq >= 65 else ("MODERADA" if freq >= 55 else "FRACA"),
                "trajetoria_200_100_50_25": por_horizonte
            }
        return relatorio

    def analisar_comportamento_pos_numero(self):
        relatorio = {}
        for num in range(15):
            dados = getattr(self, "unidade_analise", {}).get(num, {})
            total = dados.get("ocorrencias", 0)
            if total == 0: continue
            cores_pos = {"VERMELHO": dados.get("pos_numero_V", 0), "PRETO": dados.get("pos_numero_P", 0), "BRANCO": dados.get("pos_numero_B", 0)}
            cor_dominante = max(cores_pos, key=cores_pos.get)
            freq_dominante = round((cores_pos[cor_dominante] / total) * 100, 2)
            ultimas = dados.get("ultimas_cores", [])
            if len(ultimas) >= 8:
                ultimas_dominantes = sum(1 for c in ultimas if c == ('V' if cor_dominante == "VERMELHO" else 'P'))
                taxa_ultimas = ultimas_dominantes / len(ultimas)
                if taxa_ultimas < 0.5: tendencia = "EM MUDANÇA / SATURAÇÃO POSSÍVEL"
                elif taxa_ultimas >= 0.75: tendencia = "ESTÁVEL"
                else: tendencia = "MODERADO"
            else: tendencia = "DADOS INSUFICIENTES"
            relatorio[num] = {
                "total_aparicoes": total, "cor_mais_frequente_apos": cor_dominante,
                "frequencia_cor_dominante_%": freq_dominante, "distribuicao_pos": cores_pos,
                "comportamento_dominante": dados.get("comportamento_dominante", "NEUTRO"), "estabilidade": dados.get("estabilidade", "NEUTRO"),
                "saturacao": dados.get("saturacao", "NORMAL"), "tendencia_recente": tendencia
            }
        return relatorio
