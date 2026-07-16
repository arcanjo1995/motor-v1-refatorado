class MotorNoCall:
    @staticmethod
    def checar_no_call(sub_num, sub_pol):
        cenarios_duplas = [(7, 8), (8, 9), (9, 10), (10, 11)]
        for idx1, idx2 in cenarios_duplas:
            if sub_num[idx1] == sub_num[idx2]:
                return True, "Volume 2 Cap 6: Trava das Duplas Ativa"
        posicoes_criticas_6 = [5, 8, 9, 10, 11]
        for pos in posicoes_criticas_6:
            if sub_num[pos] == 6:
                return True, "Volume 2 Cap 4: Trava Número 6 (Posição de No Call Ativa)"
        posicoes_criticas_2 = [8, 9, 10, 11]
        for pos in posicoes_criticas_2:
            if sub_num[pos] == 2:
                return True, "Volume 2 Cap 3: Trava Número 2"
        posicoes_criticas_b = [5, 8, 9, 10, 11]
        for pos in posicoes_criticas_b:
            if sub_pol[pos] == "B":
                return True, "Volume 2 Cap 5: Trava do Branco"
        return False, "Evento Neutro Operacional"

    @staticmethod
    def checar_risco_preditivo_g0(sub_num, ia_modelo):
        if len(sub_num) < 2 or not hasattr(ia_modelo, 'bigramas_numericos'):
            return False, ""

        penultimo_num, ultimo_num = sub_num[-2], sub_num[-1]
        chave_dupla = f"{penultimo_num}-{ultimo_num}"
        stats_dupla = ia_modelo.bigramas_numericos.get(chave_dupla)

        if not stats_dupla or stats_dupla.get("total", 0) == 0:
            return False, ""

        total_ocorrencias = int(stats_dupla.get("total", 0) or 0)
        prox_numeros_historico = stats_dupla.get("prox_numero", {}) or {}

        if total_ocorrencias <= 0 or not prox_numeros_historico:
            return False, ""

        BASE_TEORICA_NUMERO = 100.0 / 15.0
        MULTIPLICADOR_ALERTA = 1.50
        MULTIPLICADOR_CRITICO = 2.00
        DOMINANCIA_MINIMA_SOBRE_SEGUNDO = 1.25

        distribuicao = {}
        for numero in range(15):
            ocorrencias = int(prox_numeros_historico.get(numero, 0) or 0)
            distribuicao[numero] = (ocorrencias / total_ocorrencias) * 100.0

        ranking = sorted(distribuicao.items(), key=lambda item: item[1], reverse=True)
        numero_lider, chance_lider = ranking[0]
        chance_segundo = ranking[1][1] if len(ranking) > 1 else 0.0

        elevacao_lider = (chance_lider / BASE_TEORICA_NUMERO) if BASE_TEORICA_NUMERO > 0 else 0.0
        dominancia_lider = (chance_lider / chance_segundo) if chance_segundo > 0 else float("inf")

        chance_6 = distribuicao.get(6, 0.0)
        chance_2 = distribuicao.get(2, 0.0)
        chance_repeticao = distribuicao.get(ultimo_num, 0.0)

        elevacao_6 = chance_6 / BASE_TEORICA_NUMERO
        elevacao_2 = chance_2 / BASE_TEORICA_NUMERO
        elevacao_repeticao = chance_repeticao / BASE_TEORICA_NUMERO

        if elevacao_6 >= MULTIPLICADOR_CRITICO:
            return True, (f"Radar de Ameaça Numérica: Detectado risco crítico de {chance_6:.1f}% do número 6 cair no G0 ({elevacao_6:.2f}x a referência teórica de {BASE_TEORICA_NUMERO:.2f}%).")
        if elevacao_2 >= MULTIPLICADOR_CRITICO:
            return True, (f"Radar de Ameaça Numérica: Detectado risco crítico de {chance_2:.1f}% do número 2 cair no G0 ({elevacao_2:.2f}x a referência teórica de {BASE_TEORICA_NUMERO:.2f}%).")
        if elevacao_repeticao >= MULTIPLICADOR_CRITICO:
            return True, (f"Radar de Ameaça Numérica: Detectado risco de {chance_repeticao:.1f}% do número {ultimo_num} repetir no G0 e formar dupla numérica exata ({elevacao_repeticao:.2f}x a referência teórica).")

        if elevacao_lider >= MULTIPLICADOR_CRITICO and dominancia_lider >= DOMINANCIA_MINIMA_SOBRE_SEGUNDO:
            try:
                ia_modelo._ultimo_radar_numerico_contextual = {
                    "ativo": True,
                    "numero_lider": numero_lider,
                    "chance": round(chance_lider,2),
                    "elevacao": round(elevacao_lider,2),
                    "dominancia": round(dominancia_lider,2),
                }
            except:
                pass
            return False, ""

        if numero_lider in (2, 6, ultimo_num) and elevacao_lider >= MULTIPLICADOR_ALERTA:
            tipo_ameaca = "dupla numérica exata" if numero_lider == ultimo_num else f"número instável {numero_lider}"
            return True, (f"Radar de Ameaça Numérica: Concentração relevante no G0 para {tipo_ameaca}: {chance_lider:.1f}% ({elevacao_lider:.2f}x a referência teórica de {BASE_TEORICA_NUMERO:.2f}%).")
        return False, ""
