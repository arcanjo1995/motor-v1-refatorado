# rules/contagens.py

class MotorContagensProjetivas:
    """
    Motor oficial de contagens e cenários estruturais.

    Regras são EVIDÊNCIAS contextuais, não verdades absolutas. O mesmo número
    pode produzir efeitos diferentes em estruturas diferentes. Todas as
    evidências válidas coexistem e seguem para a arbitragem hierárquica.
    """

    REGRAS_PROJECAO = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    CONTINUIDADE_PRETA = {(8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,13), (13,12), (12,11), (11,10)}
    CONTINUIDADE_VERMELHA = {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,6), (6,5), (5,4), (4,3)}

    @staticmethod
    def _adicionar(lista, direcao, tipo, origem, peso="MEDIO", familia="CONTAGENS_OFICIAIS", **extras):
        item = {
            "direcao": direcao, "tipo_regra": tipo, "origem": origem,
            "peso": peso, "familia": familia
        }
        item.update(extras)
        lista.append(item)

    @classmethod
    def _mapear_contagens(cls, sub_num, sub_pol):
class MotorContagensProjetivas:
    """
    Motor oficial de contagens e cenários estruturais.

    Regras são EVIDÊNCIAS contextuais, não verdades absolutas. O mesmo número
    pode produzir efeitos diferentes em estruturas diferentes. Todas as
    evidências válidas coexistem e seguem para a arbitragem hierárquica.
    """

    REGRAS_PROJECAO = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    CONTINUIDADE_PRETA = {(8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,13), (13,12), (12,11), (11,10)}
    CONTINUIDADE_VERMELHA = {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,6), (6,5), (5,4), (4,3)}

    @staticmethod
    def _adicionar(lista, direcao, tipo, origem, peso="MEDIO", familia="CONTAGENS_OFICIAIS", **extras):
        item = {
            "direcao": direcao, "tipo_regra": tipo, "origem": origem,
            "peso": peso, "familia": familia
        }
        item.update(extras)
        lista.append(item)

    @classmethod
    def _mapear_contagens(cls, sub_num, sub_pol):
        contagens = []
        n = len(sub_num)
        for pos, numero in enumerate(sub_num):
            if numero not in cls.REGRAS_PROJECAO:
                continue
            casas_exigidas = cls.REGRAS_PROJECAO[numero] - 1
            fechamento = pos + casas_exigidas
            expectativa_g0 = fechamento + 1
            expectativa_g1 = fechamento + 2
            caminho_fim = min(fechamento, n - 1)
            tem_branco_estrutura = any(sub_num[k] == 0 for k in range(pos + 1, caminho_fim + 1))

            if fechamento >= n:
                status = "ABERTA"
            else:
                status = "FECHADA"
                if expectativa_g0 < n and sub_pol[expectativa_g0] in ("V", "B"):
                    status = "PAGA"
                elif expectativa_g1 < n and sub_pol[expectativa_g1] in ("V", "B"):
                    status = "PAGA"
                elif expectativa_g1 < n:
                    status = "MORTA"
                else:
                    status = "VIVA"

            contagens.append({
                "numero": numero, "origem_idx": pos, "origem_posicao": pos + 1,
                "casas_exigidas": casas_exigidas, "fechamento_idx": fechamento,
                "fechamento_posicao": fechamento + 1, "expectativa_g0_idx": expectativa_g0,
                "expectativa_g1_idx": expectativa_g1, "status": status,
                "tem_branco_estrutura": tem_branco_estrutura, "assumida_por": None,
                "coexistente": False, "transicional": False
            })

        # Coexistência: duas ou mais contagens ABERTAS/VIVAS com alcance temporal sobreposto.
        relevantes = [c for c in contagens if c["status"] in ("ABERTA", "VIVA")]
        if len(relevantes) >= 2:
            for c in relevantes:
                c["coexistente"] = True

        # Transição, chance dupla, finalização conjunta e assunção.
        for atual in contagens:
            for nova in contagens:
                if nova["origem_idx"] <= atual["origem_idx"]:
                    continue
                if nova["origem_idx"] == atual["fechamento_idx"]:
                    atual["transicional"] = True
                    nova["transicional"] = True
                if atual["status"] in ("FECHADA", "VIVA") and nova["origem_idx"] in (atual["expectativa_g0_idx"], atual["expectativa_g1_idx"]):
                    atual["assumida_por"] = nova["numero"]

        return contagens

    @classmethod
    def mapear_janela(cls, sub_num, sub_pol, geometry_mercado, ia_modelo=None):
        lista_bruta = []
        if len(sub_num) < 12 or len(sub_pol) < 12:
            return lista_bruta

        sub_num = [int(x) for x in sub_num]
        sub_pol = [str(x).upper() for x in sub_pol]
        contagens = cls._mapear_contagens(sub_num, sub_pol)

        # CONTAGENS PROJETIVAS OFICIAIS 1..7. Somente após o fechamento a
        # expectativa vermelha é ativa; validade operacional G0 até G1.
        for c in contagens:
            if c["tem_branco_estrutura"]:
                continue
            if c["status"] in ("FECHADA", "VIVA") and c["expectativa_g0_idx"] in (11, 12):
                cls._adicionar(
                    lista_bruta, "VERMELHO", f"V3_ATIVADOR_{c['numero']}",
                    "Contagem Projetiva Oficial", "ALTO" if c["numero"] in (2, 3) else "MEDIO",
                    "CONTAGENS_PROJETIVAS", status_contagem=c["status"],
                    origem_posicao=c["origem_posicao"], fechamento_posicao=c["fechamento_posicao"],
                    validade="G0_G1", coexistente=c["coexistente"], transicional=c["transicional"]
                )

        # REGRA OFICIAL DO NÚMERO 4 — quatro cenários pretos contextuais.
        cenarios_4 = []
        if sub_pol[-2] == "P" and sub_num[-1] == 4:
            cenarios_4.append("CENARIO_1_P_4")
        if sub_pol[-3] == "P" and sub_num[-2] == 4 and sub_pol[-1] == "P":
            cenarios_4.append("CENARIO_2_P_4_P")
        if sub_num[-3] == 4 and sub_pol[-2:] == ["P", "P"]:
            cenarios_4.append("CENARIO_3_4_P_P")
        if sub_num[-2] == 4 and sub_pol[-1] == "P":
            cenarios_4.append("CENARIO_4_4_P")
        for cenario in cenarios_4:
            cls._adicionar(lista_bruta, "PRETO", f"REGRA_4_{cenario}",
                           "Regra Oficial Número 4", "MEDIO", "REGRA_OFICIAL_4",
                           validade="G0_G1", contexto="CONTINUIDADE_PRETA_ESTRUTURAL")

        # REGRA OFICIAL DO NÚMERO 10 — ativador estrutural preto no fechamento.
        if sub_num[-1] == 10:
            saturacao_extrema = len(set(sub_pol[-4:])) == 1
            continuidade_preta = any((sub_num[i-1], sub_num[i]) in cls.CONTINUIDADE_PRETA for i in range(1, len(sub_num)))
            cls._adicionar(lista_bruta, "PRETO", "REGRA_10_ATIVADOR_ESTRUTURAL",
                           "Regra Oficial Número 10", "ALTO", "REGRA_OFICIAL_10",
                           validade="G0_G1", validacao_saturacao=not saturacao_extrema,
                           validacao_continuidade=continuidade_preta, validacao_no_call=True)

        # REGRA OFICIAL 5-10 — combinação preta e continuação preta forte.
        if sub_num[-2:] == [5, 10]:
            saturacao_extrema = len(set(sub_pol[-4:])) == 1
            cls._adicionar(lista_bruta, "PRETO", "REGRA_5_10_CENARIO_1",
                           "Regra Oficial 5-10", "ALTO", "REGRA_OFICIAL_5_10",
                           validade="G0_G1", continuidade_intacta=True,
                           saturacao_extrema=saturacao_extrema)
        if sub_num[-3:-1] == [5, 10] and sub_pol[-1] == "P":
            saturacao_extrema = len(set(sub_pol[-4:])) == 1
            cls._adicionar(lista_bruta, "PRETO", "REGRA_5_10_CENARIO_2_CONTINUACAO_FORTE",
                           "Regra Oficial 5-10", "ALTO", "REGRA_OFICIAL_5_10",
                           validade="G0_G1", continuidade_intacta=True,
                           saturacao_extrema=saturacao_extrema, forca="FORTE")

        # CONTINUIDADE NUMÉRICA — percorre posição 1 -> final, sem saltos.
        continuidades = []
        for i in range(1, len(sub_num)):
            par = (sub_num[i - 1], sub_num[i])
            if par in cls.CONTINUIDADE_PRETA:
                continuidades.append((i, "PRETO", par))
            elif par in cls.CONTINUIDADE_VERMELHA:
                continuidades.append((i, "VERMELHO", par))
        # A continuidade que alcança o fechamento da janela mantém polaridade até G1.
        for i, direcao, par in continuidades:
            if i == len(sub_num) - 1:
                cls._adicionar(lista_bruta, direcao,
                               "V2_CONTINUIDADE_PRETA" if direcao == "PRETO" else "V2_CONTINUIDADE_VERMELHA",
                               "Continuidade Numérica Oficial", "ALTO", "CONTINUIDADE_NUMERICA",
                               par=par, validade="G0_G1", fluxo_dominante=direcao)

        # Dinâmicas oficiais entre contagens. São mapeadas sem apagar as
        # expectativas originais. A autoridade direcional imediata, porém, só
        # existe quando a contagem já fechou e sua expectativa G0/G1 alcança o
        # horizonte operacional atual. Contagem ABERTA continua mapeada como
        # conhecimento estrutural, mas não assume o CALL antes do próprio tempo.
        ativas = [c for c in contagens if c["status"] in ("ABERTA", "FECHADA", "VIVA")]
        horizonte_g0_idx = len(sub_num)
        horizonte_g1_idx = len(sub_num) + 1
        operacionais = [
            c for c in ativas
            if (
                c["status"] in ("FECHADA", "VIVA")
                and c["expectativa_g0_idx"] <= horizonte_g1_idx
                and c["expectativa_g1_idx"] >= horizonte_g0_idx
            )
        ]
        operacionais_ordenadas = sorted(
            operacionais,
            key=lambda c: (
                abs(c["expectativa_g0_idx"] - horizonte_g0_idx),
                c["expectativa_g0_idx"],
                -c["origem_idx"]
            )
        )
        dominante = operacionais_ordenadas[0] if operacionais_ordenadas else None
        direcao_dinamica = "VERMELHO" if dominante else "NEUTRO"

        coexistentes = [c for c in ativas if c["coexistente"]]
        if len(coexistentes) >= 2:
            cls._adicionar(lista_bruta, direcao_dinamica, "COEXISTENCIA_CONTAGENS_ATIVA",
                           "Capítulo Coexistências", "ALTO", "DINAMICA_CONTAGENS",
                           contagens=[c["numero"] for c in coexistentes], status="COEXISTENTE")

        transicionais = [c for c in contagens if c["transicional"]]
        if transicionais:
            cls._adicionar(lista_bruta, direcao_dinamica, "TRANSICAO_CONTAGENS_ATIVA",
                           "Capítulo Transições", "ALTO", "DINAMICA_CONTAGENS", status="TRANSICIONAL")

        # Chance dupla: fechamento de uma contagem e nascimento de outra na mesma casa.
        chances_duplas = []
        for a in contagens:
            for b in contagens:
                if a is not b and a["fechamento_idx"] == b["origem_idx"]:
                    chances_duplas.append((a["numero"], b["numero"], b["origem_posicao"]))
        if chances_duplas:
            cls._adicionar(lista_bruta, direcao_dinamica, "CHANCE_DUPLA_ATIVA",
                           "Capítulo Chance Dupla", "ALTO", "DINAMICA_CONTAGENS",
                           eventos=chances_duplas)

        assumidas = [c for c in contagens if c["assumida_por"] is not None]
        if assumidas:
            cls._adicionar(lista_bruta, direcao_dinamica, "ASSUNCAO_CONTAGEM_OFICIAL",
                           "Capítulo Assunção de Contagens", "ALTO", "DINAMICA_CONTAGENS",
                           eventos=[(c["numero"], c["assumida_por"]) for c in assumidas],
                           pergunta_obrigatoria="PAGOU_OU_ASSUMIU")

        finais = {}
        for c in contagens:
            if c["fechamento_idx"] < len(sub_num):
                finais.setdefault(c["fechamento_idx"], []).append(c)
        conjuntas = [grupo for grupo in finais.values() if len(grupo) >= 2]
        if conjuntas:
            cls._adicionar(lista_bruta, direcao_dinamica, "FINALIZACAO_CONJUNTA_ATIVA",
                           "Capítulo Finalização Conjunta", "ALTO", "DINAMICA_CONTAGENS",
                           grupos=[[c["numero"] for c in grupo] for grupo in conjuntas])

        # Hierarquia oficial: fechamento recente -> consequência ativa -> respeito
        # estrutural -> apoio posicional -> consequência futura. Mantém coexistência
        # de evidências; apenas registra qual contagem tem maior autoridade.
        if dominante is not None:
            cls._adicionar(lista_bruta, "VERMELHO", f"HIERARQUIA_CONTAGEM_{dominante['numero']}",
                           "Hierarquia entre Contagens", "ALTO", "HIERARQUIA_CONTAGENS",
                           contagem_dominante=dominante["numero"],
                           fechamento_recente=dominante["fechamento_posicao"],
                           status_contagem=dominante["status"], validade="G0_G1")

        if ia_modelo and hasattr(ia_modelo, 'verificar_quebrador_historico'):
            eh_quebrador, dir_quebra, motivo_quebra = ia_modelo.verificar_quebrador_historico(sub_num, sub_pol)
            if eh_quebrador:
                cls._adicionar(lista_bruta, dir_quebra, "QUEBRADOR_HISTORICO_ATIVO",
                               "IA_Quebradores", "ALTO", "QUEBRADOR_HISTORICO",
                               motivo=motivo_quebra)

        return lista_bruta

        pass
