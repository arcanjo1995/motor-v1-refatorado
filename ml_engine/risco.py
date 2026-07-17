import math
from collections import defaultdict
from utils.helpers import hash_chave
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from config.settings import VERSAO_CHAVES_HASH


class RiscoMixin:
    """
    Mixin para Filtros Discriminativos, Risco G2+ e Conflitos.
    """

    # ============================================================
    # MÉTODOS EXISTENTES (mantidos)
    # ============================================================

    def _chave_conflito(self, expectations, geometria, modo_mercado, probabilidade_markov):
        peso_v = 0
        peso_p = 0
        for item in expectations or []:
            peso = 3 if any(k in item.get("tipo_regra", "") for k in ["CONTINUIDADE", "ASSUNCAO", "QUEBRADOR"]) else 1
            if item.get("direcao") == "VERMELHO":
                peso_v += peso
            elif item.get("direcao") == "PRETO":
                peso_p += peso
        direcao_pos = "V" if peso_v > peso_p else ("P" if peso_p > peso_v else "N")

        direcao_geo = "N"
        if geometria == "CICLO_FECHADO_PVVP":
            direcao_geo = "V"
        elif geometria == "CICLO_FECHADO_VPPV":
            direcao_geo = "P"
        elif geometria == "SATURAÇÃO ESTRUTURAL (V)":
            direcao_geo = "P"
        elif geometria == "SATURAÇÃO ESTRUTURAL (P)":
            direcao_geo = "V"

        mv = float((probabilidade_markov or {}).get("V", 0.0))
        mp = float((probabilidade_markov or {}).get("P", 0.0))
        direcao_markov = "V" if mv > mp else ("P" if mp > mv else "N")
        faixa_markov = int(min(20, abs(mv - mp)) // 2)
        return hash_chave(f"M={direcao_markov}:{faixa_markov}|POS={direcao_pos}|GEO={direcao_geo}|REG={modo_mercado}")

    def registrar_resultado_conflito(self, expectations, geometria, modo_mercado, probabilidade_markov, correcoes):
        if not correcoes:
            return
        chave = self._chave_conflito(expectations, geometria, modo_mercado, probabilidade_markov)
        stats = self.memoria_conflitos[chave]
        stats["total"] += 1

        c0 = correcoes[0] if len(correcoes) > 0 else None
        c1 = correcoes[1] if len(correcoes) > 1 else None
        if c0 in ("V", "B"):
            stats["V_g0"] += 1
            stats["V_g0g1"] += 1
        elif c1 in ("V", "B"):
            stats["V_g0g1"] += 1
        else:
            stats["falhas_v"] += 1

        if c0 in ("P", "B"):
            stats["P_g0"] += 1
            stats["P_g0g1"] += 1
        elif c1 in ("P", "B"):
            stats["P_g0g1"] += 1
        else:
            stats["falhas_p"] += 1

    def obter_voto_contextual(self, expectations, geometria, modo_mercado, probabilidade_markov):
        if not hasattr(self, "memoria_conflitos"):
            return {"direcao": "NEUTRO", "peso": 0.0, "total": 0}
        chave = self._chave_conflito(expectations, geometria, modo_mercado, probabilidade_markov)
        stats = self.memoria_conflitos.get(chave)
        if not stats or stats.get("total", 0) < 30:
            return {"direcao": "NEUTRO", "peso": 0.0, "total": stats.get("total", 0) if stats else 0}

        total = stats["total"]
        taxa_v = stats["V_g0g1"] / total
        taxa_p = stats["P_g0g1"] / total
        taxa_v_g0 = stats["V_g0"] / total
        taxa_p_g0 = stats["P_g0"] / total

        score_v = (taxa_v_g0 * 0.70) + (taxa_v * 0.30)
        score_p = (taxa_p_g0 * 0.70) + (taxa_p * 0.30)
        margem_score = abs(score_v - score_p)
        margem_g0 = abs(taxa_v_g0 - taxa_p_g0)

        if margem_score < 0.035 and margem_g0 < 0.04:
            return {
                "direcao": "NEUTRO", "peso": 0.0, "total": total,
                "taxa_v": taxa_v, "taxa_p": taxa_p,
                "taxa_v_g0": taxa_v_g0, "taxa_p_g0": taxa_p_g0,
                "score_v": score_v, "score_p": score_p,
                "prioridade_g0": True
            }

        direcao = "VERMELHO" if score_v > score_p else "PRETO"
        melhor_taxa = taxa_v if direcao == "VERMELHO" else taxa_p
        melhor_g0 = taxa_v_g0 if direcao == "VERMELHO" else taxa_p_g0
        if margem_score >= 0.08 and total >= 80:
            peso = 4.5
        elif margem_score >= 0.055 and total >= 50:
            peso = 3.5
        else:
            peso = 2.0

        return {
            "direcao": direcao, "peso": peso, "total": total,
            "taxa_v": taxa_v, "taxa_p": taxa_p,
            "taxa_v_g0": taxa_v_g0, "taxa_p_g0": taxa_p_g0,
            "score_v": score_v, "score_p": score_p,
            "margem_score": margem_score, "margem_g0": margem_g0,
            "melhor_taxa_g0_g1": melhor_taxa, "melhor_taxa_g0": melhor_g0,
            "prioridade_g0": True
        }

    def _treinar_memoria_conflitos_base_longa(self, dados):
        self.memoria_conflitos = defaultdict(lambda: {
            "total": 0, "V_g0g1": 0, "P_g0g1": 0,
            "V_g0": 0, "P_g0": 0, "falhas_v": 0, "falhas_p": 0
        })
        if not dados or len(dados) < 15:
            self.memoria_conflitos_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        eventos = 0
        for i in range(11, len(dados) - 2):
            janela = dados[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
            expectativas = MotorContagensProjetivas.mapear_janela(nums, pol, geometria, None)
            modo = self._detectar_regime_temporal(pol)
            prob_markov = self.calcular_probabilidade_exata_markov(pol)
            correcoes = [dados[i+1]["cor"], dados[i+2]["cor"]]
            self.registrar_resultado_conflito(expectativas, geometria, modo, prob_markov, correcoes)
            eventos += 1

        self.memoria_conflitos_metricas = {
            "ativo": True,
            "metodo": "RECONSTRUCAO_CRONOLOGICA_BASE_LONGA",
            "eventos_processados": eventos,
            "contextos_aprendidos": len(self.memoria_conflitos),
            "depende_auditoria_mutavel": False,
            "altera_no_call": False,
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        }

    # ============================================================
    # MÉTODO ADICIONADO (estava faltando)
    # ============================================================
    def _avaliar_risco_cartografia_veto(self, sub_num, sub_pol, direcao):
        """
        MAIN 88 — veto cartográfico discriminativo por famílias independentes.
        Mede diretamente G2/FALHA contra G0/G1 para a direção preliminar,
        compara cada contexto com a linha-base global da própria direção e
        impede que granularidades correlacionadas da mesma estrutura contem
        como confirmações independentes.
        """
        cfg = getattr(self, "filtro_discriminativo_config", {})
        if not bool(cfg.get("cartografia_no_veto", True)):
            return {"ativo": False, "leituras": [], "consultados": 0, "risco_alto": 0}
        if not getattr(self, "cartografia_xls_metricas", {}).get("ativo"):
            return {"ativo": False, "leituras": [], "consultados": 0, "risco_alto": 0}
        if direcao not in ("VERMELHO", "PRETO"):
            return {"ativo": False, "leituras": [], "consultados": 0, "risco_alto": 0}

        letra = "V" if direcao == "VERMELHO" else "P"
        suporte_min = int(cfg.get("suporte_cartografia_minimo", 30))
        risco_alto = float(cfg.get("risco_contexto_alto", 0.22))
        lift_minimo = float(cfg.get("lift_risco_minimo", 0.03))
        razao_minima = float(cfg.get("razao_risco_minima", 1.10))
        leituras = []

        def fonte_da_chave(chave):
            prefixo = str(chave).split("|", 1)[0]
            if prefixo == "NUMERO":
                return "CARTOGRAFIA_NUMERO"
            if prefixo.startswith("STREAK"):
                return "CARTOGRAFIA_STREAK"
            if prefixo.startswith("XADREZ"):
                return "CARTOGRAFIA_XADREZ"
            return "CARTOGRAFIA_PADRAO"

        total_base = 0
        resolveu_base = 0
        for chave_base, st_base in self.cartografia_padroes_xls.items():
            if not str(chave_base).startswith("NUMERO|"):
                continue
            total_ctx_base = int(st_base.get("total", 0))
            total_base += total_ctx_base
            if letra == "V":
                resolveu_base += int(st_base.get("V_g0", 0)) + int(st_base.get("V_g1", 0))
            else:
                resolveu_base += int(st_base.get("P_g0", 0)) + int(st_base.get("P_g1", 0))

        taxa_base = resolveu_base / max(total_base, 1)
        risco_base = max(0.0, min(1.0, 1.0 - taxa_base))

        for chave in self._chaves_cartografia_padrao(sub_num, sub_pol):
            st = self.cartografia_padroes_xls.get(chave)
            if not st:
                continue
            total = int(st.get("total", 0))
            if total < suporte_min:
                continue

            if letra == "V":
                resolveu = int(st.get("V_g0", 0)) + int(st.get("V_g1", 0))
            else:
                resolveu = int(st.get("P_g0", 0)) + int(st.get("P_g1", 0))

            taxa_resolucao = max(0.0, min(1.0, resolveu / max(total, 1)))
            risco = 1.0 - taxa_resolucao
            lift_risco = risco - risco_base
            razao_risco = risco / max(risco_base, 1e-9)
            fonte = fonte_da_chave(chave)

            prefixo = str(chave).split("|", 1)[0]
            if "EXATO" in prefixo:
                especificidade = 1.00
            elif "TRI" in prefixo:
                especificidade = 0.94
            elif "BI" in prefixo:
                especificidade = 0.88
            elif "NUM" in prefixo:
                especificidade = 0.82
            elif prefixo in ("STREAK", "XADREZ"):
                especificidade = 0.76
            elif prefixo == "NUMERO":
                especificidade = 0.68
            else:
                especificidade = 0.60

            peso = (total / (total + float(suporte_min))) * especificidade
            leituras.append({
                "fonte": fonte,
                "familia": fonte,
                "chave": chave,
                "risco": risco,
                "taxa_g0_g1": taxa_resolucao,
                "risco_base": risco_base,
                "lift_risco": lift_risco,
                "razao_risco": razao_risco,
                "suporte": total,
                "peso": max(0.05, peso),
                "tipo": "CARTOGRAFIA"
            })

        # Trajetória projetiva (apenas VERMELHO, pois V3)
        if letra == "V":
            total_proj_base = 0
            resolveu_proj_base = 0
            for stats_proj in self.estatisticas_projecoes_respeito.values():
                total_proj_base += int(stats_proj.get("total", 0))
                resolveu_proj_base += (
                    int(stats_proj.get("respeitada_g0", 0))
                    + int(stats_proj.get("respeitada_g1", 0))
                )
            risco_proj_base = 1.0 - (resolveu_proj_base / max(total_proj_base, 1))

            projecoes_ativas = [
                numero for pos, numero in enumerate(sub_num)
                if 1 <= int(numero) <= 7 and pos + int(numero) in (11, 12)
            ]
            for numero in projecoes_ativas:
                leitura_traj = self._obter_cartografia_projecao_atual(
                    numero, sub_num, sub_pol
                )
                suporte = int(leitura_traj.get("suporte", 0))
                if leitura_traj.get("ativo") and suporte >= suporte_min:
                    taxa_resolucao = max(
                        0.0, min(1.0, float(leitura_traj.get("taxa_respeito", 0.0)))
                    )
                    risco = 1.0 - taxa_resolucao
                    lift_risco = risco - risco_proj_base
                    razao_risco = risco / max(risco_proj_base, 1e-9)
                    peso = suporte / (suporte + float(suporte_min))
                    leituras.append({
                        "fonte": "CARTOGRAFIA_TRAJETORIA",
                        "familia": "CARTOGRAFIA_TRAJETORIA",
                        "chave": f"TRAJETORIA_ATIVA|N={int(numero)}",
                        "risco": risco,
                        "taxa_g0_g1": taxa_resolucao,
                        "risco_base": risco_proj_base,
                        "lift_risco": lift_risco,
                        "razao_risco": razao_risco,
                        "suporte": suporte,
                        "peso": max(0.05, peso),
                        "tipo": "CARTOGRAFIA"
                    })

        discriminativas = [
            item for item in leituras
            if item["risco"] >= risco_alto
            and item["lift_risco"] >= lift_minimo
            and item["razao_risco"] >= razao_minima
        ]

        melhores_por_familia = {}
        for item in discriminativas:
            familia = item["familia"]
            atual = melhores_por_familia.get(familia)
            score = (item["lift_risco"], item["risco"], item["suporte"])
            score_atual = (
                atual["lift_risco"], atual["risco"], atual["suporte"]
            ) if atual else None
            if atual is None or score > score_atual:
                melhores_por_familia[familia] = item

        independentes = list(melhores_por_familia.values())
        if independentes:
            soma_pesos = sum(item["peso"] for item in independentes)
            risco_estimado = sum(
                item["risco"] * item["peso"] for item in independentes
            ) / max(soma_pesos, 1e-9)
            lift_estimado = sum(
                item["lift_risco"] * item["peso"] for item in independentes
            ) / max(soma_pesos, 1e-9)
            razao_estimada = sum(
                item["razao_risco"] * item["peso"] for item in independentes
            ) / max(soma_pesos, 1e-9)
        else:
            risco_estimado = 0.0
            lift_estimado = 0.0
            razao_estimada = 1.0

        return {
            "ativo": True,
            "leituras": leituras,
            "consultados": len(leituras),
            "risco_alto": len(independentes),
            "contextos_discriminativos_brutos": len(discriminativas),
            "fontes_risco_alto": [item["fonte"] for item in independentes],
            "fontes_distintas_risco_alto": len(independentes),
            "familias_independentes_risco_alto": len(independentes),
            "risco_estimado": risco_estimado,
            "risco_base": risco_base,
            "lift_risco_estimado": lift_estimado,
            "razao_risco_estimada": razao_estimada
        }

    # ============================================================
    # MÉTODOS JÁ EXISTENTES
    # ============================================================
    def _avaliar_instabilidade_decisoria(self, sub_num, sub_pol, direcao):
        """
        MAIN 96 — mede o desacordo entre famílias competentes.
        """
        analise_geo = {"geometria": AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)}
        voto_especialistas = self.obter_voto_competencia_especialistas(
            sub_num, sub_pol, analise_geo
        )
        voto_camadas = self.obter_voto_camadas_ampliadas(sub_num, sub_pol)
        expectativas = MotorContagensProjetivas.mapear_janela(
            sub_num, sub_pol, analise_geo["geometria"], None
        )

        acumulado = defaultdict(lambda: {"V": 0.0, "P": 0.0, "fontes": []})

        familias_especialistas = {
            "MARKOV": "SEQUENCIAL",
            "BIGRAMA": "SEQUENCIAL",
            "TRIGRAMA": "SEQUENCIAL",
            "PROJETIVA": "PROJECAO",
            "ESPELHO_INVERSAO": "PROJECAO",
            "NUMERO": "NUMERICA",
            "GEOMETRIA": "GEOMETRIA",
        }
        for nome, d, peso in voto_especialistas.get("fontes", []):
            familia = familias_especialistas.get(nome)
            if familia and d in ("V", "P"):
                acumulado[familia][d] += max(0.05, float(peso))
                acumulado[familia]["fontes"].append(nome)

        familias_camadas = {
            "DNA_NUMERICO": "NUMERICA",
            "FECHAMENTO_NUMERICO": "NUMERICA",
            "NUMEROLOGIA_ESTATISTICA": "NUMERICA",
            "REGRAS_POSICIONAIS": "REGRAS",
            "STREAK": "GEOMETRIA",
            "XADREZ": "GEOMETRIA",
        }
        for item in voto_camadas.get("votos", []):
            familia = familias_camadas.get(item.get("camada"))
            d = "V" if item.get("direcao") == "VERMELHO" else (
                "P" if item.get("direcao") == "PRETO" else None
            )
            if familia and d:
                acumulado[familia][d] += max(0.05, float(item.get("peso", 0.0)))
                acumulado[familia]["fontes"].append(item.get("camada"))

        for regra in expectativas or []:
            d = "V" if regra.get("direcao") == "VERMELHO" else (
                "P" if regra.get("direcao") == "PRETO" else None
            )
            if not d:
                continue
            peso_manual = str(regra.get("peso", "MEDIO")).upper()
            peso = {"BAIXO": 1.0, "MEDIO": 2.0, "MÉDIO": 2.0, "ALTO": 3.0}.get(peso_manual, 2.0)
            acumulado["REGRAS"][d] += peso
            acumulado["REGRAS"]["fontes"].append(regra.get("tipo_regra", "REGRA"))

        familias = []
        peso_v = 0.0
        peso_p = 0.0
        for nome_familia in ("SEQUENCIAL", "PROJECAO", "NUMERICA", "REGRAS", "GEOMETRIA"):
            dados = acumulado.get(nome_familia)
            if not dados:
                continue
            v = float(dados["V"])
            p = float(dados["P"])
            if v == p or max(v, p) <= 0:
                continue
            d = "V" if v > p else "P"
            peso_familia = 1.0
            if d == "V":
                peso_v += peso_familia
            else:
                peso_p += peso_familia
            familias.append({
                "familia": nome_familia,
                "direcao": "VERMELHO" if d == "V" else "PRETO",
                "score_v_interno": round(v, 2),
                "score_p_interno": round(p, 2),
                "margem_interna": round(abs(v - p), 2),
                "fontes": list(dict.fromkeys(dados["fontes"])),
            })

        total = peso_v + peso_p
        if total <= 0:
            entropia = 0.0
        else:
            pv = peso_v / total
            pp = peso_p / total
            entropia = 0.0
            for prob in (pv, pp):
                if prob > 0:
                    entropia -= prob * math.log2(prob)

        letra_sinal = "V" if direcao == "VERMELHO" else "P"
        familias_sinal = sum(1 for item in familias if item["direcao"] == direcao)
        familias_contrarias = len(familias) - familias_sinal
        conflito = peso_v > 0 and peso_p > 0

        return {
            "ativo": True,
            "entropia_decisoria": entropia,
            "conflito_familias": conflito,
            "familias_ativas": len(familias),
            "familias_sinal": familias_sinal,
            "familias_contrarias": familias_contrarias,
            "peso_familias_vermelho": peso_v,
            "peso_familias_preto": peso_p,
            "direcao_sinal": letra_sinal,
            "familias": familias,
            "expectativas": expectativas,
            "geometria": analise_geo["geometria"],
        }

    def _avaliar_memoria_historica_conflito_g2_mais(self, sub_num, sub_pol, direcao, instabilidade):
        prob_markov = self.calcular_probabilidade_exata_markov(sub_pol)
        modo = AnalisadorContextoAvancado.detectar_modo_mercado(sub_pol, False, None)
        chave = self._chave_conflito(
            instabilidade.get("expectativas", []),
            instabilidade.get("geometria"),
            modo,
            prob_markov,
        )
        stats = self.memoria_conflitos.get(chave)
        suporte = int(stats.get("total", 0)) if stats else 0
        letra = "V" if direcao == "VERMELHO" else "P"
        resolveu = int(stats.get(f"{letra}_g0g1", 0)) if stats else 0
        risco = 1.0 - (resolveu / max(suporte, 1)) if suporte else 0.0

        total_base = 0
        resolveu_base = 0
        for item in self.memoria_conflitos.values():
            total_ctx = int(item.get("total", 0))
            total_base += total_ctx
            resolveu_base += int(item.get(f"{letra}_g0g1", 0))
        risco_base = 1.0 - (resolveu_base / max(total_base, 1)) if total_base else 0.0
        lift = risco - risco_base
        razao = risco / max(risco_base, 1e-9) if risco_base > 0 else 1.0

        return {
            "chave": chave,
            "suporte": suporte,
            "risco": risco,
            "risco_base": risco_base,
            "lift_risco": lift,
            "razao_risco": razao,
        }

    def avaliar_filtro_discriminativo_g0_g1(self, sub_num, sub_pol, direcao):
        """
        MAIN 96 — veto por instabilidade da decisão final.
        """
        if not self.filtro_discriminativo_metricas.get("ativo"):
            return {"ativo": False, "vetar": False, "motivo": "FILTRO_NAO_TREINADO"}
        if direcao not in ("VERMELHO", "PRETO") or len(sub_num) < 12 or len(sub_pol) < 12:
            return {"ativo": True, "vetar": False, "motivo": "CONTEXTO_INVALIDO"}

        cfg = self.filtro_discriminativo_config
        entropia_minima = float(cfg.get("entropia_decisoria_minima", 0.90))
        familias_contrarias_minimas = int(cfg.get("familias_conflitantes_minimas", 1))
        suporte_memoria_minimo = int(cfg.get("suporte_memoria_conflito_minimo", 30))
        risco_veto = float(cfg.get("risco_veto", 0.24))
        risco_precision_minimo = float(cfg.get("risco_precision_minimo", 0.0))
        lift_minimo = float(cfg.get("lift_risco_minimo", 0.03))
        razao_minima = float(cfg.get("razao_risco_minima", 1.10))

        instabilidade = self._avaliar_instabilidade_decisoria(sub_num, sub_pol, direcao)
        memoria = self._avaliar_memoria_historica_conflito_g2_mais(
            sub_num, sub_pol, direcao, instabilidade
        )
        cartografia = self._avaliar_risco_cartografia_veto(sub_num, sub_pol, direcao)

        entropia_alta = float(instabilidade.get("entropia_decisoria", 0.0)) >= entropia_minima
        conflito_familias = (
            bool(instabilidade.get("conflito_familias"))
            and int(instabilidade.get("familias_contrarias", 0)) >= familias_contrarias_minimas
        )

        risco_historico = float(memoria.get("risco", 0.0))
        risco_base = float(memoria.get("risco_base", 0.0))
        lift_historico = float(memoria.get("lift_risco", 0.0))
        razao_historica = float(memoria.get("razao_risco", 1.0))
        limite_historico = max(
            risco_veto, risco_base + lift_minimo, risco_precision_minimo
        )
        risco_historico_comprovado = (
            int(memoria.get("suporte", 0)) >= suporte_memoria_minimo
            and risco_historico >= limite_historico
            and lift_historico >= lift_minimo
            and razao_historica >= razao_minima
        )

        risco_cartografia = float(cartografia.get("risco_estimado", 0.0))
        lift_cartografia = float(cartografia.get("lift_risco_estimado", 0.0))
        razao_cartografia = float(cartografia.get("razao_risco_estimada", 1.0))
        familias_cartografia = int(cartografia.get("familias_independentes_risco_alto", 0))
        cartografia_agravante = (
            bool(cartografia.get("ativo"))
            and familias_cartografia >= int(cfg.get("contextos_cartografia_minimos", 2))
            and risco_cartografia >= max(risco_veto, float(cartografia.get("risco_base", 0.0)) + lift_minimo)
            and lift_cartografia >= lift_minimo
            and razao_cartografia >= razao_minima
        )

        rota_memoria = bool(
            entropia_alta and conflito_familias and risco_historico_comprovado
        )
        rota_cartografia_confirmada = bool(
            entropia_alta
            and conflito_familias
            and cartografia_agravante
            and familias_cartografia >= int(cfg.get("fontes_cartografia_minimas", 2))
        )
        vetar = bool(rota_memoria or rota_cartografia_confirmada)
        fontes_risco_alto = []
        if entropia_alta:
            fontes_risco_alto.append("ENTROPIA_DECISORIA")
        if conflito_familias:
            fontes_risco_alto.append("CONFLITO_FAMILIAS")
        if risco_historico_comprovado:
            fontes_risco_alto.append("MEMORIA_CONFLITO_G2_MAIS")
        if cartografia_agravante:
            fontes_risco_alto.append("CARTOGRAFIA_AGRAVANTE")

        return {
            "ativo": True,
            "vetar": vetar,
            "tipo_veto": "INSTABILIDADE_DECISORIA_G2_FALHA" if vetar else None,
            "rota_veto": (
                "ENTROPIA_CONFLITO_MEMORIA" if rota_memoria
                else ("ENTROPIA_CONFLITO_CARTOGRAFIA_CONFIRMADA" if rota_cartografia_confirmada else None)
            ),
            "risco_estimado": round(
                (risco_historico if rota_memoria else risco_cartografia) * 100, 2
            ),
            "taxa_resolucao_estimada_g0_g1": round(
                (1.0 - (risco_historico if rota_memoria else risco_cartografia)) * 100, 2
            ),
            "lift_risco_estimado_percent": round(
                (lift_historico if rota_memoria else lift_cartografia) * 100, 2
            ),
            "razao_risco_estimada": round(
                razao_historica if rota_memoria else razao_cartografia, 4
            ),
            "leituras_validas": int(instabilidade.get("familias_ativas", 0)) + int(cartografia.get("consultados", 0)),
            "contextos_risco_alto": len(fontes_risco_alto),
            "contextos_discriminativos_g2_falha": 1 if risco_historico_comprovado else 0,
            "especialistas_alinhados": int(instabilidade.get("familias_sinal", 0)),
            "fontes_risco_alto": fontes_risco_alto,
            "fontes_avaliadas": [item.get("familia") for item in instabilidade.get("familias", [])],
            "ENTROPIA_DECISORIA": round(float(instabilidade.get("entropia_decisoria", 0.0)), 4),
            "ENTROPIA_DECISORIA_ALTA": bool(entropia_alta),
            "CONFLITO_FAMILIAS": bool(conflito_familias),
            "FAMILIAS_ATIVAS": int(instabilidade.get("familias_ativas", 0)),
            "FAMILIAS_SINAL": int(instabilidade.get("familias_sinal", 0)),
            "FAMILIAS_CONTRARIAS": int(instabilidade.get("familias_contrarias", 0)),
            "MAPA_FAMILIAS_DECISORIAS": instabilidade.get("familias", []),
            "MEMORIA_CONFLITO_CHAVE": memoria.get("chave"),
            "MEMORIA_CONFLITO_SUPORTE": int(memoria.get("suporte", 0)),
            "MEMORIA_CONFLITO_G2_MAIS": bool(risco_historico_comprovado),
            "RISCO_MEMORIA_CONFLITO_PERCENT": round(risco_historico * 100, 2),
            "RISCO_BASE_MEMORIA_CONFLITO_PERCENT": round(risco_base * 100, 2),
            "CONTEXTOS_CARTOGRAFIA_CONSULTADOS": int(cartografia.get("consultados", 0)),
            "CONTEXTOS_CARTOGRAFIA_RISCO_ALTO": familias_cartografia,
            "CONTEXTOS_CARTOGRAFIA_DISCRIMINATIVOS_BRUTOS": int(cartografia.get("contextos_discriminativos_brutos", 0)),
            "FAMILIAS_CARTOGRAFIA_INDEPENDENTES": familias_cartografia,
            "FONTES_CARTOGRAFIA_RISCO_ALTO": cartografia.get("fontes_risco_alto", []),
            "RISCO_CARTOGRAFIA_PERCENT": round(risco_cartografia * 100, 2),
            "LIFT_CARTOGRAFIA_PERCENT": round(lift_cartografia * 100, 2),
            "RAZAO_RISCO_CARTOGRAFIA": round(razao_cartografia, 4),
            "CARTOGRAFIA_AGRAVANTE": bool(cartografia_agravante),
            "ROTA_V5_MEMORIA": bool(rota_memoria),
            "ROTA_V5_CARTOGRAFIA_CONFIRMADA": bool(rota_cartografia_confirmada),
            "VETO_POR_CARTOGRAFIA": bool(rota_cartografia_confirmada),
            "VETO_POR_ESPECIALISTAS": False,
            "limite_veto_percent": round(limite_historico * 100, 2),
            "limite_especialistas_percent": round(limite_historico * 100, 2),
            "limite_cartografia_percent": round(max(risco_veto, float(cartografia.get("risco_base", 0.0)) + lift_minimo) * 100, 2),
            "risco_base_especialistas_percent": round(risco_base * 100, 2),
            "risco_base_cartografia_percent": round(float(cartografia.get("risco_base", 0.0)) * 100, 2),
            "limite_tecnico_original_percent": round(risco_veto * 100, 2),
            "protecao_falso_positivo_percent": round(risco_precision_minimo * 100, 2),
            "lift_risco_minimo_percent": round(lift_minimo * 100, 2),
            "razao_risco_minima": round(razao_minima, 4),
            "entropia_decisoria_minima": round(entropia_minima, 4),
            "suporte_memoria_conflito_minimo": suporte_memoria_minimo,
            "acao": "NO_CALL" if vetar else "PRESERVAR_SINAL",
            "altera_direcao": False
        }

    def _chaves_contexto_risco_g2_mais(self, sub_num, sub_pol, direcao):
        if direcao not in ("VERMELHO", "PRETO") or len(sub_num) < 12 or len(sub_pol) < 12:
            return []
        d = "V" if direcao == "VERMELHO" else "P"
        regime = self._detectar_regime_temporal(sub_pol)
        regime_hmm = self._obter_regime_hmm_contextual(sub_pol)
        geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
        cores3 = "".join(sub_pol[-3:])
        cores5 = "".join(sub_pol[-5:])
        par_num = f"{sub_num[-2]}-{sub_num[-1]}"
        ultimo_num = str(sub_num[-1])
        return [
            f"EXATO|{d}|{regime}|{geometria}|{cores5}|{par_num}",
            f"PADRAO|{d}|{regime}|{geometria}|{cores5}",
            f"GEOMETRIA|{d}|{regime}|{geometria}|{cores3}",
            f"REGIME|{d}|{regime}|{cores3}",
            f"HMM|{d}|{regime_hmm}|{cores3}",
            f"REGIME_HMM|{d}|{regime}|{regime_hmm}",
            f"REGIME_DIRECAO|{d}|{regime}",
            f"DIRECAO|{d}",
            f"NUMERO|{d}|{regime}|{ultimo_num}",
        ]

    def _treinar_risco_g2_mais_base_longa(self):
        self.risco_g2_mais_contextos = defaultdict(lambda: {"peso_total": 0.0, "peso_risco": 0.0})
        dados = self.dados_longo or []
        if len(dados) < 500:
            self.risco_g2_mais_metricas = {"ativo": False, "motivo": "BASE_LONGA_INSUFICIENTE"}
            return

        total = len(dados)
        meia_vida = max(20000.0, total / 4.0)
        piso = 0.12
        eventos = 0

        for i in range(11, total - 2):
            janela = dados[i-11:i+1]
            sub_num = [int(d["numero"]) for d in janela]
            sub_pol = [str(d["cor"]).upper() for d in janela]
            c0 = str(dados[i+1]["cor"]).upper()
            c1 = str(dados[i+2]["cor"]).upper()

            idade = (total - 1) - i
            peso_temporal = max(piso, 0.5 ** (idade / meia_vida))

            for direcao, letra in (("VERMELHO", "V"), ("PRETO", "P")):
                resolveu_g0_g1 = c0 in (letra, "B") or c1 in (letra, "B")
                risco = 0.0 if resolveu_g0_g1 else 1.0
                for chave in self._chaves_contexto_risco_g2_mais(sub_num, sub_pol, direcao):
                    stats = self.risco_g2_mais_contextos[chave]
                    stats["peso_total"] += peso_temporal
                    stats["peso_risco"] += peso_temporal * risco
            eventos += 1

        total_peso_risco = sum(float(v.get("peso_risco", 0.0)) for k, v in self.risco_g2_mais_contextos.items() if str(k).startswith("DIRECAO|"))
        total_peso_base = sum(float(v.get("peso_total", 0.0)) for k, v in self.risco_g2_mais_contextos.items() if str(k).startswith("DIRECAO|"))
        risco_base_global = total_peso_risco / max(total_peso_base, 1e-9)
        self.risco_g2_mais_metricas = {
            "ativo": True,
            "eventos_base_longa": eventos,
            "contextos_aprendidos": len(self.risco_g2_mais_contextos),
            "backoff_hierarquico_ativo": True,
            "risco_relativo_adaptativo_ativo": True,
            "risco_base_global": round(risco_base_global, 6),
            "versao_risco": 3,
            "niveis_hierarquicos": [
                "EXATO", "PADRAO", "GEOMETRIA", "REGIME",
                "HMM", "REGIME_HMM", "REGIME_DIRECAO", "DIRECAO", "NUMERO"
            ],
            "objetivo": "MINIMIZAR_NAO_RESOLUCAO_ATE_G1",
            "g2_tratado_como_risco_operacional": True,
            "falha_tratada_como_risco_operacional": True,
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        }

    def avaliar_risco_g2_mais(self, sub_num, sub_pol, direcao):
        if not self.risco_g2_mais_metricas.get("ativo"):
            return {"ativo": False, "vetar": False, "motivo": "ESPECIALISTA_NAO_TREINADO"}

        cfg = self.risco_g2_mais_config
        limite_original = float(cfg.get("risco_veto", 0.34))
        limite_precision = float(cfg.get("risco_precision_minimo", 0.0))
        risco_base_global = float(self.risco_g2_mais_metricas.get("risco_base_global", 0.20))
        lift_minimo_risco = 0.03
        limite = max(limite_original, risco_base_global + lift_minimo_risco, limite_precision)
        suporte_padrao = float(cfg.get("suporte_efetivo_minimo", 50.0))
        min_concordantes = int(cfg.get("contextos_minimos_concordantes", 2))
        suportes_nivel = cfg.get("suportes_minimos_por_nivel", {})
        pesos_especificidade = cfg.get("pesos_especificidade", {})

        suportes_default = {
            "EXATO": 12.0, "PADRAO": 20.0, "GEOMETRIA": 30.0,
            "REGIME": 50.0, "HMM": 40.0, "REGIME_HMM": 35.0,
            "REGIME_DIRECAO": 80.0,
            "DIRECAO": 120.0, "NUMERO": 35.0
        }
        pesos_default = {
            "EXATO": 1.00, "PADRAO": 0.90, "GEOMETRIA": 0.80,
            "REGIME": 0.70, "HMM": 0.78, "REGIME_HMM": 0.86,
            "REGIME_DIRECAO": 0.60,
            "DIRECAO": 0.45, "NUMERO": 0.65
        }

        leituras = []
        for chave in self._chaves_contexto_risco_g2_mais(sub_num, sub_pol, direcao):
            stats = self.risco_g2_mais_contextos.get(chave)
            if not stats:
                continue
            nivel = chave.split("|", 1)[0]
            suporte = float(stats.get("peso_total", 0.0))
            suporte_min = float(
                suportes_nivel.get(nivel, suportes_default.get(nivel, suporte_padrao))
            )
            if suporte < suporte_min:
                continue

            risco = float(stats.get("peso_risco", 0.0)) / max(suporte, 1e-9)
            confianca_suporte = suporte / (suporte + suporte_min)
            especificidade = float(
                pesos_especificidade.get(nivel, pesos_default.get(nivel, 0.50))
            )
            peso_evidencia = suporte * confianca_suporte * especificidade
            leituras.append({
                "chave": nivel,
                "risco": risco,
                "suporte": suporte,
                "peso_evidencia": peso_evidencia
            })

        concordantes = [x for x in leituras if x["risco"] >= limite]
        if len(concordantes) >= min_concordantes:
            peso_total = sum(x["peso_evidencia"] for x in concordantes)
            risco_estimado = sum(
                x["risco"] * x["peso_evidencia"] for x in concordantes
            ) / max(peso_total, 1e-9)
            if risco_estimado >= limite:
                return {
                    "ativo": True,
                    "vetar": True,
                    "tipo_veto": "ABSOLUTO",
                    "risco_estimado": round(risco_estimado * 100, 2),
                    "contextos_validos": len(leituras),
                    "contextos_risco_alto": len(concordantes),
                    "suporte_efetivo": round(sum(x["suporte"] for x in concordantes), 2),
                    "peso_evidencia_hierarquica": round(peso_total, 2),
                    "limite_veto_percent": round(limite * 100, 2),
                    "limite_tecnico_original_percent": round(limite_original * 100, 2),
                    "protecao_falso_positivo_percent": round(limite_precision * 100, 2),
                    "backoff_hierarquico": True,
                    "niveis_concordantes": [x["chave"] for x in concordantes]
                }

        return {
            "ativo": True,
            "vetar": False,
            "tipo_veto": None,
            "risco_estimado": None,
            "contextos_validos": len(leituras),
            "contextos_risco_alto": len(concordantes),
            "backoff_hierarquico": True,
            "niveis_validos": [x["chave"] for x in leituras]
        }
