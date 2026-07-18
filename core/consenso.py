# core/consenso.py
"""
Módulo de Consenso Ponderado por Evidência (PDE).
Substitui a hierarquia fixa por pesos dinâmicos baseados em desempenho histórico.
"""

import math
from collections import defaultdict

class ConsensoPonderado:
    """
    Calcula o consenso entre todas as evidências, cada uma com peso dinâmico
    baseado em sua taxa de acerto na base longa e na recência.
    """

    # Fatores de ponderação para cada tipo de fonte (base vs recência)
    FATORES = {
        "REGRA": {"base": 0.50, "recencia": 0.50},
        "CONTAGEM": {"base": 0.40, "recencia": 0.60},
        "RADAR": {"base": 0.20, "recencia": 0.80},
        "IA": {"base": 0.40, "recencia": 0.60},
        "CARTOGRAFIA": {"base": 0.50, "recencia": 0.50},
        "MARKOV": {"base": 0.40, "recencia": 0.60},
        "GEOMETRIA": {"base": 0.50, "recencia": 0.50},
        "STREAK": {"base": 0.40, "recencia": 0.60},
        "MORFOLOGIA": {"base": 0.40, "recencia": 0.60},
    }

    # Suporte mínimo para considerar a taxa de recência (se menor, usa apenas base)
    SUPORTE_MINIMO_RECENCIA = 5

    @staticmethod
    def calcular_peso(taxa_base, taxa_recencia, suporte_recencia, tipo_fonte):
        """
        Calcula o peso dinâmico de uma fonte.
        taxa_base: float 0..1 (acerto na base longa)
        taxa_recencia: float 0..1 (acerto na recência)
        suporte_recencia: int (número de ocorrências na recência)
        tipo_fonte: str (ex: "REGRA", "RADAR", "IA")
        Retorna: float (peso final)
        """
        fatores = ConsensoPonderado.FATORES.get(tipo_fonte, {"base": 0.50, "recencia": 0.50})
        fator_base = fatores["base"]
        fator_recencia = fatores["recencia"]

        # Se não houver suporte suficiente na recência, usa apenas a base
        if suporte_recencia < ConsensoPonderado.SUPORTE_MINIMO_RECENCIA:
            peso = taxa_base
        else:
            peso = (taxa_base * fator_base) + (taxa_recencia * fator_recencia)

        # Shrinkage por suporte: fontes com poucas ocorrências têm peso reduzido
        suporte_total = suporte_recencia + 30  # 30 é o mínimo para confiança plena
        shrink = min(1.0, suporte_total / 60.0)
        peso = peso * shrink

        return round(peso, 4)

    @staticmethod
    def coletar_votos(ia_modelo, expectations, geometria_mercado, previsao_ia,
                      probabilidade_markov, influencia_radar, sub_num, sub_pol):
        """
        Coleta todas as evidências disponíveis e retorna uma lista de votos.
        Cada voto é um dict: {"direcao": "VERMELHO"/"PRETO", "score": float, "fonte": str, "tipo": str}
        """
        votos = []

        # 1. Regras posicionais
        for regra in (expectativas or []):
            direcao = regra.get("direcao")
            if direcao not in ("VERMELHO", "PRETO"):
                continue
            # Força do voto = autoridade da regra (já calculada na matriz evolutiva)
            if ia_modelo and hasattr(ia_modelo, "_autoridade_evolutiva_regra"):
                autoridade = ia_modelo._autoridade_evolutiva_regra(regra.get("tipo_regra", ""))
            else:
                autoridade = 0.50  # fallback
            # Peso dinâmico: taxa de acerto da regra na base e recência
            tipo_regra = regra.get("tipo_regra", "DESCONHECIDA")
            taxa_base = ConsensoPonderado._taxa_regra_base(ia_modelo, tipo_regra)
            taxa_recencia, suporte_recencia = ConsensoPonderado._taxa_regra_recencia(ia_modelo, tipo_regra)
            peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "REGRA")
            score = autoridade * peso
            votos.append({
                "direcao": direcao,
                "score": round(score, 4),
                "fonte": f"REGRA_{tipo_regra}",
                "tipo": "REGRA",
                "peso": peso,
                "autoridade": autoridade,
                "taxa_base": taxa_base,
                "taxa_recencia": taxa_recencia,
                "suporte_recencia": suporte_recencia
            })

        # 2. Contagens consolidadas (V3, coexistência, etc.)
        if ia_modelo and hasattr(ia_modelo, "obter_voto_contagens_consolidado"):
            try:
                voto_contagens = ia_modelo.obter_voto_contagens_consolidado(
                    sub_num, sub_pol, expectations
                )
                if voto_contagens and voto_contagens.get("ativo"):
                    direcao = voto_contagens.get("direcao")
                    if direcao in ("VERMELHO", "PRETO"):
                        forca = float(voto_contagens.get("peso", 0.0)) / 18.0  # normaliza 0..1
                        taxa_base = ConsensoPonderado._taxa_contagem_base(ia_modelo, sub_num, sub_pol)
                        taxa_recencia, suporte_recencia = ConsensoPonderado._taxa_contagem_recencia(ia_modelo, sub_num, sub_pol)
                        peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "CONTAGEM")
                        score = forca * peso
                        votos.append({
                            "direcao": direcao,
                            "score": round(score, 4),
                            "fonte": "CONTAGENS_CONSOLIDADAS",
                            "tipo": "CONTAGEM",
                            "peso": peso,
                            "forca": forca,
                            "taxa_base": taxa_base,
                            "taxa_recencia": taxa_recencia,
                            "suporte_recencia": suporte_recencia
                        })
            except Exception:
                pass

        # 3. IA observacional
        direcao_ia, confianca_ia, raciocinio_ia = previsao_ia
        if direcao_ia in ("VERMELHO", "PRETO") and float(confianca_ia) >= 52.5:
            forca = float(confianca_ia) / 100.0
            taxa_base = ConsensoPonderado._taxa_ia_base(ia_modelo)
            taxa_recencia, suporte_recencia = ConsensoPonderado._taxa_ia_recencia(ia_modelo)
            peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "IA")
            score = forca * peso
            votos.append({
                "direcao": direcao_ia,
                "score": round(score, 4),
                "fonte": "IA_OBSERVACIONAL",
                "tipo": "IA",
                "peso": peso,
                "forca": forca,
                "taxa_base": taxa_base,
                "taxa_recencia": taxa_recencia,
                "suporte_recencia": suporte_recencia
            })

        # 4. Geometria de mercado
        if geometria_mercado == "CICLO_FECHADO_PVVP":
            direcao = "VERMELHO"
            forca = 0.30  # peso fixo para geometria
            taxa_base = 0.50  # pode ser melhorado com dados históricos
            taxa_recencia = 0.50
            suporte_recencia = 10
            peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "GEOMETRIA")
            score = forca * peso
            votos.append({
                "direcao": direcao,
                "score": round(score, 4),
                "fonte": "GEOMETRIA_PVVP",
                "tipo": "GEOMETRIA",
                "peso": peso,
                "forca": forca
            })
        elif geometria_mercado == "CICLO_FECHADO_VPPV":
            direcao = "PRETO"
            forca = 0.30
            taxa_base = 0.50
            taxa_recencia = 0.50
            suporte_recencia = 10
            peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "GEOMETRIA")
            score = forca * peso
            votos.append({
                "direcao": direcao,
                "score": round(score, 4),
                "fonte": "GEOMETRIA_VPPV",
                "tipo": "GEOMETRIA",
                "peso": peso,
                "forca": forca
            })

        # 5. Radar Numérico
        if influencia_radar:
            direcao, forca, numero = ConsensoPonderado._extrair_voto_radar(influencia_radar)
            if direcao and forca > 0:
                taxa_base = ConsensoPonderado._taxa_radar_base(ia_modelo, numero)
                taxa_recencia, suporte_recencia = ConsensoPonderado._taxa_radar_recencia(ia_modelo, numero)
                peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "RADAR")
                # Bônus para números críticos (2, 6)
                if numero in (2, 6):
                    forca = min(1.0, forca * 1.4)
                score = forca * peso
                votos.append({
                    "direcao": direcao,
                    "score": round(score, 4),
                    "fonte": f"RADAR_NUMERO_{numero}",
                    "tipo": "RADAR",
                    "peso": peso,
                    "forca": forca,
                    "numero": numero,
                    "taxa_base": taxa_base,
                    "taxa_recencia": taxa_recencia,
                    "suporte_recencia": suporte_recencia
                })

        # 6. Cartografias (XLS, padrões, regras)
        cartografias = [
            ("CARTOGRAFIA_XLS", ia_modelo.obter_voto_cartografia_xls if ia_modelo else None),
            ("PADRAO_CONTEXTUAL", ia_modelo.obter_voto_padrao_contextual if ia_modelo else None),
            ("REGRA_CONTEXTUAL", ia_modelo.obter_voto_regra_contextual if ia_modelo else None),
        ]
        for nome, metodo in cartografias:
            if metodo:
                try:
                    voto = metodo(sub_num, sub_pol)
                    if voto and voto.get("direcao") in ("VERMELHO", "PRETO"):
                        direcao = voto["direcao"]
                        forca = float(voto.get("peso", 0.0)) / 18.0
                        # Para cartografias, usamos a taxa de acerto geral (se disponível)
                        taxa_base = 0.50  # fallback
                        taxa_recencia = 0.50
                        suporte_recencia = 10
                        if ia_modelo and hasattr(ia_modelo, "cartografia_xls_metricas"):
                            # Podemos extrair uma taxa global da cartografia
                            pass
                        peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "CARTOGRAFIA")
                        score = forca * peso
                        votos.append({
                            "direcao": direcao,
                            "score": round(score, 4),
                            "fonte": nome,
                            "tipo": "CARTOGRAFIA",
                            "peso": peso,
                            "forca": forca
                        })
                except Exception:
                    pass

        # 7. Streak consolidada
        if ia_modelo and hasattr(ia_modelo, "obter_voto_streak_consolidado"):
            try:
                voto_streak = ia_modelo.obter_voto_streak_consolidado(sub_num, sub_pol)
                if voto_streak and voto_streak.get("ativo") and voto_streak.get("direcao") in ("VERMELHO", "PRETO"):
                    direcao = voto_streak["direcao"]
                    forca = float(voto_streak.get("peso", 0.0))
                    taxa_base = 0.50
                    taxa_recencia = 0.50
                    suporte_recencia = 10
                    peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "STREAK")
                    score = forca * peso
                    votos.append({
                        "direcao": direcao,
                        "score": round(score, 4),
                        "fonte": "STREAK_CONSOLIDADA",
                        "tipo": "STREAK",
                        "peso": peso,
                        "forca": forca,
                        "suporte": voto_streak.get("suporte", 0)
                    })
            except Exception:
                pass

        # 8. Morfologia estrutural
        if ia_modelo and hasattr(ia_modelo, "obter_voto_morfologia_estrutural"):
            try:
                voto_morf = ia_modelo.obter_voto_morfologia_estrutural(sub_num, sub_pol)
                if voto_morf and voto_morf.get("ativo") and voto_morf.get("direcao") in ("VERMELHO", "PRETO"):
                    direcao = voto_morf["direcao"]
                    forca = float(voto_morf.get("peso", 0.0))
                    taxa_base = 0.50
                    taxa_recencia = 0.50
                    suporte_recencia = 10
                    peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "MORFOLOGIA")
                    score = forca * peso
                    votos.append({
                        "direcao": direcao,
                        "score": round(score, 4),
                        "fonte": "MORFOLOGIA_ESTRUTURAL",
                        "tipo": "MORFOLOGIA",
                        "peso": peso,
                        "forca": forca,
                        "suporte": voto_morf.get("suporte", 0)
                    })
            except Exception:
                pass

        # 9. Markov observacional
        if probabilidade_markov:
            v = float(probabilidade_markov.get("V", 0.0))
            p = float(probabilidade_markov.get("P", 0.0))
            if abs(v - p) >= 2.0:
                direcao = "VERMELHO" if v > p else "PRETO"
                forca = abs(v - p) / 20.0  # normaliza 0..1 (20% de diferença = 1.0)
                taxa_base = 0.50
                taxa_recencia = 0.50
                suporte_recencia = 10
                peso = ConsensoPonderado.calcular_peso(taxa_base, taxa_recencia, suporte_recencia, "MARKOV")
                score = forca * peso
                votos.append({
                    "direcao": direcao,
                    "score": round(score, 4),
                    "fonte": "MARKOV_OBSERVACIONAL",
                    "tipo": "MARKOV",
                    "peso": peso,
                    "forca": forca,
                    "v": v,
                    "p": p
                })

        return votos

    # ============================================================
    # MÉTODOS AUXILIARES PARA TAXAS DE ACERTO (BASE E RECÊNCIA)
    # ============================================================

    @staticmethod
    def _taxa_regra_base(ia_modelo, tipo_regra):
        """Taxa de acerto G0/G1 da regra na base longa."""
        if ia_modelo and hasattr(ia_modelo, "regras_competencia_cronologica"):
            stats = ia_modelo.regras_competencia_cronologica.get(tipo_regra, {})
            total = stats.get("total_validacao", 0)
            if total > 0:
                return stats.get("acertos_g0_g1", 0) / total
        return 0.50

    @staticmethod
    def _taxa_regra_recencia(ia_modelo, tipo_regra):
        """Taxa de acerto G0/G1 da regra na recência e suporte."""
        # Será implementada no motor_unificado, armazenando em uma estrutura
        if ia_modelo and hasattr(ia_modelo, "regras_competencia_recencia"):
            stats = ia_modelo.regras_competencia_recencia.get(tipo_regra, {})
            total = stats.get("total", 0)
            if total > 0:
                return stats.get("acertos", 0) / total, total
        return 0.50, 0

    @staticmethod
    def _taxa_contagem_base(ia_modelo, sub_num, sub_pol):
        """Taxa de respeito da contagem na base longa."""
        # Simplificado: usa a taxa global de respeito das projeções
        if ia_modelo and hasattr(ia_modelo, "projecoes_respeito_metricas"):
            return ia_modelo.projecoes_respeito_metricas.get("taxa_respeito_g0_g1_percent", 50.0) / 100.0
        return 0.50

    @staticmethod
    def _taxa_contagem_recencia(ia_modelo, sub_num, sub_pol):
        """Taxa de respeito da contagem na recência."""
        # Será implementada
        return 0.50, 0

    @staticmethod
    def _taxa_ia_base(ia_modelo):
        """Acurácia da IA na base longa."""
        if ia_modelo and hasattr(ia_modelo, "ml_metricas"):
            acc_gb = ia_modelo.ml_metricas.get("acuracia_gb", 50.0)
            acc_mlp = ia_modelo.ml_metricas.get("acuracia_mlp", 50.0)
            return (acc_gb + acc_mlp) / 200.0  # média / 100
        return 0.50

    @staticmethod
    def _taxa_ia_recencia(ia_modelo):
        """Acurácia da IA na recência."""
        # Será implementada
        return 0.50, 0

    @staticmethod
    def _taxa_radar_base(ia_modelo, numero):
        """Taxa de acerto G0/G1 do Radar para um número específico na base longa."""
        if ia_modelo and hasattr(ia_modelo, "memoria_radar"):
            total_global = 0
            acertos_global = 0
            for chave, stats in ia_modelo.memoria_radar.items():
                if stats.get("total", 0) > 0:
                    total_global += stats["total"]
                    acertos_global += stats.get("acertos_g0_por_numero", {}).get(numero, 0)
                    acertos_global += stats.get("acertos_g1_por_numero", {}).get(numero, 0)
            if total_global > 0:
                return acertos_global / total_global
        return 0.50

    @staticmethod
    def _taxa_radar_recencia(ia_modelo, numero):
        """Taxa de acerto G0/G1 do Radar na recência."""
        # Será implementada
        return 0.50, 0

    @staticmethod
    def _extrair_voto_radar(influencia_radar):
        """Extrai direção, força e número do Radar."""
        if not influencia_radar:
            return None, 0.0, None
        numero = influencia_radar.get("numero_dominante")
        if numero is None:
            return None, 0.0, None
        consenso = float(influencia_radar.get("consenso", 0.0))
        confiabilidade = float(influencia_radar.get("confiabilidade", 0.0))
        fator = float(influencia_radar.get("fator_influencia", 0.0))
        # Direção baseada no número
        if numero == 0:
            return None, 0.0, numero
        direcao = "VERMELHO" if 1 <= numero <= 7 else "PRETO" if 8 <= numero <= 14 else None
        if direcao is None:
            return None, 0.0, numero
        # Força: combinação de consenso, confiabilidade e fator normalizado
        fator_norm = abs(fator) / 0.25  # 0..1
        forca = (consenso * 0.50) + (confiabilidade * 0.30) + (fator_norm * 0.20)
        return direcao, min(1.0, forca), numero

    @staticmethod
    def calcular_consenso(votos, limiar_minimo=0.05):
        """
        Soma os scores por direção e decide.
        Retorna: (direcao, score_vencedor, score_perdedor, detalhes)
        """
        score_v = sum(v["score"] for v in votos if v["direcao"] == "VERMELHO")
        score_p = sum(v["score"] for v in votos if v["direcao"] == "PRETO")
        diferenca = abs(score_v - score_p)
        total_votos = len(votos)

        detalhes = {
            "score_vermelho": round(score_v, 4),
            "score_preto": round(score_p, 4),
            "diferenca": round(diferenca, 4),
            "total_votos": total_votos,
            "votos": votos
        }

        if diferenca < limiar_minimo:
            return "NO_CALL", score_v, score_p, detalhes

        direcao = "VERMELHO" if score_v > score_p else "PRETO"
        return direcao, score_v, score_p, detalhes
