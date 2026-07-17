from collections import defaultdict
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from config.settings import HAS_ML

class EvolucaoMixin:
    @staticmethod
    def _nivel_hierarquico_regra(regra):
        tipo = str((regra or {}).get("tipo_regra", "")).upper()
        familia = str((regra or {}).get("familia", "")).upper()

        if tipo == "COEXISTENCIA_CONTAGENS_ATIVA":
            return 4, "COEXISTENCIAS"
        if tipo in ("TRANSICAO_CONTAGENS_ATIVA", "CHANCE_DUPLA_ATIVA"):
            return 5, "TRANSICOES"
        if "ASSUNCAO" in tipo:
            return 6, "ASSUNCOES"
        if (
            tipo.startswith("V3_ATIVADOR_")
            or tipo.startswith("HIERARQUIA_CONTAGEM_")
            or tipo == "FINALIZACAO_CONJUNTA_ATIVA"
            or familia in ("CONTAGENS_PROJETIVAS", "HIERARQUIA_CONTAGENS")
        ):
            return 3, "CONTAGENS"
        if tipo:
            return 2, "REGRAS_POSICIONAIS"
        return 8, "PADROES_VISUAIS"

    def _autoridade_evolutiva_regra(self, tipo_regra):
        item = (getattr(self, "matriz_evolutiva", {}) or {}).get("regras", {}).get(str(tipo_regra), {})
        return float(item.get("autoridade_atual", 0.0) or 0.0)

    def _avaliar_regras_em_dados_evolutivos(self, dados, limite_janelas=6000):
        mapa = defaultdict(lambda: {
            "total": 0, "respeito": 0, "atraso": 0, "deslocamento": 0, "falha": 0,
            "VERMELHO": 0, "PRETO": 0
        })
        if not dados or len(dados) < 15:
            return mapa

        inicio = max(0, len(dados) - int(max(100, limite_janelas)) - 15)
        fim = len(dados) - 14
        for i in range(inicio, fim):
            janela = dados[i:i+12]
            sub_num = [int(d["numero"]) for d in janela]
            sub_pol = [str(d["cor"]).upper() for d in janela]
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
            regras = MotorContagensProjetivas.mapear_janela(sub_num, sub_pol, geometria, None)
            futuros = [str(dados[i+j]["cor"]).upper() for j in (12, 13, 14)]
            vistos = set()
            for regra in regras:
                tipo = str(regra.get("tipo_regra", ""))
                direcao = str(regra.get("direcao", ""))
                if not tipo or direcao not in ("VERMELHO", "PRETO"):
                    continue
                chave_evento = (tipo, direcao)
                if chave_evento in vistos:
                    continue
                vistos.add(chave_evento)
                letra = "V" if direcao == "VERMELHO" else "P"
                st = mapa[tipo]
                st["total"] += 1
                st[direcao] += 1
                if futuros[0] in (letra, "B"):
                    st["respeito"] += 1
                elif futuros[1] in (letra, "B"):
                    st["atraso"] += 1
                elif futuros[2] in (letra, "B"):
                    st["deslocamento"] += 1
                else:
                    st["falha"] += 1
        return mapa

    @staticmethod
    def _classificar_tendencia_evolutiva(hist, rec):
        total_h = max(1, int(hist.get("total", 0)))
        total_r = max(1, int(rec.get("total", 0)))
        sucesso_h = (hist.get("respeito", 0) + hist.get("atraso", 0)) / total_h
        sucesso_r = (rec.get("respeito", 0) + rec.get("atraso", 0)) / total_r
        risco_h = (hist.get("deslocamento", 0) + hist.get("falha", 0)) / total_h
        risco_r = (rec.get("deslocamento", 0) + rec.get("falha", 0)) / total_r
        delta = sucesso_r - sucesso_h
        delta_risco = risco_r - risco_h

        if delta <= -0.18 or delta_risco >= 0.18:
            estado = "DEGRADACAO_CRITICA"
        elif delta <= -0.12 or delta_risco >= 0.12:
            estado = "DEGRADACAO_FORTE"
        elif delta <= -0.07 or delta_risco >= 0.07:
            estado = "DEGRADACAO_MODERADA"
        elif delta <= -0.03 or delta_risco >= 0.03:
            estado = "DEGRADACAO_LEVE"
        elif delta >= 0.12 and delta_risco <= -0.08:
            estado = "RECUPERACAO_FORTE"
        elif delta >= 0.07 and delta_risco <= -0.04:
            estado = "RECUPERACAO_MODERADA"
        elif delta >= 0.03:
            estado = "RECUPERACAO_LEVE"
        else:
            estado = "ESTAVEL"

        if delta > 0.03:
            tendencia = "TENDENCIA_DE_ALTA"
        elif delta < -0.03:
            tendencia = "TENDENCIA_DE_BAIXA"
        else:
            tendencia = "TENDENCIA_LATERAL"
        return estado, tendencia, sucesso_h, sucesso_r, risco_h, risco_r

    def atualizar_matriz_evolutiva(self):
        longo = list(getattr(self, "dados_longo", []) or [])
        rec = list(getattr(self, "dados_recencia", []) or [])
        if len(rec) < 30:
            rec = longo[-min(1200, len(longo)):]
        hist = longo[:-len(rec)] if rec and len(longo) > len(rec) else longo
        hist_stats = self._avaliar_regras_em_dados_evolutivos(hist, limite_janelas=6000)
        rec_stats = self._avaliar_regras_em_dados_evolutivos(rec, limite_janelas=1500)

        regras = {}
        tipos = set(hist_stats.keys()) | set(rec_stats.keys())
        for tipo in tipos:
            h = hist_stats.get(tipo, {})
            r = rec_stats.get(tipo, {})
            if int(r.get("total", 0)) < 3 and int(h.get("total", 0)) < 5:
                continue
            estado, tendencia, sucesso_h, sucesso_r, risco_h, risco_r = self._classificar_tendencia_evolutiva(h, r)
            total_r = max(1, int(r.get("total", 0)))
            suporte = min(1.0, int(r.get("total", 0)) / 30.0)
            autoridade = max(0.0, min(1.0, (sucesso_r * 0.70 + (1.0 - risco_r) * 0.30) * suporte))
            regras[tipo] = {
                "total_historico": int(h.get("total", 0)),
                "total_recente": int(r.get("total", 0)),
                "taxa_respeito": round(100.0 * r.get("respeito", 0) / total_r, 2),
                "taxa_atraso": round(100.0 * r.get("atraso", 0) / total_r, 2),
                "taxa_deslocamento": round(100.0 * r.get("deslocamento", 0) / total_r, 2),
                "taxa_falha": round(100.0 * r.get("falha", 0) / total_r, 2),
                "sucesso_historico_ate_g1": round(sucesso_h * 100.0, 2),
                "sucesso_recente_ate_g1": round(sucesso_r * 100.0, 2),
                "risco_historico_g2_mais": round(risco_h * 100.0, 2),
                "risco_recente_g2_mais": round(risco_r * 100.0, 2),
                "estado_evolutivo": estado,
                "tendencia": tendencia,
                "autoridade_atual": round(autoridade, 4)
            }

        def extremo(campo, maior=True, filtro=None):
            itens = [(k, v) for k, v in regras.items() if filtro is None or filtro(v)]
            if not itens:
                return None
            return (max if maior else min)(itens, key=lambda kv: float(kv[1].get(campo, 0.0)))[0]

        self.matriz_evolutiva = {
            "ativo": True,
            "metodo": "JANELAS_MOVEIS_HISTORICO_X_RECENCIA_G0_G1_G2_FALHA",
            "regras": regras,
            "regra_mais_forte": extremo("autoridade_atual", True),
            "regra_mais_fraca": extremo("autoridade_atual", False),
            "regra_em_recuperacao": extremo("autoridade_atual", True, lambda v: str(v.get("estado_evolutivo", "")).startswith("RECUPERACAO")),
            "regra_em_degradacao": extremo("taxa_falha", True, lambda v: str(v.get("estado_evolutivo", "")).startswith("DEGRADACAO")),
            "regra_mais_respeitada": extremo("taxa_respeito", True),
            "regra_mais_atrasada": extremo("taxa_atraso", True),
            "regra_mais_deslocada": extremo("taxa_deslocamento", True),
            "regra_mais_falha": extremo("taxa_falha", True),
            "registros_historicos_considerados": len(hist),
            "registros_recentes_considerados": len(rec)
        }
        return self.matriz_evolutiva

    def _atualizar_ml_controlada_incremental(self, dados_combinados):
        if not HAS_ML:
            self.ml_atualizacao_incremental_metricas = {"ativo": False, "motivo": "ML_INDISPONIVEL"}
            return
        dados = list(dados_combinados or [])
        if len(dados) < 200:
            self.ml_atualizacao_incremental_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        janela_maxima = 12000
        recorte = dados[-min(janela_maxima, len(dados)):]
        inicio = len(dados) - len(recorte)
        try:
            self._treinar_ml_avancado(recorte)
            self.ml_atualizacao_incremental_metricas = {
                "ativo": True,
                "metodo": "RETREINO_CRONOLOGICO_CONTROLADO_EM_CAUDA",
                "registros_base_acumulada": len(dados),
                "registros_usados_ml": len(recorte),
                "inicio_cronologico_recorte": inicio,
                "janela_maxima": janela_maxima,
                "ml_ready": bool(getattr(self, "ml_ready", False)),
                "metricas_validacao": dict(getattr(self, "ml_metricas", {}) or {})
            }
        except Exception as e:
            self.ml_atualizacao_incremental_metricas = {
                "ativo": False,
                "motivo": "ERRO_ATUALIZACAO_CONTROLADA",
                "erro": f"{type(e).__name__}: {e}",
                "modelo_anterior_preservado": bool(getattr(self, "ml_ready", False))
            }
