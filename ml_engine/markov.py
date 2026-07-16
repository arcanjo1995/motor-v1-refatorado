from collections import defaultdict

class MarkovMixin:
    """
    Mixin isolando lógicas de Cadeias de Markov Multiescala e Memória Temporal.
    """

    def _treinar_markov_multiescala(self):
        self.markov_ordens = {ordem: defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0}) for ordem in range(1, 7)}

        def absorver(dados, peso_base=1, temporal=False):
            if not dados: return
            cores = [str(d.get("cor", "B")).upper() for d in dados]
            total = len(cores)
            for ordem in range(1, 7):
                if total <= ordem: continue
                for i in range(ordem, total):
                    estado = tuple(cores[i-ordem:i])
                    proxima = cores[i]
                    if proxima not in ("V", "P", "B"): continue
                    fator = peso_base
                    if temporal and total > 1000:
                        fator = max(1, int(peso_base * (1.0 + (i / total) * 1.5)))
                    stats = self.markov_ordens[ordem][estado]
                    stats[proxima] += fator
                    stats["total"] += fator

        absorver(getattr(self, 'dados_longo', []), peso_base=1, temporal=True)
        absorver(getattr(self, 'dados_recencia', []), peso_base=6, temporal=False)

    @staticmethod
    def _detectar_regime_temporal(cores):
        janela = [c for c in (cores or [])[-12:] if c in ("V", "P")]
        if len(janela) < 4: return "MISTO"
        alternancias = sum(1 for i in range(1, len(janela)) if janela[i] != janela[i-1])
        freq_alt = alternancias / max(1, len(janela) - 1)
        streaks = []
        atual = 1
        for i in range(1, len(janela)):
            if janela[i] == janela[i-1]: atual += 1
            else:
                streaks.append(atual)
                atual = 1
        streaks.append(atual)
        streak_medio = sum(streaks) / len(streaks)
        if freq_alt >= 0.60: return "XADREZ_DOMINANTE"
        if streak_medio >= 2.5 or max(streaks) >= 4: return "STREAK_DOMINANTE"
        return "MISTO"

    def _treinar_memoria_temporal_adaptativa(self):
        self.markov_temporal = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        self.markov_temporal_regime = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}

        dados = getattr(self, 'dados_longo', [])
        total = len(dados)
        if total < 30:
            self.temporal_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        cores = [str(d.get("cor", "B")).upper() for d in dados]
        meia_vida = max(int(self.temporal_config.get("meia_vida_minima", 20000)), max(1, total // 4))
        piso = float(self.temporal_config.get("piso_memoria_historica", 0.12))
        soma_pesos = 0.0

        for i in range(1, total):
            idade = (total - 1) - i
            peso_tempo = max(piso, 0.5 ** (idade / meia_vida))
            proxima = cores[i]
            if proxima not in ("V", "P", "B"): continue
            regime = self._detectar_regime_temporal(cores[max(0, i-12):i])
            soma_pesos += peso_tempo

            for ordem in range(1, min(6, i) + 1):
                estado = tuple(cores[i-ordem:i])
                stats = self.markov_temporal[ordem][estado]
                stats[proxima] += peso_tempo
                stats["total"] += peso_tempo

                chave_regime = (regime, estado)
                stats_regime = self.markov_temporal_regime[ordem][chave_regime]
                stats_regime[proxima] += peso_tempo
                stats_regime["total"] += peso_tempo

        self.temporal_metricas = {
            "ativo": True, "registros_base_longa": total,
            "meia_vida_registros": meia_vida, "piso_memoria_historica": piso,
            "massa_temporal_efetiva": round(soma_pesos, 2), "recencia_oficial_preservada_peso": 6
        }

    def calcular_probabilidade_exata_markov(self, ultimas_cores):
        if not ultimas_cores:
            return {"V": 0.0, "P": 0.0, "B": 0.0}

        if hasattr(self, "markov_ordens") and self.markov_ordens:
            acumulado = {"V": 0.0, "P": 0.0, "B": 0.0}
            peso_total = 0.0
            detalhes = []

            min_amostra = {6: 12, 5: 16, 4: 22, 3: 30, 2: 40, 1: 50}
            peso_ordem = {6: 6.0, 5: 5.0, 4: 4.0, 3: 3.0, 2: 2.0, 1: 1.0}

            for ordem in range(min(6, len(ultimas_cores)), 0, -1):
                estado = tuple(ultimas_cores[-ordem:])
                stats = self.markov_ordens.get(ordem, {}).get(estado)
                if not stats: continue
                total = stats.get("total", 0)
                if total < min_amostra[ordem]: continue

                denom = total + 3.0
                probs = {"V": (stats.get("V", 0) + 1.0) / denom, "P": (stats.get("P", 0) + 1.0) / denom, "B": (stats.get("B", 0) + 1.0) / denom}
                confianca_amostra = min(1.0, total / (min_amostra[ordem] * 4.0))
                peso = peso_ordem[ordem] * (0.5 + 0.5 * confianca_amostra)

                for cor in acumulado:
                    acumulado[cor] += probs[cor] * peso
                peso_total += peso
                detalhes.append({"ordem": ordem, "amostra": total})

            if peso_total > 0:
                return {
                    "V": round((acumulado["V"] / peso_total) * 100, 2),
                    "P": round((acumulado["P"] / peso_total) * 100, 2),
                    "B": round((acumulado["B"] / peso_total) * 100, 2),
                    "ordens_utilizadas": detalhes
                }

        # Fallback legado
        if len(ultimas_cores) < 2: return {"V": 0.0, "P": 0.0, "B": 0.0}
        estado_inicial_2 = (ultimas_cores[-2], ultimas_cores[-1])
        resultados = {'V': 0, 'P': 0, 'B': 0}
        transicoes_2_longa = getattr(self, 'modelo_transicao', {}).get(estado_inicial_2, [])
        for cor in transicoes_2_longa: resultados[cor] += 1
        
        dados_rec = getattr(self, 'dados_recencia', [])
        for i in range(len(dados_rec) - 1):
            if i > 0 and (dados_rec[i-1]['cor'], dados_rec[i]['cor']) == estado_inicial_2:
                cor_seguinte = dados_rec[i+1]['cor']
                resultados[cor_seguinte] += 5

        total = sum(resultados.values())
        if total == 0: return {"V": 0.0, "P": 0.0, "B": 0.0}
        return {
            "V": round((resultados['V'] / total) * 100, 2),
            "P": round((resultados['P'] / total) * 100, 2),
            "B": round((resultados['B'] / total) * 100, 2)
        }
