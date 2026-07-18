# ml_engine/radar_numerico.py
from collections import defaultdict
import math
from utils.helpers import hash_chave
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from config.settings import VERSAO_CHAVES_HASH

class RadarNumericoMixin:
    """
    Mixin para o Radar Numérico Evoluído.
    Aprende continuamente com Base, Recência e Feedback ao vivo.
    Gera consenso entre três fontes e influencia a arbitragem sem inverter direção.
    """

    def _inicializar_memoria_radar(self):
        """Inicializa a memória do Radar se ainda não existir."""
        if not hasattr(self, 'memoria_radar'):
            self.memoria_radar = defaultdict(lambda: {
                "total": 0,
                "acertos_g0": 0,
                "acertos_g1": 0,
                "erros": 0,
                "historico_numeros": defaultdict(int),  # número real -> contagem
                "acertos_g0_por_numero": defaultdict(int),
                "acertos_g1_por_numero": defaultdict(int),
                "erros_por_numero": defaultdict(int),
                # Para pesos adaptativos (opcional)
                "fonte_base_acertos": 0,
                "fonte_recencia_acertos": 0,
                "fonte_ao_vivo_acertos": 0,
            })
        if not hasattr(self, 'pesos_radar'):
            # Pesos iniciais fixos; serão ajustados com o tempo se habilitado
            self.pesos_radar = {
                "base": 0.40,
                "recencia": 0.35,
                "ao_vivo": 0.25
            }
        if not hasattr(self, 'config_radar'):
            self.config_radar = {
                "versao": 1,
                "aprendizado_continuo": True,
                "pesos_adaptativos": False,  # True para ativar ajuste dinâmico
                "minimo_amostras_para_peso": 30,
                "limiar_ameaca_critica": 3.0,   # multiplicador sobre teórico
                "limiar_ameaca_alta": 2.0,
                "limiar_ameaca_media": 1.5,
                "influencia_maxima": 0.25,      # quanto pode alterar a autoridade da regra
                "influencia_minima": 0.05,
            }

    def _chave_contexto_radar(self, sub_num, sub_pol, analise_contexto=None):
        """
        Gera uma chave compacta para o contexto atual.
        Inclui: geometria, regime HMM, Markov, regra dominante, bigrama, streak.
        """
        if not sub_num or not sub_pol:
            return "RADAR_SEM_CONTEXTO"

        nums = [int(x) for x in sub_num[-12:]]
        pol = [str(x).upper() for x in sub_pol[-12:]]

        geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
        regime_hmm = self._obter_regime_hmm_contextual(pol)
        markov = self.calcular_probabilidade_exata_markov(pol)
        mv = float(markov.get("V", 0.0))
        mp = float(markov.get("P", 0.0))
        if abs(mv - mp) < 0.50:
            faixa_markov = "NEUTRO"
        elif mv > mp:
            faixa_markov = "V_FORTE" if abs(mv - mp) >= 2.0 else "V_LEVE"
        else:
            faixa_markov = "P_FORTE" if abs(mv - mp) >= 2.0 else "P_LEVE"

        # Regra dominante (se disponível)
        regra_dominante = "SEM_REGRA"
        if analise_contexto:
            regras = analise_contexto.get("regras_posicionais", [])
            if regras:
                # Pega a primeira regra (já ordenada por hierarquia no juiz)
                regra_dominante = str(regras[0].get("tipo_regra", "SEM_REGRA"))

        bigrama = "-".join(str(x) for x in nums[-2:])
        # Streak atual
        streak = 0
        if pol and pol[-1] in ("V", "P"):
            cor = pol[-1]
            for c in reversed(pol):
                if c == cor:
                    streak += 1
                else:
                    break
        streak = min(streak, 8)

        chave = (
            f"RADAR|GEO={geometria}|HMM={regime_hmm}|MK={faixa_markov}"
            f"|REGRA={regra_dominante}|BI={bigrama}|STREAK={streak}"
        )
        return hash_chave(chave)

    def _prever_numero_radar(self, sub_num, sub_pol, analise_contexto=None):
        """
        Calcula a distribuição de probabilidades para cada número (0..14)
        usando consenso entre Base, Recência e Ao Vivo.
        Retorna um dicionário com:
        - distribuição: dict {numero: prob}
        - numero_dominante: int
        - consenso_max: float (probabilidade do dominante)
        - confiabilidade: float (baseada no histórico de acertos do Radar)
        - fontes: dict com as probabilidades de cada fonte
        """
        nums = [int(x) for x in sub_num[-12:]]
        pol = [str(x).upper() for x in sub_pol[-12:]]

        # 1. Probabilidade da Base de Longo Prazo (usando transições numéricas)
        prob_base = self._probabilidade_base_numerica(nums, pol)

        # 2. Probabilidade da Recência (últimos 200 registros)
        prob_recencia = self._probabilidade_recencia_numerica(nums, pol)

        # 3. Probabilidade da sequência ao vivo (usando heurística atual)
        prob_ao_vivo = self._probabilidade_ao_vivo_numerica(nums, pol)

        # Consenso ponderado
        pesos = self.pesos_radar
        consenso = {}
        for num in range(15):
            consenso[num] = (
                prob_base.get(num, 0.0) * pesos["base"] +
                prob_recencia.get(num, 0.0) * pesos["recencia"] +
                prob_ao_vivo.get(num, 0.0) * pesos["ao_vivo"]
            )

        # Normalizar para soma 1
        total = sum(consenso.values())
        if total > 0:
            for num in consenso:
                consenso[num] /= total
        else:
            # Fallback: distribuição uniforme
            for num in consenso:
                consenso[num] = 1.0 / 15.0

        # Número dominante
        numero_dominante = max(consenso, key=consenso.get)
        consenso_max = consenso[numero_dominante]

        # Confiabilidade: baseada no histórico de acertos do Radar para esse contexto/número
        chave_contexto = self._chave_contexto_radar(sub_num, sub_pol, analise_contexto)
        memoria = self.memoria_radar.get(chave_contexto, {})
        total_prev = memoria.get("total", 0)
        acertos_g0 = memoria.get("acertos_g0_por_numero", {}).get(numero_dominante, 0)
        acertos_g1 = memoria.get("acertos_g1_por_numero", {}).get(numero_dominante, 0)
        erros = memoria.get("erros_por_numero", {}).get(numero_dominante, 0)
        total_num = acertos_g0 + acertos_g1 + erros

        if total_num >= 5:
            # Precisão G0 + G1
            precisao = (acertos_g0 + acertos_g1) / total_num
            confiabilidade = min(1.0, precisao * 1.2)  # leve ajuste
        else:
            # Poucos dados: usa a confiabilidade global do Radar
            total_global = sum(m.get("total", 0) for m in self.memoria_radar.values())
            if total_global > 0:
                acertos_global = sum(m.get("acertos_g0", 0) + m.get("acertos_g1", 0) for m in self.memoria_radar.values())
                confiabilidade = acertos_global / total_global
            else:
                confiabilidade = 0.5  # neutro

        # Atualiza os pesos das fontes se adaptativo
        if self.config_radar.get("pesos_adaptativos"):
            self._ajustar_pesos_fonte(chave_contexto, numero_dominante)

        return {
            "distribuicao": consenso,
            "numero_dominante": numero_dominante,
            "consenso_max": consenso_max,
            "confiabilidade": round(confiabilidade, 4),
            "fontes": {
                "base": prob_base,
                "recencia": prob_recencia,
                "ao_vivo": prob_ao_vivo
            },
            "memoria": memoria  # para relatório
        }

    def _probabilidade_base_numerica(self, nums, pol):
        """Probabilidade baseada na Base de Longo Prazo (transições numéricas)."""
        ultimo = nums[-1]
        prob = {i: 0.0 for i in range(15)}
        trans = getattr(self, 'transicoes_numericas', {}).get(ultimo)
        if trans:
            total = float(trans.get("total", 0.0))
            if total > 0:
                for num, qtd in trans.get("proximos", {}).items():
                    prob[int(num)] = qtd / total
        # Suavização para evitar zeros
        for i in range(15):
            prob[i] = (prob[i] + 0.01) / 1.15  # Laplace smoothing
        return prob

    def _probabilidade_recencia_numerica(self, nums, pol):
        """Probabilidade baseada na Recência (últimos 200 registros)."""
        dados_rec = getattr(self, 'dados_recencia', [])[-200:]
        prob = {i: 0.0 for i in range(15)}
        if not dados_rec or len(dados_rec) < 3:
            return prob

        ultimo = nums[-1]
        total = 0
        contagem = defaultdict(int)
        for i in range(len(dados_rec) - 1):
            if int(dados_rec[i].get("numero", -1)) == ultimo:
                prox = int(dados_rec[i+1].get("numero", -1))
                if 0 <= prox <= 14:
                    contagem[prox] += 1
                    total += 1
        if total > 0:
            for num in contagem:
                prob[num] = contagem[num] / total
        # Suavização
        for i in range(15):
            prob[i] = (prob[i] + 0.01) / 1.15
        return prob

    def _probabilidade_ao_vivo_numerica(self, nums, pol):
        """
        Probabilidade baseada na sequência ao vivo.
        Usa a mesma lógica heurística atual (simulacao de rotas, bigrama, etc.)
        """
        # Reaproveita a função existente simular_rotas_proximos_resultados,
        # mas extrai apenas a distribuição de números.
        resultado = self.simular_rotas_proximos_resultados(nums, pol, limite_rotas=15)
        prob = {i: 0.0 for i in range(15)}
        if resultado.get("ativo") and resultado.get("rotas"):
            for rota in resultado["rotas"]:
                num = rota["numero_g0_possivel"]
                prob_rota = rota["probabilidade_rota_percent"] / 100.0
                prob[int(num)] += prob_rota
            # Normalizar
            total = sum(prob.values())
            if total > 0:
                for num in prob:
                    prob[num] /= total
        return prob

    def _ajustar_pesos_fonte(self, chave_contexto, numero_previsto):
        """Atualiza os pesos das fontes com base no desempenho histórico no contexto."""
        memoria = self.memoria_radar.get(chave_contexto, {})
        total = memoria.get("total", 0)
        if total < self.config_radar.get("minimo_amostras_para_peso", 30):
            return

        # Supondo que armazenamos acertos por fonte (Base, Recência, Ao Vivo)
        # Precisamos de campos adicionais na memória para isso.
        # Para simplificar, usaremos uma abordagem baseada em erro médio (não implementado agora)
        # Deixamos como placeholder; pode ser implementado depois.
        # Por enquanto, mantém pesos fixos.
        pass

    def _atualizar_memoria_radar(self, chave_contexto, numero_real, resultado):
        """
        Atualiza a memória do Radar com o resultado real.
        resultado: 'G0', 'G1', ou 'FALHA'
        """
        memoria = self.memoria_radar[chave_contexto]
        memoria["total"] += 1
        memoria["historico_numeros"][numero_real] += 1

        if resultado == "G0":
            memoria["acertos_g0"] += 1
            memoria["acertos_g0_por_numero"][numero_real] += 1
        elif resultado == "G1":
            memoria["acertos_g1"] += 1
            memoria["acertos_g1_por_numero"][numero_real] += 1
        else:
            memoria["erros"] += 1
            memoria["erros_por_numero"][numero_real] += 1

    def _treinar_radar_em_janela(self, sub_num, sub_pol, numero_g0, numero_g1, analise_contexto=None):
        """Treina o Radar com uma janela histórica, onde conhecemos G0 e G1."""
        chave = self._chave_contexto_radar(sub_num, sub_pol, analise_contexto)
        # Prever número dominante (apenas para registrar, não usado no treino)
        previsao = self._prever_numero_radar(sub_num, sub_pol, analise_contexto)
        # Atualizar com o número real de G0
        self._atualizar_memoria_radar(chave, numero_g0, "G0")
        # Se G1 for diferente e também for relevante, podemos registrar como "atraso"
        if numero_g1 != numero_g0 and numero_g1 is not None:
            self._atualizar_memoria_radar(chave, numero_g1, "G1")
        # Se ambos falharem (não ocorreram em G0/G1), isso é tratado no feedback ao vivo

    def _processar_feedback_radar(self, sub_num, sub_pol, numeros_saidos, analise_contexto, classificacao):
        """
        Processa o feedback ao vivo: atualiza a memória com o resultado real.
        """
        if not numeros_saidos or len(numeros_saidos) < 2:
            return
        g0 = int(numeros_saidos[0])
        g1 = int(numeros_saidos[1]) if len(numeros_saidos) > 1 else None

        chave = self._chave_contexto_radar(sub_num, sub_pol, analise_contexto)
        # Atualiza com G0
        self._atualizar_memoria_radar(chave, g0, "G0")
        # Se G1 existe e é diferente, atualiza também
        if g1 is not None and g1 != g0:
            self._atualizar_memoria_radar(chave, g1, "G1")
        # Se a classificação for FALHA, registramos que o número previsto não saiu
        # (mas já registramos os números que saíram, então a memória já tem a informação)

    def obter_influencia_radar(self, sub_num, sub_pol, analise_contexto=None):
        """
        Retorna um dicionário com a influência do Radar para a arbitragem.
        """
        previsao = self._prever_numero_radar(sub_num, sub_pol, analise_contexto)
        numero = previsao["numero_dominante"]
        consenso = previsao["consenso_max"]
        confiabilidade = previsao["confiabilidade"]

        # Determinar se é uma ameaça (2, 6, dupla) -> isso é tratado fora
        # Aqui só geramos o fator de influência
        influencia = {
            "numero_dominante": numero,
            "consenso": consenso,
            "confiabilidade": confiabilidade,
            "distribuicao": previsao["distribuicao"],
            "fontes": previsao["fontes"],
            "memoria": previsao["memoria"],
            "fator_influencia": 0.0,  # será calculado com base no contexto
        }

        # Calcular fator de influência (entre -1 e 1)
        # Se o Radar aponta um número que historicamente favorece a direção proposta,
        # podemos ajustar. Por enquanto, apenas calculamos um fator baseado na confiança
        # e na magnitude do consenso.
        fator_base = consenso * confiabilidade
        # Normalizar para um fator entre 0 e 0.25 (influência máxima configurada)
        max_influencia = self.config_radar.get("influencia_maxima", 0.25)
        fator = min(max_influencia, fator_base * 2.0)  # ajuste empírico
        influencia["fator_influencia"] = round(fator, 4)

        # Guardar a última influência para relatório
        self._ultima_influencia_radar = influencia
        return influencia

    def gerar_relatorio_radar(self, sub_num, sub_pol, analise_contexto=None):
        """
        Gera um relatório detalhado do Radar para exibição.
        """
        influencia = self.obter_influencia_radar(sub_num, sub_pol, analise_contexto)
        distrib = influencia["distribuicao"]
        fontes = influencia["fontes"]
        numero_dominante = influencia["numero_dominante"]
        consenso = influencia["consenso"]

        # Preparar dados para tabela
        tabela = []
        for num in range(15):
            tabela.append({
                "numero": num,
                "base": round(fontes["base"].get(num, 0.0) * 100, 2),
                "recencia": round(fontes["recencia"].get(num, 0.0) * 100, 2),
                "ao_vivo": round(fontes["ao_vivo"].get(num, 0.0) * 100, 2),
                "consenso": round(distrib.get(num, 0.0) * 100, 2)
            })

        # Desempenho histórico para o número dominante
        memoria = influencia["memoria"]
        total_num = (memoria.get("acertos_g0_por_numero", {}).get(numero_dominante, 0) +
                     memoria.get("acertos_g1_por_numero", {}).get(numero_dominante, 0) +
                     memoria.get("erros_por_numero", {}).get(numero_dominante, 0))
        acertos_g0 = memoria.get("acertos_g0_por_numero", {}).get(numero_dominante, 0)
        acertos_g1 = memoria.get("acertos_g1_por_numero", {}).get(numero_dominante, 0)
        erros = memoria.get("erros_por_numero", {}).get(numero_dominante, 0)
        if total_num > 0:
            taxa_g0 = acertos_g0 / total_num * 100
            taxa_g1 = acertos_g1 / total_num * 100
            taxa_erro = erros / total_num * 100
        else:
            taxa_g0 = taxa_g1 = taxa_erro = 0.0

        # Classificação de ameaça
        referencia_teorica = 100.0 / 15.0
        multiplicador = consenso / referencia_teorica if referencia_teorica > 0 else 0
        if multiplicador >= self.config_radar.get("limiar_ameaca_critica", 3.0):
            classificacao = "🚨 AMEAÇA CRÍTICA"
        elif multiplicador >= self.config_radar.get("limiar_ameaca_alta", 2.0):
            classificacao = "⚠️ AMEAÇA ALTA"
        elif multiplicador >= self.config_radar.get("limiar_ameaca_media", 1.5):
            classificacao = "⚡ AMEAÇA MÉDIA"
        else:
            classificacao = "✅ NORMAL"

        # Ranking dos números mais prováveis (top 5)
        ranking = sorted([(num, distrib.get(num, 0.0)) for num in range(15)],
                         key=lambda x: x[1], reverse=True)[:5]
        ranking_formatado = [f"{num} → {prob*100:.1f}%" for num, prob in ranking]

        relatorio = {
            "tabela": tabela,
            "numero_dominante": numero_dominante,
            "consenso": consenso * 100,
            "referencia_teorica": referencia_teorica,
            "multiplicador": multiplicador,
            "classificacao": classificacao,
            "confiabilidade": influencia["confiabilidade"] * 100,
            "taxa_acerto_g0": taxa_g0,
            "taxa_acerto_g1": taxa_g1,
            "taxa_erro": taxa_erro,
            "fator_influencia": influencia["fator_influencia"],
            "ranking": ranking_formatado,
            "memoria_contagem": memoria.get("total", 0)
        }
        return relatorio
