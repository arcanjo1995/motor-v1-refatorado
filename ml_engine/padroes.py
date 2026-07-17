import math
from collections import defaultdict
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas

class PadroesMixin:
    """
    Mixin para leitura de Z-Score, Quebradores Históricos e Simulação de Rotas.
    """

    def _garantir_quebradores_defaultdict(self, info):
        """
        Garante que o campo 'quebradores' seja um defaultdict(int).
        """
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
                    info = self.padroes_xadrez_detalhado[chave]
                    info = self._garantir_quebradores_defaultdict(info)
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
                    info = self.padroes_streak_detalhado[chave]
                    info = self._garantir_quebradores_defaultdict(info)
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
                info = self.padroes_gerais_detalhado[chave]
                info = self._garantir_quebradores_defaultdict(info)
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

    # O restante dos métodos ( _processar_bloco_dados, _calcular_comportamento_dominante, etc.)
    # permanece idêntico ao que você já tem. Para economia de espaço, mantenha o código
    # original após este ponto, apenas substitua o método mapear_padroes_avancados e adicione
    # o método _garantir_quebradores_defaultdict.
