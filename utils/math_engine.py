import math

class EngineMatematicoAvancado:
    @staticmethod
    def calcular_entropia_shannon(sub_pol):
        if not sub_pol: 
            return 0.0
        total = len(sub_pol)
        counts = {'V': sub_pol.count('V'), 'P': sub_pol.count('P'), 'B': sub_pol.count('B')}
        entropia = 0.0
        for cor, count in counts.items():
            if count > 0:
                p = count / total
                entropia -= p * math.log2(p)
        return round(entropia, 3)

    @staticmethod
    def calcular_raridade_sequencia(sub_pol):
        if not sub_pol: 
            return {"streak": 0, "probabilidade": 100.0, "status": "SEM DADOS"}
        ultima_cor = sub_pol[-1]
        if ultima_cor not in ['V', 'P']: 
            return {"streak": 0, "probabilidade": 100.0, "status": "BRANCO NO FECHAMENTO"}
        streak = 0
        for cor in reversed(sub_pol):
            if cor == ultima_cor: 
                streak += 1
            else: 
                break
        probabilidade_sequencia = ((7 / 15) ** streak) * 100
        status = "SATURAÇÃO CRÍTICA" if streak >= 5 else ("DESVIO PADRÃO EM CURSO" if streak >= 3 else "ESTRUTURA DENTRO DA NORMALIDADE")
        return {
            "streak": streak, 
            "cor_sequencia": "VERMELHO" if ultima_cor == 'V' else "PRETO", 
            "probabilidade": round(probabilidade_sequencia, 2), 
            "status": status
        }

    @staticmethod
    def calcular_vies_surfe(caminho_base, janela=100):
        # Importação interna para evitar dependências circulares com a camada de dados
        from data.leitor_xls import LeitorXLS
        
        leitor = LeitorXLS(caminho_base)
        dados = leitor.ler_e_validar()
        
        if not dados: 
            return {
                "vies": "INDISPONÍVEL", "desvio_v": 0.0, "desvio_p": 0.0, 
                "frequencia_v": 46.67, "frequencia_p": 46.67, "frequencia_b": 6.67
            }
            
        ultimos = dados[-janela:]
        v = sum(1 for d in ultimos if d['cor'] == 'V')
        p = sum(1 for d in ultimos if d['cor'] == 'P')
        b = sum(1 for d in ultimos if d['cor'] == 'B')
        
        pct_v = (v / len(ultimos)) * 100
        pct_p = (p / len(ultimos)) * 100
        pct_b = (b / len(ultimos)) * 100
        
        desvio_v = round(pct_v - 46.67, 2)
        desvio_p = round(pct_p - 46.67, 2)
        vies = "SURFE DE MACROFREQUÊNCIA: VIÁS PARA VERMELHO ATIVO" if pct_v >= 53.0 else ("SURFE DE MACROFREQUÊNCIA: VIÁS PARA PRETO ATIVO" if pct_p >= 53.0 else "MACROANÁLISE EQUILIBRADA")
        
        return {
            "frequencia_v": round(pct_v, 2), "frequencia_p": round(pct_p, 2), 
            "frequencia_b": round(pct_b, 2), "desvio_v": desvio_v, 
            "desvio_p": desvio_p, "vies": vies
        }
