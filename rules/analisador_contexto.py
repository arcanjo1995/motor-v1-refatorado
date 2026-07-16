class AnalisadorContextoAvancado:
    @staticmethod
    def mapear_padroes_geometria(sub_pol):
        texto = "".join(sub_pol)
        if texto.endswith("VPPV"):
            return "CICLO_FECHADO_VPPV"
        if texto.endswith("PVVP"):
            return "CICLO_FECHADO_PVVP"
        if texto.endswith("VVVV"):
            return "SATURAÇÃO ESTRUTURAL (V)"
        if texto.endswith("PPPP"):
            return "SATURAÇÃO ESTRUTURAL (P)"
        if "VPVP" in texto or "PVPV" in texto:
            return "XADREZ ATIVO"
        return "ESTÁVEL"

    @staticmethod
    def detectar_modo_mercado(sub_pol, eh_sinal_real=False, ia_modelo=None):
        # Usa o HMM se disponível
        if ia_modelo and getattr(ia_modelo, 'ml_ready', False):
            try:
                import numpy as np
                if getattr(ia_modelo, 'ml_hmm', None) is not None:
                    c_map = {'P': 0, 'V': 1, 'B': 2}
                    seq = [[c_map.get(c, 2)] for c in sub_pol]
                    estado_oculto = ia_modelo.ml_hmm.predict(seq)[-1]
                    regimes = {0: "HMM_CONSOLIDACAO (Estável)", 1: "HMM_TENDENCIA (Surfe)", 2: "HMM_CAOS (Recolhimento)"}
                    return regimes.get(estado_oculto, "NEUTRO")
            except:
                pass

        texto = "".join(sub_pol)
        alternancias = sum(1 for i in range(len(texto)-1) if texto[i] != texto[i+1])
        if eh_sinal_real:
            if alternancias >= 7:
                return "REGIME_RECOLHIMENTO"
            elif alternancias <= 3:
                return "REGIME_PAGADOR"
            return "NEUTRO"
        else:
            if alternancias >= 7:
                return "CHUVA"
            elif alternancias <= 3:
                return "RECUPERACAO"
            return "NEUTRO"
