class MotorNoCall:
    """
    Motor exclusivo de segurança e gestão de risco (NO CALL).
    """
    
    @staticmethod
    def checar_no_call(sub_num, sub_pol):
        if not sub_num or not sub_pol:
            return False, ""
        
        texto_pol = "".join(sub_pol)
        ultimo_num = sub_num[-1]
        
        # Bloqueio de proteção contra o Zero/Branco
        if ultimo_num == 0:
            return True, "O Branco acabou de sair. O mercado precisa de tempo para resetar a tendência."
            
        # Alerta de espelho perfeito (Padrão 1-2-1) sem direção clara
        if len(sub_pol) >= 3 and sub_pol[-3] == sub_pol[-1] and sub_pol[-2] != sub_pol[-1]:
            if len(sub_pol) >= 5 and sub_pol[-5:] == list("PVPVP") or sub_pol[-5:] == list("VPVPV"):
                pass 
            else:
                return True, "Instabilidade Direcional: Formação de espelho curto detectada (Ex: V-P-V). Direção incerta."
        
        # Saturação Crítica
        if texto_pol.endswith("VVVVV") or texto_pol.endswith("PPPPP"):
            return True, "Mercado Saturado: 5 ou mais sequências seguidas da mesma cor. Risco extremo de reversão imprevisível."
            
        return False, ""

    @staticmethod
    def checar_risco_preditivo_g0(sub_num, ia_modelo):
        """
        MAIN 96 — delega o veto preditivo exclusivamente ao filtro discriminativo
        (G0/G1 vs G2+) se ele estiver ativo. Caso contrário, aplica as proteções
        históricas e determinísticas tradicionais.
        """
        if not sub_num or len(sub_num) < 2:
            return False, ""

        if ia_modelo is None:
            return False, ""

        if getattr(ia_modelo, "filtro_discriminativo_metricas", {}).get("ativo"):
            return False, ""

        ultimo_num = sub_num[-1]
        penultimo_num = sub_num[-2]

        if hasattr(ia_modelo, 'padroes_gerais_detalhado') and ia_modelo.padroes_gerais_detalhado:
            if penultimo_num == ultimo_num:
                chave_dupla = f"PADRAO_GERAL_2 [{penultimo_num}-{ultimo_num}]"
                padrao = ia_modelo.padroes_gerais_detalhado.get(chave_dupla)
                if padrao and padrao.get("total", 0) >= 5:
                    v = padrao.get("apos_V", 0)
                    p = padrao.get("apos_P", 0)
                    total = v + p
                    if total > 0:
                        taxa_v = v / total
                        taxa_p = p / total
                        if abs(taxa_v - taxa_p) < 0.15:
                            return True, f"Proteção de Dupla Numérica: A dupla {penultimo_num}-{ultimo_num} quebrou o padrão estatístico. As probabilidades estão empatadas ({round(taxa_v*100)}% V vs {round(taxa_p*100)}% P)."

        if hasattr(ia_modelo, 'estatisticas_bigramas_globais') and ia_modelo.estatisticas_bigramas_globais:
            bigrama = f"{penultimo_num}-{ultimo_num}"
            stats_bi = ia_modelo.estatisticas_bigramas_globais.get(bigrama)
            if stats_bi and stats_bi.get("total", 0) >= 8:
                taxa_v = ((stats_bi.get("V_g0", 0) + stats_bi.get("V_g1", 0)) / stats_bi["total"])
                taxa_p = ((stats_bi.get("P_g0", 0) + stats_bi.get("P_g1", 0)) / stats_bi["total"])
                if abs(taxa_v - taxa_p) < 0.12:
                    return True, f"Proteção de Bigrama: O histórico de {bigrama} gera taxas conflituosas (V={round(taxa_v*100)}%, P={round(taxa_p*100)}%). Aposta suspensa até G1."

        if hasattr(ia_modelo, 'unidade_analise'):
            stats_ultimo = ia_modelo.unidade_analise.get(ultimo_num, {})
            if stats_ultimo and stats_ultimo.get("ocorrencias", 0) >= 15:
                freq_v = float(stats_ultimo.get("freq_v", 0.0)) / 100.0
                freq_p = float(stats_ultimo.get("freq_p", 0.0)) / 100.0
                if abs(freq_v - freq_p) < 0.05:
                     return True, f"Risco Posicional: O número {ultimo_num} está em fase de instabilidade absoluta entre as cores."

        return False, ""
