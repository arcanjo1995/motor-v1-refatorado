import os
from main import LeitorXLS, MotorNoCall, MotorContagensProjetivas, AnalisadorContextoAvancado, JuizHierarquicoModificado, IAPreditivaV1

class ProcessadorTipoB:
    def __init__(self, sequencia_12_numeros, caminho_base_dados):
        self.entrada_usuario = sequencia_12_numeros
        self.caminho_base = caminho_base_dados
        
        # LEGENDA REAL IMUTÁVEL PARA ENTRADA MANUAL (Volume 8)
        self.polaridades_usuario = []
        for num in self.entrada_usuario:
            if num == 0: 
                self.polaridades_usuario.append("B")
            elif 1 <= num <= 7: 
                self.polaridades_usuario.append("V")
            else: 
                self.polaridades_usuario.append("P")

    # CORREÇÃO CRÍTICA: Alterado de 'ejecutar' para 'executar' para alinhar com a interface
    def executar_sinal_real(self):
        if len(self.entrada_usuario) != 12: 
            return "[ERRO] Requisito de exatamente 12 números violado."
            
        leitor = LeitorXLS(self.caminho_base)
        base_historica = leitor.ler_e_validar()
        if not base_historica: 
            return "[ERRO] Base de dados resultados_blaze.xlsx ausente."

        # Separação das listas globais da cronologia
        num_global = [d['numero'] for d in base_historica]
        pol_global = [d['cor'] for d in base_historica]

        # 1. Instancia e treina a IA Preditiva com a base de longo prazo
        ia_operacional = IAPreditivaV1(base_historica)
        previsao_ia = ia_operacional.predizer_proxima_casa(self.entrada_usuario, self.polaridades_usuario)

        # 2. Mapeia a geometria de mercado com base na entrada atual do usuário
        saturacao = AnalisadorContextoAvancado.mapear_padroes_geometria(self.polaridades_usuario)
        
        # 3. Executa a checagem rigorosa de travas de Nível 1 (Duplas, 2, 6 e Branco nas posições do manual)
        nc_ativo, motivo_nc = MotorNoCall.checar_no_call(self.entrada_usuario, self.polaridades_usuario)
        
        # 4. Mapeia as contagens projetivas e regras do 4, 10 e 5-10
        expectativas = MotorContagensProjetivas.mapear_janela(self.entrada_usuario, self.polaridades_usuario, saturacao)
        
        # 5. Calcula a inclinação histórica pós-número baseando-se no fechamento atual
        num_fechamento = self.entrada_usuario[-1]
        inclinacao_num = AnalisadorContextoAvancado.calcular_numerologia_pos_numero(num_fechamento, num_global, pol_global)
        
        # 6. O Juiz Hierárquico unifica tudo e dita o veredito final operativo (Com IA e Geometria incluídas)
        sinal_final, justificativa = JuizHierarquicoModificado.arbitrar_sinal(
            nc_ativo, motivo_nc, expectativas, inclinacao_num, saturacao, previsao_ia
        )

        # 7. Preditor de atraso do Branco estatístico
        chance_branco, casas_atraso = AnalisadorContextoAvancado.preditor_estatistico_branco(num_fechamento, num_global, pol_global)

        # Construção da memória de cálculo formatada para a interface (Volume 22 - Capítulo 1)
        output = "[MEMÓRIA DE CÁLCULO]\n"
        output += f"- Mapeamento: Sequência {self.entrada_usuario} processada.\n"
        output += f"- Geometria da Janela: {saturacao}\n"
        output += f"- Previsão IA: {previsao_ia[0]} ({previsao_ia[1]:.1f}%)\n"
        output += f"- Inclinação Histórica ({num_fechamento}): {inclinacao_num[0]} ({inclinacao_num[1]:.1f}%)\n"
        output += f"- Resolução de Conflitos: {justificativa}\n\n"
        output += "[RESULTADO FINAL TIPO B]\n"
        output += f"SINAL: {sinal_final}\n"
        output += f"BRANCO: {chance_branco} CHANCE (Atraso: {casas_atraso} rodadas)\n"
        output += f"ESTADO DO MERCADO: {saturacao}\n"
        return output
