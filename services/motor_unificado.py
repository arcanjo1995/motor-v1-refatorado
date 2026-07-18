# services/motor_unificado.py
import os
import gc
import pandas as pd
from datetime import datetime

from config.settings import NOME_BASE_DEFINITIVA
from data.leitor_xls import LeitorXLS
from data.persistence import carregar_modelo_longo_prazo, salvar_modelo_longo_prazo
from ml_engine.preditor_base import IAPreditivaV1
from core.motor_analise import MotorAnalise
from core.juiz_hierarquico import JuizHierarquicoModificado
from services.treinador import (
    analisar_regime_recencia,
    integrar_recencia_no_modelo,
    adicionar_a_base_longo_prazo,
    treinar_base_longo_prazo_com_janelas
)


class MotorUnificadoV1:
    def __init__(self):
        self.ia = None
        self.regime_recencia = None
        self.ultima_atualizacao = None
        self.base_longa_carregada = False
        self.recencia_injetada = False

    def carregar_tudo(self, forcar_recencia=True):
        print("[MOTOR UNIFICADO] Iniciando carregamento completo...")
        self.ia = carregar_modelo_longo_prazo()
        if self.ia and len(self.ia.dados_recencia) > 200:
            self.ia.dados_recencia = self.ia.dados_recencia[-200:]
        if self.ia is None:
            if os.path.exists(NOME_BASE_DEFINITIVA):
                dados_longos = LeitorXLS(NOME_BASE_DEFINITIVA).ler_e_validar()
                if dados_longos and len(dados_longos) >= 50:
                    relatorio = treinar_base_longo_prazo_com_janelas(dados_longos)
                    self.ia = relatorio.get("ia_treinada")
                    self.base_longa_carregada = self.ia is not None
                    if self.ia is not None:
                        salvar_modelo_longo_prazo(self.ia)
                del dados_longos
                gc.collect()
        else:
            self.base_longa_carregada = True
        if forcar_recencia:
            if self.ia and not getattr(self.ia, "recencia_foi_injetada_na_sessao", False):
                self._carregar_e_injetar_recencia()
        self.ultima_atualizacao = datetime.now()
        gc.collect()
        print("[MOTOR UNIFICADO] Carregamento concluído.")

    def _carregar_e_injetar_recencia(self):
        if getattr(self, "recencia_injetada", False):
            return
        if not os.path.exists("base_recencia_ativa.xlsx"):
            return
        dados_rec = LeitorXLS("base_recencia_ativa.xlsx").ler_e_validar()
        if not dados_rec or len(dados_rec) < 20:
            return
        dados_rec = dados_rec[-200:]
        print(f"[MOTOR UNIFICADO] Injetando {len(dados_rec)} registros de recência com peso alto...")
        if self.ia is None:
            self.ia = IAPreditivaV1([], [])
        self.ia.dados_recencia = list(dados_rec)
        self.ia.injetar_aprendizado_imediato(
            dados_rec,
            multiplicador_peso=6,
            salvar_na_recencia=False
        )
        self.ia.treinar_q_learning_contextual(
            dados_rec,
            multiplicador_peso=6,
            origem="RECENCIA"
        )
        self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
        self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
        self.ia.regime_recencia = self.regime_recencia
        self.ia.recencia_foi_injetada_na_sessao = True
        self.recencia_injetada = True

        # <-- RADAR: treinar Radar sobre a recência injetada
        if self.ia is not None and hasattr(self.ia, '_treinar_radar_em_janela') and len(dados_rec) >= 13:
            try:
                nums = [int(d.get("numero")) for d in dados_rec]
                pol = [str(d.get("cor", "B")).upper() for d in dados_rec]
                for i in range(11, len(dados_rec) - 2):
                    sub_num = nums[i-11:i+1]
                    sub_pol = pol[i-11:i+1]
                    g0 = int(dados_rec[i+1].get("numero", -1))
                    g1 = int(dados_rec[i+2].get("numero", -1)) if i+2 < len(dados_rec) else None
                    if g0 < 0:
                        continue
                    try:
                        analise = MotorAnalise.analisar_janela(sub_num, sub_pol, self.ia, eh_sinal_real=False)
                        self.ia._treinar_radar_em_janela(sub_num, sub_pol, g0, g1, analise)
                    except Exception as e:
                        self.ia._treinar_radar_em_janela(sub_num, sub_pol, g0, g1, None)
            except Exception as e:
                print(f"[RADAR] Erro ao treinar Radar com recência: {e}")

    def absorver_base_longa(self, dados_novos):
        if not dados_novos or len(dados_novos) < 30:
            return {"sucesso": False, "mensagem": "Base muito pequena."}
        try:
            if os.path.exists(NOME_BASE_DEFINITIVA):
                backup_name = NOME_BASE_DEFINITIVA.replace(
                    ".xlsx", f"_backup_substituicao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                )
                os.replace(NOME_BASE_DEFINITIVA, backup_name)
            df_base = pd.DataFrame([{"numero": d["numero"], "cor": d["cor"]} for d in dados_novos])
            df_base.iloc[::-1].reset_index(drop=True).to_excel(NOME_BASE_DEFINITIVA, index=False)
        except Exception as e:
            return {"sucesso": False, "mensagem": f"Falha ao substituir a base definitiva: {e}"}
        relatorio = treinar_base_longo_prazo_com_janelas(dados_novos)
        self.ia = relatorio.get("ia_treinada")
        self.base_longa_carregada = True
        if os.path.exists("base_recencia_ativa.xlsx"):
            self._carregar_e_injetar_recencia()
        sucesso = salvar_modelo_longo_prazo(self.ia)
        return {"sucesso": True, "registros_absorvidos": len(dados_novos), "modelo_salvo": sucesso, "mensagem": "Absorvido."}

    def processar_novo_lote(self, novos_dados):
        """
        Encadeia o delta sem recarregar o XLS nem reconstruir a IA após a atualização.
        """
        if not novos_dados:
            return {"sucesso": False, "mensagem": "Nenhum dado novo foi fornecido."}
        if self.ia is None:
            self.carregar_tudo(forcar_recencia=False)
        if self.ia is None:
            return {
                "sucesso": False,
                "mensagem": "Modelo persistido não encontrado. Substitua/treine a base definitiva uma vez antes do encadeamento incremental."
            }

        relatorio = adicionar_a_base_longo_prazo(novos_dados, origem_feedback_ao_vivo=False)

        if isinstance(relatorio, dict) and relatorio.get("sucesso"):
            ia_atualizada = relatorio.get("ia_treinada")
            if ia_atualizada is not None:
                self.ia = ia_atualizada
            self.base_longa_carregada = self.ia is not None
            self.ultima_atualizacao = datetime.now()

        gc.collect()
        return relatorio

    def processar_recencia(self, dados_recencia):
        if not dados_recencia or len(dados_recencia) < 20:
            return {"sucesso": False, "mensagem": "Base de recência muito pequena."}
        if self.ia is None:
            self.carregar_tudo(forcar_recencia=False)
        dados_recencia_ativos = [
            {"numero": d["numero"], "cor": d["cor"]}
            for d in dados_recencia[-200:]
        ]
        self.ia.dados_recencia = list(dados_recencia_ativos)
        self.ia.injetar_aprendizado_imediato(
            dados_recencia_ativos,
            multiplicador_peso=6,
            salvar_na_recencia=False
        )
        self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
        self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
        self.ia.regime_recencia = self.regime_recencia
        self.recencia_injetada = True
        self.ia.treinar_q_learning_contextual(
            dados_recencia_ativos,
            multiplicador_peso=6,
            origem="RECENCIA"
        )
        self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
        self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
        self.ia.regime_recencia = self.regime_recencia
        self.ia.atualizar_matriz_evolutiva()
        salvar_modelo_longo_prazo(self.ia)

        # <-- RADAR: treinar Radar sobre a recência processada
        if self.ia is not None and hasattr(self.ia, '_treinar_radar_em_janela') and len(dados_recencia_ativos) >= 13:
            try:
                nums = [int(d.get("numero")) for d in dados_recencia_ativos]
                pol = [str(d.get("cor", "B")).upper() for d in dados_recencia_ativos]
                for i in range(11, len(dados_recencia_ativos) - 2):
                    sub_num = nums[i-11:i+1]
                    sub_pol = pol[i-11:i+1]
                    g0 = int(dados_recencia_ativos[i+1].get("numero", -1))
                    g1 = int(dados_recencia_ativos[i+2].get("numero", -1)) if i+2 < len(dados_recencia_ativos) else None
                    if g0 < 0:
                        continue
                    try:
                        analise = MotorAnalise.analisar_janela(sub_num, sub_pol, self.ia, eh_sinal_real=False)
                        self.ia._treinar_radar_em_janela(sub_num, sub_pol, g0, g1, analise)
                    except Exception as e:
                        self.ia._treinar_radar_em_janela(sub_num, sub_pol, g0, g1, None)
            except Exception as e:
                print(f"[RADAR] Erro ao treinar Radar com recência processada: {e}")

        return {
            "sucesso": True,
            "registros_processados": len(dados_recencia),
            "registros_recencia_ativos": len(dados_recencia_ativos),
            "recencia_separada_base_mestra": True,
            "regime_recencia": self.regime_recencia,
            "matriz_evolutiva": self.ia.matriz_evolutiva,
            "mensagem": "Recência processada com sucesso em buffer separado da Base Mestra."
        }

    def gerar_sinal_tipo_b(self, sequencia_12):
        if len(sequencia_12) != 12:
            return {"erro": "Necessário exatamente 12 números"}
        if self.ia is None:
            self.carregar_tudo()
        polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in sequencia_12]
        analise = MotorAnalise.analisar_janela(sequencia_12, polaridades, self.ia, eh_sinal_real=True)
        if analise["no_call"]["ativo"]:
            return {
                "sinal": "NO CALL",
                "justificativa": analise["no_call"]["motivo"],
                "no_call": True,
                "regime_recencia": self.regime_recencia,
                "motivo_real": f"NO CALL: {analise['no_call']['motivo']}",
                "regra_id": "SISTEMA_TRAVADO",
                "entropia": analise.get("entropia"),
                "probabilidade_markov": analise.get("probabilidade_markov"),
                "raciocinio_trace": analise.get("camadas", []),
                "decisao_final": {"sinal": "NO CALL", "justificativa": analise['no_call']['motivo'], "regra_id": "SISTEMA_TRAVADO"}
            }
        geometria = analise["geometria"]
        expectativas = analise["regras_posicionais"]
        direcao_ia = analise["ia"]["direcao"]
        conf_ia = analise["ia"]["confianca"]
        raciocinio_ia = analise["ia"]["raciocinio"]
        streak = analise["contexto_reversao"]["streak"]
        xadrez_len = analise["contexto_reversao"]["xadrez_len"]
        xadrez_quebrou = analise["contexto_reversao"]["xadrez_quebrou"]
        contexto_exaustao = analise["contexto_reversao"]["exaustao"]
        modo_mercado = analise["contexto_avancado"].get("modo_mercado", "NEUTRO")
        sinal_final, justificativa_final, regra_id_final = JuizHierarquicoModificado.arbitrar_sinal(
            no_call_ativo=False, motivo_nc="", expectations=expectativas, inclinacao_num=None,
            geometria_mercado=geometria,
            previsao_ia=(direcao_ia, conf_ia, raciocinio_ia), status_inversao=None,
            historico_regras=self.ia.historico_regras if self.ia else {},
            modo_mercado=modo_mercado, streak_atual=streak, xadrez_len=xadrez_len,
            xadrez_quebrou=xadrez_quebrou,
            contexto_exaustao=contexto_exaustao, probabilidade_markov=analise.get("probabilidade_markov"),
            ia_modelo=self.ia, entropia_shannon=analise.get("entropia", 0.0),
            influencia_radar=analise.get("influencia_radar")  # <-- RADAR: passa influência
        )
        validacao_contextual = getattr(self.ia, "_ultima_validacao_autoridade_contextual", {}) or {}
        if validacao_contextual.get("ativo"):
            componentes_ctx = validacao_contextual.get("componentes", {}) or {}
            resumo_componentes = []
            for nome, item in componentes_ctx.items():
                parte = (
                    f"{nome}: macro {item.get('taxa_direcao_g0_g1', 0):.2f}% direção vs "
                    f"{item.get('taxa_contraria_g0_g1', 0):.2f}% contrária (n={item.get('suporte', 0)})"
                )
                if item.get('suporte_recente', 0):
                    parte += (
                        f" | recente {item.get('taxa_direcao_recente_g0_g1', 0):.2f}% direção vs "
                        f"{item.get('taxa_contraria_recente_g0_g1', 0):.2f}% contrária "
                        f"(n={item.get('suporte_recente', 0)})"
                    )
                resumo_componentes.append(parte)
            detalhe_ctx = (
                f"Regra: {validacao_contextual.get('regra')} | Direção preservada: {validacao_contextual.get('direcao')} | "
                f"G0/G1 direção: {validacao_contextual.get('taxa_direcao_g0_g1', 0):.2f}% | "
                f"Contrária: {validacao_contextual.get('taxa_contraria_g0_g1', 0):.2f}% | "
                f"Deriva número final: {validacao_contextual.get('estado_deriva_numero_final', 'SEM_SUPORTE')} | "
                f"Fragmentação: {validacao_contextual.get('fragmentacao_contextual', False)} | "
                f"Contextos: {'; '.join(resumo_componentes) if resumo_componentes else 'sem componentes com suporte'}"
            )
            analise["camadas"].append({
                "camada": 7.5,
                "nome": "Validação Contextual da Autoridade Hierárquica",
                "resultado": validacao_contextual.get("status", "SEM_VALIDACAO"),
                "detalhe": detalhe_ctx,
                "impacto": "BLOQUEIO" if validacao_contextual.get("vetar") else "VALIDACAO"
            })
        else:
            analise["camadas"].append({
                "camada": 7.5,
                "nome": "Validação Contextual da Autoridade Hierárquica",
                "resultado": "SEM_VALIDACAO",
                "detalhe": "Nenhuma validação contextual disponível para esta janela.",
                "impacto": "NEUTRO"
            })
        if sinal_final != "NO CALL" and streak >= 6:
            sinal_final = "NO CALL"
            justificativa_final = f"Veto de streak {streak}x (segurança anti-tendência)"
            regra_id_final = "VETO_STREAK"

        # <-- RADAR: gerar relatório detalhado do Radar se disponível
        relatorio_radar = None
        if self.ia is not None and hasattr(self.ia, 'gerar_relatorio_radar'):
            try:
                relatorio_radar = self.ia.gerar_relatorio_radar(sequencia_12, polaridades, analise)
            except Exception as e:
                relatorio_radar = {"erro": str(e)}

        return {
            "sinal": sinal_final,
            "justificativa": justificativa_final,
            "confianca_ia": round(conf_ia, 2),
            "no_call": False,
            "regime_recencia": self.regime_recencia,
            "motivo_real": justificativa_final,
            "raciocinio_trace": analise["camadas"],
            "regra_id": regra_id_final,
            "entropia": analise.get("entropia"),
            "probabilidade_markov": analise.get("probabilidade_markov"),
            "simulacao_rotas_proximos_resultados": getattr(self.ia, "ultima_simulacao_rotas", {}),
            "confluencia_camadas_ampliadas": getattr(self.ia, "ultima_confluencia_camadas_ampliadas", {}),
            "validacao_contextual_autoridade": getattr(self.ia, "_ultima_validacao_autoridade_contextual", {}),
            "oposicao_causal_consolidada": getattr(self.ia, "_ultima_oposicao_causal_consolidada", {}),
            "auditoria_contrafactual_autorizacao": getattr(self.ia, "auditoria_contrafactual_autorizacao", {}),
            "relatorio_radar": relatorio_radar,  # <-- RADAR
            "decisao_final": {
                "sinal": sinal_final,
                "justificativa": justificativa_final,
                "regra_id": regra_id_final
            }
        }

    def processar_feedback_real(self, sequencia_12, sinal_indicado, regra_id, numeros_saidos, classificacao,
                                entropia_shannon=0.0, probabilidade_markov=None):
        if self.ia is None:
            self.carregar_tudo()
        polaridades = ['B' if n == 0 else ('V' if 1 <= n <= 7 else 'P') for n in sequencia_12]
        analise = MotorAnalise.analisar_janela(sequencia_12, polaridades, self.ia)
        modo_mercado = analise.get("contexto_avancado", {}).get("modo_mercado", "NEUTRO")
        geometria = analise.get("geometria", "ESTÁVEL")
        expectativas = analise.get("regras_posicionais", [])
        classificacao_limpa = classificacao.split(" ")[0].upper()
        if "LOSS" in classificacao_limpa or "FALHA" in classificacao_limpa:
            classificacao_limpa = "FALHA"
        estado_rl = self.ia.construir_estado_q_contextual(
            sequencia_12,
            polaridades,
            analise=analise,
            entropia_shannon=entropia_shannon,
            probabilidade_markov=probabilidade_markov or analise.get("probabilidade_markov")
        )
        acao_rl = "APOSTAR" if sinal_indicado != "NO CALL" else "NO_CALL"
        if classificacao_limpa in ["G0", "G1"]:
            recompensa = 1.0
        elif classificacao_limpa == "G2":
            recompensa = -0.5
        elif classificacao_limpa == "FALHA":
            recompensa = -2.0
        else:
            recompensa = 0.0
        self.ia.atualizar_q_learning(estado_rl, acao_rl, recompensa)
        contexto_analise = {
            "geometria": geometria,
            "regras_posicionais": expectativas,
            "controlador_retardador": analise.get("controlador_retardador", {}),
            "contexto_avancado": {"modo_mercado": modo_mercado},
            "entropia_shannon": entropia_shannon,
            "monte_carlo_indicou": probabilidade_markov
        }
        if classificacao_limpa in ["G0", "G1"]:
            self.ia.registrar_padrao_vencedor(contexto_analise, classificacao_limpa)
        try:
            memoria_cf = getattr(self.ia, "auditoria_contrafactual_autorizacao", None)
            if not isinstance(memoria_cf, dict):
                memoria_cf = {
                    "total": 0, "oficial_g0_g1": 0, "oposta_g0_g1": 0,
                    "no_call_protegeria_g2_falha": 0, "eventos": []
                }
                self.ia.auditoria_contrafactual_autorizacao = memoria_cf
            direcao_oficial = sinal_indicado if sinal_indicado in ("VERMELHO", "PRETO") else None
            direcao_oposta = (
                "PRETO" if direcao_oficial == "VERMELHO"
                else "VERMELHO" if direcao_oficial == "PRETO"
                else None
            )
            cores_reais = [
                "BRANCO" if int(n) == 0 else ("VERMELHO" if 1 <= int(n) <= 7 else "PRETO")
                for n in numeros_saidos[:2]
            ]
            oficial_g0_g1 = bool(
                direcao_oficial and any(c in (direcao_oficial, "BRANCO") for c in cores_reais)
            )
            oposta_g0_g1 = bool(
                direcao_oposta and any(c in (direcao_oposta, "BRANCO") for c in cores_reais)
            )
            risco_real = classificacao_limpa in ("G2", "FALHA")
            memoria_cf["total"] = int(memoria_cf.get("total", 0)) + 1
            if oficial_g0_g1:
                memoria_cf["oficial_g0_g1"] = int(memoria_cf.get("oficial_g0_g1", 0)) + 1
            if oposta_g0_g1:
                memoria_cf["oposta_g0_g1"] = int(memoria_cf.get("oposta_g0_g1", 0)) + 1
            if risco_real:
                memoria_cf["no_call_protegeria_g2_falha"] = int(
                    memoria_cf.get("no_call_protegeria_g2_falha", 0)
                ) + 1
            eventos_cf = list(memoria_cf.get("eventos", []))
            eventos_cf.append({
                "regra": regra_id,
                "direcao_oficial": direcao_oficial,
                "direcao_oposta": direcao_oposta,
                "classificacao_real": classificacao_limpa,
                "oficial_g0_g1": oficial_g0_g1,
                "oposta_g0_g1": oposta_g0_g1,
                "no_call_protegeria": risco_real,
                "validacao_contextual": dict(
                    getattr(self.ia, "_ultima_validacao_autoridade_contextual", {}) or {}
                ),
                "oposicao_causal": dict(
                    getattr(self.ia, "_ultima_oposicao_causal_consolidada", {}) or {}
                )
            })
            memoria_cf["eventos"] = eventos_cf[-500:]
        except Exception as e:
            print(f"[AUDITORIA CONTRAFACTUAL] Registro ignorado por segurança: {e}")
        if regra_id and regra_id not in ["NENHUMA", "SISTEMA_TRAVADO"]:
            self.ia.historico_regras[regra_id]["total"] += 1
            if classificacao_limpa in ["G0", "G1"]:
                self.ia.historico_regras[regra_id]["acertos"] += 1
        evento_ao_vivo = [int(n) for n in list(sequencia_12) + list(numeros_saidos)]
        cronologia_ao_vivo = list(getattr(self.ia, "cronologia_ao_vivo", []) or [])
        maior_sobreposicao = 0
        limite_sobreposicao = min(len(cronologia_ao_vivo), len(evento_ao_vivo))
        for tamanho_sobreposicao in range(limite_sobreposicao, 0, -1):
            if cronologia_ao_vivo[-tamanho_sobreposicao:] == evento_ao_vivo[:tamanho_sobreposicao]:
                maior_sobreposicao = tamanho_sobreposicao
                break
        if cronologia_ao_vivo and maior_sobreposicao == 0:
            cronologia_ao_vivo = []
        numeros_cronologicamente_novos = evento_ao_vivo[maior_sobreposicao:]
        cronologia_ao_vivo.extend(numeros_cronologicamente_novos)
        self.ia.cronologia_ao_vivo = cronologia_ao_vivo[-5000:]
        dados_novos_completos = []
        for n in numeros_cronologicamente_novos:
            cor = 'B' if n == 0 else ('V' if 1 <= n <= 7 else 'P')
            dados_novos_completos.append({"numero": n, "cor": cor})
        contexto_injecao = {
            "regras_posicionais": expectativas,
            "controlador_retardador": analise.get("controlador_retardador", {}),
            "geometria": geometria
        }
        self.ia.injetar_aprendizado_imediato(
            dados_novos_completos,
            multiplicador_peso=6,
            analise_contexto=contexto_injecao,
            salvar_na_recencia=False
        )
        self.ia.dados_recencia.extend([
            {"numero": n, "cor": ('B' if n == 0 else ('V' if 1 <= n <= 7 else 'P'))}
            for n in numeros_saidos
        ])
        self.ia.dados_recencia = self.ia.dados_recencia[-200:]
        dados_novos_para_arquivo = []
        for n in numeros_cronologicamente_novos:
            cor = 'B' if n == 0 else ('V' if 1 <= n <= 7 else 'P')
            dados_novos_para_arquivo.append({"numero": n, "cor": cor})
        recencia_atual = self.ia.dados_recencia.copy() if self.ia else []
        cronologia_ao_vivo_atual = list(getattr(self.ia, "cronologia_ao_vivo", []) or [])
        rel = adicionar_a_base_longo_prazo(dados_novos_para_arquivo, origem_feedback_ao_vivo=True)

        # <-- RADAR: processar feedback para o Radar
        if self.ia is not None and hasattr(self.ia, '_processar_feedback_radar'):
            try:
                self.ia._processar_feedback_radar(
                    sequencia_12,
                    polaridades,
                    numeros_saidos,
                    analise,
                    classificacao_limpa
                )
            except Exception as e:
                print(f"[RADAR] Falha ao processar feedback: {e}")

        self.carregar_tudo(forcar_recencia=False)
        if self.ia:
            self.ia.cronologia_ao_vivo = cronologia_ao_vivo_atual[-5000:]
            self.ia.dados_recencia = recencia_atual[-200:]
            self.ia.analise_recencia = self.ia.analisar_comportamento_pos_numero_recencia(self.ia.dados_recencia)
            self.regime_recencia = analisar_regime_recencia(self.ia.dados_recencia)
            self.ia.regime_recencia = self.regime_recencia
            self.ia.atualizar_matriz_evolutiva()
            self.ia.mapear_deriva_comportamental_numeros()
            salvar_modelo_longo_prazo(self.ia)
        return rel

    def status(self):
        volume_longo = len(self.ia.dados_longo) if self.ia and self.ia.dados_longo else 0
        volume_rec = len(self.ia.dados_recencia) if self.ia and self.ia.dados_recencia else 0
        return {
            "ia_carregada": self.ia is not None,
            "base_longa_carregada": self.base_longa_carregada,
            "recencia_injetada": self.recencia_injetada,
            "regime_recencia": self.regime_recencia,
            "volume_longo_prazo": volume_longo,
            "volume_recencia": volume_rec,
            "ultima_atualizacao": self.ultima_atualizacao.isoformat() if self.ultima_atualizacao else None
        }


motor_unificado = MotorUnificadoV1()
