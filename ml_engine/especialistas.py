from collections import defaultdict
from utils.helpers import hash_chave
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from config.settings import VERSAO_CHAVES_HASH

class EspecialistasMixin:
    """
    Mixin isolando as validações Walk-Forward 80/20 e os pesos
    dinâmicos dos Especialistas e Camadas Ampliadas.
    """
    
    def _validar_competencia_especialistas_cronologica(self, dados):
        """
        Mede cada especialista em ordem cronológica e aprende ONDE ele é
        competente: regime, geometria, direção do voto e padrão V/P recente.
        O bloco futuro valida; os mapas do treino permanecem congelados.
        """
        self.competencia_especialistas = {}
        self.competencia_contextual = defaultdict(lambda: {"total": 0, "acertos": 0})
        self.competencia_contextual_detalhada = defaultdict(lambda: {"total": 0, "acertos": 0})
        self.competencia_contextual_metricas = {}
        self.competencia_metricas = {}

        if not dados or len(dados) < 500:
            self.competencia_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        corte = int(len(dados) * 0.80)
        treino = dados[:corte]
        validacao = dados[corte:]
        if len(validacao) < 50:
            self.competencia_metricas = {"ativo": False, "motivo": "VALIDACAO_INSUFICIENTE"}
            return

        def nova_stats():
            return {"V": 0, "P": 0, "total": 0}

        mapas = {
            "MARKOV": defaultdict(nova_stats),
            "BIGRAMA": defaultdict(nova_stats),
            "TRIGRAMA": defaultdict(nova_stats),
            "NUMERO": defaultdict(nova_stats),
            "GEOMETRIA": defaultdict(nova_stats),
            "ESPELHO_INVERSAO": defaultdict(nova_stats),
        }
        mapa_projetiva_respeito = defaultdict(
            lambda: {"total": 0, "respeitada": 0, "nao_respeitada": 0}
        )

        def registrar(mapa, chave, c0, c1):
            stats = mapa[chave]
            stats["total"] += 1
            if c0 in ("V", "B") or c1 in ("V", "B"):
                stats["V"] += 1
            if c0 in ("P", "B") or c1 in ("P", "B"):
                stats["P"] += 1

        for i in range(11, len(treino) - 2):
            janela = treino[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            c0, c1 = treino[i+1]["cor"], treino[i+2]["cor"]

            for ordem in range(1, 7):
                registrar(mapas["MARKOV"], (ordem, tuple(pol[-ordem:])), c0, c1)
            registrar(mapas["BIGRAMA"], tuple(nums[-2:]), c0, c1)
            registrar(mapas["TRIGRAMA"], tuple(nums[-3:]), c0, c1)
            registrar(mapas["NUMERO"], nums[-1], c0, c1)

            geo = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
            registrar(mapas["GEOMETRIA"], geo, c0, c1)

            chave_espelho = self._identificar_contexto_espelho_inversao(nums, pol)
            if chave_espelho:
                registrar(mapas["ESPELHO_INVERSAO"], chave_espelho, c0, c1)

            # PROJETIVA não aprende V contra P. Aprende RESPEITADA contra NÃO RESPEITADA.
            for pos, numero in enumerate(nums):
                if 1 <= numero <= 7 and pos + numero in (11, 12):
                    stats_proj = mapa_projetiva_respeito[numero]
                    stats_proj["total"] += 1
                    respeitada = c0 in ("V", "B") or c1 in ("V", "B")
                    if respeitada:
                        stats_proj["respeitada"] += 1
                    else:
                        stats_proj["nao_respeitada"] += 1

        desempenho = defaultdict(lambda: {"total": 0, "acertos": 0})

        def voto_mapa(nome, chave, minimo=12, margem_min=0.06):
            stats = mapas[nome].get(chave)
            if not stats or stats["total"] < minimo:
                return None
            taxa_v = stats["V"] / stats["total"]
            taxa_p = stats["P"] / stats["total"]
            if abs(taxa_v - taxa_p) < margem_min:
                return None
            return "V" if taxa_v > taxa_p else "P"

        def projetiva_respeitada(numero, minimo=30, taxa_minima=0.58):
            stats = mapa_projetiva_respeito.get(numero)
            if not stats or stats["total"] < minimo:
                return False
            return (stats["respeitada"] / stats["total"]) >= taxa_minima

        inicio_global = max(corte, 11)
        for i in range(inicio_global, len(dados) - 2):
            janela = dados[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            c0, c1 = dados[i+1]["cor"], dados[i+2]["cor"]
            regime = self._detectar_regime_temporal(pol)
            geo = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
            padrao = "".join(pol[-3:])
            numero_final = nums[-1]

            votos = {}
            for ordem in range(6, 0, -1):
                voto = voto_mapa(
                    "MARKOV", (ordem, tuple(pol[-ordem:])),
                    minimo={6:8, 5:10, 4:14, 3:20, 2:28, 1:35}[ordem]
                )
                if voto:
                    votos["MARKOV"] = voto
                    break
            votos["BIGRAMA"] = voto_mapa("BIGRAMA", tuple(nums[-2:]), minimo=15)
            votos["TRIGRAMA"] = voto_mapa("TRIGRAMA", tuple(nums[-3:]), minimo=10)
            votos["NUMERO"] = voto_mapa("NUMERO", numero_final, minimo=30)
            votos["GEOMETRIA"] = voto_mapa("GEOMETRIA", geo, minimo=30)

            chave_espelho = self._identificar_contexto_espelho_inversao(nums, pol)
            if chave_espelho:
                votos["ESPELHO_INVERSAO"] = voto_mapa(
                    "ESPELHO_INVERSAO", chave_espelho, minimo=15
                )

            projecoes_ativas = [
                numero for pos, numero in enumerate(nums)
                if 1 <= numero <= 7 and pos + numero in (11, 12)
            ]
            if projecoes_ativas and all(
                projetiva_respeitada(numero) for numero in projecoes_ativas
            ):
                votos["PROJETIVA"] = "V"

            for especialista, voto in votos.items():
                if voto not in ("V", "P"):
                    continue
                acertou = c0 in (voto, "B") or c1 in (voto, "B")
                desempenho[especialista]["total"] += 1
                if acertou:
                    desempenho[especialista]["acertos"] += 1

                # Compatibilidade com a competência contextual anterior.
                chave_contexto = f"{especialista}|{regime}"
                self.competencia_contextual[chave_contexto]["total"] += 1
                if acertou:
                    self.competencia_contextual[chave_contexto]["acertos"] += 1

                # Nova memória contextual granular. Não muda o voto do especialista;
                # mede a competência do especialista exatamente no contexto do voto.
                chaves_detalhadas = [
                    f"{especialista}|REGIME|{regime}",
                    f"{especialista}|REGIME_GEOMETRIA|{regime}|{geo}",
                    f"{especialista}|REGIME_DIRECAO|{regime}|{voto}",
                    f"{especialista}|REGIME_PADRAO|{regime}|{padrao}",
                    f"{especialista}|GEOMETRIA_DIRECAO|{geo}|{voto}",
                    f"{especialista}|EXATO|{regime}|{geo}|{voto}|{padrao}|N={numero_final}",
                ]
                for chave in chaves_detalhadas:
                    self.competencia_contextual_detalhada[chave]["total"] += 1
                    if acertou:
                        self.competencia_contextual_detalhada[chave]["acertos"] += 1

        for especialista, stats in desempenho.items():
            total = stats["total"]
            acertos = stats["acertos"]
            taxa = (acertos / total) if total else 0.0
            self.competencia_especialistas[especialista] = {
                "total_validacao": total,
                "acertos_g0_g1": acertos,
                "taxa_g0_g1": round(taxa * 100, 2)
            }

        contextos_com_suporte = sum(
            1 for stats in self.competencia_contextual_detalhada.values()
            if stats["total"] >= 20
        )
        self.competencia_contextual_metricas = {
            "ativo": True,
            "contextos_detalhados_aprendidos": len(self.competencia_contextual_detalhada),
            "contextos_com_suporte_minimo_20": contextos_com_suporte,
            "dimensoes": [
                "REGIME", "REGIME_GEOMETRIA", "REGIME_DIRECAO",
                "REGIME_PADRAO", "GEOMETRIA_DIRECAO", "EXATO"
            ],
            "objetivo": "PESAR_CADA_ESPECIALISTA_ONDE_ELE_PROVOU_COMPETENCIA_G0_G1",
            "validacao": "80_20_CRONOLOGICA_CONGELADA"
        }
        self.competencia_metricas = {
            "ativo": True,
            "treino_cronologico_registros": len(treino),
            "validacao_cronologica_registros": len(validacao),
            "especialistas_validados": len(self.competencia_especialistas),
            "metodo": "80_20_CRONOLOGICO_G0_G1",
            "competencia_contextual_detalhada_ativa": True
        }

        self.filtro_discriminativo_metricas = {
            "ativo": True,
            "versao": 5,
            "contextos_contextuais_disponiveis": len(self.competencia_contextual_detalhada),
            "contextos_com_suporte_minimo_20": contextos_com_suporte,
            "especialistas_validados": len(self.competencia_especialistas),
            "projecoes_respeito_ativas": bool(getattr(self, "projecoes_respeito_metricas", {}).get("ativo")),
            "cartografia_no_veto_discriminativo": bool(getattr(self, "cartografia_xls_metricas", {}).get("ativo")),
            "contextos_cartografia_padroes_disponiveis": len(getattr(self, "cartografia_padroes_xls", {})),
            "contextos_cartografia_trajetorias_disponiveis": len(getattr(self, "cartografia_projecoes_trajetoria", {})),
            "rota_cartografia_independe_especialistas_alinhados": False,
            "projetiva_no_veto_discriminativo": False,
            "motivo_exclusao_projetiva_veto": "ESPECIALISTAS_SAO_FONTES_DE_DIRECAO_COMPETENCIA_NAO_FONTES_ISOLADAS_DE_ERRO",
            "objetivo": "VETAR_INSTABILIDADE_DECISORIA_COM_RISCO_HISTORICO_G2_FALHA_COMPROVADO",
            "criterio_seletividade": "ENTROPIA_DECISORIA_MAIS_CONFLITO_FAMILIAS_INDEPENDENTES_MAIS_MEMORIA_HISTORICA_G2_FALHA_CARTOGRAFIA_AGRAVANTE",
            "acao_permitida": "VETAR_PARA_NO_CALL",
            "altera_direcao": False,
            "validacao_origem": "COMPETENCIA_80_20_CRONOLOGICA_CONGELADA",
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        }


    def _treinar_memoria_conflitos_base_longa(self, dados):
        """Reconstrói a memória de conflitos diretamente da cronologia histórica.

        Corrige a lacuna criada quando a auditoria passou a ser congelada: a
        memória deixa de depender de ``aprender_durante_auditoria=True``.
        Nenhuma regra de NO CALL, peso de RECÊNCIA ou direção fixa é alterada.
        """
        self.memoria_conflitos = defaultdict(lambda: {
            "total": 0, "V_g0g1": 0, "P_g0g1": 0,
            "V_g0": 0, "P_g0": 0, "falhas_v": 0, "falhas_p": 0
        })
        if not dados or len(dados) < 15:
            self.memoria_conflitos_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        eventos = 0
        for i in range(11, len(dados) - 2):
            janela = dados[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
            expectativas = MotorContagensProjetivas.mapear_janela(nums, pol, geometria, None)
            modo = self._detectar_regime_temporal(pol)
            prob_markov = self.calcular_probabilidade_exata_markov(pol)
            correcoes = [dados[i+1]["cor"], dados[i+2]["cor"]]
            self.registrar_resultado_conflito(expectativas, geometria, modo, prob_markov, correcoes)
            eventos += 1

        self.memoria_conflitos_metricas = {
            "ativo": True,
            "metodo": "RECONSTRUCAO_CRONOLOGICA_BASE_LONGA",
            "eventos_processados": eventos,
            "contextos_aprendidos": len(self.memoria_conflitos),
            "depende_auditoria_mutavel": False,
            "altera_no_call": False,
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        }

    def _validar_regras_posicionais_cronologica(self, dados):
        """Mede cada regra posicional em 80/20 cronológico congelado.

        A regra continua com a direção original definida pelo motor. Esta camada
        apenas mede G0/G1 real e restaura uma métrica que havia zerado porque o
        relatório consultava o histórico local da auditoria congelada.
        """
        self.regras_competencia_cronologica = {}
        self.regras_competencia_metricas = {}
        if not dados or len(dados) < 500:
            self.regras_competencia_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        corte = int(len(dados) * 0.80)
        desempenho = defaultdict(lambda: {"total": 0, "acertos": 0, "g0": 0, "g1": 0})
        for i in range(max(corte, 11), len(dados) - 2):
            janela = dados[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
            regras = MotorContagensProjetivas.mapear_janela(nums, pol, geometria, None)
            c0, c1 = dados[i+1]["cor"], dados[i+2]["cor"]
            for regra in regras:
                nome = regra.get("tipo_regra", "DESCONHECIDO")
                direcao = regra.get("direcao")
                letra = "V" if direcao == "VERMELHO" else ("P" if direcao == "PRETO" else None)
                if letra is None:
                    continue
                st = desempenho[nome]
                st["total"] += 1
                if c0 in (letra, "B"):
                    st["acertos"] += 1
                    st["g0"] += 1
                elif c1 in (letra, "B"):
                    st["acertos"] += 1
                    st["g1"] += 1

        self.regras_competencia_cronologica = {
            nome: {
                "total_validacao": st["total"],
                "acertos_g0_g1": st["acertos"],
                "g0": st["g0"],
                "g1": st["g1"],
                "taxa_g0_g1": round((st["acertos"] / max(1, st["total"])) * 100, 2)
            }
            for nome, st in desempenho.items()
        }
        boas = sum(
            1 for st in self.regras_competencia_cronologica.values()
            if st["total_validacao"] >= 5 and st["taxa_g0_g1"] >= 55.0
        )
        self.regras_competencia_metricas = {
            "ativo": True,
            "metodo": "80_20_CRONOLOGICO_CONGELADO_G0_G1",
            "regras_avaliadas": len(self.regras_competencia_cronologica),
            "regras_com_boa_performance": boas,
            "altera_direcao_original": False,
            "altera_regras_no_call": False
        }

    def _chaves_camadas_ampliadas(self, nums, pol):
        """Extrai contextos estatísticos sem consultar memórias futuras."""
        if len(nums) < 12 or len(pol) < 12:
            return {}
        ultimo, penultimo = int(nums[-1]), int(nums[-2])
        delta = ultimo - penultimo
        direcao_delta = "SOBE" if delta > 0 else ("DESCE" if delta < 0 else "IGUAL")

        streak = 1
        if pol[-1] in ("V", "P"):
            for cor in reversed(pol[:-1]):
                if cor == pol[-1]:
                    streak += 1
                else:
                    break
        else:
            streak = 0

        xadrez = 1
        for i in range(len(pol) - 1, 0, -1):
            if pol[i] in ("V", "P") and pol[i-1] in ("V", "P") and pol[i] != pol[i-1]:
                xadrez += 1
            else:
                break

        geo = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
        regras = MotorContagensProjetivas.mapear_janela(nums, pol, geo, None)

        chaves = {
            "NUMEROLOGIA_ESTATISTICA": [
                f"ULTIMO={ultimo}",
                f"PAR={penultimo}-{ultimo}",
                f"DELTA={min(abs(delta), 14)}|{direcao_delta}",
                f"FAIXA={penultimo//4}-{ultimo//4}|PARIDADE={penultimo%2}-{ultimo%2}",
            ],
            "DNA_NUMERICO": [
                "DNA3=" + "-".join(map(str, nums[-3:])),
                "DNA4=" + "-".join(map(str, nums[-4:])),
            ],
            "FECHAMENTO_NUMERICO": [
                f"P3={''.join(pol[-3:])}|N={ultimo}",
                f"P4={''.join(pol[-4:])}|PAR={penultimo}-{ultimo}",
                f"P5={''.join(pol[-5:])}|N={ultimo}",
            ],
            "STREAK": [f"COR={pol[-1]}|LEN={min(streak, 8)}"] if streak >= 2 else [],
            "XADREZ": [f"LEN={min(xadrez, 8)}|ULT={pol[-1]}"] if xadrez >= 3 else [],
            "REGRAS_POSICIONAIS": [
                f"{r.get('tipo_regra','SEM_REGRA')}|DIR={r.get('direcao','NEUTRO')}"
                for r in regras
            ],
        }
        return chaves

    @staticmethod
    def _registrar_camadas_ampliadas(mapas, chaves, c0, c1):
        for camada, lista_chaves in chaves.items():
            mapa = mapas.setdefault(camada, {})
            for chave in lista_chaves:
                stats = mapa.setdefault(chave, {"V": 0, "P": 0, "total": 0})
                stats["total"] += 1
                if c0 in ("V", "B") or c1 in ("V", "B"):
                    stats["V"] += 1
                if c0 in ("P", "B") or c1 in ("P", "B"):
                    stats["P"] += 1

    @staticmethod
    def _voto_stats_camadas_ampliadas(stats, minimo=12, margem=0.06):
        if not stats or int(stats.get("total", 0)) < minimo:
            return None
        total = max(1, int(stats.get("total", 0)))
        taxa_v = float(stats.get("V", 0)) / total
        taxa_p = float(stats.get("P", 0)) / total
        if abs(taxa_v - taxa_p) < margem:
            return None
        return "V" if taxa_v > taxa_p else "P"

    def _validar_competencia_camadas_ampliadas_cronologica(self, dados):
        """
        MAIN 74: mede NUMEROLOGIA estatística, DNA, fechamento, regras
        posicionais, streak e xadrez em 80/20 cronológico congelado. Depois
        monta a memória operacional com toda a base. Não cria veto/NO CALL.
        """
        self.camadas_ampliadas_mapas = {}
        self.camadas_ampliadas_competencia = {}
        self.camadas_ampliadas_contextual = {}
        self.camadas_ampliadas_metricas = {}

        if not dados or len(dados) < 500:
            self.camadas_ampliadas_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        corte = int(len(dados) * 0.80)
        treino = dados[:corte]
        mapas_treino = {}
        for i in range(11, len(treino) - 2):
            janela = treino[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            self._registrar_camadas_ampliadas(
                mapas_treino, self._chaves_camadas_ampliadas(nums, pol),
                treino[i+1]["cor"], treino[i+2]["cor"]
            )

        desempenho = defaultdict(lambda: {"total": 0, "acertos": 0})
        contextual = defaultdict(lambda: {"total": 0, "acertos": 0})
        minimos = {
            "NUMEROLOGIA_ESTATISTICA": 20,
            "DNA_NUMERICO": 8,
            "FECHAMENTO_NUMERICO": 12,
            "REGRAS_POSICIONAIS": 20,
            "STREAK": 25,
            "XADREZ": 25,
        }

        for i in range(max(corte, 11), len(dados) - 2):
            janela = dados[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            c0, c1 = dados[i+1]["cor"], dados[i+2]["cor"]
            regime = self._detectar_regime_temporal(pol)
            chaves = self._chaves_camadas_ampliadas(nums, pol)
            for camada, lista_chaves in chaves.items():
                votos = []
                for chave in lista_chaves:
                    voto = self._voto_stats_camadas_ampliadas(
                        mapas_treino.get(camada, {}).get(chave),
                        minimos.get(camada, 20)
                    )
                    if voto:
                        votos.append(voto)
                if not votos:
                    continue
                v = votos.count("V")
                p = votos.count("P")
                if v == p:
                    continue
                voto = "V" if v > p else "P"
                acertou = c0 in (voto, "B") or c1 in (voto, "B")
                desempenho[camada]["total"] += 1
                contextual[f"{camada}|{regime}"]["total"] += 1
                if acertou:
                    desempenho[camada]["acertos"] += 1
                    contextual[f"{camada}|{regime}"]["acertos"] += 1

        self.camadas_ampliadas_competencia = {
            camada: {
                "total_validacao": stats["total"],
                "acertos_g0_g1": stats["acertos"],
                "taxa_g0_g1": round((stats["acertos"] / max(1, stats["total"])) * 100, 2),
            }
            for camada, stats in desempenho.items()
        }
        self.camadas_ampliadas_contextual = {
            chave: dict(stats) for chave, stats in contextual.items()
        }

        # Memória operacional aprende a base completa somente após a validação
        # congelada ter medido a competência de cada camada.
        mapas_operacionais = {}
        for i in range(11, len(dados) - 2):
            janela = dados[i-11:i+1]
            nums = [d["numero"] for d in janela]
            pol = [d["cor"] for d in janela]
            self._registrar_camadas_ampliadas(
                mapas_operacionais, self._chaves_camadas_ampliadas(nums, pol),
                dados[i+1]["cor"], dados[i+2]["cor"]
            )
        self.camadas_ampliadas_mapas = mapas_operacionais
        self.camadas_ampliadas_metricas = {
            "ativo": True,
            "versao": 1,
            "metodo": "80_20_CRONOLOGICO_CONGELADO_COMPETENCIA_BASE_COMPLETA_OPERACIONAL",
            "camadas_estudadas": list(minimos.keys()),
            "camadas_validadas": len(self.camadas_ampliadas_competencia),
            "camadas_com_voto_estatistico_validado": list(self.camadas_ampliadas_competencia.keys()),
            "regras_posicionais_ativas_nativamente_no_juiz": True,
            "regras_posicionais_competencia_cronologica": getattr(self, "regras_competencia_metricas", {}),
            "contextos_competencia_aprendidos": len(self.camadas_ampliadas_contextual),
            "contextos_operacionais_aprendidos": sum(len(m) for m in self.camadas_ampliadas_mapas.values()),
            "objetivo": "APRENDER_ONDE_CADA_CAMADA_ACERTA_G0_G1_E_DAR_VOZ_CONTEXTUAL",
            "cria_no_call": False,
            "altera_regras_no_call": False,
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH,
        }

    def obter_voto_camadas_ampliadas(self, sub_num, sub_pol):
        """Confluência aditiva: só dá voz à camada que provou competência."""
        if not getattr(self, "camadas_ampliadas_metricas", {}).get("ativo"):
            return {"direcao": "NEUTRO", "peso": 0.0, "votos": []}

        regime = self._detectar_regime_temporal(sub_pol)
        chaves = self._chaves_camadas_ampliadas(sub_num, sub_pol)
        minimos = {
            "NUMEROLOGIA_ESTATISTICA": 20, "DNA_NUMERICO": 8,
            "FECHAMENTO_NUMERICO": 12, "REGRAS_POSICIONAIS": 20,
            "STREAK": 25, "XADREZ": 25,
        }
        votos_finais = []
        score_v = score_p = 0.0

        for camada, lista_chaves in chaves.items():
            comp = self.camadas_ampliadas_competencia.get(camada, {})
            total_comp = int(comp.get("total_validacao", 0))
            taxa_global = float(comp.get("taxa_g0_g1", 0.0)) / 100.0
            if total_comp < 30 or taxa_global < 0.74:
                continue

            ctx = self.camadas_ampliadas_contextual.get(f"{camada}|{regime}", {})
            total_ctx = int(ctx.get("total", 0))
            taxa_ctx = (float(ctx.get("acertos", 0)) / total_ctx) if total_ctx else taxa_global
            taxa_comp = ((taxa_global * 0.40) + (taxa_ctx * 0.60)) if total_ctx >= 25 else taxa_global
            if taxa_comp < 0.76:
                continue

            votos = []
            suportes = []
            for chave in lista_chaves:
                stats = self.camadas_ampliadas_mapas.get(camada, {}).get(chave)
                voto = self._voto_stats_camadas_ampliadas(stats, minimos.get(camada, 20))
                if voto:
                    votos.append(voto)
                    suportes.append(int(stats.get("total", 0)))
            if not votos or votos.count("V") == votos.count("P"):
                continue

            direcao = "V" if votos.count("V") > votos.count("P") else "P"
            suporte = max(suportes) if suportes else 0
            # Peso deliberadamente limitado: camada ampliada complementa o
            # motor existente e nunca substitui as memórias já funcionais.
            peso = min(6.0, max(1.0, (taxa_comp - 0.74) * 30.0 + min(suporte, 100) / 100.0))
            if direcao == "V":
                score_v += peso
            else:
                score_p += peso
            votos_finais.append({
                "camada": camada,
                "direcao": "VERMELHO" if direcao == "V" else "PRETO",
                "peso": round(peso, 2),
                "taxa_competencia_g0_g1": round(taxa_comp * 100, 2),
                "suporte": suporte,
            })

        score_v = min(score_v, 18.0)
        score_p = min(score_p, 18.0)
        if score_v == score_p:
            direcao, peso = "NEUTRO", 0.0
        elif score_v > score_p:
            direcao, peso = "VERMELHO", score_v - score_p
        else:
            direcao, peso = "PRETO", score_p - score_v

        self.ultima_confluencia_camadas_ampliadas = {
            "regime": regime,
            "score_vermelho": round(score_v, 2),
            "score_preto": round(score_p, 2),
            "direcao": direcao,
            "peso_liquido": round(peso, 2),
            "votos": votos_finais,
        }
        return {"direcao": direcao, "peso": round(peso, 2), "votos": votos_finais}

    def _avaliar_competencia_contextual_detalhada(self, especialista, regime, geometria, direcao, padrao, numero_final):
        """
        Retorna a taxa contextual ajustada e o suporte do especialista sem
        alterar o peso já usado pelo motor. É uma leitura paralela exclusiva
        do filtro discriminativo G0/G1 x G2+.
        """
        global_stats = self.competencia_especialistas.get(especialista, {})
        total_global = int(global_stats.get("total_validacao", 0))
        taxa_global = float(global_stats.get("taxa_g0_g1", 0.0)) / 100.0
        if total_global < 30 or taxa_global <= 0:
            return {"valido": False, "taxa_g0_g1": 0.0, "risco_g2_mais": 1.0, "suporte": 0}

        geometria = geometria or "NEUTRO"
        direcao = direcao if direcao in ("V", "P") else "NEUTRO"
        padrao = padrao or "SEM_PADRAO"
        numero_final = int(numero_final) if numero_final is not None else -1

        candidatos = [
            (f"{especialista}|EXATO|{regime}|{geometria}|{direcao}|{padrao}|N={numero_final}", 20, 1.00),
            (f"{especialista}|REGIME_PADRAO|{regime}|{padrao}", 25, 0.90),
            (f"{especialista}|REGIME_GEOMETRIA|{regime}|{geometria}", 30, 0.85),
            (f"{especialista}|GEOMETRIA_DIRECAO|{geometria}|{direcao}", 30, 0.80),
            (f"{especialista}|REGIME_DIRECAO|{regime}|{direcao}", 35, 0.75),
            (f"{especialista}|REGIME|{regime}", 40, 0.65),
        ]

        evidencias = []
        for chave, suporte_min, especificidade in candidatos:
            stats = self.competencia_contextual_detalhada.get(chave)
            if not stats:
                continue
            total = int(stats.get("total", 0))
            if total < suporte_min:
                continue
            taxa_ctx = float(stats.get("acertos", 0)) / max(total, 1)
            forca_ctx = total / (total + suporte_min)
            taxa_ajustada = (taxa_ctx * forca_ctx) + (taxa_global * (1.0 - forca_ctx))
            peso = especificidade * (total / (total + 50.0))
            evidencias.append((taxa_ajustada, total, peso, chave.split("|", 2)[1]))

        if evidencias:
            soma_pesos = sum(item[2] for item in evidencias)
            taxa = sum(item[0] * item[2] for item in evidencias) / max(soma_pesos, 1e-9)
            suporte = max(item[1] for item in evidencias)
            niveis = [item[3] for item in evidencias]
        else:
            legado = self.competencia_contextual.get(f"{especialista}|{regime}")
            if legado and int(legado.get("total", 0)) >= 30:
                total_legado = int(legado.get("total", 0))
                taxa_ctx = float(legado.get("acertos", 0)) / max(total_legado, 1)
                taxa = (taxa_global * 0.40) + (taxa_ctx * 0.60)
                suporte = total_legado
                niveis = ["REGIME_LEGADO"]
            else:
                taxa = taxa_global
                suporte = total_global
                niveis = ["GLOBAL"]

        risco_contexto = max(0.0, min(1.0, 1.0 - taxa))
        risco_global = max(0.0, min(1.0, 1.0 - taxa_global))
        lift_risco = risco_contexto - risco_global
        razao_risco = risco_contexto / max(risco_global, 1e-9)

        return {
            "valido": suporte >= int(self.filtro_discriminativo_config.get("suporte_contextual_minimo", 30)),
            "taxa_g0_g1": taxa,
            "risco_g2_mais": risco_contexto,
            "risco_global_especialista": risco_global,
            "lift_risco": lift_risco,
            "razao_risco": razao_risco,
            "suporte": suporte,
            "niveis": niveis
        }
    def _peso_competencia(self, especialista, regime, geometria=None, direcao=None, padrao=None, numero_final=None):
        """
        Peso do especialista condicionado ao contexto em que ele está votando.
        Usa validação cronológica e backoff de especificidade; nunca troca a
        direção do voto, apenas mede quanto confiar naquele especialista agora.
        """
        global_stats = self.competencia_especialistas.get(especialista, {})
        total_global = int(global_stats.get("total_validacao", 0))
        taxa_global = float(global_stats.get("taxa_g0_g1", 0.0)) / 100.0
        if total_global < 30 or taxa_global < 0.58:
            return 0.0

        geometria = geometria or "NEUTRO"
        direcao = direcao if direcao in ("V", "P") else "NEUTRO"
        padrao = padrao or "SEM_PADRAO"
        numero_final = int(numero_final) if numero_final is not None else -1

        candidatos = [
            (f"{especialista}|EXATO|{regime}|{geometria}|{direcao}|{padrao}|N={numero_final}", 20, 1.00),
            (f"{especialista}|REGIME_PADRAO|{regime}|{padrao}", 25, 0.90),
            (f"{especialista}|REGIME_GEOMETRIA|{regime}|{geometria}", 30, 0.85),
            (f"{especialista}|GEOMETRIA_DIRECAO|{geometria}|{direcao}", 30, 0.80),
            (f"{especialista}|REGIME_DIRECAO|{regime}|{direcao}", 35, 0.75),
            (f"{especialista}|REGIME|{regime}", 40, 0.65),
        ]

        evidencias = []
        for chave, suporte_min, especificidade in candidatos:
            stats = self.competencia_contextual_detalhada.get(chave)
            if not stats:
                continue
            total = int(stats.get("total", 0))
            if total < suporte_min:
                continue
            taxa_ctx = stats.get("acertos", 0) / max(total, 1)
            # Shrinkage para a taxa global evita supervalorizar contexto pequeno.
            forca_ctx = total / (total + suporte_min)
            taxa_ajustada = (taxa_ctx * forca_ctx) + (taxa_global * (1.0 - forca_ctx))
            evidencias.append((taxa_ajustada, total, especificidade))

        if evidencias:
            # Os contextos mais específicos pesam mais, sem deixar o maior volume
            # esmagar a leitura contextual.
            soma_pesos = sum(
                esp * (total / (total + 50.0))
                for _, total, esp in evidencias
            )
            taxa = sum(
                taxa * esp * (total / (total + 50.0))
                for taxa, total, esp in evidencias
            ) / max(soma_pesos, 1e-9)
            suporte_contextual = max(total for _, total, _ in evidencias)
        else:
            legado = self.competencia_contextual.get(f"{especialista}|{regime}")
            if legado and legado.get("total", 0) >= 30:
                taxa_ctx = legado["acertos"] / legado["total"]
                taxa = (taxa_global * 0.40) + (taxa_ctx * 0.60)
                suporte_contextual = legado["total"]
            else:
                taxa = taxa_global
                suporte_contextual = total_global

        if suporte_contextual < 30 or taxa < 0.58:
            return 0.0
        if taxa >= 0.84:
            return 4.0
        if taxa >= 0.80:
            return 3.5
        if taxa >= 0.76:
            return 3.0
        if taxa >= 0.70:
            return 2.0
        if taxa >= 0.63:
            return 1.25
        return 0.5

    def obter_voto_competencia_especialistas(self, sub_num, sub_pol, analise_contexto=None):
        """
        Consolida especialistas validados e pesa cada voto pela competência
        comprovada no contexto atual. A direção original de cada especialista
        é preservada.
        """
        if not self.competencia_metricas.get("ativo") or len(sub_num) < 12:
            return {"direcao": "NEUTRO", "peso": 0.0, "fontes": []}

        regime = self._detectar_regime_temporal(sub_pol)
        geo = (
            (analise_contexto or {}).get("geometria")
            or AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
        )
        padrao = "".join(sub_pol[-3:])
        numero_final = sub_num[-1]
        votos = []

        def adicionar(nome, direcao):
            if direcao not in ("V", "P"):
                return
            peso = self._peso_competencia(
                nome, regime, geo, direcao, padrao, numero_final
            )
            if peso > 0:
                votos.append((nome, direcao, peso))

        prob_markov = self.calcular_probabilidade_exata_markov(sub_pol)
        if prob_markov.get("V", 0) > prob_markov.get("P", 0):
            adicionar("MARKOV", "V")
        elif prob_markov.get("P", 0) > prob_markov.get("V", 0):
            adicionar("MARKOV", "P")

        bi = self.estatisticas_bigramas_globais.get(f"{sub_num[-2]}-{sub_num[-1]}")
        if bi and bi.get("total", 0) >= 15:
            tv = (bi["V_g0"] + bi["V_g1"]) / bi["total"]
            tp = (bi["P_g0"] + bi["P_g1"]) / bi["total"]
            if abs(tv - tp) >= 0.06:
                adicionar("BIGRAMA", "V" if tv > tp else "P")

        tri = self.estatisticas_trigramas_globais.get(
            f"{sub_num[-3]}-{sub_num[-2]}-{sub_num[-1]}"
        )
        if tri and tri.get("total", 0) >= 10:
            tv = (tri["V_g0"] + tri["V_g1"]) / tri["total"]
            tp = (tri["P_g0"] + tri["P_g1"]) / tri["total"]
            if abs(tv - tp) >= 0.06:
                adicionar("TRIGRAMA", "V" if tv > tp else "P")

        stats_num = self.unidade_analise.get(sub_num[-1], {})
        if stats_num.get("ocorrencias", 0) >= 30:
            fv = float(stats_num.get("freq_v", 0.0))
            fp = float(stats_num.get("freq_p", 0.0))
            if abs(fv - fp) >= 0.06:
                adicionar("NUMERO", "V" if fv > fp else "P")

        direcao_geo = None
        if geo == "CICLO_FECHADO_PVVP":
            direcao_geo = "V"
        elif geo == "CICLO_FECHADO_VPPV":
            direcao_geo = "P"
        elif geo == "SATURAÇÃO ESTRUTURAL (V)":
            direcao_geo = "P"
        elif geo == "SATURAÇÃO ESTRUTURAL (P)":
            direcao_geo = "V"
        adicionar("GEOMETRIA", direcao_geo)

        voto_espelho = self.obter_voto_espelho_inversao(sub_num, sub_pol)
        if voto_espelho.get("direcao") == "VERMELHO":
            adicionar("ESPELHO_INVERSAO", "V")
        elif voto_espelho.get("direcao") == "PRETO":
            adicionar("ESPELHO_INVERSAO", "P")

        # Contagem projetiva: somente RESPEITADA/NÃO RESPEITADA.
        # Nunca gera voto PRETO. A regra V3 original continua VERMELHA.
        projecoes_ativas = [
            numero for pos, numero in enumerate(sub_num)
            if 1 <= numero <= 7 and pos + numero in (11, 12)
        ]
        votos_respeito = []
        for numero in projecoes_ativas:
            leitura = self._obter_respeito_projecao_contextual(
                numero, sub_num, sub_pol
            )
            if leitura.get("suporte", 0) >= 30:
                votos_respeito.append(leitura.get("taxa_respeito", 0.0))
        if votos_respeito and min(votos_respeito) >= 0.58:
            adicionar("PROJETIVA", "V")

        score_v = sum(p for _, d, p in votos if d == "V")
        score_p = sum(p for _, d, p in votos if d == "P")
        margem = abs(score_v - score_p)
        if margem < 1.0 or max(score_v, score_p) <= 0:
            return {
                "direcao": "NEUTRO", "peso": 0.0, "fontes": votos,
                "regime": regime, "geometria": geo
            }

        direcao = "VERMELHO" if score_v > score_p else "PRETO"
        peso = min(15.0, 4.0 + (margem * 2.0))
        return {
            "direcao": direcao,
            "peso": round(peso, 2),
            "fontes": votos,
            "regime": regime,
            "geometria": geo,
            "score_v": round(score_v, 2),
            "score_p": round(score_p, 2)
        }
