import numpy as np
from collections import defaultdict
from utils.helpers import hash_chave
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas

class CartografiaMixin:
    """
    Mixin que isola o mapeamento contextual do Excel, cruzando 
    Padrões, Regras e Projeções.
    """
    
    @staticmethod
    def _resultado_ate_g1(c0, c1, letra):
        if c0 in (letra, "B"):
            return "G0"
        if c1 in (letra, "B"):
            return "G1"
        return "FALHA"

    def _chaves_cartografia_padrao(self, sub_num, sub_pol):
        """Cria chaves hierárquicas para streak, pares/trincas e xadrez."""
        if not sub_num or not sub_pol:
            return []
        chaves = []
        max_tam = min(8, len(sub_pol))
        for tam in range(1, max_tam + 1):
            cores = "".join(sub_pol[-tam:])
            nums = "-".join(str(n) for n in sub_num[-tam:])
            ultimo = sub_num[-1]
            bi = "-".join(str(n) for n in sub_num[-2:]) if len(sub_num) >= 2 else str(ultimo)
            tri = "-".join(str(n) for n in sub_num[-3:]) if len(sub_num) >= 3 else bi

            if tam == 1:
                chaves.append(f"NUMERO|N={ultimo}")
            if tam >= 2 and len(set(cores)) == 1 and "B" not in cores:
                chaves.extend([
                    f"STREAK|T={tam}|C={cores[-1]}",
                    f"STREAK_NUM|T={tam}|C={cores[-1]}|ULT={ultimo}",
                    f"STREAK_BI|T={tam}|C={cores[-1]}|BI={bi}",
                    f"STREAK_TRI|T={tam}|C={cores[-1]}|TRI={tri}",
                    f"STREAK_EXATO|T={tam}|C={cores[-1]}|NUM={nums}",
                ])
            if tam >= 3 and "B" not in cores:
                eh_xadrez = all(cores[j] != cores[j - 1] for j in range(1, len(cores)))
                if eh_xadrez:
                    chaves.extend([
                        f"XADREZ|T={tam}|C={cores}",
                        f"XADREZ_NUM|T={tam}|C={cores}|ULT={ultimo}",
                        f"XADREZ_BI|T={tam}|C={cores}|BI={bi}",
                        f"XADREZ_TRI|T={tam}|C={cores}|TRI={tri}",
                        f"XADREZ_EXATO|T={tam}|C={cores}|NUM={nums}",
                    ])
        return list(dict.fromkeys(chaves))

    
    def _obter_regime_hmm_contextual(self, sub_pol):
        """
        Retorna somente o estado latente HMM da janela atual. É condicionante de
        cartografia; não soma bônus V/P e não cria NO CALL isoladamente.
        """
        if not getattr(self, "ml_hmm", None):
            return "HMM_INDISPONIVEL"
        try:
            c_map = {"P": 0, "V": 1, "B": 2}
            seq = np.asarray(
                [[c_map.get(str(c).upper(), 2)] for c in sub_pol],
                dtype=int
            )
            if len(seq) < 2:
                return "HMM_INSUFICIENTE"
            estado = int(self.ml_hmm.predict(seq)[-1])
            return f"HMM_ESTADO_{estado}"
        except Exception:
            return "HMM_INDISPONIVEL"

    def _chaves_cartografia_contextual_padrao(self, sub_num, sub_pol):
        """
        MAIN 98 — descreve o padrão RAIZ e suas condicionantes internas.

        O padrão de cores não recebe direção fixa. Cada raiz é estudada novamente
        por último número, bigrama, trigrama, regime, regime HMM, Markov, geometria,
        transição geométrica, regras e contagens ativas. As chaves são independentes para
        permitir backoff quando um contexto exato ainda possui pouco suporte.
        """
        if len(sub_num) < 2 or len(sub_pol) < 2:
            return []

        nums = [int(x) for x in sub_num]
        pol = [str(x).upper() for x in sub_pol]
        ultimo = nums[-1]
        bi = "-".join(str(x) for x in nums[-2:])
        tri = "-".join(str(x) for x in nums[-3:]) if len(nums) >= 3 else bi
        regime = self._detectar_regime_temporal(pol)
        regime_hmm = self._obter_regime_hmm_contextual(pol)
        geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)

        markov = self.calcular_probabilidade_exata_markov(pol)
        mv = float(markov.get("V", 0.0))
        mp = float(markov.get("P", 0.0))
        dif_markov = abs(mv - mp)
        if dif_markov < 0.50:
            faixa_markov = "NEUTRO"
        elif mv > mp:
            faixa_markov = "V_FORTE" if dif_markov >= 2.0 else "V_LEVE"
        else:
            faixa_markov = "P_FORTE" if dif_markov >= 2.0 else "P_LEVE"

        geo_anterior = "SEM_GEOMETRIA_ANTERIOR"
        if len(pol) >= 3:
            geo_anterior = AnalisadorContextoAvancado.mapear_padroes_geometria(pol[:-1])

        regras_ativas = []
        if len(nums) >= 12:
            try:
                regras_ativas = MotorContagensProjetivas.mapear_janela(
                    nums[-12:], pol[-12:], geometria, None
                )
            except Exception:
                regras_ativas = []

        tipos_regras = sorted({
            str(r.get("tipo_regra", "SEM_REGRA"))
            for r in regras_ativas
            if r.get("tipo_regra")
        })
        contagens = sorted({
            str(r.get("tipo_regra"))
            for r in regras_ativas
            if (
                str(r.get("familia", "")).upper() in (
                    "CONTAGENS_PROJETIVAS",
                    "DINAMICA_CONTAGENS",
                    "HIERARQUIA_CONTAGENS"
                )
                or "CONTAGEM" in str(r.get("tipo_regra", "")).upper()
                or str(r.get("tipo_regra", "")).upper().startswith("V3_ATIVADOR_")
            )
        })

        chaves = []
        max_tam = min(8, len(pol))
        for tam in range(2, max_tam + 1):
            raiz = "".join(pol[-tam:])
            prefixo = f"PADRAO_CTX|T={tam}|C={raiz}"
            chaves.extend([
                prefixo,
                f"{prefixo}|ULT={ultimo}",
                f"{prefixo}|BI={bi}",
                f"{prefixo}|TRI={tri}",
                f"{prefixo}|REG={regime}",
                f"{prefixo}|HMM={regime_hmm}",
                f"{prefixo}|MK={faixa_markov}",
                f"{prefixo}|GEO={geometria}",
                f"{prefixo}|GEO_TRANS={geo_anterior}>{geometria}",
                f"{prefixo}|ULT={ultimo}|BI={bi}",
                f"{prefixo}|TRI={tri}|REG={regime}",
                f"{prefixo}|REG={regime}|HMM={regime_hmm}",
                f"{prefixo}|HMM={regime_hmm}|MK={faixa_markov}",
                f"{prefixo}|REG={regime}|MK={faixa_markov}",
            ])
            for tipo in tipos_regras:
                chaves.append(f"{prefixo}|REGRA={tipo}")
            for tipo in contagens:
                chaves.append(f"{prefixo}|CONTAGEM={tipo}")

        return list(dict.fromkeys(chaves))

    def _registrar_cartografia_contextual_padrao(self, sub_num, sub_pol, c0, c1):
        """Registra o desfecho real G0/G1 de cada contexto interno do padrão."""
        rv = self._resultado_ate_g1(c0, c1, "V")
        rp = self._resultado_ate_g1(c0, c1, "P")
        for chave in self._chaves_cartografia_contextual_padrao(sub_num, sub_pol):
            st = self.cartografia_padroes_contextual[hash_chave(chave)]
            st["total"] += 1
            if c0 == "B":
                st["B_g0"] += 1
            if rv == "G0":
                st["V_g0"] += 1
            elif rv == "G1":
                st["V_g1"] += 1
            else:
                st["V_falha"] += 1
            if rp == "G0":
                st["P_g0"] += 1
            elif rp == "G1":
                st["P_g1"] += 1
            else:
                st["P_falha"] += 1

    def obter_voto_padrao_contextual(self, sub_num, sub_pol):
        """
        Consulta a cartografia interna do padrão atual.

        Prioriza G0 e usa G0/G1 como confirmação. Contextos específicos recebem
        mais peso, porém shrinkage por suporte impede uma amostra pequena de
        transformar um padrão em verdade absoluta.
        """
        mapa = getattr(self, "cartografia_padroes_contextual", {})
        if not mapa:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "contextos": 0, "leituras": []
            }

        leituras = []
        for chave in self._chaves_cartografia_contextual_padrao(sub_num, sub_pol):
            st = mapa.get(hash_chave(chave))
            if not st:
                continue
            suporte = int(st.get("total", 0))
            if suporte < 20:
                continue

            v_g0 = float(st.get("V_g0", 0)) / suporte
            p_g0 = float(st.get("P_g0", 0)) / suporte
            v_g01 = (float(st.get("V_g0", 0)) + float(st.get("V_g1", 0))) / suporte
            p_g01 = (float(st.get("P_g0", 0)) + float(st.get("P_g1", 0))) / suporte

            # G0 é o objetivo prioritário; G1 confirma que a direção não é frágil.
            score_v = (0.65 * v_g0) + (0.35 * v_g01)
            score_p = (0.65 * p_g0) + (0.35 * p_g01)

            if "|ULT=" in chave and "|BI=" in chave:
                especificidade = 1.00
            elif "|TRI=" in chave and "|REG=" in chave:
                especificidade = 0.98
            elif "|REG=" in chave and "|HMM=" in chave:
                especificidade = 0.96
            elif "|HMM=" in chave and "|MK=" in chave:
                especificidade = 0.95
            elif "|REG=" in chave and "|MK=" in chave:
                especificidade = 0.95
            elif "|GEO_TRANS=" in chave:
                especificidade = 0.92
            elif "|REGRA=" in chave or "|CONTAGEM=" in chave:
                especificidade = 0.90
            elif "|TRI=" in chave:
                especificidade = 0.88
            elif "|BI=" in chave:
                especificidade = 0.82
            elif "|ULT=" in chave:
                especificidade = 0.78
            elif "|REG=" in chave or "|MK=" in chave or "|GEO=" in chave:
                especificidade = 0.74
            else:
                especificidade = 0.55

            shrink = suporte / (suporte + 30.0)
            peso_leitura = especificidade * shrink
            leituras.append({
                "chave": chave,
                "suporte": suporte,
                "score_v": score_v,
                "score_p": score_p,
                "peso": peso_leitura,
                "v_g0": v_g0,
                "p_g0": p_g0,
                "v_g0_g1": v_g01,
                "p_g0_g1": p_g01
            })

        if not leituras:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "contextos": 0, "leituras": []
            }

        peso_total = sum(x["peso"] for x in leituras)
        score_v = sum(x["score_v"] * x["peso"] for x in leituras) / max(peso_total, 1e-9)
        score_p = sum(x["score_p"] * x["peso"] for x in leituras) / max(peso_total, 1e-9)
        margem = abs(score_v - score_p)

        if margem < 0.025:
            direcao = "NEUTRO"
            peso = 0.0
        else:
            direcao = "VERMELHO" if score_v > score_p else "PRETO"
            if margem >= 0.10 and len(leituras) >= 3:
                peso = 3.0
            elif margem >= 0.055 and len(leituras) >= 2:
                peso = 2.0
            else:
                peso = 1.0

        melhores = sorted(
            leituras,
            key=lambda x: abs(x["score_v"] - x["score_p"]) * x["peso"],
            reverse=True
        )[:12]
        resultado = {
            "ativo": True,
            "direcao": direcao,
            "peso": peso,
            "score_vermelho": round(score_v, 6),
            "score_preto": round(score_p, 6),
            "margem": round(margem, 6),
            "contextos": len(leituras),
            "leituras": melhores
        }
        self.ultima_leitura_padrao_contextual = resultado
        return resultado

    @staticmethod
    def _resultado_direcional_ate_g2(c0, c1, c2, letra):
        """Classifica G0/G1/G2/FALHA para uma direção V/P, aceitando Branco."""
        if c0 in (letra, "B"):
            return "G0"
        if c1 in (letra, "B"):
            return "G1"
        if c2 in (letra, "B"):
            return "G2"
        return "FALHA"

    def _contexto_cartografia_regra(self, sub_num, sub_pol, eventos_override=None):
        nums = [int(x) for x in sub_num]
        pol = [str(x).upper() for x in sub_pol]
        if not nums or not pol:
            return {
                "ultimo": -1, "bi": "SEM_BI", "tri": "SEM_TRI",
                "padrao": "SEM_PADRAO", "geometria": "ESTÁVEL",
                "regime": "NEUTRO", "markov": "NEUTRO", "eventos": [],
                "coexistentes": "SEM_COEXISTENCIA", "contagens": "SEM_CONTAGEM"
            }

        ultimo = nums[-1]
        bi = "-".join(str(x) for x in nums[-2:]) if len(nums) >= 2 else str(ultimo)
        tri = "-".join(str(x) for x in nums[-3:]) if len(nums) >= 3 else bi
        padrao = "".join(pol[-4:])
        geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
        regime = self._detectar_regime_temporal(pol)

        markov = self.calcular_probabilidade_exata_markov(pol)
        mv = float(markov.get("V", 0.0))
        mp = float(markov.get("P", 0.0))
        dif = abs(mv - mp)
        if dif < 0.50:
            faixa_markov = "NEUTRO"
        elif mv > mp:
            faixa_markov = "V_FORTE" if dif >= 2.0 else "V_LEVE"
        else:
            faixa_markov = "P_FORTE" if dif >= 2.0 else "P_LEVE"

        if eventos_override is None:
            eventos = self._eventos_regras_contagens_contextuais(nums, pol)
        else:
            eventos = list(eventos_override)

        tipos = sorted({e["tipo"] for e in eventos})
        contagens = sorted({
            e["tipo"] for e in eventos
            if (
                e["familia"] in (
                    "CONTAGENS_PROJETIVAS", "DINAMICA_CONTAGENS",
                    "HIERARQUIA_CONTAGENS", "CICLO_CONTAGEM_HISTORICO"
                )
                or "CONTAGEM" in e["tipo"]
                or e["tipo"].startswith("V3_ATIVADOR_")
            )
        })
        coexistentes = ",".join(tipos) if tipos else "SEM_COEXISTENCIA"
        estado_contagens = ",".join(contagens) if contagens else "SEM_CONTAGEM"

        regime_hmm = self._obter_regime_hmm_contextual(pol)

        return {
            "ultimo": ultimo, "bi": bi, "tri": tri, "padrao": padrao,
            "geometria": geometria, "regime": regime,
            "regime_hmm": regime_hmm,
            "markov": faixa_markov, "eventos": eventos,
            "coexistentes": coexistentes, "contagens": estado_contagens
        }

    def _chaves_cartografia_contextual_eventos(self, sub_num, sub_pol, eventos):
        """Gera as mesmas chaves contextuais para uma lista explícita de eventos."""
        if not sub_num or not sub_pol or not eventos:
            return []

        ctx = self._contexto_cartografia_regra(
            sub_num, sub_pol, eventos_override=eventos
        )
        chaves = []
        for evento in ctx["eventos"]:
            prefixo = f"REGRA_CTX|E={evento['tipo']}"
            chaves.extend([
                prefixo,
                f"{prefixo}|ULT={ctx['ultimo']}",
                f"{prefixo}|BI={ctx['bi']}",
                f"{prefixo}|TRI={ctx['tri']}",
                f"{prefixo}|PAD={ctx['padrao']}",
                f"{prefixo}|GEO={ctx['geometria']}",
                f"{prefixo}|REG={ctx['regime']}",
                f"{prefixo}|HMM={ctx['regime_hmm']}",
                f"{prefixo}|REG={ctx['regime']}|HMM={ctx['regime_hmm']}",
                f"{prefixo}|MK={ctx['markov']}",
                f"{prefixo}|ULT={ctx['ultimo']}|BI={ctx['bi']}",
                f"{prefixo}|TRI={ctx['tri']}|PAD={ctx['padrao']}",
                f"{prefixo}|PAD={ctx['padrao']}|REG={ctx['regime']}",
                f"{prefixo}|REG={ctx['regime']}|MK={ctx['markov']}",
                f"{prefixo}|TRI={ctx['tri']}|PAD={ctx['padrao']}|REG={ctx['regime']}",
                f"{prefixo}|COEX={ctx['coexistentes']}",
                f"{prefixo}|CONT={ctx['contagens']}",
            ])
        return list(dict.fromkeys(chaves))

    def _cartografia_recente_regra_atual(self, regra_id):
        """
        MAIN 119 — cartografia RECENTE separada da memória macro.
        Varre somente a recência oficial ativa e conserva o detector oficial:
        uma regra só é contabilizada quando realmente esteve ativa na janela.
        """
        dados = list(getattr(self, "dados_recencia", []) or [])[-200:]
        if len(dados) < 15 or not regra_id:
            return {}
        nums = [int(d.get("numero")) for d in dados]
        pol = [str(d.get("cor", "B")).upper() for d in dados]
        mapa = defaultdict(lambda: {
            "total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0
        })
        for i in range(0, len(nums) - 14):
            sub_num = nums[i:i + 12]
            sub_pol = pol[i:i + 12]
            eventos = self._eventos_regras_contagens_contextuais(sub_num, sub_pol)
            evento = next((e for e in eventos if str(e.get("tipo")) == str(regra_id)), None)
            if evento is None:
                continue
            c0, c1 = pol[i + 12], pol[i + 13]
            for chave in self._chaves_cartografia_contextual_eventos(
                sub_num, sub_pol, [evento]
            ):
                st = mapa[chave]
                st["total"] += 1
                rv = self._resultado_ate_g1(c0, c1, "V")
                rp = self._resultado_ate_g1(c0, c1, "P")
                if rv in ("G0", "G1"):
                    st[f"V_{rv.lower()}"] += 1
                if rp in ("G0", "G1"):
                    st[f"P_{rp.lower()}"] += 1
        return dict(mapa)
    def _chaves_cartografia_contextual_regra(self, sub_num, sub_pol):
        """
        Gera níveis de backoff por REGRA/CONTAGEM. A ativação atual continua vindo
        exclusivamente do detector oficial. A base histórica das regras 4, 10 e
        5-10 é catalogada diretamente, ocorrência por ocorrência.
        """
        if len(sub_num) < 12 or len(sub_pol) < 12:
            return []

        eventos = self._eventos_regras_contagens_contextuais(
            sub_num[-12:], sub_pol[-12:]
        )
        return self._chaves_cartografia_contextual_eventos(
            sub_num[-12:], sub_pol[-12:], eventos
        )

    def _registrar_cartografia_contextual_regra(
        self, sub_num, sub_pol, c0, c1, c2, eventos_override=None
    ):
        ctx = self._contexto_cartografia_regra(
            sub_num, sub_pol, eventos_override=eventos_override
        )
        if not ctx["eventos"]:
            return 0

        chaves_por_evento = defaultdict(list)
        for chave in self._chaves_cartografia_contextual_eventos(
            sub_num, sub_pol, ctx["eventos"]
        ):
            tipo = chave.split("|E=", 1)[1].split("|", 1)[0]
            chaves_por_evento[tipo].append(chave)

        registrados = 0
        for evento in ctx["eventos"]:
            tipo = evento["tipo"]
            direcao_oficial = "V" if evento["direcao"] == "VERMELHO" else "P"
            resultado_oficial = self._resultado_direcional_ate_g2(
                c0, c1, c2, direcao_oficial
            )
            resultado_v = self._resultado_direcional_ate_g2(c0, c1, c2, "V")
            resultado_p = self._resultado_direcional_ate_g2(c0, c1, c2, "P")

            for chave in chaves_por_evento.get(tipo, []):
                st = self.cartografia_regras_contextual[chave]
                st["total"] += 1
                st[resultado_oficial.lower()] += 1
                st[f"V_{resultado_v.lower()}"] += 1
                st[f"P_{resultado_p.lower()}"] += 1
                st[f"direcao_g0_{c0}"] += 1
            registrados += 1
        return registrados

    def _mapear_cartografia_contextual_regras_contagens(self, dados):
        """
        MAIN 101 — varredura integral do XLS, índice por índice.

        Regra 4, Regra 10 e 5-10 são encontradas diretamente em CADA aparição
        cronológica, como números, bigramas, trigramas e padrões. A janela de 12
        permanece intacta para a operação e para as demais regras/contagens.
        """
        self.cartografia_regras_contextual = defaultdict(
            lambda: {
                "total": 0, "g0": 0, "g1": 0, "g2": 0, "falha": 0,
                "V_g0": 0, "V_g1": 0, "V_g2": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_g2": 0, "P_falha": 0,
                "direcao_g0_V": 0, "direcao_g0_P": 0, "direcao_g0_B": 0
            }
        )

        if not dados or len(dados) < 4:
            self.cartografia_regras_contextual_metricas = {
                "ativo": False, "posicoes_varridas": 0,
                "ocorrencias_catalogadas": 0, "contextos": 0
            }
            return

        numeros = [int(d["numero"]) for d in dados]
        cores = [str(d["cor"]).upper() for d in dados]
        posicoes_varridas = 0
        ocorrencias = 0
        ocorrencias_regras_diretas = 0
        ocorrencias_detector_12 = 0

        familias_diretas = {
            "REGRA_OFICIAL_4", "REGRA_OFICIAL_10", "REGRA_OFICIAL_5_10"
        }

        # i é a posição cronológica atual. i+1/i+2/i+3 são G0/G1/G2 históricos.
        for i in range(0, len(dados) - 3):
            inicio = max(0, i - 11)
            sub_num = numeros[inicio:i + 1]
            sub_pol = cores[inicio:i + 1]
            posicoes_varridas += 1

            # 4, 10 e 5-10: ocorrência direta no índice atual, sem depender da
            # janela operacional de 12. As condições são cópia lógica exata do
            # detector oficial e o detector em si permanece intocado.
            eventos_diretos = self._eventos_regras_oficiais_cronologicos_no_indice(
                numeros, cores, i
            )

            # Demais regras e ciclos de contagem continuam usando a cartografia
            # já existente. Removemos daqui apenas as três famílias diretas para
            # impedir dupla contagem da mesma aparição histórica.
            eventos_detector = []
            if len(sub_num) >= 12 and len(sub_pol) >= 12:
                eventos_detector = [
                    evento
                    for evento in self._eventos_regras_contagens_contextuais(
                        sub_num[-12:], sub_pol[-12:]
                    )
                    if evento.get("familia") not in familias_diretas
                ]

            eventos = eventos_diretos + eventos_detector
            if not eventos:
                continue

            registrados = self._registrar_cartografia_contextual_regra(
                sub_num, sub_pol,
                cores[i + 1], cores[i + 2], cores[i + 3],
                eventos_override=eventos
            )
            ocorrencias += registrados
            ocorrencias_regras_diretas += len(eventos_diretos)
            ocorrencias_detector_12 += len(eventos_detector)

        bases = [
            st for chave, st in self.cartografia_regras_contextual.items()
            if chave.count("|") == 1
        ]
        total_base = sum(int(st.get("total", 0)) for st in bases)
        g0 = sum(int(st.get("g0", 0)) for st in bases)
        g1 = sum(int(st.get("g1", 0)) for st in bases)
        g2 = sum(int(st.get("g2", 0)) for st in bases)
        falha = sum(int(st.get("falha", 0)) for st in bases)

        self.cartografia_regras_contextual_metricas = {
            "ativo": True,
            "versao": 2,
            "metodo": "VARREDURA_CRONOLOGICA_DIRETA_INDICE_A_INDICE",
            "detector_oficial_alterado": False,
            "janela_12_operacional_alterada": False,
            "regras_diretas": [
                "REGRA_4", "REGRA_10", "REGRA_5_10"
            ],
            "posicoes_varridas": posicoes_varridas,
            "ocorrencias_catalogadas": ocorrencias,
            "ocorrencias_regras_diretas": ocorrencias_regras_diretas,
            "ocorrencias_detector_12": ocorrencias_detector_12,
            "contextos": len(self.cartografia_regras_contextual),
            "total_ocorrencias_base": total_base,
            "G0": g0, "G1": g1, "G2": g2, "FALHA": falha,
            "resolucao_ate_g1_percent": round(
                ((g0 + g1) / max(1, total_base)) * 100.0, 2
            )
        }

    def obter_voto_regra_contextual(self, sub_num, sub_pol):
        """
        Consulta a cartografia das regras/contagens ativas no contexto atual.
        G0 é prioritário e G1 confirma a direção; G2 fica registrado para auditoria.
        """
        mapa = getattr(self, "cartografia_regras_contextual", {})
        if not mapa or len(sub_num) < 12 or len(sub_pol) < 12:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "contextos": 0, "leituras": []
            }

        leituras = []
        for chave in self._chaves_cartografia_contextual_regra(
            sub_num[-12:], sub_pol[-12:]
        ):
            st = mapa.get(chave)
            if not st:
                continue
            suporte = int(st.get("total", 0))
            if suporte < 20:
                continue

            v_g0 = float(st.get("V_g0", 0)) / suporte
            p_g0 = float(st.get("P_g0", 0)) / suporte
            v_g01 = (
                float(st.get("V_g0", 0)) + float(st.get("V_g1", 0))
            ) / suporte
            p_g01 = (
                float(st.get("P_g0", 0)) + float(st.get("P_g1", 0))
            ) / suporte

            score_v = (0.70 * v_g0) + (0.30 * v_g01)
            score_p = (0.70 * p_g0) + (0.30 * p_g01)

            if "|TRI=" in chave and "|PAD=" in chave and "|REG=" in chave:
                especificidade = 1.00
            elif "|ULT=" in chave and "|BI=" in chave:
                especificidade = 0.98
            elif "|REG=" in chave and "|MK=" in chave:
                especificidade = 0.95
            elif "|COEX=" in chave or "|CONT=" in chave:
                especificidade = 0.92
            elif "|TRI=" in chave or "|PAD=" in chave:
                especificidade = 0.88
            elif "|BI=" in chave:
                especificidade = 0.82
            elif "|ULT=" in chave:
                especificidade = 0.78
            elif "|REG=" in chave or "|MK=" in chave or "|GEO=" in chave:
                especificidade = 0.74
            else:
                especificidade = 0.58

            shrink = suporte / (suporte + 30.0)
            peso = especificidade * shrink
            leituras.append({
                "chave": chave, "suporte": suporte,
                "score_v": score_v, "score_p": score_p, "peso": peso,
                "g0": int(st.get("g0", 0)), "g1": int(st.get("g1", 0)),
                "g2": int(st.get("g2", 0)), "falha": int(st.get("falha", 0)),
                "direcao_g0": {
                    "V": int(st.get("direcao_g0_V", 0)),
                    "P": int(st.get("direcao_g0_P", 0)),
                    "B": int(st.get("direcao_g0_B", 0))
                }
            })

        if not leituras:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0,
                "margem": 0.0, "contextos": 0, "leituras": []
            }

        peso_total = sum(x["peso"] for x in leituras)
        score_v = sum(x["score_v"] * x["peso"] for x in leituras) / max(peso_total, 1e-9)
        score_p = sum(x["score_p"] * x["peso"] for x in leituras) / max(peso_total, 1e-9)
        margem = abs(score_v - score_p)

        if margem < 0.025:
            direcao = "NEUTRO"
        else:
            direcao = "VERMELHO" if score_v > score_p else "PRETO"

        suporte_medio = sum(x["suporte"] for x in leituras) / len(leituras)
        peso_voto = min(
            3.0,
            margem * 18.0 * min(1.0, suporte_medio / 80.0)
        )
        if direcao == "NEUTRO":
            peso_voto = 0.0

        resultado = {
            "ativo": direcao in ("VERMELHO", "PRETO"),
            "direcao": direcao,
            "peso": round(peso_voto, 4),
            "margem": round(margem, 6),
            "contextos": len(leituras),
            "score_v": round(score_v, 6),
            "score_p": round(score_p, 6),
            "leituras": sorted(
                leituras, key=lambda x: (x["peso"], x["suporte"]), reverse=True
            )[:12]
        }
        self.ultima_leitura_regra_contextual = resultado
        return resultado
    def _chaves_trajetoria_projecao(self, numero, traj_num, traj_pol):
        """Destrincha cada contagem vermelha 1..7 por posição, números e fechamento."""
        if not traj_num or not traj_pol:
            return []
        n = int(numero)
        nums = [int(x) for x in traj_num]
        cores = [str(x).upper() for x in traj_pol]
        nums_txt = "-".join(str(x) for x in nums)
        cores_txt = "".join(cores)
        ultimo = nums[-1]
        bi = "-".join(str(x) for x in nums[-2:]) if len(nums) >= 2 else str(ultimo)
        tri = "-".join(str(x) for x in nums[-3:]) if len(nums) >= 3 else bi

        chaves = [
            f"PROJ|N={n}",
            f"PROJ_FECHO|N={n}|ULT={ultimo}",
            f"PROJ_BI|N={n}|BI={bi}",
            f"PROJ_TRI|N={n}|TRI={tri}",
            f"PROJ_CORES|N={n}|C={cores_txt}",
            f"PROJ_TRAJ|N={n}|NUM={nums_txt}|C={cores_txt}",
        ]
        # Posição relativa de CADA número intermediário. Ex.: no 7,
        # descobrir que o 5 na última casa antes do alvo muda o respeito.
        for pos, valor in enumerate(nums[1:], start=1):
            distancia_alvo = len(nums) - 1 - pos
            chaves.append(
                f"PROJ_POS|N={n}|POS={pos}|DIST_ALVO={distancia_alvo}|VAL={valor}"
            )
        return list(dict.fromkeys(chaves))

    def _mapear_cartografia_completa_xls(self, dados):
        """
        MAIN 85 — autópsia cronológica completa do XLS.

        1) Mapeia todo número final e o resultado imediato até G1.
        2) Mapeia P-P, P-P-P..., V-V, V-V-V... com bigrama, trigrama,
           último número e sequência numérica exata.
        3) Mapeia todo xadrez e mede continuação/quebra até G1 nos mesmos contextos.
        4) Para CADA projeção vermelha 1..7 já existente, destrincha a trajetória
           completa por posição relativa, números intermediários, fechamento,
           bigrama, trigrama e cores. Não cria ativador novo.
        """
        self.cartografia_projecoes_trajetoria = defaultdict(
            lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0}
        )
        self.cartografia_padroes_xls = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0, "B_g0": 0}
        )
        self.cartografia_padroes_contextual = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        self.cartografia_trajetoria_streak = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        if not dados or len(dados) < 3:
            self.cartografia_xls_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return

        total = len(dados)
        eventos_padrao = 0
        eventos_projecao = 0

        for i in range(total - 2):
            inicio = max(0, i - 11)
            sub = dados[inicio:i + 1]
            sub_num = [int(d["numero"]) for d in sub]
            sub_pol = [str(d["cor"]).upper() for d in sub]
            c0 = str(dados[i + 1]["cor"]).upper()
            c1 = str(dados[i + 2]["cor"]).upper()

            for chave in self._chaves_cartografia_padrao(sub_num, sub_pol):
                st = self.cartografia_padroes_xls[chave]
                st["total"] += 1
                if c0 == "B":
                    st["B_g0"] += 1
                rv = self._resultado_ate_g1(c0, c1, "V")
                rp = self._resultado_ate_g1(c0, c1, "P")
                if rv == "G0": st["V_g0"] += 1
                elif rv == "G1": st["V_g1"] += 1
                if rp == "G0": st["P_g0"] += 1
                elif rp == "G1": st["P_g1"] += 1
            # MAIN 98 — o mesmo evento também alimenta a cartografia interna
            # do padrão. A cartografia antiga acima permanece sem alteração.
            self._registrar_cartografia_contextual_padrao(
                sub_num, sub_pol, c0, c1
            )
            self._registrar_trajetoria_streak(
                sub_num, sub_pol, c0, c1
            )
            self._registrar_morfologia_estrutural(
                sub_num, sub_pol, c0, c1
            )
            eventos_padrao += 1

            num_gatilho = int(dados[i]["numero"])
            if 1 <= num_gatilho <= 7:
                alvo_idx = i + num_gatilho
                if alvo_idx + 1 < total:
                    caminho_tem_branco = any(
                        str(dados[k]["cor"]).upper() == "B"
                        for k in range(i + 1, alvo_idx)
                    )
                    if not caminho_tem_branco:
                        traj = dados[i:alvo_idx + 1]
                        traj_num = [int(d["numero"]) for d in traj]
                        traj_pol = [str(d["cor"]).upper() for d in traj]
                        cor_alvo = str(dados[alvo_idx]["cor"]).upper()
                        cor_g1 = str(dados[alvo_idx + 1]["cor"]).upper()
                        resultado = self._resultado_ate_g1(cor_alvo, cor_g1, "V")
                        for chave in self._chaves_trajetoria_projecao(
                            num_gatilho, traj_num, traj_pol
                        ):
                            st = self.cartografia_projecoes_trajetoria[chave]
                            st["total"] += 1
                            if resultado == "G0":
                                st["respeitada_g0"] += 1
                            elif resultado == "G1":
                                st["respeitada_g1"] += 1
                            else:
                                st["nao_respeitada"] += 1
                        eventos_projecao += 1

        proj_suporte20 = sum(
            1 for st in self.cartografia_projecoes_trajetoria.values()
            if st["total"] >= 20
        )
        padroes_suporte20 = sum(
            1 for st in self.cartografia_padroes_xls.values()
            if st["total"] >= 20
        )
        contextual_suporte20 = sum(
            1 for st in self.cartografia_padroes_contextual.values()
            if st["total"] >= 20
        )
        self.cartografia_padroes_contextual_metricas = {
            "ativo": True,
            "versao": 1,
            "metodo": "PADRAO_RAIZ_COM_CONDICIONANTES_INTERNAS_G0_G1",
            "contextos_aprendidos": len(self.cartografia_padroes_contextual),
            "contextos_suporte_minimo_20": contextual_suporte20,
            "dimensoes": [
                "PADRAO_RAIZ", "ULTIMO_NUMERO", "BIGRAMA", "TRIGRAMA",
                "REGIME", "REGIME_HMM", "MARKOV", "GEOMETRIA",
                "TRANSICAO_GEOMETRIA", "REGRAS_ATIVAS", "CONTAGENS_ATIVAS"
            ],
            "prioridade": "G0_COM_CONFIRMACAO_G0_G1",
            "altera_regras": False,
            "altera_recencia": False
        }
        streak_traj_suporte20 = sum(
            1 for st in self.cartografia_trajetoria_streak.values()
            if st["total"] >= 20
        )
        morfologia_suporte20 = sum(
            1 for st in self.cartografia_morfologia_estrutural.values()
            if st["total"] >= 20
        )
        self.cartografia_morfologia_estrutural_metricas = {
            "ativo": True,
            "versao": 1,
            "metodo": "MORFOLOGIA_BLOCOS_NORMALIZADA_REPETICAO_INVERSAO_ESPELHO_ATE_G1",
            "contextos_aprendidos": len(self.cartografia_morfologia_estrutural),
            "contextos_suporte_minimo_20": morfologia_suporte20,
            "dimensoes": ["BLOCOS", "MORFOLOGIA", "TRAJETORIA", "REPETICAO", "INVERSAO_CROMATICA", "ESPELHO"],
            "nomes_didaticos_convertidos_em_regras": False,
            "altera_direcao": False,
            "processamento_incremental": False,
        }
        self.cartografia_trajetoria_streak_metricas = {
            "ativo": True,
            "versao": 1,
            "metodo": "TRAJETORIA_CAUSAL_STREAK_BILATERAL_V_P_ATE_G1",
            "contextos_aprendidos": len(self.cartografia_trajetoria_streak),
            "contextos_suporte_minimo_20": streak_traj_suporte20,
            "cores_estudadas": ["VERMELHO", "PRETO"],
            "estagios": ["NASCIMENTO", "CONFIRMACAO", "STREAK", "EXPANSAO", "RETOMADA"],
            "cruza_respiro_e_contagens": True,
            "processamento_incremental": False,
        }
        self.cartografia_xls_metricas = {
            "ativo": True,
            "versao": 1,
            "metodo": "CARTOGRAFIA_CRONOLOGICA_COMPLETA_CASO_A_CASO_ATE_G1",
            "eventos_padrao_analisados": eventos_padrao,
            "eventos_projecoes_1_a_7_analisados": eventos_projecao,
            "contextos_padroes_aprendidos": len(self.cartografia_padroes_xls),
            "contextos_padroes_suporte_minimo_20": padroes_suporte20,
            "contextos_trajetorias_projecao_aprendidos": len(self.cartografia_projecoes_trajetoria),
            "contextos_trajetorias_suporte_minimo_20": proj_suporte20,
            "mapeia_numeros_individuais": True,
            "mapeia_streak_pp_vv_e_evolucoes": True,
            "mapeia_xadrez_continuacao_quebra": True,
            "mapeia_bigrama_trigrama_ultimo_numero": True,
            "cartografia_contextual_interna_padroes_ativa": True,
            "padroes_condicionados_por_numero_bigrama_trigrama_regime_markov_regras_contagens": True,
            "simulacao_causal_proximos_numeros_ativa": True,
            "simulacao_recalcula_bigrama_trigrama_numero_markov_regras": True,
            "mapeia_posicao_relativa_na_trajetoria": True,
            "projecoes_vermelhas_existentes": [1, 2, 3, 4, 5, 6, 7],
            "cria_novo_ativador": False,
            "direcao_original_projecoes": "VERMELHO",
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        }

    def _obter_cartografia_projecao_atual(self, numero, sub_num, sub_pol):
        """Consulta a trajetória ativa da regra 1..7 com backoff hierárquico."""
        n = int(numero)
        gatilho_idx = None
        for i in range(len(sub_num) - 1, -1, -1):
            if int(sub_num[i]) == n and i + n == len(sub_num) - 1:
                gatilho_idx = i
                break
        if gatilho_idx is None:
            return {"ativo": False, "taxa_respeito": 0.0, "suporte": 0}

        traj_num = sub_num[gatilho_idx:]
        traj_pol = sub_pol[gatilho_idx:]
        leituras = []
        for chave in self._chaves_trajetoria_projecao(n, traj_num, traj_pol):
            st = self.cartografia_projecoes_trajetoria.get(chave)
            if not st:
                continue
            suporte = int(st.get("total", 0))
            if suporte < 20:
                continue
            respeitou = int(st.get("respeitada_g0", 0)) + int(st.get("respeitada_g1", 0))
            taxa = respeitou / max(suporte, 1)
            # Contextos mais específicos entram primeiro e ganham força pelo suporte.
            especificidade = 1.0
            if chave.startswith("PROJ_TRAJ|"): especificidade = 1.00
            elif chave.startswith("PROJ_TRI|"): especificidade = 0.90
            elif chave.startswith("PROJ_BI|"): especificidade = 0.82
            elif chave.startswith("PROJ_FECHO|"): especificidade = 0.78
            elif chave.startswith("PROJ_POS|"): especificidade = 0.75
            elif chave.startswith("PROJ_CORES|"): especificidade = 0.72
            else: especificidade = 0.50
            peso = suporte * (suporte / (suporte + 20.0)) * especificidade
            leituras.append((taxa, suporte, peso, chave))

        if not leituras:
            return {"ativo": False, "taxa_respeito": 0.0, "suporte": 0}
        peso_total = sum(x[2] for x in leituras)
        taxa = sum(x[0] * x[2] for x in leituras) / max(peso_total, 1e-9)
        return {
            "ativo": True,
            "taxa_respeito": taxa,
            "suporte": sum(x[1] for x in leituras),
            "contextos": len(leituras),
            "fontes": [x[3].split("|", 1)[0] for x in leituras]
        }

    def obter_voto_cartografia_xls(self, sub_num, sub_pol):
        """Voto aditivo da cartografia. Não cria NO CALL e não altera regras."""
        if not getattr(self, "cartografia_xls_metricas", {}).get("ativo"):
            return {"direcao": "NEUTRO", "peso": 0.0, "contextos": 0}
        leituras = []
        for chave in self._chaves_cartografia_padrao(sub_num, sub_pol):
            st = self.cartografia_padroes_xls.get(chave)
            if not st:
                continue
            total = int(st.get("total", 0))
            if total < 30:
                continue
            taxa_v = (int(st.get("V_g0", 0)) + int(st.get("V_g1", 0))) / max(total, 1)
            taxa_p = (int(st.get("P_g0", 0)) + int(st.get("P_g1", 0))) / max(total, 1)
            margem = abs(taxa_v - taxa_p)
            if margem < 0.08:
                continue
            especificidade = 1.0 if ("EXATO|" in chave or "_TRI|" in chave) else 0.75
            peso = total * (total / (total + 30.0)) * especificidade
            leituras.append((taxa_v, taxa_p, peso, chave))

        if not leituras:
            return {"direcao": "NEUTRO", "peso": 0.0, "contextos": 0}
        soma = sum(x[2] for x in leituras)
        tv = sum(x[0] * x[2] for x in leituras) / max(soma, 1e-9)
        tp = sum(x[1] * x[2] for x in leituras) / max(soma, 1e-9)
        margem = abs(tv - tp)
        if margem < 0.08:
            return {"direcao": "NEUTRO", "peso": 0.0, "contextos": len(leituras)}
        direcao = "VERMELHO" if tv > tp else "PRETO"
        peso_final = min(18.0, 4.0 + (margem * 40.0))
        return {
            "direcao": direcao,
            "peso": peso_final,
            "contextos": len(leituras),
            "taxa_v": round(tv * 100, 2),
            "taxa_p": round(tp * 100, 2)
        }
