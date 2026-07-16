from collections import defaultdict
from utils.helpers import hash_chave

class TrajetoriasMixin:
    """
    Mixin isolando a morfologia estrutural e a evolução da 
    família STREAK.
    """
    
    @staticmethod
    def _descrever_dna_deslocamento(numeros):
        """
        MAIN 128 — assinatura causal compacta do movimento numérico.
        Não cria voto próprio: serve somente como condicionante contextual das
        famílias já existentes (principalmente STREAK e deriva temporal).
        """
        nums = [int(x) for x in (numeros or [])]
        if len(nums) < 2:
            return {"ativo": False}

        deltas = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
        ultimos = deltas[-3:]
        sinais = [
            "S" if d > 0 else ("D" if d < 0 else "I")
            for d in ultimos
        ]
        magnitudes = [
            "CURTO" if abs(d) <= 2 else ("MEDIO" if abs(d) <= 5 else "LONGO")
            for d in ultimos
        ]

        if len(ultimos) >= 2:
            a, b = ultimos[-2], ultimos[-1]
            if a == 0 and b == 0:
                trajetoria = "ESTAGNACAO"
            elif a * b < 0:
                trajetoria = "REVERSAO_EXTREMA" if abs(a) >= 6 and abs(b) >= 6 else "REVERSAO"
            elif a > 0 and b > 0:
                trajetoria = "SUBIDA_ACELERADA" if abs(b) > abs(a) else (
                    "SUBIDA_CONSTANTE" if abs(b) == abs(a) else "SUBIDA_DESACELERADA"
                )
            elif a < 0 and b < 0:
                trajetoria = "DESCIDA_ACELERADA" if abs(b) > abs(a) else (
                    "DESCIDA_CONSTANTE" if abs(b) == abs(a) else "DESCIDA_DESACELERADA"
                )
            elif b == 0:
                trajetoria = "ESTAGNACAO_APOS_MOVIMENTO"
            else:
                trajetoria = "NASCIMENTO_MOVIMENTO"
        else:
            trajetoria = "MOVIMENTO_UNICO"

        return {
            "ativo": True,
            "delta_final": int(deltas[-1]),
            "direcao_final": sinais[-1],
            "magnitude_final": magnitudes[-1],
            "assinatura_direcoes": "".join(sinais),
            "assinatura_magnitudes": "-".join(magnitudes),
            "trajetoria": trajetoria,
        }

    @staticmethod
    def _descrever_gramatica_blocos(cores):
        """
        MAIN 128 — comprime a sequência V/P em blocos causais.
        Branco encerra a gramática local. A leitura é bilateral e sem direção fixa.
        """
        pol = [str(x).upper() for x in (cores or [])]
        if not pol or pol[-1] not in ("V", "P"):
            return {"ativo": False}

        trecho = []
        for cor in reversed(pol):
            if cor not in ("V", "P"):
                break
            trecho.append(cor)
        trecho.reverse()
        if not trecho:
            return {"ativo": False}

        blocos = []
        for cor in trecho:
            if blocos and blocos[-1][0] == cor:
                blocos[-1][1] += 1
            else:
                blocos.append([cor, 1])

        recentes = blocos[-5:]
        assinatura = "-".join(f"{cor}{min(tam, 9)}" for cor, tam in recentes)
        tamanhos = [tam for _, tam in recentes]
        alternancia_blocos = False
        if len(recentes) >= 4:
            cores_blocos = [cor for cor, _ in recentes]
            alternancia_blocos = all(
                cores_blocos[i] != cores_blocos[i - 1]
                for i in range(1, len(cores_blocos))
            ) and (max(tamanhos) - min(tamanhos) <= 1)

        return {
            "ativo": True,
            "assinatura": assinatura,
            "quantidade_blocos": len(recentes),
            "alternancia_blocos": alternancia_blocos,
            "tamanho_bloco_atual": int(recentes[-1][1]),
            "tamanho_bloco_anterior": int(recentes[-2][1]) if len(recentes) >= 2 else 0,
        }

    @staticmethod
    def _descrever_morfologia_estrutural(cores):
        """
        MAIN 129 — leitura genérica da construção da sequência.

        Converte cores em blocos, normaliza a primeira cor como A e a oposta
        como B e descreve a trajetória dos tamanhos. Os rótulos didáticos das
        imagens não são regras: 3-2-1, 2-1-2, 4-3-2-1 etc. são apenas formas
        aprendidas estatisticamente.
        """
        pol = [str(x).upper() for x in (cores or [])]
        if not pol or pol[-1] not in ("V", "P"):
            return {"ativo": False}

        trecho = []
        for cor in reversed(pol):
            if cor not in ("V", "P"):
                break
            trecho.append(cor)
        trecho.reverse()
        if len(trecho) < 2:
            return {"ativo": False}

        blocos = []
        for cor in trecho:
            if blocos and blocos[-1]["cor"] == cor:
                blocos[-1]["tamanho"] += 1
            else:
                blocos.append({"cor": cor, "tamanho": 1})

        blocos = blocos[-6:]
        if len(blocos) < 2:
            return {"ativo": False}

        cor_a = blocos[0]["cor"]
        tamanhos = [int(b["tamanho"]) for b in blocos]
        orientacao = ["A" if b["cor"] == cor_a else "B" for b in blocos]
        morfologia = "-".join(str(min(t, 9)) for t in tamanhos)
        forma_normalizada = "-".join(
            f"{orientacao[i]}{min(tamanhos[i], 9)}"
            for i in range(len(tamanhos))
        )
        deltas = [tamanhos[i] - tamanhos[i - 1] for i in range(1, len(tamanhos))]
        delta_assinatura = ",".join(
            "+" if d > 0 else ("-" if d < 0 else "0")
            for d in deltas
        )

        if deltas and all(d == 1 for d in deltas):
            trajetoria = "EXPANSAO_LINEAR"
        elif deltas and all(d == -1 for d in deltas):
            trajetoria = "CONTRACAO_LINEAR"
        elif deltas and all(d >= 0 for d in deltas) and any(d > 0 for d in deltas):
            trajetoria = "EXPANSAO_IRREGULAR"
        elif deltas and all(d <= 0 for d in deltas) and any(d < 0 for d in deltas):
            trajetoria = "CONTRACAO_IRREGULAR"
        elif len(deltas) >= 2 and all(
            deltas[i] * deltas[i - 1] < 0
            for i in range(1, len(deltas))
            if deltas[i] != 0 and deltas[i - 1] != 0
        ):
            trajetoria = "OSCILACAO"
        elif deltas and all(d == 0 for d in deltas):
            trajetoria = "ESTABILIZACAO"
        else:
            trajetoria = "MISTA"

        # Repetição/inversão estrutural: compara metades de mesmo comprimento.
        repeticao = inversao = False
        similaridade_repeticao = similaridade_inversao = 0.0
        n = len(blocos)
        for tam in range(min(3, n // 2), 1, -1):
            esq = blocos[-2 * tam:-tam]
            dir_ = blocos[-tam:]
            te = [x["tamanho"] for x in esq]
            td = [x["tamanho"] for x in dir_]
            if te != td:
                continue
            oe = [x["cor"] for x in esq]
            od = [x["cor"] for x in dir_]
            similaridade_repeticao = 1.0 if oe == od else 0.0
            similaridade_inversao = 1.0 if all(a != b for a, b in zip(oe, od)) else 0.0
            repeticao = similaridade_repeticao == 1.0
            inversao = similaridade_inversao == 1.0
            break

        # Espelho morfológico em torno dos possíveis pivôs de bloco.
        espelho = False
        espelho_tamanho = 0
        for piv in range(1, len(tamanhos) - 1):
            alcance = min(piv, len(tamanhos) - piv - 1)
            for k in range(alcance, 0, -1):
                if tamanhos[piv-k:piv] == list(reversed(tamanhos[piv+1:piv+1+k])):
                    espelho = True
                    espelho_tamanho = max(espelho_tamanho, k)
                    break

        # Coerência: quanto a trajetória consegue ser descrita por um movimento
        # consistente. Não gera direção; apenas mede qualidade estrutural.
        if trajetoria in ("EXPANSAO_LINEAR", "CONTRACAO_LINEAR", "ESTABILIZACAO"):
            coerencia = 1.0
        elif trajetoria in ("EXPANSAO_IRREGULAR", "CONTRACAO_IRREGULAR", "OSCILACAO"):
            coerencia = 0.78
        else:
            mudancas = sum(
                1 for i in range(1, len(deltas))
                if deltas[i] != 0 and deltas[i - 1] != 0 and deltas[i] * deltas[i - 1] < 0
            )
            coerencia = max(0.25, 0.60 - 0.10 * mudancas)

        return {
            "ativo": True,
            "blocos": blocos,
            "tamanhos": tamanhos,
            "morfologia": morfologia,
            "forma_normalizada": forma_normalizada,
            "orientacao": "-".join(orientacao),
            "trajetoria": trajetoria,
            "delta_assinatura": delta_assinatura,
            "repeticao": repeticao,
            "inversao_cromatica": inversao,
            "espelho": espelho,
            "espelho_tamanho": espelho_tamanho,
            "coerencia": round(float(coerencia), 4),
            "cor_atual": blocos[-1]["cor"],
            "tamanho_atual": int(blocos[-1]["tamanho"]),
            "cor_inicial": cor_a,
        }

    def _chaves_morfologia_estrutural(self, sub_num, sub_pol):
        d = self._descrever_morfologia_estrutural(sub_pol)
        if not d.get("ativo"):
            return []

        nums = [int(x) for x in (sub_num or [])]
        base = f"MORFO|M={d['morfologia']}|TRJ={d['trajetoria']}"
        chaves = [
            f"MORFO_FAMILIA|TRJ={d['trajetoria']}|DELTA={d['delta_assinatura']}",
            base,
            f"{base}|FORMA={d['forma_normalizada']}",
            f"{base}|ATUAL={d['cor_atual']}{d['tamanho_atual']}",
        ]
        if nums:
            chaves.append(f"{base}|ULT={nums[-1]}")
        if len(nums) >= 2:
            chaves.append(f"{base}|BI={nums[-2]}-{nums[-1]}")
        if d.get("repeticao"):
            chaves.append(f"{base}|REL=REPETICAO")
        if d.get("inversao_cromatica"):
            chaves.append(f"{base}|REL=INVERSAO_CROMATICA")
        if d.get("espelho"):
            chaves.append(f"{base}|REL=ESPELHO|K={d['espelho_tamanho']}")
        return list(dict.fromkeys(chaves))

    def _registrar_morfologia_estrutural(self, sub_num, sub_pol, c0, c1):
        rv = self._resultado_ate_g1(c0, c1, "V")
        rp = self._resultado_ate_g1(c0, c1, "P")
        for chave in self._chaves_morfologia_estrutural(sub_num, sub_pol):
            st = self.cartografia_morfologia_estrutural[hash_chave(chave)]
            st["total"] += 1
            if str(c0).upper() == "B":
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

    def obter_voto_morfologia_estrutural(self, sub_num, sub_pol):
        """
        Consolida a morfologia como contexto aprendido. Não cria CALL e não
        inverte direção. Pode apenas sustentar ou vetar uma autoridade já eleita
        quando a oposição estrutural é robusta.
        """
        descricao = self._descrever_morfologia_estrutural(sub_pol)
        mapa = getattr(self, "cartografia_morfologia_estrutural", {})
        if not descricao.get("ativo") or not mapa:
            return {"ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0}

        leituras = []
        for chave in self._chaves_morfologia_estrutural(sub_num, sub_pol):
            st = mapa.get(hash_chave(chave))
            if not st:
                continue
            total = int(st.get("total", 0))
            if total < 20:
                continue
            taxa_v = (int(st.get("V_g0", 0)) + int(st.get("V_g1", 0))) / max(total, 1)
            taxa_p = (int(st.get("P_g0", 0)) + int(st.get("P_g1", 0))) / max(total, 1)
            especificidade = 0.58
            if "|REL=ESPELHO" in chave or "|REL=INVERSAO_CROMATICA" in chave:
                especificidade = 0.96
            elif "|REL=REPETICAO" in chave:
                especificidade = 0.94
            elif "|BI=" in chave:
                especificidade = 0.90
            elif "|ULT=" in chave:
                especificidade = 0.84
            elif "|FORMA=" in chave:
                especificidade = 0.82
            elif "|ATUAL=" in chave:
                especificidade = 0.76
            elif chave.startswith("MORFO|"):
                especificidade = 0.70
            peso = especificidade * (total / (total + 35.0))
            leituras.append({
                "fonte": chave.split("|", 1)[0],
                "taxa_v": taxa_v, "taxa_p": taxa_p,
                "total": total, "peso": peso,
            })

        if not leituras:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0,
                "morfologia": descricao,
            }

        soma = sum(x["peso"] for x in leituras)
        taxa_v_macro = sum(x["taxa_v"] * x["peso"] for x in leituras) / max(soma, 1e-9)
        taxa_p_macro = sum(x["taxa_p"] * x["peso"] for x in leituras) / max(soma, 1e-9)
        suporte = max(x["total"] for x in leituras)

        # RECÊNCIA oficial: mesma assinatura estrutural, sem misturar com memória
        # imediata de feedback. O peso continua derivado das 200 rodadas injetadas.
        rec_total = rec_v = rec_p = 0
        dados_rec = list(getattr(self, "dados_recencia", []) or [])[-200:]
        chaves_atuais = set(hash_chave(c) for c in self._chaves_morfologia_estrutural(sub_num, sub_pol))
        if len(dados_rec) >= 3 and chaves_atuais:
            for i in range(len(dados_rec) - 2):
                inicio = max(0, i - 11)
                trecho = dados_rec[inicio:i + 1]
                n_rec = [int(x["numero"]) for x in trecho]
                c_rec = [str(x["cor"]).upper() for x in trecho]
                chaves_rec = set(hash_chave(c) for c in self._chaves_morfologia_estrutural(n_rec, c_rec))
                if not chaves_atuais.intersection(chaves_rec):
                    continue
                g0 = str(dados_rec[i + 1]["cor"]).upper()
                g1 = str(dados_rec[i + 2]["cor"]).upper()
                rec_total += 1
                if g0 in ("V", "B") or g1 in ("V", "B"):
                    rec_v += 1
                if g0 in ("P", "B") or g1 in ("P", "B"):
                    rec_p += 1

        taxa_v = taxa_v_macro
        taxa_p = taxa_p_macro
        if rec_total >= 8:
            fator_rec = min(0.60, 0.20 + rec_total / 100.0)
            taxa_v = (1.0 - fator_rec) * taxa_v_macro + fator_rec * (rec_v / rec_total)
            taxa_p = (1.0 - fator_rec) * taxa_p_macro + fator_rec * (rec_p / rec_total)

        margem = abs(taxa_v - taxa_p)
        direcao = "VERMELHO" if taxa_v > taxa_p else ("PRETO" if taxa_p > taxa_v else "NEUTRO")
        peso = min(1.0, margem * 3.0) * min(1.0, suporte / 80.0) * float(descricao.get("coerencia", 0.5))

        resultado = {
            "ativo": direcao in ("VERMELHO", "PRETO"),
            "direcao": direcao,
            "peso": round(float(peso), 4),
            "suporte": int(suporte),
            "margem": round(float(margem), 4),
            "taxa_v": round(float(taxa_v) * 100.0, 2),
            "taxa_p": round(float(taxa_p) * 100.0, 2),
            "recencia_suporte": int(rec_total),
            "morfologia": descricao,
            "fontes": [x["fonte"] for x in leituras],
            "altera_direcao": False,
        }
        self._ultimo_voto_morfologia_estrutural = resultado
        return resultado

    def _descrever_trajetoria_streak(self, sub_num, sub_pol):
        """
        MAIN 127 — reconstrói a trajetória terminal de uma sequência da MESMA cor.

        A leitura é bilateral: VERMELHO e PRETO usam exatamente o mesmo algoritmo.
        O estado diferencia nascimento (1), confirmação (2), streak (3) e expansão
        (4+), além de retomada após respiro quando a mesma cor já dominava antes.
        """
        nums = [int(x) for x in (sub_num or [])]
        pol = [str(x).upper() for x in (sub_pol or [])]
        if not nums or len(nums) != len(pol) or pol[-1] not in ("V", "P"):
            return {"ativo": False}

        cor = pol[-1]
        inicio = len(pol) - 1
        while inicio > 0 and pol[inicio - 1] == cor:
            inicio -= 1
        tamanho = len(pol) - inicio

        if tamanho == 1:
            estagio = "NASCIMENTO"
        elif tamanho == 2:
            estagio = "CONFIRMACAO"
        elif tamanho == 3:
            estagio = "STREAK"
        else:
            estagio = "EXPANSAO"

        # Segmenta o passado imediato em blocos V/P; branco encerra a ligação causal.
        runs = []
        j = inicio - 1
        while j >= 0 and len(runs) < 3:
            if pol[j] == "B":
                break
            cor_run = pol[j]
            fim_run = j
            while j >= 0 and pol[j] == cor_run:
                j -= 1
            runs.append({
                "cor": cor_run,
                "tamanho": fim_run - j,
                "inicio": j + 1,
                "fim": fim_run,
            })

        respiro = runs[0] if runs and runs[0]["cor"] != cor else None
        streak_anterior = (
            runs[1]
            if respiro is not None and len(runs) >= 2 and runs[1]["cor"] == cor
            else None
        )
        retomada = streak_anterior is not None
        tipo_trajetoria = "RETOMADA" if retomada else estagio

        ultimo = nums[-1]
        bi = "-".join(str(x) for x in nums[-2:]) if len(nums) >= 2 else str(ultimo)
        tri = "-".join(str(x) for x in nums[-3:]) if len(nums) >= 3 else bi
        numero_nascimento = nums[inicio]
        nums_respiro = []
        if respiro is not None:
            nums_respiro = nums[respiro["inicio"]:respiro["fim"] + 1]

        def _contagens_no_contexto(numeros_ctx, cores_ctx):
            if len(numeros_ctx) < 12:
                return []
            try:
                geo = AnalisadorContextoAvancado.mapear_padroes_geometria(cores_ctx[-12:])
                regras = MotorContagensProjetivas.mapear_janela(
                    numeros_ctx[-12:], cores_ctx[-12:], geo, None
                )
            except Exception:
                return []
            return sorted({
                str(r.get("tipo_regra", ""))
                for r in regras
                if (
                    str(r.get("familia", "")).upper() in (
                        "CONTAGENS_PROJETIVAS", "DINAMICA_CONTAGENS", "HIERARQUIA_CONTAGENS"
                    )
                    or "CONTAGEM" in str(r.get("tipo_regra", "")).upper()
                    or str(r.get("tipo_regra", "")).upper().startswith("V3_ATIVADOR_")
                )
                and r.get("tipo_regra")
            })

        contagens_atuais = _contagens_no_contexto(nums, pol)
        contagens_nascimento = _contagens_no_contexto(nums[:inicio + 1], pol[:inicio + 1])
        dna_deslocamento = self._descrever_dna_deslocamento(nums[max(inicio, len(nums) - 4):])
        gramatica_blocos = self._descrever_gramatica_blocos(pol)

        return {
            "ativo": True,
            "cor": cor,
            "tamanho": tamanho,
            "estagio": estagio,
            "tipo_trajetoria": tipo_trajetoria,
            "retomada": retomada,
            "streak_anterior": int(streak_anterior["tamanho"]) if streak_anterior else 0,
            "cor_respiro": str(respiro["cor"]) if respiro else "SEM_RESPIRO",
            "tamanho_respiro": int(respiro["tamanho"]) if respiro else 0,
            "nums_respiro": nums_respiro,
            "numero_nascimento": numero_nascimento,
            "ultimo": ultimo,
            "bi": bi,
            "tri": tri,
            "contagens_nascimento": contagens_nascimento,
            "contagens_atuais": contagens_atuais,
            "contagens_atravessadas": sorted(set(contagens_nascimento) - set(contagens_atuais)),
            "dna_deslocamento": dna_deslocamento,
            "gramatica_blocos": gramatica_blocos,
        }

    def _chaves_trajetoria_streak(self, sub_num, sub_pol):
        """Chaves hierárquicas e hasháveis da trajetória bilateral V/P."""
        d = self._descrever_trajetoria_streak(sub_num, sub_pol)
        if not d.get("ativo"):
            return []

        base = (
            f"STREAK_TRAJ|C={d['cor']}|EST={d['estagio']}|T={d['tamanho']}"
            f"|TIPO={d['tipo_trajetoria']}"
        )
        chaves = [
            base,
            f"{base}|NASC={d['numero_nascimento']}",
            f"{base}|ULT={d['ultimo']}",
            f"{base}|BI={d['bi']}",
            f"{base}|TRI={d['tri']}",
        ]

        if d["retomada"]:
            retomada = (
                f"{base}|ANT={d['streak_anterior']}|RESP={d['cor_respiro']}{d['tamanho_respiro']}"
            )
            chaves.extend([
                retomada,
                f"{retomada}|NASC={d['numero_nascimento']}",
                f"{retomada}|RESP_NUM={'-'.join(str(x) for x in d['nums_respiro']) or 'SEM'}",
                f"{retomada}|TRI={d['tri']}",
            ])

        if d["contagens_nascimento"]:
            assinatura = ",".join(d["contagens_nascimento"])
            chaves.append(f"{base}|CONT_NASC={assinatura}")
        if d["contagens_atuais"]:
            assinatura = ",".join(d["contagens_atuais"])
            chaves.append(f"{base}|CONT_ATUAL={assinatura}")
        if d["contagens_atravessadas"]:
            assinatura = ",".join(d["contagens_atravessadas"])
            chaves.append(f"{base}|CONT_ATRAV={assinatura}")
            if d["retomada"]:
                chaves.append(
                    f"{base}|TIPO=RETOMADA|ANT={d['streak_anterior']}"
                    f"|RESP={d['cor_respiro']}{d['tamanho_respiro']}|CONT_ATRAV={assinatura}"
                )

        # MAIN 128 — morfologia interna da STREAK sem criar uma nova família/voto.
        # DNA de deslocamento, trajetória de deltas e gramática de blocos apenas
        # condicionam a autoridade da família STREAK já existente.
        dna = d.get("dna_deslocamento", {}) or {}
        if dna.get("ativo"):
            chaves.extend([
                f"{base}|DELTA_DIR={dna.get('assinatura_direcoes')}|TRJ={dna.get('trajetoria')}",
                f"{base}|DELTA_MAG={dna.get('assinatura_magnitudes')}|TRJ={dna.get('trajetoria')}",
            ])

        gramatica = d.get("gramatica_blocos", {}) or {}
        if gramatica.get("ativo"):
            chaves.append(f"{base}|BLOCOS={gramatica.get('assinatura')}")
            if gramatica.get("alternancia_blocos"):
                chaves.append(
                    f"{base}|ALT_BLOCOS=SIM|ATUAL={gramatica.get('tamanho_bloco_atual')}"
                    f"|ANT={gramatica.get('tamanho_bloco_anterior')}"
                )

        return list(dict.fromkeys(chaves))

    def _registrar_trajetoria_streak(self, sub_num, sub_pol, c0, c1):
        rv = self._resultado_ate_g1(c0, c1, "V")
        rp = self._resultado_ate_g1(c0, c1, "P")
        for chave in self._chaves_trajetoria_streak(sub_num, sub_pol):
            st = self.cartografia_trajetoria_streak[hash_chave(chave)]
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

    def obter_voto_trajetoria_streak(self, sub_num, sub_pol, mapa=None):
        """
        Consulta UMA família causal STREAK pela trajetória de nascimento/evolução.
        O mesmo cálculo vale para streak vermelha e preta. A direção é aprendida.
        """
        mapa = mapa if mapa is not None else getattr(self, "cartografia_trajetoria_streak", {})
        descricao = self._descrever_trajetoria_streak(sub_num, sub_pol)
        if not descricao.get("ativo") or not mapa:
            return {"ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0}

        leituras = []
        for chave in self._chaves_trajetoria_streak(sub_num, sub_pol):
            st = mapa.get(hash_chave(chave))
            if not st:
                continue
            suporte = int(st.get("total", 0))
            if suporte < 20:
                continue
            taxa_v = (int(st.get("V_g0", 0)) + int(st.get("V_g1", 0))) / max(suporte, 1)
            taxa_p = (int(st.get("P_g0", 0)) + int(st.get("P_g1", 0))) / max(suporte, 1)
            especificidade = 0.60
            if "|CONT_ATRAV=" in chave:
                especificidade = 1.00
            elif "|DELTA_DIR=" in chave or "|DELTA_MAG=" in chave:
                especificidade = 0.97
            elif "|BLOCOS=" in chave or "|ALT_BLOCOS=" in chave:
                especificidade = 0.96
            elif "|RESP_NUM=" in chave:
                especificidade = 0.96
            elif "|ANT=" in chave and "|RESP=" in chave:
                especificidade = 0.92
            elif "|CONT_NASC=" in chave or "|CONT_ATUAL=" in chave:
                especificidade = 0.90
            elif "|TRI=" in chave:
                especificidade = 0.86
            elif "|BI=" in chave:
                especificidade = 0.80
            elif "|NASC=" in chave or "|ULT=" in chave:
                especificidade = 0.74
            shrink = suporte / (suporte + 30.0)
            leituras.append({
                "chave": chave, "suporte": suporte,
                "taxa_v": taxa_v, "taxa_p": taxa_p,
                "peso": especificidade * shrink,
            })

        if not leituras:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0,
                "trajetoria": descricao,
            }

        soma = sum(x["peso"] for x in leituras)
        taxa_v = sum(x["taxa_v"] * x["peso"] for x in leituras) / max(soma, 1e-9)
        taxa_p = sum(x["taxa_p"] * x["peso"] for x in leituras) / max(soma, 1e-9)
        margem = abs(taxa_v - taxa_p)
        direcao = "NEUTRO" if margem < 0.04 else ("VERMELHO" if taxa_v > taxa_p else "PRETO")
        suporte = max(x["suporte"] for x in leituras)
        suporte_fator = suporte / (suporte + 30.0)
        peso = min(0.90, margem * 5.0 * suporte_fator)

        resultado = {
            "ativo": True,
            "familia": "STREAK",
            "direcao": direcao,
            "peso": round(peso, 4),
            "margem": round(margem, 4),
            "suporte": suporte,
            "taxa_v": round(taxa_v * 100, 2),
            "taxa_p": round(taxa_p * 100, 2),
            "contextos": len(leituras),
            "trajetoria": descricao,
            "altera_direcao_automaticamente": False,
        }
        self._ultimo_voto_trajetoria_streak = dict(resultado)
        return resultado

    def obter_voto_streak_consolidado(self, sub_num, sub_pol):
        """
        MAIN 127 — voz causal única da família STREAK.

        A família é bilateral (VERMELHO e PRETO) e passa a considerar a trajetória
        desde NASCIMENTO/CONFIRMAÇÃO, não apenas o estado terminal de 3 iguais.
        STREAK legada e TRAJETÓRIA não somam votos independentes: são consolidadas
        em uma única leitura antes da arbitragem.
        """
        nums = [int(x) for x in (sub_num or [])]
        pol = [str(x).upper() for x in (sub_pol or [])]
        if not nums or len(nums) != len(pol) or pol[-1] not in ("V", "P"):
            return {"ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0}

        descricao = self._descrever_trajetoria_streak(nums, pol)
        if not descricao.get("ativo"):
            return {"ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0}

        cor_streak = descricao["cor"]
        tamanho = int(descricao["tamanho"])
        leituras = []

        # Memória terminal legada: preservada sem duplicar a causa.
        especificidade_prefixo = {
            "STREAK": 0.76, "STREAK_NUM": 0.82, "STREAK_BI": 0.88,
            "STREAK_TRI": 0.94, "STREAK_EXATO": 1.00,
        }
        for chave in self._chaves_cartografia_padrao(nums, pol):
            prefixo = str(chave).split("|", 1)[0]
            if prefixo not in especificidade_prefixo:
                continue
            st = self.cartografia_padroes_xls.get(chave)
            if not st:
                continue
            total = int(st.get("total", 0))
            if total < 20:
                continue
            taxa_v = (int(st.get("V_g0", 0)) + int(st.get("V_g1", 0))) / max(total, 1)
            taxa_p = (int(st.get("P_g0", 0)) + int(st.get("P_g1", 0))) / max(total, 1)
            leituras.append({
                "fonte": f"LEGADO_{prefixo}", "taxa_v": taxa_v, "taxa_p": taxa_p,
                "total": total,
                "peso": (total / (total + 20.0)) * especificidade_prefixo[prefixo] * 0.70,
            })

        # Trajetória causal: nascimento, confirmação, expansão, retomada/respiro
        # e relação cronológica com contagens. Mesma lógica para V e P.
        mapa_traj = getattr(self, "cartografia_trajetoria_streak", {})
        for chave in self._chaves_trajetoria_streak(nums, pol):
            st = mapa_traj.get(hash_chave(chave))
            if not st:
                continue
            total = int(st.get("total", 0))
            if total < 20:
                continue
            taxa_v = (int(st.get("V_g0", 0)) + int(st.get("V_g1", 0))) / max(total, 1)
            taxa_p = (int(st.get("P_g0", 0)) + int(st.get("P_g1", 0))) / max(total, 1)
            especificidade = 0.60
            if "|CONT_ATRAV=" in chave:
                especificidade = 1.00
            elif "|RESP_NUM=" in chave:
                especificidade = 0.96
            elif "|ANT=" in chave and "|RESP=" in chave:
                especificidade = 0.92
            elif "|CONT_NASC=" in chave or "|CONT_ATUAL=" in chave:
                especificidade = 0.90
            elif "|TRI=" in chave:
                especificidade = 0.86
            elif "|BI=" in chave:
                especificidade = 0.80
            elif "|NASC=" in chave or "|ULT=" in chave:
                especificidade = 0.74
            leituras.append({
                "fonte": "TRAJETORIA", "taxa_v": taxa_v, "taxa_p": taxa_p,
                "total": total,
                "peso": (total / (total + 30.0)) * especificidade,
            })

        if not leituras:
            return {
                "ativo": False, "direcao": "NEUTRO", "peso": 0.0, "suporte": 0,
                "streak": tamanho, "cor_streak": cor_streak, "trajetoria": descricao,
            }

        # Consolidação causal: média ponderada, nunca contagem de "quantos votos".
        soma = sum(x["peso"] for x in leituras)
        taxa_v_macro = sum(x["taxa_v"] * x["peso"] for x in leituras) / max(soma, 1e-9)
        taxa_p_macro = sum(x["taxa_p"] * x["peso"] for x in leituras) / max(soma, 1e-9)
        suporte_macro = max(x["total"] for x in leituras)

        # RECÊNCIA oficial: procura a MESMA assinatura de trajetória nas 200 rodadas.
        rec_total = rec_v = rec_p = 0
        dados_rec = list(getattr(self, "dados_recencia", []) or [])[-200:]
        chaves_atuais = set(hash_chave(c) for c in self._chaves_trajetoria_streak(nums, pol))
        if len(dados_rec) >= 3 and chaves_atuais:
            for i in range(len(dados_rec) - 2):
                inicio = max(0, i - 11)
                trecho = dados_rec[inicio:i + 1]
                n_rec = [int(x["numero"]) for x in trecho]
                c_rec = [str(x["cor"]).upper() for x in trecho]
                chaves_rec = set(hash_chave(c) for c in self._chaves_trajetoria_streak(n_rec, c_rec))
                if not chaves_atuais.intersection(chaves_rec):
                    continue
                g0 = str(dados_rec[i + 1]["cor"]).upper()
                g1 = str(dados_rec[i + 2]["cor"]).upper()
                rec_total += 1
                if g0 in ("V", "B") or g1 in ("V", "B"):
                    rec_v += 1
                if g0 in ("P", "B") or g1 in ("P", "B"):
                    rec_p += 1

        taxa_v = taxa_v_macro
        taxa_p = taxa_p_macro
        peso_recencia_efetivo = 0.0
        taxa_v_rec = taxa_p_rec = None
        if rec_total > 0:
            taxa_v_rec = rec_v / rec_total
            taxa_p_rec = rec_p / rec_total
            confianca_rec = rec_total / (rec_total + 20.0)
            peso_recencia_efetivo = 6.0 * confianca_rec
            taxa_v = (taxa_v_macro + taxa_v_rec * peso_recencia_efetivo) / (1.0 + peso_recencia_efetivo)
            taxa_p = (taxa_p_macro + taxa_p_rec * peso_recencia_efetivo) / (1.0 + peso_recencia_efetivo)

        margem = abs(taxa_v - taxa_p)
        direcao = "NEUTRO" if margem < 0.04 else ("VERMELHO" if taxa_v > taxa_p else "PRETO")
        suporte_fator = suporte_macro / (suporte_macro + 30.0)
        peso_final = min(0.85, margem * 5.0 * suporte_fator)

        resultado = {
            "ativo": True,
            "familia": "STREAK",
            "direcao": direcao,
            "peso": round(peso_final, 4),
            "margem": round(margem, 4),
            "streak": tamanho,
            "cor_streak": cor_streak,
            "estagio": descricao.get("estagio"),
            "tipo_trajetoria": descricao.get("tipo_trajetoria"),
            "trajetoria": descricao,
            "suporte": suporte_macro,
            "fontes_causais": sorted(set(x["fonte"] for x in leituras)),
            "taxa_v_macro": round(taxa_v_macro * 100, 2),
            "taxa_p_macro": round(taxa_p_macro * 100, 2),
            "recencia_suporte": rec_total,
            "taxa_v_recente": round(taxa_v_rec * 100, 2) if taxa_v_rec is not None else None,
            "taxa_p_recente": round(taxa_p_rec * 100, 2) if taxa_p_rec is not None else None,
            "peso_recencia_oficial": 6,
            "peso_recencia_efetivo": round(peso_recencia_efetivo, 4),
            "altera_direcao_automaticamente": False,
        }
        self._ultimo_voto_streak_consolidado = dict(resultado)
        return resultado
