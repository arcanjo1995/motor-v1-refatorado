import numpy as np
from collections import defaultdict
from config.settings import HAS_ML, HAS_HMM, CategoricalHMM, HMM_BACKEND, ERROS_IMPORTACAO_ML
from utils.math_engine import EngineMatematicoAvancado
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
except ImportError:
    pass

class MachineLearningMixin:
    """
    Mixin exclusivo para a extração de features e treinamento neural (GB/MLP/HMM).
    """
    def _construir_features_ml_contextuais(self, sub_num, sub_pol, prob_markov, geometria=None, expectativas=None, modo_mercado=None):
        """MAIN 95 — vetor causal de 32 features estruturais e contextuais."""
        sub_num = [int(n) for n in sub_num[-12:]]
        sub_pol = [str(c).upper() for c in sub_pol[-12:]]
        geometria = geometria or AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
        expectativas = expectativas if expectativas is not None else MotorContagensProjetivas.mapear_janela(sub_num, sub_pol, geometria, None)
        modo_mercado = modo_mercado or AnalisadorContextoAvancado.detectar_modo_mercado(sub_pol, False, None)

        entropia = EngineMatematicoAvancado.calcular_entropia_shannon(sub_pol)
        mv = float((prob_markov or {}).get("V", 0.0))
        mp = float((prob_markov or {}).get("P", 0.0))
        mb = float((prob_markov or {}).get("B", 0.0))
        freq_v = sub_pol.count("V") / max(len(sub_pol), 1)
        freq_p = sub_pol.count("P") / max(len(sub_pol), 1)
        freq_b = sub_pol.count("B") / max(len(sub_pol), 1)
        alternancias = sum(1 for a, b in zip(sub_pol, sub_pol[1:]) if a != b)

        streak = 1
        for i in range(len(sub_pol) - 2, -1, -1):
            if sub_pol[i] == sub_pol[-1]: streak += 1
            else: break
        xadrez = 1
        for i in range(len(sub_pol) - 1, 0, -1):
            if sub_pol[i] != sub_pol[i-1]: xadrez += 1
            else: break

        peso_v = 0.0
        peso_p = 0.0
        familias_v = set()
        familias_p = set()
        contagens_vivas = 0
        contagens_abertas = 0
        for e in expectativas:
            peso_texto = str(e.get("peso", "MEDIO")).upper()
            peso = {"BAIXO": 1.0, "MEDIO": 2.0, "MÉDIO": 2.0, "ALTO": 3.0}.get(peso_texto, 2.0)
            direcao = e.get("direcao")
            familia = e.get("familia", "POSICIONAL")
            if direcao == "VERMELHO": peso_v += peso; familias_v.add(familia)
            elif direcao == "PRETO": peso_p += peso; familias_p.add(familia)
            status = str(e.get("status_contagem", "")).upper()
            if status == "VIVA": contagens_vivas += 1
            elif status == "ABERTA": contagens_abertas += 1

        geo_pvvp = 1.0 if geometria == "CICLO_FECHADO_PVVP" else 0.0
        geo_vppv = 1.0 if geometria == "CICLO_FECHADO_VPPV" else 0.0
        geo_sat_v = 1.0 if geometria == "SATURAÇÃO ESTRUTURAL (V)" else 0.0
        geo_sat_p = 1.0 if geometria == "SATURAÇÃO ESTRUTURAL (P)" else 0.0
        regime_chuva = 1.0 if modo_mercado in ("CHUVA", "REGIME_RECOLHIMENTO") else 0.0
        regime_recuperacao = 1.0 if modo_mercado in ("RECUPERACAO", "REGIME_PAGADOR") else 0.0

        return [
            entropia, mv, mp, mb, mv-mp, abs(mv-mp),
            sub_num[-1], sub_num[-2], sub_num[-3],
            freq_v, freq_p, freq_b, alternancias / 11.0,
            streak, 1.0 if sub_pol[-1] == "V" else -1.0 if sub_pol[-1] == "P" else 0.0,
            xadrez, geo_pvvp, geo_vppv, geo_sat_v, geo_sat_p,
            peso_v, peso_p, peso_v-peso_p, len(familias_v), len(familias_p),
            contagens_vivas, contagens_abertas, regime_chuva, regime_recuperacao,
            sum(1 for n in sub_num[-6:] if n in (2, 6)) / 6.0,
            sum(1 for n in sub_num[-6:] if n in (4, 5, 10)) / 6.0,
            len(set(sub_num[-6:])) / 6.0
        ]

    def _treinar_ml_avancado(self, dados):
        """Validação cronológica robusta e independente para GB e MLP."""
        self.ml_metricas = {}
        if not HAS_ML or len(dados) < 50:
            self.ml_metricas = {
                "ativo": False,
                "motivo": "ML_INDISPONIVEL" if not HAS_ML else "BASE_INSUFICIENTE",
                "has_sklearn": HAS_SKLEARN,
                "has_hmm": HAS_HMM,
                "erros_importacao": dict(ERROS_IMPORTACAO_ML)
            }
            return

        X, y, seq_hmm = [], [], []
        c_map = {'P': 0, 'V': 1, 'B': 2}
        for d in dados:
            seq_hmm.append([c_map.get(d['cor'], 2)])

        if HAS_HMM and CategoricalHMM is not None:
            try:
                self.ml_hmm = CategoricalHMM(n_components=3, n_iter=100, random_state=42)
                if len(seq_hmm) > 50:
                    self.ml_hmm.fit(np.asarray(seq_hmm, dtype=int))
            except Exception as e:
                self.ml_hmm = None
                self.ml_metricas["erro_hmm"] = f"{type(e).__name__}: {e}"
        else:
            self.ml_hmm = None
            self.ml_metricas["hmm_ativo"] = False
            if ERROS_IMPORTACAO_ML.get("hmmlearn"):
                self.ml_metricas["erro_importacao_hmm"] = ERROS_IMPORTACAO_ML["hmmlearn"]

        # Markov causal incremental: mantém exatamente o mesmo conjunto de
        # transições que a reconstrução por prefixo usava em cada amostra,
        # sem reler todo o passado a cada janela.
        ordem_markov_causal = 5
        cores_ml_causais = [d["cor"] for d in dados]
        tabela_markov_causal = defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0})

        # Antes da primeira janela, deixa preparadas as transições j=5..10.
        # A transição j=i+11 é absorvida no início de cada iteração; assim,
        # ao consultar a amostra i, a tabela contém exatamente j=5..i+11,
        # igual ao antigo range(ordem_markov, len(dados[:i+12])).
        limite_inicial = min(11, len(cores_ml_causais))
        for j in range(ordem_markov_causal, limite_inicial):
            estado = tuple(cores_ml_causais[j-ordem_markov_causal:j])
            prox = cores_ml_causais[j]
            if prox in ("V", "P", "B"):
                tabela_markov_causal[estado][prox] += 1.0

        for i in range(len(dados) - 12):
            janela = dados[i:i+12]
            alvo = dados[i+12]['cor']
            pol = [d['cor'] for d in janela]
            num = [d['numero'] for d in janela]

            j_novo = i + 11
            if j_novo >= ordem_markov_causal and j_novo < len(cores_ml_causais):
                estado_novo = tuple(cores_ml_causais[j_novo-ordem_markov_causal:j_novo])
                prox_novo = cores_ml_causais[j_novo]
                if prox_novo in ("V", "P", "B"):
                    tabela_markov_causal[estado_novo][prox_novo] += 1.0

            if alvo not in ['V', 'P']:
                continue
            entropia = EngineMatematicoAvancado.calcular_entropia_shannon(pol)
            # Mesma feature Markov causal anterior, agora consultada em O(1).
            ordem_markov = min(5, len(pol))
            estado_markov = tuple(pol[-ordem_markov:])
            stats_markov_causal = tabela_markov_causal.get(estado_markov)
            if stats_markov_causal is None:
                contagens_markov = {"V": 0.0, "P": 0.0, "B": 0.0}
            else:
                contagens_markov = {
                    "V": float(stats_markov_causal["V"]),
                    "P": float(stats_markov_causal["P"]),
                    "B": float(stats_markov_causal["B"])
                }
            total_markov_causal = sum(contagens_markov.values())
            if total_markov_causal > 0:
                prob_markov = {
                    "V": (contagens_markov["V"] / total_markov_causal) * 100,
                    "P": (contagens_markov["P"] / total_markov_causal) * 100,
                    "B": (contagens_markov["B"] / total_markov_causal) * 100
                }
            else:
                prob_markov = {"V": 46.67, "P": 46.67, "B": 6.66}
            geometria = AnalisadorContextoAvancado.mapear_padroes_geometria(pol)
            expectativas = MotorContagensProjetivas.mapear_janela(num, pol, geometria, None)
            modo_ml = AnalisadorContextoAvancado.detectar_modo_mercado(pol, False, None)
            X.append(self._construir_features_ml_contextuais(
                num, pol, prob_markov, geometria, expectativas, modo_ml
            ))
            y.append(1 if alvo == 'V' else 0)

        if len(X) <= 20:
            self.ml_metricas.update({"ativo": False, "motivo": "AMOSTRAS_ML_INSUFICIENTES"})
            return

        try:
            X_np = np.asarray(X, dtype=float)
            y_np = np.asarray(y, dtype=int)
            corte = max(20, int(len(X_np) * 0.85))
            if corte >= len(X_np):
                corte = len(X_np) - max(5, len(X_np) // 10)
            X_treino, y_treino = X_np[:corte], y_np[:corte]
            X_valid, y_valid = X_np[corte:], y_np[corte:]

            # Validação cronológica do HMM como separador de regimes.
            # O modelo temporário vê somente o trecho de treino e os estados são
            # avaliados no trecho posterior. O resultado é métrica, não voto.
            self.ml_metricas["hmm_backend"] = HMM_BACKEND
            self.ml_metricas["hmm_ativo"] = self.ml_hmm is not None
            if HAS_HMM and CategoricalHMM is not None and len(seq_hmm) > 50:
                try:
                    corte_seq_hmm = max(20, int(len(seq_hmm) * 0.85))
                    hmm_valid = CategoricalHMM(
                        n_components=3, n_iter=100, random_state=42
                    )
                    hmm_valid.fit(
                        np.asarray(seq_hmm[:corte_seq_hmm], dtype=int)
                    )
                    estados_valid = hmm_valid.predict(
                        np.asarray(seq_hmm[corte_seq_hmm:], dtype=int)
                    )
                    obs_valid = np.asarray(
                        seq_hmm[corte_seq_hmm:], dtype=int
                    ).reshape(-1)
                    dist_estados = {}
                    separacoes = []
                    for estado in range(3):
                        mascara = estados_valid == estado
                        suporte_estado = int(np.sum(mascara))
                        if suporte_estado <= 0:
                            continue
                        obs_estado = obs_valid[mascara]
                        pct_v = float(np.mean(obs_estado == 1))
                        pct_p = float(np.mean(obs_estado == 0))
                        separacoes.append(abs(pct_v - pct_p))
                        dist_estados[f"ESTADO_{estado}"] = {
                            "suporte": suporte_estado,
                            "pct_vermelho": round(pct_v * 100.0, 2),
                            "pct_preto": round(pct_p * 100.0, 2),
                            "margem_v_p": round(abs(pct_v - pct_p) * 100.0, 2)
                        }
                    self.ml_metricas["hmm_validacao_cronologica"] = {
                        "ativo": True,
                        "treino_registros": int(corte_seq_hmm),
                        "validacao_registros": int(len(seq_hmm) - corte_seq_hmm),
                        "estados": dist_estados,
                        "margem_media_separacao_v_p": round(
                            (sum(separacoes) / max(1, len(separacoes))) * 100.0, 2
                        ),
                        "uso": "CONDICIONANTE_CARTOGRAFIA_SEM_VOTO_BRUTO"
                    }
                except Exception as e:
                    self.ml_metricas["hmm_validacao_cronologica"] = {
                        "ativo": False,
                        "erro": f"{type(e).__name__}: {e}"
                    }

            self.ml_metricas.update({
                "ativo": True,
                "treino_cronologico_amostras": int(len(X_treino)),
                "validacao_cronologica_amostras": int(len(X_valid)),
                "features_temporais_causais": True,
                "feature_leakage_global_removido": True,
                "features_contextuais_estruturais": True,
                "quantidade_features": 32,
                "versao_features": 2
            })

            resultados = {}
            if len(set(y_treino.tolist())) >= 2 and len(X_valid) > 0:
                try:
                    gb_valid = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
                    gb_valid.fit(X_treino, y_treino)
                    resultados["gb"] = float(np.mean(gb_valid.predict(X_valid) == y_valid))
                except Exception as e:
                    self.ml_metricas["erro_validacao_gb"] = f"{type(e).__name__}: {e}"

                try:
                    mlp_valid = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=800, random_state=42, early_stopping=True, validation_fraction=0.12, n_iter_no_change=25)
                    mlp_valid.fit(X_treino, y_treino)
                    resultados["mlp"] = float(np.mean(mlp_valid.predict(X_valid) == y_valid))
                except Exception as e:
                    self.ml_metricas["erro_validacao_mlp"] = f"{type(e).__name__}: {e}"
                    try:
                        mlp_valid = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=800, random_state=42, early_stopping=False)
                        mlp_valid.fit(X_treino, y_treino)
                        resultados["mlp"] = float(np.mean(mlp_valid.predict(X_valid) == y_valid))
                        self.ml_metricas["fallback_mlp_validacao"] = True
                    except Exception as e2:
                        self.ml_metricas["erro_fallback_validacao_mlp"] = f"{type(e2).__name__}: {e2}"
            else:
                self.ml_metricas["motivo_validacao"] = "TREINO_SEM_DUAS_CLASSES_OU_VALIDACAO_VAZIA"

            acc_gb, acc_mlp = resultados.get("gb"), resultados.get("mlp")
            if acc_gb is not None:
                self.ml_metricas["acuracia_gb"] = round(acc_gb * 100, 2)
            if acc_mlp is not None:
                self.ml_metricas["acuracia_mlp"] = round(acc_mlp * 100, 2)

            if acc_gb is not None and acc_mlp is not None:
                q_gb, q_mlp = max(0.01, acc_gb - 0.45), max(0.01, acc_mlp - 0.45)
                soma = q_gb + q_mlp
                self.ml_pesos = {"gb": q_gb / soma, "mlp": q_mlp / soma}
            elif acc_gb is not None:
                self.ml_pesos = {"gb": 1.0, "mlp": 0.0}
            elif acc_mlp is not None:
                self.ml_pesos = {"gb": 0.0, "mlp": 1.0}
            else:
                self.ml_pesos = {"gb": 0.5, "mlp": 0.5}
            self.ml_metricas["peso_gb"] = round(self.ml_pesos["gb"], 4)
            self.ml_metricas["peso_mlp"] = round(self.ml_pesos["mlp"], 4)

            self.ml_gb, self.ml_mlp = None, None
            if len(set(y_np.tolist())) >= 2:
                try:
                    self.ml_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
                    self.ml_gb.fit(X_np, y_np)
                except Exception as e:
                    self.ml_metricas["erro_treino_final_gb"] = f"{type(e).__name__}: {e}"
                try:
                    self.ml_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=800, random_state=42, early_stopping=True, validation_fraction=0.12, n_iter_no_change=25)
                    self.ml_mlp.fit(X_np, y_np)
                except Exception as e:
                    self.ml_metricas["erro_treino_final_mlp"] = f"{type(e).__name__}: {e}"
                    try:
                        self.ml_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=800, random_state=42, early_stopping=False)
                        self.ml_mlp.fit(X_np, y_np)
                        self.ml_metricas["fallback_mlp_final"] = True
                    except Exception as e2:
                        self.ml_metricas["erro_fallback_final_mlp"] = f"{type(e2).__name__}: {e2}"

            self.ml_ready = self.ml_gb is not None or self.ml_mlp is not None
            self.ml_metricas["modelos_prontos"] = {"gb": self.ml_gb is not None, "mlp": self.ml_mlp is not None, "hmm": self.ml_hmm is not None}
        except Exception as e:
            self.ml_metricas["erro_geral_ml"] = f"{type(e).__name__}: {e}"
            self.ml_ready = False
