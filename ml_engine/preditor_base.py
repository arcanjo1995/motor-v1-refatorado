# meu_motor/ml_engine/preditor_base.py
import os
import pandas as pd
from collections import defaultdict
import pickle
import json
from datetime import datetime
import time
import tempfile
import math
import random
import hashlib
import sys
import gc
import numpy as np

# Importações dos módulos refatorados
from config import NOME_BASE_DEFINITIVA, VERSAO_CHAVES_HASH
from utils.hashing import hash_chave, _mesclar_mapa_hash, fabrica_padrao_detalhado, fabrica_historico_regras_zerado, fabrica_historico_regras_auditado
from utils.math_engine import EngineMatematicoAvancado
from data.leitor_xls import LeitorXLS
from data.persistence import salvar_modelo_longo_prazo, carregar_modelo_longo_prazo
from rules.analisador_contexto import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from rules.motor_nocall import MotorNoCall
from rules.juiz_hierarquico import JuizHierarquicoModificado

# Integração ML
HAS_SKLEARN = False
HAS_HMM = False
ERROS_IMPORTACAO_ML = {}

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    HAS_SKLEARN = True
except Exception as e:
    HAS_SKLEARN = False
    ERROS_IMPORTACAO_ML["sklearn_numpy"] = f"{type(e).__name__}: {e}"

try:
    from hmmlearn.hmm import CategoricalHMM
    HAS_HMM = True
    HMM_BACKEND = "HMMLEARN"
except Exception as e:
    ERROS_IMPORTACAO_ML["hmmlearn"] = f"{type(e).__name__}: {e}"
    CategoricalHMM = None
    HMM_BACKEND = "NUMPY_FALLBACK"

    class CategoricalHMM:
        """
        Fallback categórico discreto em NumPy para ambientes onde o binário do
        hmmlearn é incompatível. Preserva somente a interface fit/predict usada
        pelo Motor V1 e não participa como voto bruto de direção.
        """
        def __init__(self, n_components=3, n_iter=100, random_state=42):
            self.n_components = int(n_components)
            self.n_iter = int(n_iter)
            self.random_state = int(random_state)
            self.n_features = 3
            self.startprob_ = None
            self.transmat_ = None
            self.emissionprob_ = None

        @staticmethod
        def _normalizar(v, eixo=None):
            soma = np.sum(v, axis=eixo, keepdims=True)
            return v / np.maximum(soma, 1e-12)

        def fit(self, X):
            obs = np.asarray(X, dtype=int).reshape(-1)
            if obs.size < 2:
                raise ValueError("Sequência insuficiente para HMM.")
            self.n_features = max(3, int(obs.max()) + 1)
            rng = np.random.default_rng(self.random_state)
            k = self.n_components
            m = self.n_features

            self.startprob_ = self._normalizar(rng.random(k) + 1.0)
            self.transmat_ = self._normalizar(rng.random((k, k)) + 1.0, eixo=1)
            self.emissionprob_ = self._normalizar(rng.random((k, m)) + 1.0, eixo=1)

            anterior = None
            max_iter = min(self.n_iter, 60)
            for _ in range(max_iter):
                t = obs.size
                alpha = np.zeros((t, k), dtype=float)
                escalas = np.zeros(t, dtype=float)

                alpha[0] = self.startprob_ * self.emissionprob_[:, obs[0]]
                escalas[0] = max(alpha[0].sum(), 1e-300)
                alpha[0] /= escalas[0]

                for i in range(1, t):
                    alpha[i] = (alpha[i - 1] @ self.transmat_) * self.emissionprob_[:, obs[i]]
                    escalas[i] = max(alpha[i].sum(), 1e-300)
                    alpha[i] /= escalas[i]

                beta = np.ones((t, k), dtype=float)
                for i in range(t - 2, -1, -1):
                    beta[i] = self.transmat_ @ (
                        self.emissionprob_[:, obs[i + 1]] * beta[i + 1]
                    )
                    beta[i] /= max(escalas[i + 1], 1e-300)

                gamma = self._normalizar(alpha * beta, eixo=1)
                xi_soma = np.zeros((k, k), dtype=float)
                for i in range(t - 1):
                    xi = (
                        alpha[i][:, None]
                        * self.transmat_
                        * (self.emissionprob_[:, obs[i + 1]] * beta[i + 1])[None, :]
                    )
                    xi_soma += xi / max(xi.sum(), 1e-300)

                self.startprob_ = self._normalizar(gamma[0] + 1e-6)
                self.transmat_ = self._normalizar(xi_soma + 1e-6, eixo=1)

                emis = np.full((k, m), 1e-6, dtype=float)
                for simbolo in range(m):
                    emis[:, simbolo] += gamma[obs == simbolo].sum(axis=0)
                self.emissionprob_ = self._normalizar(emis, eixo=1)

                loglik = float(np.log(np.maximum(escalas, 1e-300)).sum())
                if anterior is not None and abs(loglik - anterior) < 1e-5:
                    break
                anterior = loglik
            return self

        def predict(self, X):
            obs = np.asarray(X, dtype=int).reshape(-1)
            if self.startprob_ is None:
                raise ValueError("HMM ainda não treinado.")
            t = obs.size
            k = self.n_components
            log_start = np.log(np.maximum(self.startprob_, 1e-300))
            log_trans = np.log(np.maximum(self.transmat_, 1e-300))
            log_emis = np.log(np.maximum(self.emissionprob_, 1e-300))

            delta = np.zeros((t, k), dtype=float)
            psi = np.zeros((t, k), dtype=int)
            delta[0] = log_start + log_emis[:, obs[0]]

            for i in range(1, t):
                candidatos = delta[i - 1][:, None] + log_trans
                psi[i] = np.argmax(candidatos, axis=0)
                delta[i] = candidatos[psi[i], np.arange(k)] + log_emis[:, obs[i]]

            estados = np.zeros(t, dtype=int)
            estados[-1] = int(np.argmax(delta[-1]))
            for i in range(t - 2, -1, -1):
                estados[i] = psi[i + 1, estados[i + 1]]
            return estados

    HAS_HMM = HAS_SKLEARN

HAS_ML = HAS_SKLEARN

# Avisos
if not HAS_SKLEARN:
    print("Aviso: NumPy/scikit-learn indisponíveis. O Motor seguirá no modo Estatístico Avançado.")
if not HAS_HMM:
    print("Aviso: HMM indisponível. GB e MLP permanecem independentes.")
elif HMM_BACKEND == "NUMPY_FALLBACK":
    print("Aviso: hmmlearn binário indisponível. HMM categórico NumPy ativado como fallback compatível.")

# ============================================================
# CLASSE IAPreditivaV1
# ============================================================
class IAPreditivaV1:
    def __init__(self, dados_longo_prazo, dados_recencia=None):
        self.dados_longo = dados_longo_prazo
        self.dados_recencia = dados_recencia if dados_recencia else []
        self.modelo_transicao = defaultdict(list)
        self.modelo_transicao_profundo = defaultdict(list)
        self.modelo_numerico = defaultdict(list)
        self.transicoes_numericas = defaultdict(lambda: {"total": 0.0, "proximos": defaultdict(float)})
        self.bigramas_numericos = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0, "prox_numero": defaultdict(int)})
        self.saturacao_ciclica = defaultdict(lambda: {"ciclos_V": [], "ciclos_P": [], "historico_distancias": []})
        self.dna_padroes = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        self.padroes_fechamento_numerico = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        self.estatisticas_projecoes_globais = {n: {"total": 0, "g0": 0, "g1": 0, "falha": 0} for n in range(1, 8)}
        self.estatisticas_projecoes_bilaterais = {}
        self.estatisticas_projecoes_respeito = {n: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0} for n in range(1, 8)}
        self.projecoes_respeito_contextual = defaultdict(lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0})
        self.projecoes_respeito_metricas = {}
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
        self.cartografia_padroes_contextual_metricas = {}
        self.ultima_leitura_padrao_contextual = {}
        self.cartografia_trajetoria_streak = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        self.cartografia_trajetoria_streak_metricas = {}
        self._ultimo_voto_trajetoria_streak = {}
        self.cartografia_morfologia_estrutural = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        self.cartografia_morfologia_estrutural_metricas = {}
        self._ultimo_voto_morfologia_estrutural = {}
        self.cartografia_regras_contextual = defaultdict(
            lambda: {
                "total": 0, "g0": 0, "g1": 0, "g2": 0, "falha": 0,
                "V_g0": 0, "V_g1": 0, "V_g2": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_g2": 0, "P_falha": 0,
                "direcao_g0_V": 0, "direcao_g0_P": 0, "direcao_g0_B": 0
            }
        )
        self.cartografia_regras_contextual_metricas = {}
        self.ultima_leitura_regra_contextual = {}
        self.matriz_deriva_comportamental = {}
        self.cartografia_xls_metricas = {}
        self.especialista_espelho_inversao = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        self.estatisticas_bigramas_globais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        self.estatisticas_trigramas_globais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        self.estatisticas_regras_oficiais = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0}
        )
        self.regras_oficiais_metricas = {}
        self.probabilidades_globais = {"streak_v_5": 0.0, "streak_p_5": 0.0, "xadrez_5": 0.0}
        self.unidade_analise = {}
        for n in range(15):
            self.unidade_analise[n] = {
                "ocorrencias": 0, "V": 0, "P": 0, "B": 0,
                "freq_v": 0.0, "freq_p": 0.0, "freq_b": 0.0,
                "estabilidade": "NEUTRO", "saturacao": "NORMAL",
                "enfraquecimento": "ESTÁVEL", "comportamento_dominante": "NEUTRO",
                "pos_numero_V": 0, "pos_numero_P": 0, "pos_numero_B": 0,
                "pos_numero_freq_v": 0.0, "pos_numero_freq_p": 0.0, "pos_numero_freq_b": 0.0,
                "comportamento_pos_numero": "NEUTRO", "retencao_media": 0,
                "ultimas_cores": []
            }
        self.xadrez_stats = {"quebras": 0, "continuacoes": 0, "numeros_quebradores": defaultdict(int)}
        self.streak_breaker_stats = {"V": defaultdict(int), "P": defaultdict(int)}
        self.color_ngrams = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int)}
        self.padroes_xadrez_detalhado = defaultdict(fabrica_padrao_detalhado)
        self.padroes_streak_detalhado = defaultdict(fabrica_padrao_detalhado)
        self.padroes_gerais_detalhado = defaultdict(fabrica_padrao_detalhado)
        self.memoria_padroes_vencedores = []
        self.historico_regras = defaultdict(fabrica_historico_regras_zerado)
        self.auditoria_contrafactual_autorizacao = {
            "total": 0,
            "oficial_g0_g1": 0,
            "oposta_g0_g1": 0,
            "no_call_protegeria_g2_falha": 0,
            "eventos": []
        }
        self.controladores_fortes = defaultdict(int)
        self.padroes_fortes = []
        self.analise_recencia = {}
        self.regime_recencia = None
        self.ml_ready = False
        self.ml_gb = None
        self.ml_mlp = None
        self.ml_hmm = None
        self.ml_pesos = {"gb": 0.5, "mlp": 0.5}
        self.ml_metricas = {}
        self.q_table = {}
        self.q_learning_contextual_metricas = {}
        self.markov_ordens = {ordem: defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0}) for ordem in range(1, 7)}
        self.memoria_conflitos = defaultdict(lambda: {
            "total": 0, "V_g0g1": 0, "P_g0g1": 0,
            "V_g0": 0, "P_g0": 0, "falhas_v": 0, "falhas_p": 0
        })
        self.markov_temporal = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        self.markov_temporal_regime = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        self.temporal_config = {"versao": 1, "piso_memoria_historica": 0.12, "meia_vida_minima": 20000, "peso_maximo_no_juiz": 2.5}
        self.temporal_metricas = {}
        self.ml_atualizacao_incremental_metricas = {}
        self.matriz_evolutiva = {}
        self.ultima_consequencia_futura = {}
        self.ultima_cadeia_causal = {}
        self.hierarquia_oficial_metricas = {
            "ativo": True,
            "versao": 1,
            "ordem": [
                "NO_CALL", "REGRAS_POSICIONAIS", "CONTAGENS", "COEXISTENCIAS",
                "TRANSICOES", "ASSUNCOES", "CONSEQUENCIA_FUTURA",
                "PADROES_VISUAIS", "SINAL"
            ],
            "metodo": "AUTORIDADE_HIERARQUICA_SEM_VOTACAO"
        }
        self.competencia_especialistas = {}
        self.competencia_contextual = defaultdict(lambda: {"total": 0, "acertos": 0})
        self.competencia_contextual_detalhada = defaultdict(lambda: {"total": 0, "acertos": 0})
        self.competencia_contextual_metricas = {}
        self.competencia_metricas = {}
        self.camadas_ampliadas_mapas = {}
        self.camadas_ampliadas_competencia = {}
        self.camadas_ampliadas_contextual = {}
        self.camadas_ampliadas_metricas = {}
        self.ultima_confluencia_camadas_ampliadas = {}
        self.regras_competencia_cronologica = {}
        self.regras_competencia_metricas = {}
        self.memoria_conflitos_metricas = {}
        self.filtro_discriminativo_metricas = {}
        self.filtro_discriminativo_config = {
            "versao": 5,
            "risco_veto": 0.24,
            "risco_contexto_alto": 0.22,
            "risco_precision_minimo": 0.0,
            "contextos_minimos_concordantes": 2,
            "especialistas_alinhados_minimos": 2,
            "suporte_contextual_minimo": 30,
            "suporte_projetiva_minimo": 30,
            "shrinkage_contextual_ativo": True,
            "lift_risco_minimo": 0.03,
            "razao_risco_minima": 1.10,
            "cartografia_no_veto": True,
            "suporte_cartografia_minimo": 30,
            "contextos_cartografia_minimos": 2,
            "fontes_cartografia_minimas": 2,
            "entropia_decisoria_minima": 0.90,
            "familias_conflitantes_minimas": 1,
            "suporte_memoria_conflito_minimo": 30,
            "peso_recencia_oficial_alterado": False
        }
        self.risco_g2_mais_contextos = defaultdict(lambda: {"peso_total": 0.0, "peso_risco": 0.0})
        self.risco_g2_mais_metricas = {}
        self.risco_g2_mais_config = {
            "versao": 2,
            "risco_veto": 0.34,
            "risco_precision_minimo": 0.0,
            "suporte_efetivo_minimo": 50.0,
            "contextos_minimos_concordantes": 2,
            "backoff_hierarquico_ativo": True,
            "suportes_minimos_por_nivel": {
                "EXATO": 12.0,
                "PADRAO": 20.0,
                "GEOMETRIA": 30.0,
                "REGIME": 50.0,
                "REGIME_DIRECAO": 80.0,
                "DIRECAO": 120.0,
                "NUMERO": 35.0
            },
            "pesos_especificidade": {
                "EXATO": 1.00,
                "PADRAO": 0.90,
                "GEOMETRIA": 0.80,
                "REGIME": 0.70,
                "REGIME_DIRECAO": 0.60,
                "DIRECAO": 0.45,
                "NUMERO": 0.65
            },
            "peso_recencia_oficial_alterado": False
        }

        self._treinar_modelo_profundo()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'modelo_transicao' in state: state['modelo_transicao'] = dict(state['modelo_transicao'])
        if 'modelo_transicao_profundo' in state: state['modelo_transicao_profundo'] = dict(state['modelo_transicao_profundo'])
        if 'modelo_numerico' in state: state['modelo_numerico'] = dict(state['modelo_numerico'])
        if 'transicoes_numericas' in state:
            state['transicoes_numericas'] = {
                int(k): {
                    "total": float(v.get("total", 0.0)),
                    "proximos": dict(v.get("proximos", {}))
                }
                for k, v in state['transicoes_numericas'].items()
            }
        if 'bigramas_numericos' in state:
            bn_save = {}
            for k, v in state['bigramas_numericos'].items():
                v_copy = v.copy()
                if 'prox_numero' in v_copy:
                    v_copy['prox_numero'] = dict(v_copy['prox_numero'])
                bn_save[k] = v_copy
            state['bigramas_numericos'] = bn_save
        if 'saturacao_ciclica' in state: state['saturacao_ciclica'] = {k: dict(v) for k, v in state['saturacao_ciclica'].items()}
        if 'dna_padroes' in state: state['dna_padroes'] = {k: dict(v) for k, v in state['dna_padroes'].items()}
        if 'padroes_fechamento_numerico' in state: state['padroes_fechamento_numerico'] = {k: dict(v) for k, v in state['padroes_fechamento_numerico'].items()}
        if 'estatisticas_projecoes_globais' in state: state['estatisticas_projecoes_globais'] = {k: dict(v) for k, v in state['estatisticas_projecoes_globais'].items()}
        if 'estatisticas_projecoes_bilaterais' in state: state['estatisticas_projecoes_bilaterais'] = {}
        if 'estatisticas_projecoes_respeito' in state: state['estatisticas_projecoes_respeito'] = {k: dict(v) for k, v in state['estatisticas_projecoes_respeito'].items()}
        if 'projecoes_respeito_contextual' in state: state['projecoes_respeito_contextual'] = {k: dict(v) for k, v in state['projecoes_respeito_contextual'].items()}
        if 'cartografia_projecoes_trajetoria' in state: state['cartografia_projecoes_trajetoria'] = {k: dict(v) for k, v in state['cartografia_projecoes_trajetoria'].items()}
        if 'cartografia_padroes_xls' in state: state['cartografia_padroes_xls'] = {k: dict(v) for k, v in state['cartografia_padroes_xls'].items()}
        if 'cartografia_padroes_contextual' in state: state['cartografia_padroes_contextual'] = {k: dict(v) for k, v in state['cartografia_padroes_contextual'].items()}
        if 'cartografia_trajetoria_streak' in state: state['cartografia_trajetoria_streak'] = {k: dict(v) for k, v in state['cartografia_trajetoria_streak'].items()}
        if 'cartografia_morfologia_estrutural' in state: state['cartografia_morfologia_estrutural'] = {k: dict(v) for k, v in state['cartografia_morfologia_estrutural'].items()}
        if 'cartografia_regras_contextual' in state: state['cartografia_regras_contextual'] = {k: dict(v) for k, v in state['cartografia_regras_contextual'].items()}
        if 'especialista_espelho_inversao' in state: state['especialista_espelho_inversao'] = {k: dict(v) for k, v in state['especialista_espelho_inversao'].items()}
        if 'estatisticas_bigramas_globais' in state: state['estatisticas_bigramas_globais'] = dict(state['estatisticas_bigramas_globais'])
        if 'estatisticas_trigramas_globais' in state: state['estatisticas_trigramas_globais'] = dict(state['estatisticas_trigramas_globais'])
        if 'estatisticas_regras_oficiais' in state: state['estatisticas_regras_oficiais'] = {k: dict(v) for k, v in state['estatisticas_regras_oficiais'].items()}
        if 'controladores_fortes' in state: state['controladores_fortes'] = dict(state['controladores_fortes'])
        if 'historico_regras' in state: state['historico_regras'] = dict(state['historico_regras'])
        if 'color_ngrams' in state: state['color_ngrams'] = {k: dict(v) for k, v in state['color_ngrams'].items()}
        if 'xadrez_stats' in state and isinstance(state['xadrez_stats'], dict):
            xs = state['xadrez_stats'].copy()
            if 'numeros_quebradores' in xs: xs['numeros_quebradores'] = dict(xs['numeros_quebradores'])
            state['xadrez_stats'] = xs
        if 'streak_breaker_stats' in state and isinstance(state['streak_breaker_stats'], dict):
            ss = state['streak_breaker_stats'].copy()
            for cor_k in ss: ss[cor_k] = dict(ss[cor_k])
            state['streak_breaker_stats'] = ss
        for d_name in ['padroes_xadrez_detalhado', 'padroes_streak_detalhado', 'padroes_gerais_detalhado']:
            if d_name in state:
                pd_dict = {}
                for k, v in state[d_name].items():
                    v_copy = v.copy()
                    if 'quebradores' in v_copy: v_copy['quebradores'] = dict(v_copy['quebradores'])
                    pd_dict[k] = v_copy
                state[d_name] = pd_dict
        if 'q_table' in state: state['q_table'] = dict(state['q_table'])
        if 'markov_ordens' in state:
            state['markov_ordens'] = {
                int(ordem): {chave: dict(stats) for chave, stats in tabela.items()}
                for ordem, tabela in state['markov_ordens'].items()
            }
        if 'memoria_conflitos' in state:
            state['memoria_conflitos'] = {
                chave: dict(stats) for chave, stats in state['memoria_conflitos'].items()
            }
        if 'competencia_contextual' in state:
            state['competencia_contextual'] = {
                chave: dict(stats) for chave, stats in state['competencia_contextual'].items()
            }
        if 'competencia_contextual_detalhada' in state:
            state['competencia_contextual_detalhada'] = {
                chave: dict(stats) for chave, stats in state['competencia_contextual_detalhada'].items()
            }
        if 'risco_g2_mais_contextos' in state:
            state['risco_g2_mais_contextos'] = {
                chave: dict(stats) for chave, stats in state['risco_g2_mais_contextos'].items()
            }
        if 'camadas_ampliadas_mapas' in state:
            state['camadas_ampliadas_mapas'] = {
                nome: {chave: dict(stats) for chave, stats in mapa.items()}
                for nome, mapa in state['camadas_ampliadas_mapas'].items()
            }
        if 'camadas_ampliadas_contextual' in state:
            state['camadas_ampliadas_contextual'] = {
                chave: dict(stats) for chave, stats in state['camadas_ampliadas_contextual'].items()
            }
        for nome_temporal in ['markov_temporal', 'markov_temporal_regime']:
            if nome_temporal in state:
                state[nome_temporal] = {
                    int(ordem): {chave: dict(stats) for chave, stats in tabela.items()}
                    for ordem, tabela in state[nome_temporal].items()
                }
        return state

    def _normalizar_unidade_analise_compatibilidade(self):
        carregada = getattr(self, "unidade_analise", {})
        if not isinstance(carregada, dict):
            carregada = {}
        campos_padrao = {
            "ocorrencias": 0, "V": 0, "P": 0, "B": 0,
            "freq_v": 0.0, "freq_p": 0.0, "freq_b": 0.0,
            "estabilidade": "NEUTRO", "saturacao": "NORMAL",
            "enfraquecimento": "ESTÁVEL",
            "comportamento_dominante": "NEUTRO",
            "pos_numero_V": 0, "pos_numero_P": 0, "pos_numero_B": 0,
            "pos_numero_freq_v": 0.0, "pos_numero_freq_p": 0.0,
            "pos_numero_freq_b": 0.0,
            "comportamento_pos_numero": "NEUTRO",
            "retencao_media": 0,
            "ultimas_cores": []
        }
        normalizada = {}
        for n in range(15):
            registro = carregada.get(n)
            if not isinstance(registro, dict):
                registro = carregada.get(str(n))
            if not isinstance(registro, dict):
                registro = {}
            registro_normalizado = dict(registro)
            for campo, valor_padrao in campos_padrao.items():
                if campo not in registro_normalizado:
                    registro_normalizado[campo] = (
                        list(valor_padrao) if isinstance(valor_padrao, list)
                        else valor_padrao
                    )
            if not isinstance(registro_normalizado.get("ultimas_cores"), list):
                try:
                    registro_normalizado["ultimas_cores"] = list(
                        registro_normalizado.get("ultimas_cores") or []
                    )
                except Exception:
                    registro_normalizado["ultimas_cores"] = []
            normalizada[n] = registro_normalizado
        self.unidade_analise = normalizada
        return True

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._normalizar_unidade_analise_compatibilidade()
        self.modelo_transicao = defaultdict(list, state.get('modelo_transicao', {}))
        self.modelo_transicao_profundo = defaultdict(list, state.get('modelo_transicao_profundo', {}))
        self.modelo_numerico = defaultdict(list, state.get('modelo_numerico', {}))
        self.transicoes_numericas = defaultdict(
            lambda: {"total": 0.0, "proximos": defaultdict(float)}
        )
        for k, v in state.get('transicoes_numericas', {}).items():
            self.transicoes_numericas[int(k)] = {
                "total": float(v.get("total", 0.0)),
                "proximos": defaultdict(
                    float,
                    {int(n): float(q) for n, q in v.get("proximos", {}).items()}
                )
            }
        bigramas_loaded = state.get('bigramas_numericos', {})
        self.bigramas_numericos = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0, "prox_numero": defaultdict(int)})
        for k, v in bigramas_loaded.items():
            self.bigramas_numericos[k] = {"V": v.get("V",0), "P": v.get("P",0), "B": v.get("B",0), "total": v.get("total",0), "prox_numero": defaultdict(int, v.get("prox_numero", {}))}
        saturacao_loaded = state.get('saturacao_ciclica', {})
        self.saturacao_ciclica = defaultdict(lambda: {"ciclos_V": [], "ciclos_P": [], "historico_distancias": []})
        for k, v in saturacao_loaded.items():
            self.saturacao_ciclica[k] = {"ciclos_V": v.get("ciclos_V", []), "ciclos_P": v.get("ciclos_P", []), "historico_distancias": v.get("historico_distancias", [])}
        dna_loaded = state.get('dna_padroes', {})
        self.dna_padroes = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        for k, v in dna_loaded.items():
            self.dna_padroes[k] = {"V": v.get("V",0), "P": v.get("P",0), "B": v.get("B",0), "total": v.get("total",0)}
        pfn_loaded = state.get('padroes_fechamento_numerico', {})
        self.padroes_fechamento_numerico = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        for k, v in pfn_loaded.items():
            self.padroes_fechamento_numerico[k] = {"V": v.get("V",0), "P": v.get("P",0), "B": v.get("B",0), "total": v.get("total",0)}
        epg_loaded = state.get('estatisticas_projecoes_globais', {})
        self.estatisticas_projecoes_globais = {n: {"total": 0, "g0": 0, "g1": 0, "falha": 0} for n in range(1, 8)}
        for k, v in epg_loaded.items():
            self.estatisticas_projecoes_globais[int(k)] = {"total": v.get("total",0), "g0": v.get("g0",0), "g1": v.get("g1",0), "falha": v.get("falha",0)}
        self.estatisticas_projecoes_bilaterais = {}
        epr_loaded = state.get('estatisticas_projecoes_respeito', {})
        self.estatisticas_projecoes_respeito = {n: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0} for n in range(1, 8)}
        for k, v in epr_loaded.items():
            self.estatisticas_projecoes_respeito[int(k)] = {
                "total": int(v.get("total", 0)),
                "respeitada_g0": int(v.get("respeitada_g0", 0)),
                "respeitada_g1": int(v.get("respeitada_g1", 0)),
                "nao_respeitada": int(v.get("nao_respeitada", 0))
            }
        self.projecoes_respeito_contextual = defaultdict(lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0})
        for chave, v in state.get('projecoes_respeito_contextual', {}).items():
            self.projecoes_respeito_contextual[chave] = {
                "total": int(v.get("total", 0)),
                "respeitada_g0": int(v.get("respeitada_g0", 0)),
                "respeitada_g1": int(v.get("respeitada_g1", 0)),
                "nao_respeitada": int(v.get("nao_respeitada", 0))
            }
        self.projecoes_respeito_metricas = state.get('projecoes_respeito_metricas', {})
        self.cartografia_projecoes_trajetoria = defaultdict(
            lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0}
        )
        for chave, v in state.get('cartografia_projecoes_trajetoria', {}).items():
            self.cartografia_projecoes_trajetoria[chave] = {
                "total": int(v.get("total", 0)),
                "respeitada_g0": int(v.get("respeitada_g0", 0)),
                "respeitada_g1": int(v.get("respeitada_g1", 0)),
                "nao_respeitada": int(v.get("nao_respeitada", 0))
            }
        self.cartografia_padroes_xls = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0, "B_g0": 0}
        )
        for chave, v in state.get('cartografia_padroes_xls', {}).items():
            self.cartografia_padroes_xls[chave] = {
                "total": int(v.get("total", 0)),
                "V_g0": int(v.get("V_g0", 0)),
                "V_g1": int(v.get("V_g1", 0)),
                "P_g0": int(v.get("P_g0", 0)),
                "P_g1": int(v.get("P_g1", 0)),
                "B_g0": int(v.get("B_g0", 0))
            }
        self.cartografia_padroes_contextual = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        mapa_padroes_hash = _mesclar_mapa_hash(state.get('cartografia_padroes_contextual', {}))
        for chave, v in mapa_padroes_hash.items():
            self.cartografia_padroes_contextual[chave] = {
                "total": int(v.get("total", 0)),
                "V_g0": int(v.get("V_g0", 0)),
                "V_g1": int(v.get("V_g1", 0)),
                "V_falha": int(v.get("V_falha", 0)),
                "P_g0": int(v.get("P_g0", 0)),
                "P_g1": int(v.get("P_g1", 0)),
                "P_falha": int(v.get("P_falha", 0)),
                "B_g0": int(v.get("B_g0", 0))
            }
        self.cartografia_padroes_contextual_metricas = state.get('cartografia_padroes_contextual_metricas', {})
        self.ultima_leitura_padrao_contextual = state.get('ultima_leitura_padrao_contextual', {})
        self.cartografia_trajetoria_streak = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        mapa_streak_traj_hash = _mesclar_mapa_hash(state.get('cartografia_trajetoria_streak', {}))
        for chave, v in mapa_streak_traj_hash.items():
            self.cartografia_trajetoria_streak[chave] = {
                "total": int(v.get("total", 0)),
                "V_g0": int(v.get("V_g0", 0)), "V_g1": int(v.get("V_g1", 0)),
                "V_falha": int(v.get("V_falha", 0)),
                "P_g0": int(v.get("P_g0", 0)), "P_g1": int(v.get("P_g1", 0)),
                "P_falha": int(v.get("P_falha", 0)),
                "B_g0": int(v.get("B_g0", 0))
            }
        self.cartografia_trajetoria_streak_metricas = state.get('cartografia_trajetoria_streak_metricas', {})
        self._ultimo_voto_trajetoria_streak = state.get('_ultimo_voto_trajetoria_streak', {})
        self.cartografia_morfologia_estrutural = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        mapa_morfologia_hash = _mesclar_mapa_hash(state.get('cartografia_morfologia_estrutural', {}))
        for chave, v in mapa_morfologia_hash.items():
            self.cartografia_morfologia_estrutural[chave] = {
                "total": int(v.get("total", 0)),
                "V_g0": int(v.get("V_g0", 0)), "V_g1": int(v.get("V_g1", 0)),
                "V_falha": int(v.get("V_falha", 0)),
                "P_g0": int(v.get("P_g0", 0)), "P_g1": int(v.get("P_g1", 0)),
                "P_falha": int(v.get("P_falha", 0)),
                "B_g0": int(v.get("B_g0", 0))
            }
        self.cartografia_morfologia_estrutural_metricas = state.get('cartografia_morfologia_estrutural_metricas', {})
        self._ultimo_voto_morfologia_estrutural = state.get('_ultimo_voto_morfologia_estrutural', {})
        self.cartografia_regras_contextual = defaultdict(
            lambda: {
                "total": 0, "g0": 0, "g1": 0, "g2": 0, "falha": 0,
                "V_g0": 0, "V_g1": 0, "V_g2": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_g2": 0, "P_falha": 0,
                "direcao_g0_V": 0, "direcao_g0_P": 0, "direcao_g0_B": 0
            }
        )
        for chave, v in state.get('cartografia_regras_contextual', {}).items():
            self.cartografia_regras_contextual[chave] = {
                "total": int(v.get("total", 0)),
                "g0": int(v.get("g0", 0)), "g1": int(v.get("g1", 0)),
                "g2": int(v.get("g2", 0)), "falha": int(v.get("falha", 0)),
                "V_g0": int(v.get("V_g0", 0)), "V_g1": int(v.get("V_g1", 0)),
                "V_g2": int(v.get("V_g2", 0)), "V_falha": int(v.get("V_falha", 0)),
                "P_g0": int(v.get("P_g0", 0)), "P_g1": int(v.get("P_g1", 0)),
                "P_g2": int(v.get("P_g2", 0)), "P_falha": int(v.get("P_falha", 0)),
                "direcao_g0_V": int(v.get("direcao_g0_V", 0)),
                "direcao_g0_P": int(v.get("direcao_g0_P", 0)),
                "direcao_g0_B": int(v.get("direcao_g0_B", 0))
            }
        self.cartografia_regras_contextual_metricas = state.get('cartografia_regras_contextual_metricas', {})
        self.matriz_deriva_comportamental = state.get('matriz_deriva_comportamental', {})
        self.ultima_leitura_regra_contextual = state.get('ultima_leitura_regra_contextual', {})
        self.cartografia_xls_metricas = state.get('cartografia_xls_metricas', {})
        eei_loaded = state.get('especialista_espelho_inversao', {})
        self.especialista_espelho_inversao = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in eei_loaded.items():
            self.especialista_espelho_inversao[k] = {"total": v.get("total", 0), "V_g0": v.get("V_g0", 0), "V_g1": v.get("V_g1", 0), "P_g0": v.get("P_g0", 0), "P_g1": v.get("P_g1", 0)}
        ebg_loaded = state.get('estatisticas_bigramas_globais', {})
        self.estatisticas_bigramas_globais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in ebg_loaded.items():
            self.estatisticas_bigramas_globais[k] = {"total": v.get("total",0), "V_g0": v.get("V_g0",0), "V_g1": v.get("V_g1",0), "P_g0": v.get("P_g0",0), "P_g1": v.get("P_g1",0)}
        etg_loaded = state.get('estatisticas_trigramas_globais', {})
        self.estatisticas_trigramas_globais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in etg_loaded.items():
            self.estatisticas_trigramas_globais[k] = {"total": v.get("total",0), "V_g0": v.get("V_g0",0), "V_g1": v.get("V_g1",0), "P_g0": v.get("P_g0",0), "P_g1": v.get("P_g1",0)}
        ero_loaded = state.get('estatisticas_regras_oficiais', {})
        self.estatisticas_regras_oficiais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in ero_loaded.items():
            self.estatisticas_regras_oficiais[k] = {"total": v.get("total", 0), "V_g0": v.get("V_g0", 0), "V_g1": v.get("V_g1", 0), "P_g0": v.get("P_g0", 0), "P_g1": v.get("P_g1", 0)}
        self.regras_oficiais_metricas = state.get('regras_oficiais_metricas', {})
        self.probabilidades_globais = state.get('probabilidades_globais', {"streak_v_5": 0.0, "streak_p_5": 0.0, "xadrez_5": 0.0})
        self.controladores_fortes = defaultdict(int, state.get('controladores_fortes', {}))
        self.historico_regras = defaultdict(fabrica_historico_regras_zerado)
        for k, v in state.get('historico_regras', {}).items():
            self.historico_regras[k] = {"acertos": v.get("acertos", 0), "total": v.get("total", 0)}
        c_grams = state.get('color_ngrams', {1: {}, 2: {}, 3: {}})
        self.color_ngrams = {k: defaultdict(int, v) for k, v in c_grams.items()}
        x_st = state.get('xadrez_stats', {})
        self.xadrez_stats = {"quebras": x_st.get("quebras", 0), "continuacoes": x_st.get("continuacoes", 0), "numeros_quebradores": defaultdict(int, x_st.get("numeros_quebradores", {}))}
        s_st = state.get('streak_breaker_stats', {"V": {}, "P": {}})
        self.streak_breaker_stats = {"V": defaultdict(int, s_st.get("V", {})), "P": defaultdict(int, s_st.get("P", {}))}
        for d_name in ['padroes_xadrez_detalhado', 'padroes_streak_detalhado', 'padroes_gerais_detalhado']:
            if d_name in state:
                pd_dict = {}
                for k, v in state[d_name].items():
                    v_copy = v.copy()
                    if 'quebradores' in v_copy: v_copy['quebradores'] = dict(v_copy['quebradores'])
                    pd_dict[k] = v_copy
                state[d_name] = pd_dict
        self.q_table = _mesclar_mapa_hash(state.get('q_table', {}))
        self.q_learning_contextual_metricas = state.get('q_learning_contextual_metricas', {})
        self.versao_chaves_hash = VERSAO_CHAVES_HASH
        self.ml_gb = state.get('ml_gb')
        self.ml_mlp = state.get('ml_mlp')
        self.ml_hmm = state.get('ml_hmm')
        self.ml_pesos = state.get('ml_pesos', {"gb": 0.5, "mlp": 0.5})
        self.ml_metricas = state.get('ml_metricas', {})
        self.ml_ready = bool(
            state.get('ml_ready', False)
            and self.ml_gb is not None
            and self.ml_mlp is not None
        )
        markov_loaded = state.get('markov_ordens', {})
        self.markov_ordens = {
            ordem: defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
            for ordem in range(1, 7)
        }
        for ordem, tabela in markov_loaded.items():
            ordem_int = int(ordem)
            if ordem_int not in self.markov_ordens:
                self.markov_ordens[ordem_int] = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
            for chave, stats in tabela.items():
                chave_final = tuple(chave) if isinstance(chave, list) else chave
                self.markov_ordens[ordem_int][chave_final] = {
                    "V": stats.get("V", 0), "P": stats.get("P", 0),
                    "B": stats.get("B", 0), "total": stats.get("total", 0)
                }
        self.memoria_conflitos = defaultdict(lambda: {
            "total": 0, "V_g0g1": 0, "P_g0g1": 0,
            "V_g0": 0, "P_g0": 0, "falhas_v": 0, "falhas_p": 0
        })
        mapa_conflitos_hash = _mesclar_mapa_hash(state.get('memoria_conflitos', {}))
        for chave, stats in mapa_conflitos_hash.items():
            self.memoria_conflitos[chave] = {
                "total": stats.get("total", 0),
                "V_g0g1": stats.get("V_g0g1", 0),
                "P_g0g1": stats.get("P_g0g1", 0),
                "V_g0": stats.get("V_g0", 0),
                "P_g0": stats.get("P_g0", 0),
                "falhas_v": stats.get("falhas_v", 0),
                "falhas_p": stats.get("falhas_p", 0)
            }
        self.temporal_config = state.get('temporal_config', {"versao": 1, "piso_memoria_historica": 0.12, "meia_vida_minima": 20000, "peso_maximo_no_juiz": 2.5})
        self.temporal_metricas = state.get('temporal_metricas', {})
        self.ml_atualizacao_incremental_metricas = state.get('ml_atualizacao_incremental_metricas', {})
        self.matriz_evolutiva = state.get('matriz_evolutiva', {})
        self.ultima_consequencia_futura = state.get('ultima_consequencia_futura', {})
        self.ultima_cadeia_causal = state.get('ultima_cadeia_causal', {})
        self.hierarquia_oficial_metricas = state.get('hierarquia_oficial_metricas', {
            "ativo": True, "versao": 1,
            "ordem": ["NO_CALL", "REGRAS_POSICIONAIS", "CONTAGENS", "COEXISTENCIAS", "TRANSICOES", "ASSUNCOES", "CONSEQUENCIA_FUTURA", "PADROES_VISUAIS", "SINAL"],
            "metodo": "AUTORIDADE_HIERARQUICA_SEM_VOTACAO"
        })
        self.competencia_especialistas = state.get('competencia_especialistas', {})
        self.competencia_metricas = state.get('competencia_metricas', {})
        self.competencia_contextual = defaultdict(lambda: {"total": 0, "acertos": 0})
        for chave, stats in state.get('competencia_contextual', {}).items():
            self.competencia_contextual[chave] = {"total": int(stats.get("total", 0)), "acertos": int(stats.get("acertos", 0))}
        self.competencia_contextual_detalhada = defaultdict(lambda: {"total": 0, "acertos": 0})
        for chave, stats in state.get('competencia_contextual_detalhada', {}).items():
            self.competencia_contextual_detalhada[chave] = {"total": int(stats.get("total", 0)), "acertos": int(stats.get("acertos", 0))}
        self.competencia_contextual_metricas = state.get('competencia_contextual_metricas', {})
        self.camadas_ampliadas_mapas = {}
        for nome, mapa in state.get('camadas_ampliadas_mapas', {}).items():
            self.camadas_ampliadas_mapas[nome] = {
                chave: {
                    "V": int(stats.get("V", 0)),
                    "P": int(stats.get("P", 0)),
                    "total": int(stats.get("total", 0))
                }
                for chave, stats in mapa.items()
            }
        self.camadas_ampliadas_competencia = state.get('camadas_ampliadas_competencia', {})
        self.camadas_ampliadas_contextual = {
            chave: {
                "total": int(stats.get("total", 0)),
                "acertos": int(stats.get("acertos", 0))
            }
            for chave, stats in state.get('camadas_ampliadas_contextual', {}).items()
        }
        self.camadas_ampliadas_metricas = state.get('camadas_ampliadas_metricas', {})
        self.ultima_confluencia_camadas_ampliadas = state.get('ultima_confluencia_camadas_ampliadas', {})
        self.regras_competencia_cronologica = state.get('regras_competencia_cronologica', {})
        self.regras_competencia_metricas = state.get('regras_competencia_metricas', {})
        self.memoria_conflitos_metricas = state.get('memoria_conflitos_metricas', {})

        filtro_config_padrao = {
            "versao": 5,
            "risco_veto": 0.24,
            "risco_contexto_alto": 0.22,
            "risco_precision_minimo": 0.0,
            "contextos_minimos_concordantes": 2,
            "especialistas_alinhados_minimos": 2,
            "suporte_contextual_minimo": 30,
            "suporte_projetiva_minimo": 30,
            "shrinkage_contextual_ativo": True,
            "lift_risco_minimo": 0.03,
            "razao_risco_minima": 1.10,
            "cartografia_no_veto": True,
            "suporte_cartografia_minimo": 30,
            "contextos_cartografia_minimos": 2,
            "fontes_cartografia_minimas": 2,
            "entropia_decisoria_minima": 0.90,
            "familias_conflitantes_minimas": 1,
            "suporte_memoria_conflito_minimo": 30,
            "peso_recencia_oficial_alterado": False
        }
        filtro_config_salvo = state.get('filtro_discriminativo_config', {})
        self.filtro_discriminativo_config = filtro_config_padrao.copy()
        for chave_config in filtro_config_padrao:
            if (
                chave_config in filtro_config_salvo
                and chave_config not in (
                    "versao", "risco_veto", "risco_contexto_alto", "risco_precision_minimo",
                    "lift_risco_minimo", "razao_risco_minima"
                )
            ):
                self.filtro_discriminativo_config[chave_config] = filtro_config_salvo[chave_config]
        self.filtro_discriminativo_config["versao"] = 5
        self.filtro_discriminativo_config["risco_veto"] = 0.24
        self.filtro_discriminativo_config["risco_contexto_alto"] = 0.22
        self.filtro_discriminativo_config["risco_precision_minimo"] = 0.0
        self.filtro_discriminativo_config["lift_risco_minimo"] = 0.03
        self.filtro_discriminativo_config["razao_risco_minima"] = 1.10
        self.filtro_discriminativo_config["cartografia_no_veto"] = True
        self.filtro_discriminativo_config["suporte_cartografia_minimo"] = 30
        self.filtro_discriminativo_config["contextos_cartografia_minimos"] = 2
        self.filtro_discriminativo_config["fontes_cartografia_minimas"] = 2
        self.filtro_discriminativo_config["entropia_decisoria_minima"] = 0.90
        self.filtro_discriminativo_config["familias_conflitantes_minimas"] = 1
        self.filtro_discriminativo_config["suporte_memoria_conflito_minimo"] = 30
        self.filtro_discriminativo_metricas = state.get('filtro_discriminativo_metricas', {})
        self.risco_g2_mais_contextos = defaultdict(lambda: {"peso_total": 0.0, "peso_risco": 0.0})
        for chave, stats in state.get('risco_g2_mais_contextos', {}).items():
            self.risco_g2_mais_contextos[chave] = {
                "peso_total": float(stats.get("peso_total", 0.0)),
                "peso_risco": float(stats.get("peso_risco", 0.0))
            }
        self.risco_g2_mais_metricas = state.get('risco_g2_mais_metricas', {})
        risco_config_padrao = {
            "versao": 2,
            "risco_veto": 0.34,
            "risco_precision_minimo": 0.0,
            "suporte_efetivo_minimo": 50.0,
            "contextos_minimos_concordantes": 2,
            "backoff_hierarquico_ativo": True,
            "suportes_minimos_por_nivel": {
                "EXATO": 12.0, "PADRAO": 20.0, "GEOMETRIA": 30.0,
                "REGIME": 50.0, "REGIME_DIRECAO": 80.0,
                "DIRECAO": 120.0, "NUMERO": 35.0
            },
            "pesos_especificidade": {
                "EXATO": 1.00, "PADRAO": 0.90, "GEOMETRIA": 0.80,
                "REGIME": 0.70, "REGIME_DIRECAO": 0.60,
                "DIRECAO": 0.45, "NUMERO": 0.65
            },
            "peso_recencia_oficial_alterado": False
        }
        risco_config_salvo = state.get('risco_g2_mais_config', {})
        self.risco_g2_mais_config = risco_config_padrao.copy()
        for chave_config in risco_config_padrao:
            if chave_config in risco_config_salvo and chave_config != "versao":
                self.risco_g2_mais_config[chave_config] = risco_config_salvo[chave_config]
        self.risco_g2_mais_config["versao"] = 2
        self.markov_temporal = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        self.markov_temporal_regime = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        for nome_temporal in ['markov_temporal', 'markov_temporal_regime']:
            destino = getattr(self, nome_temporal)
            for ordem, tabela in state.get(nome_temporal, {}).items():
                ordem_int = int(ordem)
                for chave, stats in tabela.items():
                    chave_final = tuple(chave) if isinstance(chave, list) else chave
                    destino[ordem_int][chave_final] = {
                        "V": float(stats.get("V", 0.0)), "P": float(stats.get("P", 0.0)),
                        "B": float(stats.get("B", 0.0)), "total": float(stats.get("total", 0.0))
                    }

        for d_name in ['padroes_xadrez_detalhado', 'padroes_streak_detalhado', 'padroes_gerais_detalhado']:
            _mapa_carregado = getattr(self, d_name, {})
            _mapa_restaurado = defaultdict(fabrica_padrao_detalhado)
            for _chave_padrao, _detalhe_padrao in dict(_mapa_carregado or {}).items():
                _detalhe_restaurado = fabrica_padrao_detalhado()
                if isinstance(_detalhe_padrao, dict):
                    _detalhe_restaurado.update(_detalhe_padrao)
                    _detalhe_restaurado["quebradores"] = defaultdict(
                        int,
                        dict(_detalhe_padrao.get("quebradores", {}) or {})
                    )
                _mapa_restaurado[_chave_padrao] = _detalhe_restaurado
            setattr(self, d_name, _mapa_restaurado)
        return state

    # ============================================================
    # MÉTODOS DE TREINAMENTO E CARTÓGRAFIA
    # ============================================================

    def _treinar_modelo_profundo(self):
        if self.dados_longo and len(self.dados_longo) >= 5:
            self._processar_bloco_dados(self.dados_longo, 1, True)
            self._calcular_probabilidades_globais_cache()
        if self.dados_recencia and len(self.dados_recencia) >= 5:
            self._processar_bloco_dados(self.dados_recencia, 4, True)
        todos_dados = (self.dados_longo or []) + (self.dados_recencia or [])
        if todos_dados:
            self._mapear_projecoes_globais(todos_dados)
            self._mapear_cartografia_completa_xls(todos_dados)
            self._treinar_markov_multiescala()
            self._mapear_cartografia_contextual_regras_contagens(todos_dados)
            self._treinar_memoria_temporal_adaptativa()
            self._validar_competencia_especialistas_cronologica(todos_dados)
            self._treinar_memoria_conflitos_base_longa(todos_dados)
            self._validar_regras_posicionais_cronologica(todos_dados)
            self._validar_competencia_camadas_ampliadas_cronologica(todos_dados)
            self._treinar_risco_g2_mais_base_longa()
            self._treinar_ml_avancado(todos_dados)

    def _treinar_markov_multiescala(self):
        self.markov_ordens = {
            ordem: defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
            for ordem in range(1, 7)
        }
        def absorver(dados, peso_base=1, temporal=False):
            if not dados:
                return
            cores = [str(d.get("cor", "B")).upper() for d in dados]
            total = len(cores)
            for ordem in range(1, 7):
                if total <= ordem:
                    continue
                for i in range(ordem, total):
                    estado = tuple(cores[i-ordem:i])
                    proxima = cores[i]
                    if proxima not in ("V", "P", "B"):
                        continue
                    fator = peso_base
                    if temporal and total > 1000:
                        fator = max(1, int(peso_base * (1.0 + (i / total) * 1.5)))
                    stats = self.markov_ordens[ordem][estado]
                    stats[proxima] += fator
                    stats["total"] += fator
        absorver(self.dados_longo or [], peso_base=1, temporal=True)
        absorver(self.dados_recencia or [], peso_base=6, temporal=False)

    @staticmethod
    def _detectar_regime_temporal(cores):
        janela = [c for c in (cores or [])[-12:] if c in ("V", "P")]
        if len(janela) < 4:
            return "MISTO"
        alternancias = sum(1 for i in range(1, len(janela)) if janela[i] != janela[i-1])
        freq_alt = alternancias / max(1, len(janela) - 1)
        streaks = []
        atual = 1
        for i in range(1, len(janela)):
            if janela[i] == janela[i-1]:
                atual += 1
            else:
                streaks.append(atual)
                atual = 1
        streaks.append(atual)
        streak_medio = sum(streaks) / len(streaks)
        if freq_alt >= 0.60:
            return "XADREZ_DOMINANTE"
        if streak_medio >= 2.5 or max(streaks) >= 4:
            return "STREAK_DOMINANTE"
        return "MISTO"

    def _treinar_memoria_temporal_adaptativa(self):
        self.markov_temporal = {
            ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0})
            for ordem in range(1, 7)
        }
        self.markov_temporal_regime = {
            ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0})
            for ordem in range(1, 7)
        }
        dados = self.dados_longo or []
        total = len(dados)
        if total < 30:
            self.temporal_metricas = {"ativo": False, "motivo": "BASE_INSUFICIENTE"}
            return
        cores = [str(d.get("cor", "B")).upper() for d in dados]
        meia_vida = max(
            int(self.temporal_config.get("meia_vida_minima", 20000)),
            max(1, total // 4)
        )
        piso = float(self.temporal_config.get("piso_memoria_historica", 0.12))
        soma_pesos = 0.0
        for i in range(1, total):
            idade = (total - 1) - i
            peso_tempo = max(piso, 0.5 ** (idade / meia_vida))
            proxima = cores[i]
            if proxima not in ("V", "P", "B"):
                continue
            regime = self._detectar_regime_temporal(cores[max(0, i-12):i])
            soma_pesos += peso_tempo
            for ordem in range(1, min(6, i) + 1):
                estado = tuple(cores[i-ordem:i])
                stats = self.markov_temporal[ordem][estado]
                stats[proxima] += peso_tempo
                stats["total"] += peso_tempo
                chave_regime = (regime, estado)
                stats_regime = self.markov_temporal_regime[ordem][chave_regime]
                stats_regime[proxima] += peso_tempo
                stats_regime["total"] += peso_tempo
        self.temporal_metricas = {
            "ativo": True,
            "registros_base_longa": total,
            "meia_vida_registros": meia_vida,
            "piso_memoria_historica": piso,
            "massa_temporal_efetiva": round(soma_pesos, 2),
            "recencia_oficial_preservada_peso": 6,
            "chaves_hash_alta_cardinalidade": True,
            "versao_chaves_hash": VERSAO_CHAVES_HASH
        }

    def obter_voto_temporal(self, ultimas_cores):
        if not ultimas_cores or not getattr(self, "markov_temporal", None):
            return {"direcao": "NEUTRO", "peso": 0.0, "total": 0.0}
        regime = self._detectar_regime_temporal(ultimas_cores)
        acumulado_v = 0.0
        acumulado_p = 0.0
        peso_total = 0.0
        amostra_total = 0.0
        ordens = []
        min_amostra = {6: 8.0, 5: 10.0, 4: 14.0, 3: 20.0, 2: 28.0, 1: 35.0}
        peso_ordem = {6: 6.0, 5: 5.0, 4: 4.0, 3: 3.0, 2: 2.0, 1: 1.0}
        for ordem in range(min(6, len(ultimas_cores)), 0, -1):
            estado = tuple(ultimas_cores[-ordem:])
            stats_regime = self.markov_temporal_regime.get(ordem, {}).get((regime, estado))
            stats_global = self.markov_temporal.get(ordem, {}).get(estado)
            stats = None
            origem = None
            if stats_regime and stats_regime.get("total", 0.0) >= min_amostra[ordem]:
                stats = stats_regime
                origem = "REGIME"
            elif stats_global and stats_global.get("total", 0.0) >= min_amostra[ordem]:
                stats = stats_global
                origem = "GLOBAL"
            if not stats:
                continue
            total = float(stats.get("total", 0.0))
            denom = total + 3.0
            prob_v = (float(stats.get("V", 0.0)) + 1.0) / denom
            prob_p = (float(stats.get("P", 0.0)) + 1.0) / denom
            confianca_amostra = min(1.0, total / (min_amostra[ordem] * 4.0))
            peso = peso_ordem[ordem] * (0.5 + 0.5 * confianca_amostra)
            acumulado_v += prob_v * peso
            acumulado_p += prob_p * peso
            peso_total += peso
            amostra_total += total
            ordens.append({"ordem": ordem, "origem": origem, "massa": round(total, 2)})
        if peso_total <= 0:
            return {"direcao": "NEUTRO", "peso": 0.0, "total": 0.0, "regime": regime}
        prob_v = (acumulado_v / peso_total) * 100
        prob_p = (acumulado_p / peso_total) * 100
        margem = abs(prob_v - prob_p)
        direcao = "VERMELHO" if prob_v > prob_p else ("PRETO" if prob_p > prob_v else "NEUTRO")
        if margem >= 8.0:
            peso_voto = 2.5
        elif margem >= 5.0:
            peso_voto = 2.0
        elif margem >= 3.0:
            peso_voto = 1.0
        else:
            peso_voto = 0.0
            direcao = "NEUTRO"
        peso_voto = min(peso_voto, float(self.temporal_config.get("peso_maximo_no_juiz", 2.5)))
        return {
            "direcao": direcao,
            "peso": peso_voto,
            "total": round(amostra_total, 2),
            "regime": regime,
            "V": round(prob_v, 2),
            "P": round(prob_p, 2),
            "margem": round(margem, 2),
            "ordens_utilizadas": ordens
        }

    def _chave_conflito(self, expectations, geometria, modo_mercado, probabilidade_markov):
        peso_v = 0
        peso_p = 0
        for item in expectations or []:
            peso = 3 if any(k in item.get("tipo_regra", "") for k in ["CONTINUIDADE", "ASSUNCAO", "QUEBRADOR"]) else 1
            if item.get("direcao") == "VERMELHO":
                peso_v += peso
            elif item.get("direcao") == "PRETO":
                peso_p += peso
        direcao_pos = "V" if peso_v > peso_p else ("P" if peso_p > peso_v else "N")
        direcao_geo = "N"
        if geometria == "CICLO_FECHADO_PVVP":
            direcao_geo = "V"
        elif geometria == "CICLO_FECHADO_VPPV":
            direcao_geo = "P"
        elif geometria == "SATURAÇÃO ESTRUTURAL (V)":
            direcao_geo = "P"
        elif geometria == "SATURAÇÃO ESTRUTURAL (P)":
            direcao_geo = "V"
        mv = float((probabilidade_markov or {}).get("V", 0.0))
        mp = float((probabilidade_markov or {}).get("P", 0.0))
        direcao_markov = "V" if mv > mp else ("P" if mp > mv else "N")
        faixa_markov = int(min(20, abs(mv - mp)) // 2)
        return hash_chave(f"M={direcao_markov}:{faixa_markov}|POS={direcao_pos}|GEO={direcao_geo}|REG={modo_mercado}")

    def registrar_resultado_conflito(self, expectations, geometria, modo_mercado, probabilidade_markov, correcoes):
        if not correcoes:
            return
        chave = self._chave_conflito(expectations, geometria, modo_mercado, probabilidade_markov)
        stats = self.memoria_conflitos[chave]
        stats["total"] += 1
        c0 = correcoes[0] if len(correcoes) > 0 else None
        c1 = correcoes[1] if len(correcoes) > 1 else None
        if c0 in ("V", "B"):
            stats["V_g0"] += 1
            stats["V_g0g1"] += 1
        elif c1 in ("V", "B"):
            stats["V_g0g1"] += 1
        else:
            stats["falhas_v"] += 1
        if c0 in ("P", "B"):
            stats["P_g0"] += 1
            stats["P_g0g1"] += 1
        elif c1 in ("P", "B"):
            stats["P_g0g1"] += 1
        else:
            stats["falhas_p"] += 1

    def obter_voto_contextual(self, expectations, geometria, modo_mercado, probabilidade_markov):
        if not hasattr(self, "memoria_conflitos"):
            return {"direcao": "NEUTRO", "peso": 0.0, "total": 0}
        chave = self._chave_conflito(expectations, geometria, modo_mercado, probabilidade_markov)
        stats = self.memoria_conflitos.get(chave)
        if not stats or stats.get("total", 0) < 30:
            return {"direcao": "NEUTRO", "peso": 0.0, "total": stats.get("total", 0) if stats else 0}
        total = stats["total"]
        taxa_v = stats["V_g0g1"] / total
        taxa_p = stats["P_g0g1"] / total
        taxa_v_g0 = stats["V_g0"] / total
        taxa_p_g0 = stats["P_g0"] / total
        score_v = (taxa_v_g0 * 0.70) + (taxa_v * 0.30)
        score_p = (taxa_p_g0 * 0.70) + (taxa_p * 0.30)
        margem_score = abs(score_v - score_p)
        margem_g0 = abs(taxa_v_g0 - taxa_p_g0)
        if margem_score < 0.035 and margem_g0 < 0.04:
            return {
                "direcao": "NEUTRO", "peso": 0.0, "total": total,
                "taxa_v": taxa_v, "taxa_p": taxa_p,
                "taxa_v_g0": taxa_v_g0, "taxa_p_g0": taxa_p_g0,
                "score_v": score_v, "score_p": score_p,
                "prioridade_g0": True
            }
        direcao = "VERMELHO" if score_v > score_p else "PRETO"
        melhor_taxa = taxa_v if direcao == "VERMELHO" else taxa_p
        melhor_g0 = taxa_v_g0 if direcao == "VERMELHO" else taxa_p_g0
        if margem_score >= 0.08 and total >= 80:
            peso = 4.5
        elif margem_score >= 0.055 and total >= 50:
            peso = 3.5
        else:
            peso = 2.0
        return {
            "direcao": direcao, "peso": peso, "total": total,
            "taxa_v": taxa_v, "taxa_p": taxa_p,
            "taxa_v_g0": taxa_v_g0, "taxa_p_g0": taxa_p_g0,
            "score_v": score_v, "score_p": score_p,
            "margem_score": margem_score, "margem_g0": margem_g0,
            "melhor_taxa_g0_g1": melhor_taxa, "melhor_taxa_g0": melhor_g0,
            "prioridade_g0": True
        }

    def _validar_competencia_especialistas_cronologica(self, dados):
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
                chave_contexto = f"{especialista}|{regime}"
                self.competencia_contextual[chave_contexto]["total"] += 1
                if acertou:
                    self.competencia_contextual[chave_contexto]["acertos"] += 1
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
                        mapas_treino.get(camada, {}).get
