import sys
import numpy as np

# Limite ampliado somente para auditorias longas
sys.setrecursionlimit(5000)

NOME_BASE_DEFINITIVA = "resultados_blaze.xlsx"
VERSAO_CHAVES_HASH = 1

# ============================================================
# INTEGRAÇÃO DE MACHINE LEARNING E DEEP LEARNING
# ============================================================
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

    class CategoricalHMMFallback:
        """
        Fallback categórico discreto em NumPy para ambientes onde o binário do
        hmmlearn é incompatível.
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
    
    CategoricalHMM = CategoricalHMMFallback
    HAS_HMM = HAS_SKLEARN

HAS_ML = HAS_SKLEARN
