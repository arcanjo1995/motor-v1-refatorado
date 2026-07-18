# ml_engine/preditor_base.py
import os
import json
import time
from collections import defaultdict

# Importamos todos os "pedaços do cérebro" (Mixins) que você criou
from .q_learning import QLearningMixin
from .markov import MarkovMixin
from .cartografia import CartografiaMixin
from .trajetorias import TrajetoriasMixin
from .especialistas import EspecialistasMixin
from .risco import RiscoMixin
from .padroes import PadroesMixin
from .deriva_temporal import DerivaTemporalMixin
from .machine_learning import MachineLearningMixin
from .evolucao import EvolucaoMixin
from .comportamento import ComportamentoMixin
from .probabilidades import ProbabilidadesMixin
from .radar_numerico import RadarNumericoMixin  # <-- RADAR

# Utilitários e Configurações necessárias
from config.settings import VERSAO_CHAVES_HASH, HAS_ML
from utils.helpers import (
    hash_chave, 
    _mesclar_mapa_hash, 
    fabrica_padrao_detalhado, 
    fabrica_historico_regras_zerado
)
from rules.analisador import AnalisadorContextoAvancado
from rules.contagens import MotorContagensProjetivas
from utils.math_engine import EngineMatematicoAvancado

class IAPreditivaV1(
    QLearningMixin, 
    MarkovMixin, 
    CartografiaMixin, 
    TrajetoriasMixin, 
    EspecialistasMixin, 
    RiscoMixin, 
    PadroesMixin, 
    DerivaTemporalMixin, 
    MachineLearningMixin,
    EvolucaoMixin, 
    ComportamentoMixin, 
    ProbabilidadesMixin,
    RadarNumericoMixin   # <-- RADAR
):
    """
    O Cérebro Central do Motor V1. 
    A herança múltipla (Mixins) cola todas as funções invisivelmente.
    As variáveis e construtores ficam aqui para não corromper o Pickle.
    """
    
    def __init__(self, dados_longo_prazo, dados_recencia=None):
        self.dados_longo = dados_longo_prazo
        self.dados_recencia = dados_recencia if dados_recencia else []
        self.modelo_transicao = defaultdict(list)
        self.modelo_transicao_profundo = defaultdict(list)
        self.modelo_numerico = defaultdict(list)
        # MAIN 97 — transição NUMERO -> próximo número para simulação causal de rotas.
        # É memória adicional; não substitui bigrama, trigrama, Markov, regras ou recência.
        self.transicoes_numericas = defaultdict(lambda: {"total": 0.0, "proximos": defaultdict(float)})
        self.bigramas_numericos = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0, "prox_numero": defaultdict(int)})
        self.saturacao_ciclica = defaultdict(lambda: {"ciclos_V": [], "ciclos_P": [], "historico_distancias": []})
        self.dna_padroes = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        
        self.padroes_fechamento_numerico = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        
        # Mapeadores Globais (Até G1)
        self.estatisticas_projecoes_globais = {n: {"total": 0, "g0": 0, "g1": 0, "falha": 0} for n in range(1, 8)}
        # Memória de respeito das contagens projetivas.
        # A projeção V3 continua sendo VERMELHA; não existe leitura bilateral V/P.
        self.estatisticas_projecoes_bilaterais = {}  # legado desativado para retrocompatibilidade
        self.estatisticas_projecoes_respeito = {n: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0} for n in range(1, 8)}
        self.projecoes_respeito_contextual = defaultdict(lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0})
        self.projecoes_respeito_metricas = {}

        # MAIN 85 — cartografia completa do XLS.
        # É uma camada ADITIVA: não remove nem altera regras, pesos, RECÊNCIA,
        # NO CALL, Markov, ML ou direção original das projeções V3.
        self.cartografia_projecoes_trajetoria = defaultdict(
            lambda: {"total": 0, "respeitada_g0": 0, "respeitada_g1": 0, "nao_respeitada": 0}
        )
        self.cartografia_padroes_xls = defaultdict(
            lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0, "B_g0": 0}
        )
        # MAIN 98 — cartografia contextual INTERNA dos padrões.
        # Mantém a cartografia existente intacta e adiciona uma memória separada
        # para aprender quando o mesmo padrão muda de comportamento por número
        # final, bigrama, trigrama, regime, Markov, geometria, regras e contagens.
        self.cartografia_padroes_contextual = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        self.cartografia_padroes_contextual_metricas = {}
        self.ultima_leitura_padrao_contextual = {}

        # MAIN 127 — trajetória causal de STREAK, bilateral V/P.
        # Não cria regra fixa de continuidade ou reversão: aprende nascimento,
        # confirmação, expansão e retomada após respiro para as DUAS cores.
        self.cartografia_trajetoria_streak = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        self.cartografia_trajetoria_streak_metricas = {}
        self._ultimo_voto_trajetoria_streak = {}

        # MAIN 129 — cartografia de MORFOLOGIA E CONTINUIDADE ESTRUTURAL.
        # Camada aditiva: aprende a forma dos blocos, trajetória de tamanhos,
        # repetição/inversão/espelho e relação curta-média-longa. Não transforma
        # nomes didáticos (321, zig-zag, reflexo etc.) em regras fixas de CALL.
        self.cartografia_morfologia_estrutural = defaultdict(
            lambda: {
                "total": 0, "V_g0": 0, "V_g1": 0, "V_falha": 0,
                "P_g0": 0, "P_g1": 0, "P_falha": 0, "B_g0": 0
            }
        )
        self.cartografia_morfologia_estrutural_metricas = {}
        self._ultimo_voto_morfologia_estrutural = {}

        # MAIN 100 — cartografia contextual INTERNA das regras e contagens.
        # O detector oficial permanece intacto. Esta memória apenas percorre a
        # cronologia índice por índice e aprende o desfecho de cada ativação,
        # evolução e fechamento estrutural nos subcontextos em que ocorreram.
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

        # MAIN 89 — memória estatística das regras oficiais estruturais.
        # Mesma filosofia de bigramas/trigramas: cada regra ativa é auditada
        # cronologicamente contra G0/G1 e participa da geração do sinal.
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
        # MAIN 123 — memória diagnóstica da autorização operacional.
        # Não vota e não muda direção. Compara, após o desfecho real, se a
        # política oficial, a direção oposta ou o NO CALL teria protegido G0/G1.
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
        # MAIN 99 — métricas exclusivas do Q-Learning contextual.
        # Não altera arbitragem, regras, Markov, geometria, ML ou recência.
        self.q_learning_contextual_metricas = {}

        # Camada evolutiva retrocompatível: aprende contexto sem remover
        # nenhum mapeador, regra, peso ou memória já existente.
        self.markov_ordens = {ordem: defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0}) for ordem in range(1, 7)}
        self.memoria_conflitos = defaultdict(lambda: {
            "total": 0, "V_g0g1": 0, "P_g0g1": 0,
            "V_g0": 0, "P_g0": 0, "falhas_v": 0, "falhas_p": 0
        })

        # Memória temporal adaptativa exclusiva da BASE LONGA.
        # Não substitui a recência oficial e não altera o peso 6.
        self.markov_temporal = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        self.markov_temporal_regime = {ordem: defaultdict(lambda: {"V": 0.0, "P": 0.0, "B": 0.0, "total": 0.0}) for ordem in range(1, 7)}
        self.temporal_config = {"versao": 1, "piso_memoria_historica": 0.12, "meia_vida_minima": 20000, "peso_maximo_no_juiz": 2.5}
        self.temporal_metricas = {}

        # MAIN 114 — estado retrocompatível das nove correções estruturais.
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

        # MAIN 74 — competência formal das camadas analíticas que antes eram
        # apenas memória/apoio. Não cria NO CALL e não altera regras imutáveis.
        self.camadas_ampliadas_mapas = {}
        self.camadas_ampliadas_competencia = {}
        self.camadas_ampliadas_contextual = {}
        self.camadas_ampliadas_metricas = {}
        self.ultima_confluencia_camadas_ampliadas = {}
        self.regras_competencia_cronologica = {}
        self.regras_competencia_metricas = {}
        self.memoria_conflitos_metricas = {}

        # Filtro discriminativo G0/G1 x G2+.
        # Usa somente a competência contextual já validada cronologicamente e
        # o respeito/não respeito das contagens projetivas já aprendido.
        # Não cria direção, não altera voto, não muda a RECÊNCIA oficial e
        # somente pode converter um sinal preliminar arriscado em NO CALL.
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

        # Especialista adicional de proteção operacional G0/G1.
        # Aprende somente o risco de uma direção NÃO resolver em G0/G1.
        # Não altera regras, pesos, Markov, RECÊNCIA oficial ou direção dos sinais.
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

        # <-- RADAR: inicialização da memória do Radar
        self._inicializar_memoria_radar()

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

        # <-- RADAR: serialização da memória e configurações do Radar
        if 'memoria_radar' in state:
            state['memoria_radar'] = dict(state['memoria_radar'])
        if 'pesos_radar' in state:
            state['pesos_radar'] = dict(state['pesos_radar'])
        if 'config_radar' in state:
            state['config_radar'] = dict(state['config_radar'])

        # Persistir os modelos ML é intencional: versões anteriores removiam
        # ml_gb/ml_mlp/ml_hmm do pickle e o app perdia a camada neural no reboot.
        return state


    def _normalizar_unidade_analise_compatibilidade(self):
        """
        Garante o contrato estrutural de unidade_analise ao reutilizar um modelo
        persistido no treinamento incremental. Preserva os valores aprendidos e
        apenas recompõe chaves/campos ausentes ou chaves numéricas serializadas
        como texto.
        """
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
        for k, v in saturacao_loaded.items(): self.saturacao_ciclica[k] = {"ciclos_V": v.get("ciclos_V", []), "ciclos_P": v.get("ciclos_P", []), "historico_distancias": v.get("historico_distancias", [])}
            
        dna_loaded = state.get('dna_padroes', {})
        self.dna_padroes = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        for k, v in dna_loaded.items(): self.dna_padroes[k] = {"V": v.get("V",0), "P": v.get("P",0), "B": v.get("B",0), "total": v.get("total",0)}
            
        pfn_loaded = state.get('padroes_fechamento_numerico', {})
        self.padroes_fechamento_numerico = defaultdict(lambda: {"V": 0, "P": 0, "B": 0, "total": 0})
        for k, v in pfn_loaded.items(): self.padroes_fechamento_numerico[k] = {"V": v.get("V",0), "P": v.get("P",0), "B": v.get("B",0), "total": v.get("total",0)}
            
        epg_loaded = state.get('estatisticas_projecoes_globais', {})
        self.estatisticas_projecoes_globais = {n: {"total": 0, "g0": 0, "g1": 0, "falha": 0} for n in range(1, 8)}
        for k, v in epg_loaded.items(): self.estatisticas_projecoes_globais[int(k)] = {"total": v.get("total",0), "g0": v.get("g0",0), "g1": v.get("g1",0), "falha": v.get("falha",0)}
            
        # A memória bilateral antiga é deliberadamente desativada.
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
        self.cartografia_padroes_contextual_metricas = state.get(
            'cartografia_padroes_contextual_metricas', {}
        )
        self.ultima_leitura_padrao_contextual = state.get(
            'ultima_leitura_padrao_contextual', {}
        )

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
        self.cartografia_morfologia_estrutural_metricas = state.get(
            'cartografia_morfologia_estrutural_metricas', {}
        )
        self._ultimo_voto_morfologia_estrutural = state.get(
            '_ultimo_voto_morfologia_estrutural', {}
        )

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
        self.cartografia_regras_contextual_metricas = state.get(
            'cartografia_regras_contextual_metricas', {}
        )
        self.matriz_deriva_comportamental = state.get(
            'matriz_deriva_comportamental', {}
        )
        self.ultima_leitura_regra_contextual = state.get(
            'ultima_leitura_regra_contextual', {}
        )
        self.cartografia_xls_metricas = state.get('cartografia_xls_metricas', {})

        eei_loaded = state.get('especialista_espelho_inversao', {})
        self.especialista_espelho_inversao = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in eei_loaded.items():
            self.especialista_espelho_inversao[k] = {"total": v.get("total", 0), "V_g0": v.get("V_g0", 0), "V_g1": v.get("V_g1", 0), "P_g0": v.get("P_g0", 0), "P_g1": v.get("P_g1", 0)}

        ebg_loaded = state.get('estatisticas_bigramas_globais', {})
        self.estatisticas_bigramas_globais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in ebg_loaded.items(): self.estatisticas_bigramas_globais[k] = {"total": v.get("total",0), "V_g0": v.get("V_g0",0), "V_g1": v.get("V_g1",0), "P_g0": v.get("P_g0",0), "P_g1": v.get("P_g1",0)}

        etg_loaded = state.get('estatisticas_trigramas_globais', {})
        self.estatisticas_trigramas_globais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in etg_loaded.items(): self.estatisticas_trigramas_globais[k] = {"total": v.get("total",0), "V_g0": v.get("V_g0",0), "V_g1": v.get("V_g1",0), "P_g0": v.get("P_g0",0), "P_g1": v.get("P_g1",0)}

        ero_loaded = state.get('estatisticas_regras_oficiais', {})
        self.estatisticas_regras_oficiais = defaultdict(lambda: {"total": 0, "V_g0": 0, "V_g1": 0, "P_g0": 0, "P_g1": 0})
        for k, v in ero_loaded.items():
            self.estatisticas_regras_oficiais[k] = {"total": v.get("total", 0), "V_g0": v.get("V_g0", 0), "V_g1": v.get("V_g1", 0), "P_g0": v.get("P_g0", 0), "P_g1": v.get("P_g1", 0)}
        self.regras_oficiais_metricas = state.get('regras_oficiais_metricas', {})
            
        self.probabilidades_globais = state.get('probabilidades_globais', {"streak_v_5": 0.0, "streak_p_5": 0.0, "xadrez_5": 0.0})
        self.controladores_fortes = defaultdict(int, state.get('controladores_fortes', {}))
        self.historico_regras = defaultdict(fabrica_historico_regras_zerado)
        for k, v in state.get('historico_regras', {}).items(): self.historico_regras[k] = {"acertos": v.get("acertos", 0), "total": v.get("total", 0)}
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

        # MAIN 96 — veto por instabilidade da decisão final.
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

        # <-- RADAR: restaura memória e configurações do Radar
        if 'memoria_radar' not in self.__dict__:
            self._inicializar_memoria_radar()
        else:
            self.memoria_radar = defaultdict(lambda: {
                "total": 0,
                "acertos_g0": 0,
                "acertos_g1": 0,
                "erros": 0,
                "historico_numeros": defaultdict(int),
                "acertos_g0_por_numero": defaultdict(int),
                "acertos_g1_por_numero": defaultdict(int),
                "erros_por_numero": defaultdict(int),
                "fonte_base_acertos": 0,
                "fonte_recencia_acertos": 0,
                "fonte_ao_vivo_acertos": 0,
            }, self.memoria_radar)
        if not hasattr(self, 'pesos_radar'):
            self.pesos_radar = {"base": 0.40, "recencia": 0.35, "ao_vivo": 0.25}
        if not hasattr(self, 'config_radar'):
            self.config_radar = {
                "versao": 1,
                "aprendizado_continuo": True,
                "pesos_adaptativos": False,
                "minimo_amostras_para_peso": 30,
                "limiar_ameaca_critica": 3.0,
                "limiar_ameaca_alta": 2.0,
                "limiar_ameaca_media": 1.5,
                "influencia_maxima": 0.25,
                "influencia_minima": 0.05,
            }

        return state

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

            # <-- RADAR: treinar Radar sobre toda a base (aprendizado contínuo)
            if len(todos_dados) >= 13:
                nums = [int(d.get("numero")) for d in todos_dados]
                pol = [str(d.get("cor", "B")).upper() for d in todos_dados]
                for i in range(11, len(todos_dados) - 2):
                    sub_num = nums[i-11:i+1]
                    sub_pol = pol[i-11:i+1]
                    g0 = int(todos_dados[i+1].get("numero", -1))
                    g1 = int(todos_dados[i+2].get("numero", -1)) if i+2 < len(todos_dados) else None
                    if g0 < 0:
                        continue
                    try:
                        # Usa MotorAnalise para obter contexto (se disponível)
                        from core.motor_analise import MotorAnalise
                        analise = MotorAnalise.analisar_janela(sub_num, sub_pol, self, eh_sinal_real=False)
                        self._treinar_radar_em_janela(sub_num, sub_pol, g0, g1, analise)
                    except Exception as e:
                        # Fallback: treina sem contexto
                        self._treinar_radar_em_janela(sub_num, sub_pol, g0, g1, None)

    def construir_cadeia_causal_consequencia(self, sub_num, sub_pol, expectativas=None):
        """
        Materializa origem -> sustentador -> assunção -> controlador ->
        consequência futura. Não cria uma hierarquia paralela: a consequência
        ocupa exclusivamente o nível 7 da hierarquia oficial.
        """
        expectativas = list(expectativas or [])
        ordenadas = sorted(
            [e for e in expectativas if e.get("direcao") in ("VERMELHO", "PRETO")],
            key=lambda e: (
                self._nivel_hierarquico_regra(e)[0],
                -self._autoridade_evolutiva_regra(e.get("tipo_regra", ""))
            )
        )
        origem = ordenadas[0] if ordenadas else None
        sustentadores = [
            e for e in ordenadas[1:]
            if origem and e.get("direcao") == origem.get("direcao")
        ]
        assuncao = next((e for e in ordenadas if "ASSUNCAO" in str(e.get("tipo_regra", "")).upper()), None)

        controlador = origem
        if origem:
            mesma_direcao = [origem] + sustentadores
            controlador = max(
                mesma_direcao,
                key=lambda e: (
                    -self._nivel_hierarquico_regra(e)[0],
                    self._autoridade_evolutiva_regra(e.get("tipo_regra", ""))
                )
            )

        direcao = controlador.get("direcao") if controlador else "NEUTRO"
        autoridade = self._autoridade_evolutiva_regra(controlador.get("tipo_regra", "")) if controlador else 0.0
        estado = (getattr(self, "matriz_evolutiva", {}) or {}).get("regras", {}).get(
            controlador.get("tipo_regra", "") if controlador else "", {}
        ).get("estado_evolutivo", "SEM_MEDICAO")

        viva = bool(
            controlador
            and direcao in ("VERMELHO", "PRETO")
            and not str(estado).startswith("DEGRADACAO_CRITICA")
        )
        consequencia = {
            "origem": origem.get("tipo_regra") if origem else None,
            "sustentador": [e.get("tipo_regra") for e in sustentadores[:3]],
            "assuncao": assuncao.get("tipo_regra") if assuncao else None,
            "controlador": controlador.get("tipo_regra") if controlador else None,
            "direcao": direcao,
            "autoridade_atual": round(float(autoridade), 4),
            "estado_evolutivo": estado,
            "status": "VIVA" if viva else "ENFRAQUECIDA_OU_AUSENTE",
            "nivel_hierarquico": 7,
            "tipo": "CONSEQUENCIA_FUTURA"
        }
        self.ultima_cadeia_causal = {
            "origem": consequencia["origem"],
            "sustentador": consequencia["sustentador"],
            "assuncao": consequencia["assuncao"],
            "controlador": consequencia["controlador"]
        }
        self.ultima_consequencia_futura = consequencia
        return consequencia

    def predizer_proxima_casa(self, sub_num, sub_pol, analise_contexto=None):
        if len(sub_num) < 12:
            return "NEUTRO", 0.0, "Janela insuficiente"
        
        ultimo_num = sub_num[-1]
        penultimo_num = sub_num[-2]
        
        ultimas_cores_2 = (sub_pol[-2], sub_pol[-1])
        ultimas_cores_4 = tuple(sub_pol[-4:]) if len(sub_pol) >= 4 else None
        
        trans_2 = self.modelo_transicao.get(ultimas_cores_2, [])
        trans_4 = self.modelo_transicao_profundo.get(ultimas_cores_4, []) if hasattr(self, 'modelo_transicao_profundo') else []
        
        por_num = self.modelo_numerico.get(ultimo_num, [])
        stats = self.unidade_analise.get(ultimo_num, {"freq_v": 0, "freq_p": 0})
        
        v_bonus = stats.get("freq_v", 0) * 3.5
        p_bonus = stats.get("freq_p", 0) * 3.5

        if hasattr(self, 'estatisticas_trigramas_globais') and len(sub_num) >= 3:
            trigrama_atual = f"{sub_num[-3]}-{sub_num[-2]}-{sub_num[-1]}"
            stats_tri = self.estatisticas_trigramas_globais.get(trigrama_atual)
            if stats_tri and stats_tri["total"] >= 5:
                taxa_v_tri = ((stats_tri["V_g0"] + stats_tri["V_g1"]) / stats_tri["total"]) * 100
                taxa_p_tri = ((stats_tri["P_g0"] + stats_tri["P_g1"]) / stats_tri["total"]) * 100
                if taxa_v_tri >= 60.0: v_bonus += 25
                elif taxa_v_tri < 45.0: v_bonus -= 15
                if taxa_p_tri >= 60.0: p_bonus += 25
                elif taxa_p_tri < 45.0: p_bonus -= 15

        if hasattr(self, 'estatisticas_bigramas_globais') and len(sub_num) >= 2:
            bigrama_atual = f"{sub_num[-2]}-{sub_num[-1]}"
            stats_bi = self.estatisticas_bigramas_globais.get(bigrama_atual)
            if stats_bi and stats_bi["total"] >= 5:
                taxa_v_bi = ((stats_bi["V_g0"] + stats_bi["V_g1"]) / stats_bi["total"]) * 100
                taxa_p_bi = ((stats_bi["P_g0"] + stats_bi["P_g1"]) / stats_bi["total"]) * 100
                if taxa_v_bi >= 55.0: v_bonus += 18
                elif taxa_v_bi < 45.0: v_bonus -= 10
                if taxa_p_bi >= 55.0: p_bonus += 18
                elif taxa_p_bi < 45.0: p_bonus -= 10
        
        # Consolidação causal das CONTAGENS antes do v_bonus/p_bonus.
        # V3, coexistência, finalização conjunta e hierarquia continuam detectadas
        # normalmente, mas deixam de votar repetidamente como provas independentes.
        tipos_contagens_consolidadas = {
            "COEXISTENCIA_CONTAGENS_ATIVA",
            "FINALIZACAO_CONJUNTA_ATIVA",
        }

        def _eh_contagem_consolidada(regra):
            tipo = str(regra.get("tipo_regra", ""))
            return (
                tipo.startswith("V3_ATIVADOR_")
                or tipo.startswith("HIERARQUIA_CONTAGEM_")
                or tipo in tipos_contagens_consolidadas
            )

        if analise_contexto:
            regras_posicionais = analise_contexto.get("regras_posicionais", [])

            # Regras não pertencentes à raiz causal consolidada mantêm exatamente
            # o cálculo individual anterior.
            for regra in regras_posicionais:
                if _eh_contagem_consolidada(regra):
                    continue

                direcao_regra = regra.get("direcao")
                peso_manual = str(regra.get("peso", "MEDIO")).upper()
                bonus_base = {"BAIXO": 4.0, "MEDIO": 10.0, "MÉDIO": 10.0, "ALTO": 18.0}.get(peso_manual, 8.0)
                if direcao_regra == "VERMELHO":
                    v_bonus += bonus_base
                elif direcao_regra == "PRETO":
                    p_bonus += bonus_base

                stats_regra = getattr(self, "estatisticas_regras_oficiais", {}).get(regra.get("tipo_regra", ""))
                if stats_regra and stats_regra.get("total", 0) >= 5:
                    total_regra = float(stats_regra["total"])
                    if direcao_regra == "VERMELHO":
                        taxa = (stats_regra.get("V_g0", 0) + stats_regra.get("V_g1", 0)) / total_regra
                        if taxa >= 0.65: v_bonus += 16
                        elif taxa >= 0.58: v_bonus += 8
                        elif taxa < 0.45: v_bonus -= 12
                    elif direcao_regra == "PRETO":
                        taxa = (stats_regra.get("P_g0", 0) + stats_regra.get("P_g1", 0)) / total_regra
                        if taxa >= 0.65: p_bonus += 16
                        elif taxa >= 0.58: p_bonus += 8
                        elif taxa < 0.45: p_bonus -= 12

            # A raiz CONTAGENS gera somente um voto, calibrado pela cartografia
            # contextual histórica das próprias estruturas ativas.
            voto_contagens = self.obter_voto_contagens_consolidado(
                sub_num, sub_pol, regras_posicionais
            )
            self.ultimo_voto_contagens_consolidado = voto_contagens
            if voto_contagens.get("direcao") == "VERMELHO":
                v_bonus += float(voto_contagens.get("peso", 0.0))
            elif voto_contagens.get("direcao") == "PRETO":
                p_bonus += float(voto_contagens.get("peso", 0.0))

        voto_cartografia = self.obter_voto_cartografia_xls(sub_num, sub_pol)
        if voto_cartografia.get("direcao") == "VERMELHO":
            v_bonus += float(voto_cartografia.get("peso", 0.0))
        elif voto_cartografia.get("direcao") == "PRETO":
            p_bonus += float(voto_cartografia.get("peso", 0.0))

        # MAIN 97 — simula possíveis números em G0 e recalcula a rota estrutural
        # após cada candidato. Peso aditivo e limitado; não altera regras existentes.
        voto_rotas = self.simular_rotas_proximos_resultados(sub_num, sub_pol)
        self.ultima_simulacao_rotas = voto_rotas
        if voto_rotas.get("direcao") == "VERMELHO":
            v_bonus += float(voto_rotas.get("peso", 0.0)) * 6.0
        elif voto_rotas.get("direcao") == "PRETO":
            p_bonus += float(voto_rotas.get("peso", 0.0)) * 6.0

        voto_espelho = self.obter_voto_espelho_inversao(sub_num, sub_pol)
        if voto_espelho.get("direcao") == "VERMELHO": v_bonus += float(voto_espelho.get("peso", 0.0))
        elif voto_espelho.get("direcao") == "PRETO": p_bonus += float(voto_espelho.get("peso", 0.0))

        voto_competencia = self.obter_voto_competencia_especialistas(sub_num, sub_pol, analise_contexto)
        if voto_competencia.get("direcao") == "VERMELHO":
            v_bonus += float(voto_competencia.get("peso", 0.0))
        elif voto_competencia.get("direcao") == "PRETO":
            p_bonus += float(voto_competencia.get("peso", 0.0))

        # MAIN 74 — as camadas antes apenas analíticas agora provam competência
        # cronológica e participam de forma aditiva. Não criam veto nem NO CALL.
        voto_camadas = self.obter_voto_camadas_ampliadas(sub_num, sub_pol)
        if voto_camadas.get("direcao") == "VERMELHO":
            v_bonus += float(voto_camadas.get("peso", 0.0))
        elif voto_camadas.get("direcao") == "PRETO":
            p_bonus += float(voto_camadas.get("peso", 0.0))

        dna_3 = "-".join(map(str, sub_num[-3:]))
        if hasattr(self, 'dna_padroes') and dna_3 in self.dna_padroes:
            dna_stats = self.dna_padroes[dna_3]
            if dna_stats["total"] >= 5: 
                prob_dna_v = (dna_stats["V"] / dna_stats["total"]) * 100
                prob_dna_p = (dna_stats["P"] / dna_stats["total"]) * 100
                z_score_v = self._calcular_z_score(dna_stats["V"], dna_stats["total"])
                z_score_p = self._calcular_z_score(dna_stats["P"], dna_stats["total"])
                if prob_dna_v >= 60.0 and z_score_v > 1.64: v_bonus += 25
                elif prob_dna_p >= 60.0 and z_score_p > 1.64: p_bonus += 25

        if penultimo_num != ultimo_num:
            chave_dupla = f"{penultimo_num}-{ultimo_num}"
            stats_dupla = self.bigramas_numericos.get(chave_dupla)
            if stats_dupla and stats_dupla["total"] >= 5:
                prob_v_dupla = (stats_dupla["V"] / stats_dupla["total"]) * 100
                prob_p_dupla = (stats_dupla["P"] / stats_dupla["total"]) * 100
                z_score_v = self._calcular_z_score(stats_dupla["V"], stats_dupla["total"])
                z_score_p = self._calcular_z_score(stats_dupla["P"], stats_dupla["total"])
                if prob_v_dupla > 55.0 and z_score_v > 1.64: v_bonus += 18
                elif prob_p_dupla > 55.0 and z_score_p > 1.64: p_bonus += 18

        if hasattr(self, 'padroes_fechamento_numerico'):
            aplicou_destrinchador = False
            for tam in [5, 4, 3]:
                if len(sub_pol) >= tam and not aplicou_destrinchador:
                    padrao_atual = "".join(sub_pol[-tam:])
                    if 'B' not in padrao_atual:
                        chave_num = f"PADRAO_{padrao_atual}_{ultimo_num}"
                        chave_bigrama = f"PADRAO_{padrao_atual}_{penultimo_num}-{ultimo_num}"
                        stats_bigrama = self.padroes_fechamento_numerico.get(chave_bigrama)
                        if stats_bigrama and stats_bigrama["total"] >= 3:
                            prob_v = (stats_bigrama["V"] / stats_bigrama["total"]) * 100
                            prob_p = (stats_bigrama["P"] / stats_bigrama["total"]) * 100
                            z_score_v = self._calcular_z_score(stats_bigrama["V"], stats_bigrama["total"])
                            z_score_p = self._calcular_z_score(stats_bigrama["P"], stats_bigrama["total"])
                            if prob_v >= 55.0 and z_score_v > 1.28:
                                v_bonus += 45
                                aplicou_destrinchador = True
                            elif prob_p >= 55.0 and z_score_p > 1.28:
                                p_bonus += 45
                                aplicou_destrinchador = True
                        if not aplicou_destrinchador:
                            stats_num = self.padroes_fechamento_numerico.get(chave_num)
                            if stats_num and stats_num["total"] >= 3:
                                prob_v = (stats_num["V"] / stats_num["total"]) * 100
                                prob_p = (stats_num["P"] / stats_num["total"]) * 100
                                z_score_v = self._calcular_z_score(stats_num["V"], stats_num["total"])
                                z_score_p = self._calcular_z_score(stats_num["P"], stats_num["total"])
                                if prob_v >= 55.0 and z_score_v > 1.28:
                                    v_bonus += 35
                                    aplicou_destrinchador = True
                                elif prob_p >= 55.0 and z_score_p > 1.28:
                                    p_bonus += 35
                                    aplicou_destrinchador = True

        comportamento = stats.get("comportamento_dominante", "NEUTRO")
        estabilidade = stats.get("estabilidade", "NEUTRO")
        enfraquecimento = stats.get("enfraquecimento", "ESTÁVEL")
        if comportamento == "VERMELHO": v_bonus += 12
        elif comportamento == "PRETO": p_bonus += 12
        if estabilidade == "ESTÁVEL":
            if comportamento == "VERMELHO": v_bonus += 10
            elif comportamento == "PRETO": p_bonus += 10
        elif estabilidade == "INSTÁVEL":
            v_bonus -= 8
            p_bonus -= 8
            
        # MAIN 110 — calibração temporal LONGO PRAZO x RECÊNCIA por cenário.
        # Números, bigramas, trigramas, padrões, streaks, geometria, Markov,
        # regime e regras ativas são comparados individualmente; a deriva recente
        # entra como UMA força consolidada para não duplicar evidências.
        voto_deriva_temporal = self.obter_ajuste_deriva_temporal(
            sub_num, sub_pol, analise_contexto
        )
        if voto_deriva_temporal.get("direcao") == "VERMELHO":
            v_bonus += float(voto_deriva_temporal.get("peso", 0.0))
        elif voto_deriva_temporal.get("direcao") == "PRETO":
            p_bonus += float(voto_deriva_temporal.get("peso", 0.0))

        has_rec = len(self.dados_recencia) > 0
        p_trans_4 = 0.30 if has_rec else 0.25
        p_trans_2 = 0.12 if has_rec else 0.10
        p_num = 0.15 if has_rec else 0.14
        
        v_deep = (trans_4.count('V') * p_trans_4) if trans_4 else 0
        p_deep = (trans_4.count('P') * p_trans_4) if trans_4 else 0
        
        total_v = v_deep + (trans_2.count('V') * p_trans_2) + (por_num.count('V') * p_num) + v_bonus
        total_p = p_deep + (trans_2.count('P') * p_trans_2) + (por_num.count('P') * p_num) + p_bonus
        
        prob_v_heuristica = (total_v / (total_v + total_p)) * 100 if (total_v + total_p) > 0 else 0
        prob_p_heuristica = (total_p / (total_v + total_p)) * 100 if (total_v + total_p) > 0 else 0

        if getattr(self, 'ml_ready', False) and HAS_ML:
            try:
                entropia = EngineMatematicoAvancado.calcular_entropia_shannon(sub_pol)
                prob_markov = self.calcular_probabilidade_exata_markov(sub_pol)
                freq_v = sub_pol.count('V') / 12.0
                
                # MAIN 95 — inferência usa exatamente o mesmo vetor contextual de 32 features do treino.
                geometria_ml = (analise_contexto or {}).get("geometria") or AnalisadorContextoAvancado.mapear_padroes_geometria(sub_pol)
                expectativas_ml = (analise_contexto or {}).get("regras_posicionais")
                if expectativas_ml is None:
                    expectativas_ml = MotorContagensProjetivas.mapear_janela(sub_num, sub_pol, geometria_ml, None)
                modo_ml = ((analise_contexto or {}).get("contexto_avancado") or {}).get("modo_mercado")
                features_32 = self._construir_features_ml_contextuais(
                    sub_num, sub_pol, prob_markov, geometria_ml, expectativas_ml, modo_ml
                )
                features = [features_32]

                previsoes_ml = []
                if self.ml_gb is not None:
                    features_gb = features
                    if int(getattr(self.ml_gb, "n_features_in_", 32)) == 9:
                        features_gb = [[features_32[i] for i in (0, 1, 2, 6, 7, 9, 16, 22, 29)]]
                    prob_gb = self.ml_gb.predict_proba(features_gb)[0]
                    mapa_gb = {int(c): float(p) for c, p in zip(self.ml_gb.classes_, prob_gb)}
                    previsoes_ml.append(("gb", mapa_gb.get(1, 0.0), mapa_gb.get(0, 0.0)))
                if self.ml_mlp is not None:
                    features_mlp = features
                    if int(getattr(self.ml_mlp, "n_features_in_", 32)) == 9:
                        features_mlp = [[features_32[i] for i in (0, 1, 2, 6, 7, 9, 16, 22, 29)]]
                    prob_mlp = self.ml_mlp.predict_proba(features_mlp)[0]
                    mapa_mlp = {int(c): float(p) for c, p in zip(self.ml_mlp.classes_, prob_mlp)}
                    previsoes_ml.append(("mlp", mapa_mlp.get(1, 0.0), mapa_mlp.get(0, 0.0)))

                soma_pesos = sum(float(getattr(self, "ml_pesos", {}).get(nome, 0.0)) for nome, _, _ in previsoes_ml)
                if soma_pesos <= 0 and previsoes_ml:
                    pesos_uso = {nome: 1.0 / len(previsoes_ml) for nome, _, _ in previsoes_ml}
                else:
                    pesos_uso = {nome: float(getattr(self, "ml_pesos", {}).get(nome, 0.0)) / soma_pesos for nome, _, _ in previsoes_ml}
                ml_v_val = sum(prob_v * pesos_uso.get(nome, 0.0) for nome, prob_v, _ in previsoes_ml) * 100
                ml_p_val = sum(prob_p * pesos_uso.get(nome, 0.0) for nome, _, prob_p in previsoes_ml) * 100

                # Meta-confluência: ML deixa de apagar as memórias contextuais.
                accs_validas = [
                    float(v) for k, v in getattr(self, "ml_metricas", {}).items()
                    if k in ("acuracia_gb", "acuracia_mlp") and isinstance(v, (int, float))
                ]
                acc_media = (sum(accs_validas) / len(accs_validas)) if accs_validas else 50.0
                peso_ml_meta = min(0.75, max(0.25, 0.25 + max(0.0, acc_media - 50.0) / 50.0))
                peso_memoria_meta = 1.0 - peso_ml_meta
                meta_v = (ml_v_val * peso_ml_meta) + (prob_v_heuristica * peso_memoria_meta)
                meta_p = (ml_p_val * peso_ml_meta) + (prob_p_heuristica * peso_memoria_meta)
                self.ultima_meta_confluencia = {
                    "ml_v": round(ml_v_val, 2), "ml_p": round(ml_p_val, 2),
                    "memoria_v": round(prob_v_heuristica, 2), "memoria_p": round(prob_p_heuristica, 2),
                    "peso_ml": round(peso_ml_meta, 4), "peso_memorias": round(peso_memoria_meta, 4),
                    "meta_v": round(meta_v, 2), "meta_p": round(meta_p, 2)
                }

                BARREIRA = 52.5
                if meta_v >= BARREIRA and meta_v > meta_p:
                    return "VERMELHO", round(meta_v, 1), f"Meta-Confluência ML + Memórias: V ({meta_v:.1f}%)"
                elif meta_p >= BARREIRA and meta_p > meta_v:
                    return "PRETO", round(meta_p, 1), f"Meta-Confluência ML + Memórias: P ({meta_p:.1f}%)"
                else:
                    return "NEUTRO", round(max(meta_v, meta_p), 1), "Meta-Confluência sem direção dominante"
            except Exception as e:
                pass

        BARREIRA = 52.5
        if prob_v_heuristica >= BARREIRA and prob_v_heuristica > prob_p_heuristica + 4:
            return "VERMELHO", round(prob_v_heuristica, 1), f"Confluência de Heurística e Z-Score ({prob_v_heuristica:.1f}%)"
        elif prob_p_heuristica >= BARREIRA and prob_p_heuristica > prob_v_heuristica + 4:
            return "PRETO", round(prob_p_heuristica, 1), f"Confluência de Heurística e Z-Score ({prob_p_heuristica:.1f}%)"
        return "NEUTRO", round(max(prob_v_heuristica, prob_p_heuristica), 1), "Sem confluência clara"
