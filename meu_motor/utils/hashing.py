# utils/hashing.py
import hashlib
from collections import defaultdict

VERSAO_CHAVES_HASH = 1  # importado de config, mas mantido localmente para evitar ciclo

def hash_chave(chave):
    """Compacta chaves textuais longas das memórias de alta cardinalidade."""
    if isinstance(chave, int):
        return chave
    return int(hashlib.md5(str(chave).encode("utf-8")).hexdigest(), 16)

def _mesclar_mapa_hash(mapa, campos=None):
    """Migra chaves textuais legadas para hash preservando estatísticas acumuladas."""
    convertido = {}
    for chave, valor in (mapa or {}).items():
        chave_final = hash_chave(chave)
        if chave_final not in convertido:
            convertido[chave_final] = dict(valor) if isinstance(valor, dict) else valor
            continue
        if isinstance(valor, dict) and isinstance(convertido[chave_final], dict):
            for campo, numero in valor.items():
                if campos is None or campo in campos:
                    if isinstance(numero, (int, float)):
                        convertido[chave_final][campo] = convertido[chave_final].get(campo, 0) + numero
    return convertido

# Fábricas de estruturas de dados
def fabrica_padrao_detalhado():
    return {
        "total": 0, "apos_V": 0, "apos_P": 0, "apos_B": 0,
        "quebradores": defaultdict(int), "g0": 0, "g1": 0,
        "_futuros": []
    }

def fabrica_historico_regras_zerado():
    return {"acertos": 0, "total": 0}

def fabrica_historico_regras_auditado():
    return {"acertos": 1, "total": 1}
