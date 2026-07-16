import os
import json
import hashlib
from collections import defaultdict
from datetime import datetime

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

def salvar_log_json(dados, nome_arquivo="logs/sinais_tipo_b.jsonl"):
    os.makedirs("logs", exist_ok=True)
    dados["timestamp"] = datetime.now().isoformat()
    with open(nome_arquivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(dados, ensure_ascii=False) + "\n")
