# data/persistence.py
import os
import time
import pickle
import tempfile

def salvar_modelo_longo_prazo(ia, caminho="modelo_longo_prazo.pkl"):
    try:
        pasta = os.path.dirname(caminho)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        dir_alvo = pasta if pasta else "."
        for tentativa in range(5):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir=dir_alvo) as tmp:
                    tmp_path = tmp.name
                    pickle.dump(ia, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                if os.path.exists(caminho):
                    os.remove(caminho)
                os.replace(tmp_path, caminho)
                return True
            except Exception as e:
                print(f"Tentativa {tentativa + 1} de salvar o modelo falhou: {e}")
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                time.sleep(0.7)
        return False
    except Exception as e:
        print(f"Erro crítico ao salvar modelo: {e}")
        return False

def carregar_modelo_longo_prazo(caminho="modelo_longo_prazo.pkl"):
    if os.path.exists(caminho):
        try:
            with open(caminho, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return None
    return None
