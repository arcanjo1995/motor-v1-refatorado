import os
import pandas as pd

class LeitorXLS:
    def __init__(self, caminho_arquivo):
        self.caminho = caminho_arquivo

    def ler_e_validar(self):
        if not os.path.exists(self.caminho): 
            return None
        try:
            df = pd.read_excel(self.caminho)
            df.columns = [str(col).strip().lower() for col in df.columns]
            col_num = None
            
            for possible in ['número', 'numero', 'num', 'number', 'result']:
                if possible in df.columns:
                    col_num = possible
                    break
            
            if col_num is None:
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        col_num = col
                        break
            
            if col_num is None: 
                return None
                
            df = df.rename(columns={col_num: 'numero'})
            df = df.iloc[::-1].reset_index(drop=True)
            
            if len(df) < 5: 
                return None
                
            dados = []
            for _, row in df.iterrows():
                try:
                    num = int(float(row['numero']))
                    if num == 0: 
                        cor = 'B'
                    elif 1 <= num <= 7: 
                        cor = 'V'
                    elif 8 <= num <= 14: 
                        cor = 'P'
                    else: 
                        continue
                    dados.append({"numero": num, "cor": cor})
                except: 
                    continue
                    
            return dados if len(dados) >= 5 else None
            
        except Exception as e:
            print(f"[LeitorXLS] Erro: {e}")
            return None
