# ETAPA 0 — Leitura do CSV (com checagens)

import re
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path("input.csv")

assert CSV_PATH.exists(), f"Arquivo não encontrado: {CSV_PATH.resolve()}"

# Tentativas de leitura com encodings comuns
encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
last_err = None
df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(CSV_PATH, encoding=enc)
        print(f"[OK] Lido com encoding: {enc}")
        break
    except Exception as e:
        last_err = e

if df is None:
    raise RuntimeError(f"Falha ao ler CSV. Último erro: {last_err}")

# Mostra um resumo rápido
print("\n[INFO] Formato do DataFrame:", df.shape)
print("[INFO] Primeiras colunas:", list(df.columns[:10]))
print("\n[INFO] Amostra (5 linhas):")
print(df.head(5))

# Helper para localizar colunas mesmo com sufixos (ex.: 'Type (Remove)')
def find_col(cols, key):
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

# Detecta as colunas principais (só para conferência visual nesta etapa)
col_type  = find_col(df.columns, "Type")
col_env   = find_col(df.columns, "Environment")
col_cr    = find_col(df.columns, "CR")
col_size  = find_col(df.columns, "Size")

print("\n[DETECÇÃO DE COLUNAS]")
print("Type:", col_type)
print("Environment:", col_env)
print("CR:", col_cr)
print("Size:", col_size)

# ETAPA 1 -Seleção de colunas essenciais
print("\nETAPA 1 - Eliminando colunas não essenciais...")

# Nesta etapa vamos preparar um DataFrame de trabalho limpo, apenas com o que importa
# para a geração de encontros.

# Dicionário com nomes detectados (da etapa anterior)
ESSENTIALS = {
    "type": col_type,
    "env": col_env,
    "cr": col_cr,
    "size": col_size
}

print("\n[INFO] Colunas essenciais selecionadas:")
for k, v in ESSENTIALS.items():
    print(f"  {k:>8} -> {v}")

# Cria uma cópia só com essas colunas
df_work = df[[v for v in ESSENTIALS.values() if v is not None]].copy()

print("\n[INFO] DataFrame de trabalho criado.")
print("[INFO] Formato:", df_work.shape)
print("[INFO] Colunas:", list(df_work.columns))
print("\nPrévia:")
print(df_work.head(10))


# ETAPA 2 - Limpeza e transformação dos dados
print("\nETAPA 2 - Iniciando Limpeza e Transdormação...")

# Funções Auxiliares
def parse_cr(x):
    # converter valores de CR para float
    s = str(x).strip()
    if s in ("nan", "", "—", "-", "None"):
        return np.nan
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        except:
            pass
    try:
        return float(s)
    except:
        return np.nan

def clean_text_basic(s):
    # Remove parenteses e normaliza o caps
    s = re.sub(r"\(.*?\)", "", str(s))
    s = re.sub(r"\s+", " ", s)
    return s.strip().title()

def normalize_env(env_str):
    # padronizar ambientes (environment)
    parts = [p.strip().title() for p in str(env_str).split(",") if p.strip()]
    if not parts:
        return "Unknown"
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)

#Aplicando limpeza
df_work["Type"] = df_work["Type"].apply(clean_text_basic)
df_work["Size"] =  df_work["Size"].apply(clean_text_basic)
df_work["Environment"] = df_work["Environment"].apply(normalize_env)
df_work["CR_float"] = df_work["CR"].apply(parse_cr)

#Tratando valores faltantes
df_work["CR_float"] = df_work["CR_float"].fillna(0.0)
df_work["Environment"] = df_work["Environment"].replace("", "Unknown")
df_work["Type"] = df_work["Type"].replace("", "Unknown")

# Eliminar duplicatas
before = len(df_work)
df_work.drop_duplicates(inplace=True)
after = len(df_work)

print(f"[INFO] Duplicatas removidas: {before - after}")
print("[INFO] Visualização dos dados limpos:")
print(df_work.head(10))

# ETAPA 3 — Transformação (one-hot e multi-one-hot)
print("\nETAPA 3 - Iniciando transformação (one-hot)...")

# ID estável após a limpeza
df_work = df_work.reset_index(drop=True)
df_work.insert(0, "MonsterID", df_work.index + 1)

# One-hot de Type e Size
type_dummies = pd.get_dummies(df_work["Type"], prefix="type")
size_dummies = pd.get_dummies(df_work["Size"], prefix="size")

# Multi one-hot de Environment
KNOWN_ENVS = [
    "Arctic","Cave","Desert","Dungeon","Forest","Hell","Mountain",
    "Plains","Sky","Underground","Urban","Water","Unknown"
]

def env_one_hot(env_str):
    envs = [e.strip() for e in str(env_str).split(",") if e.strip()]
    return {f"env_{e}": int(e in envs) for e in KNOWN_ENVS}

env_dummies = df_work["Environment"].apply(env_one_hot).apply(pd.Series)

features = pd.concat([
    df_work[["MonsterID","CR_float"]],
    type_dummies, size_dummies, env_dummies
], axis=1)

print("[INFO] Preview das features:", features.shape)
print(features.head(5))

# ETAPA 4 — Exportação final
print("\nETAPA 3 - Exportando arquivos...")
from pathlib import Path
OUT_CLEAN = Path("monsters_clean.csv")
OUT_TRAIN = Path("monsters_train.csv")

# 1) Dataset limpo
df_clean = df_work[["MonsterID","Type","Environment","Size","CR","CR_float"]]
df_clean.to_csv(OUT_CLEAN, index=False)

# 2) Dataset de treino
features.to_csv(OUT_TRAIN, index=False)

print(f"[OK] Salvos:\n - {OUT_CLEAN.resolve()}\n - {OUT_TRAIN.resolve()}")
