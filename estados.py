import numpy as np
import pandas as pd 
import joblib 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RUTA = "Medicaldataset.csv"
VARS_ESTADO = ["Troponin","CK-MB"]
K= 4
SEMILLA= 42
NOMBRES= ["Bajo","Dudoso","Intermedio","Alto"]

#Limpieza 

df = pd.read_csv(RUTA)

print("Filas Originales", len(df))
print("Nulos", df.isna().sum().sum(), "| Duplicados", df.duplicated().sum())

# Valores Outliers

malas = ~df["Heart rate"].between(30,200)
print("Filas con frecuencia cardiaca imposible", malas.sum())
df = df[~malas].copy()

df["y"] = (df["Result"] == "positive").astype(int)

for c in VARS_ESTADO:
    df["log_" + c] = np.log10(df[c])
VARS_LOG = ["log_" + c for c in VARS_ESTADO]
print("Filas tras limpieza:",len(df))
