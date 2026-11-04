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

#Estados K-MEANS

scaler = StandardScaler().fit(df[VARS_LOG])
X = scaler.transform(df[VARS_LOG])
kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEMILLA).fit(X)
print("Silhoutte con K =", K, ":", round(silhouette_score(X, kmeans.labels_), 3))

#Re orden de los Clsuters

crudo = pd.Series(kmeans.labels_, index=df.index)
orden = df.groupby(crudo)["y"].mean().sort_values().index.tolist()
mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
df["estado"] = crudo.map(mapa)
df["estado_nombre"] = df["estado"].map(dict(enumerate(NOMBRES)))

#Perfil de cada estado
perfil = df.groupby(['estado','estado_nombre']).agg(
    n=("y","size"),
    troponina_mediana=("Troponin","median"),
    ckmb_mediana=("CK-MB","median"),
    pct_positivo=("y","mean"),
    edad_mediana=("Age","median"),
).round(3)
perfil["pct_positivo"] = (perfil["pct_positivo"]*100).round(1)
print("\n", perfil)

