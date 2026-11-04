# ============================================================
# PERSONA 2 - Dinámica del mundo: Acciones, Recompensa R(s,a) y Transición P(s'|s,a)
# (Paso 3a y Paso 3b del taller). Adaptado al repositorio de Persona 1.
# Ejecutar desde la carpeta del repo:  python persona2_R_y_P.py
# ============================================================
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# ------------------------------------------------------------
# 0. CONFIGURACIÓN
# ------------------------------------------------------------
ARCHIVO_DF_ESTADOS = "dataset_con_estados.csv"   # Persona 1
ARCHIVO_ESTADOS    = "modelo_estados.joblib"     # Persona 1: dict {scaler, kmeans, mapa, vars, nombres}
ARCHIVO_ADABOOST   = "modelo_adaboost.joblib"    # opcional: si no existe, se entrena uno aquí

COL_ESTADO = "estado"
COL_CLASE  = "Result"      # 'positive' / 'negative'
MARCADORES = ["Troponin", "CK-MB"]   # variables que usa el K-Means (en log10)

FEATURES_SUPERVISADO = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
                        "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]
SEED = 42

# ------------------------------------------------------------
# 1. ACCIONES (Pregunta 2)
# ------------------------------------------------------------
ACCIONES = ["ESCALAR", "REPETIR", "SEGUIMIENTO"]
NOMBRE_ACCION = {
    "ESCALAR":     "Alerta ROJA: escalar al médico y priorizar atención",
    "REPETIR":     "Alerta AMARILLA: repetir medición y pedir valoración clínica",
    "SEGUIMIENTO": "Alerta VERDE: seguimiento habitual",
}

# ------------------------------------------------------------
# 2. RECOMPENSAS (Pregunta 4)
# ------------------------------------------------------------
RECOMPENSA = {
    #                  si había infarto | si NO había infarto
    "ESCALAR":     {"infarto":  +100,  "sano":  -10},
    "REPETIR":     {"infarto":   -40,  "sano":   -3},
    "SEGUIMIENTO": {"infarto": -1000,  "sano":   +5},
}

# ------------------------------------------------------------
# 3. SUPUESTOS DE DINÁMICA (Pregunta 5): multiplicador de troponina y CK-MB
#    en la siguiente evaluación del paciente
# ------------------------------------------------------------
EFECTO_MARCADORES = {
    "ESCALAR":     {"infarto": 0.40, "sano": 1.00},
    "REPETIR":     {"infarto": 1.15, "sano": 1.00},
    "SEGUIMIENTO": {"infarto": 1.60, "sano": 1.00},
}
RUIDO_SD   = 0.10
N_REPLICAS = 20


# ------------------------------------------------------------
# 4. CARGAR ENTREGABLES DE PERSONA 1
# ------------------------------------------------------------
print("--- Cargando datos y modelo de estados ---")
df = pd.read_csv(ARCHIVO_DF_ESTADOS)
est = joblib.load(ARCHIVO_ESTADOS)
scaler, kmeans, mapa = est["scaler"], est["kmeans"], est["mapa"]
NOMBRES = est["nombres"]                       # ['Bajo','Dudoso','Alto-Troponina','Alto-CKMB']


def asignar_estado(X):
    """Replica la lógica de Persona 1: log10 -> scaler -> KMeans.predict -> reorden (mapa)."""
    Xlog = pd.DataFrame(np.log10(X[MARCADORES].to_numpy(dtype=float)),
                        columns=["log_" + c for c in MARCADORES])   # mismos nombres que en el fit
    crudo = kmeans.predict(scaler.transform(Xlog))
    return np.array([mapa[c] for c in crudo])


# Chequeo de consistencia: reproducir la columna 'estado' del CSV con .predict()
coincide = (asignar_estado(df) == df[COL_ESTADO].to_numpy()).mean()
print(f"Coincidencia de .predict() con la columna 'estado' del CSV: {coincide:.3%}")
assert coincide > 0.99, "El .predict() no reproduce los estados de Persona 1."

y = (df[COL_CLASE].astype(str).str.lower() == "positive").astype(int).to_numpy()
S, A = len(NOMBRES), len(ACCIONES)
s_idx = df[COL_ESTADO].to_numpy()               # los estados ya son 0..K-1 ordenados por riesgo
print(f"Estados: {dict(enumerate(NOMBRES))}  |  Acciones: {ACCIONES}")


# ------------------------------------------------------------
# 5. PASO 3a - Recompensa R(s, a)
# ------------------------------------------------------------
print("\n--- PASO 3a: Construyendo R(s, a) ---")
R_df = pd.DataFrame({"estado": s_idx})
for a in ACCIONES:
    R_df[a] = np.where(y == 1, RECOMPENSA[a]["infarto"], RECOMPENSA[a]["sano"])
R_table = R_df.groupby("estado")[ACCIONES].mean().reindex(range(S))
R_table.index = NOMBRES
R = R_table.to_numpy()                          # (S x A)

resumen_estados = pd.DataFrame({
    "n_pacientes": np.bincount(s_idx, minlength=S),
    "pct_infarto_real": [round(y[s_idx == s].mean() * 100, 1) for s in range(S)],
}, index=NOMBRES)
print(R_table.round(2).to_string())
print("\nResumen por estado:")
print(resumen_estados.to_string())


# ------------------------------------------------------------
# 6. PASO 3b - Transición P(s' | s, a)  (Opción B)
# ------------------------------------------------------------
print("\n--- PASO 3b: Construyendo P(s' | s, a) ---")

# Probabilidad de infarto según el modelo supervisado (AdaBoost)
if os.path.exists(ARCHIVO_ADABOOST):
    ada = joblib.load(ARCHIVO_ADABOOST)
    clases = list(ada.classes_)
    col_pos = clases.index(1) if 1 in clases else clases.index("positive")
    p_inf = ada.predict_proba(df[FEATURES_SUPERVISADO])[:, col_pos]
    print("AdaBoost cargado desde", ARCHIVO_ADABOOST)
else:
    # Respaldo: se entrena un AdaBoost y se usan probabilidades fuera de muestra (5-fold)
    ada = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    p_inf = cross_val_predict(ada, df[FEATURES_SUPERVISADO], y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              method="predict_proba")[:, 1]
    print("No se encontró", ARCHIVO_ADABOOST, "-> AdaBoost entrenado aquí (probabilidades 5-fold).")

rng = np.random.default_rng(SEED)
counts = np.zeros((A, S, S))

for ai, a in enumerate(ACCIONES):
    for _ in range(N_REPLICAS):
        for escenario, peso in [("infarto", p_inf), ("sano", 1.0 - p_inf)]:
            X_next = df[MARCADORES].copy()
            for m in MARCADORES:
                ruido = rng.lognormal(0.0, RUIDO_SD, len(X_next))
                X_next[m] = X_next[m] * EFECTO_MARCADORES[a][escenario] * ruido
            s_next = asignar_estado(X_next)       # <-- .predict() del K-Means de Persona 1
            np.add.at(counts[ai], (s_idx, s_next), peso)

tot = counts.sum(axis=2, keepdims=True)
P = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)
assert np.allclose(P.sum(axis=2), 1.0), "Alguna fila de P no suma 1."
print("P construida con forma (A, S, S) =", P.shape, "-> todas las filas suman 1.")

for ai, a in enumerate(ACCIONES):
    print(f"\nP(s' | s, a={a}):")
    print(pd.DataFrame(P[ai], index=NOMBRES, columns=NOMBRES).round(3).to_string())


# ------------------------------------------------------------
# 7. ENTREGABLES PARA PERSONA 3
# ------------------------------------------------------------
np.savez("mdp_matrices.npz", R=R, P=P,
         states=np.array(NOMBRES, dtype=str), actions=np.array(ACCIONES, dtype=str))
R_table.to_csv("R_table.csv")
resumen_estados.to_csv("resumen_estados.csv")
with open("mdp_metadata.json", "w", encoding="utf-8") as f:
    json.dump({"states": NOMBRES, "actions": ACCIONES, "nombre_accion": NOMBRE_ACCION,
               "recompensa": RECOMPENSA, "efecto_marcadores": EFECTO_MARCADORES},
              f, ensure_ascii=False, indent=2)
print("\nGuardados: mdp_matrices.npz, R_table.csv, resumen_estados.csv, mdp_metadata.json")

# NOTA PARA PERSONA 3 (Bellman):  Q = R + GAMMA * np.einsum("asn,n->sa", P, V)
#   V = Q.max(axis=1);  política = Q.argmax(axis=1)
