# PERSONA 2 - Paso 3b: Transiciones P(s' | s, a)   (Opcion B: consultar los modelos)
# Taller 1 Corte 3 - RL  |  MDP: Deteccion temprana de pacientes con posible infarto
#
# Requiere estados.py (dataset_con_estados.csv y modelo_estados.joblib).
#
# Se descarto la Opcion A (shift(-1)): el dataset no es una serie de tiempo, cada
# fila es un paciente distinto. En su lugar (Opcion B):
#   1. AdaBoost estima p = P(infarto | paciente).
#   2. Para cada accion se simula como cambian troponina y CK-MB en el siguiente
#      control (multiplicador + ruido lognormal), en el escenario "infarto" (peso p)
#      y "sano" (peso 1-p).
#   3. El K-Means (.predict) asigna el estado siguiente s'.
#   4. Se acumulan los pesos en un conteo (s, a, s') y cada fila se normaliza a 1.
#
# Salida: transiciones.npz (tensor P) y P_ESCALAR.csv / P_REPETIR.csv / P_SEGUIMIENTO.csv

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier

SEMILLA = 42
N_REPETICIONES = 50          # repeticiones por paciente para suavizar el ruido
SIGMA_RUIDO = 0.10           # ~10% de ruido de medicion (lognormal)
rng = np.random.default_rng(SEMILLA)

ESTADOS  = ["Bajo", "Dudoso", "Alto-T", "Alto-CK"]
ACCIONES = ["ESCALAR", "REPETIR", "SEGUIMIENTO"]
nS, nA = len(ESTADOS), len(ACCIONES)

# Multiplicador sobre troponina y CK-MB en el siguiente control de un paciente CON infarto.
# Pacientes sanos: sin cambio (x1.00). Son supuestos de modelado (validar con personal medico).
MULT_INFARTO = {"ESCALAR": 0.40, "REPETIR": 1.15, "SEGUIMIENTO": 1.60}
MULT_SANO    = {"ESCALAR": 1.00, "REPETIR": 1.00, "SEGUIMIENTO": 1.00}

# ---- Cargar datos y modelos de Persona 1 ----
df = pd.read_csv("dataset_con_estados.csv")
modelo = joblib.load("modelo_estados.joblib")
scaler, kmeans, mapa = modelo["scaler"], modelo["kmeans"], modelo["mapa"]
VARS = modelo["vars"]                       # ["Troponin", "CK-MB"]

# ---- AdaBoost: p = P(infarto | paciente) ----
FEATURES = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
            "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]
ada = AdaBoostClassifier(n_estimators=200, random_state=SEMILLA).fit(df[FEATURES], df["y"])
p_infarto = ada.predict_proba(df[FEATURES])[:, 1]
print("AdaBoost entrenado | p promedio de infarto:", round(p_infarto.mean(), 3))

COLS_LOG = ["log_" + v for v in VARS]               # ["log_Troponin", "log_CK-MB"]

def estado_de(trop, ckmb):
    """Asigna estado (0..3) a marcadores simulados usando el pipeline de Persona 1."""
    X = pd.DataFrame({COLS_LOG[0]: np.log10(trop), COLS_LOG[1]: np.log10(ckmb)})
    crudo = kmeans.predict(scaler.transform(X))
    return np.array([mapa[c] for c in crudo])      # re-ordena al indice Bajo..Alto-CK

# ---- Construir el conteo (s, a, s') ----
conteo = np.zeros((nA, nS, nS))
trop0 = df[VARS[0]].to_numpy()
ckmb0 = df[VARS[1]].to_numpy()
s_actual = df["estado"].to_numpy()

for ai, a in enumerate(ACCIONES):
    for _ in range(N_REPETICIONES):
        # Escenario INFARTO (peso p): marcadores x multiplicador, con ruido
        ruido = rng.lognormal(0.0, SIGMA_RUIDO, size=len(df))
        s_inf = estado_de(trop0 * MULT_INFARTO[a] * ruido,
                           ckmb0 * MULT_INFARTO[a] * ruido)
        # Escenario SANO (peso 1-p): sin cambio, con ruido
        ruido = rng.lognormal(0.0, SIGMA_RUIDO, size=len(df))
        s_san = estado_de(trop0 * MULT_SANO[a] * ruido,
                          ckmb0 * MULT_SANO[a] * ruido)
        for i in range(len(df)):
            conteo[ai, s_actual[i], s_inf[i]] += p_infarto[i]
            conteo[ai, s_actual[i], s_san[i]] += (1 - p_infarto[i])

# ---- Normalizar cada fila a 1 ----
P = conteo / conteo.sum(axis=2, keepdims=True)

# ---- Guardar ----
np.savez("transiciones.npz", P=P, estados=ESTADOS, acciones=ACCIONES)
for ai, a in enumerate(ACCIONES):
    tabla = pd.DataFrame(P[ai].round(2), index=ESTADOS, columns=ESTADOS)
    tabla.to_csv(f"P_{a}.csv")
    print(f"\nP(s' | s, {a}):")
    print(tabla.to_string())

print("\nGuardados: transiciones.npz, P_ESCALAR.csv, P_REPETIR.csv, P_SEGUIMIENTO.csv")
