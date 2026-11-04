# PERSONA 3 - Pasos 4 y 5: Iteracion de Valor y Extraccion de Politica
# Taller 1 Corte 3 - RL  |  MDP: Deteccion temprana de pacientes con posible infarto
#
# Requiere haber corrido antes:
#   - recompensa.py    -> recompensa.csv        (R(s,a), Paso 3a)
#   - transiciones.py  -> transiciones.npz      (P(s'|s,a), Paso 3b)
#
# Con R y P ya construidas, resolver el MDP es la parte directa: el codigo de
# Iteracion de Valor (Paso 4) y Extraccion de Politica (Paso 5) funciona casi sin
# cambios. Salidas: valores_estado.csv y politica_optima.csv.

import numpy as np
import pandas as pd

GAMMA = 0.9       # factor de descuento (valora el futuro)
THETA = 1e-9      # umbral de convergencia
MAX_IT = 1000

# ---- Cargar el MDP (salidas de Persona 2) ----
R_df = pd.read_csv("recompensa.csv", index_col=0)
ESTADOS = list(R_df.index)
ACCIONES = list(R_df.columns)
R = R_df.to_numpy(dtype=float)                       # (nS, nA)

datos = np.load("transiciones.npz", allow_pickle=True)
P_tensor = datos["P"]                                # (nA, nS, nS)
nS, nA = len(ESTADOS), len(ACCIONES)
assert P_tensor.shape == (nA, nS, nS), "P no coincide con R"

# ---- PASO 4 - Iteracion de Valor ----
V = np.zeros(nS)
print("=" * 60)
print("PASO 4 - ITERACION DE VALOR   (gamma =", GAMMA, ")")
print("=" * 60)
for it in range(1, MAX_IT + 1):
    Q = R + GAMMA * (P_tensor @ V).T                 # Q[s,a]
    V_nuevo = Q.max(axis=1)
    delta = np.abs(V_nuevo - V).max()
    V = V_nuevo
    if it <= 5 or delta < THETA:
        print(f"  iter {it:>3} | delta = {delta:.3e} | V = {np.round(V, 2)}")
    if delta < THETA:
        print(f"  --> Convergio en {it} iteraciones.")
        break

# ---- PASO 5 - Extraccion de la Politica Optima ----
Q_final = R + GAMMA * (P_tensor @ V).T
politica = [ACCIONES[i] for i in Q_final.argmax(axis=1)]

print("\n" + "=" * 60)
print("PASO 5 - POLITICA OPTIMA   (pi*(s) = argmax_a Q(s,a))")
print("=" * 60)
tabla = pd.DataFrame({
    "estado": ESTADOS,
    "V_optimo": np.round(V, 2),
    "accion_optima": politica,
    **{f"Q_{a}": np.round(Q_final[:, j], 2) for j, a in enumerate(ACCIONES)},
})
print(tabla.to_string(index=False))

tabla.to_csv("politica_optima.csv", index=False)
pd.DataFrame({"estado": ESTADOS, "V_optimo": np.round(V, 4)}).to_csv(
    "valores_estado.csv", index=False)
print("\nGuardados: politica_optima.csv, valores_estado.csv")
