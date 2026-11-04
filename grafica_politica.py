# PERSONA 3 - Parte 3: "Batalla de las Politicas" (evaluar y justificar)
# Taller 1 Corte 3 - RL  |  MDP: Deteccion temprana de pacientes con posible infarto
#
# Requiere haber corrido antes:
#   - recompensa.py    -> recompensa.csv
#   - transiciones.py  -> transiciones.npz
#   - politica_mdp.py  -> politica_optima.csv
#
# Genera:
#   - grafica_distribucion_politica.png : que acciones favorece la politica optima
#   - grafica_batalla_politicas.png     : valor de la politica optima vs politicas base
#   - batalla_politicas.csv             : valor de cada politica

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 0.9

# ---- Cargar el MDP y la politica optima ----
R_df = pd.read_csv("recompensa.csv", index_col=0)
ESTADOS, ACCIONES = list(R_df.index), list(R_df.columns)
R = R_df.to_numpy(dtype=float)
nS, nA = len(ESTADOS), len(ACCIONES)

P = np.load("transiciones.npz", allow_pickle=True)["P"]          # (nA, nS, nS)
P_por_accion = {a: P[j] for j, a in enumerate(ACCIONES)}

pol_opt = pd.read_csv("politica_optima.csv")["accion_optima"].tolist()

def evaluar_politica(acc_por_estado):
    """Valor exacto V^pi de una politica fija: V = (I - gamma*P_pi)^-1 R_pi."""
    P_pi = np.array([P_por_accion[acc_por_estado[s]][s] for s in range(nS)])
    R_pi = np.array([R[s, ACCIONES.index(acc_por_estado[s])] for s in range(nS)])
    return np.linalg.solve(np.eye(nS) - GAMMA * P_pi, R_pi)

colores_acc = {"ESCALAR": "#c0392b", "REPETIR": "#e9c46a", "SEGUIMIENTO": "#2a9d8f"}

# ---- GRAFICA 1: distribucion de la politica optima ----
conteo = {a: pol_opt.count(a) for a in ACCIONES}
fig, ax = plt.subplots(figsize=(7, 4.5))
barras = ax.bar(list(conteo.keys()), list(conteo.values()),
                color=[colores_acc[a] for a in conteo])
for b, (a, c) in zip(barras, conteo.items()):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f"{c}/{nS}\n({100*c/nS:.0f}%)", ha="center", fontsize=10)
ax.set_ylabel("N.o de estados que eligen la accion")
ax.set_ylim(0, nS + 1)
ax.set_title("Distribucion de la politica optima\n(Deteccion temprana de infarto)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("grafica_distribucion_politica.png", dpi=200)

# ---- GRAFICA 2 + batalla: optima vs politicas base ----
politicas = {
    "Optima (agente)":     pol_opt,
    "Siempre REPETIR":     ["REPETIR"] * nS,
    "Siempre SEGUIMIENTO": ["SEGUIMIENTO"] * nS,   # "nunca actuar"
}
resultados = {n: evaluar_politica(p) for n, p in politicas.items()}
# Politica aleatoria: promedio de las acciones en cada estado
P_rand = P.mean(axis=0)
R_rand = R.mean(axis=1)
resultados["Aleatoria"] = np.linalg.solve(np.eye(nS) - GAMMA * P_rand, R_rand)

filas = [{"politica": n, "V_promedio": round(V.mean(), 1),
          **{f"V_{ESTADOS[s]}": round(V[s], 1) for s in range(nS)}}
         for n, V in resultados.items()]
pd.DataFrame(filas).to_csv("batalla_politicas.csv", index=False)

orden = sorted(resultados, key=lambda k: resultados[k].mean(), reverse=True)
valores = [resultados[k].mean() for k in orden]
cols = ["#2a9d8f" if k.startswith("Optima") else "#9aa0a6" for k in orden]
fig, ax = plt.subplots(figsize=(8, 4.8))
barras = ax.bar(orden, valores, color=cols)
for b, v in zip(barras, valores):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + (8 if v >= 0 else -8),
            f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Valor promedio de los estados  (V^pi)")
ax.set_title("Batalla de politicas: valor esperado por estrategia\n(gamma = 0.9)")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig("grafica_batalla_politicas.png", dpi=200)

print("Batalla de politicas:")
print(pd.DataFrame(filas).to_string(index=False))
print("\nGuardadas: grafica_distribucion_politica.png, grafica_batalla_politicas.png, batalla_politicas.csv")
