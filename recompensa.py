# PERSONA 2 - Paso 3a: Recompensa R(s, a)
# Taller 1 Corte 3 - RL  |  MDP: Deteccion temprana de pacientes con posible infarto
#
# Requiere haber corrido antes estados.py (usa dataset_con_estados.csv).
#
# Logica de negocio: la recompensa depende de la accion y del resultado REAL del
# paciente (si habia infarto o no). El error mas costoso es el falso negativo:
# dejar en SEGUIMIENTO a quien si tiene infarto (-1000), ~100x una falsa alarma.
#
# R(s, a) = promedio, por estado, de la recompensa de cada paciente segun su
# resultado real. Salida: recompensa.csv (filas = estado, columnas = accion).

import numpy as np
import pandas as pd

ESTADOS  = ["Bajo", "Dudoso", "Alto-T", "Alto-CK"]
ACCIONES = ["ESCALAR", "REPETIR", "SEGUIMIENTO"]

# Recompensa base: [accion][resultado]  (resultado: 1 = habia infarto, 0 = sano)
R_BASE = {
    #            infarto   sano
    "ESCALAR":     ( +100,  -10),   # atencion a tiempo / falsa alarma
    "REPETIR":     (  -40,   -3),   # demora en la atencion / costo de repetir
    "SEGUIMIENTO": (-1000,   +5),   # FALSO NEGATIVO / ahorro de recursos
}

df = pd.read_csv("dataset_con_estados.csv")   # trae columnas 'estado' y 'y'

# Recompensa por paciente y accion segun su resultado real (y)
for a in ACCIONES:
    r_inf, r_sano = R_BASE[a]
    df[f"R_{a}"] = np.where(df["y"] == 1, r_inf, r_sano)

# R(s, a) = promedio por estado
R = df.groupby("estado")[[f"R_{a}" for a in ACCIONES]].mean()
R.index = [ESTADOS[i] for i in R.index]
R.columns = ACCIONES
# Se redondea a entero para presentar R(s,a) como en el informe (ej. +27, -16, -336).
R = R.round(0).astype(int)

print("Recompensa base por decision:")
print(pd.DataFrame(R_BASE, index=["infarto", "sano"]).T, "\n")
print("% infarto real por estado:")
print((df.groupby("estado")["y"].mean() * 100).round(1).to_string(), "\n")
print("R(s, a) - recompensa esperada por estado:")
print(R.to_string())

R.to_csv("recompensa.csv")
print("\nGuardado: recompensa.csv")
